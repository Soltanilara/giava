"""Train ACT or Diffusion Policy on a GIAVA real-robot LeRobot dataset.

Why this wrapper exists instead of calling `lerobot-train` directly: the GIAVA
recorder stores MORE than a policy should see -- per-camera/per-arm float64
epoch timestamps and end-effector poses ride along in every frame for sync
analysis and replay.  lerobot's factory turns EVERY `observation.*` feature
into a policy input unless input_features is set explicitly, so a naive
`lerobot-train` run would feed ~1.7e9-magnitude epoch timestamps into the
normalization buffers and the policy.  This wrapper builds the config
programmatically: inputs are observation.state + the chosen cameras, nothing
else (ee_pose opt-in via --use-ee-pose), and everything downstream is
lerobot's own training pipeline, checkpointing included.

Usage (gym_av312 env):

  python train_real.py --root <dataset_run_dir> --policy act
  python train_real.py --root <dataset_run_dir> --policy smolvla   # needs env vla312
  python train_real.py --root <dataset_run_dir> --policy diffusion \
      --cameras left_wrist right_wrist top_scene --successes-only

Both policies train from the SAME dataset -- action space (absolute joint
positions) and observations are identical; only chunking differs and it is
derived from the dataset's own fps.
"""

import argparse
import json
import sys
from pathlib import Path


def select_features(ds_meta, cameras, use_ee_pose, use_env_state=False):
    from lerobot.configs.types import FeatureType
    from lerobot.utils.constants import OBS_ENV_STATE
    from lerobot.utils.feature_utils import dataset_to_policy_features

    features = dataset_to_policy_features(ds_meta.features)
    output_features = {k: f for k, f in features.items()
                       if f.type is FeatureType.ACTION}

    available_cams = [k.removeprefix("observation.images.")
                      for k in features if k.startswith("observation.images.")]
    if cameras == ["all"]:
        cameras = available_cams
    missing = [c for c in cameras if c not in available_cams]
    if missing:
        raise SystemExit(f"camera(s) {missing} not in dataset; "
                         f"available: {available_cams}")

    input_features = {"observation.state": features["observation.state"]}
    for cam in cameras:
        input_features[f"observation.images.{cam}"] = \
            features[f"observation.images.{cam}"]
    if use_ee_pose:
        for k, f in features.items():
            if k.startswith("observation.ee_pose"):
                input_features[k] = f
    if use_env_state:
        ## ACT reads exactly ONE low-dim key besides observation.state, and it
        ## matches on this literal name with FeatureType.ENV.  A geometry
        ## vector under any other key (observation.object_centroid, say) is
        ## normalized and then silently dropped -- the whole reason the Nov
        ## 2025 study's centroid variants were identical to plain RGB.
        if OBS_ENV_STATE not in features:
            raise SystemExit(
                f"--env-state given but {OBS_ENV_STATE} is not in this "
                "dataset; build it with build_envstate_dataset.py")
        f = features[OBS_ENV_STATE]
        if f.type is not FeatureType.ENV:
            raise SystemExit(
                f"{OBS_ENV_STATE} typed {f.type}, expected FeatureType.ENV")
        input_features[OBS_ENV_STATE] = f

    excluded = sorted(set(features) - set(input_features) - set(output_features))
    print(f"policy inputs:  {sorted(input_features)}")
    print(f"policy outputs: {sorted(output_features)}")
    print(f"excluded from policy (recorded for analysis/replay): {excluded}")
    return input_features, output_features


def episode_outcomes(root):
    """episode_index -> latest record from episode_outcomes.jsonl (the
    recorder appends one line per saved episode: outcome + task string)."""
    for candidate in (root / "episode_outcomes.jsonl",
                      root.parent / "episode_outcomes.jsonl"):
        if candidate.exists():
            recs = {}
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        recs[int(rec["episode_index"])] = rec
            return recs
    raise SystemExit(f"no episode_outcomes.jsonl found under {root}")


def successful_episodes(root):
    """Episode indices labelled 'success' in episode_outcomes.jsonl."""
    return sorted(i for i, r in episode_outcomes(root).items()
                  if r.get("outcome") == "success")


def dagger_episodes(root, start, min_interventions):
    """Episodes recorded under --policy (index >= start) that carry at least
    `min_interventions` operator takeovers, read from meta/robustness.jsonl
    (teleop_enable_count = times a controller was gripped).  An episode the
    policy finished alone teaches nothing the base demos did not; the
    corrections are the information."""
    path = root / "meta" / "robustness.jsonl"
    if not path.exists():
        raise SystemExit(f"--dagger-from: {path} not found")
    keep = []
    dropped = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            i = int(rec["episode_index"])
            if i < start:
                continue
            n = int(rec.get("stats", {}).get("teleop_enable_count", 0))
            (keep if n >= min_interventions else dropped).append(i)
    print(f"  dagger episodes >= {start}: keeping {len(keep)} with >= "
          f"{min_interventions} intervention(s), dropping {len(dropped)} "
          f"the policy finished alone: {dropped}")
    return sorted(set(keep))


def piece_episodes(root, pieces):
    """Episode indices whose task string names one of `pieces` (shape_sorter
    runs: 'insert the red cube into the square hole' -> cube)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interbotix_ws"
                           / "src" / "av_aloha" / "data_collection_scripts"))
    import shape_sorter as ss
    bad = [p for p in pieces if p not in ss.CLASSES]
    if bad:
        raise SystemExit(f"--pieces: unknown {bad}; one of {ss.CLASS_NAMES}")
    keep = sorted(i for i, r in episode_outcomes(root).items()
                  if ss.target_from_task(r.get("task", "")) in pieces)
    if not keep:
        raise SystemExit(f"--pieces {pieces}: no episode in {root} names them")
    return keep


## Released SmolVLA checkpoint we fine-tune from (config + weights + processors).
SMOLVLA_BASE = "lerobot/smolvla_base"
## pi0 / pi0.5 base checkpoints.  3.3B params: a FULL fine-tune wants ~40 GB+
## (an A100 80GB is comfortable at batch 16-32); on 24 GB use a smaller batch
## and gradient checkpointing.  Unlike SmolVLA above, the vision encoder is
## NOT frozen by default -- which is the setting the 2026-09-03 SmolVLA run
## got wrong, so leave it alone unless you are deliberately ablating it.
PI0_BASE = {"pi0": "lerobot/pi0", "pi05": "lerobot/pi05_base"}


def build_policy_config(policy, fps, input_features, output_features,
                        init_from=None, dtype=None,
                        gradient_checkpointing=False,
                        train_vision_encoder=False,
                        chunk_size=None, n_action_steps=None):
    if init_from is not None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy_config  # noqa: F401  registers types

        ## Fine-tune: config + weights from an existing checkpoint's
        ## pretrained_model dir; the I/O features must match it exactly (same
        ## arm mode, cameras, env-state variant).  lerobot re-derives the
        ## normalization stats from THIS dataset, so fine-tune on a dataset
        ## whose stats resemble the base's (the pooled base + new-piece
        ## episodes, not the 10 new ones alone).
        ckpt = Path(init_from)
        if not (ckpt / "config.json").exists():
            raise SystemExit(f"--init-from: no config.json in {ckpt} -- point "
                             f"at .../checkpoints/<step>/pretrained_model")
        cfg = PreTrainedConfig.from_pretrained(str(ckpt))
        if cfg.type != policy:
            raise SystemExit(f"--init-from checkpoint is {cfg.type}, "
                             f"--policy says {policy}")
        for name, ours, theirs in (("input", input_features, cfg.input_features),
                                   ("output", output_features, cfg.output_features)):
            if {k: tuple(v.shape) for k, v in ours.items()} != \
               {k: tuple(v.shape) for k, v in theirs.items()}:
                raise SystemExit(
                    f"--init-from: {name} features differ from the "
                    f"checkpoint's.\n  dataset:    "
                    f"{ {k: tuple(v.shape) for k, v in ours.items()} }\n"
                    f"  checkpoint: "
                    f"{ {k: tuple(v.shape) for k, v in theirs.items()} }")
        cfg.pretrained_path = str(ckpt)
        cfg.push_to_hub = False
        return cfg

    if policy == "smolvla":
        from lerobot.configs.policies import PreTrainedConfig
        ## Importing the config class is what registers "smolvla" as a draccus
        ## choice; without it from_pretrained cannot decode the hub config.json.
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401

        ## Fine-tune, not train-from-scratch: the config comes from the hub
        ## checkpoint and `pretrained_path` makes lerobot load its weights and
        ## processor pipelines too.  lerobot_train then swaps the SO-100
        ## normalization stats for this dataset's own, so only the I/O
        ## features need overriding here.  The base's chunk_size=50 is 2 s at
        ## 25 Hz -- the same horizon ACT uses below.  State/action are padded
        ## to max_state_dim/max_action_dim=32 inside the model, so a 7-dof arm
        ## loads into the 6-dof SO-100 projections without surgery.
        ## SmolVLA needs transformers>=5.4 (+ huggingface-hub>=1.5), which the
        ## data-collection env pins away from -- run this in `vla312`.
        cfg = PreTrainedConfig.from_pretrained(SMOLVLA_BASE)
        cfg.input_features = input_features
        cfg.output_features = output_features
        cfg.pretrained_path = SMOLVLA_BASE
        cfg.push_to_hub = False
        ## THE BASE CONFIG FREEZES THE VISION ENCODER AND TRAINS ONLY THE
        ## ACTION EXPERT.  That is the hub default, and it is what the
        ## 2026-09-03 flower run used -- so its visual features never adapted
        ## to this rig at all, which is a large part of why it lost to an ACT
        ## trained from scratch.  --train-vision-encoder undoes both.
        if train_vision_encoder:
            cfg.freeze_vision_encoder = False
            cfg.train_expert_only = False
        if dtype is not None:
            cfg.dtype = dtype
        return cfg

    if policy in ("pi0", "pi05"):
        from lerobot.configs.policies import PreTrainedConfig
        ## Importing the config class registers the type as a draccus choice,
        ## exactly as for smolvla.
        if policy == "pi05":
            from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401
        else:
            from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: F401
        base = PI0_BASE[policy]
        cfg = PreTrainedConfig.from_pretrained(base)
        cfg.input_features = input_features
        cfg.output_features = output_features
        cfg.pretrained_path = base
        cfg.push_to_hub = False
        ## MEMORY.  The hub preset trains in float32 with no AMP and no
        ## gradient checkpointing: for a 3.3B model that is ~13 GB of weights
        ## + ~13 GB of grads + ~26 GB of AdamW state before a single
        ## activation.  Fine on an 80 GB card at a small batch, not fine
        ## anywhere smaller or at a large batch.  bfloat16 roughly halves the
        ## first three; gradient checkpointing trades compute for activation
        ## memory.  Neither changes the recipe, only what fits.
        if dtype is not None:
            cfg.dtype = dtype
        if gradient_checkpointing:
            cfg.gradient_checkpointing = True
        ## State/action pad to max_state_dim/max_action_dim = 32 inside the
        ## model, so a 14-d right_av vector loads without surgery.
        return cfg

    if policy == "act":
        from lerobot.policies.act.configuration_act import ACTConfig

        ## ~2 seconds of actions per chunk, the ALOHA-validated horizon,
        ## expressed in this dataset's own ticks so a 25 Hz and a 50 Hz
        ## dataset both predict the same wall-clock span.
        chunk = chunk_size or max(20, int(round(2.0 * fps)))
        ## n_action_steps == chunk_size means the chunk is executed fully
        ## open-loop.  That is fine for ABSOLUTE actions but wrong for a
        ## delta-action dataset, where each predicted step is integrated onto
        ## the last: 100 integrated deltas drift badly.  --n-action-steps lets
        ## a delta run re-plan often while keeping the same prediction horizon.
        n_act = n_action_steps or chunk
        if n_act > chunk:
            raise SystemExit(
                f"--n-action-steps {n_act} cannot exceed --chunk-size {chunk}")
        return ACTConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=chunk,
            n_action_steps=n_act,
            push_to_hub=False,
        )

    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    ## Horizon must be divisible by the UNet's 2^3 downsampling.  64 ticks
    ## spans 1.3 s at 50 Hz / 3.2 s at 20 Hz -- both workable; executing half
    ## the horizon before re-planning is the standard receding-horizon split.
    horizon, n_action_steps, n_obs_steps = 64, 32, 2
    return DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        drop_n_last_frames=horizon - n_action_steps - n_obs_steps + 1,
        push_to_hub=False,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True,
                    help="dataset run directory (holds meta.json + meta/)")
    ap.add_argument("--policy", required=True,
                    choices=["act", "diffusion", "smolvla", "pi0", "pi05"])
    ap.add_argument("--cameras", nargs="+", default=["all"],
                    help="camera names to train on (default: all in dataset)")
    ap.add_argument("--env-state", action="store_true",
                    help="feed observation.environment_state (object centroid "
                         "+ target vectors) to the policy as its own encoder "
                         "token. Requires a dataset built by "
                         "build_envstate_dataset.py.")
    ap.add_argument("--use-ee-pose", action="store_true",
                    help="also feed observation.ee_pose.* to the policy")
    ap.add_argument("--successes-only", action="store_true",
                    help="train only on episodes labelled success (ss)")
    ap.add_argument("--episodes", type=int, nargs="+", default=None,
                    help="explicit episode indices (overrides --successes-only "
                         "and --pieces)")
    ap.add_argument("--pieces", nargs="+", default=None,
                    help="shape_sorter runs: train only on episodes whose task "
                         "string names these pieces (e.g. cube triangle). "
                         "Combines with --successes-only.")
    ap.add_argument("--dagger-from", type=int, default=None, metavar="EP",
                    help="episodes with index >= EP were recorded under "
                         "data_collection --policy. Of those, keep only the "
                         "ones with >= --dagger-min-interventions takeovers; "
                         "episodes below EP (the base demos) are all kept.")
    ap.add_argument("--dagger-min-interventions", type=int, default=1)
    ap.add_argument("--exclude-episodes", type=int, nargs="+", default=[],
                    help="drop these episode indices whatever else selects them")
    ap.add_argument("--init-from", default=None,
                    help="fine-tune: a checkpoint's pretrained_model dir to "
                         "load config + weights from (features must match). "
                         "Use with a small --steps; ACT's default lr is 1e-5.")
    ap.add_argument("--lr", type=float, default=None, metavar="LR",
                    help="override the policy's optimizer learning rate. ACT's "
                         "default is 1e-5, which is a FROM-SCRATCH rate: "
                         "re-applied to an already-converged checkpoint it "
                         "moves the weights far enough to undo what the base "
                         "demos taught. For a correction fine-tune use 1e-6 "
                         "(one tenth) and few steps.")
    ap.add_argument("--lr-backbone", type=float, default=None,
                    help="ACT only: separate rate for the vision backbone "
                         "(default: whatever --lr is set to)")
    ap.add_argument("--train-vision-encoder", action="store_true",
                    help="smolvla only: unfreeze the vision encoder and "
                         "train the whole model, not just the action "
                         "expert. The hub default freezes both.")
    ap.add_argument("--policy-dtype", default=None,
                    choices=["float32", "bfloat16"],   # pi05 accepts only these
                    help="pi0/pi05 only: override the preset's dtype. "
                         "bfloat16 roughly halves weight/grad/optimizer "
                         "memory; use it if float32 will not fit.")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="pi0/pi05 only: recompute activations in the "
                         "backward pass -- slower per step, much less "
                         "activation memory.")
    ap.add_argument("--chunk-size", type=int, default=None, metavar="N",
                    help="ACT action-chunk length in ticks (default: 2 s at "
                         "the dataset's fps, i.e. 100 at 50 Hz)")
    ap.add_argument("--n-action-steps", type=int, default=None, metavar="N",
                    help="how many of the predicted chunk to execute before "
                         "re-planning (default: the whole chunk, fully "
                         "open-loop).  REQUIRED to be well below --chunk-size "
                         "for a delta-action dataset -- integrating 100 "
                         "predicted deltas open-loop drifts.  Try 20.")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--save-freq", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--output-dir", default=None,
                    help="default: outputs/train/<task>_<policy>")
    ap.add_argument("--job-name", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint in --output-dir. "
                         "To EXTEND a finished run, resume it with a larger "
                         "--steps: lerobot trains `range(step, cfg.steps)`, so "
                         "--resume --steps 40000 on a completed 20000-step run "
                         "trains 20000 more, continuing the step numbering and "
                         "keeping the optimizer state (unlike --init-from, "
                         "which starts Adam from scratch).")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    meta_json = root / "meta.json"
    if not meta_json.exists():
        raise SystemExit(f"{meta_json} not found -- --root must be a dataset "
                         f"RUN directory (task/<timestamp>)")
    with open(meta_json) as f:
        giava_meta = json.load(f)
    task = giava_meta["task"]
    repo_id = f"deviamar/{task}"

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    ds_meta = LeRobotDatasetMetadata(repo_id, root=str(root))
    fps = ds_meta.fps
    print(f"dataset: {root}")
    print(f"  task={task}  episodes={ds_meta.total_episodes} "
          f"frames={ds_meta.total_frames}  fps={fps}")
    if fps != giava_meta.get("fps"):
        raise SystemExit(f"fps mismatch between lerobot info ({fps}) and "
                         f"meta.json ({giava_meta.get('fps')})")

    episodes = args.episodes
    any_filter = (args.successes_only or args.pieces or args.dagger_from is not None
                  or args.exclude_episodes)
    if episodes is None and any_filter:
        chosen = set(episode_outcomes(root))
        if args.pieces:
            chosen &= set(piece_episodes(root, args.pieces))
        if args.successes_only:
            chosen &= set(successful_episodes(root))
        if args.dagger_from is not None:
            base = {i for i in chosen if i < args.dagger_from}
            corr = set(dagger_episodes(root, args.dagger_from,
                                       args.dagger_min_interventions))
            chosen = base | (chosen & corr)
        chosen -= set(args.exclude_episodes)
        episodes = sorted(chosen)
        if not episodes:
            raise SystemExit("episode filter left nothing to train on")
        print(f"  training on {len(episodes)} episodes"
              + (f" of pieces {args.pieces}" if args.pieces else "")
              + (" (successes only)" if args.successes_only else "")
              + (f" (base <{args.dagger_from} + corrected)" if args.dagger_from is not None else "")
              + (f" minus {args.exclude_episodes}" if args.exclude_episodes else "")
              + f": {episodes}")

    input_features, output_features = select_features(
        ds_meta, args.cameras, args.use_ee_pose, args.env_state)
    policy_cfg = build_policy_config(
        args.policy, fps, input_features, output_features, args.init_from,
        dtype=args.policy_dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        train_vision_encoder=args.train_vision_encoder,
        chunk_size=args.chunk_size, n_action_steps=args.n_action_steps)
    if args.init_from:
        print(f"  fine-tuning from {args.init_from}")

    ## LEARNING RATE.  Set on the policy config, because that is where lerobot
    ## reads it from: cfg.get_optimizer_preset() builds the AdamW config out of
    ## optimizer_lr / optimizer_lr_backbone / optimizer_weight_decay, and
    ## nothing on TrainPipelineConfig can override it afterwards.
    ##
    ## The optimizer STATE is not restored by --init-from (only --resume does
    ## that), so this is a fresh AdamW at the new rate rather than a continued
    ## schedule.
    if args.lr is not None:
        if not hasattr(policy_cfg, "optimizer_lr"):
            raise SystemExit(f"--lr: {args.policy} config has no optimizer_lr")
        _was = policy_cfg.optimizer_lr
        policy_cfg.optimizer_lr = args.lr
        _msg = f"  learning rate {_was:g} -> {args.lr:g}"
        if hasattr(policy_cfg, "optimizer_lr_backbone"):
            _wasb = policy_cfg.optimizer_lr_backbone
            policy_cfg.optimizer_lr_backbone = (args.lr_backbone
                                                if args.lr_backbone is not None
                                                else args.lr)
            _msg += (f"   backbone {_wasb:g} -> "
                     f"{policy_cfg.optimizer_lr_backbone:g}")
        print(_msg)
    elif args.lr_backbone is not None:
        raise SystemExit("--lr-backbone without --lr")

    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts.lerobot_train import train

    job_name = args.job_name or f"{task}_{args.policy}"
    output_dir = Path(args.output_dir or f"outputs/train/{job_name}")

    ## lerobot's resume path re-reads --config_path from sys.argv (draccus has
    ## normally consumed it before validate() runs); since this wrapper builds
    ## the config programmatically, plant the checkpoint path there.
    if args.resume:
        ckpt = output_dir / "checkpoints" / "last" / "pretrained_model"
        if not ckpt.exists():
            raise SystemExit(f"--resume: no checkpoint at {ckpt}")
        sys.argv.append(f"--config_path={ckpt / 'train_config.json'}")

    ## RESUMING NEEDS THE OPTIMIZER SPELLED OUT.  lerobot only fills
    ## cfg.optimizer from the policy preset when resume is False (see
    ## TrainPipelineConfig.validate), because its own CLI resume path loads
    ## the whole saved config through --config_path.  This wrapper builds the
    ## config in Python instead, so on resume cfg.optimizer would stay None
    ## and make_optimizer_and_scheduler() raises "Optimizer config is
    ## required".  Build it here; load_training_state() then overwrites the
    ## state (including the learning rate in param_groups) from the
    ## checkpoint, so a resumed run keeps the rate it was training at.
    _optim = _sched = None
    if args.resume:
        _optim = policy_cfg.get_optimizer_preset()
        _sched = policy_cfg.get_scheduler_preset()

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=repo_id, root=str(root),
                              episodes=episodes),
        policy=policy_cfg,
        optimizer=_optim,
        scheduler=_sched,
        output_dir=output_dir,
        job_name=job_name,
        resume=args.resume,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        env_eval_freq=0,
        log_freq=100,
        save_freq=args.save_freq,
        wandb=WandBConfig(enable=args.wandb, project="giava"),
    )

    train(cfg)


if __name__ == "__main__":
    main()
