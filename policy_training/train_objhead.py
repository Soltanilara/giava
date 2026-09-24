"""Launch an ACT + auxiliary-object-head run (or its weight-0 control).

Deliberately a SEPARATE launcher rather than a `--policy act_objhead` branch
inside train_real.py: this adds no lines to a file that is being read and
re-commented right now, and it reuses train_real's feature selection and
episode filtering by import, so the two cannot drift on what a policy is
allowed to see.

    python train_objhead.py --root <mask_dataset_run_dir> \
        --cameras right_wrist top_scene low_scene \
        --obj-head-weight 1.0 --job-name flower_objhead_joint

The control arm is the same command with --obj-head-weight 0.

`--root` MUST point at a dataset built by build_mask_dataset.py: the label
stream `observation.images.obj_mask` rides in the dataset and is never passed
to --cameras, so it is a target and not an input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

## Registers "act_objhead" and patches lerobot's policy factory. Must precede
## any lerobot.configs import that resolves policy choices.
import act_objhead
import train_real


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True,
                    help="dataset run dir built by build_mask_dataset.py")
    ap.add_argument("--cameras", nargs="+",
                    default=["right_wrist", "top_scene", "low_scene"],
                    help="policy input cameras; NEVER include obj_mask")
    ap.add_argument("--obj-head-weight", type=float, default=1.0,
                    help="0 for the control arm")
    ap.add_argument("--obj-head-camera", default="top_scene",
                    help="which camera's feature map the head attaches to")
    ap.add_argument("--obj-head-pos-weight", type=float, default=150.0,
                    help="~n_neg/n_pos on this dataset; see act_objhead.py")
    ap.add_argument("--obj-head-dilate", type=int, default=0,
                    help="grow the target by N cells before the loss")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="required < chunk-size for a delta-action dataset")
    ap.add_argument("--successes-only", action="store_true")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--save-freq", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--job-name", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    if "obj_mask" in args.cameras:
        raise SystemExit(
            "obj_mask is the auxiliary LABEL, not a policy input. Drop it "
            "from --cameras, or use train_real.py if you meant the "
            "mask-as-fourth-camera variant.")

    root = Path(args.root).resolve()
    if not (root / "meta" / "info.json").exists():
        raise SystemExit(f"--root: no meta/info.json under {root}")

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    task = root.parent.name
    repo_id = f"giava/{task}"
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)

    if act_objhead.OBJ_MASK_KEY not in ds_meta.features:
        raise SystemExit(
            f"--root has no {act_objhead.OBJ_MASK_KEY}. Build it first:\n"
            f"  python build_mask_dataset.py --src {root} --dst <new_root>")

    input_features, output_features = train_real.select_features(
        ds_meta, args.cameras, use_ee_pose=False)

    episodes = (train_real.successful_episodes(root)
                if args.successes_only else None)
    if episodes is not None:
        print(f"  successes-only: {len(episodes)} of "
              f"{ds_meta.total_episodes} episodes")

    fps = ds_meta.fps
    chunk = args.chunk_size or max(20, int(round(2.0 * fps)))
    n_act = args.n_action_steps or chunk
    if n_act > chunk:
        raise SystemExit(f"--n-action-steps {n_act} > --chunk-size {chunk}")

    policy_cfg = act_objhead.ACTObjHeadConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk,
        n_action_steps=n_act,
        push_to_hub=False,
        obj_head_weight=args.obj_head_weight,
        obj_head_camera=f"observation.images.{args.obj_head_camera}",
        obj_head_pos_weight=args.obj_head_pos_weight,
        obj_head_dilate=args.obj_head_dilate,
    )
    print(f"  auxiliary head: weight {args.obj_head_weight} on "
          f"{policy_cfg.obj_head_camera} "
          f"(pos_weight {args.obj_head_pos_weight})")
    if args.obj_head_weight == 0:
        print("  == CONTROL ARM: head built and scored, gradient off ==")

    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts.lerobot_train import train

    job_name = args.job_name or f"{task}_act_objhead"
    output_dir = Path(args.output_dir or f"outputs/train/{job_name}")

    ## Same resume contract as train_real.py: lerobot re-reads --config_path
    ## off sys.argv, and the optimizer preset must be spelled out because this
    ## config is built in Python rather than decoded from the saved json.
    _optim = _sched = None
    if args.resume:
        ckpt = output_dir / "checkpoints" / "last" / "pretrained_model"
        if not ckpt.exists():
            raise SystemExit(f"--resume: no checkpoint at {ckpt}")
        sys.argv.append(f"--config_path={ckpt / 'train_config.json'}")
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
