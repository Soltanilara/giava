"""Launch an ACT + block->hole auxiliary-head run on the shape sorter, or its control.

    python train_sorterhead.py \\
        --root <shape_sorter_envstate run dir> \\
        --sorter-head-weight 1.0 --job-name sorter166_act_head_w1

    # control: same class, same data order, aux gradient off
    python train_sorterhead.py --root <same> --sorter-head-weight 0 \\
        --job-name sorter166_act_head_w0

`--root` MUST be the envstate tree: `observation.environment_state` is the
label.  It is never passed as a policy input here (there is deliberately no
--env-state flag); the input arm of the ablation is train_real.py --env-state
on the same root.  See act_sorterhead.py for what the head does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

## Registers "act_sorterhead" and patches lerobot's policy factory. Must
## precede any lerobot.configs import that resolves policy choices.
import act_sorterhead
import train_real


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True,
                    help="shape_sorter_envstate run dir (label source)")
    ap.add_argument("--cameras", nargs="+",
                    default=["right_wrist", "top_scene", "oak_left"])
    ap.add_argument("--sorter-head-weight", type=float, default=1.0,
                    help="0 for the control arm")
    ap.add_argument("--sorter-head-camera", default="top_scene",
                    help="which camera's feature map the head reads")
    ap.add_argument("--sorter-head-channels", type=int, default=32)
    ap.add_argument("--sorter-head-found-weight", type=float, default=1.0)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--n-action-steps", type=int, default=100,
                    help="deploy setting for this task is 100; see shape-sorter notes")
    ap.add_argument("--successes-only", action="store_true")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--save-freq", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--job-name", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not (root / "meta" / "info.json").exists():
        raise SystemExit(f"--root: no meta/info.json under {root}")

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    task = root.parent.name
    repo_id = f"giava/{task}"
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)

    env = ds_meta.features.get(act_sorterhead.ENV_KEY)
    if env is None:
        raise SystemExit(
            f"--root has no {act_sorterhead.ENV_KEY}. Point it at the "
            f"shape_sorter_envstate tree, not shape_sorter_canon.")
    names = env.get("names") or []
    want = ["obj_found", "obj_to_hole_dx", "obj_to_hole_dy"]
    have = [names[i] if i < len(names) else None
            for i in (act_sorterhead.IDX_FOUND, act_sorterhead.IDX_DX,
                      act_sorterhead.IDX_DY)]
    if have != want:
        raise SystemExit(f"env-state layout is {names}; expected indices "
                         f"2/6/7 to be {want}, got {have}")

    n_tasks = len(ds_meta.tasks)
    if n_tasks != 3:
        print(f"  WARNING: {n_tasks} tasks in dataset, head has 3 outputs")

    ## use_env_state=False is the whole point: the vector is a target here.
    input_features, output_features = train_real.select_features(
        ds_meta, args.cameras, use_ee_pose=False, use_env_state=False)

    episodes = (train_real.successful_episodes(root)
                if args.successes_only else None)
    if episodes is not None:
        print(f"  successes-only: {len(episodes)} of "
              f"{ds_meta.total_episodes} episodes")

    if args.n_action_steps > args.chunk_size:
        raise SystemExit(f"--n-action-steps {args.n_action_steps} > "
                         f"--chunk-size {args.chunk_size}")

    policy_cfg = act_sorterhead.ACTSorterHeadConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        push_to_hub=False,
        sorter_head_weight=args.sorter_head_weight,
        sorter_head_camera=f"observation.images.{args.sorter_head_camera}",
        sorter_head_n_pieces=3,
        sorter_head_channels=args.sorter_head_channels,
        sorter_head_found_weight=args.sorter_head_found_weight,
    )
    print(f"  auxiliary head: weight {args.sorter_head_weight} on "
          f"{policy_cfg.sorter_head_camera}, label from "
          f"{act_sorterhead.ENV_KEY}[2,6,7], conditioned on task_index")
    if args.sorter_head_weight == 0:
        print("  == CONTROL ARM: head built and scored, gradient off ==")

    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts.lerobot_train import train

    job_name = args.job_name or f"{task}_act_sorterhead"
    output_dir = Path(args.output_dir or f"outputs/train/{job_name}")

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
