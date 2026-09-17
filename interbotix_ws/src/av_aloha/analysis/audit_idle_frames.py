"""How much of a recorded dataset is the arm sitting still?

WHY YOU WOULD RUN THIS
======================
Recording starts when the operator presses 'r', but teleoperation starts when
they put the headset on and squeeze a button.  Every frame in between is the
arm PARKED AT THE RESET POSE, recorded with `action` == that pose.  A few
seconds of it per episode, times a hundred episodes, and the reset pose becomes
the single most common action in the dataset -- at near-zero variance, paired
with observations of a scene that is not moving.

A behaviour-cloning policy trained on that learns "static scene -> reset pose"
as a high-confidence mapping, and reproduces it exactly whenever a rollout
reaches a state where nothing is changing, which in practice means right after
it finishes the task.  It looks like the reset motion leaked into the data.  It
did not: the reset pose was at the FRONT of every episode all along.

This script measures the effect so it is a number rather than a theory:

    python audit_idle_frames.py --dataset-root <run folder>
    python audit_idle_frames.py --task active_vision_data_collection

Reads the parquet directly -- no lerobot, no ROS, no video decoding -- so it
runs on a laptop against a copied dataset.

FIXING IT is GIAVA_RECORD_GATE=teleop in data_collection.py (now the default),
which simply does not write frames until teleop is first enabled.  For datasets
already recorded, the lead-in reported here is what you would trim.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pyarrow is required: pip install pyarrow")


def episode_files(root: Path):
    """Every episode parquet under a dataset root, in episode order.

    Handles both chunked (data/chunk-XXX/episode_YYYYYY.parquet) and flat
    layouts, since that path has changed across lerobot versions."""
    files = sorted((root / "data").rglob("episode_*.parquet"))
    if not files:
        files = sorted(root.rglob("episode_*.parquet"))
    return files


def leading_run(actions: np.ndarray, tol: float) -> int:
    """How many frames at the START hold within tol of the first action."""
    if len(actions) == 0:
        return 0
    d = np.linalg.norm(actions - actions[0], axis=1)
    moved = np.nonzero(d > tol)[0]
    return int(moved[0]) if len(moved) else len(actions)


def trailing_run(actions: np.ndarray, tol: float) -> int:
    if len(actions) == 0:
        return 0
    d = np.linalg.norm(actions - actions[-1], axis=1)
    moved = np.nonzero(d > tol)[0]
    return int(len(actions) - moved[-1] - 1) if len(moved) else len(actions)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument("--base-root", type=str, default=None,
                    help="Defaults to data_col_config.DATASET_ROOT.")
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--run", type=str, default="latest")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="Joint-space L2 distance (rad) below which two "
                         "actions count as the same pose. Default 0.01 -- "
                         "well under teleop motion, well over encoder noise.")
    ap.add_argument("--fps", type=float, default=50.0,
                    help="Only used to print seconds alongside frame counts.")
    ap.add_argument("--per-episode", action="store_true",
                    help="One line per episode as well as the summary.")
    args = ap.parse_args()

    if args.dataset_root:
        root = Path(args.dataset_root).expanduser().resolve()
    else:
        if not args.task:
            sys.exit("give --dataset-root, or --task (and optionally --run)")
        base = args.base_root
        if base is None:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from data_col_config import DATASET_ROOT
            base = DATASET_ROOT
        task_dir = Path(base).expanduser().resolve() / args.task
        if not task_dir.is_dir():
            sys.exit(f"no such task directory: {task_dir}")
        runs = [d for d in task_dir.iterdir() if d.is_dir()]
        if not runs:
            sys.exit(f"no runs under {task_dir}")
        root = (sorted(runs)[-1] if args.run == "latest"
                else task_dir / args.run).resolve()

    files = episode_files(root)
    if not files:
        sys.exit(f"no episode parquet files under {root}")

    print(f"dataset: {root}")
    print(f"{len(files)} episode(s), tol={args.tol} rad\n")

    tot = lead = tail = 0
    ## Fraction of the WHOLE dataset within tol of its own episode's first
    ## action -- the number that matters, because it is how often "output the
    ## reset pose" was the correct answer during training.
    at_start_pose = 0
    leads, tails = [], []

    if args.per_episode:
        print(f"{'episode':>10} {'frames':>7} {'lead':>6} {'lead_s':>7} "
              f"{'tail':>6} {'tail_s':>7} {'@start':>7}")

    for f in files:
        actions = np.asarray(
            pq.read_table(f, columns=["action"])["action"].to_pylist(),
            dtype=np.float64)
        if actions.ndim != 2 or len(actions) == 0:
            print(f"  skipping {f.name}: unexpected action shape "
                  f"{actions.shape}")
            continue

        n = len(actions)
        l = leading_run(actions, args.tol)
        t = trailing_run(actions, args.tol)
        near = int((np.linalg.norm(actions - actions[0], axis=1)
                    <= args.tol).sum())

        tot += n
        lead += l
        tail += t
        at_start_pose += near
        leads.append(l)
        tails.append(t)

        if args.per_episode:
            print(f"{f.stem.split('_')[-1]:>10} {n:>7} {l:>6} "
                  f"{l / args.fps:>7.1f} {t:>6} {t / args.fps:>7.1f} "
                  f"{100 * near / n:>6.1f}%")

    if not tot:
        sys.exit("no usable frames found")

    def pct(x):
        return f"{100 * x / tot:.1f}%"

    print(f"\n{'-' * 62}")
    print(f"total frames                      {tot}")
    print(f"parked at the START of an episode {lead:>8}  {pct(lead):>7}  "
          f"({lead / args.fps:.0f} s)   <- the lead-in")
    print(f"held at the END of an episode     {tail:>8}  {pct(tail):>7}  "
          f"({tail / args.fps:.0f} s)")
    print(f"anywhere within tol of the        {at_start_pose:>8}  "
          f"{pct(at_start_pose):>7}")
    print("  episode's own first action")
    if leads:
        print(f"\nlead-in per episode: median {np.median(leads):.0f} frames "
              f"({np.median(leads) / args.fps:.1f} s), "
              f"max {max(leads)} ({max(leads) / args.fps:.1f} s)")
        print(f"tail    per episode: median {np.median(tails):.0f} frames "
              f"({np.median(tails) / args.fps:.1f} s), "
              f"max {max(tails)} ({max(tails) / args.fps:.1f} s)")

    share = at_start_pose / tot
    print()
    if share > 0.10:
        print(f"VERDICT: {pct(at_start_pose)} of every action label in this "
              f"dataset is the episode's starting pose.")
        print("A policy trained on this will reproduce that pose whenever the "
              "observation is\nambiguous or the scene stops moving -- which "
              "is what 'it goes back to reset when\nit finishes' looks like. "
              "Re-record with GIAVA_RECORD_GATE=teleop (now the default),\n"
              "or trim the lead-in reported above from the existing episodes.")
    else:
        print(f"VERDICT: only {pct(at_start_pose)} of actions sit at the "
              "starting pose -- the lead-in is\nnot large enough to explain a "
              "reliable return-to-reset. Look elsewhere.")


if __name__ == "__main__":
    main()
