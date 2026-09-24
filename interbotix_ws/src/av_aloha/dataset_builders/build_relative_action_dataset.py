"""Rewrite `action` as a per-step DELTA, in a new dataset root.

WHY, and the design decision that matters
-----------------------------------------
GIAVA's `action` column is the absolute commanded joint position.  Measured on
transfer_flower_merged:

    mean |action|                  = 0.427   rad
    mean |action[t] - action[t-1]| = 0.00212 rad   (per tick @ 50 Hz)
                             ratio ~ 200 : 1

The motion that matters is 0.5% of the number the network predicts, so a 1%
error in the absolute prediction is a 200% error in the movement.  Predicting
displacement instead makes the target placement-invariant: the same reach
executed at the left and right edges of the table becomes the same action
sequence, which is the generalization property we actually want.

THE CHUNKING CONSTRAINT -- read before changing the formulation.

The obvious encoding is "delta from the state at chunk start", a[t] - s[t0].
It cannot be precomputed here.  A LeRobot dataset stores one action per ROW,
and ACT's `delta_timestamps` gathers a window at load time, so t0 differs for
every training sample drawn from the same row.  Storing a[t] - s[t] (each row
relative to its own state) is worse than useless: reconstructing it at rollout
would need s[t] for future t, which is exactly what is not known.

So this writes the one encoding that is exactly invertible from information
available at rollout time -- the per-step delta:

    action_rel[t] = action[t] - action[t-1]      within an episode
    action_rel[0] = action[0] - observation.state[0]

At rollout the commanded position is recovered by cumulative sum from the
current measured state.  That integration is why `n_action_steps` MUST be
dropped well below `chunk_size` for a policy trained on this: all four
flower128 policies run chunk_size = n_action_steps = 100, i.e. fully open-loop
for 2 s at 50 Hz, and integrating 100 predicted deltas open-loop compounds
drift.  Use --n-action-steps 20 or lower.  See train_real.py.

Scale is handled for free: unlike `observation.environment_state`, the ACTION
key IS in ACT's normalization_mapping (MEAN_STD), so LeRobot normalizes these
small deltas to unit variance automatically.  Do not pre-scale them here.

    python build_relative_action_dataset.py --src <run_dir> --dst <new_run_dir>
    python build_relative_action_dataset.py --src ... --dst ... --verify
"""
from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ACTION = "action"
STATE = "observation.state"


def destaircase(actions: np.ndarray, episode_index: np.ndarray,
                cols) -> np.ndarray:
    """Linearly interpolate the commanded joints between command UPDATES.

    WHY.  The teleop command is written every 50 Hz tick but only CHANGES
    when the headset/IK pipeline produces a new target -- measured on
    transfer_flower_merged: 71% of base-session ticks and 65% of hull ticks
    carry exactly the previous command (effective update rate ~27 Hz, and
    lower in the earliest sessions).  Differencing that staircase gives a
    delta stream that is mostly exact zeros, and an L1 regression on it
    learns the median: zero.  The first delta policy (flower128_act_rel_rgb,
    2026-09-19) did exactly that -- predicted |d| 0.00002 rad on frames
    whose chunks were two-thirds stationary, and matched ground truth to a
    ratio of 1.02 on the one frame with continuous motion.  Verified with
    the checkpoint on its own training frames.

    Interpolating between updates recovers the continuous trajectory the
    operator actually traced; every tick then carries a real displacement.
    Genuine pauses (the arm truly held still) survive as runs of unchanged
    commands that are ALSO unchanged in the measured state, so the
    interpolation is only applied to runs shorter than `max_hold` ticks.
    """
    out = actions.astype(np.float64).copy()
    starts = np.flatnonzero(np.r_[True, episode_index[1:] != episode_index[:-1]])
    ends = np.r_[starts[1:], len(actions)]
    for s0, s1 in zip(starts, ends):
        for c in cols:
            x = out[s0:s1, c]
            chg = np.flatnonzero(np.r_[True, x[1:] != x[:-1]])     # ticks where the command changed
            if len(chg) < 2:
                continue
            ## only bridge gaps up to MAX_HOLD ticks; longer holds are real
            keep = [chg[0]]
            for a, b in zip(chg[:-1], chg[1:]):
                if b - a <= MAX_HOLD:
                    pass
                else:
                    keep.append(b - 1)      # end of a genuine hold: anchor there
                keep.append(b)
            keep = np.unique(keep)
            out[s0:s1, c] = np.interp(np.arange(s1 - s0), keep, x[keep])
    return out


MAX_HOLD = 6   # ticks (120 ms); command gaps longer than this are treated as real pauses


def to_relative(actions: np.ndarray, states: np.ndarray,
                episode_index: np.ndarray, absolute_cols=()) -> np.ndarray:
    """Per-step deltas, reset at every episode boundary.

    Columns listed in `absolute_cols` are copied through unchanged.  The
    gripper belongs there: it is a binary open/close command, not a
    trajectory, so differencing it yields a mostly-zero column punctuated by
    +/-1.5 spikes (437 of them in transfer_flower_merged).  ACTION is
    normalized MEAN_STD, so those spikes land ~20 sigma out and become a
    target an L1 regression cannot hit -- while the informative signal, WHEN
    to close, is destroyed.  Keep it absolute.
    """
    rel = np.zeros_like(actions)
    starts = np.flatnonzero(np.r_[True, episode_index[1:] != episode_index[:-1]])
    start_set = set(starts.tolist())
    for i in range(len(actions)):
        if i in start_set:
            ## First frame: delta from the MEASURED state, which is the only
            ## thing a rollout knows before it has commanded anything.
            rel[i] = actions[i] - states[i]
        else:
            rel[i] = actions[i] - actions[i - 1]
    for c in absolute_cols:
        rel[:, c] = actions[:, c]
    return rel


def to_absolute(rel: np.ndarray, states: np.ndarray,
                episode_index: np.ndarray, absolute_cols=()) -> np.ndarray:
    """Inverse of to_relative -- the operation a rollout performs."""
    out = np.zeros_like(rel)
    starts = np.flatnonzero(np.r_[True, episode_index[1:] != episode_index[:-1]])
    start_set = set(starts.tolist())
    for i in range(len(rel)):
        if i in start_set:
            out[i] = states[i] + rel[i]
        else:
            out[i] = out[i - 1] + rel[i]
    for c in absolute_cols:
        out[:, c] = rel[:, c]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--absolute-cols", type=int, nargs="*", default=None,
                    metavar="J",
                    help="joint indices left ABSOLUTE rather than differenced. "
                         "Default: the last column, i.e. the gripper -- see "
                         "to_relative().  Pass an empty list to difference "
                         "everything.")
    ap.add_argument("--no-destaircase", action="store_true",
                    help="difference the raw command staircase (the 2026-09-19 "
                         "v1 behaviour; produces a mostly-zero delta stream -- "
                         "see destaircase())")
    ap.add_argument("--verify", action="store_true",
                    help="reintegrate the deltas and compare against the "
                         "original absolute actions")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise SystemExit(f"{dst} already exists -- refusing to overwrite")

    pq = sorted(glob.glob(str(src / "data" / "*" / "*.parquet")))
    if not pq:
        raise SystemExit(f"no parquet under {src}/data")

    frames = pd.concat([pd.read_parquet(f, columns=["episode_index"]) for f in pq],
                       ignore_index=True)
    print(f"[src] {src}  ({len(pq)} parquet file(s), {len(frames)} frames)")

    ## Copy meta + root files, symlink videos -- same layout convention as
    ## build_envstate_dataset.py so the two variants compose.
    dst.mkdir(parents=True)
    shutil.copytree(src / "meta", dst / "meta")
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    (dst / "videos").mkdir()
    for cam_dir in (src / "videos").iterdir():
        (dst / "videos" / cam_dir.name).symlink_to(cam_dir.resolve())
    print(f"[dst] {dst}  (meta + root files copied, videos symlinked)")

    all_rel = []
    for f in pq:
        df = pd.read_parquet(f)
        A = np.stack(df[ACTION].values).astype(np.float64)
        S = np.stack(df[STATE].values).astype(np.float64)
        ep = df["episode_index"].to_numpy()
        if A.shape[1] != S.shape[1]:
            raise SystemExit(
                f"action is {A.shape[1]}-d but state is {S.shape[1]}-d -- "
                "relative actions need them in the same space.  This script "
                "is for JOINT-space runs; an EE-pose action column would need "
                "quaternion-aware differencing, which is not implemented.")
        abs_cols = ([A.shape[1] - 1] if args.absolute_cols is None
                    else list(args.absolute_cols))
        A_src = A
        if not args.no_destaircase:
            joint_cols = [c for c in range(A.shape[1]) if c not in abs_cols]
            A_src = destaircase(A, ep, joint_cols)
            z_before = np.mean(np.abs(np.diff(A[:, joint_cols], axis=0)).sum(1) == 0)
            z_after = np.mean(np.abs(np.diff(A_src[:, joint_cols], axis=0)).sum(1) == 0)
            print(f"[deltas] de-staircased: zero-delta ticks {z_before:.1%} -> {z_after:.1%}  "
                  f"(interpolated command deviates from the recorded one by "
                  f"max {np.abs(A_src - A).max():.4f} rad)")
        rel = to_relative(A_src, S, ep, abs_cols)

        if args.verify:
            back = to_absolute(rel, S, ep, abs_cols)
            err = np.abs(back - A_src)
            print(f"[verify] {Path(f).name}: max reintegration error "
                  f"{err.max():.3e} rad  (mean {err.mean():.3e})")
            if err.max() > 1e-6:
                raise SystemExit("reintegration does not reproduce the "
                                 "original actions -- refusing to write")

        df[ACTION] = list(rel.astype(np.float32))
        out = dst / Path(f).relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        all_rel.append(rel)
        print(f"[data] {out.name}: action -> per-step delta {rel.shape}")

    R = np.concatenate(all_rel)

    ## info.json: keep the shape, rename the columns so nothing downstream
    ## mistakes these for absolute positions.
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    feat = info["features"][ACTION]
    if feat.get("names"):
        feat["names"] = [n if j in abs_cols else f"d_{n}"
                         for j, n in enumerate(feat["names"])]
    info["action_space"] = "joint_delta"
    info["delta_absolute_cols"] = list(abs_cols)
    info_path.write_text(json.dumps(info, indent=4))
    print(f"[meta] info.json: action_space=joint_delta, "
          f"absolute columns kept: {abs_cols}")

    ## stats.json MUST be recomputed -- ACTION is normalized with MEAN_STD, so
    ## stale absolute-position statistics would destroy the signal.
    stats_path = dst / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        stats[ACTION] = {
            "mean": R.mean(0).tolist(),
            "std": (R.std(0) + 1e-8).tolist(),
            "min": R.min(0).tolist(),
            "max": R.max(0).tolist(),
            "count": [int(R.shape[0])],
        }
        stats_path.write_text(json.dumps(stats, indent=4))
        print("[meta] stats.json: action statistics recomputed")

    ## Per-episode stats live in meta/episodes/*.parquet and are also stale.
    ep_files = sorted(glob.glob(str(dst / "meta" / "episodes" / "**" / "*.parquet"),
                                recursive=True))
    ep_all = pd.concat([pd.read_parquet(f, columns=["episode_index"]) for f in pq],
                       ignore_index=True)["episode_index"].to_numpy()
    for f in ep_files:
        edf = pd.read_parquet(f)
        cols = [c for c in edf.columns if c.startswith(f"stats/{ACTION}/")]
        if not cols:
            continue
        for _i, row in edf.iterrows():
            e = int(row["episode_index"])
            blk = R[ep_all == e]
            if not len(blk):
                continue
            for c in cols:
                which = c.split("/")[-1]
                if which == "mean":
                    edf.at[_i, c] = blk.mean(0).tolist()
                elif which == "std":
                    edf.at[_i, c] = (blk.std(0) + 1e-8).tolist()
                elif which == "min":
                    edf.at[_i, c] = blk.min(0).tolist()
                elif which == "max":
                    edf.at[_i, c] = blk.max(0).tolist()
                elif which == "count":
                    edf.at[_i, c] = [int(len(blk))]
                elif which.startswith("q"):
                    q = int(which[1:])
                    edf.at[_i, c] = np.percentile(blk, q, axis=0).tolist()
        edf.to_parquet(f, index=False)
        print(f"[meta] {Path(f).name}: per-episode action stats recomputed")

    print(f"\naction delta magnitude per joint (rad/tick @ 50 Hz):")
    names = info["features"][ACTION].get("names") or [f"d{j}" for j in range(R.shape[1])]
    for j, n in enumerate(names):
        print(f"  {n:24s} mean|d|={np.abs(R[:, j]).mean():.5f}  "
              f"std={R[:, j].std():.5f}  max|d|={np.abs(R[:, j]).max():.4f}")
    print(f"\noverall mean |delta| = {np.abs(R).mean():.5f} rad  "
          f"(absolute action mean |a| was ~0.427 rad)")
    print("\nREMINDER: train this with --n-action-steps well below --chunk-size "
          "(try 20 vs 100).  Integrating 100 predicted deltas open-loop drifts.")


if __name__ == "__main__":
    main()
