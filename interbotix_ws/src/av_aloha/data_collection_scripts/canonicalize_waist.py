"""Put every episode's middle_base on ONE 2-pi branch, in a new dataset root.

    python canonicalize_waist.py --src <run_dir> --dst <new_run_dir>

WHY.  middle_base is a multi-turn joint whose driver frame can boot 2-pi
shifted (see middle_joint_offsets.json, "_middle_base"), and the recorder
stores DRIVER values.  In shape_sorter/20260907_204429 that split the session
in two: episodes 0-67 recorded the camera waist near -3.03 rad and episodes
68+ recorded the SAME physical poses near +3.26 rad, 2*pi apart.  Nothing
jumps inside an episode, so it is invisible until you look at the statistics:

    observation.state[middle_base]  std 2.768 rad   <- almost entirely the
                                                       branch flag
    real camera-arm motion          std 0.109 rad

lerobot normalizes with those stats, so the camera arm's actual motion would
reach the network divided by ~25x -- the policy would see which half of the
session an episode came from, loud, and the camera motion as noise.

WHAT IT DOES.  Picks the branch holding the most frames, maps every frame of
every other branch onto it by adding a multiple of 2*pi, and rewrites
observation.state and action.  The mapping is exact (a multiple of 2*pi is
the same physical angle), so no measurement is altered -- only its
representation.  Videos are symlinked; meta/ and the root files are copied;
meta/stats.json and the per-episode stats in meta/episodes/ are recomputed
for the two features that changed.

ROLLOUT MUST MATCH.  A policy trained on this convention needs live servo
reads canonicalized the same way, or it meets a 2*pi offset at deployment.
Use `canonical_branch()` from here on both sides -- do not re-derive it.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

TWO_PI = 2.0 * np.pi
JOINT = "middle_base"
FEATURES = ("observation.state", "action")


def canonical_branch(values, reference):
    """`values` (rad) moved by whole turns onto the branch of `reference`.

    Exact: adds a multiple of 2*pi, so the physical angle is untouched."""
    v = np.asarray(values, dtype=np.float64)
    return v - TWO_PI * np.round((v - reference) / TWO_PI)


def joint_index(src: Path) -> int:
    names = json.loads((src / "meta.json").read_text())["joint_names"]
    if JOINT not in names:
        raise SystemExit(f"{JOINT} not in {src}/meta.json joint_names={names}")
    return names.index(JOINT)


def read_parts(src: Path):
    files = sorted(glob.glob(str(src / "data" / "*" / "*.parquet")))
    if not files:
        raise SystemExit(f"no data parquet under {src}")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if dst.exists():
        raise SystemExit(f"{dst} already exists -- refusing to overwrite")

    j = joint_index(src)
    files = read_parts(src)

    ## The reference branch is the one holding the most frames, so the
    ## majority of the data keeps its recorded numbers exactly.
    allv = np.concatenate([
        np.stack(pd.read_parquet(f, columns=["observation.state"])
                 ["observation.state"].to_numpy())[:, j] for f in files])
    ref = float(np.median(allv[np.abs(allv - np.median(allv)) < np.pi]))
    print(f"[src] {src}\n[joint] {JOINT} (index {j})")
    print(f"[branch] reference {ref:+.3f} rad; before: "
          f"min {allv.min():+.3f} max {allv.max():+.3f} std {allv.std():.3f}")

    dst.mkdir(parents=True)
    shutil.copytree(src / "meta", dst / "meta")
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    ## Record the branch in the run's own meta.json.  rollout_policy.py reads
    ## it to canonicalize LIVE servo reads the same way, which is the only
    ## thing standing between a policy trained here and a 2*pi offset at
    ## deployment.  A dataset without this key was never canonicalized.
    mj = dst / "meta.json"
    m = json.loads(mj.read_text())
    m["waist_reference"] = float(ref)
    m["waist_joint"] = JOINT
    mj.write_text(json.dumps(m, indent=2))
    (dst / "videos").mkdir()
    for cam in (src / "videos").iterdir():
        (dst / "videos" / cam.name).symlink_to(cam.resolve())
    print(f"[dst] {dst}  (meta + root files copied, videos symlinked)")

    moved_eps, kept = set(), 0
    per_ep = {}
    for f in files:
        df = pd.read_parquet(f)
        for feat in FEATURES:
            M = np.stack(df[feat].to_numpy()).astype(np.float32)
            fixed = canonical_branch(M[:, j], ref)
            if feat == "observation.state":
                shift = np.abs(fixed - M[:, j]) > 1e-6
                for e in np.unique(df["episode_index"].to_numpy()[shift]):
                    moved_eps.add(int(e))
                kept += int((~shift).sum())
            M[:, j] = fixed.astype(np.float32)
            df[feat] = list(M)
        for e, g in df.groupby("episode_index"):
            per_ep[int(e)] = {
                feat: np.stack(g[feat].to_numpy()).astype(np.float64)
                for feat in FEATURES}
        out = dst / Path(f).relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"[data] {out.name}: {JOINT} canonicalized")

    after = np.concatenate([
        np.stack(pd.read_parquet(f, columns=["observation.state"])
                 ["observation.state"].to_numpy())[:, j]
        for f in sorted(glob.glob(str(dst / "data" / "*" / "*.parquet")))])
    print(f"[branch] after:  min {after.min():+.3f} max {after.max():+.3f} "
          f"std {after.std():.3f}")
    print(f"[branch] {len(moved_eps)} episode(s) moved, {kept} frames unchanged")

    ## Stats drive lerobot's normalization, so they must describe the data
    ## that is actually on disk now.  Only the two changed features are
    ## recomputed; every other entry is left exactly as recorded.
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    for feat in FEATURES:
        M = np.concatenate([per_ep[e][feat] for e in sorted(per_ep)])
        col = M[:, j]
        for key, val in (("mean", col.mean()), ("std", col.std()),
                         ("min", col.min()), ("max", col.max())):
            if key in stats.get(feat, {}):
                stats[feat][key][j] = float(val)
        for q, p in (("q01", 1), ("q99", 99)):
            if q in stats.get(feat, {}):
                stats[feat][q][j] = float(np.percentile(col, p))
        print(f"[stats] {feat}[{JOINT}] mean {col.mean():+.4f} std {col.std():.4f}")
    stats_path.write_text(json.dumps(stats, indent=4))

    ## Per-episode stats live in meta/episodes/*.parquet as
    ## stats/<feature>/<key> columns holding one list per episode.
    for f in sorted(glob.glob(str(dst / "meta" / "episodes" / "*" / "*.parquet"))):
        d = pd.read_parquet(f)
        changed = False
        for feat in FEATURES:
            for key in ("mean", "std", "min", "max", "q01", "q99"):
                col = f"stats/{feat}/{key}"
                if col not in d.columns:
                    continue
                vals = []
                for _, row in d.iterrows():
                    e = int(row["episode_index"])
                    arr = np.asarray(row[col], dtype=np.float64).copy()
                    if e in per_ep:
                        c = per_ep[e][feat][:, j]
                        arr[j] = {"mean": c.mean(), "std": c.std(),
                                  "min": c.min(), "max": c.max(),
                                  "q01": np.percentile(c, 1),
                                  "q99": np.percentile(c, 99)}[key]
                    vals.append(arr)
                d[col] = vals
                changed = True
        if changed:
            d.to_parquet(f, index=False)
            print(f"[meta] {Path(f).name}: per-episode stats updated")

    print("\nDone.  Train on this root; roll out with the same convention "
          "(canonical_branch()).")


if __name__ == "__main__":
    main()
