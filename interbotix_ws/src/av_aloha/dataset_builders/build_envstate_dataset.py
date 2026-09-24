"""Add observation.environment_state to a LeRobot dataset, in a new root.

Videos are SYMLINKED (1.5 GB) and only meta/ + data/ are copied, so a variant
costs a few MB.  The features come from scene_features.py -- the same module
rollout_policy.py imports, so train and deploy cannot drift.

    python build_envstate_dataset.py --src <run_dir> --dst <new_run_dir>                  # flower scene
    python build_envstate_dataset.py --src <run_dir> --dst <new_run_dir> --scene shape_sorter

shape_sorter: the tracked piece and its hole change per episode, read from
each frame's task string (shape_sorter.target_from_task), so every episode in
the run must have been recorded with --target.
"""
from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

import scene_features  # noqa: E402
import shape_sorter  # noqa: E402
import wrist_features  # noqa: E402

KEY = "observation.environment_state"


class _OneHotScene:
    """Module-shaped view of shape_sorter's one-hot extractor, so the scene
    table below stays uniform (FEATURE_NAMES / FEATURE_DIM / Extractor)."""
    FEATURE_NAMES = shape_sorter.ONEHOT_FEATURE_NAMES
    FEATURE_DIM = shape_sorter.ONEHOT_FEATURE_DIM
    Extractor = shape_sorter.OneHotExtractor
    target_from_task = staticmethod(shape_sorter.target_from_task)


SCENES = {"flower": scene_features, "shape_sorter": shape_sorter,
          "shape_sorter_onehot": _OneHotScene, "wrist": wrist_features}


def _frames(src: Path, camera: str):
    """Decode one camera's frames in dataset row order."""
    paths = sorted(glob.glob(
        str(src / "videos" / f"observation.images.{camera}" / "*" / "*.mp4")))
    if not paths:
        raise SystemExit(f"no {camera} video under {src}")
    for p in paths:
        c = av.open(p)
        for frame in c.decode(video=0):
            yield frame.to_ndarray(format="rgb24")
        c.close()


def report_rank(feats: np.ndarray, names) -> int:
    """Print the numerical rank of the feature block and name the degenerate
    dimensions.

    This exists because the original 12-d top_scene vector turned out to have
    rank 4 -- four dimensions constant, four affine copies of another two --
    and nothing in the pipeline noticed for two weeks of training runs.  A
    hand-engineered feature vector should never reach a policy without this
    check.
    """
    ## float64 ON PURPOSE.  The column is stored float32, and an exactly
    ## affine relationship between dimensions leaves a residual singular value
    ## around 0.04 at that precision -- enough to report rank 5 for a vector
    ## whose true rank is 4.  Rank is a question about the data, not about the
    ## storage format.
    feats = np.asarray(feats, dtype=np.float64)
    sd = feats.std(0)
    const = [names[i] for i in range(len(names)) if sd[i] < 1e-9]
    centred = feats - feats.mean(0)
    sv = np.linalg.svd(centred, compute_uv=False)
    rank = int((sv > sv[0] * 1e-6).sum()) if sv[0] > 0 else 0
    print(f"\n[rank] numerical rank {rank} of {len(names)}")
    print(f"[rank] singular values: "
          f"{np.array2string(sv, precision=2, suppress_small=True)}")
    if const:
        print(f"[rank] CONSTANT dimensions (zero information): {const}")
    if rank < len(names):
        print(f"[rank] WARNING: {len(names) - rank} dimension(s) are linearly "
              f"dependent -- they cost parameters and carry nothing.")
    else:
        print("[rank] full rank: every dimension carries independent signal.")
    return rank


def row_columns(pq_files):
    """(episode_index, task_index) per row, in the order extract() decodes."""
    eps, tis = [], []
    for f in pq_files:
        df = pd.read_parquet(f, columns=["episode_index", "task_index"])
        eps.append(df["episode_index"].to_numpy())
        tis.append(df["task_index"].to_numpy())
    return np.concatenate(eps), np.concatenate(tis)


def row_targets(src: Path, task_index: np.ndarray):
    """Per-row target piece for the shape_sorter scene, via
    task_index -> meta/tasks.parquet -> task string -> piece."""
    tasks = pd.read_parquet(src / "meta" / "tasks.parquet")
    idx_to_piece = {int(row["task_index"]): shape_sorter.target_from_task(str(task))
                    for task, row in tasks.iterrows()}
    bad = sorted({str(t) for t, row in tasks.iterrows()
                  if idx_to_piece[int(row["task_index"])] is None})
    if bad:
        raise SystemExit(
            f"task string(s) {bad} name no shape_sorter piece -- was this run "
            f"recorded with --task shape_sorter --target <piece>?")
    return np.array([idx_to_piece[int(t)] for t in task_index], dtype=object)


def extract(src: Path, scene: str, episode_index: np.ndarray,
            targets) -> np.ndarray:
    """One feature vector per frame, in dataset row order.  The extractor's
    hold-last state resets at every episode boundary (and, for shape_sorter,
    the tracked piece follows the row's target)."""
    mod = SCENES[scene]
    n_expected = len(episode_index)
    out = np.zeros((n_expected, mod.FEATURE_DIM), dtype=np.float32)

    ## The wrist scene needs two streams decoded in lockstep; every other
    ## scene reads top_scene alone.
    two_cams = getattr(mod, "NEEDS_TWO_CAMERAS", False)
    if two_cams:
        streams = zip(_frames(src, mod.CAMERAS[0]), _frames(src, mod.CAMERAS[1]))
    else:
        streams = ((f, None) for f in _frames(src, "top_scene"))

    i = 0
    ex = None
    for a, b in streams:
        if i >= n_expected:
            break
        new_ep = i == 0 or episode_index[i] != episode_index[i - 1]
        if targets is not None:
            if ex is None or new_ep or ex.target != targets[i]:
                ex = mod.Extractor(targets[i])
        elif ex is None:
            ex = mod.Extractor()
        elif new_ep:
            ex.reset()
        out[i] = ex(a, b) if two_cams else ex(a)
        i += 1
    if i != n_expected:
        raise SystemExit(f"decoded {i} frames but dataset has {n_expected}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--scene", default="flower", choices=sorted(SCENES),
                    help="which extractor: flower (blue block -> printed "
                         "targets); shape_sorter (per-episode piece centroid "
                         "+ its hole, needs assets/shape_sorter_targets.json); "
                         "shape_sorter_onehot (target class one-hot only -- "
                         "'which', not 'where')")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    sf = SCENES[args.scene]
    if dst.exists():
        raise SystemExit(f"{dst} already exists -- refusing to overwrite")

    pq = sorted(glob.glob(str(src / "data" / "*" / "*.parquet")))
    episode_index, task_index = row_columns(pq)
    frames = len(episode_index)
    print(f"[src] {src}  ({len(pq)} parquet file(s), {frames} frames, "
          f"scene={args.scene})")

    targets = None
    if args.scene.startswith("shape_sorter"):
        targets = row_targets(src, task_index)
        counts = {p: int((targets == p).sum()) for p in shape_sorter.CLASS_NAMES}
        print(f"[targets] frames per piece: {counts}")

    feats = extract(src, args.scene, episode_index, targets)
    print(f"[features] {feats.shape}  found-rate="
          f"{feats[:, sf.FEATURE_NAMES.index('obj_found')].mean():.1%}")
    report_rank(feats, sf.FEATURE_NAMES)

    ## meta + data are copied; videos are symlinked.  Root-level files
    ## (meta.json, teleop_config.json, episode_outcomes.jsonl, robustness
    ## logs) come along too -- train_real.py and rollout_policy.py read them
    ## from the run dir, and the first variant had to have them copied by hand.
    dst.mkdir(parents=True)
    shutil.copytree(src / "meta", dst / "meta")
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    (dst / "videos").mkdir()
    for cam_dir in (src / "videos").iterdir():
        (dst / "videos" / cam_dir.name).symlink_to(cam_dir.resolve())
    print(f"[dst] {dst}  (meta + root files copied, videos symlinked)")

    ## Reset the hold-last carry at every episode boundary: a position held
    ## across a cut would describe the PREVIOUS episode's scene.
    off = 0
    for f in pq:
        df = pd.read_parquet(f)
        block = feats[off:off + len(df)].copy()
        ep = df["episode_index"].to_numpy()
        starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
        found_i = sf.FEATURE_NAMES.index("obj_found")
        ## On the first frame of an episode there is no previous vector to
        ## hold, so a miss must fall back to the scene's documented
        ## no-detection value.  These are NOT all zero: for the wrist scene
        ## `w_range` = 0 would mean "at grasp distance", i.e. a miss would be
        ## encoded as a perfect grasp.  MISS_VECTOR is defined by each scene
        ## module; the extractor already produces it (reset() clears the
        ## hold-last carry), so this is a guard, not the primary path.
        miss = np.asarray(getattr(sf, "MISS_VECTOR", None)
                          if getattr(sf, "MISS_VECTOR", None) is not None
                          else np.zeros(sf.FEATURE_DIM), dtype=np.float32)
        for s in starts:
            if block[s, found_i] == 0.0:
                keep = block[s, found_i]
                block[s] = miss
                block[s, found_i] = keep
        df[KEY] = list(block.astype(np.float32))
        out = dst / Path(f).relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"[data] {out.name}: +{KEY} {block.shape}")
        off += len(df)

    ## info.json: declare the feature so dataset_to_policy_features types it
    ## ENV (it keys off this exact name).
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"][KEY] = {
        "dtype": "float32",
        "shape": [sf.FEATURE_DIM],
        "names": list(sf.FEATURE_NAMES),
    }
    info_path.write_text(json.dumps(info, indent=4))
    print(f"[meta] info.json: declared {KEY} {sf.FEATURE_DIM}d")

    ## stats.json: written for completeness/inspection.  ACT's
    ## normalization_mapping has no ENV entry, so these are NOT applied --
    ## the features are hand-scaled in scene_features.py instead.
    stats_path = dst / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        stats[KEY] = {
            "mean": feats.mean(0).tolist(),
            "std": (feats.std(0) + 1e-8).tolist(),
            "min": feats.min(0).tolist(),
            "max": feats.max(0).tolist(),
            "count": [int(feats.shape[0])],
        }
        stats_path.write_text(json.dumps(stats, indent=4))
        print("[meta] stats.json updated")

    print("\nper-feature range (must sit near [-1, 1] -- ENV is NOT normalized):")
    for j, name in enumerate(sf.FEATURE_NAMES):
        print(f"  {name:18s} min={feats[:, j].min():7.3f} "
              f"mean={feats[:, j].mean():7.3f} max={feats[:, j].max():7.3f}")


if __name__ == "__main__":
    main()
