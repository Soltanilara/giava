"""Rebuild our v2.1 depth datasets in the lerobot v3.0 format.

Why the stock converter cannot do this
--------------------------------------
Our v2.1 depth datasets store each depth map as a raw 480x640 float32 array in a
parquet column. That breaks `convert_dataset_v21_to_v30.py` two ways:

1. pyarrow rejects the nested float32 arrays
   ("Did not pass numpy.dtype object ... observation.depth.right_wrist").
2. Per-episode stats are computed element-wise over the whole depth map, so
   `episodes_stats.jsonl` balloons (948 MB for 21 episodes in one dataset) and
   blows the 100 MB single-file limit for episode metadata.

v0.6.0 has a real depth pipeline instead: depth features declared with
`info={"is_depth_map": True}` are encoded as 12-bit lossless HEVC (gray12le) via
`DepthEncoderConfig`, with log quantization between depth_min and depth_max.

So rather than patching the old layout through, this script re-records each
dataset: it reads the v2.1 parquet + RGB videos and writes a fresh v3.0 dataset
with depth as a proper depth video stream.

Originals are moved aside to <name>_old_v21 and never deleted.

Usage:
    python convert_depth_datasets.py --list
    python convert_depth_datasets.py --only block_square/20260528_130259
    python convert_depth_datasets.py
"""

import argparse
import json
import shutil
import traceback
from pathlib import Path

import av
import numpy as np
import pandas as pd

from lerobot.configs.video import DepthEncoderConfig
from lerobot.datasets import LeRobotDataset

ROOT = Path(__file__).resolve().parent / "dataset/lerobot"
BACKUP_SUFFIX = "_old_v21"

DEPTH_DATASETS = [
    "block_square/20260528_183732",
    "block_square/20260528_185043",
    "block_square/20260528_164157",
    "block_square/20260528_131838",
    "grasp cube/20260528_235229",
]


def load_v21_info(root: Path) -> dict:
    return json.loads((root / "meta/info.json").read_text())


def build_v3_features(info: dict) -> tuple[dict, list[str], list[str]]:
    """Translate v2.1 features into v3.0 features.

    Depth columns become depth *video* features; RGB video features stay video.
    """
    features: dict[str, dict] = {}
    rgb_keys: list[str] = []
    depth_keys: list[str] = []

    for key, ft in info["features"].items():
        if key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            continue

        if ft["dtype"] == "video":
            rgb_keys.append(key)
            features[key] = {
                "dtype": "video",
                "shape": tuple(ft["shape"]),
                "names": ft.get("names"),
            }
        elif key.startswith("observation.depth.") and ft["dtype"] == "float32":
            # 480x640 float32 depth map -> 12-bit lossless depth video
            depth_keys.append(key)
            h, w = ft["shape"][0], ft["shape"][1]
            features[key] = {
                "dtype": "video",
                "shape": (h, w, 1),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": True},
            }
        else:
            features[key] = {
                "dtype": ft["dtype"],
                "shape": tuple(ft["shape"]),
                "names": ft.get("names"),
            }

    return features, rgb_keys, depth_keys


def decode_episode_video(
    root: Path, info: dict, key: str, ep: int
) -> tuple[list[np.ndarray], float]:
    """Decode one episode's video, returning its frames and its native fps."""
    path = root / info["video_path"].format(
        episode_chunk=ep // info.get("chunks_size", 1000),
        video_key=key,
        episode_index=ep,
    )
    container = av.open(str(path))
    stream = container.streams.video[0]
    native_fps = float(stream.average_rate) if stream.average_rate else float(info["fps"])
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    return frames, native_fps


def convert_one(
    rel: str,
    fps_override: float | None = None,
    stride: int = 1,
    out_rel: str | None = None,
) -> dict:
    """Convert one v2.1 depth dataset to v3.0.

    Args:
        stride: keep every Nth frame. Used to regenerate downsampled variants
            straight from the full-rate v2.1 source, so depth is quantized to
            12 bits exactly once rather than re-encoded from an already-encoded
            copy. The output fps is scaled accordingly.
        out_rel: write to this path instead of replacing `rel` in place. When
            set, the source is left completely untouched.
    """
    src = ROOT / rel
    info = load_v21_info(src)
    features, rgb_keys, depth_keys = build_v3_features(info)

    episodes = [json.loads(l) for l in (src / "meta/episodes.jsonl").read_text().splitlines()]
    tasks = [json.loads(l) for l in (src / "meta/tasks.jsonl").read_text().splitlines()]
    default_task = tasks[0]["task"] if tasks else rel.split("/")[0]

    dest = ROOT / out_rel if out_rel else src
    staging = dest.parent / f"{dest.name}_v3depth"
    if staging.exists():
        shutil.rmtree(staging)

    # fps may legitimately be fractional (grasp_cube_downsampled_20 is 50/4 = 12.5).
    raw_fps = fps_override or (info["fps"] / stride)
    fps = int(raw_fps) if float(raw_fps).is_integer() else float(raw_fps)
    print(f"  features: {len(features)} | rgb={rgb_keys} | depth={depth_keys} | fps={fps}")

    ds = LeRobotDataset.create(
        repo_id=f"deviamar/{rel.split('/')[0].replace(' ', '_')}",
        root=str(staging),
        fps=fps,
        features=features,
        depth_encoder=DepthEncoderConfig(),
        streaming_encoding=True,
        encoder_queue_maxsize=120,
    )

    written_eps = 0
    written_frames = 0
    skipped: list[str] = []

    for ep_meta in episodes:
        ep = ep_meta["episode_index"]
        pq = src / info["data_path"].format(
            episode_chunk=ep // info.get("chunks_size", 1000), episode_index=ep
        )
        if not pq.exists():
            skipped.append(f"ep{ep}: no parquet")
            continue

        df = pd.read_parquet(pq)
        decoded = {k: decode_episode_video(src, info, k, ep) for k in rgb_keys}
        videos = {k: v[0] for k, v in decoded.items()}
        native_fps = {k: v[1] for k, v in decoded.items()}

        n = len(df)

        # Map each parquet row to its video frame. Normally row i is frame i, but
        # in downsampled datasets (e.g. grasp_cube_downsampled_20) the parquet was
        # subsampled while the videos were left at full rate, so row i is NOT
        # frame i. The retained rows keep their original timestamps, so
        # timestamp * native_video_fps recovers the true frame index.
        row_to_frame: dict[str, list[int]] = {}
        for k in rgb_keys:
            if len(videos[k]) == n:
                row_to_frame[k] = list(range(n))
            else:
                ts = df["timestamp"].to_numpy(dtype=np.float64)
                idx = np.rint(ts * native_fps[k]).astype(int)
                idx = np.clip(idx, 0, len(videos[k]) - 1)
                row_to_frame[k] = idx.tolist()
                skipped.append(
                    f"ep{ep}: {k} has {len(videos[k])} frames vs {n} parquet rows; "
                    f"mapped by timestamp x {native_fps[k]:g}fps "
                    f"(rows 0..2 -> frames {idx[:3].tolist()})"
                )

        # Pull columns out individually: pandas cannot build a row Series across
        # these nested array-extension dtypes (df.iloc[i] raises on them).
        cols = {k: df[k].to_list() for k in features if k not in rgb_keys}

        for i in range(0, n, stride):
            frame: dict = {}

            for k in rgb_keys:
                frame[k] = videos[k][row_to_frame[k][i]]

            for k in depth_keys:
                h, w = features[k]["shape"][0], features[k]["shape"][1]
                raw = cols[k][i]
                try:
                    arr = np.asarray(raw, dtype=np.float32)
                except (ValueError, TypeError):
                    # Some datasets store the depth map as an array-of-row-arrays
                    # rather than a flat 2D block; stack the rows back together.
                    arr = np.stack([np.asarray(r, dtype=np.float32) for r in raw])
                if arr.ndim == 1:
                    arr = arr.reshape(h, w)
                frame[k] = arr.reshape(h, w)[..., None]  # (H, W, 1), metres

            for k in features:
                if k in rgb_keys or k in depth_keys:
                    continue
                frame[k] = np.asarray(cols[k][i], dtype=np.float32).reshape(
                    features[k]["shape"]
                )

            ds.add_frame(frame, default_task)

        ds.save_episode()
        written_eps += 1
        kept = len(range(0, n, stride))
        written_frames += kept
        print(f"    ep{ep}: {kept} frames"
              + (f" (stride {stride} of {n})" if stride > 1 else ""), flush=True)

    ds.finalize()

    # Verify before swapping anything.
    check = LeRobotDataset(ds.repo_id, root=str(staging))
    result = {
        "dataset": rel,
        "episodes": check.num_episodes,
        "frames": check.num_frames,
        "skipped": skipped,
    }
    _ = check[0]

    if out_rel:
        if dest.exists():
            raise RuntimeError(f"destination already exists: {dest}")
        shutil.move(str(staging), str(dest))
        result["written_to"] = str(dest)
    else:
        backup = src.parent / f"{src.name}{BACKUP_SUFFIX}"
        if backup.exists():
            raise RuntimeError(f"backup already exists, refusing to overwrite: {backup}")
        shutil.move(str(src), str(backup))
        shutil.move(str(staging), str(src))
        result["backup"] = str(backup)
    result["status"] = "ok"
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth frame; output fps is scaled by 1/N")
    ap.add_argument("--out", type=str, default=None,
                    help="write to this dataset path instead of converting in place")
    args = ap.parse_args()

    targets = [args.only] if args.only else DEPTH_DATASETS
    if args.list:
        for t in targets:
            print(t, "exists" if (ROOT / t).exists() else "MISSING")
        return 0

    results = []
    for i, rel in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {rel}", flush=True)
        if not (ROOT / rel).exists():
            print("  MISSING, skipping")
            results.append({"dataset": rel, "status": "missing"})
            continue
        try:
            r = convert_one(rel, args.fps, stride=args.stride, out_rel=args.out)
            print(f"  OK: {r['episodes']} eps, {r['frames']} frames")
            if r["skipped"]:
                print("  NOTE:")
                for s in r["skipped"]:
                    print("   ", s)
            results.append(r)
        except Exception:
            traceback.print_exc()
            results.append({"dataset": rel, "status": "failed",
                            "error": traceback.format_exc()[-2000:]})

    out = Path(__file__).parent / "depth_conversion_report.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nreport: {out}")
    for r in results:
        print(f"  {r.get('status')}: {r['dataset']}")
    return 0 if all(r.get("status") == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
