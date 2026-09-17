"""Strip phantom depth features from the *_rgb_only block_square datasets.

These datasets were derived by dropping the depth columns from a depth recording,
but only the parquet data was rewritten: `meta/info.json` still *declares* the
depth features, and `meta/episodes_stats.jsonl` still carries their per-pixel
statistics -- 23 MB per episode per depth camera, up to 1.77 GB per dataset.

That is what makes the stock v2.1 -> v3.0 converter fail with
"Episodes dataset is too large (750 MB) ... limit is 100 MB".

Nothing is lost by removing them: the corresponding parquet columns do not exist.
This script drops any declared feature that is absent from the actual data, then
runs the stock converter.

Original metadata is copied to meta_backup_pre_repair/ before any edit.
"""

import json
import shutil
import sys
import traceback
from pathlib import Path

import pandas as pd

from lerobot.datasets import LeRobotDataset
from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

ROOT = Path(__file__).resolve().parent / "dataset/lerobot"

TARGETS = [
    "block_square/20260528_164157_rgb_only",
    "block_square/20260528_164157_rgb_only_grasp",
    "block_square/20260528_131838_rgb_only",
    "block_square/20260528_131838_rgb_only_grasp",
]

# Bookkeeping columns that live in the parquet but are not "features" to check.
ALWAYS_KEEP = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def fix_one(rel: str) -> dict:
    src = ROOT / rel
    info = json.loads((src / "meta/info.json").read_text())

    # Columns that actually exist in the data (any episode will do).
    pq = next(iter(sorted((src / "data").rglob("episode_*.parquet"))))
    present = set(pd.read_parquet(pq).columns)

    video_keys = {k for k, ft in info["features"].items() if ft["dtype"] == "video"}
    declared = set(info["features"])
    # A feature is phantom if it is neither a video stream nor an actual column.
    phantom = {
        k for k in declared
        if k not in ALWAYS_KEEP and k not in video_keys and k not in present
    }

    if not phantom:
        return {"dataset": rel, "status": "nothing_to_fix"}

    backup = src / "meta_backup_pre_repair"
    if not backup.exists():
        shutil.copytree(src / "meta", backup)

    for k in phantom:
        del info["features"][k]
    (src / "meta/info.json").write_text(json.dumps(info, indent=4))

    stats_path = src / "meta/episodes_stats.jsonl"
    before_mb = stats_path.stat().st_size / 1e6
    out = []
    for line in stats_path.read_text().splitlines():
        e = json.loads(line)
        for k in phantom:
            e.get("stats", {}).pop(k, None)
        out.append(json.dumps(e))
    stats_path.write_text("\n".join(out) + "\n")
    after_mb = stats_path.stat().st_size / 1e6

    return {
        "dataset": rel,
        "status": "metadata_fixed",
        "removed_features": sorted(phantom),
        "episodes_stats_mb": [round(before_mb, 1), round(after_mb, 1)],
    }


def main() -> int:
    results = []
    for i, rel in enumerate(TARGETS, 1):
        print(f"\n[{i}/{len(TARGETS)}] {rel}", flush=True)
        src = ROOT / rel
        if not src.exists():
            print("  MISSING")
            results.append({"dataset": rel, "status": "missing"})
            continue
        try:
            r = fix_one(rel)
            print(f"  {r['status']}")
            if r.get("removed_features"):
                print(f"  removed: {r['removed_features']}")
                print(f"  episodes_stats.jsonl: {r['episodes_stats_mb'][0]} MB "
                      f"-> {r['episodes_stats_mb'][1]} MB")

            repo_id = f"deviamar/{rel.split('/')[0]}"
            convert_dataset(repo_id=repo_id, root=src, push_to_hub=False,
                            force_conversion=True)

            ds = LeRobotDataset(repo_id, root=src)
            _ = ds[0]
            r.update({"status": "ok", "episodes": ds.num_episodes,
                      "frames": ds.num_frames})
            print(f"  OK: {ds.num_episodes} eps, {ds.num_frames} frames")
            results.append(r)
        except Exception:
            traceback.print_exc()
            results.append({"dataset": rel, "status": "failed",
                            "error": traceback.format_exc()[-2000:]})

    out = Path(__file__).parent / "rgb_only_fix_report.json"
    out.write_text(json.dumps(results, indent=2))
    print("\n===== SUMMARY =====")
    for r in results:
        print(f"  {r.get('status')}: {r['dataset']}")
    print(f"report: {out}")
    return 0 if all(r.get("status") == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
