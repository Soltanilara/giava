"""Move a dataset's middle_base into the RE-CLOCKED driver frame.

    python shift_waist_frame.py --src <run_dir> --dst <new_run_dir> --theta -3.1416

WHY THIS IS NOT canonicalize_waist.py
=====================================
They look alike and mean opposite things.

  canonicalize_waist.py  adds a multiple of 2*pi.  EXACT -- the physical angle
                         is untouched, only which branch represents it.  Fixes
                         a boot-branch flip.
  this script            adds theta, the PHYSICAL re-clock of the motor.  The
                         number changes because the hardware changed; the
                         physical angle each row describes is the same one it
                         always was, now named correctly for the arm that
                         exists today.

Running canonicalize_waist.py can NOT do this job and will silently appear to
succeed: for any |theta| < pi it rounds to k=0 and changes nothing.  That is
the trap -- a dataset that looks canonicalized, trains fine, and then drives
the camera arm theta radians away from where every frame of its training data
says it should be.  With theta near a half turn that is not a subtle error.

ORDER MATTERS.  Shift FIRST, canonicalize second.  Canonicalizing first picks
a reference branch in the old frame, and the shift then walks the data off it.

Videos are symlinked; meta/ and root files are copied; stats.json and the
per-episode stats are recomputed for the two features that change.
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

JOINT = "middle_base"
FEATURES = ("observation.state", "action")
STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--theta", type=float, required=True,
                    help="physical re-clock [rad]; driver_new = driver_old + theta")
    args = ap.parse_args()
    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if dst.exists():
        raise SystemExit(f"{dst} already exists -- refusing to overwrite")

    meta = json.loads((src / "meta.json").read_text())
    names = meta["joint_names"]
    if JOINT not in names:
        raise SystemExit(f"{JOINT} not in {src}/meta.json (joint_names={names})")
    j = names.index(JOINT)

    ## A second shift would be silent and unrecoverable -- the rows carry no
    ## record of their own frame beyond this key.
    if "waist_reclock" in meta:
        raise SystemExit(
            f"{src}/meta.json already records waist_reclock="
            f"{meta['waist_reclock']} -- this dataset is already shifted.")

    files = sorted(glob.glob(str(src / "data" / "*" / "*.parquet")))
    if not files:
        raise SystemExit(f"no data parquet under {src}")

    print(f"[src]   {src}")
    print(f"[joint] {JOINT} (index {j} of {len(names)})")
    print(f"[theta] {args.theta:+.6f} rad ({np.degrees(args.theta):+.2f} deg)")

    dst.mkdir(parents=True)
    shutil.copytree(src / "meta", dst / "meta")
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    (dst / "videos").mkdir()
    for cam in (src / "videos").iterdir():
        (dst / "videos" / cam.name).symlink_to(cam.resolve())

    ## Record the shift so this is idempotent and auditable, and move any
    ## canonicalization reference with it -- a waist_reference left in the old
    ## frame would send rollout's canonical_branch to the wrong branch.
    meta["waist_reclock"] = float(args.theta)
    meta["waist_joint"] = JOINT
    if "waist_reference" in meta:
        old_ref = float(meta["waist_reference"])
        meta["waist_reference"] = old_ref + args.theta
        print(f"[ref]   waist_reference {old_ref:+.4f} -> "
              f"{meta['waist_reference']:+.4f}")
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))

    per_ep: dict[int, dict[str, np.ndarray]] = {}
    for f in files:
        df = pd.read_parquet(f)
        for feat in FEATURES:
            M = np.stack(df[feat].to_numpy()).astype(np.float32)
            M[:, j] = (M[:, j].astype(np.float64) + args.theta).astype(np.float32)
            df[feat] = list(M)
        for e, g in df.groupby("episode_index"):
            per_ep[int(e)] = {feat: np.stack(g[feat].to_numpy()).astype(np.float64)
                              for feat in FEATURES}
        out = dst / Path(f).relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"[data]  {out.name}: {JOINT} shifted")

    def col(feat):
        return np.concatenate([per_ep[e][feat][:, j] for e in sorted(per_ep)])

    ## lerobot normalizes from these, so they must describe what is on disk.
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    for feat in FEATURES:
        c = col(feat)
        for key, val in (("mean", c.mean()), ("std", c.std()),
                         ("min", c.min()), ("max", c.max()),
                         ("q01", np.percentile(c, 1)), ("q99", np.percentile(c, 99))):
            if key in stats.get(feat, {}):
                stats[feat][key][j] = float(val)
        print(f"[stats] {feat}[{JOINT}] mean {c.mean():+.4f} std {c.std():.4f} "
              f"range [{c.min():+.3f}, {c.max():+.3f}]")
    stats_path.write_text(json.dumps(stats, indent=4))

    for f in sorted(glob.glob(str(dst / "meta" / "episodes" / "*" / "*.parquet"))):
        d = pd.read_parquet(f)
        changed = False
        for feat in FEATURES:
            for key in STAT_KEYS:
                c = f"stats/{feat}/{key}"
                if c not in d.columns:
                    continue
                vals = []
                for _, row in d.iterrows():
                    arr = np.asarray(row[c], dtype=np.float64).copy()
                    e = int(row["episode_index"])
                    if e in per_ep:
                        v = per_ep[e][feat][:, j]
                        arr[j] = {"mean": v.mean(), "std": v.std(),
                                  "min": v.min(), "max": v.max(),
                                  "q01": np.percentile(v, 1),
                                  "q99": np.percentile(v, 99)}[key]
                    vals.append(arr)
                d[c] = vals
                changed = True
        if changed:
            d.to_parquet(f, index=False)
            print(f"[meta]  {Path(f).name}: per-episode stats updated")

    print(f"\n[dst]   {dst}")
    print("Retrain on this root. A checkpoint trained on the OLD root must not")
    print("be rolled out against the re-clocked arm without the same theta")
    print("applied live -- see policy_driver.waist_reclock.")


if __name__ == "__main__":
    main()
