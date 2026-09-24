"""Add a metric gripper->object distance vector as observation.environment_state.

The vector is, per frame,

    [dx, dy, dz, obj_found]

where (dx, dy, dz) = obj_xyz - ee_xyz in METRES in the arm's base frame:

  * ee_xyz  is the recorded `observation.ee_pose.<arm>[4:7]` -- forward
    kinematics of the commanded joints, the same frame gates.ik.solve uses.
  * obj_xyz is the object's top_scene centroid pushed through the empirical
    pixel -> grasp-pose fit in assets/flower_px2ee.json (analysis/fit_px2ee.py):
    the EE position at which an object seen at that pixel gets grasped.

So the vector reads "how far, in metres, is the gripper from its grasp pose
for this object" -- zero at grasp, and directly in the units the arm moves in.
It is the metric counterpart of wrist_features' pixel-space w_err/w_range,
and it uses a signal the policy already has (its own EE pose) rather than
asking the network to learn the camera geometry.

WHAT IT CANNOT DO, stated up front.  obj_xyz assumes the object is ON THE
TABLE, because the fit was made from frame-0 placements.  Once the gripper
lifts it the top camera mostly loses it anyway (found-rate ~84% overall, low
while held), and hold-last then keeps obj_xyz at the pickup spot while ee_xyz
moves away -- so during transport the vector becomes "displacement since
pickup", not "distance to object".  `obj_found` is in the vector precisely so
the policy can tell the two regimes apart.  During approach -- where the
height error that motivated all of this happens -- the vector is exact to the
fit's residual (xy median 11 mm, z std 6 mm).

Inputs are read from parquet only (no video decode): the top_scene centroid
comes from an existing *_envstate dataset's `obj_cx/obj_cy/obj_found`, and
the EE pose from the source run.  Videos are symlinked from --videos-from
(default: the source), so this composes with a mask variant:

    python build_eedist_dataset.py \
        --src dataset/lerobot/transfer_flower_merged_wristmask/<run> \
        --envstate dataset/lerobot/transfer_flower_merged_envstate/<run> \
        --dst dataset/lerobot/transfer_flower_merged_wristmask_eedist/<run>
"""
from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from paths import ASSETS_DIR  # noqa: E402

KEY = "observation.environment_state"
FEATURE_NAMES = ["ee_to_obj_dx", "ee_to_obj_dy", "ee_to_obj_dz", "obj_found"]
FIT = ASSETS_DIR / "flower_px2ee.json"


def report_rank(feats, names):
    sd = feats.std(0)
    sv = np.linalg.svd(feats - feats.mean(0), compute_uv=False)
    rank = int((sv > sv[0] * 1e-6).sum())
    print(f"[rank] numerical rank {rank} of {len(names)}; singular values "
          f"{np.array2string(sv, precision=2)}")
    const = [n for n, s in zip(names, sd) if s < 1e-9]
    if const:
        print(f"[rank] CONSTANT dimensions: {const}")
    return rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset to copy data/meta from and whose videos to symlink")
    ap.add_argument("--envstate", required=True, help="a *_envstate dataset with obj_cx/obj_cy/obj_found from top_scene")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--arm", default="right")
    args = ap.parse_args()
    src, env, dst = Path(args.src), Path(args.envstate), Path(args.dst)
    if dst.exists():
        raise SystemExit(f"{dst} already exists -- refusing to overwrite")

    fit = json.loads(FIT.read_text())
    W = np.asarray(fit["W_affine"], dtype=np.float64)
    print(f"[fit] {FIT.name}: {fit['n_episodes']} episodes, xy residual median "
          f"{fit['residual_xy_mm']['median']} mm")

    src_pq = sorted(glob.glob(str(src / "data" / "*" / "*.parquet")))
    env_pq = sorted(glob.glob(str(env / "data" / "*" / "*.parquet")))
    if len(src_pq) != len(env_pq):
        raise SystemExit("src and envstate datasets have different parquet layouts")
    einfo = json.loads((env / "meta" / "info.json").read_text())
    enames = einfo["features"][KEY]["names"]
    ix, iy, ifound = (enames.index(n) for n in ("obj_cx", "obj_cy", "obj_found"))
    tsh = einfo["features"]["observation.images.top_scene"]["shape"]
    H_, W_ = tsh[0], tsh[1]

    dst.mkdir(parents=True)
    shutil.copytree(src / "meta", dst / "meta")
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    (dst / "videos").mkdir()
    for cam_dir in (src / "videos").iterdir():
        ## Relative links, so the tree survives an rsync to another machine.
        ## dst sits at the same depth as src (lerobot/<name>/<run>/videos/),
        ## so a link src already carries can be copied verbatim; a real
        ## directory in src is linked by the same ../../../ convention.
        if cam_dir.is_symlink():
            target = Path(os.readlink(cam_dir))
        else:
            target = Path("..") / ".." / ".." / src.parent.name / src.name / "videos" / cam_dir.name
        (dst / "videos" / cam_dir.name).symlink_to(target)
    print(f"[dst] {dst}  (meta + root files copied, videos symlinked relative)")

    all_feats = []
    for sf, ef in zip(src_pq, env_pq):
        df = pd.read_parquet(sf)
        edf = pd.read_parquet(ef, columns=["episode_index", "frame_index", KEY])
        if not (df["episode_index"].to_numpy() == edf["episode_index"].to_numpy()).all():
            raise SystemExit("row order differs between src and envstate")
        E = np.stack(edf[KEY].values).astype(np.float64)
        P = np.stack(df[f"observation.ee_pose.{args.arm}"].values).astype(np.float64)
        u = (E[:, ix] + 1.0) / 2.0 * W_        # undo scene_features._norm_xy
        v = (E[:, iy] + 1.0) / 2.0 * H_
        obj_xyz = np.c_[u, v, np.ones(len(u))] @ W
        ee_xyz = P[:, 4:7]
        d = obj_xyz - ee_xyz
        feats = np.c_[d, E[:, ifound]].astype(np.float32)
        df[KEY] = list(feats)
        out = dst / Path(sf).relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        all_feats.append(feats)
        print(f"[data] {out.name}: +{KEY} {feats.shape}")
    F = np.concatenate(all_feats)

    info = json.loads((dst / "meta" / "info.json").read_text())
    info["features"][KEY] = {"dtype": "float32", "shape": [len(FEATURE_NAMES)],
                             "names": list(FEATURE_NAMES)}
    info["env_state"] = {"kind": "ee_to_obj_metric", "fit": FIT.name, "arm": args.arm,
                         "units": "metres in the arm base frame; obj from top_scene via px->EE fit"}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats[KEY] = {"mean": F.mean(0).tolist(), "std": (F.std(0) + 1e-8).tolist(),
                  "min": F.min(0).tolist(), "max": F.max(0).tolist(), "count": [int(len(F))]}
    stats_path.write_text(json.dumps(stats, indent=4))
    print(f"[meta] info.json + stats.json updated ({len(FEATURE_NAMES)}-d)")

    report_rank(F, FEATURE_NAMES)
    print("\nper-feature range (metres; ENV is NOT normalized by ACT):")
    for j, n in enumerate(FEATURE_NAMES):
        print(f"  {n:14s} min={F[:, j].min():7.3f} mean={F[:, j].mean():7.3f} max={F[:, j].max():7.3f}")
    print(f"found-rate {F[:, 3].mean():.1%}")


if __name__ == "__main__":
    main()
