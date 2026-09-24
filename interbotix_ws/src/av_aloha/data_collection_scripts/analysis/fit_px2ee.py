"""Fit top_scene pixel -> right-arm grasp pose from recorded episodes, and
write assets/flower_px2ee.json for auto_reset.py.

WHY THIS AND NOT A CAMERA CALIBRATION.  The rig has no solved top_scene
extrinsic (calibration/data/extrinsics is empty).  Every training episode,
however, records both the flower's pixel position at frame 0 (via the env-state
column of the *_envstate dataset) and the end-effector pose at the moment the
gripper closed.  Fitting those directly gives "where the gripper goes to grasp
the thing at pixel p", in the arm's own base frame, with the operator's grasp
offset already included -- which is exactly what a scripted pick needs, and
nothing a camera model would add.

Run it again whenever the top camera, the table, or the gripper moves; the
residual it prints is the honest accuracy of the resulting reset.

    python analysis/fit_px2ee.py --root dataset/lerobot/transfer_flower_merged_envstate/20260916_centre_plus_rim
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from paths import ASSETS_DIR  # noqa: E402

OUT = ASSETS_DIR / "flower_px2ee.json"
ENV = "observation.environment_state"
EE = "observation.ee_pose.right"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="an *_envstate dataset root (needs obj_cx/obj_cy at index 0/1, obj_found at 2)")
    ap.add_argument("--drop", type=float, default=0.5, help="gripper command drop that counts as 'closed'")
    ap.add_argument("--write", action="store_true", help=f"overwrite {OUT}")
    args = ap.parse_args()

    pq = sorted(glob.glob(str(Path(args.root) / "data" / "*" / "*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "action", EE, ENV]) for f in pq], ignore_index=True)
    E = np.stack(df[ENV].values); A = np.stack(df["action"].values); P = np.stack(df[EE].values)
    ep = df["episode_index"].to_numpy()
    info = json.loads((Path(args.root) / "meta" / "info.json").read_text())
    names = info["features"][ENV].get("names") or []
    if names[:3] != ["obj_cx", "obj_cy", "obj_found"]:
        raise SystemExit(f"expected env-state to start obj_cx, obj_cy, obj_found; got {names[:3]}")
    W_, H_ = info["features"]["observation.images.top_scene"]["shape"][1], info["features"]["observation.images.top_scene"]["shape"][0]

    X, Y, Q = [], [], []
    for e in np.unique(ep):
        i = np.flatnonzero(ep == e); g = A[i, -1]
        c = np.flatnonzero(g < g.max() - args.drop)
        if not len(c) or E[i[0], 2] == 0:
            continue
        X.append(((E[i[0], 0] + 1) / 2 * W_, (E[i[0], 1] + 1) / 2 * H_))   # undo scene_features._norm_xy
        Y.append(P[i[c[0]], 4:7]); Q.append(P[i[c[0]], 0:4])
    X, Y, Q = np.array(X), np.array(Y), np.array(Q)
    print(f"{len(X)} episodes with a frame-0 detection and a gripper close")

    M = np.c_[X, np.ones(len(X))]
    W, *_ = np.linalg.lstsq(M, Y, rcond=None)
    R = Y - M @ W
    d = np.linalg.norm(R[:, :2], axis=1) * 1e3
    print(f"affine px->xyz residual: xy median {np.median(d):.1f} mm  p90 {np.percentile(d, 90):.1f} mm  "
          f"max {d.max():.1f} mm | z std {R[:, 2].std() * 1e3:.1f} mm")
    Q = Q * np.sign(Q[:, :1]); qm = Q.mean(0); qm /= np.linalg.norm(qm)
    ang = 2 * np.degrees(np.arccos(np.clip(np.abs(Q @ qm), 0, 1)))
    print(f"grasp orientation: mean wxyz {np.round(qm, 4).tolist()}  deviation median {np.median(ang):.1f} deg  p90 {np.percentile(ang, 90):.1f} deg")

    doc = {
        "_README": ("top_scene pixel (u,v) of the flower at episode start -> right-arm EE grasp pose. "
                    "xyz = [u,v,1] @ W_affine (metres, URDF base frame, same frame as observation.ee_pose.right / "
                    "gates.ik.solve). q_grasp is the mean close-frame quaternion (wxyz). Refit with analysis/fit_px2ee.py "
                    "if the top camera, table or gripper moves."),
        "source_root": str(args.root), "n_episodes": int(len(X)),
        "W_affine": W.tolist(), "q_grasp_wxyz": qm.tolist(), "z_grasp_mean_m": float(Y[:, 2].mean()),
        "residual_xy_mm": {"median": round(float(np.median(d)), 1), "p90": round(float(np.percentile(d, 90)), 1), "max": round(float(d.max()), 1)},
        "m_per_px": {"x": round(float(abs(W[0, 0])), 5), "y": round(float(abs(W[1, 1])), 5)},
    }
    if args.write:
        OUT.write_text(json.dumps(doc, indent=2)); print(f"wrote {OUT}")
    else:
        print(f"(dry run -- pass --write to update {OUT})")


if __name__ == "__main__":
    main()
