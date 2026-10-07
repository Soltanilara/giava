"""Provisional top_scene pose in the URDF world frame, solved from recorded grasps.

WHY.  The rig has no solved top_scene extrinsic (calibration/data/extrinsics
is empty, ee_camera_transforms.json "cameras" is {}).  Every flower episode
already pairs a 2D and a 3D observation of the same physical thing:

    2D  the flower's top_scene pixel (the env-state detector, per frame)
    3D  the closed-fingertip point when the gripper closes on it
        = T_world_flange (observation.ee_pose.right) @ TCP (0, 0, 0.12753)
          -- the TCP measured by touch_calibrate on 2026-09-03

So one PnP solve over ~130 episodes gives a camera pose with no ChArUco
board.  The pixel used is the LAST frame the detector still saw the flower
before the close, so a nudge earlier in the episode does not corrupt the pair.

THIS IS NOT A CALIBRATION.  The fingertips close around the flower's middle,
the detector reports its top-view centroid, and the operator's grasp is not
perfectly centred.  The printed residual is the honest accuracy; the output
goes to outputs/, never into ee_camera_transforms.json.  Replace it with
calibration/scene_extrinsics.py when the board session happens.

    python top_pose_from_grasps.py --root <..._envstate run dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from camera import load_intrinsics  # noqa: E402

TCP_M = np.array([0.0, 0.0, 0.12753])     # calibration/tcp_offsets.json, right arm
ENV, EE = "observation.environment_state", "observation.ee_pose.right"


def quat_wxyz_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def pairs(root: Path, drop: float = 0.5):
    info = json.loads((root / "meta" / "info.json").read_text())
    names = info["features"][ENV]["names"]
    assert names[:3] == ["obj_cx", "obj_cy", "obj_found"], names[:3]
    H, W = info["features"]["observation.images.top_scene"]["shape"][:2]
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "action", EE, ENV])
                    for f in sorted(glob.glob(str(root / "data" / "*" / "*.parquet")))], ignore_index=True)
    E, A, P = (np.stack(df[c].values) for c in (ENV, "action", EE))
    ep = df["episode_index"].to_numpy()
    uv, xyz, eps = [], [], []
    for e in np.unique(ep):
        i = np.flatnonzero(ep == e)
        g = A[i, -1]
        close = np.flatnonzero(g < g.max() - drop)
        if not len(close):
            continue
        seen = np.flatnonzero(E[i[: close[0] + 1], 2] > 0.5)
        if not len(seen):
            continue
        f = i[seen[-1]]
        uv.append(((E[f, 0] + 1) / 2 * W, (E[f, 1] + 1) / 2 * H))   # undo _norm_xy
        c = i[close[0]]
        xyz.append(P[c, 4:7] + quat_wxyz_to_R(P[c, :4]) @ TCP_M)
        eps.append(int(e))
    return np.array(uv, float), np.array(xyz, float), np.array(eps), (W, H)


def solve(uv, xyz, cam):
    import cv2
    n = cam.unproject(uv)                            # distortion handled by the model
    ok, rv, tv, inl = cv2.solvePnPRansac(xyz, n, np.eye(3), None, reprojectionError=0.02,
                                         flags=cv2.SOLVEPNP_EPNP, iterationsCount=2000)
    if not ok:
        raise SystemExit("PnP failed")
    inl = inl.ravel()
    rv, tv = cv2.solvePnPRefineLM(xyz[inl], n[inl], np.eye(3), None, rv, tv)
    R, _ = cv2.Rodrigues(rv)
    T_cam_world = np.eye(4); T_cam_world[:3, :3] = R; T_cam_world[:3, 3] = tv.ravel()
    return np.linalg.inv(T_cam_world), inl


def solve_fixed_height(uv, xyz, cam, T_world_cam0, height_m):
    """Re-solve with the camera centre's world z held at a measured height.

    The grasp points all lie within a few cm of the table, so a free PnP pins
    the viewing direction well but trades camera distance against the rest of
    the pose.  A tape-measured height removes that degree of freedom; the
    remaining five (rotation + centre x, y) are refined by least squares on
    pixel error, starting from the free solution."""
    import cv2
    from scipy.optimize import least_squares
    R0 = T_world_cam0[:3, :3].T                      # world -> camera rotation
    x0 = np.r_[cv2.Rodrigues(R0)[0].ravel(), T_world_cam0[:2, 3]]

    def T_cw(x):
        R, _ = cv2.Rodrigues(x[:3])
        C = np.array([x[3], x[4], height_m])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = -R @ C
        return T

    def resid(x):
        T = T_cw(x)
        pc = xyz @ T[:3, :3].T + T[:3, 3]
        return (cam.project(pc) - uv).ravel()

    r = least_squares(resid, x0, loss="huber", f_scale=3.0)
    return np.linalg.inv(T_cw(r.x))


def report(tag, T_world_cam, uv, xyz, cam, inl):
    pc = (np.linalg.inv(T_world_cam) @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
    err = np.linalg.norm(cam.project(pc) - uv, axis=1)
    z = pc[:, 2]
    print(f"{tag:22s} centre {np.round(T_world_cam[:3, 3], 3)}  "
          f"reprojection px on the {len(inl)} inliers: median {np.median(err[inl]):.2f}  "
          f"p90 {np.percentile(err[inl], 90):.2f}  (~{np.median(err[inl]) * np.median(z) / cam.fx * 1e3:.1f} mm at the table)")
    return err


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "top_scene_pose_from_grasps.json")
    ap.add_argument("--camera-height", type=float, default=None,
                    help="measured height of the camera above the table [m] (table = world z 0); "
                         "holds the camera centre's z fixed")
    a = ap.parse_args()

    cam = load_intrinsics("top_scene")
    uv, xyz, eps, (W, H) = pairs(a.root)
    print(f"{len(uv)} episodes paired  (pixels span u {uv[:,0].min():.0f}-{uv[:,0].max():.0f}, "
          f"v {uv[:,1].min():.0f}-{uv[:,1].max():.0f}; z {xyz[:,2].min():.3f}-{xyz[:,2].max():.3f} m)")
    print(f"intrinsics: {cam.source}")
    T_world_cam, inl = solve(uv, xyz, cam)
    print(f"inliers {len(inl)}/{len(uv)}")
    err = report("free 6-DoF", T_world_cam, uv, xyz, cam, inl)
    if a.camera_height is not None:
        T_world_cam = solve_fixed_height(uv[inl], xyz[inl], cam, T_world_cam, a.camera_height)
        err = report(f"height held {a.camera_height:.3f} m", T_world_cam, uv, xyz, cam, inl)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "_README": "PROVISIONAL top_scene pose from recorded grasps (top_pose_from_grasps.py). "
                   "T_world_camera, OpenCV optical frame, URDF world frame, row-major. Not a calibration.",
        "camera": "top_scene", "rigid_to": "base", "matrix_4x4_row_major": T_world_cam.tolist(),
        "intrinsics_source": cam.source, "source_root": str(a.root),
        "n_pairs": int(len(uv)), "n_inliers": int(len(inl)),
        "reprojection_px": {"median": float(np.median(err[inl])), "p90": float(np.percentile(err[inl], 90))},
        "tcp_m": TCP_M.tolist(),
        "camera_height_held_m": a.camera_height,
    }, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
