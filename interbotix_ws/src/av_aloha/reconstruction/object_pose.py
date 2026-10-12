"""Object position AND orientation from the same points seen by two cameras.

One triangulated point gives a position.  Orientation needs at least three
points that are not on one line, seen in both views, and a rule for turning
them into axes.  There are two rules here:

    with a model    You know where those points sit on the object (its own
                    frame, metres) -- corners of a cube, the tips of a
                    triangle.  The pose is the rigid transform that carries
                    the model onto the triangulated points (Kabsch), and the
                    fit residual is a direct check on the whole measurement:
                    a rigid object cannot fit badly unless a click, the
                    correspondence or the calibration is wrong.

    without         The frame is built from the first three points: origin
                    at point 0, +x towards point 1, +z normal to the plane
                    of points 0-1-2 (right-handed), +y completing it.  Fine
                    for tracking how something moves; it has no idea what
                    the object's "real" axes are.

The input is pixels, so anything that finds the points -- clicks today, a
keypoint detector later -- can feed `object_pose` unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from stereo import StereoRig, Triangulation


def kabsch(src: np.ndarray, dst: np.ndarray):
    """R, t minimising sum |R src_i + t - dst_i|^2, and the RMS residual."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = cd - R @ cs
    res = dst - (src @ R.T + t)
    return R, t, float(np.sqrt(np.mean(np.sum(res * res, axis=1))))


def frame_from_three(p0, p1, p2) -> np.ndarray:
    """Rotation whose columns are the axes described in the module docstring."""
    x = np.asarray(p1, float) - np.asarray(p0, float)
    z = np.cross(x, np.asarray(p2, float) - np.asarray(p0, float))
    if np.linalg.norm(x) < 1e-6 or np.linalg.norm(z) < 1e-9:
        raise ValueError("points 0, 1, 2 are (nearly) on one line -- they do "
                         "not define a plane, so there is no orientation")
    x /= np.linalg.norm(x)
    z /= np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], axis=1)


def board_frame(rig: StereoRig, img_points: Dict[str, np.ndarray],
                grid) -> Dict[str, Any]:
    """T_ref_board for a chessboard seen by both cameras, z pointing up.

    `img_points` is each camera's detection (row-major corners, from
    calibration/board.detect_chessboard).  The two cameras may number the
    corners differently -- see board.GridSpec -- so every symmetry of the
    grid is tried for the second camera and the one whose rays actually
    meet is kept.  Every corner is then triangulated and the board model
    fitted to them, so the frame uses both cameras, and the fit residual
    says how well the rig reproduces a known flat grid."""
    import board as B

    a, b = list(img_points)
    obj = grid.obj_points().astype(float)
    ua = np.asarray(img_points[a], dtype=float)
    ub = np.asarray(img_points[b], dtype=float)
    best = None
    for p in B.grid_symmetries(grid.cols, grid.rows):
        inv = np.argsort(p)
        tri = [rig.triangulate({a: ua[k], b: ub[inv[k]]}, sigma_px=0)
               for k in range(len(obj))]
        gap = float(np.mean([t.ray_gap_mm for t in tri]))
        if best is None or gap < best[0]:
            best = (gap, np.stack([t.point for t in tri]))
    gap, P = best
    R, t, rms = kabsch(obj, P)
    ## Corner 0 and the axis directions come from how the detector happened
    ## to number the grid.  Only the sign of z is fixed here: up, towards
    ## the cameras, so heights above the table come out positive.
    if np.dot(R[:, 2], rig[a].centre - t) < 0:
        R = R @ np.diag([1.0, -1.0, -1.0])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return {"T_ref_board": T, "fit_rms_mm": rms * 1e3,
            "mean_ray_gap_mm": gap, "n_corners": len(obj)}


def board_frame_single(rig: StereoRig, camera: str, img_points: np.ndarray,
                       grid) -> Dict[str, Any]:
    """T_ref_board from ONE camera -- for when the other sees it edge-on.

    PnP on normalised rays (the camera knows its own distortion direction),
    so only this camera's intrinsics and pose in the rig are involved.  A
    flat board 0.7 m below a camera that looks almost straight down is a
    well-conditioned PnP; the reprojection RMS says how well it fitted."""
    import cv2

    cam = rig[camera]
    obj = grid.obj_points().astype(np.float64)
    uv = np.asarray(img_points, dtype=float).reshape(-1, 2)
    rays = cam.unproject(uv)
    ok, rvec, tvec = cv2.solvePnP(obj.reshape(-1, 1, 3),
                                  rays.reshape(-1, 1, 2).astype(np.float64),
                                  np.eye(3), np.zeros(5),
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError(f"PnP failed on {camera}")
    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_cam_board[:3, 3] = tvec.ravel()
    T = cam.T_ref_cam @ T_cam_board
    ## Residual BEFORE the z flip below: the flip relabels the frame, it
    ## does not move the corners, and the model points would no longer match.
    P = obj @ T[:3, :3].T + T[:3, 3]
    d = cam.project_ref(P) - uv
    if np.dot(T[:3, 2], cam.centre - T[:3, 3]) < 0:
        T[:3, :3] = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    return {"T_ref_board": T,
            "reprojection_rms_px": float(np.sqrt(np.mean(np.sum(d * d, 1)))),
            "n_corners": len(obj), "cameras": [camera]}


def object_pose(rig: StereoRig, clicks: Sequence[Dict[str, Any]],
                model: Optional[np.ndarray] = None,
                sigma_px: float = 1.0) -> Dict[str, Any]:
    """Pose of an object in the rig's frame.

    clicks   one dict per point, {camera_name: (u, v)}, in a fixed order
    model    (N, 3) the same points in the object's own frame, metres
    """
    if len(clicks) < 3:
        raise ValueError(f"need >= 3 points for an orientation, got "
                         f"{len(clicks)}")
    tri: list[Triangulation] = [rig.triangulate(dict(c), sigma_px=sigma_px)
                                for c in clicks]
    P = np.stack([t.point for t in tri])

    if model is not None:
        model = np.asarray(model, dtype=float)
        if model.shape != P.shape:
            raise ValueError(f"model has {len(model)} points but "
                             f"{len(P)} were clicked")
        R, t, rms = kabsch(model, P)
        method = "kabsch fit of the model to the triangulated points"
        fit_res = np.linalg.norm(P - (model @ R.T + t), axis=1)
    else:
        R = frame_from_three(P[0], P[1], P[2])
        t = P[0]
        rms, fit_res = None, None
        method = "frame from points 0,1,2 (origin 0, +x to 1, +z normal)"

    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return {
        "T_ref_object": T,
        "position_m": t,
        "R": R,
        "centroid_m": P.mean(axis=0),
        "points_m": P,
        "triangulations": tri,
        "ray_gap_mm": [t_.ray_gap_mm for t_ in tri],
        "fit_rms_mm": None if rms is None else rms * 1e3,
        "fit_residual_mm": None if fit_res is None else fit_res * 1e3,
        "method": method,
        "ref_frame": rig.ref_frame,
    }
