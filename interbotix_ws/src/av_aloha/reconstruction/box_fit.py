"""Pose of a box resting on the table, from its colour silhouette in each view.

The object is assumed to sit flat on the board, so its pose has three
degrees of freedom: x, y and yaw in the board frame.  Its centre is then
H/2 above the board by construction.

No points are clicked.  For a candidate (x, y, yaw) the box's 8 corners are
projected into each camera; since a box is convex, its silhouette is exactly
the convex hull of those projections.  The cost compares that outline with
the outline of the colour mask, both ways round:

    hull -> mask   points sampled along the predicted outline, scored by
                   their distance to the observed outline (a distance
                   transform, so sub-pixel and smooth);
    mask -> hull   observed outline pixels, scored by their exact distance
                   to the predicted polygon.

Either term alone can be cheated -- a tiny hull inside the blob, or a huge
one around it -- together they cannot.  The result is in pixels RMS per
camera, which is the quality figure: a good fit is about a pixel.  Fitting
each camera ALONE as well, and comparing, tests the calibration itself --
the two must agree on where the box is if the rig is right.

Only for CONVEX prisms.  The flower is not convex; its silhouette is not
the hull of its vertices, and this would fit its hull instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Observation:
    """One camera's view of the object: its mask outline, preprocessed."""
    mask: np.ndarray                 # uint8 0/1, full image
    contour: np.ndarray              # (N, 2) float, outline pixels
    dt: np.ndarray                   # float32, distance to the outline
    area: int
    centroid: np.ndarray             # (2,) pixels


def observe(mask: np.ndarray, roi: Optional[np.ndarray] = None,
            min_area: int = 40) -> Optional[Observation]:
    """Largest blob of `mask` (inside `roi` if given) as an Observation."""
    import cv2

    m = (mask > 0).astype(np.uint8)
    if roi is not None:
        m &= (roi > 0).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < min_area:
        return None
    blob = (lab == i).astype(np.uint8)
    cs, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=len).reshape(-1, 2).astype(np.float32)
    edge = np.full(blob.shape, 255, np.uint8)
    edge[c[:, 1].astype(int), c[:, 0].astype(int)] = 0
    dt = cv2.distanceTransform(edge, cv2.DIST_L2, 5)
    return Observation(blob, c, dt, area, cents[i].astype(float))


def box_corners(pose: Sequence[float], size: Tuple[float, float, float]
                ) -> np.ndarray:
    """(8, 3) corners in the BOARD frame: bottom four at z=0, then top four."""
    x, y, yaw = pose
    W, D, H = size
    c, s = np.cos(yaw), np.sin(yaw)
    xy = np.array([[-W, -D], [W, -D], [W, D], [-W, D]]) / 2.0
    xy = xy @ np.array([[c, s], [-s, c]]) + [x, y]
    return np.vstack([np.c_[xy, np.zeros(4)], np.c_[xy, np.full(4, H)]])


BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def _bilinear(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = np.clip(pts[:, 0], 0, w - 1.001)
    y = np.clip(pts[:, 1], 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def camera_cost(uv: np.ndarray, obs: Observation) -> float:
    """Mean squared outline disagreement, px^2 (see module docstring)."""
    import cv2

    if not np.isfinite(uv).all():
        return 1e6
    hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
    nxt = np.roll(hull, -1, axis=0)
    seg = np.linalg.norm(nxt - hull, axis=1)
    samples = []
    for p, q, L in zip(hull, nxt, seg):
        k = max(2, int(np.ceil(L)))
        t = np.linspace(0.0, 1.0, k, endpoint=False)[:, None]
        samples.append(p + t * (q - p))
    a = _bilinear(obs.dt, np.vstack(samples))
    poly = hull.reshape(-1, 1, 2)
    b = np.array([cv2.pointPolygonTest(poly, (float(u), float(v)), True)
                  for u, v in obs.contour])
    return 0.5 * (float(np.mean(a * a)) + float(np.mean(b * b)))


def fit_box(rig, T_ref_board: np.ndarray, obs: Dict[str, Observation],
            size: Tuple[float, float, float],
            init: Optional[Sequence[float]] = None,
            yaw_period: float = np.pi / 2,
            fixed_yaw: Optional[float] = None) -> Dict[str, object]:
    """Best (x, y, yaw) in the board frame for the cameras in `obs`.

    `init` warm-starts from a previous answer (the live loop); without it
    every yaw over one symmetry period is tried from the blob's position.
    `fixed_yaw` holds yaw at a value measured some other way (see
    `top_face_yaw`) and fits x, y only -- the silhouettes alone constrain
    yaw poorly at this resolution, and a wrong yaw drags x, y with it."""
    from scipy.optimize import minimize

    names = list(obs)
    R, t = T_ref_board[:3, :3], T_ref_board[:3, 3]

    def project(pose):
        P = box_corners(pose, size) @ R.T + t
        return {n: rig[n].project_ref(P) for n in names}

    def cost(pose):
        uv = project(pose)
        return sum(camera_cost(uv[n], obs[n]) for n in names) / len(names)

    if fixed_yaw is not None:
        xy0 = (np.asarray(init[:2], dtype=float) if init is not None
               else _blob_xy(rig, T_ref_board, obs, size[2]))
        r = minimize(lambda p: cost([p[0], p[1], fixed_yaw]), xy0,
                     method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-4, "maxiter": 300,
                              "initial_simplex": xy0 + np.array(
                                  [[0, 0], [0.004, 0], [0, 0.004]])})
        x, y = r.x
        uv = project([x, y, fixed_yaw])
        return {"x": float(x), "y": float(y), "yaw": float(fixed_yaw),
                "z_centre": size[2] / 2.0,
                "rms_px": {n: float(np.sqrt(camera_cost(uv[n], obs[n])))
                           for n in names},
                "uv": uv}

    starts = []
    if init is not None:
        starts.append(np.asarray(init, dtype=float))
    else:
        xy0 = _blob_xy(rig, T_ref_board, obs, size[2])
        yaws = np.arange(0.0, yaw_period, np.radians(7.5))
        scored = sorted((cost([*xy0, y]), y) for y in yaws)
        starts = [np.array([*xy0, y]) for _, y in scored[:3]]

    best = None
    for s0 in starts:
        r = minimize(cost, s0, method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-4, "maxiter": 400,
                              "initial_simplex": s0 + np.array(
                                  [[0, 0, 0], [0.004, 0, 0], [0, 0.004, 0],
                                   [0, 0, 0.05]])})
        if best is None or r.fun < best.fun:
            best = r
    x, y, yaw = best.x
    yaw = (yaw + yaw_period / 2) % yaw_period - yaw_period / 2
    uv = project([x, y, yaw])
    return {
        "x": float(x), "y": float(y), "yaw": float(yaw),
        "z_centre": size[2] / 2.0,
        "rms_px": {n: float(np.sqrt(camera_cost(uv[n], obs[n])))
                   for n in names},
        "uv": uv,
    }


def top_face_yaw(image: np.ndarray, mask: np.ndarray, cam,
                 T_ref_board: np.ndarray, xy: Sequence[float], height: float,
                 symmetry: int = 4, mm_per_px: float = 0.5,
                 extent_mm: float = 80.0) -> Tuple[float, float]:
    """Yaw from the DIRECTIONS of the edges on the object's top face.

    The top view is resampled onto the plane of the top face (z = height),
    which removes the perspective, and image gradients on the object are
    averaged as exp(i * symmetry * angle): a cube's edges run in 4
    directions 90 deg apart, which multiplying by 4 maps onto one.  The
    outline of a 16 px blob barely changes with yaw; its edges' directions
    do, and there are hundreds of edge pixels to average.

    Returns (yaw in radians within +-pi/symmetry, coherence 0..1).  Low
    coherence (< ~0.3) means no dominant edge direction -- do not trust it.
    Known risk: side faces' vertical edges run radially from the image
    centre, so far from the centre they may bias this by a few degrees."""
    import cv2

    n = int(round(extent_mm / mm_per_px))
    s = mm_per_px * 1e-3
    gx, gy = np.meshgrid((np.arange(n) - n / 2) * s + xy[0],
                         (np.arange(n) - n / 2) * s + xy[1])
    P = np.stack([gx, gy, np.full_like(gx, height)], -1).reshape(-1, 3)
    uv = cam.project_ref(P @ T_ref_board[:3, :3].T + T_ref_board[:3, 3])
    uv = uv.reshape(n, n, 2).astype(np.float32)
    rect = cv2.remap(image, uv[..., 0], uv[..., 1], cv2.INTER_CUBIC)
    m = cv2.remap(mask.astype(np.float32), uv[..., 0], uv[..., 1],
                  cv2.INTER_LINEAR) > 0.5
    m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    g = rect if rect.ndim == 2 else cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 2.0)
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    w = np.hypot(dx, dy) * m
    if w.sum() <= 0:
        return 0.0, 0.0
    z = (w * np.exp(1j * symmetry * np.arctan2(dy, dx))).sum()
    return float(np.angle(z) / symmetry), float(abs(z) / w.sum())


def _blob_xy(rig, T_ref_board, obs, height) -> np.ndarray:
    """Rough start: the first camera's blob centroid, ray to mid-height."""
    n = next(iter(obs))
    o, d = rig[n].ray(obs[n].centroid)
    Tb = np.linalg.inv(T_ref_board)
    o_b = Tb[:3, :3] @ o + Tb[:3, 3]
    d_b = Tb[:3, :3] @ d
    s = (height / 2.0 - o_b[2]) / d_b[2]
    return (o_b + s * d_b)[:2]
