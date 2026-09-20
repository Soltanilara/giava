"""Dense tabletop point cloud from one RealSense: capture, segment, view.

    python reconstruction/pointcloud.py capture --camera top_scene
    python reconstruction/pointcloud.py capture --camera top_scene --no-view
    python reconstruction/pointcloud.py view --load outputs/cloud_top_scene.npz

WHAT FRAME THE NUMBERS ARE IN -- READ THIS FIRST
================================================
**The camera's own optical frame, NOT the robot world frame.**  +x right,
+y down, +z forward along the optical axis, metres.

That is not a design preference, it is the only honest option today.
Putting these points in the world frame needs `T_world_camera`, and
nothing has measured it: `calibration/ee_camera_transforms.json` has an
empty `cameras` object, and every entry in `camera_mount.CAMERA_MOUNTS`
for the scene cameras still reads `provenance: "unknown"`.  The rest of
this package RAISES rather than invent a missing extrinsic (see
`rig.load_rig`), and this module holds that line: it never claims a world
pose it does not have.

To promote these clouds to the world frame later, run

    python calibration/scene_extrinsics.py collect
    python calibration/scene_extrinsics.py solve      # camera-to-camera
    python calibration/scene_extrinsics.py anchor     # tie to the robot
    python calibration/scene_extrinsics.py write      # -> ee_camera_transforms.json

and then a single 4x4 multiply moves every point here into `base`.  Until
`anchor` has run, two clouds from two cameras CANNOT be merged -- there is
no common frame to merge them in, and averaging them anyway is how you get
a confidently wrong map.  Solve before anchor; that order is not optional.

WHY DENSE, WHEN stereo.py ALREADY TRIANGULATES
==============================================
`stereo.py` is sparse and exact: you supply a correspondence, it returns
ONE point with a real error bar.  That is the right tool for measuring a
known feature and the wrong one for "what is on the table".  The D405
gives a depth value at every pixel for free -- the correspondence search
already happened, on the camera's own ASIC -- so a surface costs one frame
grab instead of 300k clicks.

The trade is honesty about error.  A triangulated point from `stereo.py`
carries a covariance derived from the calibration.  A depth pixel does
not: it carries the projector's opinion, which degrades on dark, shiny,
and transparent surfaces and fails outright past the D405's range.  So
this module reports COVERAGE (how many pixels survived each filter) rather
than an error bar it cannot compute.  If you need a number you can quote,
measure it with `measure.py`; if you need to see the scene, use this.

WHICH CAMERA TO POINT AT THE TABLE  (measured 2026-09-20, not guessed)
======================================================================
The D405 is a SHORT range sensor: Intel spec it from 7 cm to about 50 cm,
and its depth error grows with the square of distance.  `top_scene` sits
**86 cm** above the table, well outside that, and it shows:

    residual RMS against the fitted table plane      19.2 mm
    near-plane points more than 15 mm off the plane  41.3 %

At that noise floor nothing shorter than roughly 3 sigma -- about **58 mm**
-- can be told apart from the table it is sitting on, and DBSCAN duly
turns the noise ripple on the bare wood into dozens of "objects".  This is
a sensor-placement fact, not a parameter to tune around: no `--eps` makes
a 19 mm noise floor resolve a 20 mm block.

So:
  * small objects (sorter pieces, blocks)  -> use a WRIST camera, brought
    to 15-30 cm over the table.  Same sensor, in spec, ~1-2 mm noise.
  * whole-table layout, big objects        -> `top_scene` is fine, with
    `--plane-tol 0.015` and only trusting clusters over ~60 mm tall.
  * `low_scene` is a grazing view from table level.  Grazing angles are
    the worst case for stereo depth; it is not a tabletop-mapping camera.

THE PIPELINE, AND WHY EACH STAGE IS THERE
=========================================
  1. median of N depth frames   the D405's per-pixel depth flickers frame
                                to frame; the median of ~10 kills that
                                without smearing edges the way a mean does
  2. align depth -> color       so every 3D point gets its true colour,
                                not a colour from a neighbouring pixel
  3. deproject                  pixels + depth -> metres, using the COLOR
                                intrinsics (alignment resampled depth into
                                the colour camera, so colour intrinsics are
                                the correct ones -- using depth intrinsics
                                here is a classic and silent ~1 cm error)
  4. range gate                 D405 is a SHORT range sensor; readings
                                outside [--min-depth, --max-depth] are
                                noise dressed as geometry
  5. voxel downsample           one point per voxel: uniform density,
                                averaged colour, and DBSCAN that finishes
                                this decade
  6. RANSAC plane               the table is the single biggest plane in
                                view; find it explicitly rather than
                                assuming z or y is "up"
  7. cluster                    what is left, grouped into objects

Nothing here writes a calibration, consistent with the rest of the package.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_CALIB = _HERE.parent / "calibration"
_DCS = _HERE.parent / "data_collection_scripts"
for _p in (str(_HERE), str(_CALIB), str(_DCS), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

## Serials live in camera_manager so that the cloud provably comes from the
## same physical device the recorder opens under that name.  Duplicating the
## table here would let the two drift apart silently.
try:
    from camera_manager import CAMERA_SERIALS
except Exception:  # pragma: no cover - lets --serial still work standalone
    CAMERA_SERIALS = {}

DEFAULT_OUT = _HERE / "outputs"


## ------------------------------------------------------------------ ##
##  capture
## ------------------------------------------------------------------ ##

def grab_frames(serial: str, n_median: int, warmup: int,
                width: int, height: int, fps: int):
    """Open one RealSense and return (depth_m, color_rgb, intrinsics).

    Depth is the per-pixel MEDIAN of `n_median` frames, already aligned to
    the colour stream.  Zeros mean "no reading" and are preserved as zeros
    rather than being interpolated -- an invented depth is indistinguishable
    from a real one downstream, which is exactly the failure this avoids.
    """
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    profile = pipeline.start(config)
    try:
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)

        ## Auto-exposure needs a few frames to settle.  Colour captured
        ## before it settles is the wrong colour, and it is baked into every
        ## point, so this wait is not optional.
        for _ in range(warmup):
            pipeline.wait_for_frames()

        depth_stack, color = [], None
        for _ in range(n_median):
            frames = align.process(pipeline.wait_for_frames())
            d = frames.get_depth_frame()
            c = frames.get_color_frame()
            if not d or not c:
                continue
            depth_stack.append(np.asanyarray(d.get_data()))
            color = np.asanyarray(c.get_data())

        if not depth_stack:
            raise RuntimeError(f"camera {serial} produced no aligned frames")

        intr = frames.get_color_frame().profile.as_video_stream_profile().intrinsics
        intrinsics = {
            "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
            "width": intr.width, "height": intr.height,
            "model": str(intr.model), "coeffs": list(intr.coeffs),
        }
    finally:
        pipeline.stop()

    raw = np.stack(depth_stack, axis=0).astype(np.float32)
    ## Median over frames, ignoring the zeros -- a pixel that reads 0 in
    ## half the frames should take the median of the readings it DID get,
    ## not be dragged toward zero by the dropouts.
    raw[raw == 0] = np.nan
    with np.errstate(all="ignore"):
        depth_u16 = np.nanmedian(raw, axis=0)
    depth_m = np.nan_to_num(depth_u16, nan=0.0) * depth_scale

    return depth_m, color, intrinsics


def deproject(depth_m: np.ndarray, color: np.ndarray, intrinsics: dict):
    """Pixels + depth -> (points Nx3 metres, colors Nx3 uint8).

    Distortion is handled rather than assumed away.  The D405's colour
    stream usually reports near-zero coefficients, but "usually" is not a
    guarantee and a silently un-undistorted cloud bows at the edges in a
    way that looks like a real curved surface.
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    ppx, ppy = intrinsics["ppx"], intrinsics["ppy"]
    coeffs = np.asarray(intrinsics["coeffs"], dtype=np.float64)

    h, w = depth_m.shape
    vs, us = np.mgrid[0:h, 0:w]
    us = us.astype(np.float64).ravel()
    vs = vs.astype(np.float64).ravel()

    if np.max(np.abs(coeffs)) > 1e-6:
        import cv2
        K = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]], dtype=np.float64)
        pts = np.stack([us, vs], axis=1).reshape(-1, 1, 2)
        norm = cv2.undistortPoints(pts, K, coeffs).reshape(-1, 2)
        x_n, y_n = norm[:, 0], norm[:, 1]
    else:
        x_n = (us - ppx) / fx
        y_n = (vs - ppy) / fy

    z = depth_m.ravel().astype(np.float64)
    points = np.stack([x_n * z, y_n * z, z], axis=1)
    colors = color.reshape(-1, 3)
    return points, colors


## ------------------------------------------------------------------ ##
##  filtering and segmentation
## ------------------------------------------------------------------ ##

def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel: float):
    """One point per occupied voxel, with the voxel's mean position/colour.

    This is the 'clean set of points' step.  Raw depth gives wildly uneven
    density -- surfaces near the camera get far more pixels per square
    centimetre than far ones -- and every density-based algorithm
    downstream (DBSCAN above all) reads that as structure that is not
    there.  A voxel grid makes density uniform by construction.
    """
    if voxel <= 0:
        return points, colors, np.arange(len(points))

    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                   return_counts=True)
    n_vox = len(counts)

    sums = np.zeros((n_vox, 3))
    csum = np.zeros((n_vox, 3))
    np.add.at(sums, inverse, points)
    np.add.at(csum, inverse, colors.astype(np.float64))

    out_pts = sums / counts[:, None]
    out_col = (csum / counts[:, None]).astype(np.uint8)
    return out_pts, out_col, counts


def fit_plane_ransac(points: np.ndarray, tol: float, iters: int,
                     rng: np.random.Generator):
    """Largest plane by inlier count -> (normal, d, inlier_mask).

    Plane is {p : normal . p + d = 0} with |normal| = 1.

    RANSAC and not least-squares: a least-squares fit over the whole cloud
    is pulled by the objects sitting ON the table, which is precisely the
    signal we are trying to separate from it.  The final refit uses only
    the inliers, so the objects never vote on where the table is.
    """
    best_mask = np.zeros(len(points), dtype=bool)
    best_count = 0
    n = len(points)
    if n < 3:
        raise RuntimeError("not enough points to fit a plane")

    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -normal @ p0
        dist = np.abs(points @ normal + d)
        mask = dist < tol
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask

    if best_count < 3:
        raise RuntimeError("RANSAC found no plane; try a larger --plane-tol")

    ## Refit on the inliers: the 3-point hypothesis is only a seed, and its
    ## normal is as noisy as the three pixels that produced it.
    inliers = points[best_mask]
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    d = -normal @ centroid

    dist = np.abs(points @ normal + d)
    return normal, d, dist < tol


def plane_basis(normal: np.ndarray):
    """Two orthonormal in-plane axes for a plane with this normal."""
    seed = np.array([1.0, 0.0, 0.0])
    if abs(normal @ seed) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(normal, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    return e1, e2


def table_footprint(points: np.ndarray, table_mask: np.ndarray,
                    normal: np.ndarray, margin: float, pct: float):
    """In-plane bounds of the table itself -> a mask over ALL points.

    WHY THIS STAGE EXISTS.  A scene camera sees the room, not a tabletop:
    the first run of this tool on `top_scene` reported 87 "objects", and
    most of them were the arms, the frame bars, and the far wall.  No
    clustering parameter fixes that, because those things ARE dense
    clusters sitting above the table plane -- an infinite plane extends
    behind the wall just as happily as under the blocks.

    The physical prior that does fix it: an object on the table is within
    the table's own footprint.  So the table inliers define the region of
    interest, and everything outside it is not a tabletop object no matter
    how solid it looks.  `pct` trims the inlier extremes before taking the
    bounds, so one stray inlier on the floor cannot enlarge the table.
    """
    e1, e2 = plane_basis(normal)
    a_all, b_all = points @ e1, points @ e2
    a_t, b_t = a_all[table_mask], b_all[table_mask]

    lo_a, hi_a = np.percentile(a_t, [pct, 100 - pct])
    lo_b, hi_b = np.percentile(b_t, [pct, 100 - pct])

    inside = ((a_all > lo_a - margin) & (a_all < hi_a + margin) &
              (b_all > lo_b - margin) & (b_all < hi_b + margin))
    extent = (float(hi_a - lo_a), float(hi_b - lo_b))
    return inside, extent


def cluster_objects(points: np.ndarray, eps: float, min_samples: int):
    """DBSCAN over the non-table points -> integer labels (-1 = noise).

    DBSCAN rather than k-means because the number of objects is exactly
    what we are trying to find out, and k-means would happily split one
    object in two to hit a k it was told.
    """
    from sklearn.cluster import DBSCAN
    if len(points) == 0:
        return np.zeros(0, dtype=int)
    return DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_


## ------------------------------------------------------------------ ##
##  output
## ------------------------------------------------------------------ ##

def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Binary little-endian PLY with per-vertex colour.

    Hand-rolled because the format is twelve lines of header and this
    package should not grow an open3d dependency for a file writer.
    Readable by MeshLab, CloudCompare, Blender and open3d alike.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float32)
    cols = np.asarray(colors, dtype=np.uint8)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec = np.empty(len(pts), dtype=dtype)
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["red"], rec["green"], rec["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(rec.tobytes())


def distinct_colors(n: int) -> np.ndarray:
    """n visually separable RGB colours, golden-ratio spaced in hue."""
    import colorsys
    out = []
    for i in range(max(n, 1)):
        h = (i * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.98)
        out.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.asarray(out, dtype=np.uint8)


def render_topdown(path: Path, points: np.ndarray, colors: np.ndarray,
                   table_mask: np.ndarray, obj_mask: np.ndarray,
                   labels: np.ndarray, normal: np.ndarray, d: float,
                   mm_per_px: float = 2.0) -> None:
    """Orthographic bird's-eye of the table plane, as a PNG.

    The camera's own image is a perspective view from 24 degrees off
    vertical, which is a bad way to judge a segmentation: near things look
    big, the table is a trapezoid, and two objects at different heights
    overlap.  Re-projecting onto the plane's own axes removes all three
    problems -- every pixel is the same number of millimetres, so a cluster
    that looks like a block IS block-shaped.

    Left panel is true colour, right panel is the segmentation.
    """
    import cv2

    e1, e2 = plane_basis(normal)
    a, b = points @ e1, points @ e2
    lo_a, hi_a = a.min(), a.max()
    lo_b, hi_b = b.min(), b.max()

    scale = 1000.0 / mm_per_px
    w = max(int((hi_a - lo_a) * scale) + 1, 2)
    h = max(int((hi_b - lo_b) * scale) + 1, 2)
    xs = np.clip(((a - lo_a) * scale).astype(int), 0, w - 1)
    ys = np.clip(((b - lo_b) * scale).astype(int), 0, h - 1)

    ## Painter's order: the highest point wins each pixel, so an object
    ## is drawn over the table it stands on rather than under it.
    height = points @ normal + d
    order = np.argsort(height)

    rgb = np.zeros((h, w, 3), np.uint8)
    seg = np.zeros((h, w, 3), np.uint8)

    seg_colors = np.zeros_like(colors)
    seg_colors[table_mask] = (70, 70, 70)
    ids = sorted(int(l) for l in set(labels.tolist()) if l >= 0)
    palette = distinct_colors(len(ids))
    obj_idx = np.where(obj_mask)[0]
    for i, label in enumerate(ids):
        seg_colors[obj_idx[labels == label]] = palette[i]
    seg_colors[obj_idx[labels == -1]] = (35, 35, 35)

    rgb[ys[order], xs[order]] = colors[order]
    seg[ys[order], xs[order]] = seg_colors[order]

    gap = np.full((h, 8, 3), 255, np.uint8)
    canvas = np.hstack([rgb, gap, seg])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def serve_viser(payload: dict, port: int) -> None:
    """Show the segmentation in viser: table grey, each object its own hue."""
    import viser

    points = payload["points"]
    colors = payload["colors"]
    table_mask = payload["table_mask"]
    labels = payload["labels"]
    point_size = float(payload.get("point_size", 0.002))

    server = viser.ViserServer(port=port)
    server.scene.add_point_cloud(
        "/table", points=points[table_mask].astype(np.float32),
        colors=np.full((int(table_mask.sum()), 3), 110, dtype=np.uint8),
        point_size=point_size,
    )

    obj_pts = points[~table_mask]
    ids = sorted(int(l) for l in set(labels.tolist()) if l >= 0)
    palette = distinct_colors(len(ids))
    for i, label in enumerate(ids):
        sel = labels == label
        server.scene.add_point_cloud(
            f"/objects/obj_{label:02d}",
            points=obj_pts[sel].astype(np.float32),
            colors=np.tile(palette[i], (int(sel.sum()), 1)),
            point_size=point_size * 1.5,
        )

    noise = obj_pts[labels == -1]
    if len(noise):
        server.scene.add_point_cloud(
            "/unclustered", points=noise.astype(np.float32),
            colors=np.full((len(noise), 3), 60, dtype=np.uint8),
            point_size=point_size,
        )

    print(f"\n  viser serving on http://localhost:{port}")
    print("  table = grey, each object a colour, unclustered = dark grey")
    print("  Ctrl-C to stop.\n")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("  stopped.")


## ------------------------------------------------------------------ ##
##  commands
## ------------------------------------------------------------------ ##

def cmd_capture(args) -> None:
    serial = args.serial or CAMERA_SERIALS.get(args.camera)
    if not serial:
        raise SystemExit(
            f"no serial for camera '{args.camera}'. Known: "
            f"{sorted(CAMERA_SERIALS)}. Pass --serial to use another device."
        )

    print(f"\n  camera {args.camera}  serial {serial}")
    print(f"  grabbing {args.median} frames (after {args.warmup} warmup) ...")
    depth_m, color, intrinsics = grab_frames(
        serial, args.median, args.warmup, args.width, args.height, args.fps)

    points, colors = deproject(depth_m, color, intrinsics)
    n_raw = len(points)

    valid = (points[:, 2] > args.min_depth) & (points[:, 2] < args.max_depth)
    points, colors = points[valid], colors[valid]
    n_gated = len(points)
    if n_gated == 0:
        raise SystemExit(
            f"no depth in [{args.min_depth}, {args.max_depth}] m. The D405 is "
            f"a short-range sensor -- check the camera is actually looking at "
            f"the table, and widen --max-depth if the stand is tall."
        )

    points, colors, _ = voxel_downsample(points, colors, args.voxel)
    n_vox = len(points)

    rng = np.random.default_rng(args.seed)
    normal, d, table_mask = fit_plane_ransac(
        points, args.plane_tol, args.ransac_iters, rng)

    ## Orient the normal toward the camera so "above the table" is a
    ## consistent sign regardless of which three points RANSAC happened to
    ## seed with.
    if normal @ np.array([0.0, 0.0, 1.0]) > 0:
        normal, d = -normal, -d

    signed = points @ normal + d
    inside, extent = table_footprint(points, table_mask, normal,
                                     args.footprint_margin, args.footprint_pct)

    ## An object on this table is: not the table itself, on the camera side
    ## of it, no taller than --max-height, and within the table's own
    ## footprint.  Drop any one of these four and the room comes back in.
    above = (~table_mask) & (signed > 0) & (signed < args.max_height) & inside
    obj_points = points[above]
    obj_colors = colors[above]

    labels = cluster_objects(obj_points, args.eps, args.min_samples)
    ids = sorted(int(l) for l in set(labels.tolist()) if l >= 0)

    print(f"\n  pixels                 {n_raw:>9,}")
    print(f"  in depth range         {n_gated:>9,}  "
          f"({100.0 * n_gated / max(n_raw, 1):.1f}% coverage)")
    print(f"  after {args.voxel * 1000:.0f} mm voxels     {n_vox:>9,}")
    print(f"  table inliers          {int(table_mask.sum()):>9,}  "
          f"({100.0 * table_mask.sum() / max(n_vox, 1):.1f}%)")
    print(f"  in table footprint     {int(inside.sum()):>9,}")
    print(f"  objects (on the table) {len(obj_points):>9,}")
    print(f"\n  table plane  n = [{normal[0]:+.3f} {normal[1]:+.3f} "
          f"{normal[2]:+.3f}]  d = {d:+.4f} m")
    print(f"  camera height above it   {abs(d):.3f} m")
    print(f"  table footprint          {extent[0]:.2f} x {extent[1]:.2f} m "
          f"(+{args.footprint_margin * 100:.0f} cm margin)")
    if table_mask.sum() / max(n_vox, 1) < 0.10:
        print("\n  WARNING: the plane claims under 10% of the cloud. The "
              "biggest plane in\n           view may not be your table -- "
              "check --max-depth is not letting a\n           wall or the "
              "floor outvote it.")

    ## Extents are reported in the TABLE's frame, not the camera's.  The
    ## camera looks down at ~24 deg here, so a camera-frame bounding box
    ## mixes footprint into "height" and reads a flat sheet of paper as
    ## 34 cm tall.  Footprint is measured along the in-plane axes and
    ## height along the plane normal, which is what an object state wants.
    e1, e2 = plane_basis(normal)
    print(f"\n  {len(ids)} object cluster(s)   "
          f"(eps {args.eps * 1000:.0f} mm, min_samples {args.min_samples})")
    print(f"  {'id':>4}  {'points':>7}  {'centroid (x y z) m':>26}  "
          f"{'footprint mm':>14}  {'height mm':>9}")
    print("  " + "-" * 74)
    objects_meta = []
    for label in ids:
        sel = labels == label
        pts = obj_points[sel]
        c = pts.mean(axis=0)
        a, b = pts @ e1, pts @ e2
        foot = ((a.max() - a.min()) * 1000.0, (b.max() - b.min()) * 1000.0)
        h = pts @ normal + d
        height = float(h.max() * 1000.0)
        print(f"  {label:>4}  {int(sel.sum()):>7,}  "
              f"[{c[0]:+.3f} {c[1]:+.3f} {c[2]:+.3f}]  "
              f"{foot[0]:6.0f} x{foot[1]:6.0f}  {height:9.0f}")
        objects_meta.append({
            "id": label, "n_points": int(sel.sum()),
            "centroid_m": c.tolist(),
            "footprint_mm": list(foot),
            "height_above_table_mm": height,
        })
    n_noise = int((labels == -1).sum())
    if n_noise:
        print(f"\n  {n_noise:,} points unclustered (DBSCAN noise) -- raise "
              f"--eps or lower --min-samples if objects are being dropped")

    out = Path(args.out or DEFAULT_OUT)
    stem = f"cloud_{args.camera}"
    write_ply(out / f"{stem}_full.ply", points, colors)
    write_ply(out / f"{stem}_table.ply", points[table_mask], colors[table_mask])
    write_ply(out / f"{stem}_objects.ply", obj_points, obj_colors)
    for label in ids:
        sel = labels == label
        write_ply(out / f"{stem}_obj{label:02d}.ply",
                  obj_points[sel], obj_colors[sel])

    np.savez_compressed(
        out / f"{stem}.npz",
        points=points, colors=colors, table_mask=table_mask,
        labels=labels, plane_normal=normal, plane_d=d,
    )
    (out / f"{stem}_report.json").write_text(json.dumps({
        "frame": "camera_optical",
        "world_frame_available": False,
        "why": "ee_camera_transforms.json has no entry for this camera; "
               "run calibration/scene_extrinsics.py collect|solve|anchor|write",
        "camera": args.camera, "serial": serial,
        "intrinsics": intrinsics,
        "plane": {"normal": normal.tolist(), "d": float(d),
                  "footprint_m": list(extent)},
        "gates": {"max_depth_m": args.max_depth, "voxel_m": args.voxel,
                  "plane_tol_m": args.plane_tol,
                  "max_height_m": args.max_height,
                  "footprint_margin_m": args.footprint_margin,
                  "eps_m": args.eps, "min_samples": args.min_samples},
        "counts": {"pixels": n_raw, "in_range": n_gated, "voxels": n_vox,
                   "table": int(table_mask.sum()),
                   "in_footprint": int(inside.sum()),
                   "objects": int(len(obj_points)), "noise": n_noise},
        "objects": objects_meta,
    }, indent=2))

    render_topdown(out / f"{stem}_topdown.png", points, colors, table_mask,
                   above, labels, normal, d, args.mm_per_px)

    print(f"\n  wrote {out}/{stem}_*.ply  +  {stem}.npz  +  "
          f"{stem}_report.json\n        {stem}_topdown.png "
          f"(bird's-eye: colour | segmentation)")
    print("  NOTE: points are in the CAMERA optical frame, not the robot "
          "world frame.\n        See this file's docstring.")

    if not args.no_view:
        serve_viser({"points": points, "colors": colors,
                     "table_mask": table_mask, "labels": labels,
                     "point_size": args.point_size}, args.port)


def cmd_view(args) -> None:
    data = np.load(args.load)
    serve_viser({
        "points": data["points"], "colors": data["colors"],
        "table_mask": data["table_mask"], "labels": data["labels"],
        "point_size": args.point_size,
    }, args.port)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="grab a cloud, segment it, save it")
    cap.add_argument("--camera", default="top_scene",
                     help="camera name in camera_manager.CAMERA_SERIALS")
    cap.add_argument("--serial", default=None, help="override the serial")
    cap.add_argument("--median", type=int, default=10,
                     help="depth frames to median together (default 10)")
    cap.add_argument("--warmup", type=int, default=30,
                     help="frames discarded for auto-exposure (default 30)")
    cap.add_argument("--width", type=int, default=640)
    cap.add_argument("--height", type=int, default=480)
    cap.add_argument("--fps", type=int, default=30)
    cap.add_argument("--min-depth", type=float, default=0.05,
                     help="metres; below this the D405 does not report")
    cap.add_argument("--max-depth", type=float, default=1.20,
                     help="metres; beyond this D405 depth is mostly noise")
    cap.add_argument("--voxel", type=float, default=0.003,
                     help="voxel edge in metres (default 3 mm; 0 disables)")
    cap.add_argument("--plane-tol", type=float, default=0.015,
                     help="table inlier band, metres (default 15 mm, sized for\n                          a scene camera; drop to 0.004 for a wrist cam)")
    cap.add_argument("--ransac-iters", type=int, default=400)
    cap.add_argument("--max-height", type=float, default=0.30,
                     help="metres above the table an object may reach "
                          "(default 0.30; taller things are arms and walls)")
    cap.add_argument("--footprint-margin", type=float, default=0.05,
                     help="metres of slack around the table's own extent "
                          "(default 0.05)")
    cap.add_argument("--footprint-pct", type=float, default=2.0,
                     help="percentile trimmed off each end of the table "
                          "extent before bounding it (default 2)")
    cap.add_argument("--eps", type=float, default=0.012,
                     help="DBSCAN neighbourhood, metres (default 12 mm)")
    cap.add_argument("--min-samples", type=int, default=25)
    cap.add_argument("--seed", type=int, default=0)
    cap.add_argument("--out", default=None)
    cap.add_argument("--no-view", action="store_true")
    cap.add_argument("--port", type=int, default=8097)
    cap.add_argument("--point-size", type=float, default=0.002)
    cap.add_argument("--mm-per-px", type=float, default=2.0,
                     help="resolution of the top-down PNG (default 2 mm/px)")
    cap.set_defaults(func=cmd_capture)

    vw = sub.add_parser("view", help="re-open a saved cloud, no hardware")
    vw.add_argument("--load", required=True, help="the .npz written by capture")
    vw.add_argument("--port", type=int, default=8097)
    vw.add_argument("--point-size", type=float, default=0.002)
    vw.set_defaults(func=cmd_view)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
