"""Live accuracy test of object pose from the two scene cameras.

    python reconstruction/pose_live_test.py --grid 9 6 --grid-square 0.0225 \
        --object cube --square-face 0.040

1.  ORIGIN.  Lay the chessboard flat on the table, nothing on it, where both
    cameras see every corner.  Press o.  All corners are triangulated and
    the board fitted to them: that is the origin, z up.  Its axes are drawn
    on the live view from then on -- if they stop lining up with the board,
    a camera has moved.  Saved, so `--origin-file` skips this next time.

2.  MEASURE.  Put the object on the board.  SPACE freezes both views.
    Click the same points on the object in each view (a curve shows where
    the second click can be), ENTER solves, n accepts and goes back live.

3.  MOVE IT BY A KNOWN AMOUNT and measure again.  Each result is printed
    relative to the previous one, in millimetres AND in squares, so "slid
    two squares along x" should read +2.00 sq.  Rotations of 90 degrees are
    easy to make exactly by lining an edge up with the grid lines.

The model.  `--square-face S` uses the four corners of a square face of
side S (the cube's top): this knows the true geometry, so its fit residual
checks the absolute size.  Without it, the first accepted measurement
becomes the model and later ones are fitted to that -- this works for any
rigid object (the flower's petal tips), and the residual checks that the
rig sees the same shape everywhere.

Click order is the one thing that cannot be automated here.  On a
symmetric object the cameras cannot tell one corner or petal from another,
so YOU must: mark one corner/petal with a bit of tape and always start
there, and go round COUNTER-CLOCKWISE as seen from above.  A tilt near 180
deg in the output means the clicks went the other way.

Everything is in the board frame: origin at the corner the drawn axes start
from, x red, y green, z blue (up).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_CALIB = _HERE.parent / "calibration"
for _p in (str(_HERE), str(_CALIB), str(_HERE.parent / "data_collection")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import board as B  # noqa: E402
from object_pose import board_frame, board_frame_single, object_pose  # noqa: E402
from rig import SCENE_CAMERAS, load_stereo_rig  # noqa: E402

COLOURS = [(0, 255, 255), (255, 128, 0), (255, 0, 255), (0, 200, 0),
           (0, 128, 255), (255, 255, 0), (128, 0, 255), (0, 255, 128)]
AXIS_BGR = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]


def draw_axes(img, cam, T, length):
    import cv2
    pts = np.array([T[:3, 3]] + [T[:3, 3] + length * T[:3, k]
                                 for k in range(3)])
    uv = cam.project_ref(pts)
    if not np.isfinite(uv).all():
        return
    o = tuple(int(v) for v in uv[0])
    for k in range(3):
        cv2.line(img, o, tuple(int(v) for v in uv[k + 1]), AXIS_BGR[k], 2)


def in_board(T_ref_board, r) -> Dict[str, Any]:
    """Object pose re-expressed in the board frame, as the numbers we print."""
    Tb = np.linalg.inv(T_ref_board)
    T = Tb @ r["T_ref_object"]
    c = Tb[:3, :3] @ r["centroid_m"] + Tb[:3, 3]
    R = T[:3, :3]
    return {
        "T_board_object": T,
        "centroid_mm": c * 1e3,
        "yaw_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
        "tilt_deg": float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))),
    }


def _median_corners(frames: List[np.ndarray]) -> np.ndarray:
    """Per-corner median over frames.  Frames whose corner numbering flipped
    relative to the latest one are re-ordered first, or the median would
    average corners from opposite ends of the board."""
    ref = np.asarray(frames[-1], dtype=float)
    fixed = []
    for f in frames:
        f = np.asarray(f, dtype=float)
        fixed.append(f if np.abs(f - ref).mean() < np.abs(f[::-1] - ref).mean()
                     else f[::-1])
    return np.median(np.stack(fixed), axis=0)


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def main() -> None:
    import cv2
    from common import DATA_ROOT, timestamp
    from rs_camera import resolve as rs_resolve
    from sync_capture import CameraWorker

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", nargs=2, type=int, required=True,
                    metavar=("COLS", "ROWS"), help="INTERIOR corners")
    ap.add_argument("--grid-square", type=float, required=True,
                    help="measured square side, metres")
    ap.add_argument("--object", default="object",
                    help="label for the log, e.g. cube / flower")
    ap.add_argument("--square-face", type=float, default=None,
                    help="side of a square face, metres: click its 4 "
                         "corners counter-clockwise from above")
    ap.add_argument("--origin-file", default=None,
                    help="reuse a saved origin instead of capturing one")
    ap.add_argument("--stereo-file", default=None,
                    help="default: newest calibration/data/extrinsics/"
                         "stereo_*.json")
    ap.add_argument("--prefer-intrinsics", default="factory",
                    choices=["charuco", "factory"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--depth-range", nargs=2, type=float, default=[0.15, 2.0])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    names = list(SCENE_CAMERAS)
    rig = load_stereo_rig(names, stereo_file=args.stereo_file,
                          prefer_intrinsics=args.prefer_intrinsics)
    print(f"\n  rig: {rig.source}")
    grid = B.GridSpec(args.grid[0], args.grid[1], args.grid_square)
    grid.validate()
    sq_mm = args.grid_square * 1e3

    out = Path(args.outdir) if args.outdir else (
        Path(DATA_ROOT) / "measurements"
        / f"pose_test_{args.object}_{timestamp()}")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "measurements.jsonl"

    model: Optional[np.ndarray] = None
    if args.square_face:
        s = args.square_face
        model = np.array([[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0]], float)
    learned = model is None

    T_rb: Optional[np.ndarray] = None
    if args.origin_file:
        T_rb = np.asarray(json.loads(Path(args.origin_file).read_text())
                          ["T_ref_board_row_major"])
        print(f"  origin from {args.origin_file}")

    workers = {}
    for name, _ in rs_resolve(names):
        w = CameraWorker(name, args.width, args.height, args.fps,
                         global_time=True, keep_images=True)
        w.start()
        workers[name] = w

    W = args.width
    mode = "origin" if T_rb is None else "live"
    frozen: Dict[str, np.ndarray] = {}
    points: List[Dict[str, np.ndarray]] = [{}]
    clicks: List[str] = []
    result: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []
    top_hist: List[np.ndarray] = []

    def on_mouse(event, x, y, flags, param):
        nonlocal result
        if mode != "click" or event != cv2.EVENT_LBUTTONDOWN:
            return
        name = names[0] if x < W else names[1]
        if name not in points[-1]:
            clicks.append(name)
        points[-1][name] = np.array([float(x if x < W else x - W), float(y)])
        result = None
        if len(points[-1]) == 2:
            points.append({})

    win = "pose live test"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(__doc__.split("The model.")[0])

    try:
        while True:
            if mode in ("origin", "live"):
                recs = {n: workers[n].snapshot() for n in names}
                if any(r is None or r.image is None for r in recs.values()):
                    cv2.waitKey(10)
                    continue
                imgs = {n: cv2.cvtColor(r.image, cv2.COLOR_RGB2BGR)
                        for n, r in recs.items()}
            else:
                imgs = frozen
            tiles = {n: imgs[n].copy() for n in names}

            if mode == "origin":
                dets = {n: B.detect_chessboard(imgs[n], grid) for n in names}
                for n in names:
                    tiles[n] = B.annotate(tiles[n], dets[n])
                both = all(d.ok for d in dets.values())
                ready = dets[names[0]].ok
                ## The board is static while the origin is set, so its
                ## corners can be averaged over frames; a single frame's
                ## sub-pixel noise tilts a near-face-on PnP by a degree or two.
                if ready:
                    top_hist.append(dets[names[0]].img_points)
                    del top_hist[:-30]
                else:
                    top_hist.clear()
                footer = ("board in BOTH -- press o to set the origin" if both
                          else f"board in {names[0]} -- press o to set the "
                               f"origin from it alone" if ready
                          else f"lay the board flat, every corner visible "
                               f"in {names[0]}")
            elif mode == "live":
                for n in names:
                    draw_axes(tiles[n], rig[n], T_rb, 3 * args.grid_square)
                footer = (f"{args.object}: SPACE freezes  |  o redoes the "
                          f"origin  |  q quits   ({len(history)} logged)")
            else:
                for i, pt in enumerate(points):
                    col = COLOURS[i % len(COLOURS)]
                    for n, uv in pt.items():
                        x, y = (int(round(v)) for v in uv)
                        cv2.drawMarker(tiles[n], (x, y), col,
                                       cv2.MARKER_CROSS, 16, 2)
                        cv2.putText(tiles[n], str(i), (x + 6, y - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                pend = points[-1]
                if len(pend) == 1:
                    src = next(iter(pend))
                    dst = names[1] if src == names[0] else names[0]
                    try:
                        c = rig.epipolar_curve(
                            src, pend[src], dst,
                            depth_range=tuple(args.depth_range))
                        px = np.asarray(c["pixels"])
                        pts = px[np.isfinite(px).all(axis=1)].astype(int)
                        for p, q in zip(pts[:-1], pts[1:]):
                            cv2.line(tiles[dst], tuple(p), tuple(q),
                                     (60, 220, 255), 1)
                    except Exception:  # noqa: BLE001
                        pass
                for n in names:
                    draw_axes(tiles[n], rig[n], T_rb, 3 * args.grid_square)
                    if result is not None:
                        draw_axes(tiles[n], rig[n], result["T_ref_object"],
                                  0.03)
                footer = ("n accepts + logs, z undo, r reset, b back to live"
                          if result is not None else
                          "click each point in BOTH views, ENTER solves, "
                          "z undo, r reset, b back")

            mont = np.hstack([tiles[n] for n in names])
            cv2.putText(mont, footer, (8, mont.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(win, mont)
            k = cv2.waitKey(1 if mode != "click" else 30) & 0xFF
            if k == ord("q"):
                break

            if mode == "origin" and k == ord("o") and ready:
                if both:
                    bf = board_frame(
                        rig, {n: dets[n].img_points for n in names}, grid)
                    print(f"\n  origin set from BOTH cameras: "
                          f"{bf['n_corners']} corners, board fit "
                          f"{bf['fit_rms_mm']:.2f} mm rms, rays meet within "
                          f"{bf['mean_ray_gap_mm']:.2f} mm on average")
                    if bf["fit_rms_mm"] > 2.0:
                        print("  ! board fit above 2 mm -- check the "
                              "calibration before trusting anything below")
                else:
                    ## The low camera looks across the table almost edge-on
                    ## and usually cannot see a flat board whole.  The rig's
                    ## frame IS the top camera, so it alone suffices.
                    bf = board_frame_single(rig, names[0],
                                            _median_corners(top_hist), grid)
                    print(f"\n  origin set from {names[0]} alone: "
                          f"{bf['n_corners']} corners averaged over "
                          f"{len(top_hist)} frames, reprojection "
                          f"{bf['reprojection_rms_px']:.2f} px rms")
                    if bf["reprojection_rms_px"] > 1.0:
                        print("  ! above 1 px -- glare or a bent board; "
                              "flatten it and press o again")
                T_rb = bf["T_ref_board"]
                (out / "origin.json").write_text(json.dumps({
                    "T_ref_board_row_major": T_rb.tolist(),
                    "ref_frame": rig.ref_frame, "rig": rig.source,
                    "grid": grid.to_dict(), **{k_: v for k_, v in bf.items()
                                               if k_ != "T_ref_board"}},
                    indent=2))
                for n in names:
                    cv2.imwrite(str(out / f"origin_{n}.png"), imgs[n])
                print(f"  saved -> {out / 'origin.json'}   (reuse with "
                      f"--origin-file)")
                mode = "live"
            elif mode == "live" and k == ord("o"):
                mode = "origin"
            elif mode == "live" and k == 32:
                frozen = imgs
                points[:] = [{}]
                clicks.clear()
                result = None
                mode = "click"
            elif mode == "click":
                done = [p for p in points if len(p) == 2]
                if k == ord("b"):
                    mode = "live"
                elif k == ord("r"):
                    points[:] = [{}]
                    clicks.clear()
                    result = None
                elif k == ord("z") and clicks:
                    if not points[-1]:
                        points.pop()
                    points[-1].pop(clicks.pop(), None)
                    result = None
                elif k in (10, 13):
                    if model is not None and len(done) != len(model):
                        print(f"  the model has {len(model)} points, "
                              f"{len(done)} clicked")
                        continue
                    try:
                        result = object_pose(rig, done, model=model)
                    except ValueError as e:
                        print(f"  cannot solve: {e}")
                        continue
                    report(result, T_rb, history, sq_mm, learned and
                           model is None)
                elif k == ord("n") and result is not None:
                    if model is None:
                        ## First accepted view of an unknown shape: its own
                        ## points, in its own frame, become the model.
                        R, t = result["R"], result["position_m"]
                        model = (result["points_m"] - t) @ R
                        print(f"  model learned from this view "
                              f"({len(model)} points); later views are "
                              f"fitted to it")
                    entry = in_board(T_rb, result)
                    i = len(history)
                    for n in names:
                        cv2.imwrite(str(out / f"m{i:02d}_{n}.png"), frozen[n])
                    rec = {
                        "index": i, "object": args.object, "time": time.time(),
                        "T_board_object_row_major":
                            entry["T_board_object"].tolist(),
                        "centroid_mm": entry["centroid_mm"].tolist(),
                        "yaw_deg": entry["yaw_deg"],
                        "tilt_deg": entry["tilt_deg"],
                        "fit_rms_mm": result["fit_rms_mm"],
                        "ray_gap_mm": result["ray_gap_mm"],
                        "clicks": [{n: p.tolist() for n, p in c.items()}
                                   for c in done],
                    }
                    with open(log_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    history.append(entry)
                    print(f"  logged #{i} -> {log_path}")
                    mode = "live"
    finally:
        cv2.destroyAllWindows()
        for w in workers.values():
            w.stop()
        if history:
            print(f"\n  {len(history)} measurements in {log_path}")


def report(r, T_rb, history, sq_mm, first_unknown: bool) -> None:
    e = in_board(T_rb, r)
    c = e["centroid_mm"]
    print(f"\n  centroid  x {c[0]:7.1f}  y {c[1]:7.1f}  z {c[2]:6.1f} mm   "
          f"({c[0] / sq_mm:.2f}, {c[1] / sq_mm:.2f} sq)")
    print(f"  yaw {e['yaw_deg']:7.1f} deg   tilt {e['tilt_deg']:5.1f} deg")
    if e["tilt_deg"] > 90:
        print("  ! tilt > 90: the clicks went clockwise as seen from above")
    print("  ray gaps " + ", ".join(f"{g:.1f}" for g in r["ray_gap_mm"])
          + " mm  (a big one = that pair of clicks were different features)")
    if r["fit_rms_mm"] is not None:
        print(f"  shape fit {r['fit_rms_mm']:.1f} mm rms  (per point "
              + ", ".join(f"{v:.1f}" for v in r["fit_residual_mm"]) + ")")
    elif first_unknown:
        P = r["points_m"] * 1e3
        d = [np.linalg.norm(P[i] - P[(i + 1) % len(P)]) for i in range(len(P))]
        print("  side lengths " + ", ".join(f"{v:.1f}" for v in d)
              + " mm  -- check one against a ruler; n makes this the model")
    if history:
        p = history[-1]
        d = c - p["centroid_mm"]
        print(f"  vs previous: dx {d[0]:+7.1f}  dy {d[1]:+7.1f}  dz "
              f"{d[2]:+6.1f} mm   ({d[0] / sq_mm:+.2f}, {d[1] / sq_mm:+.2f} "
              f"sq)   moved {np.linalg.norm(d[:2]):.1f} mm   dyaw "
              f"{wrap180(e['yaw_deg'] - p['yaw_deg']):+.1f} deg")


if __name__ == "__main__":
    main()
