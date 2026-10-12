"""Live 3-DOF pose of a box on the table from colour masks -- no clicking.

    python reconstruction/cube_pose_live.py \
        --origin-file calibration/data/measurements/<run>/origin.json

The origin comes from pose_live_test.py (lay the board flat, press o, q).
Leave the board where it was; the cube goes on it.

Every frame: the cube's colour is thresholded in both views (the hue ranges
from data_collection/shape_sorter.py, limited to the area around the board
so the shelves cannot interfere), and a 31.5 x 31.5 x 40 mm box resting on
the board is fitted to both silhouettes at once -- see box_fit.py.  The
fitted box is drawn back into both views in green; the mask outline is
cyan.  If they do not hug each other, do not trust the number.

    SPACE   average the next --frames fits, print and log them, with the
            change since the previous logged one (mm and squares)
    m       show the raw colour masks (for checking the thresholds)
    q       quit

Output is the cube's CENTRE in the board frame: x, y from the origin
corner, z = 20 mm by construction (it rests on the board), yaw about the
vertical in (-45, 45] -- a square cube looks the same every 90 degrees.
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
for _p in (str(_HERE), str(_HERE.parent / "calibration"),
           str(_HERE.parent / "data_collection")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import box_fit as BF  # noqa: E402
from rig import SCENE_CAMERAS, load_stereo_rig  # noqa: E402


def colour_mask(bgr: np.ndarray, hues, sat_min: int, val_min: int
                ) -> np.ndarray:
    """Same rule as shape_sorter.class_mask, on a BGR image."""
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = np.zeros(H.shape, dtype=bool)
    for lo, hi in hues:
        m |= (H >= lo) & (H <= hi)
    m &= (S >= sat_min) & (V >= val_min)
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def board_roi(cam, T_ref_board, grid: Dict[str, Any], margin_sq: float,
              height: float, shape) -> np.ndarray:
    """Image region covering the board (plus a margin) up to `height`."""
    import cv2
    s = grid["square_length_m"]
    x0, x1 = -margin_sq * s, (grid["cols"] - 1 + margin_sq) * s
    y0, y1 = -margin_sq * s, (grid["rows"] - 1 + margin_sq) * s
    ## The saved origin may have had its y axis flipped to make z point up,
    ## so the board can lie on either side of y=0: cover both.
    y0, y1 = min(y0, -y1), max(y1, -y0)
    P = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1)
                  for z in (0.0, height)])
    uv = cam.project_ref(P @ T_ref_board[:3, :3].T + T_ref_board[:3, 3])
    roi = np.zeros(shape[:2], np.uint8)
    uv = uv[np.isfinite(uv).all(axis=1)]
    if len(uv) >= 3:
        cv2.fillConvexPoly(roi, cv2.convexHull(uv.astype(np.int32)), 1)
    return roi


def draw_origin(img, cam, T, length) -> None:
    """Board axes: x red, y green, z blue -- the frame every number is in."""
    import cv2
    pts = np.array([T[:3, 3]] + [T[:3, 3] + length * T[:3, k]
                                 for k in range(3)])
    uv = cam.project_ref(pts)
    if not np.isfinite(uv).all():
        return
    o = tuple(int(v) for v in uv[0])
    for k, col in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.line(img, o, tuple(int(v) for v in uv[k + 1]), col, 2)


def wrap(a: float, period: float) -> float:
    return (a + period / 2) % period - period / 2


def main() -> None:
    import cv2
    from common import timestamp
    from rs_camera import resolve as rs_resolve
    from shape_sorter import CLASSES, SAT_MIN, VAL_MIN
    from sync_capture import CameraWorker

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origin-file", required=True)
    ap.add_argument("--object", default="cube", choices=list(CLASSES),
                    help="which shape_sorter colour to segment")
    ap.add_argument("--size", nargs=3, type=float,
                    default=[0.0315, 0.0315, 0.040],
                    metavar=("W", "D", "H"), help="box size, metres")
    ap.add_argument("--hue", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"), help="override hue range (0..179)")
    ap.add_argument("--sat-min", type=int, default=SAT_MIN)
    ap.add_argument("--val-min", type=int, default=VAL_MIN)
    ap.add_argument("--roi-margin", type=float, default=3.0,
                    help="squares around the board to search")
    ap.add_argument("--frames", type=int, default=15,
                    help="fits averaged per SPACE")
    ap.add_argument("--no-truth", action="store_true",
                    help="do not ask for the measured position after SPACE")
    ap.add_argument("--stereo-file", default=None)
    ap.add_argument("--prefer-intrinsics", default="factory",
                    choices=["charuco", "factory"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    names = list(SCENE_CAMERAS)
    rig = load_stereo_rig(names, stereo_file=args.stereo_file,
                          prefer_intrinsics=args.prefer_intrinsics)
    origin = json.loads(Path(args.origin_file).read_text())
    T_rb = np.asarray(origin["T_ref_board_row_major"])
    grid = origin["grid"]
    sq_mm = grid["square_length_m"] * 1e3
    size = tuple(args.size)
    period = np.pi / 2 if abs(size[0] - size[1]) < 1e-4 else np.pi
    hues = [tuple(args.hue)] if args.hue else CLASSES[args.object]["hue"]
    print(f"\n  rig {rig.source}\n  origin {args.origin_file}\n  "
          f"{args.object}: hue {hues}, S>={args.sat_min}, V>={args.val_min}, "
          f"box {size[0]*1e3:.1f} x {size[1]*1e3:.1f} x {size[2]*1e3:.1f} mm")

    out = Path(args.outdir) if args.outdir else (
        Path(args.origin_file).parent / f"{args.object}_hsv_{timestamp()}")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "measurements.jsonl"

    workers = {}
    for name, _ in rs_resolve(names):
        w = CameraWorker(name, args.width, args.height, args.fps,
                         global_time=True, keep_images=True)
        w.start()
        workers[name] = w

    rois: Dict[str, np.ndarray] = {}
    prev: Optional[List[float]] = None
    show_mask = False
    history: List[Dict[str, Any]] = []
    win = "cube pose (HSV) -- SPACE logs, m masks, q quits"
    print("\n  SPACE logs an averaged measurement, m shows the masks, q quits")

    def grab():
        recs = {n: workers[n].snapshot() for n in names}
        if any(r is None or r.image is None for r in recs.values()):
            return None
        return {n: cv2.cvtColor(r.image, cv2.COLOR_RGB2BGR)
                for n, r in recs.items()}

    def measure(imgs, init):
        obs = {}
        for n in names:
            if n not in rois:
                rois[n] = board_roi(rig[n], T_rb, grid, args.roi_margin,
                                    size[2] * 2, imgs[n].shape)
            m = colour_mask(imgs[n], hues, args.sat_min, args.val_min)
            o = BF.observe(m, rois[n])
            if o is not None:
                obs[n] = o
        if len(obs) < 2:
            return obs, None
        free = BF.fit_box(rig, T_rb, obs, size, init=init, yaw_period=period)
        ## A warm start can get stuck if the cube was moved a lot between
        ## frames; a bad fit triggers the full search.
        if init is not None and max(free["rms_px"].values()) > 2.5:
            free = BF.fit_box(rig, T_rb, obs, size, yaw_period=period)
        ## Yaw from the top face's edge directions, then x, y refitted with
        ## yaw held there.  The free fit's yaw is kept for comparison only:
        ## at 640x480 the silhouettes barely constrain it.
        top = names[0]
        yaw, coh = BF.top_face_yaw(imgs[top], obs[top].mask, rig[top], T_rb,
                                   (free["x"], free["y"]), size[2],
                                   symmetry=int(round(2 * np.pi / period)))
        r = BF.fit_box(rig, T_rb, obs, size, init=[free["x"], free["y"]],
                       yaw_period=period, fixed_yaw=yaw)
        r.update(coherence=coh, yaw_silhouette=free["yaw"],
                 x_free=free["x"], y_free=free["y"])
        return obs, r

    try:
        while True:
            imgs = grab()
            if imgs is None:
                cv2.waitKey(10)
                continue
            obs, r = measure(imgs, prev)
            prev = None if r is None else [r["x_free"], r["y_free"],
                                           r["yaw_silhouette"]]

            tiles = {}
            for n in names:
                t = imgs[n].copy()
                if show_mask:
                    m = colour_mask(imgs[n], hues, args.sat_min, args.val_min)
                    t[m > 0] = (255, 0, 255)
                    cs, _ = cv2.findContours(rois[n], cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE) \
                        if n in rois else ([], None)
                    cv2.drawContours(t, cs, -1, (0, 165, 255), 1)
                draw_origin(t, rig[n], T_rb, 3 * grid["square_length_m"])
                if n in obs:
                    cv2.polylines(t, [obs[n].contour.astype(np.int32)], True,
                                  (255, 255, 0), 1)
                if r is not None:
                    uv = r["uv"][n]
                    for i, j in BF.BOX_EDGES:
                        cv2.line(t, tuple(int(v) for v in uv[i]),
                                 tuple(int(v) for v in uv[j]), (0, 255, 0), 1)
                tiles[n] = t
            mont = np.hstack([tiles[n] for n in names])
            if r is not None:
                txt = (f"x {r['x']*1e3:6.1f}  y {r['y']*1e3:6.1f}  z "
                       f"{r['z_centre']*1e3:4.1f} mm   yaw "
                       f"{np.degrees(r['yaw']):5.1f} (coh {r['coherence']:.2f})"
                       f"   fit "
                       + " / ".join(f"{v:.1f}" for v in r["rms_px"].values())
                       + " px")
                col = (0, 255, 0) if max(r["rms_px"].values()) < 2.0 \
                    else (0, 165, 255)
            else:
                miss = [n for n in names if n not in obs]
                txt = f"no {args.object} in {', '.join(miss)} (m shows masks)"
                col = (0, 0, 255)
            cv2.putText(mont, txt, (8, mont.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.imshow(win, mont)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("m"):
                show_mask = not show_mask
            if k == 32 and r is not None:
                log_one(args, rig, T_rb, size, period, sq_mm, grab, measure,
                        r, history, out, log_path, names)
    finally:
        cv2.destroyAllWindows()
        for w in workers.values():
            w.stop()
        if history:
            print(f"\n  {len(history)} measurements in {log_path}")


def log_one(args, rig, T_rb, size, period, sq_mm, grab, measure, r0,
            history, out, log_path, names) -> None:
    """Average --frames fits, cross-check the cameras, print and log."""
    import cv2

    fits, last_imgs, last_obs = [], None, None
    init = [r0["x"], r0["y"], r0["yaw"]]
    while len(fits) < args.frames:
        imgs = grab()
        if imgs is None:
            cv2.waitKey(5)
            continue
        obs, r = measure(imgs, init)
        if r is None:
            continue
        fits.append(r)
        init = [r["x"], r["y"], r["yaw"]]
        last_imgs, last_obs = imgs, obs
    X = np.array([[f["x"], f["y"]] for f in fits]) * 1e3
    ## Average yaw on the circle of its symmetry period.
    k = 2 * np.pi / period
    yaws = np.array([f["yaw"] for f in fits])
    yaw = np.arctan2(np.sin(k * yaws).mean(), np.cos(k * yaws).mean()) / k
    yaw_sd = np.degrees(np.std([wrap(y - yaw, period) for y in yaws]))
    rms = {n: float(np.mean([f["rms_px"][n] for f in fits])) for n in names}
    ys = np.array([f["yaw_silhouette"] for f in fits])
    yaw_sil = np.arctan2(np.sin(k * ys).mean(), np.cos(k * ys).mean()) / k
    coh = float(np.mean([f["coherence"] for f in fits]))
    Xf = np.array([[f["x_free"], f["y_free"]] for f in fits]) * 1e3

    ## Calibration cross-check: each camera alone, on the last frame.
    alone = {n: BF.fit_box(rig, T_rb, {n: last_obs[n]}, size,
                           init=init, yaw_period=period) for n in names}
    a, b = (alone[n] for n in names)
    gap_mm = np.hypot(a["x"] - b["x"], a["y"] - b["y"]) * 1e3
    gap_deg = np.degrees(wrap(a["yaw"] - b["yaw"], period))

    x, y = X.mean(axis=0)
    print(f"\n  #{len(history)}  {args.object} centre   x {x:7.1f}  y {y:7.1f}"
          f"  z {size[2] * 500:5.1f} mm   ({x / sq_mm:+.2f}, {y / sq_mm:+.2f} "
          f"sq)   yaw {np.degrees(yaw):+6.1f} deg")
    print(f"      yaw from top-face edges (coherence {coh:.2f}"
          + (", LOW -- do not trust" if coh < 0.3 else "")
          + f"); silhouette-only yaw would say {np.degrees(yaw_sil):+.1f}")
    print(f"      spread over {len(fits)} frames: x {X[:, 0].std():.2f}  "
          f"y {X[:, 1].std():.2f} mm, yaw {yaw_sd:.2f} deg")
    print("      silhouette fit " + ", ".join(
        f"{n} {v:.2f} px" for n, v in rms.items()))
    print(f"      each camera alone: {names[0]} vs {names[1]} differ by "
          f"{gap_mm:.1f} mm, {gap_deg:+.1f} deg   (the calibration check)")
    if history:
        p = history[-1]
        dx, dy = x - p["x_mm"], y - p["y_mm"]
        print(f"      vs previous: dx {dx:+7.1f}  dy {dy:+7.1f} mm   "
              f"({dx / sq_mm:+.2f}, {dy / sq_mm:+.2f} sq)   moved "
              f"{np.hypot(dx, dy):.1f} mm   dyaw "
              f"{np.degrees(wrap(yaw - p['yaw_rad'], period)):+.1f} deg")
    i = len(history)
    for n in names:
        cv2.imwrite(str(out / f"m{i:02d}_{n}.png"), last_imgs[n])
    rec = {"index": i, "object": args.object, "time": time.time(),
           "x_mm": float(x), "y_mm": float(y), "z_mm": size[2] * 500,
           "yaw_deg": float(np.degrees(yaw)), "yaw_rad": float(yaw),
           "sd_x_mm": float(X[:, 0].std()), "sd_y_mm": float(X[:, 1].std()),
           "sd_yaw_deg": float(yaw_sd), "fit_rms_px": rms,
           "yaw_coherence": coh,
           "yaw_silhouette_deg": float(np.degrees(yaw_sil)),
           "x_free_mm": float(Xf[:, 0].mean()),
           "y_free_mm": float(Xf[:, 1].mean()),
           "camera_alone_gap_mm": float(gap_mm),
           "camera_alone_gap_deg": float(gap_deg),
           "alone": {n: {"x_mm": alone[n]["x"] * 1e3,
                         "y_mm": alone[n]["y"] * 1e3,
                         "yaw_deg": float(np.degrees(alone[n]["yaw"]))}
                     for n in names}}
    if not args.no_truth:
        rec.update(ask_truth(rec, sq_mm, period))
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    history.append(rec)
    print(f"      logged -> {log_path}")


def ask_truth(rec, sq_mm: float, period: float) -> Dict[str, Any]:
    """Ask for the measured placement and score this measurement against it.

    Typed in the TERMINAL, not the image window.  Position in squares from
    the origin corner along the drawn axes (red x, green y), yaw in degrees
    from the grid lines; blank skips."""
    try:
        txt = input("      measured  x_sq y_sq yaw_deg  (blank = skip): ")
    except EOFError:
        return {}
    try:
        tx, ty, tyaw = (float(v) for v in txt.split())
    except ValueError:
        if txt.strip():
            print("      not three numbers -- no truth recorded")
        return {}
    ex, ey = rec["x_mm"] - tx * sq_mm, rec["y_mm"] - ty * sq_mm
    eyaw = np.degrees(wrap(np.radians(rec["yaw_deg"] - tyaw), period))
    esil = np.degrees(wrap(np.radians(rec["yaw_silhouette_deg"] - tyaw),
                           period))
    print(f"      error: x {ex:+.1f}  y {ey:+.1f} mm  ({np.hypot(ex, ey):.1f} mm)"
          f"   yaw {eyaw:+.1f} deg  (silhouette-only {esil:+.1f})")
    return {"truth_x_mm": tx * sq_mm, "truth_y_mm": ty * sq_mm,
            "truth_yaw_deg": tyaw, "err_x_mm": float(ex),
            "err_y_mm": float(ey), "err_yaw_deg": float(eyaw)}


if __name__ == "__main__":
    main()
