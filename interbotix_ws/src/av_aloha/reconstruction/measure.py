"""Measuring with the scene rig, from the command line.

    status      what the rig is and where it is
    precision   how accurate it can be, before you rely on it
    pair        click a feature in two saved images and get its position
    pose        click >= 3 points on an object in both views: position AND
                orientation (live cameras, or a saved pair)
    known-size  depth from an object of known length, using NO extrinsics

The interactive work normally happens in `world_view.py --measure`, which
has live images.  These exist for the things a GUI is bad at: reporting
what the rig can actually resolve, and re-measuring from images that were
recorded earlier, reproducibly, after the fact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_CALIB = _HERE.parent / "calibration"
for _p in (str(_HERE), str(_CALIB), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rig import SCENE_CAMERAS, load_rig, rig_status  # noqa: E402
from stereo import distance_with_error  # noqa: E402


def _load(args):
    return load_rig(list(args.cameras), frame=args.frame,
                    prefer_intrinsics=args.prefer_intrinsics)


## ------------------------------------------------------------------ ##

def cmd_status(args) -> None:
    st = rig_status(tuple(args.cameras))
    print("\n  scene rig")
    print("  " + "=" * 62)
    for name, e in st["cameras"].items():
        print(f"\n  {name}")
        print(f"    intrinsics : {e.get('intrinsics') or 'MISSING'}")
        print(f"    pose       : "
              + ("known, rigid to '%s' (%s)"
                 % (e.get("rigid_to"), e.get("provenance"))
                 if e.get("pose_known") else "UNKNOWN"))
    print(f"\n  camera-to-camera : {st.get('stereo_file') or 'MISSING'}")
    print(f"  anchored to robot: {'YES' if st['anchored'] else 'NO'}")
    try:
        rig = _load(args)
    except Exception as e:  # noqa: BLE001
        print(f"\n  cannot assemble a rig yet:\n    {e}")
        return
    print("\n" + rig.describe())


def cmd_precision(args) -> None:
    """What this rig can resolve, per point -- before trusting a number.

    A calibration report says how well the solver fitted its own data.
    This says something more useful and much harder to fool: given where
    the cameras ended up, how many millimetres does one pixel of click
    error cost, at the places you actually care about."""
    rig = _load(args)
    if len(rig.posed) < 2:
        raise SystemExit("need at least two posed cameras")
    a, b = rig.posed[0], rig.posed[1]

    print(f"\n  rig frame '{rig.ref_frame}'   baseline "
          f"{rig.baseline_m(a, b) * 1e3:.0f} mm")
    print(f"  assuming a {args.sigma:.1f} px click error\n")
    print(f"  {'point (mm)':<26} {'view angle':>10} {'1-sigma (mm)':>26} "
          f"{'worst':>7}")
    print("  " + "-" * 74)

    lo, hi, n = args.extent[0], args.extent[1], args.grid
    zs = args.heights
    rows, unseen = [], 0
    for z in zs:
        for x in np.linspace(lo, hi, n):
            for y in np.linspace(lo, hi, n):
                p = np.array([x, y, z])
                obs = {}
                for cam in (a, b):
                    uv = rig[cam].project_ref(p)
                    if not rig[cam].in_image(uv)[0]:
                        break
                    obs[cam] = uv
                if len(obs) < 2:
                    unseen += 1
                    continue
                t = rig.triangulate(obs, sigma_px=args.sigma)
                s = t.sigma_axes_mm[0]
                rows.append({
                    "point_mm": (p * 1e3).tolist(),
                    "viewing_angle_deg": t.meta.get("viewing_angle_deg"),
                    "sigma_mm": s.tolist(),
                    "worst_mm": float(s[0]),
                })
    if not rows:
        raise SystemExit(
            "no sampled point is visible to both cameras. Either the "
            "extrinsics are wrong or --extent/--heights do not overlap "
            "the shared field of view.")

    rows.sort(key=lambda r: r["worst_mm"])
    show = rows if args.all else (rows[:3] + rows[-3:])
    for r in show:
        p = r["point_mm"]
        print(f"  [{p[0]:+7.0f},{p[1]:+7.0f},{p[2]:+7.0f}]"
              f"  {r['viewing_angle_deg']:>8.1f}deg"
              f"   {' x '.join(f'{v:5.2f}' for v in r['sigma_mm'])}"
              f"  {r['worst_mm']:6.2f}")
        if not args.all and r is show[2]:
            print("  " + "." * 74)

    worst = np.array([r["worst_mm"] for r in rows])
    ang = np.array([r["viewing_angle_deg"] for r in rows])
    print("  " + "-" * 74)
    print(f"  {len(rows)} points visible to both, {unseen} seen by fewer")
    print(f"  worst-axis 1-sigma: {worst.min():.2f} best, "
          f"{np.median(worst):.2f} median, {worst.max():.2f} worst")
    print(f"  viewing angle     : {ang.min():.0f}-{ang.max():.0f} deg "
          f"(90 is ideal; near 0 or 180 cannot resolve depth)")
    print(f"""
  What that means for the questions this rig was built to answer:

    a 10 cm frame offset      {'RESOLVED EASILY' if worst.max() < 20 else 'still resolvable'}
      -- {100.0 / max(worst.max(), 1e-9):.0f}x the worst-case uncertainty
    a 1 mm link-length check   {'usable' if np.median(worst) < 1.0 else 'NOT single-shot'}
      -- median uncertainty is {np.median(worst):.2f} mm per point, so
         average over repeats, or measure a longer span and divide

  These are CLICK-noise figures only. They say nothing about a wrong
  baseline, which biases every length by the same factor and can only be
  caught against a physical ruler.""")


def cmd_pair(args) -> None:
    """Click the same feature in two saved images."""
    import cv2

    rig = _load(args)
    imgs, names = {}, list(args.cameras)
    for name, path in zip(names, args.images):
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"could not read {path}")
        imgs[name] = img
        if (img.shape[1], img.shape[0]) != (rig[name].width, rig[name].height):
            raise SystemExit(
                f"{path} is {img.shape[1]}x{img.shape[0]} but {name}'s "
                f"intrinsics are for {rig[name].width}x{rig[name].height}. "
                f"Intrinsics are RESOLUTION SPECIFIC -- measuring across "
                f"that mismatch would be wrong by the scale ratio.")

    picks: Dict[str, np.ndarray] = {}
    print("\n  click the SAME feature in each window; q when done, r to reset")

    def make_cb(name):
        def cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                picks[name] = np.array([float(x), float(y)])
                print(f"    {name}: ({x}, {y})")
        return cb

    for name in names:
        cv2.namedWindow(name)
        cv2.setMouseCallback(name, make_cb(name))

    result = None
    while True:
        for name in names:
            vis = imgs[name].copy()
            if name in picks:
                x, y = (int(v) for v in picks[name])
                cv2.drawMarker(vis, (x, y), (0, 255, 255),
                               cv2.MARKER_CROSS, 20, 1)
            ## Once one image has a pick, draw where the feature can be in
            ## the other -- the same epipolar constraint the GUI shows.
            others = [n for n in names if n in picks and n != name]
            if others and name not in picks:
                try:
                    c = rig.epipolar_curve(
                        others[0], picks[others[0]], name,
                        depth_range=tuple(args.depth_range))
                    px = np.asarray(c["pixels"])
                    ok = np.isfinite(px).all(axis=1)
                    pts = px[ok].astype(int)
                    for p, q in zip(pts[:-1], pts[1:]):
                        cv2.line(vis, tuple(p), tuple(q), (60, 220, 255), 1)
                except Exception:  # noqa: BLE001
                    pass
            cv2.imshow(name, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            picks.clear()
        if len(picks) == len(names):
            result = rig.triangulate(dict(picks), sigma_px=args.sigma)
            print("\n" + result.describe(rig.ref_frame))
            picks.clear() if args.repeat else None
            if not args.repeat:
                break
    cv2.destroyAllWindows()
    if result is None:
        print("\n  nothing measured")


def _grab_pair(args, names: List[str]) -> Dict[str, np.ndarray]:
    """Live preview of every camera in one window; SPACE freezes a set."""
    import cv2
    from common import DATA_ROOT, timestamp
    from rs_camera import resolve as rs_resolve
    from sync_capture import CameraWorker

    workers = {}
    for name, _ in rs_resolve(names):
        w = CameraWorker(name, args.width, args.height, args.fps,
                         global_time=True, keep_images=True)
        w.start()
        workers[name] = w
    print("\n  live: SPACE freezes the views to click on, q quits")
    frozen = None
    try:
        while frozen is None:
            recs = {n: workers[n].snapshot() for n in names}
            if any(r is None or r.image is None for r in recs.values()):
                cv2.waitKey(10)
                continue
            imgs = {n: cv2.cvtColor(r.image, cv2.COLOR_RGB2BGR)
                    for n, r in recs.items()}
            cv2.imshow("pose -- SPACE freezes, q quits",
                       np.hstack([imgs[n] for n in names]))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                raise SystemExit("  quit before freezing")
            if k == 32:
                frozen = imgs
    finally:
        cv2.destroyAllWindows()
        for w in workers.values():
            w.stop()
    out = Path(args.outdir) if args.outdir else (
        Path(DATA_ROOT) / "measurements" / f"pose_{timestamp()}")
    out.mkdir(parents=True, exist_ok=True)
    for n, img in frozen.items():
        cv2.imwrite(str(out / f"{n}.png"), img)
    args.outdir = str(out)
    print(f"  frozen views saved -> {out}  (re-measure later with "
          f"--images {' '.join(str(out / f'{n}.png') for n in names)})")
    return frozen


def cmd_pose(args) -> None:
    """Click the same N points on an object in both views -> its pose."""
    import cv2
    from common import DATA_ROOT, matrix_to_quat_wxyz, save_json, timestamp
    from object_pose import object_pose

    rig = _load(args)
    names = list(args.cameras)
    if len(names) != 2:
        raise SystemExit("pose works on exactly two cameras")
    model = None
    if args.model:
        from common import load_json
        model = np.asarray(load_json(Path(args.model))["points"], dtype=float)
        print(f"  model: {len(model)} points from {args.model} -- click them "
              f"in THAT order")

    if args.images:
        imgs = {}
        for name, path in zip(names, args.images):
            img = cv2.imread(str(path))
            if img is None:
                raise SystemExit(f"could not read {path}")
            imgs[name] = img
    else:
        imgs = _grab_pair(args, names)
    for name, img in imgs.items():
        if (img.shape[1], img.shape[0]) != (rig[name].width, rig[name].height):
            raise SystemExit(
                f"{name} image is {img.shape[1]}x{img.shape[0]} but its "
                f"intrinsics are for {rig[name].width}x{rig[name].height}")

    W = imgs[names[0]].shape[1]
    points: List[Dict[str, np.ndarray]] = [{}]   # last entry is in progress
    clicks: List[str] = []                       # camera of each click, for undo
    result: Optional[Dict[str, Any]] = None

    def on_mouse(event, x, y, flags, param):
        nonlocal result
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        name = names[0] if x < W else names[1]
        u = float(x if x < W else x - W)
        if name not in points[-1]:
            clicks.append(name)
        points[-1][name] = np.array([u, float(y)])
        result = None
        if len(points[-1]) == 2:
            print(f"    point {len(points) - 1}: "
                  + "  ".join(f"{n} ({p[0]:.0f}, {p[1]:.0f})"
                              for n, p in points[-1].items()))
            points.append({})

    win = "pose -- click points in BOTH views; ENTER solves, z undo, r reset, q quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(f"""
  Click a point on the object in one view, then the SAME point in the
  other (a curve shows where it can be). Repeat for >= 3 points that are
  not on one line -- {"in the model's order" if model is not None else
  "point 0 is the origin, 0->1 is +x, and 0-1-2 sets the plane"}.
  ENTER solves, z undoes a click, r resets, s saves, q quits.
""")
    colours = [(0, 255, 255), (255, 128, 0), (255, 0, 255), (0, 200, 0),
               (0, 128, 255), (255, 255, 0)]
    while True:
        tiles = {n: imgs[n].copy() for n in names}
        for i, pt in enumerate(points):
            col = colours[i % len(colours)]
            for n, uv in pt.items():
                x, y = (int(round(v)) for v in uv)
                cv2.drawMarker(tiles[n], (x, y), col, cv2.MARKER_CROSS, 16, 2)
                cv2.putText(tiles[n], str(i), (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        pend = points[-1]
        if len(pend) == 1:
            src = next(iter(pend))
            dst = names[1] if src == names[0] else names[0]
            try:
                c = rig.epipolar_curve(src, pend[src], dst,
                                       depth_range=tuple(args.depth_range))
                px = np.asarray(c["pixels"])
                pts = px[np.isfinite(px).all(axis=1)].astype(int)
                for p, q in zip(pts[:-1], pts[1:]):
                    cv2.line(tiles[dst], tuple(p), tuple(q), (60, 220, 255), 1)
            except Exception:  # noqa: BLE001
                pass
        if result is not None:
            T = result["T_ref_object"]
            L = args.axis_length
            axes = np.array([T[:3, 3]] + [T[:3, 3] + L * T[:3, k]
                                          for k in range(3)])
            for n in names:
                uv = rig[n].project_ref(axes)
                if not np.isfinite(uv).all():
                    continue
                o = tuple(int(v) for v in uv[0])
                for k, col in enumerate([(0, 0, 255), (0, 255, 0),
                                         (255, 0, 0)]):
                    cv2.line(tiles[n], o, tuple(int(v) for v in uv[k + 1]),
                             col, 2)
        cv2.imshow(win, np.hstack([tiles[n] for n in names]))
        k = cv2.waitKey(30) & 0xFF
        done = [p for p in points if len(p) == 2]
        if k == ord("q"):
            break
        if k == ord("r"):
            points[:] = [{}]
            clicks.clear()
            result = None
        elif k == ord("z") and clicks:
            if not points[-1]:
                points.pop()
            points[-1].pop(clicks.pop(), None)
            result = None
        elif k in (13, 10) and result is None:
            if model is not None and len(done) != len(model):
                print(f"  model has {len(model)} points, {len(done)} clicked")
                continue
            try:
                result = object_pose(rig, done, model=model,
                                     sigma_px=args.sigma)
            except ValueError as e:
                print(f"  cannot solve: {e}")
                continue
            _print_pose(result, matrix_to_quat_wxyz)
        elif k == ord("s") and result is not None:
            out = Path(args.outdir) if args.outdir else (
                Path(DATA_ROOT) / "measurements" / f"pose_{timestamp()}")
            out.mkdir(parents=True, exist_ok=True)
            doc = {
                "ref_frame": result["ref_frame"],
                "rig": rig.source,
                "method": result["method"],
                "T_ref_object_row_major": result["T_ref_object"].tolist(),
                "quat_wxyz": matrix_to_quat_wxyz(result["R"]).tolist(),
                "points_m": result["points_m"].tolist(),
                "clicks": [{n: p.tolist() for n, p in c.items()}
                           for c in done],
                "ray_gap_mm": result["ray_gap_mm"],
                "fit_rms_mm": result["fit_rms_mm"],
                "images": args.images or [str(out / f"{n}.png")
                                          for n in names],
            }
            path = save_json(doc, out / f"pose_{timestamp()}.json")
            cv2.imwrite(str(out / f"pose_{timestamp()}_overlay.png"),
                        np.hstack([tiles[n] for n in names]))
            print(f"  saved -> {path}")
    cv2.destroyAllWindows()


def _print_pose(r: Dict[str, Any], to_quat) -> None:
    p = r["position_m"] * 1e3
    R = r["R"]
    print(f"\n  object pose in '{r['ref_frame']}'   ({r['method']})")
    print(f"    position  [{p[0]:8.1f} {p[1]:8.1f} {p[2]:8.1f}] mm")
    c = r["centroid_m"] * 1e3
    print(f"    centroid  [{c[0]:8.1f} {c[1]:8.1f} {c[2]:8.1f}] mm")
    print(f"    quat wxyz {np.array2string(to_quat(R), precision=4)}")
    for k, ax in enumerate("xyz"):
        print(f"    object +{ax} = [{R[0, k]:+.3f} {R[1, k]:+.3f} "
              f"{R[2, k]:+.3f}]")
    gaps = ", ".join(f"{g:.1f}" for g in r["ray_gap_mm"])
    print(f"    ray gaps per point: {gaps} mm   (how far the two rays "
          f"miss each other;\n      a big one = that point was clicked on "
          f"different features)")
    if r["fit_rms_mm"] is not None:
        res = ", ".join(f"{v:.1f}" for v in r["fit_residual_mm"])
        print(f"    model fit {r['fit_rms_mm']:.1f} mm rms  (per point: "
              f"{res})")
    print("    axes drawn: x red, y green, z blue.  s saves.")


def cmd_known_size(args) -> None:
    rig = _load(args)
    r = rig.depth_from_known_size(args.camera, args.a, args.b, args.size)
    print(f"""
  {args.camera}: two clicks {r['pixel_separation']:.1f} px apart subtend
  {r['subtended_angle_deg']:.3f} deg. For a known length of
  {args.size * 1e3:.1f} mm that puts it at

      {r['depth_m'] * 1e3:.1f} mm      (small-angle form: """
          f"""{r['depth_small_angle_m'] * 1e3:.1f} mm, """
          f"""{r['small_angle_disagreement_pct']:.2f} % apart)

  Uses NO extrinsics -- this is an independent check on a triangulated
  depth, not another consumer of the same calibration.

  {r['assumption']}
  One pixel of click error moves this by """
          f"""{r['sensitivity_pct_per_px']:.2f} %, i.e. """
          f"""{r['depth_m'] * r['sensitivity_pct_per_px'] * 10:.1f} mm.""")


## ------------------------------------------------------------------ ##

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    def common(p):
        p.add_argument("--cameras", nargs="+", default=list(SCENE_CAMERAS))
        p.add_argument("--frame", default="auto",
                       choices=["auto", "base", "stereo"],
                       help="'base' insists on robot world coordinates and "
                            "refuses if the rig is not anchored")
        p.add_argument("--prefer-intrinsics", default="charuco",
                       choices=["charuco", "factory"])
        p.add_argument("--sigma", type=float, default=1.0,
                       help="assumed click error, pixels")
        return p

    sub = ap.add_subparsers(dest="cmd", required=True)
    common(sub.add_parser("status")).set_defaults(fn=cmd_status)
    ## --list is what scene_extrinsics.py's success message suggests.
    common(sub.add_parser("list")).set_defaults(fn=cmd_status)

    pr = common(sub.add_parser(
        "precision", help="how many mm one pixel of click error costs"))
    pr.add_argument("--extent", nargs=2, type=float, default=[-0.35, 0.35],
                    metavar=("LO", "HI"), help="x and y sweep, metres")
    pr.add_argument("--heights", nargs="+", type=float,
                    default=[0.0, 0.15, 0.30], help="z values, metres")
    pr.add_argument("--grid", type=int, default=7)
    pr.add_argument("--all", action="store_true",
                    help="every sampled point, not just the extremes")
    pr.set_defaults(fn=cmd_precision)

    pa = common(sub.add_parser("pair", help="click a feature in two images"))
    pa.add_argument("--images", nargs="+", required=True,
                    help="one image per --cameras entry, same order")
    pa.add_argument("--depth-range", nargs=2, type=float, default=[0.15, 2.0])
    pa.add_argument("--repeat", action="store_true",
                    help="keep measuring instead of exiting after one")
    pa.set_defaults(fn=cmd_pair)

    po = common(sub.add_parser(
        "pose", help="object position + orientation from >= 3 points"))
    po.add_argument("--images", nargs="+", default=None,
                    help="one image per --cameras entry; omit to grab live")
    po.add_argument("--model", default=None,
                    help='JSON {"points": [[x,y,z], ...]} -- the clicked '
                         "points in the object's own frame, metres")
    po.add_argument("--depth-range", nargs=2, type=float, default=[0.15, 2.0])
    po.add_argument("--axis-length", type=float, default=0.05,
                    help="drawn axis length, metres")
    po.add_argument("--outdir", default=None)
    po.add_argument("--width", type=int, default=640)
    po.add_argument("--height", type=int, default=480)
    po.add_argument("--fps", type=int, default=30)
    po.set_defaults(fn=cmd_pose)

    ks = common(sub.add_parser(
        "known-size", help="depth from an object of known length"))
    ks.add_argument("--camera", required=True)
    ks.add_argument("--a", nargs=2, type=float, required=True,
                    metavar=("U", "V"))
    ks.add_argument("--b", nargs=2, type=float, required=True,
                    metavar=("U", "V"))
    ks.add_argument("--size", type=float, required=True,
                    help="the known length, in METRES")
    ks.set_defaults(fn=cmd_known_size)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
