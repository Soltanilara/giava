"""Measuring with the scene rig, from the command line.

    status      what the rig is and where it is
    precision   how accurate it can be, before you rely on it
    pair        click a feature in two saved images and get its position
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
