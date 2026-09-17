"""Compare RealSense factory intrinsics against an OpenCV ChArUco result
(PHASE 6).

Reports two independent things, because they answer different questions.

1. PARAMETER COMPARISON
   fx, fy, cx, cy and the image resolution are directly comparable -- both
   calibrations mean the same thing by them.  Absolute and relative
   differences are reported.

2. PIXEL-DOMAIN COMPARISON
   The distortion coefficients are NOT directly comparable, and subtracting
   them term by term would be meaningless:

     * librealsense reports the D405 colour stream as
       ``inverse_brown_conrady``.  Its coefficients are applied when going
       from a PIXEL to a RAY (rs2_deproject_pixel_to_point).
     * OpenCV's coefficients are applied in the opposite direction, when
       projecting a RAY to a PIXEL (cv2.projectPoints).

   They are different parameterisations of different mappings, so k1 vs k1
   tells you nothing.  Instead this compares what the two calibrations
   actually DO: a grid of pixels is deprojected to normalised rays under
   each model, and the disagreement is reported back in pixels.  That
   number is parameterisation independent and is the one that matters for
   reconstruction.

WHAT THIS TOOL DOES NOT DO
   It does not decide which calibration is right.  Two calibrations
   agreeing tells you they are consistent, not that either matches the
   physical optics -- a systematic error in the board's measured square
   length, or a stale factory calibration, is invisible to any numerical
   comparison.  Settle it with a physical measurement: image an object of
   known size at a known distance and check the reconstructed size.

Examples

    # newest factory + newest charuco result for one camera
    python calibration/compare_intrinsics.py --camera right_wrist

    # specific files
    python calibration/compare_intrinsics.py \\
        --factory .../factory_intrinsics_A.json \\
        --charuco .../charuco_intrinsics_B.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ is a SIBLING of this package now, not the
## parent it used to be.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
for _p in (str(HERE), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    DIR_CAMERAS,
    DIR_COMPARE,
    ensure_dirs,
    latest_matching,
    load_json,
    provenance,
    save_json,
    timestamp,
)


## ------------------------------------------------------------------ ##
## Loading
## ------------------------------------------------------------------ ##

def camera_dir(camera: str) -> Path:
    """calibration/data/cameras/<camera>_<serial>/ for a camera name."""
    hits = sorted(DIR_CAMERAS.glob(f"{camera}_*"))
    if not hits:
        raise FileNotFoundError(
            f"no results directory for camera '{camera}' under "
            f"{DIR_CAMERAS}.\n"
            f"  Run rs_intrinsics.py and charuco_calibrate.py first.")
    return hits[-1]


def load_factory(path: Path, stream: str = "color") -> Dict[str, Any]:
    doc = load_json(path)
    if stream not in doc.get("streams", {}):
        raise KeyError(
            f"{path} has no '{stream}' stream "
            f"(has: {sorted(doc.get('streams', {}))})")
    s = doc["streams"][stream]
    i = s["intrinsics"]
    return {
        "label": "RealSense factory",
        "path": str(path),
        "camera": doc.get("camera"),
        "serial": doc.get("serial"),
        "width": i["width"], "height": i["height"],
        "fx": i["fx"], "fy": i["fy"], "cx": i["cx"], "cy": i["cy"],
        "model": i["model"],
        "coeffs": list(i["coeffs"]),
        "date": doc.get("metadata", {}).get("date"),
        "extra": {"fov_deg": i.get("fov_deg"),
                  "fps": s["identity"].get("fps"),
                  "format": s["identity"].get("format")},
    }


def load_charuco(path: Path) -> Dict[str, Any]:
    doc = load_json(path)
    return {
        "label": "OpenCV ChArUco",
        "path": str(path),
        "camera": doc.get("camera"),
        "serial": doc.get("serial"),
        "width": doc["image_size"]["width"],
        "height": doc["image_size"]["height"],
        "fx": doc["fx"], "fy": doc["fy"], "cx": doc["cx"], "cy": doc["cy"],
        "model": doc["distortion_model"],
        "coeffs": list(doc["distortion_coefficients"]),
        "date": doc.get("metadata", {}).get("date"),
        "extra": {
            "rms_px": doc.get("rms_reprojection_error_px"),
            "n_images": doc.get("n_images_used"),
            "coverage_fraction": doc.get("coverage_fraction"),
            "board": doc.get("board"),
        },
    }


## ------------------------------------------------------------------ ##
## Comparison
## ------------------------------------------------------------------ ##

def compare_parameters(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key in ("fx", "fy", "cx", "cy"):
        va, vb = float(a[key]), float(b[key])
        diff = vb - va
        out[key] = {
            "factory": va,
            "charuco": vb,
            "absolute_difference_px": diff,
            ## Relative to the factory value.  Meaningful for fx/fy (a
            ## focal length has a natural zero); for cx/cy the absolute
            ## pixel difference is the number to look at, since the origin
            ## is the image corner, not anything physical.
            "relative_difference_percent": (100.0 * diff / va) if va else None,
        }
    out["resolution"] = {
        "factory": [a["width"], a["height"]],
        "charuco": [b["width"], b["height"]],
        "match": [a["width"], a["height"]] == [b["width"], b["height"]],
    }
    ## Aspect ratio and principal-point offset from centre are the two
    ## derived quantities where a mistake usually shows up first.
    out["derived"] = {
        "aspect_fy_over_fx": {"factory": a["fy"] / a["fx"],
                              "charuco": b["fy"] / b["fx"]},
        "principal_offset_from_centre_px": {
            "factory": [a["cx"] - a["width"] / 2.0,
                        a["cy"] - a["height"] / 2.0],
            "charuco": [b["cx"] - b["width"] / 2.0,
                        b["cy"] - b["height"] / 2.0]},
    }
    return out


def rays_realsense(uv: np.ndarray, c: Dict[str, Any]) -> np.ndarray:
    """Pixels -> normalised rays using librealsense's own model."""
    import pyrealsense2 as rs

    intr = rs.intrinsics()
    intr.width, intr.height = int(c["width"]), int(c["height"])
    intr.fx, intr.fy = float(c["fx"]), float(c["fy"])
    intr.ppx, intr.ppy = float(c["cx"]), float(c["cy"])
    model = str(c["model"]).split(".")[-1]
    intr.model = getattr(rs.distortion, model, rs.distortion.brown_conrady)
    intr.coeffs = [float(v) for v in c["coeffs"]][:5]
    out = np.empty((len(uv), 2), dtype=float)
    for i, (u, v) in enumerate(uv):
        p = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], 1.0)
        out[i] = (p[0], p[1])
    return out


def rays_opencv(uv: np.ndarray, c: Dict[str, Any]) -> np.ndarray:
    """Pixels -> normalised rays using OpenCV's model (same direction)."""
    import cv2

    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])
    dist = np.array([float(v) for v in c["coeffs"]], dtype=float)
    pts = uv.reshape(-1, 1, 2).astype(np.float64)
    und = cv2.undistortPoints(pts, K, dist)
    return und.reshape(-1, 2)


def compare_mapping(a: Dict[str, Any], b: Dict[str, Any],
                    grid: int = 25) -> Dict[str, Any]:
    """How far apart do the two calibrations put the same pixel's ray?

    Both directions are 'undistort': pixel -> normalised ray.  The
    disagreement is converted back to pixels using the factory focal
    lengths, so the number reads as 'this many pixels of disagreement'."""
    w, h = int(a["width"]), int(a["height"])
    us = np.linspace(0, w - 1, grid)
    vs = np.linspace(0, h - 1, grid)
    uu, vv = np.meshgrid(us, vs)
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1)

    ra = rays_realsense(uv, a)
    rb = rays_opencv(uv, b)
    d = rb - ra
    ## Back to pixels through the factory focal lengths.
    dpx = np.stack([d[:, 0] * a["fx"], d[:, 1] * a["fy"]], axis=1)
    mag = np.linalg.norm(dpx, axis=1)

    ## Radius from the principal point: distortion disagreement grows
    ## outward, and the edge value is what limits wide-baseline matching.
    r = np.linalg.norm(uv - np.array([a["cx"], a["cy"]]), axis=1)
    r_norm = r / r.max()
    bands = []
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        m = (r_norm >= lo) & (r_norm < hi + (1e-9 if hi == 1.0 else 0))
        if m.any():
            bands.append({"radius_fraction": [lo, hi],
                          "n_points": int(m.sum()),
                          "mean_px": float(mag[m].mean()),
                          "max_px": float(mag[m].max())})

    return {
        "description": (
            "Each sampled pixel is deprojected to a normalised ray under "
            "both calibrations; the difference is expressed back in pixels "
            "using the factory focal lengths. Direction is pixel->ray for "
            "both, so the models' opposite conventions are handled "
            "correctly."),
        "grid": grid,
        "n_points": int(len(uv)),
        "mean_px": float(mag.mean()),
        "median_px": float(np.median(mag)),
        "p95_px": float(np.percentile(mag, 95)),
        "max_px": float(mag.max()),
        "max_at_pixel": uv[int(np.argmax(mag))].tolist(),
        "by_radius": bands,
    }


## ------------------------------------------------------------------ ##
## Report
## ------------------------------------------------------------------ ##

def report(a, b, params, mapping) -> None:
    print()
    print("=" * 78)
    print("  INPUTS")
    print("=" * 78)
    for c in (a, b):
        print(f"\n  {c['label']}")
        print(f"      file    : {c['path']}")
        print(f"      camera  : {c['camera']}   serial {c['serial']}")
        print(f"      date    : {c['date']}")
        print(f"      size    : {c['width']} x {c['height']}")
        print(f"      model   : {c['model']}")
        for k, v in (c.get("extra") or {}).items():
            if k == "board":
                continue
            print(f"      {k:8s}: {v}")

    print()
    print("=" * 78)
    print("  1. DIRECTLY COMPARABLE PARAMETERS")
    print("=" * 78)
    res = params["resolution"]
    if not res["match"]:
        print(f"\n  *** RESOLUTION MISMATCH: factory {res['factory']} vs "
              f"charuco {res['charuco']} ***")
        print("  Intrinsics are resolution specific. These two calibrations")
        print("  describe different image sizes and are NOT comparable as")
        print("  they stand -- recalibrate at a matching resolution.")
    else:
        print(f"\n  resolution: {res['factory'][0]} x {res['factory'][1]}  "
              f"(match)")
    print()
    print(f"  {'':6s} {'factory':>14s} {'charuco':>14s} {'diff':>12s} "
          f"{'rel %':>10s}")
    print("  " + "-" * 60)
    for key in ("fx", "fy", "cx", "cy"):
        p = params[key]
        rel = p["relative_difference_percent"]
        rel_s = f"{rel:>+10.4f}" if rel is not None else f"{'--':>10s}"
        print(f"  {key:6s} {p['factory']:>14.5f} {p['charuco']:>14.5f} "
              f"{p['absolute_difference_px']:>+12.5f} {rel_s}")
    print("  " + "-" * 60)
    print("  (cx/cy: read the absolute pixel difference, not the percentage "
          "-- the")
    print("   principal point's origin is the image corner, not anything "
          "physical.)")
    d = params["derived"]
    print(f"  aspect fy/fx : factory {d['aspect_fy_over_fx']['factory']:.6f}"
          f"   charuco {d['aspect_fy_over_fx']['charuco']:.6f}")
    po = d["principal_offset_from_centre_px"]
    print(f"  principal pt offset from image centre [px]:")
    print(f"      factory ({po['factory'][0]:+.2f}, {po['factory'][1]:+.2f})"
          f"   charuco ({po['charuco'][0]:+.2f}, {po['charuco'][1]:+.2f})")

    print()
    print("=" * 78)
    print("  2. DISTORTION")
    print("=" * 78)
    print(f"""
  factory model : {a['model']}
  charuco model : {b['model']}

  These are DIFFERENT PARAMETERISATIONS RUNNING IN OPPOSITE DIRECTIONS.
  librealsense's inverse_brown_conrady coefficients undistort (pixel->ray);
  OpenCV's distort (ray->pixel).  Comparing them term by term is not
  meaningful, so no differences are computed below -- the coefficients are
  listed only for the record.
""")
    names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
    n = max(len(a["coeffs"]), len(b["coeffs"]))
    print(f"  {'':4s} {'factory':>14s} {'charuco':>14s}")
    print("  " + "-" * 36)
    for i in range(n):
        va = f"{a['coeffs'][i]:+.8f}" if i < len(a["coeffs"]) else "--"
        vb = f"{b['coeffs'][i]:+.8f}" if i < len(b["coeffs"]) else "--"
        print(f"  {names[i]:4s} {va:>14s} {vb:>14s}")

    print()
    print("=" * 78)
    print("  3. WHAT THE TWO CALIBRATIONS ACTUALLY DO  (pixel domain)")
    print("=" * 78)
    m = mapping
    print(f"""
  {m['n_points']} pixels sampled on a {m['grid']}x{m['grid']} grid, each
  deprojected to a ray under both calibrations. Disagreement, in pixels:

      mean   {m['mean_px']:8.3f}
      median {m['median_px']:8.3f}
      p95    {m['p95_px']:8.3f}
      max    {m['max_px']:8.3f}   at pixel {m['max_at_pixel']}

  by distance from the principal point:
""")
    print(f"      {'radius':>14s} {'points':>8s} {'mean px':>10s} "
          f"{'max px':>10s}")
    print("      " + "-" * 44)
    for band in m["by_radius"]:
        lo, hi = band["radius_fraction"]
        print(f"      {f'{lo:.2f}-{hi:.2f}':>14s} {band['n_points']:>8d} "
              f"{band['mean_px']:>10.3f} {band['max_px']:>10.3f}")

    print()
    print("=" * 78)
    print("  INTERPRETATION")
    print("=" * 78)
    print("""
  Nothing here says which calibration is correct. Both could share a
  systematic error, and the ChArUco result additionally inherits any error
  in the board's measured square length (which shifts metric scale, though
  not fx/fy/cx/cy).

  To settle it experimentally:
    * Image an object of known size at a known, independently measured
      distance and check the size each calibration reconstructs.
    * Undistort a picture of a straight edge near the image border with
      each calibration and see which one straightens it.
    * Repeat the ChArUco calibration from a fresh capture session; the
      spread between two of your own runs bounds how much of any
      factory-vs-ChArUco difference is just noise.
""")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default=None,
                    help="GIAVA camera name; picks the newest factory and "
                         "charuco result for it")
    ap.add_argument("--factory", default=None,
                    help="explicit factory intrinsics JSON")
    ap.add_argument("--charuco", default=None,
                    help="explicit charuco intrinsics JSON")
    ap.add_argument("--stream", default="color",
                    help="which factory stream to compare (default: color)")
    ap.add_argument("--grid", type=int, default=25,
                    help="pixel grid resolution for the mapping comparison")
    ap.add_argument("--out", default=None, help="write the report to JSON")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    print()
    print("#" * 78)
    print("#  Factory vs OpenCV calibration comparison  (PHASE 6)")
    print("#" * 78)

    fpath, cpath = _resolve_inputs(args)
    a = load_factory(fpath, args.stream)
    b = load_charuco(cpath)

    params = compare_parameters(a, b)
    mapping = compare_mapping(a, b, args.grid)
    report(a, b, params, mapping)

    path = (Path(args.out) if args.out else
            DIR_COMPARE / f"compare_{a['camera'] or 'camera'}_{timestamp()}.json")
    save_json({
        "metadata": provenance("intrinsics_comparison",
                               camera=a["camera"], serial=a["serial"],
                               stream=args.stream),
        "factory": a,
        "charuco": b,
        "parameter_comparison": params,
        "mapping_comparison": mapping,
        "caveat": (
            "Numerical agreement does not establish correctness. The "
            "distortion coefficients use different parameterisations in "
            "opposite directions and are NOT differenced. Validate against "
            "a physical measurement."),
    }, path, overwrite=args.overwrite)
    print(f"  report saved to {path}\n")


def _resolve_inputs(args) -> Tuple[Path, Path]:
    if args.factory and args.charuco:
        return Path(args.factory), Path(args.charuco)
    if not args.camera:
        raise SystemExit(
            "  give --camera, or both --factory and --charuco")
    d = camera_dir(args.camera)
    f = Path(args.factory) if args.factory else latest_matching(
        d, "factory_intrinsics_*.json")
    c = Path(args.charuco) if args.charuco else latest_matching(
        d, "charuco_intrinsics_*.json")
    if f is None:
        raise SystemExit(
            f"  no factory_intrinsics_*.json in {d}\n"
            f"  run: python calibration/rs_intrinsics.py --cameras "
            f"{args.camera}")
    if c is None:
        raise SystemExit(
            f"  no charuco_intrinsics_*.json in {d}\n"
            f"  run: python calibration/charuco_capture.py --camera "
            f"{args.camera}\n"
            f"  then: python calibration/charuco_calibrate.py --images "
            f"<dir> --camera {args.camera}")
    return f, c


if __name__ == "__main__":
    main()
