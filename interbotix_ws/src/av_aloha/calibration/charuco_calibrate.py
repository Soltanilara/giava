"""OpenCV ChArUco intrinsic calibration  (PHASE 5, solve half).

Processes a directory of images (from ``charuco_capture.py`` or anywhere
else), detects the board, runs ``cv2.calibrateCamera``, and writes a JSON
result that ``compare_intrinsics.py`` can put next to the RealSense factory
values.

Reported quality numbers
    rms                 what cv2.calibrateCamera returns: the root-mean-
                        square reprojection error over ALL points in ALL
                        views, in pixels.  It is an optimisation residual,
                        not an accuracy figure -- a low RMS on a poorly
                        conditioned set of views (all fronto-parallel, all
                        in the frame centre) is perfectly possible and
                        means very little.
    per-view error      lets you find and drop the bad views.
    coverage            fraction of the image where corners were seen.
                        Distortion is unconstrained everywhere else, so a
                        calibration with 30% coverage should not be trusted
                        at the edges regardless of its RMS.

Examples

    python calibration/charuco_calibrate.py \\
        --images calibration/data/charuco/right_wrist_20260818_120000/images \\
        --camera right_wrist

    # rational (8-coefficient) model, and drop views worse than 1 px
    python calibration/charuco_calibrate.py --images IMGDIR --camera right_wrist \\
        --model rational --reject-above 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ is a SIBLING of this package now, not the
## parent it used to be.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
for _p in (str(HERE), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import board as B  # noqa: E402
from common import (  # noqa: E402
    DIR_CAMERAS,
    ensure_dirs,
    provenance,
    save_json,
    timestamp,
)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

## cv2.calibrateCamera distortion models, by the name used on the CLI.
## OpenCV's coefficient vector is ordered (k1, k2, p1, p2, k3[, k4, k5, k6]).
MODELS = {
    ## 5 coefficients: the classic "plumb bob" / Brown-Conrady used by
    ## OpenCV, ROS and most of the ecosystem.
    "plumb_bob": 0,
    ## 8 coefficients: adds a rational denominator, better for wide FOV
    ## lenses like the D405's, but needs good corner coverage to constrain.
    "rational": None,   # filled in at run time from cv2.CALIB_RATIONAL_MODEL
    ## 4 coefficients: k3 forced to zero. More stable on small datasets.
    "no_k3": None,      # cv2.CALIB_FIX_K3
}


def gather_images(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out += sorted(f for f in path.iterdir()
                          if f.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file():
            out.append(path)
        else:
            raise FileNotFoundError(f"{p} is neither a file nor a directory")
    if not out:
        raise FileNotFoundError(f"no images found in {paths}")
    return out


def detect_all(files: List[Path], spec: "B.BoardSpec", min_corners: int,
               annotate_dir: Optional[Path]) -> Tuple[List[Dict[str, Any]],
                                                      Tuple[int, int]]:
    """Detect the board in every image; returns per-image records."""
    import cv2

    detector, board = B.make_detector(spec)
    records: List[Dict[str, Any]] = []
    size: Optional[Tuple[int, int]] = None

    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  {f.name:28s} UNREADABLE")
            continue
        h, w = img.shape[:2]
        if size is None:
            size = (w, h)
        elif (w, h) != size:
            ## Intrinsics are per resolution; mixing sizes silently
            ## produces a meaningless average.
            print(f"  {f.name:28s} SKIPPED - {w}x{h} != {size[0]}x{size[1]}")
            continue

        det = B.detect(img, detector, board, min_corners=min_corners)
        if annotate_dir is not None:
            annotate_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(annotate_dir / f.name), B.annotate(img, det))

        if not det.ok:
            print(f"  {f.name:28s} rejected: {det.reason}")
            continue

        records.append({
            "file": str(f),
            "name": f.name,
            "n_corners": det.n_corners,
            "n_markers": det.n_markers,
            "obj_points": det.obj_points,
            "img_points": det.img_points,
            "ids": det.charuco_ids,
        })
        print(f"  {f.name:28s} ok  {det.n_corners:3d} corners")

    if size is None:
        raise RuntimeError("no readable images")
    return records, size


def calibrate(records: List[Dict[str, Any]], size: Tuple[int, int],
              flags: int) -> Dict[str, Any]:
    import cv2

    obj = [r["obj_points"].reshape(-1, 1, 3).astype(np.float32)
           for r in records]
    img = [r["img_points"].reshape(-1, 1, 2).astype(np.float32)
           for r in records]

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, size, None, None, flags=flags)

    ## Per-view error: cv2's single RMS hides which views are bad.
    per_view = []
    for i, r in enumerate(records):
        proj, _ = cv2.projectPoints(obj[i], rvecs[i], tvecs[i], K, dist)
        err = np.linalg.norm(
            proj.reshape(-1, 2) - img[i].reshape(-1, 2), axis=1)
        per_view.append({
            "name": r["name"],
            "n_corners": int(r["n_corners"]),
            "rms_px": float(np.sqrt(np.mean(err ** 2))),
            "max_px": float(err.max()),
            ## Board distance, useful for spotting views taken far outside
            ## the working range.
            "distance_m": float(np.linalg.norm(tvecs[i])),
        })

    return {"rms": float(rms), "K": np.asarray(K), "dist": np.asarray(dist).ravel(),
            "per_view": per_view, "rvecs": rvecs, "tvecs": tvecs}


def coverage_fraction(records: List[Dict[str, Any]],
                      size: Tuple[int, int], cells: int = 8) -> float:
    grid = np.zeros((cells, cells), dtype=bool)
    w, h = size
    for r in records:
        p = r["img_points"]
        cx = np.clip((p[:, 0] / w * cells).astype(int), 0, cells - 1)
        cy = np.clip((p[:, 1] / h * cells).astype(int), 0, cells - 1)
        grid[cy, cx] = True
    return float(grid.sum()) / grid.size


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True,
                    help="image directories and/or individual files")
    ap.add_argument("--camera", default=None,
                    help="GIAVA camera name this calibration belongs to. "
                         "Strongly recommended -- it is how the result gets "
                         "filed and how compare_intrinsics.py finds it.")
    ap.add_argument("--serial", default=None,
                    help="camera serial (looked up from --camera if known)")
    ap.add_argument("--model", default="plumb_bob",
                    choices=("plumb_bob", "rational", "no_k3"),
                    help="distortion model (default: plumb_bob, 5 coeffs)")
    ap.add_argument("--min-corners", type=int, default=8,
                    help="reject views with fewer charuco corners")
    ap.add_argument("--reject-above", type=float, default=None,
                    help="drop views whose per-view RMS exceeds this many "
                         "pixels and re-run the calibration once")
    ap.add_argument("--annotate", default=None,
                    help="write annotated detection images to this directory")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--overwrite", action="store_true")
    B.add_board_args(ap)
    args = ap.parse_args()

    import cv2

    MODELS["rational"] = cv2.CALIB_RATIONAL_MODEL
    MODELS["no_k3"] = cv2.CALIB_FIX_K3

    spec = B.spec_from_args(args)
    ensure_dirs()

    print()
    print("#" * 74)
    print("#  OpenCV ChArUco intrinsic calibration  (PHASE 5)")
    print("#" * 74)
    print(f"\n  board: {spec.describe()}")
    print(f"  model: {args.model}\n")

    files = gather_images(args.images)
    print(f"  {len(files)} candidate image(s)\n")
    records, size = detect_all(files, spec, args.min_corners,
                               Path(args.annotate) if args.annotate else None)

    if len(records) < 4:
        raise SystemExit(
            f"\n  only {len(records)} usable view(s). cv2.calibrateCamera "
            f"needs several, and a trustworthy result needs 15-30 with "
            f"varied board poses.")

    flags = MODELS[args.model] or 0
    result = calibrate(records, size, flags)
    dropped: List[str] = []

    if args.reject_above is not None:
        bad = {v["name"] for v in result["per_view"]
               if v["rms_px"] > args.reject_above}
        if bad and len(records) - len(bad) >= 4:
            print(f"\n  dropping {len(bad)} view(s) above "
                  f"{args.reject_above} px and re-calibrating:")
            for n in sorted(bad):
                print(f"      {n}")
            dropped = sorted(bad)
            records = [r for r in records if r["name"] not in bad]
            result = calibrate(records, size, flags)
        elif bad:
            print(f"\n  {len(bad)} view(s) exceed {args.reject_above} px but "
                  f"dropping them would leave too few -- keeping all.")

    cov = coverage_fraction(records, size)
    _report(result, size, cov, spec, args.model)
    _save(result, records, size, cov, spec, args, dropped)


def _report(result, size, cov, spec, model) -> None:
    K, dist = result["K"], result["dist"]
    print()
    print("=" * 74)
    print("  RESULT")
    print("=" * 74)
    print(f"\n  image size        : {size[0]} x {size[1]}")
    print(f"  views used        : {len(result['per_view'])}")
    print(f"  distortion model  : {model} "
          f"(OpenCV order k1,k2,p1,p2,k3[,k4,k5,k6])")
    print(f"  RMS reprojection  : {result['rms']:.5f} px")
    print()
    print("  camera matrix K:")
    for row in K:
        print("      [" + "  ".join(f"{v:12.5f}" for v in row) + "]")
    print()
    print(f"      fx = {K[0, 0]:12.5f}     fy = {K[1, 1]:12.5f}")
    print(f"      cx = {K[0, 2]:12.5f}     cy = {K[1, 2]:12.5f}")
    print()
    print("  distortion coefficients:")
    names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
    for n, v in zip(names, dist):
        print(f"      {n} = {v:+.8f}")

    pv = result["per_view"]
    errs = np.array([v["rms_px"] for v in pv])
    print()
    print(f"  per-view reprojection error: min {errs.min():.4f}  "
          f"median {np.median(errs):.4f}  max {errs.max():.4f} px")
    worst = sorted(pv, key=lambda v: -v["rms_px"])[:5]
    print("  worst views:")
    for v in worst:
        print(f"      {v['name']:24s} {v['rms_px']:7.4f} px  "
              f"({v['n_corners']:3d} corners, {v['distance_m']:.3f} m)")

    dists = np.array([v["distance_m"] for v in pv])
    print()
    print(f"  board distance range: {dists.min():.3f} .. {dists.max():.3f} m")
    print(f"  corner coverage     : {cov * 100:.0f}% of the image")
    print()
    print("  " + "-" * 70)
    if cov < 0.6:
        print("  WARNING: coverage below 60%. Distortion is unconstrained")
        print("  where no corners were seen -- do not trust this calibration")
        print("  near the image edges.")
    if dists.max() - dists.min() < 0.10:
        print("  WARNING: all views at nearly the same distance. Focal length")
        print("  and board distance are correlated; a narrow range leaves")
        print("  them poorly separated.")
    if not spec.measured:
        print("  WARNING: the board geometry is not marked as measured.")
        print("  fx/fy/cx/cy and the distortion coefficients are unaffected,")
        print("  but every metric quantity derived from this camera is")
        print("  scaled by whatever square_length_m really is.")
    print("  RMS is an optimisation residual, not an accuracy figure. Compare")
    print("  against the factory calibration and against physical reality.")
    print("  " + "-" * 70)


def _save(result, records, size, cov, spec, args, dropped) -> None:
    from rs_camera import serial_for

    serial = args.serial
    if serial is None and args.camera:
        try:
            serial = serial_for(args.camera)
        except Exception:
            serial = None

    K = result["K"]
    dist = result["dist"]
    doc = {
        "metadata": provenance(
            "opencv_charuco",
            camera=args.camera,
            serial=serial,
            opencv_version=_cv_version(),
            distortion_model_flag=args.model,
            image_sources=[str(p) for p in args.images],
        ),
        "camera": args.camera,
        "serial": serial,
        "image_size": {"width": size[0], "height": size[1]},
        "board": spec.to_dict(),
        "n_images_used": len(records),
        "n_images_dropped": len(dropped),
        "dropped_images": dropped,
        "rms_reprojection_error_px": result["rms"],
        "coverage_fraction": cov,
        "camera_matrix": K.tolist(),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "distortion_model": ("opencv_rational" if args.model == "rational"
                             else "opencv_plumb_bob"),
        "distortion_coefficient_order": ["k1", "k2", "p1", "p2", "k3",
                                         "k4", "k5", "k6"][:len(dist)],
        "distortion_coefficients": dist.tolist(),
        "distortion_direction": (
            "OpenCV convention: these coefficients DISTORT an ideal "
            "normalised point when projecting 3D->2D (cv2.projectPoints). "
            "librealsense's inverse_brown_conrady runs the opposite way -- "
            "see compare_intrinsics.py."
        ),
        "per_view": result["per_view"],
    }

    if args.out:
        path = Path(args.out)
    else:
        tag = f"{args.camera}_{serial}" if serial else (args.camera or "camera")
        path = DIR_CAMERAS / tag / f"charuco_intrinsics_{timestamp()}.json"
    save_json(doc, path, overwrite=args.overwrite)
    print(f"\n  saved to {path}")
    if args.camera:
        print(f"\n  Compare against the factory calibration with:")
        print(f"      python calibration/compare_intrinsics.py "
              f"--camera {args.camera}")
    print()


def _cv_version() -> str:
    import cv2
    return cv2.__version__


if __name__ == "__main__":
    main()
