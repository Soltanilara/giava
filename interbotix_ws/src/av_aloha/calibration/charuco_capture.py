"""Capture ChArUco calibration images  (PHASE 5, collection half).

Streams one camera, detects the board live, and saves only the frames that
actually contain a usable view.  Saved images are RAW -- no undistortion,
no resizing -- because they are the calibration's input data.

It also enforces the two things that actually determine calibration
quality, which a human eye is bad at judging:

  * POSE DIVERSITY.  A pile of fronto-parallel views is nearly singular:
    focal length and board distance trade off against each other and the
    solution wanders.  The tilt of each accepted view is estimated and
    reported, and --min-tilt rejects views that are too flat.
  * IMAGE COVERAGE.  Distortion is only constrained where you put corners.
    A running coverage grid shows which parts of the frame are still empty.

Examples

    # generate a board to print or display, then measure it
    python calibration/charuco_capture.py --generate-board /tmp/board.png

    # interactive capture (SPACE saves, q quits) -- needs a display
    python calibration/charuco_capture.py --camera right_wrist

    # headless: auto-save a good view every 1.5 s until 40 are collected
    python calibration/charuco_capture.py --camera right_wrist \\
        --auto --target 40 --interval 1.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    DIR_CHARUCO,
    ensure_dirs,
    provenance,
    save_json,
    timestamp,
)
from rs_camera import PRODUCTION_COLOR, RealSenseCamera, require_rs  # noqa: E402


def estimate_tilt_deg(det: "B.Detection", image_shape) -> Optional[float]:
    """Angle between the board plane normal and the camera's optical axis.

    Computed with a pinhole guess for K (fx = fy = image width, principal
    point at the centre).  That is only good to a few degrees, which is all
    a diversity check needs -- it is never used as a calibration input."""
    import cv2

    h, w = image_shape[:2]
    K = np.array([[float(w), 0, w / 2.0],
                  [0, float(w), h / 2.0],
                  [0, 0, 1.0]])
    try:
        ok, rvec, _ = cv2.solvePnP(
            det.obj_points.reshape(-1, 1, 3), det.img_points.reshape(-1, 1, 2),
            K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        ## Board normal is its +z axis, in camera coordinates.
        normal = R[:, 2]
        cos = abs(float(normal[2]))
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    except Exception:
        return None


class Coverage:
    """Which cells of the image have ever held a detected corner."""

    def __init__(self, width: int, height: int, cells: int = 8):
        self.w, self.h, self.n = width, height, cells
        self.grid = np.zeros((cells, cells), dtype=int)

    def add(self, pts: np.ndarray) -> None:
        cx = np.clip((pts[:, 0] / self.w * self.n).astype(int), 0, self.n - 1)
        cy = np.clip((pts[:, 1] / self.h * self.n).astype(int), 0, self.n - 1)
        for x, y in zip(cx, cy):
            self.grid[y, x] += 1

    @property
    def fraction(self) -> float:
        return float((self.grid > 0).sum()) / self.grid.size

    def render(self) -> List[str]:
        out = []
        for row in self.grid:
            out.append("      " + " ".join("#" if v else "." for v in row))
        out.append(f"      coverage {self.fraction * 100:.0f}% of the frame")
        return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default=None,
                    help="camera name or serial (camera_manager.CAMERA_SERIALS)")
    ap.add_argument("--width", type=int, default=PRODUCTION_COLOR[0])
    ap.add_argument("--height", type=int, default=PRODUCTION_COLOR[1])
    ap.add_argument("--fps", type=int, default=PRODUCTION_COLOR[2])
    ap.add_argument("--outdir", default=None,
                    help="where to save (default: a timestamped session dir "
                         "under calibration/data/charuco/)")
    ap.add_argument("--target", type=int, default=30,
                    help="how many views to collect (default: 30)")
    ap.add_argument("--min-corners", type=int, default=8,
                    help="reject views with fewer charuco corners")
    ap.add_argument("--min-tilt", type=float, default=0.0,
                    help="reject views flatter than this many degrees; try "
                         "10-15 to force pose diversity")
    ap.add_argument("--auto", action="store_true",
                    help="save automatically instead of waiting for SPACE")
    ap.add_argument("--interval", type=float, default=1.5,
                    help="seconds between auto saves (default: 1.5)")
    ap.add_argument("--no-display", action="store_true",
                    help="never open a window (implied when --auto and no "
                         "display is available)")
    ap.add_argument("--generate-board", default=None,
                    help="write a board image to this path and exit")
    ap.add_argument("--board-dpi", type=float, default=None,
                    help="with --generate-board: render at this DPI instead "
                         "of the default 4000 px/m")
    ap.add_argument("--board-square-px", type=int, default=None,
                    help="with --generate-board: render exactly this many "
                         "pixels per square. The direct way to fit a "
                         "screen -- pick (screen height / squares_y) and "
                         "the board fills it at 100%% zoom with no "
                         "resampling.")
    ap.add_argument("--set-square-length", type=float, default=None,
                    metavar="MM",
                    help="record a MEASURED span across the displayed or "
                         "printed board, in millimetres, into "
                         "charuco_board.json. Use with --span-squares. "
                         "marker_length_m is rescaled by the same factor, "
                         "because the two are fixed by the rendering, not "
                         "independent.")
    ap.add_argument("--span-squares", type=int, default=None,
                    help="how many whole squares --set-square-length spans")
    B.add_board_args(ap)
    args = ap.parse_args()

    spec = B.spec_from_args(args)

    print()
    print("#" * 74)
    print("#  ChArUco capture  (PHASE 5)")
    print("#" * 74)
    print(f"\n  board: {spec.describe()}\n")

    if args.set_square_length is not None:
        _set_size(spec, args, ap)
        return

    if args.generate_board:
        _generate(spec, Path(args.generate_board), args.board_dpi,
                  args.board_square_px)
        return

    if not args.camera:
        ap.error("--camera is required (or use --generate-board)")

    require_rs()
    ensure_dirs()
    _capture(args, spec)


def _set_size(spec: "B.BoardSpec", args, ap) -> None:
    """Record a measured span into charuco_board.json.

    Why this is a command rather than a hand edit: `marker_length_m` is not
    an independent number.  The ArUco marker is drawn at a fixed fraction
    of the square by whatever rendered the board, so if the square turns
    out to be 4% larger than nominal the marker is too.  Editing only
    `square_length_m` -- the natural thing to do by hand -- leaves the
    marker/square ratio wrong, and ChArUco uses both.

    Every metric quantity downstream scales linearly with these, so this
    also prints the size change as a percentage: that is exactly the error
    that would otherwise appear in every distance the rig ever reports."""
    from common import load_json, save_json

    n = args.span_squares
    if not n or n < 1:
        ap.error("--set-square-length needs --span-squares N (how many "
                 "whole squares you measured across)")
    if n > spec.squares_x and n > spec.squares_y:
        ap.error(f"--span-squares {n} exceeds the board "
                 f"({spec.squares_x} x {spec.squares_y} squares)")

    new_square = (args.set_square_length * 1e-3) / n
    if not (0.001 < new_square < 0.5):
        ap.error(f"that works out to {new_square * 1e3:.3f} mm per square, "
                 f"which is not plausible -- is --set-square-length in "
                 f"MILLIMETRES and --span-squares a whole-square count?")

    ratio = new_square / spec.square_length_m
    new_marker = spec.marker_length_m * ratio

    path = Path(args.board_config)
    doc = load_json(path) if path.exists() else {}
    old_sq, old_mk = spec.square_length_m, spec.marker_length_m
    doc["square_length_m"] = float(new_square)
    doc["marker_length_m"] = float(new_marker)
    doc["measured"] = True
    doc["note"] = (
        f"Measured {args.set_square_length:.2f} mm across {n} squares on "
        f"{time.strftime('%Y-%m-%d')}. RECORD HOW IT WAS DISPLAYED -- "
        f"screen model and zoom level, or printer and paper. If the "
        f"display changes, this must be re-measured: every distance the "
        f"rig reports scales with it.")
    save_json(doc, path, overwrite=True)

    print(f"""
  square_length_m  {old_sq:.6f} -> {new_square:.6f}   ({new_square*1e3:.3f} mm)
  marker_length_m  {old_mk:.6f} -> {new_marker:.6f}   ({new_marker*1e3:.3f} mm)
                   marker rescaled by the same {ratio:.5f} factor, because
                   the renderer fixed their ratio -- it is not free

  SIZE CHANGE: {(ratio - 1) * 100:+.2f} %

  That is the factor by which every metric result from this board would
  have been wrong: board pose, camera extrinsics, and every distance the
  reconstruction stage reports. It is NOT visible in fx/fy/cx/cy or in any
  reprojection error -- those are in pixels and are unaffected.

  Written to {path}. Now edit its "note" to say WHICH SCREEN and what zoom.

  Any calibration collected against the old size must be re-solved:
      python calibration/scene_extrinsics.py solve --dir <session>
  (the stored corner pixels are still valid; only the board geometry changed)
""")


def _generate(spec: "B.BoardSpec", path: Path, dpi: Optional[float],
              square_px: Optional[int] = None) -> None:
    import cv2

    margin = 20
    size_px = None
    if square_px is not None:
        if square_px < 20:
            raise SystemExit("--board-square-px below ~20 leaves the ArUco "
                             "bits unresolvable; use 60 or more")
        ## OpenCV fits the board inside (size - 2*margin), so ask for a
        ## size that leaves exactly square_px per square after that.
        size_px = (spec.squares_x * square_px + 2 * margin,
                   spec.squares_y * square_px + 2 * margin)
        ppm = square_px / spec.square_length_m
    else:
        ppm = (dpi / 0.0254) if dpi else 4000.0
    img = B.generate_image(spec, pixels_per_metre=ppm, margin_px=margin,
                           size_px=size_px)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    w_mm = spec.squares_x * spec.square_length_m * 1e3
    h_mm = spec.squares_y * spec.square_length_m * 1e3
    ## OpenCV preserves the squares' aspect and enlarges whichever margin
    ## it has to, so the board does NOT fill the image and the two margins
    ## are usually unequal. Measuring the image edges would therefore be
    ## wrong; report the actual square size instead.
    sq_px = min((img.shape[1] - 2 * margin) / spec.squares_x,
                (img.shape[0] - 2 * margin) / spec.squares_y)
    if square_px is not None and abs(sq_px - square_px) > 0.51:
        raise SystemExit(
            f"--board-square-px asked for {square_px} but the render came "
            f"out at {sq_px:.1f} px/square. Refusing to write a file whose "
            f"size is not what was requested.")
    print(f"  board image written to {path}  ({img.shape[1]}x{img.shape[0]} px)")
    print(f"""
  The image encodes the board's PROPORTIONS, not its size.  At the nominal
  geometry in your config it would be {w_mm:.1f} x {h_mm:.1f} mm, but what
  matters is what it measures once printed or displayed.

  One square is {sq_px:.0f} px in this file; the board occupies
  {sq_px * spec.squares_x:.0f} x {sq_px * spec.squares_y:.0f} px of it and is
  CENTRED, with unequal margins -- so measure the squares, never the image
  edges.

  NEXT:
    1. Print it, or open it on the monitor at 100% zoom (no scaling, no
       'fit to window').
    2. Measure one square edge -- ideally span several squares and divide.
    3. Put the measured values into {B.BOARD_CONFIG}
       as square_length_m and marker_length_m, and set "measured": true.

  Until then every metric result from this board is scaled by an unknown
  factor.
""")


def _capture(args, spec: "B.BoardSpec") -> None:
    import cv2

    detector, board = B.make_detector(spec)
    outdir = (Path(args.outdir) if args.outdir
              else DIR_CHARUCO / f"{args.camera}_{timestamp()}")
    (outdir / "images").mkdir(parents=True, exist_ok=True)
    (outdir / "annotated").mkdir(parents=True, exist_ok=True)

    display = not args.no_display and _has_display()
    if not display and not args.auto:
        print("  No display available -- switching to --auto mode.")
        args.auto = True

    print(f"  saving to {outdir}")
    print(f"  target {args.target} views, min {args.min_corners} corners"
          + (f", min tilt {args.min_tilt} deg" if args.min_tilt else ""))
    print("  controls: SPACE = save this view,  q = finish\n"
          if display else "  auto-capture: press Ctrl-C to finish early\n")

    saved: List[Dict[str, Any]] = []
    coverage: Optional[Coverage] = None
    last_save = 0.0

    with RealSenseCamera(args.camera, width=args.width, height=args.height,
                         fps=args.fps, stream="color") as cam:
        print(f"  streaming {cam.name} (serial {cam.serial}) at "
              f"{args.width}x{args.height}\n")
        coverage = Coverage(args.width, args.height)
        try:
            while len(saved) < args.target:
                rec = cam.capture()
                ## The stream is rgb8; OpenCV wants BGR for both detection
                ## drawing and imwrite.
                bgr = cv2.cvtColor(rec.image, cv2.COLOR_RGB2BGR)
                det = B.detect(bgr, detector, board,
                               min_corners=args.min_corners)

                tilt = (estimate_tilt_deg(det, bgr.shape)
                        if det.ok else None)
                acceptable = det.ok and (
                    args.min_tilt <= 0 or (tilt is not None
                                           and tilt >= args.min_tilt))

                if display:
                    vis = B.annotate(bgr, det)
                    _hud(vis, len(saved), args.target, tilt, coverage,
                         acceptable)
                    cv2.imshow("charuco capture  [SPACE] save  [q] quit", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    want = (key == ord(" "))
                else:
                    want = False

                if args.auto and acceptable and \
                        (time.time() - last_save) >= args.interval:
                    want = True

                if want and acceptable:
                    entry = _save_view(outdir, len(saved), bgr, det, rec,
                                       tilt)
                    coverage.add(det.img_points)
                    saved.append(entry)
                    last_save = time.time()
                    tilt_s = "  tilt --" if tilt is None \
                        else f"  tilt {tilt:5.1f} deg"
                    print(f"  [{len(saved):3d}/{args.target}] "
                          f"{det.n_corners:3d} corners{tilt_s}"
                          f"  coverage {coverage.fraction * 100:3.0f}%")
                elif want:
                    print(f"  rejected: {det.reason or 'tilt too low'}")
        except KeyboardInterrupt:
            print("\n  stopped by user")
        finally:
            if display:
                cv2.destroyAllWindows()

    _finish(outdir, saved, coverage, spec, args)


def _has_display() -> bool:
    import os
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _hud(vis, n, target, tilt, coverage, acceptable) -> None:
    import cv2
    h = vis.shape[0]
    lines = [f"saved {n}/{target}",
             f"tilt {tilt:.1f} deg" if tilt is not None else "tilt --",
             f"coverage {coverage.fraction * 100:.0f}%"]
    for i, t in enumerate(lines):
        cv2.putText(vis, t, (8, h - 12 - 20 * (len(lines) - 1 - i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if acceptable else (0, 165, 255), 1)


def _save_view(outdir: Path, idx: int, bgr, det, rec, tilt) -> Dict[str, Any]:
    import cv2

    name = f"view_{idx:03d}.png"
    cv2.imwrite(str(outdir / "images" / name), bgr)
    cv2.imwrite(str(outdir / "annotated" / name), B.annotate(bgr, det))
    return {
        "index": idx,
        "image": f"images/{name}",
        "n_corners": det.n_corners,
        "n_markers": det.n_markers,
        "tilt_deg": tilt,
        "frame_number": rec.frame_number,
        "timestamp_ms": rec.timestamp_ms,
        "timestamp_domain": rec.timestamp_domain,
        "host_epoch_s": rec.host_epoch_s,
    }


def _finish(outdir: Path, saved: List[Dict[str, Any]],
            coverage: Optional[Coverage], spec, args) -> None:
    print()
    print("=" * 74)
    print(f"  collected {len(saved)} views in {outdir}")
    print("=" * 74)
    if coverage is not None and saved:
        print("\n  corner coverage of the image plane:")
        for line in coverage.render():
            print(line)
    if saved:
        tilts = [s["tilt_deg"] for s in saved if s["tilt_deg"] is not None]
        if tilts:
            print(f"\n  tilt spread: {min(tilts):.1f} .. {max(tilts):.1f} deg"
                  f"  (mean {np.mean(tilts):.1f})")
            if max(tilts) - min(tilts) < 20:
                print("  WARNING: all views have similar tilt. Calibration is")
                print("  poorly conditioned without varied board angles --")
                print("  focal length and distance trade off against each")
                print("  other. Re-capture with the board tilted more.")
        corners = [s["n_corners"] for s in saved]
        print(f"  corners per view: {min(corners)} .. {max(corners)} "
              f"(of {spec.n_corners} possible)")

    manifest = {
        "metadata": provenance("charuco_capture", camera=args.camera,
                               width=args.width, height=args.height,
                               fps=args.fps),
        "board": spec.to_dict(),
        "image_size": [args.width, args.height],
        "n_views": len(saved),
        "coverage_fraction": (coverage.fraction if coverage else None),
        "views": saved,
    }
    save_json(manifest, outdir / "capture_manifest.json", overwrite=True)
    print(f"\n  manifest: {outdir / 'capture_manifest.json'}")
    print(f"\n  Calibrate with:")
    print(f"      python calibration/charuco_calibrate.py "
          f"--images {outdir / 'images'} --camera {args.camera}")
    print()


if __name__ == "__main__":
    main()
