"""Stereo-calibrate the OAK pair from checkerboard images saved by teleop.py.

Reads the raw pairs `teleop.py`'s 's' key writes, runs the standard OpenCV
pipeline (per-eye intrinsics -> stereoCalibrate with those FIXED ->
stereoRectify -> remap tables) and writes a `stereo.npz` in exactly the layout
`camera_manager.load_oak_rectify_maps_from_file()` reads, so the result drops
into the headset stream with no glue.

    # 1. collect (in the headset, camera arm live, board on the table)
    python teleop.py --mode middle --capture-dir ../calibration/data/oak_calib/run1

    # 2. calibrate
    python oak_stereo_calibrate.py --dir ../calibration/data/oak_calib/run1 --square 0.025

    # 3. install it as the calibration the headset stream uses
    python oak_stereo_calibrate.py --dir ../calibration/data/oak_calib/run1 --square 0.025 --install

WHY MONO-THEN-STEREO, RATHER THAN ONE JOINT SOLVE
    Calibrating everything at once lets a badly-conditioned extrinsic pull the
    intrinsics with it -- and the intrinsics are the half that stays valid when
    the cameras are re-cased, so corrupting them costs more than it saves.
    Each eye is solved on its own images first, then `CALIB_FIX_INTRINSIC`
    leaves stereoCalibrate exactly one thing to find: the rigid transform
    between the two.

THE NUMBER THAT MATTERS IS NOT THE RMS
    Reprojection RMS says the model fits the corners.  What the operator feels
    is RESIDUAL VERTICAL DISPARITY: after rectification the same point must sit
    on the same image ROW in both eyes, because vertical disparity is the one
    stereo error human vision cannot fuse -- it reads as eye strain rather than
    as a picture fault.  So this reports the rectified y-error over every
    detected corner, and that is the number to judge the run by:

        < 0.3 px   excellent
        < 0.5 px   good, ship it
        < 1.0 px   usable, mildly tiring over a long session
        > 1.0 px   recapture -- more tilt, more coverage, more pairs

BOARD GEOMETRY
    --board is INNER corners, i.e. one less than the squares in each direction:
    a 10x7-square board is `--board 9x6`.  --square is the side of one square
    in METRES, measured on the printed board (not the nominal PDF size -- most
    printers scale).  Get either wrong and the intrinsics still come out fine
    while the BASELINE is wrong by the same factor, which is the error that
    silently breaks depth and the viewer's stereo placement.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def find_pairs(root: Path, left_dir: Path | None, right_dir: Path | None):
    """[(left_path, right_path)], from teleop.py's layout or two folders."""
    if left_dir is not None and right_dir is not None:
        lefts = {p.stem.replace("_left", ""): p
                 for p in sorted(left_dir.glob("*.png"))}
        rights = {p.stem.replace("_right", ""): p
                  for p in sorted(right_dir.glob("*.png"))}
        keys = sorted(set(lefts) & set(rights))
        return [(lefts[k], rights[k]) for k in keys]

    pairs = []
    for left in sorted(root.glob("*_left.png")):
        right = left.with_name(left.name.replace("_left.png", "_right.png"))
        if right.exists():
            pairs.append((left, right))
    return pairs


def object_points(cols: int, rows: int, square: float) -> np.ndarray:
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * square


def detect(path: Path, board, debug_dir: Path | None):
    img = cv2.imread(str(path))
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## No FAST_CHECK here, unlike the live feedback in teleop.py: this is the
    ## run whose answer gets used, and FAST_CHECK trades recall for latency
    ## that does not matter offline.
    found, corners = cv2.findChessboardCorners(
        gray, board,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None, img.shape[1::-1]
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    if debug_dir is not None:
        vis = img.copy()
        cv2.drawChessboardCorners(vis, board, corners, True)
        cv2.imwrite(str(debug_dir / f"{path.stem}_corners.png"), vis)
    return corners, img.shape[1::-1]


def coverage_report(points, size, grid=6):
    """Fraction of an NxN image grid that any detected corner ever landed in.

    Distortion is only constrained where corners were observed.  A calibration
    fitted entirely in the middle of the frame extrapolates at the edges, which
    is precisely where distortion is largest -- and it does so with a low RMS,
    because nothing there was ever measured."""
    w, h = size
    hit = np.zeros((grid, grid), bool)
    for pts in points:
        for x, y in pts.reshape(-1, 2):
            gx = min(grid - 1, max(0, int(x / w * grid)))
            gy = min(grid - 1, max(0, int(y / h * grid)))
            hit[gy, gx] = True
    return hit


def rectified_y_error(objpoints, left_pts, right_pts, K1, D1, K2, D2,
                      R1, R2, P1, P2):
    """Per-corner |y_left - y_right| after rectification, in pixels.

    Computed by pushing the DETECTED corners through the same undistort +
    rectify transform the remap tables apply, so it measures the calibration
    the headset will actually run rather than the fit's own residual."""
    errs = []
    for lp, rp in zip(left_pts, right_pts):
        ul = cv2.undistortPoints(lp, K1, D1, R=R1, P=P1).reshape(-1, 2)
        ur = cv2.undistortPoints(rp, K2, D2, R=R2, P=P2).reshape(-1, 2)
        errs.append(np.abs(ul[:, 1] - ur[:, 1]))
    return np.concatenate(errs) if errs else np.zeros(0)


def main():
    ap = argparse.ArgumentParser(
        description="Stereo-calibrate the OAK pair from checkerboard images.")
    ap.add_argument("--dir", type=Path,
                    help="directory of pair_NNNN_left.png / _right.png "
                         "(teleop.py's --capture-dir)")
    ap.add_argument("--left-dir", type=Path,
                    help="alternative layout: one folder per eye")
    ap.add_argument("--right-dir", type=Path)
    ap.add_argument("--board", default="9x6",
                    help="INNER corners, cols x rows (default 9x6)")
    ap.add_argument("--square", type=float, required=True,
                    help="checkerboard square side in METRES, as printed")
    ap.add_argument("--out", type=Path,
                    help="output npz (default <dir>/stereo.npz)")
    ap.add_argument("--min-pairs", type=int, default=12,
                    help="refuse to calibrate on fewer usable pairs")
    ap.add_argument("--model", choices=("standard", "rational"),
                    default="standard",
                    help="distortion model. 'standard' = k1 k2 p1 p2 k3 "
                         "(default). 'rational' adds k4 k5 k6, which fits "
                         "wide lenses better in principle but is easy to "
                         "over-parameterise: on this rig it made the recovered "
                         "field of view swing 8 deg as pairs were dropped, "
                         "while 'standard' held to within 1 deg.")
    ap.add_argument("--max-y-err", type=float, default=1.0,
                    help="drop pairs whose worst rectified vertical disparity "
                         "exceeds this many pixels, then refit (default 1.0)")
    ap.add_argument("--reject-passes", type=int, default=3,
                    help="how many rejection+refit rounds to run (0 disables)")
    ap.add_argument("--samples", type=int, default=4,
                    help="rectified sample images to write for eyeballing")
    ap.add_argument("--debug-corners", action="store_true",
                    help="also write per-image corner overlays")
    ap.add_argument("--swap", action="store_true",
                    help="treat *_right.png as the LEFT eye and vice versa. "
                         "For a capture shot with the eye labels reversed -- "
                         "which is a wiring fact nothing in the capture path "
                         "verifies, and which was wrong on this rig until "
                         "2026-08-27 (T[0] came out +62.63 mm where a "
                         "correctly ordered pair must be negative). Lets an "
                         "existing capture be re-fitted correctly instead of "
                         "recaptured.")
    ap.add_argument("--install", action="store_true",
                    help="copy the result over camera_manager's "
                         "OAK_CALIB_STEREO_NPZ so the headset stream uses it")
    args = ap.parse_args()

    if args.dir is None and (args.left_dir is None or args.right_dir is None):
        ap.error("give --dir, or both --left-dir and --right-dir")

    try:
        cols, rows = (int(v) for v in args.board.lower().split("x"))
    except Exception:
        ap.error(f"--board must look like 9x6, got '{args.board}'")

    root = args.dir or args.left_dir.parent
    pairs = find_pairs(root, args.left_dir, args.right_dir)
    if not pairs:
        raise SystemExit(f"no image pairs found under {root}")
    if args.swap:
        pairs = [(r, l) for l, r in pairs]
        print("--swap: the file named *_right.png is being used as the LEFT "
              "eye.")
    print(f"Found {len(pairs)} pairs in {root}")

    out_dir = (args.out.parent if args.out else root)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = None
    if args.debug_corners:
        debug_dir = out_dir / "corner_debug"
        debug_dir.mkdir(exist_ok=True)

    objp = object_points(cols, rows, args.square)
    objpoints, left_pts, right_pts, used = [], [], [], []
    image_size = None

    for lp, rp in pairs:
        lc, lsize = detect(lp, (cols, rows), debug_dir)
        rc, rsize = detect(rp, (cols, rows), debug_dir)
        if lsize is not None and rsize is not None and lsize != rsize:
            print(f"  skip {lp.name}: eyes are {lsize} vs {rsize}")
            continue
        if lc is None or rc is None:
            which = ("neither eye" if lc is None and rc is None
                     else "right eye" if lc is None else "left eye")
            print(f"  skip {lp.name}: board not found in {which}")
            continue
        if image_size is None:
            image_size = lsize
        elif lsize != image_size:
            ## Mixing resolutions silently averages two different cameras.
            print(f"  skip {lp.name}: {lsize} but the set is {image_size}")
            continue
        objpoints.append(objp)
        left_pts.append(lc)
        right_pts.append(rc)
        used.append((lp, rp))
        print(f"  ok   {lp.name}")

    n = len(objpoints)
    print(f"\n{n} of {len(pairs)} pairs usable at {image_size}")

    ## PER-PAIR HANDEDNESS, before any fitting.
    ##
    ## A left camera is displaced to the left, so everything it sees sits
    ## FURTHER RIGHT in its own image: mean_u(left) - mean_u(right) must be
    ## positive for every pair, with no calibration involved.  Uniformly
    ## negative means the whole capture is labelled backwards -- fixable with
    ## --swap.  MIXED means the directory contains pairs from two different
    ## labelling conventions, which --swap cannot fix because it is a
    ## whole-directory flag.
    ##
    ## Mixing is not hypothetical: 2026-08-27, three capture folders were
    ## merged by renaming, two of them shot after the eye labels were
    ## corrected at the source and one before.  The fit came out at stereo
    ## rms 29.9 px, a 44 deg field of view and T[0] = +234 mm -- numbers that
    ## say "something is very wrong" without saying what.  This says what.
    if n:
        disp = np.array([float(np.asarray(l).reshape(-1, 2)[:, 0].mean()
                               - np.asarray(r).reshape(-1, 2)[:, 0].mean())
                         for l, r in zip(left_pts, right_pts)])
        bad = np.nonzero(disp < 0)[0]
        if len(bad) == n:
            print(f"\n  *** HANDEDNESS: all {n} pairs have NEGATIVE board "
                  f"disparity (mean {disp.mean():+.1f} px).")
            print(f"  *** The files named *_left.png were taken by the "
                  f"physically-RIGHT camera. Re-run with --swap.")
        elif len(bad):
            good = n - len(bad)
            names = [used[i][0].name for i in bad]
            print(f"\n  *** MIXED HANDEDNESS: {len(bad)} pairs are labelled "
                  f"one way and {good} the other.")
            print(f"  *** --swap cannot fix this -- it flips the WHOLE "
                  f"directory. Rename only the minority back, then re-run.")
            print(f"  *** Reversed ({len(bad)}): "
                  + ", ".join(names[:6])
                  + (f", ... (+{len(names) - 6} more)" if len(names) > 6 else ""))
            print(f"  *** Fitting a mixed set is meaningless: half the pairs "
                  f"claim the cameras are on opposite sides of each other, so "
                  f"stereoCalibrate averages two incompatible geometries.")
            raise SystemExit("refusing to fit a mixed-handedness capture")
    if n < args.min_pairs:
        raise SystemExit(
            f"need at least {args.min_pairs} usable pairs, got {n}. Capture "
            f"more, and vary the board's TILT -- a stack of fronto-parallel "
            f"views is nearly singular (focal length and distance to the board "
            f"trade off against each other, so the solution wanders).")

    for name, pts in (("left", left_pts), ("right", right_pts)):
        hit = coverage_report(pts, image_size)
        print(f"\n{name} eye coverage ({int(hit.sum())}/{hit.size} cells):")
        for row in hit:
            print("   " + "".join("#" if c else "." for c in row))
    print("  empty cells = image regions where distortion is unconstrained "
          "and the model is extrapolating.")

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
    model_flag = (cv2.CALIB_RATIONAL_MODEL if args.model == "rational" else 0)

    def solve(objp_l, lpts, rpts):
        """One full mono->stereo->rectify pass over the given pairs."""
        r1, k1, d1, _, _ = cv2.calibrateCamera(
            objp_l, lpts, image_size, None, None, flags=model_flag, criteria=crit)
        r2, k2, d2, _, _ = cv2.calibrateCamera(
            objp_l, rpts, image_size, None, None, flags=model_flag, criteria=crit)
        rs, k1, d1, k2, d2, rr, tt, e, f = cv2.stereoCalibrate(
            objp_l, lpts, rpts, k1, d1, k2, d2, image_size,
            flags=cv2.CALIB_FIX_INTRINSIC | model_flag, criteria=crit)
        ## alpha=0 crops to the all-valid-pixels rectangle.  The alternative
        ## (alpha=1, keep everything) leaves black wedges at the frame edge,
        ## which in a headset is not cosmetic: the wedge differs between the
        ## eyes, so the views disagree at the periphery and fight fusion
        ## exactly where the operator is least able to ignore it.
        rect = cv2.stereoRectify(k1, d1, k2, d2, image_size, rr, tt,
                                 flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        r1r, r2r, p1, p2 = rect[:4]
        per = [rectified_y_error(None, [l], [r], k1, d1, k2, d2, r1r, r2r, p1, p2)
               for l, r in zip(lpts, rpts)]
        return dict(rms_l=r1, rms_r=r2, rms=rs, K1=k1, D1=d1, K2=k2, D2=d2,
                    R=rr, T=tt, E=e, F=f, R1=r1r, R2=r2r, P1=p1, P2=p2,
                    Q=rect[4], roi1=rect[5], roi2=rect[6], per_pair=per)

    def hfov_deg(P1):
        return 2.0 * np.degrees(np.arctan(image_size[0] * 0.5 / float(P1[0, 0])))

    ## ITERATIVE OUTLIER REJECTION.
    ##
    ## A single bad pair is not a small error that averages out.  A board
    ## detected one row off, or caught mid-motion with a rolling shutter, is a
    ## CONSISTENT set of 54 wrong correspondences, and least squares treats
    ## consistency as evidence.  On this rig's first real capture, 14 of 46
    ## pairs carried >1 px of rectified vertical disparity and dragged the
    ## right eye's reprojection RMS to 1.26 px; dropping them took BOTH eyes to
    ## 0.27 px and the worst corner from 29 px to 0.9 px.
    ##
    ## Rejection is by rectified y-error rather than by reprojection residual
    ## on purpose: y-error is the quantity the operator actually feels, and a
    ## pair can fit its own eye beautifully while being wrong about the pair.
    print(f"\nCalibrating ({args.model} distortion model)...")
    keep_names = [lp.name for lp, _ in used]
    res = solve(objpoints, left_pts, right_pts)
    history = []
    for it in range(args.reject_passes + 1):
        worst = np.array([float(p.max()) for p in res["per_pair"]])
        allerr = np.concatenate(res["per_pair"])
        history.append((len(worst), hfov_deg(res["P1"]),
                        abs(float(res["P2"][0, 3]) / float(res["P1"][0, 0])),
                        float(np.percentile(allerr, 95))))
        print(f"  pass {it}: {len(worst):3d} pairs  "
              f"rms L{res['rms_l']:.3f} R{res['rms_r']:.3f} stereo {res['rms']:.3f}  "
              f"hfov {hfov_deg(res['P1']):5.1f} deg  "
              f"y-err p95 {np.percentile(allerr, 95):.2f} max {worst.max():.1f} px")
        if it == args.reject_passes:
            break
        keep = worst <= args.max_y_err
        if keep.all():
            print(f"  every pair is within {args.max_y_err} px; nothing to reject")
            break
        if int(keep.sum()) < args.min_pairs:
            print(f"  rejecting at {args.max_y_err} px would leave "
                  f"{int(keep.sum())} pairs (< --min-pairs {args.min_pairs}); "
                  f"stopping here. The capture, not the threshold, is the "
                  f"problem -- see the board-flatness note below.")
            break
        dropped = [keep_names[i] for i in range(len(keep)) if not keep[i]]
        print(f"    dropping {len(dropped)} pair(s) over {args.max_y_err} px: "
              + ", ".join(d.replace("_left.png", "") for d in dropped))
        objpoints = [objpoints[i] for i in range(len(keep)) if keep[i]]
        left_pts = [left_pts[i] for i in range(len(keep)) if keep[i]]
        right_pts = [right_pts[i] for i in range(len(keep)) if keep[i]]
        used = [used[i] for i in range(len(keep)) if keep[i]]
        keep_names = [keep_names[i] for i in range(len(keep)) if keep[i]]
        res = solve(objpoints, left_pts, right_pts)
    n = len(objpoints)

    ## STABILITY IS THE REAL TEST OF CONDITIONING.
    ##
    ## If the focal length is determined by the data, dropping a few pairs
    ## barely moves it.  If it swings, the data does not determine it and the
    ## solver is choosing a focal length to cancel a distortion error -- the
    ## classic focal/distance degeneracy.  Measured on this rig: the standard
    ## model held 95.2 -> 94.7 -> 95.5 deg across rejection passes while the
    ## rational model swung 100.7 -> 96.8 -> 105.0 on the SAME images, which is
    ## why `standard` is the default.  A field of view that moves by degrees is
    ## a field of view the headset will place wrongly.
    if len(history) > 1:
        fovs = [h[1] for h in history]
        spread = max(fovs) - min(fovs)
        print(f"\n  field of view across passes: "
              + " -> ".join(f"{f:.1f}" for f in fovs)
              + f"  (spread {spread:.1f} deg)")
        if spread > 2.0:
            print("  UNSTABLE. The images do not determine the focal length, so "
                  "the viewer will place them at the wrong angular size however "
                  "good the RMS looks.")
            if args.model == "rational":
                print("  Try --model standard: fewer distortion coefficients "
                      "cannot absorb as much noise into the focal length.")
            print("  The usual physical cause is a board that is not FLAT "
                  "(taped to a box, bowed, on foam): calibration assumes an "
                  "exact plane, and a bow is absorbed as focal length plus "
                  "distortion, differently in each eye. Mount it on glass, "
                  "hardboard or a clipboard and recapture.")

    K1, D1, K2, D2 = res["K1"], res["D1"], res["K2"], res["D2"]
    R, T, E, F = res["R"], res["T"], res["E"], res["F"]
    R1, R2, P1, P2, Q = res["R1"], res["R2"], res["P1"], res["P2"], res["Q"]
    roi1, roi2 = res["roi1"], res["roi2"]
    rms1, rms2, rms = res["rms_l"], res["rms_r"], res["rms"]

    print(f"\nFinal fit on {n} pairs:")
    print(f"  left  rms {rms1:.3f} px   fx={K1[0, 0]:.1f} fy={K1[1, 1]:.1f} "
          f"cx={K1[0, 2]:.1f} cy={K1[1, 2]:.1f}")
    print(f"  right rms {rms2:.3f} px   fx={K2[0, 0]:.1f} fy={K2[1, 1]:.1f} "
          f"cx={K2[0, 2]:.1f} cy={K2[1, 2]:.1f}")
    baseline = float(np.linalg.norm(T))
    tilt_deg = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))

    ## HANDEDNESS GUARD.
    ##
    ## stereoCalibrate returns R, T with X_2 = R*X_1 + T.  If camera 1 really
    ## is the left one, camera 2's origin sits at +x in camera 1's frame, so
    ## T = [-baseline, 0, 0] and T[0] < 0.  A POSITIVE T[0] means the images
    ## fed in as "left" were taken by the physically-right camera.
    ##
    ## This is the only signal in the whole fit that can see a swap, and
    ## until 2026-08-27 every place that touched it took a magnitude:
    ## norm(T) here, abs(P2[0,3]/fx) in camera_manager twice.  Meanwhile the
    ## rectified-vertical-disparity verdict below is STRUCTURALLY blind to it
    ## -- a swap leaves the rows just as well aligned -- so the calibration
    ## printed EXCELLENT while the headset showed inverted depth for months.
    T0 = float(np.asarray(T).ravel()[0])
    if T0 >= 0.0:
        print(f"\n  *** HANDEDNESS: T[0] = {T0 * 1000:+.2f} mm, "
              f"but a correctly ordered (left, right) pair MUST be negative.")
        print(f"  *** The images given as the LEFT eye were taken by the "
              f"physically-RIGHT camera.")
        print(f"  *** Re-run with --swap (this capture is fine, its labels "
              f"are not), and fix the labelling at the source so new "
              f"captures are correct.")
        print(f"  *** Everything below is a valid rectification of a SWAPPED "
              f"pair: rows will align, depth will be inverted, and no "
              f"quality number here can tell.")
        if args.install:
            raise SystemExit("refusing --install with reversed handedness")
    print(f"  stereo rms {rms:.3f} px")
    print(f"  baseline   {baseline * 1000:.2f} mm   "
          f"(T = {np.round(T.flatten() * 1000, 2)} mm)")
    print(f"  inter-camera rotation {tilt_deg:.3f} deg")
    print(f"  rectified field of view {hfov_deg(P1):.1f} deg horizontal "
          f"-- THIS is what the viewer places each eye's image at")

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, image_size, cv2.CV_16SC2)

    rect_baseline = abs(float(P2[0, 3]) / float(P1[0, 0]))
    yerr = rectified_y_error(None, left_pts, right_pts,
                             K1, D1, K2, D2, R1, R2, P1, P2)
    y_mean, y_p95, y_max = (float(np.mean(yerr)), float(np.percentile(yerr, 95)),
                            float(np.max(yerr)))
    if y_p95 < 0.3:
        verdict = "EXCELLENT"
    elif y_p95 < 0.5:
        verdict = "GOOD -- ship it"
    elif y_p95 < 1.0:
        verdict = "USABLE -- mildly tiring over a long session"
    else:
        verdict = "TOO HIGH -- recapture with more tilt and coverage"

    print(f"\nRectified vertical disparity over {len(yerr)} corners:")
    print(f"  mean {y_mean:.3f} px   p95 {y_p95:.3f} px   max {y_max:.3f} px")
    print(f"  --> {verdict}")

    ## Compare against whatever the headset is running right now.  A new
    ## calibration that moves the field of view by several degrees changes how
    ## the viewer places both images, and "the images sit too far apart" is
    ## what that feels like -- so the change is worth seeing before --install
    ## rather than discovering it with the headset on.
    try:
        import camera_manager as _cm
        prev = np.load(_cm.OAK_CALIB_STEREO_NPZ)
        pfx = float(prev["P1"][0, 0])
        pfov = 2.0 * np.degrees(np.arctan(image_size[0] * 0.5 / pfx))
        print(f"\nAgainst the calibration currently installed:")
        print(f"  field of view {pfov:.1f} -> {hfov_deg(P1):.1f} deg "
              f"({hfov_deg(P1) - pfov:+.1f})")
        print(f"  baseline      {abs(float(prev['P2'][0, 3]) / pfx) * 1000:.2f} "
              f"-> {rect_baseline * 1000:.2f} mm")
    except Exception:
        pass

    for i in range(min(args.samples, len(used))):
        lp, rp = used[i]
        li = cv2.remap(cv2.imread(str(lp)), map1x, map1y, cv2.INTER_LINEAR)
        ri = cv2.remap(cv2.imread(str(rp)), map2x, map2y, cv2.INTER_LINEAR)
        ## Side by side with horizontal rules drawn across both: if the
        ## rectification is right, every board corner sits on the same rule in
        ## both halves.  One glance replaces reading the number above.
        side = np.hstack([li, ri])
        for y in range(0, side.shape[0], 40):
            cv2.line(side, (0, y), (side.shape[1], y), (0, 255, 0), 1)
        cv2.imwrite(str(out_dir / f"rectified_check_{i:02d}.png"), side)
    print(f"\nWrote {min(args.samples, len(used))} rectified_check_*.png "
          f"(green rules should cross the same corners in both halves)")

    npz_path = args.out or (out_dir / "stereo.npz")
    np.savez_compressed(
        npz_path,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        roi1=roi1, roi2=roi2,
        map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
        K1=K1, D1=D1, K2=K2, D2=D2,
        image_size=np.asarray(image_size, dtype=np.int32),
        rms=rms, num_pairs=n,
    )
    summary = {
        "source": str(root),
        "board_inner_corners": [cols, rows],
        "square_m": args.square,
        "pairs_found": len(pairs),
        "pairs_used": n,
        "image_size": list(image_size),
        "distortion_model": args.model,
        "max_y_err_px": args.max_y_err,
        "rms_left": float(rms1), "rms_right": float(rms2),
        "rms_stereo": float(rms),
        "hfov_deg": hfov_deg(P1),
        "fov_across_passes_deg": [h[1] for h in history],
        "baseline_m": baseline,
        "rectified_baseline_m": rect_baseline,
        "inter_camera_rotation_deg": tilt_deg,
        "rectified_y_error_px": {"mean": y_mean, "p95": y_p95, "max": y_max},
        "verdict": verdict,
    }
    (out_dir / "stereo_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {npz_path}")
    print(f"Wrote {out_dir / 'stereo_summary.json'}")

    ## The rig guard in camera_manager compares the calibration's baseline
    ## against a hand-measured constant, because a stereo npz records nothing
    ## identifying the physical arrangement it was shot on -- re-casing the
    ## cameras leaves a file that still loads, still matches on resolution, and
    ## is silently wrong.  A fresh calibration is exactly when that constant
    ## needs revisiting, so say so with the number in hand.
    try:
        import camera_manager as cm
        expected = cm.OAK_EXPECTED_BASELINE_M
        tol = cm.OAK_BASELINE_TOLERANCE_M
        target = Path(cm.OAK_CALIB_STEREO_NPZ)
    except Exception as exc:
        print(f"\n[install] could not read camera_manager's settings ({exc})")
        expected, tol, target = None, None, None

    if expected is not None:
        off = abs(rect_baseline - expected)
        print(f"\ncamera_manager expects a {expected * 1000:.1f} mm baseline "
              f"(tolerance {tol * 1000:.1f} mm); this calibration measures "
              f"{rect_baseline * 1000:.1f} mm.")
        if off > tol:
            print("  MISMATCH. Measure the lens centres with a ruler:")
            print(f"    - ruler says ~{rect_baseline * 1000:.0f} mm -> the rig "
                  f"changed; set OAK_EXPECTED_BASELINE_M = "
                  f"{rect_baseline:.4f} in camera_manager.py")
            print(f"    - ruler says ~{expected * 1000:.0f} mm -> this "
                  f"calibration is wrong; --square is the usual culprit "
                  f"(a wrongly stated square size scales the baseline by "
                  f"exactly that factor and leaves the RMS untouched)")
            print("  Until they agree, camera_manager REFUSES this file and "
                  "streams raw -- which is the intended behaviour.")
        else:
            print("  Within tolerance.")

    if args.install:
        if target is None:
            raise SystemExit("--install needs camera_manager to be importable")
        if target.exists():
            backup = target.with_suffix(".npz.bak")
            shutil.copy2(target, backup)
            print(f"\n[install] previous calibration backed up to {backup}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(npz_path, target)
        print(f"[install] {npz_path} -> {target}")
        print("[install] restart teleop.py / data_collection.py to pick it up.")
    else:
        print(f"\nNot installed. Re-run with --install to copy this over "
              f"{target} (the file camera_manager loads).")


if __name__ == "__main__":
    main()
