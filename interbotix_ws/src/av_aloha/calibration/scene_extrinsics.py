"""Extrinsic calibration of the STATIC scene cameras -- top_scene, low_scene.

    collect   capture ChArUco views both cameras can see
    solve     camera-to-camera geometry from those views    -> stereo_*.json
    anchor    tie the pair to the robot world frame         -> anchor_*.json
    write     compose the two into ee_camera_transforms.json
    status    what exists and what is missing

These two cameras are bolted to the rig, not to an arm, so what they need
is `T_base_camera` directly -- there is no `T_ee_camera` for them.  Before
this script, `camera_mount.CAMERA_MOUNTS` listed both with
`provenance: unknown` and `T_parent_optical: None`, and `T_world_camera()`
raised for them.  That was correct: nothing knew where they were.


WHY IT IS SPLIT INTO TWO STAGES, AND WHY THAT ORDER
===================================================
`solve` and `anchor` answer different questions and must not be merged.

`solve` uses only a board held in the air.  It learns where the two
cameras are RELATIVE TO EACH OTHER, and its metric scale comes from the
measured ChArUco square -- no robot, no URDF, no forward kinematics
anywhere in the chain.  After this stage alone you can already measure
lengths, sizes and separations, and check them against the rulers on the
table.  **Everything measured at this stage is independent evidence about
the robot**, and that is the entire point.

`anchor` then finds where that rig sits in the robot's world frame, by
comparing points the robot claims to be at with where the cameras see it.
The fit residual is not a nuisance -- it IS the measurement of how far
forward kinematics disagrees with reality.

Doing it the other way round -- calibrating the cameras against the robot
first -- would make the cameras inherit whatever the URDF gets wrong, and
they could then never detect it.  The measurement would be circular and
would look perfect.  It is worth being explicit about this because the
circular version is easier to run and gives lower residuals.


THE DISTORTION-DIRECTION TRAP, HANDLED
======================================
The D405 factory intrinsics are `inverse_brown_conrady`: the coefficients
run PIXEL -> RAY.  OpenCV's `solvePnP` and `stereoCalibrate` expect
coefficients running RAY -> PIXEL.  Handing the factory numbers straight
to OpenCV applies them backwards -- roughly a double distortion, a few
pixels at the image edge, and nothing anywhere would report an error.

So this script never passes distortion coefficients to OpenCV.  Detected
corners are converted to NORMALISED RAYS first, by
`reconstruction.PinholeCamera.unproject`, which knows which direction each
model runs.  OpenCV is then called with `K = I` and `dist = 0`, for which
both conventions agree trivially.  A consequence worth knowing: the RMS
figures below are in NORMALISED units scaled back to pixels by the mean
focal length, not raw OpenCV pixel residuals.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ is a SIBLING of this package now, not the
## parent it used to be.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
for _p in (str(HERE), str(SCRIPTS_DIR), str(HERE.parent / "reconstruction")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import board as B  # noqa: E402
from common import (DATA_ROOT, DIR_CAMERAS, ensure_dirs, format_T,  # noqa: E402
                    invert_T, latest_matching, load_json, make_T,
                    matrix_to_quat_wxyz, provenance, rotation_angle_deg,
                    save_json, timestamp)

DIR_EXTRINSICS = DATA_ROOT / "extrinsics"
DIR_SCENE = DATA_ROOT / "scene_views"
SCENE_CAMERAS = ("top_scene", "low_scene")
MOUNT_CONFIG = HERE / "ee_camera_transforms.json"


## ------------------------------------------------------------------ ##
## Rotation averaging
## ------------------------------------------------------------------ ##

def average_rotation(Rs: Sequence[np.ndarray]) -> np.ndarray:
    """Chordal L2 mean of rotations, via the SVD of their sum.

    The naive alternative -- averaging Euler angles or matrix entries --
    does not produce a rotation at all, and averaging quaternions is
    sign-ambiguous.  This projects the arithmetic mean back onto SO(3),
    which is the closed-form minimiser of summed squared Frobenius
    distance and needs no iteration or sign bookkeeping."""
    M = np.zeros((3, 3))
    for R in Rs:
        M += np.asarray(R, dtype=float)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def average_transform(Ts: Sequence[np.ndarray]) -> np.ndarray:
    Ts = [np.asarray(T, dtype=float) for T in Ts]
    return make_T(np.mean([T[:3, 3] for T in Ts], axis=0),
                  average_rotation([T[:3, :3] for T in Ts]))


def spread(Ts: Sequence[np.ndarray], mean: Optional[np.ndarray] = None
           ) -> Dict[str, Any]:
    """How much the per-view estimates disagree -- the real quality figure.

    A calibration's RMS tells you how well the optimiser fitted the data it
    was given.  This tells you whether independent views AGREE, which is a
    different and much harder thing to fake: a systematically wrong board
    size, a mis-detected corner set or a moved camera all show up here and
    in none of the residuals."""
    Ts = [np.asarray(T, dtype=float) for T in Ts]
    if not Ts:
        return {"n": 0}
    mean = average_transform(Ts) if mean is None else mean
    t_err = [float(np.linalg.norm(T[:3, 3] - mean[:3, 3])) for T in Ts]
    r_err = [rotation_angle_deg(mean[:3, :3].T @ T[:3, :3]) for T in Ts]
    return {
        "n": len(Ts),
        "translation_mm": {
            "mean": float(np.mean(t_err) * 1e3),
            "max": float(np.max(t_err) * 1e3),
            "std": float(np.std(t_err) * 1e3),
            "per_view": [v * 1e3 for v in t_err],
        },
        "rotation_deg": {
            "mean": float(np.mean(r_err)),
            "max": float(np.max(r_err)),
            "std": float(np.std(r_err)),
            "per_view": r_err,
        },
    }


## ------------------------------------------------------------------ ##
## Board pose from normalised rays  (never hands coefficients to OpenCV)
## ------------------------------------------------------------------ ##

def board_pose(det, cam) -> Optional[np.ndarray]:
    """T_camera_board from a ChArUco detection, distortion handled correctly.

    Corners are undistorted to normalised rays by `cam` -- which knows
    whether its coefficients run pixel->ray or ray->pixel -- and PnP then
    runs on an ideal K=I camera, where the question does not arise."""
    import cv2

    rays = cam.unproject(det.img_points.reshape(-1, 2))
    ok, rvec, tvec = cv2.solvePnP(
        det.obj_points.reshape(-1, 1, 3).astype(np.float64),
        rays.reshape(-1, 1, 2).astype(np.float64),
        np.eye(3), np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return make_T(tvec.ravel(), R)


def reprojection_px(det, cam, T_cam_board: np.ndarray) -> float:
    """RMS reprojection error in pixels, through the real distortion model."""
    p_cam = (det.obj_points.reshape(-1, 3) @ T_cam_board[:3, :3].T
             + T_cam_board[:3, 3])
    uv = cam.project(p_cam)
    d = uv - det.img_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


## ------------------------------------------------------------------ ##
## collect
## ------------------------------------------------------------------ ##

def _has_display() -> bool:
    import os
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _pose_change(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """(millimetres, degrees) between two board poses."""
    return (float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1e3),
            rotation_angle_deg(a[:3, :3].T @ b[:3, :3]))


def cmd_collect(args) -> None:
    from rs_camera import resolve as rs_resolve
    from sync_capture import CameraWorker

    import cv2

    from camera import load_intrinsics

    B.require_cv2()

    ## cv2.imshow without a display does not raise -- Qt calls abort() and
    ## the process core-dumps, taking the open camera pipelines with it.
    ## Nothing downstream can catch that, so it has to be refused here.
    show = not args.no_display and _has_display()
    if not show and not args.auto:
        raise SystemExit(
            "no display available (DISPLAY and WAYLAND_DISPLAY are both "
            "unset), and collecting by hand needs the preview window.\n"
            "  Over SSH:      ssh -X, or run from the desktop session\n"
            "  On this box:   DISPLAY=:1 python calibration/"
            "scene_extrinsics.py collect ...\n"
            "  Headless:      add --auto (saves whenever both cameras see "
            "the board and it has moved enough)")
    spec = B.spec_from_args(args)
    print(f"\n  board: {spec.describe()}")
    if not spec.measured:
        print("\n  *** the board is not marked as measured -- every distance "
              "this rig\n      ever reports will be scaled by the error "
              "(charuco_board.json) ***")

    cams = list(args.cameras)
    pair = rs_resolve(cams)
    intr = {n: load_intrinsics(n, prefer=args.prefer_intrinsics) for n in cams}
    for c in intr.values():
        print("\n" + c.describe())

    detector, cvboard = B.make_detector(spec)
    outdir = Path(args.outdir) if args.outdir else (
        DIR_SCENE / f"{'_'.join(cams)}_{timestamp()}")
    (outdir / "images").mkdir(parents=True, exist_ok=True)

    workers = {}
    for name, serial in pair:
        w = CameraWorker(name, args.width, args.height, args.fps,
                         global_time=True, keep_images=True)
        w.start()
        workers[name] = w
        print(f"  streaming {name} ({serial})")

    print(f"""
  Hold the board so BOTH cameras see it, and vary the pose a lot.

    * Tilt it. A pile of fronto-parallel views is nearly singular -- the
      board's distance and the camera's focal length trade off against
      each other and neither is pinned down.
    * Move it through the whole shared volume, not one spot. What is
      being solved is a rigid transform between two cameras; views
      clustered in one place constrain it only there.
    * The top camera looks down and the low one looks across, so the
      board wants to sit at roughly 45 degrees to satisfy both at once.
      A view that is good for one and edge-on for the other is worth
      little, and this refuses those below --min-tilt.

  {'SPACE saves a view, q finishes.' if show else
     f'AUTO mode: saving every {args.interval:.1f} s once the board has moved.'}
  Target {args.target} views.

  A view is only kept if the board has moved at least {args.min_move:.0f} mm or
  turned {args.min_rot:.0f} deg since the last one. Twenty-five views of the same
  spot constrain the transform only at that spot, and would otherwise
  look like twenty-five times the evidence.
""")

    views: List[Dict[str, Any]] = []
    saved = 0
    last_pose: Optional[np.ndarray] = None
    last_save = 0.0
    last_beat = 0.0
    try:
        while True:
            frames, dets, tiles = {}, {}, []
            ok_all = True
            for name in cams:
                rec = workers[name].snapshot()
                if rec is None or rec.image is None:
                    ok_all = False
                    break
                frames[name] = rec
                det = B.detect(rec.image, detector, cvboard,
                               min_corners=args.min_corners)
                dets[name] = det
                vis = B.annotate(rec.image.copy(), det) if det.ok \
                    else rec.image.copy()
                tilt = ""
                if det.ok:
                    T = board_pose(det, intr[name])
                    if T is not None:
                        tilt = f"  tilt {_tilt_deg(T):.0f}d"
                cv2.putText(vis, f"{name}: {det.n_corners} corners{tilt}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if det.ok else (0, 0, 255), 2)
                tiles.append(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            if not ok_all:
                continue

            both = all(dets[n].ok for n in cams)
            montage = np.hstack(tiles)
            cv2.putText(montage,
                        f"saved {saved}/{args.target}   "
                        + ("BOTH OK - press SPACE" if both
                           else "need the board in BOTH"),
                        (8, montage.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0) if both else (0, 165, 255), 2)
            if show:
                cv2.imshow("scene extrinsics -- SPACE saves, q quits", montage)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                want = (key == 32)
            else:
                now = time.time()
                want = (now - last_save) >= args.interval
                ## Heartbeat: with no preview the terminal IS the feedback
                ## channel. Without it you cannot tell "not detected" from
                ## "not moved enough" from "tilt rejected", and you end up
                ## holding a laptop wondering why the count is not rising.
                if now - last_beat >= args.status_interval:
                    last_beat = now
                    bits = []
                    for n in cams:
                        d0 = dets[n]
                        t = ""
                        if d0.ok:
                            Tb = board_pose(d0, intr[n])
                            if Tb is not None:
                                t = (f" {np.linalg.norm(Tb[:3, 3]) * 1e3:.0f}mm"
                                     f" tilt {_tilt_deg(Tb):.0f}d")
                        bits.append(f"{n} {d0.n_corners:2d}c{t}" if d0.ok
                                    else f"{n} -- ({d0.reason[:24]})")
                    wait = max(0.0, args.interval - (now - last_save))
                    if both:
                        tail = "BOTH OK" + (f", save in {wait:.1f}s"
                                            if wait > 0.05 else ", SAVING")
                    else:
                        tail = "need the board in BOTH"
                    print(f"  [{saved}/{args.target}] "
                          + "  |  ".join(bits) + f"  ->  {tail}")
                time.sleep(0.02)
            if not want or not both:
                continue

            ## Pose diversity, enforced rather than hoped for.
            if last_pose is not None:
                T_now = board_pose(dets[cams[0]], intr[cams[0]])
                if T_now is not None:
                    mm, deg = _pose_change(T_now, last_pose)
                    if mm < args.min_move and deg < args.min_rot:
                        ## Printed in BOTH modes: gating this on `show` was
                        ## backwards, since headless is when it matters.
                        print(f"  skipped: board moved only {mm:.0f} mm / "
                              f"{deg:.0f} deg since the last view "
                              f"(need {args.min_move:.0f} mm or "
                              f"{args.min_rot:.0f} deg)")
                        if not show:
                            last_save = time.time()
                        continue

            entry, reject = _make_view(cams, frames, dets, intr, saved,
                                       outdir, args)
            if reject:
                print(f"  rejected: {reject}")
                continue
            views.append(entry)
            saved += 1
            last_pose = np.asarray(
                entry["cameras"][cams[0]]["T_cam_board_row_major"])
            last_save = time.time()
            print(f"  [{saved}] " + "  ".join(
                f"{n}: {dets[n].n_corners} corners, "
                f"{entry['cameras'][n]['distance_m']*1e3:.0f} mm, "
                f"tilt {entry['cameras'][n]['tilt_deg']:.0f}d"
                for n in cams))
            if saved >= args.target:
                print("\n  target reached (q to stop, or keep going -- more "
                      "is better)")
    finally:
        cv2.destroyAllWindows()
        for w in workers.values():
            w.stop()

    if not views:
        print("\n  no views saved; nothing written")
        return

    doc = {
        "metadata": provenance("scene_extrinsics_collect", cameras=cams,
                               n_views=len(views),
                               resolution=[args.width, args.height]),
        "cameras": cams,
        "board": spec.to_dict(),
        "intrinsics_source": {n: c.source for n, c in intr.items()},
        "min_corners": args.min_corners,
        "min_tilt_deg": args.min_tilt,
        "views": views,
    }
    path = save_json(doc, outdir / "views.json", overwrite=args.overwrite)
    print(f"\n  {len(views)} views -> {path}")
    print(f"\n  Next:\n      python calibration/scene_extrinsics.py solve "
          f"--dir {outdir}")


def _tilt_deg(T_cam_board: np.ndarray) -> float:
    """Angle between the board normal and the camera's viewing direction.

    0 means fronto-parallel, which is the degenerate case for calibration;
    larger is better up to the point where the corners stop resolving."""
    n = T_cam_board[:3, 2]
    view = T_cam_board[:3, 3]
    view = view / max(np.linalg.norm(view), 1e-9)
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(n, view)), 0, 1))))


def _make_view(cams, frames, dets, intr, index, outdir, args
               ) -> Tuple[Dict[str, Any], Optional[str]]:
    import cv2

    entry: Dict[str, Any] = {"index": index, "cameras": {}}
    for name in cams:
        det = dets[name]
        T = board_pose(det, intr[name])
        if T is None:
            return entry, f"{name}: PnP failed"
        tilt = _tilt_deg(T)
        if tilt < args.min_tilt:
            return entry, (f"{name}: tilt {tilt:.1f} deg < --min-tilt "
                           f"{args.min_tilt} (too fronto-parallel to be "
                           f"worth much)")
        rel = f"images/view{index:03d}_{name}.png"
        cv2.imwrite(str(outdir / rel),
                    cv2.cvtColor(frames[name].image, cv2.COLOR_RGB2BGR))
        entry["cameras"][name] = {
            "image": rel,
            "n_corners": int(det.n_corners),
            "corner_ids": det.charuco_ids.ravel().astype(int).tolist(),
            "img_points": det.img_points.reshape(-1, 2).tolist(),
            "obj_points": det.obj_points.reshape(-1, 3).tolist(),
            "T_cam_board_row_major": T.tolist(),
            "distance_m": float(np.linalg.norm(T[:3, 3])),
            "tilt_deg": tilt,
            "reprojection_rms_px": reprojection_px(det, intr[name], T),
            "timing": frames[name].timing_dict(),
        }
    ## Both cameras free-run; the board must not have moved between the two
    ## exposures. Record the gap so a shaky view can be identified later.
    ts = [entry["cameras"][n]["timing"].get("timestamp_ms") for n in cams]
    if all(t is not None for t in ts):
        entry["capture_spread_ms"] = float(max(ts) - min(ts))
    return entry, None


## ------------------------------------------------------------------ ##
## solve
## ------------------------------------------------------------------ ##

def cmd_solve(args) -> None:
    from camera import load_intrinsics

    session = Path(args.dir)
    doc = load_json(session / "views.json" if session.is_dir() else session)
    cams = list(doc["cameras"])
    if args.reference and args.reference not in cams:
        raise SystemExit(f"--reference must be one of {cams}")
    ref = args.reference or cams[0]
    others = [c for c in cams if c != ref]
    views = doc["views"]

    print(f"\n  {len(views)} views, cameras {cams}, reference '{ref}'")
    intr = {n: load_intrinsics(n, prefer=args.prefer_intrinsics) for n in cams}

    result: Dict[str, Any] = {}
    for other in others:
        per_view, kept = [], []
        for v in views:
            a = np.asarray(v["cameras"][ref]["T_cam_board_row_major"])
            b = np.asarray(v["cameras"][other]["T_cam_board_row_major"])
            ## T_ref_other = T_ref_board @ T_board_other
            per_view.append(a @ invert_T(b))
            kept.append(v)

        if len(per_view) < args.min_views:
            raise SystemExit(
                f"only {len(per_view)} usable views for {ref}<->{other}; "
                f"--min-views is {args.min_views}. More views, spread over "
                f"more of the shared volume, is the fix.")

        mean = average_transform(per_view)
        sp = spread(per_view, mean)
        print(f"\n  {ref} <-> {other}   from {len(per_view)} views")
        print(f"    per-view agreement: "
              f"{sp['translation_mm']['mean']:.2f} mm mean, "
              f"{sp['translation_mm']['max']:.2f} mm worst;  "
              f"{sp['rotation_deg']['mean']:.3f} deg mean, "
              f"{sp['rotation_deg']['max']:.3f} deg worst")

        refined, refine_info = (mean, {"method": "chordal_mean_only"})
        if not args.no_refine:
            try:
                refined, refine_info = _stereo_refine(
                    kept, ref, other, intr, mean)
                dt = np.linalg.norm(refined[:3, 3] - mean[:3, 3]) * 1e3
                dr = rotation_angle_deg(mean[:3, :3].T @ refined[:3, :3])
                print(f"    bundle refine moved it {dt:.2f} mm / {dr:.3f} deg"
                      f"   rms {refine_info['rms_px']:.3f} px")
                ## A refinement that moves the answer a long way means the
                ## per-view estimates were not really consistent.
                if dt > 10.0:
                    print("    ! that is a large correction -- the per-view "
                          "estimates disagreed; treat this calibration as "
                          "provisional and collect more varied views")
            except Exception as e:  # noqa: BLE001
                print(f"    refinement skipped: {type(e).__name__}: {e}")
                refine_info = {"method": "chordal_mean_only",
                               "refine_error": str(e)}

        result[other] = {
            "T_ref_cam_row_major": refined.tolist(),
            "chordal_mean_row_major": mean.tolist(),
            "per_view_agreement": sp,
            "refinement": refine_info,
            "baseline_mm": float(np.linalg.norm(refined[:3, 3]) * 1e3),
            "relative_orientation_deg": rotation_angle_deg(refined[:3, :3]),
        }
        print(f"    baseline {result[other]['baseline_mm']:.1f} mm, "
              f"relative rotation "
              f"{result[other]['relative_orientation_deg']:.2f} deg")
        print(format_T(refined, indent="      "))

    ## Board-frame distances measured by the finished rig, versus the board
    ## geometry that produced it. Not independent -- but it does catch a
    ## rig that solved to something self-inconsistent.
    check = _closure_check(views, cams, ref, result, intr)
    print(f"\n  closure check (board corners re-triangulated by the solved "
          f"rig vs their known board positions):")
    print(f"    {check['rms_mm']:.2f} mm rms over {check['n_points']} "
          f"corners in {check['n_views']} views, worst "
          f"{check['max_mm']:.2f} mm")
    if check["rms_mm"] > 3.0:
        print("    ! above 3 mm -- something is off. Most likely causes, in "
              "order:\n"
              "      the board square length in charuco_board.json is wrong;\n"
              "      a camera moved during collection;\n"
              "      too few / too similar views.")

    ensure_dirs()
    DIR_EXTRINSICS.mkdir(parents=True, exist_ok=True)
    entries = {ref: {"T_ref_cam_row_major": np.eye(4).tolist(),
                     "provenance": "reference camera (identity by definition)"}}
    for other, r in result.items():
        entries[other] = {"T_ref_cam_row_major": r["T_ref_cam_row_major"],
                          "provenance": f"scene_extrinsics:{Path(args.dir).name}"}

    doc_out = {
        "metadata": provenance("scene_extrinsics_solve",
                               session=str(args.dir), cameras=cams,
                               reference_camera=ref, n_views=len(views)),
        "convention": (
            "T_ref_cam maps points from the CAMERA optical frame into the "
            "reference camera's optical frame, and IS that camera's pose "
            "expressed in the reference frame. OpenCV optical convention: "
            "+x image-right, +y image-down, +z along the viewing axis."),
        "reference_camera": ref,
        "ref_frame": ref,
        "cameras": entries,
        "detail": result,
        "closure_check": check,
        "board": doc.get("board"),
        "intrinsics_source": {n: c.source for n, c in intr.items()},
        "anchored_to_robot": False,
        "note": ("Camera-to-camera only. Lengths and sizes measured with "
                 "this rig are already correct; they are expressed in the "
                 "reference camera's frame, not the robot's. Run `anchor` "
                 "to place it in the robot world frame."),
    }
    out = Path(args.out) if args.out else (
        DIR_EXTRINSICS / f"stereo_{timestamp()}.json")
    save_json(doc_out, out, overwrite=args.overwrite)
    print(f"\n  saved -> {out}")
    print(f"""
  You can measure with this right now, in '{ref}' coordinates:

      python reconstruction/measure.py --frame stereo --list

  To get robot-world coordinates as well:

      python calibration/scene_extrinsics.py anchor --help
""")


def _stereo_refine(views, ref, other, intr, T_init) -> Tuple[np.ndarray, Dict]:
    """Joint refinement over all views, on normalised rays.

    `cv2.stereoCalibrate` with CALIB_FIX_INTRINSIC, fed undistorted
    normalised coordinates and an identity K.  Feeding it the raw pixels
    and the factory coefficients would apply an inverse_brown_conrady
    model as though it were brown_conrady -- see this module's docstring."""
    import cv2

    obj, pa, pb = [], [], []
    for v in views:
        ca, cb = v["cameras"][ref], v["cameras"][other]
        ida = {int(i): k for k, i in enumerate(ca["corner_ids"])}
        idb = {int(i): k for k, i in enumerate(cb["corner_ids"])}
        shared = sorted(set(ida) & set(idb))
        if len(shared) < 6:
            continue
        o = np.asarray(ca["obj_points"], dtype=np.float32)
        obj.append(np.array([o[ida[i]] for i in shared], dtype=np.float32))
        ra = intr[ref].unproject(np.asarray(ca["img_points"], dtype=float))
        rb = intr[other].unproject(np.asarray(cb["img_points"], dtype=float))
        pa.append(np.array([ra[ida[i]] for i in shared], dtype=np.float32))
        pb.append(np.array([rb[idb[i]] for i in shared], dtype=np.float32))

    if len(obj) < 3:
        raise ValueError(
            f"only {len(obj)} views share >=6 corners between the two "
            f"cameras -- not enough to refine")

    I = np.eye(3)
    z = np.zeros(5)
    ## R, t map REF into OTHER (p_other = R p_ref + t), i.e. T_other_ref.
    rms, _, _, _, _, R, t, _, _ = cv2.stereoCalibrate(
        obj, pa, pb, I.copy(), z.copy(), I.copy(), z.copy(), (1, 1),
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                  200, 1e-12))
    T_other_ref = make_T(np.asarray(t).ravel(), np.asarray(R))
    f = (intr[ref].fx + intr[ref].fy + intr[other].fx + intr[other].fy) / 4.0
    return invert_T(T_other_ref), {
        "method": "cv2.stereoCalibrate on normalised rays, CALIB_FIX_INTRINSIC",
        "n_views_used": len(obj),
        "n_points": int(sum(len(o) for o in obj)),
        "rms_normalised": float(rms),
        "rms_px": float(rms * f),
        "rms_px_note": ("normalised residual scaled by the mean focal "
                        "length -- comparable to, but not identical with, "
                        "an OpenCV pixel RMS"),
    }


def _closure_check(views, cams, ref, result, intr) -> Dict[str, Any]:
    """Re-triangulate the board's own corners and compare with the board.

    Every corner's position in the board frame is known exactly from the
    board geometry.  Triangulating them with the freshly solved rig and
    comparing distances between corner PAIRS tests the rig's metric scale
    against the printed square -- which is the one thing a self-consistent
    but wrongly-scaled calibration cannot fake."""
    from camera import PinholeCamera
    from stereo import StereoRig

    posed = []
    for n in cams:
        c = intr[n]
        T = (np.eye(4) if n == ref
             else np.asarray(result[n]["T_ref_cam_row_major"], dtype=float))
        posed.append(PinholeCamera(
            name=n, width=c.width, height=c.height, fx=c.fx, fy=c.fy,
            cx=c.cx, cy=c.cy, model=c.model, coeffs=c.coeffs,
            T_ref_cam=T, ref_frame=ref))
    rig = StereoRig(posed, ref_frame=ref, source="closure check")

    errors, n_pts, n_views = [], 0, 0
    for v in views:
        entries = {n: v["cameras"][n] for n in cams if n in v["cameras"]}
        if len(entries) < 2:
            continue
        idx = {n: {int(i): k for k, i in enumerate(e["corner_ids"])}
               for n, e in entries.items()}
        shared = sorted(set.intersection(*(set(m) for m in idx.values())))
        if len(shared) < 4:
            continue
        pts3d, truth = [], []
        for cid in shared:
            obs = {n: np.asarray(entries[n]["img_points"])[idx[n][cid]]
                   for n in cams}
            try:
                pts3d.append(rig.triangulate(obs, sigma_px=0).point)
            except ValueError:
                continue
            truth.append(np.asarray(entries[ref]["obj_points"])[idx[ref][cid]])
        if len(pts3d) < 4:
            continue
        n_views += 1
        n_pts += len(pts3d)
        ## Compare pairwise DISTANCES: invariant to where the board was,
        ## so this tests scale and shape, not pose.
        P = np.stack(pts3d)
        Q = np.stack(truth)
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                errors.append(float(np.linalg.norm(P[i] - P[j])
                                    - np.linalg.norm(Q[i] - Q[j])))
    if not errors:
        return {"n_points": 0, "n_views": 0, "rms_mm": float("nan"),
                "max_mm": float("nan"),
                "note": "no view had enough shared corners"}
    e = np.abs(np.asarray(errors))
    return {
        "n_points": n_pts, "n_views": n_views,
        "n_pairs": len(errors),
        "rms_mm": float(np.sqrt(np.mean(e ** 2)) * 1e3),
        "max_mm": float(np.max(e) * 1e3),
        "bias_mm": float(np.mean(np.asarray(errors)) * 1e3),
        "note": ("distances between triangulated board corners minus the "
                 "same distances from the board geometry; a nonzero BIAS "
                 "is a scale error, scatter is noise"),
    }



## ------------------------------------------------------------------ ##
## anchor -- placing the solved rig in the robot world frame
## ------------------------------------------------------------------ ##
##
## The input is a set of CORRESPONDENCES: for each one, where the robot
## says a point is (forward kinematics from the measured joint angles) and
## where the cameras see it (triangulated).  `world_view.py`'s Measure tab
## records these; the format is documented in the README.
##
## The fit is deliberately run TWICE:
##
##   rigid       6 dof.  The residual is the honest disagreement between
##               forward kinematics and the cameras.  It cannot be reduced
##               by the fit, which is exactly why it is the measurement.
##
##   similarity  7 dof, adding a scale factor.  If a scale materially
##               different from 1 fits better, something is wrong with a
##               LENGTH: the ChArUco square measurement (which sets the
##               cameras' scale) or the URDF link lengths (which set the
##               robot's).  Reporting only the rigid fit would smear that
##               into the residual and hide it.
##
## The residual is then split into a CONSTANT OFFSET and SCATTER, because
## they have different causes and different fixes.  A constant offset that
## survives the fit means a frame is defined in the wrong place -- which is
## precisely the shape of a "the gripper is 10 cm from where the software
## thinks" complaint.  Scatter means joint calibration, backlash,
## compliance or click noise.


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False
            ) -> Tuple[np.ndarray, float]:
    """Least-squares similarity transform mapping src onto dst.

    Umeyama (1991).  Returns (T, scale) with T such that
    ``dst ~= scale * R @ src + t``; scale is exactly 1.0 for the rigid
    case.  The reflection guard matters: without it a noisy set of nearly
    coplanar points can fit a mirrored 'rotation', which has a lower
    residual and is physically impossible."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src and dst must both be (N, 3) and the same shape")
    n = len(src)
    if n < 3:
        raise ValueError(f"need at least 3 correspondences, got {n}")

    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s0, d0 = src - mu_s, dst - mu_d
    H = d0.T @ s0 / n
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    scale = 1.0
    if with_scale:
        var = float(np.mean(np.sum(s0 ** 2, axis=1)))
        if var < 1e-15:
            raise ValueError("all source points coincide -- no scale to fit")
        scale = float(np.trace(np.diag(D) @ S) / var)
    return make_T(mu_d - scale * R @ mu_s, scale * R), scale


def fit_report(src: np.ndarray, dst: np.ndarray, T: np.ndarray,
               scale: float = 1.0) -> Dict[str, Any]:
    """Residuals, split into the constant part and the scattered part."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    pred = src @ (T[:3, :3]).T + T[:3, 3]
    res = dst - pred
    norms = np.linalg.norm(res, axis=1)
    offset = res.mean(axis=0)
    scatter = res - offset
    return {
        "n": int(len(src)),
        "scale": float(scale),
        "scale_error_pct": float(abs(scale - 1.0) * 100.0),
        "residual_mm": {
            "rms": float(np.sqrt(np.mean(norms ** 2)) * 1e3),
            "mean": float(np.mean(norms) * 1e3),
            "max": float(np.max(norms) * 1e3),
            "per_point": (norms * 1e3).tolist(),
        },
        "constant_offset_mm": (offset * 1e3).tolist(),
        "constant_offset_magnitude_mm": float(np.linalg.norm(offset) * 1e3),
        "scatter_rms_mm": float(
            np.sqrt(np.mean(np.sum(scatter ** 2, axis=1))) * 1e3),
        "residual_vectors_mm": (res * 1e3).tolist(),
    }


def cmd_anchor(args) -> None:
    doc = load_json(Path(args.points))
    corr = doc.get("correspondences", [])
    if len(corr) < args.min_points:
        raise SystemExit(
            f"{len(corr)} correspondences, --min-points is {args.min_points}.\n"
            f"  Collect more in world_view.py's Measure tab: drive an arm to "
            f"a pose,\n  click the same gripper feature in both camera "
            f"images, RECORD.")

    keep = [c for c in corr if not c.get("excluded")]
    if args.max_ray_gap_mm is not None:
        dropped = [c for c in keep
                   if c.get("ray_gap_mm", 0.0) > args.max_ray_gap_mm]
        keep = [c for c in keep if c not in dropped]
        if dropped:
            print(f"  dropped {len(dropped)} correspondence(s) whose two "
                  f"clicks disagreed by more than "
                  f"{args.max_ray_gap_mm} mm")
    if len(keep) < args.min_points:
        raise SystemExit(f"only {len(keep)} correspondences survived filtering")

    src = np.array([c["p_ref_triangulated"] for c in keep], dtype=float)
    dst = np.array([c["p_base_fk"] for c in keep], dtype=float)
    ref_frame = doc.get("ref_frame", "unknown")

    ## Degenerate geometry check. Three collinear points fit a rotation
    ## about that line arbitrarily; the fit succeeds and is meaningless.
    sv = np.linalg.svd(src - src.mean(axis=0), compute_uv=False)
    extent_mm = (sv / np.sqrt(max(len(src) - 1, 1))) * 1e3
    print(f"\n  {len(keep)} correspondences, rig frame '{ref_frame}'")
    print(f"  spatial spread (principal, mm): "
          + " x ".join(f"{v:.0f}" for v in extent_mm))
    if extent_mm[2] < args.min_spread_mm:
        print(f"  ! the points are nearly COPLANAR (thinnest direction "
              f"{extent_mm[2]:.0f} mm < {args.min_spread_mm:.0f}).\n"
              f"    The fit will be poorly constrained out of that plane. "
              f"Add poses at\n    clearly different heights.")

    T_rigid, _ = umeyama(src, dst, with_scale=False)
    rep_rigid = fit_report(src, dst, T_rigid, 1.0)
    T_sim, scale = umeyama(src, dst, with_scale=True)
    rep_sim = fit_report(src, dst, T_sim, scale)

    print(f"""
  RIGID FIT (6 dof) -- this residual IS the forward-kinematics
  disagreement; the fit cannot absorb it.

    rms {rep_rigid['residual_mm']['rms']:.2f} mm    worst {rep_rigid['residual_mm']['max']:.2f} mm
    constant offset {rep_rigid['constant_offset_magnitude_mm']:.2f} mm  """
          f"""[{rep_rigid['constant_offset_mm'][0]:+.1f}, """
          f"""{rep_rigid['constant_offset_mm'][1]:+.1f}, """
          f"""{rep_rigid['constant_offset_mm'][2]:+.1f}]
    scatter about that offset {rep_rigid['scatter_rms_mm']:.2f} mm

  SIMILARITY FIT (7 dof) -- does a scale error explain it better?

    scale {scale:.6f}  ({rep_sim['scale_error_pct']:+.3f} %)
    rms {rep_sim['residual_mm']['rms']:.2f} mm""")

    improvement = (rep_rigid["residual_mm"]["rms"]
                   - rep_sim["residual_mm"]["rms"])
    if rep_sim["scale_error_pct"] > 0.5 and improvement > 0.2 * rep_rigid["residual_mm"]["rms"]:
        print(f"""
    ! A {rep_sim['scale_error_pct']:.2f} % scale error fits materially
      better. Something is wrong with a LENGTH, and there are only two
      candidates: the ChArUco square in charuco_board.json (which sets
      the cameras' scale) or the URDF link lengths (which set the
      robot's). Measure the printed square again first -- it is the
      cheaper of the two to check.""")
    else:
        print("      no meaningful scale error -- the two rulers agree")

    ## Per-arm breakdown: a systematic difference between arms is a base
    ## placement error, not a joint error, and is invisible in the pooled
    ## number.
    per_arm: Dict[str, Any] = {}
    arms = sorted({c.get("arm") for c in keep if c.get("arm")})
    if len(arms) > 1:
        print("\n  per-arm residual (a systematic difference between arms "
              "is a BASE\n  placement error, not a joint error):")
        res = np.asarray(rep_rigid["residual_vectors_mm"])
        for a in arms:
            idx = [i for i, c in enumerate(keep) if c.get("arm") == a]
            v = res[idx]
            per_arm[a] = {
                "n": len(idx),
                "mean_residual_mm": v.mean(axis=0).tolist(),
                "rms_mm": float(np.sqrt(np.mean(np.sum(v ** 2, axis=1)))),
            }
            m = per_arm[a]["mean_residual_mm"]
            print(f"    {a:<7s} n={len(idx):<3d} mean "
                  f"[{m[0]:+6.1f}, {m[1]:+6.1f}, {m[2]:+6.1f}] mm   "
                  f"rms {per_arm[a]['rms_mm']:.2f} mm")
        if len(arms) == 2:
            a, b = arms
            d = (np.asarray(per_arm[a]["mean_residual_mm"])
                 - np.asarray(per_arm[b]["mean_residual_mm"]))
            print(f"    difference {a} - {b}: "
                  f"[{d[0]:+.1f}, {d[1]:+.1f}, {d[2]:+.1f}] mm")
            print(f"      -> the x component is the direct measurement of "
                  f"the base separation error")

    DIR_EXTRINSICS.mkdir(parents=True, exist_ok=True)
    out_doc = {
        "metadata": provenance("scene_extrinsics_anchor",
                               points_file=str(args.points),
                               n_correspondences=len(keep),
                               n_supplied=len(corr)),
        "convention": (
            "T_base_ref maps points from the rig reference frame into the "
            "robot world frame 'base'; it IS the reference camera's pose "
            "in 'base'."),
        "ref_frame": ref_frame,
        "stereo_file": doc.get("stereo_file"),
        "T_base_ref_row_major": T_rigid.tolist(),
        "rigid_fit": rep_rigid,
        "similarity_fit": {**rep_sim,
                           "T_base_ref_row_major": T_sim.tolist()},
        "per_arm": per_arm,
        "spatial_spread_mm": extent_mm.tolist(),
        "correspondences_used": keep,
        "interpretation": (
            "residual_mm is how far the cameras and forward kinematics "
            "disagree. constant_offset_mm surviving a 6-dof fit means a "
            "frame is defined in the wrong PLACE (a missing tool offset, a "
            "wrong link origin); scatter_rms_mm means joint calibration, "
            "backlash or compliance. The rigid transform is the one "
            "written to ee_camera_transforms.json -- the similarity fit is "
            "diagnostic only, since accepting its scale would silently "
            "rescale every future measurement."),
    }
    out = Path(args.out) if args.out else (
        DIR_EXTRINSICS / f"anchor_{timestamp()}.json")
    save_json(out_doc, out, overwrite=args.overwrite)
    print(f"\n  saved -> {out}")
    print(f"""
  Next, compose the two and make the cameras usable in world coordinates:

      python calibration/scene_extrinsics.py write \\
          --stereo <stereo_*.json> --anchor {out}
""")


## ------------------------------------------------------------------ ##
## write -- into the one file camera_mount.py reads
## ------------------------------------------------------------------ ##

def cmd_write(args) -> None:
    stereo = load_json(Path(args.stereo))
    anchor = load_json(Path(args.anchor))

    ref = stereo["reference_camera"]
    if anchor.get("ref_frame") not in (ref, None):
        raise SystemExit(
            f"the anchor was computed in frame '{anchor.get('ref_frame')}' "
            f"but this stereo calibration is referenced to '{ref}'.\n"
            f"  Composing them would produce a plausible, wrong pose. Re-run "
            f"the anchor against this stereo file.")

    T_base_ref = np.asarray(anchor["T_base_ref_row_major"], dtype=float)
    res = anchor.get("rigid_fit", {}).get("residual_mm", {})
    n = anchor.get("rigid_fit", {}).get("n")

    path = Path(args.config)
    doc = load_json(path) if path.exists() else {"cameras": {}}
    doc.setdefault("cameras", {})

    written = []
    for name, entry in stereo["cameras"].items():
        T_ref_cam = np.asarray(entry["T_ref_cam_row_major"], dtype=float)
        T_base_cam = T_base_ref @ T_ref_cam
        if name in doc["cameras"] and not args.overwrite:
            raise SystemExit(
                f"'{name}' already has a calibrated transform in {path}.\n"
                f"  A camera pose is an experimental record; pass --overwrite "
                f"to replace it deliberately, after noting the old value.")
        doc["cameras"][name] = {
            "rigid_to": "base",
            "matrix_4x4_row_major": T_base_cam.tolist(),
            "provenance": f"calibrated:scene_extrinsics_{timestamp()}",
            ## Not validated by a ruler yet -- only by its own fit.
            "validated": False,
            "rms_error": res.get("rms"),
            "rms_error_units": "mm (robot-vs-camera anchor residual)",
            "n_samples": n,
            "quaternion_wxyz": matrix_to_quat_wxyz(
                T_base_cam[:3, :3]).tolist(),
            "note": (
                f"Static scene camera. Camera-to-camera from "
                f"{Path(args.stereo).name}, anchored to the robot by "
                f"{Path(args.anchor).name} ({n} correspondences, "
                f"{res.get('rms', float('nan')):.2f} mm rms). "
                f"'validated' stays false until a distance measured through "
                f"this rig has been checked against a ruler."),
        }
        written.append(name)

    save_json(doc, path, overwrite=True)
    print(f"\n  wrote {', '.join(written)} -> {path}")
    for name in written:
        T = np.asarray(doc["cameras"][name]["matrix_4x4_row_major"])
        print(f"\n  {name}  (T_base_camera)")
        print(format_T(T, indent="      "))
    print(f"""
  camera_mount.py now reports these as CALIBRATED, T_world_camera() stops
  raising for them, and world_view.py draws their frusta in the right place.

      python calibration/camera_mount.py --world
      python calibration/world_view.py --from-robot \\
          --cameras top_scene low_scene
""")

## ------------------------------------------------------------------ ##
## status
## ------------------------------------------------------------------ ##

def cmd_status(args) -> None:
    import camera_mount as CM
    from rig import rig_status

    st = rig_status(tuple(args.cameras))
    print("\n  SCENE CAMERA CALIBRATION STATUS")
    print("  " + "=" * 62)
    for name, e in st["cameras"].items():
        print(f"\n  {name}")
        print(f"    intrinsics : {e.get('intrinsics') or 'MISSING'}")
        if e.get("intrinsics_error"):
            print(f"                 {e['intrinsics_error']}")
        if e.get("pose_known"):
            mount = CM.resolve_mount(name)
            T = np.asarray(mount["T_parent_optical"], dtype=float)
            print(f"    pose       : KNOWN, rigid to '{mount['rigid_to']}' "
                  f"({mount['provenance']})")
            print(format_T(T, indent="                 "))
        else:
            print(f"    pose       : *** UNKNOWN ***")

    print(f"\n  camera-to-camera : {st.get('stereo_file') or 'MISSING'}")
    if st.get("stereo_error"):
        print(f"                     {st['stereo_error']}")
    print(f"  anchored to robot: {'YES' if st['anchored'] else 'NO'}")
    if not st["anchored"]:
        print("""
  Without an anchor you can still measure -- lengths and sizes are
  already correct in the reference camera's frame. What you cannot do is
  compare a measurement against forward kinematics, which is the whole
  point of anchoring.""")


## ------------------------------------------------------------------ ##
## CLI
## ------------------------------------------------------------------ ##

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="capture shared ChArUco views")
    c.add_argument("--cameras", nargs="+", default=list(SCENE_CAMERAS))
    c.add_argument("--target", type=int, default=25)
    c.add_argument("--min-corners", type=int, default=8)
    c.add_argument("--min-tilt", type=float, default=15.0,
                   help="reject views more fronto-parallel than this, in "
                        "degrees -- they carry almost no information")
    c.add_argument("--auto", action="store_true",
                   help="save automatically instead of waiting for SPACE; "
                        "the only way to collect without a display")
    c.add_argument("--interval", type=float, default=1.5,
                   help="with --auto: minimum seconds between saves")
    c.add_argument("--no-display", action="store_true")
    c.add_argument("--status-interval", type=float, default=1.0,
                   help="with --auto: seconds between terminal status "
                        "lines, which are the only feedback you get "
                        "without a preview window")
    c.add_argument("--min-move", type=float, default=40.0,
                   help="reject a view unless the board moved this many mm "
                        "since the last one (or turned --min-rot)")
    c.add_argument("--min-rot", type=float, default=12.0,
                   help="degrees, see --min-move")
    c.add_argument("--outdir", default=None)
    c.add_argument("--width", type=int, default=640)
    c.add_argument("--height", type=int, default=480)
    c.add_argument("--fps", type=int, default=60)
    c.add_argument("--prefer-intrinsics", default="charuco",
                   choices=["charuco", "factory"])
    c.add_argument("--overwrite", action="store_true")
    B.add_board_args(c)
    c.set_defaults(fn=cmd_collect)

    s = sub.add_parser("solve", help="camera-to-camera geometry")
    s.add_argument("--dir", required=True, help="a collect session directory")
    s.add_argument("--reference", default=None,
                   help="which camera's frame the rig is expressed in "
                        "(default: the first)")
    s.add_argument("--min-views", type=int, default=6)
    s.add_argument("--no-refine", action="store_true",
                   help="skip the joint bundle refinement")
    s.add_argument("--prefer-intrinsics", default="charuco",
                   choices=["charuco", "factory"])
    s.add_argument("--out", default=None)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(fn=cmd_solve)

    a = sub.add_parser("anchor", help="tie the rig to the robot world frame")
    a.add_argument("--points", required=True,
                   help="correspondences JSON from world_view.py's Measure "
                        "tab (robot FK point vs triangulated point)")
    a.add_argument("--min-points", type=int, default=6)
    a.add_argument("--max-ray-gap-mm", type=float, default=8.0,
                   help="drop correspondences whose two clicks disagreed by "
                        "more than this; None keeps everything")
    a.add_argument("--min-spread-mm", type=float, default=60.0,
                   help="warn below this extent in the thinnest principal "
                        "direction -- near-coplanar points fit badly")
    a.add_argument("--out", default=None)
    a.add_argument("--overwrite", action="store_true")
    a.set_defaults(fn=cmd_anchor)

    w = sub.add_parser("write", help="compose into ee_camera_transforms.json")
    w.add_argument("--stereo", required=True)
    w.add_argument("--anchor", required=True)
    w.add_argument("--config", default=str(MOUNT_CONFIG))
    w.add_argument("--overwrite", action="store_true",
                   help="replace an existing calibrated transform")
    w.set_defaults(fn=cmd_write)

    st = sub.add_parser("status", help="what exists and what is missing")
    st.add_argument("--cameras", nargs="+", default=list(SCENE_CAMERAS))
    st.set_defaults(fn=cmd_status)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
