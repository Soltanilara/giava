"""Hand-eye calibration: measuring where the camera actually is.

This is how you replace the nominal `T_flange_camera` with a measured one.

THE IDEA
========
You cannot put a ruler on a camera's optical centre -- it is inside the
lens, a few millimetres from anything you can touch, and its ORIENTATION is
hopeless to measure by hand (1 degree of error at a 100 mm working distance
is already ~1.7 mm).  So you measure it indirectly, using the robot as the
instrument.

Fix a ChArUco board somewhere the camera can see it.  Move the arm to
several poses.  At each pose you know two things:

    T_base_flange   from forward kinematics (the encoders)
    T_cam_board     from solvePnP on the detected board

The board never moves, so for every pose

    T_base_flange(i) @ X @ T_cam_board(i)  =  T_base_board  =  constant

where X = T_flange_camera is the unknown.  Taking any two poses i, j and
eliminating the constant gives the classic

    A X = X B,      A = T_base_flange(j)^-1 @ T_base_flange(i)
                    B = T_cam_board(j) @ T_cam_board(i)^-1

which is solved here by Park & Martin's method (1994): the rotation from a
closed-form matrix square root over the log-map of the relative rotations,
then the translation by linear least squares.

    NOTE: OpenCV's `cv2.calibrateHandEye` is NOT available in the OpenCV 5
    build in this environment (only the CALIB_HAND_EYE_* constants survive),
    so the solver is implemented here rather than called out to.  It is
    validated against synthetic data with a known ground truth in
    selftest.py.

WHAT MAKES OR BREAKS IT
=======================
ROTATION DIVERSITY.  This is the one that catches people.  Pure translations
carry NO information about the rotation part of X: if every A has identity
rotation the problem is singular.  You need poses whose ORIENTATIONS differ,
ideally about several different axes.  Two rotations about the same axis are
nearly as bad as one.  The tool reports the rotation spread and refuses to
pretend a degenerate set gave an answer.

The board must be rigidly fixed, and it must not move between poses -- if it
shifts, the constant is not constant and the result is quietly wrong.

Scale comes from the board's square length, so `charuco_board.json` must be
measured before this means anything in metres.

USAGE
    1. Fix the board where the wrist camera can see it from many angles.
    2. Collect poses -- position the arm with world_view's Jog/Poses tabs,
       or by hand with torque off, and capture at each:

         python calibration/handeye.py collect --camera right_wrist

    3. Solve:

         python calibration/handeye.py solve --dir <session dir>

    4. It writes the result into ee_camera_transforms.json, after which
       camera_mount.py reports the camera as CALIBRATED.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ is a SIBLING of this package now, not the
## parent it used to be.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
for _p in (str(HERE), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    DATA_ROOT,
    T_to_dict,
    ensure_dirs,
    ensure_ros_path,
    invert_T,
    load_json,
    make_T,
    provenance,
    rotation_angle_deg,
    save_json,
    timestamp,
    wait_for_enter,
)

DIR_HANDEYE = DATA_ROOT / "handeye"


## ------------------------------------------------------------------ ##
## SO(3) log / exp
## ------------------------------------------------------------------ ##

def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector (magnitude = angle in rad)."""
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    theta = np.arccos(c)
    if theta < 1e-9:
        return np.zeros(3)
    if abs(np.pi - theta) < 1e-6:
        # Near pi the antisymmetric part vanishes; recover the axis from
        # the symmetric part instead.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        k = int(np.argmax(axis))
        if axis[k] > 1e-12:
            axis = A[:, k] / axis[k]
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return axis * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (theta / (2.0 * np.sin(theta)))


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Axis-angle vector -> rotation matrix (Rodrigues)."""
    w = np.asarray(w, dtype=float)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + np.sin(theta) * K
            + (1.0 - np.cos(theta)) * (K @ K))


## ------------------------------------------------------------------ ##
## Park & Martin
## ------------------------------------------------------------------ ##

def solve_ax_xb(A_list: Sequence[np.ndarray],
                B_list: Sequence[np.ndarray]) -> np.ndarray:
    """Solve A X = X B for X, given matched relative motions.

    Park & Martin (1994).  Rotation first, in closed form; then translation
    by least squares using the recovered rotation."""
    if len(A_list) < 2:
        raise ValueError("need at least 2 relative motions (3+ poses)")

    M = np.zeros((3, 3))
    for A, B in zip(A_list, B_list):
        a = so3_log(A[:3, :3])
        b = so3_log(B[:3, :3])
        M += np.outer(b, a)

    # R_X = (M^T M)^(-1/2) M^T
    MtM = M.T @ M
    evals, evecs = np.linalg.eigh(MtM)
    evals = np.maximum(evals, 1e-15)
    inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    R_X = inv_sqrt @ M.T

    # Re-project onto SO(3): numerical drift leaves it slightly off.
    U, _, Vt = np.linalg.svd(R_X)
    R_X = U @ Vt
    if np.linalg.det(R_X) < 0:
        U[:, -1] *= -1
        R_X = U @ Vt

    # (R_A - I) t_X = R_X t_B - t_A
    C, d = [], []
    for A, B in zip(A_list, B_list):
        C.append(A[:3, :3] - np.eye(3))
        d.append(R_X @ B[:3, 3] - A[:3, 3])
    t_X, *_ = np.linalg.lstsq(np.vstack(C), np.concatenate(d), rcond=None)
    return make_T(t_X, R_X)


def build_motions(T_base_flange: Sequence[np.ndarray],
                  T_cam_board: Sequence[np.ndarray]
                  ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Relative motions between every pair of poses.

    Using ALL pairs rather than consecutive ones gives far more constraints
    from the same data, and avoids a result that depends on the order the
    poses happened to be collected in."""
    A_list, B_list = [], []
    n = len(T_base_flange)
    for i in range(n):
        for j in range(i + 1, n):
            A_list.append(invert_T(T_base_flange[j]) @ T_base_flange[i])
            B_list.append(T_cam_board[j] @ invert_T(T_cam_board[i]))
    return A_list, B_list


def residuals(X: np.ndarray, A_list, B_list) -> Dict[str, Any]:
    """How well A X = X B actually holds -- the honest quality figure."""
    rot, trans = [], []
    for A, B in zip(A_list, B_list):
        E = invert_T(A @ X) @ (X @ B)
        rot.append(rotation_angle_deg(E[:3, :3]))
        trans.append(float(np.linalg.norm(E[:3, 3])))
    r, t = np.asarray(rot), np.asarray(trans)
    return {
        "n_pairs": int(len(rot)),
        "rotation_deg": {"mean": float(r.mean()), "median": float(np.median(r)),
                         "max": float(r.max())},
        "translation_mm": {"mean": float(t.mean() * 1e3),
                           "median": float(np.median(t) * 1e3),
                           "max": float(t.max() * 1e3)},
    }


def rotation_diversity(T_base_flange: Sequence[np.ndarray]) -> Dict[str, Any]:
    """Is there enough rotation variety to constrain X at all?

    Pure translation carries no information about the rotation of X.  This
    measures how much the poses actually rotate, and whether they rotate
    about more than one axis."""
    axes, angles = [], []
    n = len(T_base_flange)
    for i in range(n):
        for j in range(i + 1, n):
            R = (invert_T(T_base_flange[j]) @ T_base_flange[i])[:3, :3]
            w = so3_log(R)
            ang = float(np.linalg.norm(w))
            if ang > 1e-6:
                angles.append(np.degrees(ang))
                axes.append(w / ang)
    if not axes:
        return {"n_rotations": 0, "max_angle_deg": 0.0,
                "axis_rank": 0, "adequate": False,
                "why": "the poses have no relative rotation at all"}
    A = np.asarray(axes)
    s = np.linalg.svd(A, compute_uv=False)
    # How many independent axes the rotations span (of a possible 3).
    rank = int(np.sum(s > 0.1 * s[0]))
    ang = np.asarray(angles)
    adequate = bool(rank >= 2 and ang.max() >= 15.0)
    return {
        "n_rotations": len(angles),
        "max_angle_deg": float(ang.max()),
        "mean_angle_deg": float(ang.mean()),
        "axis_singular_values": s.tolist(),
        "axis_rank": rank,
        "adequate": adequate,
        "why": ("ok" if adequate else
                ("rotations span only %d independent axis/axes (need >=2) "
                 "and the largest is %.1f deg (want >=15). Pure translation "
                 "cannot constrain the camera's orientation."
                 % (rank, ang.max()))),
    }


## ------------------------------------------------------------------ ##
## Solve from a collected session
## ------------------------------------------------------------------ ##

def board_pose_in_camera(image, spec, K, dist, min_corners: int = 8):
    """T_cam_board from a ChArUco detection, or None."""
    import cv2

    import board as B

    detector, board = B.make_detector(spec)
    det = B.detect(image, detector, board, min_corners=min_corners)
    if not det.ok:
        return None, det.reason
    ok, rvec, tvec = cv2.solvePnP(
        det.obj_points.reshape(-1, 1, 3), det.img_points.reshape(-1, 1, 2),
        K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, "solvePnP failed"
    R, _ = cv2.Rodrigues(rvec)
    return make_T(tvec.ravel(), R), f"{det.n_corners} corners"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="capture image + robot pose at each "
                                       "arm position you set")
    c.add_argument("--camera", required=True)
    c.add_argument("--arm", default=None,
                   help="arm the camera rides on (default: inferred)")
    c.add_argument("--outdir", default=None)
    c.add_argument("--width", type=int, default=640)
    c.add_argument("--height", type=int, default=480)
    c.add_argument("--fps", type=int, default=60)

    s = sub.add_parser("solve", help="solve AX=XB from a collected session")
    s.add_argument("--dir", required=True)
    s.add_argument("--intrinsics", default=None,
                   help="intrinsics JSON (default: newest factory file for "
                        "this camera)")
    s.add_argument("--write", action="store_true",
                   help="write the result into ee_camera_transforms.json")
    s.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()
    ensure_dirs()
    DIR_HANDEYE.mkdir(parents=True, exist_ok=True)

    if args.cmd == "collect":
        _collect(args)
    else:
        _solve(args)


def _collect(args) -> None:
    import cv2

    from capture_session import CAMERA_ARM
    from kinematics import ARM_ORDER, JointStateListener, RobotFrames
    from rs_camera import RealSenseCamera, require_rs
    from arm_config import ARM_CONFIG

    require_rs()
    ensure_ros_path()
    import rospy
    from kinematics import JointFrameBridge

    arm = args.arm or CAMERA_ARM.get(args.camera)
    if arm is None:
        raise SystemExit(f"  cannot infer which arm '{args.camera}' is on; "
                         f"pass --arm")

    frames = RobotFrames()
    if not rospy.core.is_initialized():
        rospy.init_node("giava_handeye", anonymous=True, disable_signals=True)
    bridge = JointFrameBridge(frames.robot)
    listener = JointStateListener(ARM_ORDER)
    rospy.sleep(1.0)
    if not listener.ready(arm):
        raise SystemExit(f"  no joint_states for '{arm}' -- is the driver "
                         f"running?")

    outdir = Path(args.outdir) if args.outdir else (
        DIR_HANDEYE / f"{args.camera}_{timestamp()}")
    (outdir / "images").mkdir(parents=True, exist_ok=True)

    print(f"""
  Hand-eye collection -- {args.camera} on the {arm} arm
  output: {outdir}

  Fix the ChArUco board where this camera can see it, then move the arm to
  a new pose and capture.  ROTATE the wrist between poses, about DIFFERENT
  axes -- pure translation cannot constrain the camera's orientation and
  the solve will refuse a degenerate set.

  15-20 poses is plenty.  Type ENTER to capture, 'q' then ENTER to finish.
""")
    entries: List[Dict[str, Any]] = []
    link = ARM_CONFIG[arm]["ee_link"]
    with RealSenseCamera(args.camera, width=args.width, height=args.height,
                         fps=args.fps, stream="color") as cam:
        while True:
            try:
                r = input(f"  [{len(entries)} captured] ENTER to capture, "
                          f"q to finish: ").strip().lower()
            except EOFError:
                break
            if r == "q":
                break
            rec = cam.capture()
            q = listener.q_driver(frames)
            T = frames.ee_pose(bridge.to_urdf(q), arm)
            i = len(entries)
            fn = f"images/pose{i:03d}.png"
            cv2.imwrite(str(outdir / fn),
                        cv2.cvtColor(rec.image, cv2.COLOR_RGB2BGR))
            entries.append({
                "index": i, "image": fn,
                "q_driver": q[frames.joint_indices(arm)].tolist(),
                "T_base_flange": T_to_dict(T, "base", link),
                "frame_number": rec.frame_number,
            })
            print(f"      captured pose {i}: flange at "
                  f"[{T[0,3]:+.3f}, {T[1,3]:+.3f}, {T[2,3]:+.3f}] m")

    if len(entries) < 3:
        print("\n  fewer than 3 poses -- not enough to solve.")
    save_json({
        "metadata": provenance("handeye_collect", camera=args.camera,
                               arm=arm, width=args.width,
                               height=args.height),
        "camera": args.camera, "arm": arm, "ee_link": link,
        "n_poses": len(entries), "poses": entries,
    }, outdir / "collection.json", overwrite=True)
    print(f"\n  {len(entries)} pose(s) saved to {outdir}")
    print(f"  solve with:\n"
          f"      python calibration/handeye.py solve --dir {outdir}\n")


def _solve(args) -> None:
    import cv2

    import board as B
    from common import latest_matching, DIR_CAMERAS

    d = Path(args.dir)
    coll = load_json(d / "collection.json")
    camera, arm = coll["camera"], coll["arm"]
    spec = B.load_spec()

    ## Intrinsics: a hand-eye result is only as good as the camera model
    ## used for solvePnP.
    ipath = Path(args.intrinsics) if args.intrinsics else None
    if ipath is None:
        hits = sorted(DIR_CAMERAS.glob(f"{camera}_*"))
        if hits:
            ipath = latest_matching(hits[-1], "charuco_intrinsics_*.json") \
                or latest_matching(hits[-1], "factory_intrinsics_*.json")
    if ipath is None:
        raise SystemExit(
            f"  no intrinsics for '{camera}'. Run:\n"
            f"      python calibration/rs_intrinsics.py --cameras {camera}")
    idoc = load_json(ipath)
    if "streams" in idoc:      # factory file
        i = idoc["streams"]["color"]["intrinsics"]
        K = np.asarray(i["camera_matrix"], dtype=float)
        dist = np.asarray(i["coeffs"], dtype=float)
        isrc = "realsense_factory"
    else:                       # charuco file
        K = np.asarray(idoc["camera_matrix"], dtype=float)
        dist = np.asarray(idoc["distortion_coefficients"], dtype=float)
        isrc = "opencv_charuco"

    print()
    print("#" * 74)
    print(f"#  Hand-eye solve  --  {camera} on the {arm} arm")
    print("#" * 74)
    print(f"\n  board      : {spec.describe()}")
    print(f"  intrinsics : {ipath.name}  [{isrc}]")
    print(f"  poses      : {coll['n_poses']}\n")

    T_bf, T_cb, used = [], [], []
    for e in coll["poses"]:
        img = cv2.imread(str(d / e["image"]), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  pose {e['index']:3d}: unreadable image")
            continue
        T, why = board_pose_in_camera(img, spec, K, dist)
        if T is None:
            print(f"  pose {e['index']:3d}: board not usable ({why})")
            continue
        T_bf.append(np.asarray(e["T_base_flange"]["matrix_4x4_row_major"]))
        T_cb.append(T)
        used.append(e["index"])
        print(f"  pose {e['index']:3d}: {why}, board at "
              f"{np.linalg.norm(T[:3,3])*1e3:.0f} mm")

    if len(T_bf) < 3:
        raise SystemExit(f"\n  only {len(T_bf)} usable pose(s); need >= 3 "
                         f"(and realistically 10+).")

    div = rotation_diversity(T_bf)
    print(f"\n  rotation diversity: {div['n_rotations']} relative rotations, "
          f"max {div['max_angle_deg']:.1f} deg, "
          f"spanning {div['axis_rank']} independent axis/axes")
    if not div["adequate"]:
        print(f"\n  *** NOT ENOUGH ROTATION VARIETY ***\n  {div['why']}")
        print("  Collect more poses with the wrist rotated about different")
        print("  axes, then solve again. Refusing to report a number that")
        print("  the data does not support.\n")
        raise SystemExit(1)

    A_list, B_list = build_motions(T_bf, T_cb)
    X = solve_ax_xb(A_list, B_list)
    res = residuals(X, A_list, B_list)

    t = X[:3, 3]
    print(f"""
  RESULT   T_{arm}_gripper_base -> {camera}   (the camera's pose in the flange frame)

      translation [mm] : [{t[0]*1e3:+.2f}, {t[1]*1e3:+.2f}, {t[2]*1e3:+.2f}]
      |t|              : {np.linalg.norm(t)*1e3:.2f} mm

  residuals of A X = X B over {res['n_pairs']} pose pairs:
      rotation    mean {res['rotation_deg']['mean']:.3f} deg   """
          f"""max {res['rotation_deg']['max']:.3f}
      translation mean {res['translation_mm']['mean']:.2f} mm    """
          f"""max {res['translation_mm']['max']:.2f}
""")
    from common import format_T
    print(format_T(X, indent="      "))

    nominal = _nominal(camera)
    if nominal is not None:
        dt = np.linalg.norm(X[:3, 3] - nominal[:3, 3]) * 1e3
        da = rotation_angle_deg(nominal[:3, :3].T @ X[:3, :3])
        print(f"\n  vs the NOMINAL (MuJoCo-derived) mount transform:")
        print(f"      translation differs by {dt:.2f} mm")
        print(f"      rotation    differs by {da:.2f} deg")
        print("      A large difference is not automatically an error -- the "
              "nominal\n      value was never measured. Judge by the "
              "residuals above.")

    doc = {
        "metadata": provenance("handeye_park_martin", camera=camera, arm=arm,
                               intrinsics_source=isrc,
                               intrinsics_file=str(ipath),
                               opencv_calibrateHandEye_available=False),
        "camera": camera, "arm": arm,
        "n_poses_used": len(T_bf), "pose_indices_used": used,
        "board": spec.to_dict(),
        "rotation_diversity": div,
        "residuals": res,
        "transform": T_to_dict(X, coll["ee_link"], camera),
        "method": ("Park & Martin (1994) closed-form AX=XB. "
                   "cv2.calibrateHandEye is unavailable in this OpenCV 5 "
                   "build, so the solver is implemented in handeye.py and "
                   "validated against synthetic ground truth in selftest.py."),
        "caveat": ("Scale comes from the board's square_length_m. If the "
                   "board is not measured, the translation is scaled by "
                   "whatever that really is."),
    }
    out = d / f"handeye_result_{timestamp()}.json"
    save_json(doc, out, overwrite=args.overwrite)
    print(f"\n  saved to {out}")

    if args.write:
        _write_mount(camera, X, coll["ee_link"], res, out)
    else:
        print(f"\n  Not written to ee_camera_transforms.json. Re-run with "
              f"--write once you are happy with the residuals.\n")


def _nominal(camera: str):
    import camera_mount as CM
    return CM.CAMERA_MOUNTS.get(camera, {}).get("T_parent_optical")


def _write_mount(camera: str, X: np.ndarray, link: str, res, src: Path) -> None:
    import camera_mount as CM

    path = CM.TRANSFORM_CONFIG
    doc = load_json(path)
    doc.setdefault("cameras", {})[camera] = {
        "rigid_to": link,
        "matrix_4x4_row_major": X.tolist(),
        "provenance": "calibrated:handeye_park_martin",
        "validated": False,
        "rms_error": res["translation_mm"]["mean"],
        "note": (f"Hand-eye AX=XB, see {src.name}. 'validated' stays false "
                 f"until checked against an independent measurement."),
    }
    save_json(doc, path, overwrite=True)
    print(f"\n  written to {path}")
    print(f"  camera_mount.py will now report {camera} as CALIBRATED.")
    print(f"  Note 'validated' is still false -- residuals are a "
          f"self-consistency\n  check, not proof against the physical rig.\n")


if __name__ == "__main__":
    main()
