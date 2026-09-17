"""Paired image capture with a fully recorded robot pose.

The experiment this exists for: put the arms in a known configuration,
photograph an object (and a stopwatch) with both wrist cameras, and be able
to recover the camera extrinsics afterwards from the recorded pose.

WHAT MAKES THE EXTRINSICS RECOVERABLE
=====================================
Every shot stores the INGREDIENTS, not just the answer:

    measured joint vector          (driver AND URDF coordinates)
    T_world_flange   per arm       (forward kinematics)
    T_flange_camera  per camera    (the mount transform, WITH provenance)
    T_world_camera   per camera    (the composition)
    T_camA_camB                    (the relative extrinsic, what stereo needs)

That matters because `T_flange_camera` is currently NOMINAL -- derived from
the MuJoCo model, never measured.  When a real hand-eye calibration lands,
every session recorded here can be RE-DERIVED offline from the stored joint
angles, without re-shooting anything.  Storing only the composed camera pose
would have baked today's guess in permanently.

The relative extrinsic is the one stereo actually uses, and it is partly
self-correcting: both wrist cameras use the same mount transform, so an
error common to both cancels in T_camA_camB.  Errors that differ between
the two arms do not.

MODES
    burst   consecutive frames as fast as the cameras deliver.  Checks the
            achieved frame rate against the configured one, and the
            inter-camera offset at speed.
    drift   shots spaced out over time, to see whether the offset between
            the cameras CHANGES.  These cameras free-run on independent
            crystals; a constant offset can be calibrated out, a growing
            one cannot.

Examples

    # 5 images from both wrist cameras, right now, at full speed
    python calibration/capture_session.py --cameras left_wrist right_wrist \\
        --mode burst --shots 5

    # drift: one shot every 10 s for 5 minutes
    python calibration/capture_session.py --cameras left_wrist right_wrist \\
        --mode drift --shots 30 --interval 10

    # move to a named pose first, then capture
    python calibration/capture_session.py --cameras left_wrist right_wrist \\
        --pose forward --arms left right --mode burst --shots 5
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

import camera_mount as CM  # noqa: E402
import tcp as TCP  # noqa: E402
from common import (  # noqa: E402
    DIR_VISER,
    T_to_dict,
    ensure_dirs,
    ensure_ros_path,
    invert_T,
    provenance,
    save_json,
    timestamp,
    wait_for_enter,
)
from kinematics import (  # noqa: E402
    ARM_ORDER,
    WORLD_FRAME,
    JointStateListener,
    RobotFrames,
)
from rs_camera import PRODUCTION_COLOR, require_rs, resolve  # noqa: E402
from sync_capture import CameraWorker, _check_delivering  # noqa: E402
from arm_config import ARM_CONFIG  # noqa: E402

## Which arm each wrist camera rides on, for the pose block.
CAMERA_ARM = {"left_wrist": "left", "right_wrist": "right",
              "oak_left": "middle", "oak_right": "middle"}


## ------------------------------------------------------------------ ##
## Robot pose block
## ------------------------------------------------------------------ ##

def pose_block(frames: RobotFrames, bridge, q_driver: np.ndarray,
               cameras: List[str]) -> Dict[str, Any]:
    """Everything needed to rebuild the extrinsics later."""
    q_urdf = bridge.to_urdf(q_driver)
    fk = frames.fk(q_urdf)

    arms: Dict[str, Any] = {}
    for arm in ARM_ORDER:
        link = ARM_CONFIG[arm]["ee_link"]
        T = fk[link]
        idx = frames.joint_indices(arm)
        entry = {
            "ee_link": link,
            "ee_link_meaning": ("gripper mounting flange, NOT a TCP"
                                if arm in ("left", "right")
                                else "camera mount link"),
            "q_driver": q_driver[idx].tolist(),
            "q_urdf": q_urdf[idx].tolist(),
            "joint_names": ARM_CONFIG[arm]["joint_names"],
            "T_world_flange": T_to_dict(T, WORLD_FRAME, link),
        }
        t = TCP.resolve(arm)
        if t["T_flange_tcp"] is not None:
            entry["T_world_tcp"] = T_to_dict(
                TCP.tcp_pose(T, arm), WORLD_FRAME, f"{arm}_tcp")
            entry["tcp_provenance"] = t["provenance"]
            entry["tcp_measured"] = t["measured"]
        arms[arm] = entry

    cams: Dict[str, Any] = {}
    for cam in cameras:
        mount = CM.resolve_mount(cam)
        rec: Dict[str, Any] = {
            "rigid_to": mount["rigid_to"],
            "arm": CAMERA_ARM.get(cam),
            "mount_provenance": mount["provenance"],
            "mount_validated": mount["validated"],
            "mount_note": mount.get("note"),
        }
        if mount["T_parent_optical"] is None:
            rec["T_flange_camera"] = None
            rec["T_world_camera"] = None
            rec["recoverable"] = False
            rec["why_not"] = ("no mount transform exists for this camera; "
                              "its world pose cannot be composed")
        else:
            T_wc = fk[mount["rigid_to"]] @ mount["T_parent_optical"]
            rec["T_flange_camera"] = T_to_dict(
                mount["T_parent_optical"], mount["rigid_to"], cam)
            rec["T_world_camera"] = T_to_dict(T_wc, WORLD_FRAME, cam)
            rec["recoverable"] = True
        cams[cam] = rec

    ## Relative extrinsic for EVERY ordered pair that has one -- what a
    ## stereo/MVS pipeline consumes, and it saves inverting a 4x4 by hand
    ## just to get the direction you happen to want.
    usable = [c for c in cameras if cams[c].get("T_world_camera")]
    pairs: Dict[str, Any] = {}
    for a in usable:
        for b in usable:
            if a == b:
                continue
            T_wa = np.asarray(cams[a]["T_world_camera"]["matrix_4x4_row_major"])
            T_wb = np.asarray(cams[b]["T_world_camera"]["matrix_4x4_row_major"])
            T_ab = invert_T(T_wa) @ T_wb
            e = T_to_dict(T_ab, a, b)
            e["baseline_m"] = float(np.linalg.norm(T_ab[:3, 3]))
            e["baseline_mm"] = e["baseline_m"] * 1e3
            ## Split out so rotation and translation can be checked by hand
            ## without slicing the 4x4.
            e["rotation_3x3_row_major"] = T_ab[:3, :3].tolist()
            e["translation_m"] = T_ab[:3, 3].tolist()
            pairs[f"{a}__to__{b}"] = e
    relative = None
    if len(usable) >= 2:
        relative = dict(pairs[f"{usable[0]}__to__{usable[1]}"])
        relative["note"] = (
            "The relative extrinsic stereo uses. Both wrist cameras share "
            "the same nominal mount transform, so an error COMMON to both "
            "largely cancels here; an error that differs between the two "
            "arms does not.")

    return {
        "convention": ("T_a_b maps points from frame b into frame a. "
                       "T_world_camera = T_world_flange @ T_flange_camera."),
        "world_frame": WORLD_FRAME,
        "world_axes": {"+x": "operator's left", "+y": "operator's backward",
                       "+z": "up"},
        ## Every link's world pose, so the FK chain can be checked by hand
        ## rather than taken on trust.
        "all_link_poses_world": {
            name: {"xyz_m": T[:3, 3].tolist(),
                   "rotation_3x3_row_major": T[:3, :3].tolist()}
            for name, T in fk.items()},
        "arms": arms,
        "cameras": cams,
        "relative_extrinsic": relative,
        "relative_extrinsics_all_pairs": pairs,
        "recompute_note": (
            "The joint angles and T_world_flange above are MEASUREMENTS. "
            "T_flange_camera is currently nominal, so T_world_camera and "
            "the relative extrinsic inherit that. When a hand-eye "
            "calibration exists, recompose from q_driver + the new mount "
            "transform -- no need to re-shoot this session."),
    }


def intrinsics_block(workers) -> Dict[str, Any]:
    """Intrinsics of the streams being captured, read from the live devices.

    Read from the OPEN camera rather than looked up in a file: intrinsics
    are resolution specific, and reading them here guarantees they describe
    the exact stream these images came out of.  Without this a downstream
    pipeline has images, poses and timestamps but no camera model, and has
    to go matching serials against files by hand.
    """
    out = {}
    for w in workers:
        cam = w.cam
        i = dict(cam.intrinsics or {})
        i["source"] = "realsense_factory (read live at capture time)"
        i["distortion_direction"] = (
            "librealsense convention: for inverse_brown_conrady these "
            "coefficients map PIXEL -> RAY (rs2_deproject_pixel_to_point). "
            "OpenCV's map RAY -> PIXEL. Convert before using with cv2.")
        i["identity"] = cam.stream_metadata()
        out[cam.name] = i
    return out


## ------------------------------------------------------------------ ##

def take_shot(workers: List[CameraWorker], require_new: bool,
              last_frames: Dict[str, Optional[int]],
              timeout_s: float = 2.0) -> Optional[Dict[str, Any]]:
    """One aligned pair. With require_new, waits until BOTH cameras have
    produced a frame not seen in the previous shot -- otherwise a fast loop
    re-reads the same frame and the 'frame rate' is an artefact of the
    polling, not the camera."""
    deadline = time.time() + timeout_s
    while True:
        newest = [w.newest_timestamp_s() for w in workers]
        if all(t is not None for t in newest):
            ref = min(newest)
            picks = [(w, w.nearest(ref)) for w in workers]
            if all(r is not None for _, r in picks):
                fresh = all(
                    last_frames.get(w.cam.name) is None
                    or r.frame_number != last_frames.get(w.cam.name)
                    for w, r in picks)
                if fresh or not require_new:
                    ts = {w.cam.name: r.timestamp_ms * 1e-3 for w, r in picks}
                    return {
                        "reference_s": ref,
                        "spread_s": max(ts.values()) - min(ts.values()),
                        "timestamps_s": ts,
                        "records": {w.cam.name: r for w, r in picks},
                    }
        if time.time() > deadline:
            return None
        time.sleep(0.001)


def analyse(shots: List[Dict[str, Any]], names: List[str],
            mode: str, fps: int) -> Dict[str, Any]:
    """Frame rate and inter-camera offset behaviour."""
    if len(shots) < 2:
        return {"n_shots": len(shots)}
    t = np.array([s["reference_s"] for s in shots])
    spread = np.array([s["spread_s"] for s in shots]) * 1e3
    a, b = names[0], names[1]
    delta = np.array([s["timestamps_s"][a] - s["timestamps_s"][b]
                      for s in shots]) * 1e3
    gaps = np.diff(t) * 1e3

    out = {
        "n_shots": len(shots),
        "elapsed_s": float(t[-1] - t[0]),
        "inter_shot_gap_ms": {
            "mean": float(gaps.mean()), "min": float(gaps.min()),
            "max": float(gaps.max())},
        "achieved_rate_hz": float(1000.0 / gaps.mean()) if gaps.mean() else None,
        "configured_fps": fps,
        "alignment_spread_ms": {
            "mean": float(spread.mean()), "max": float(spread.max())},
        "delta_definition": f"{a} minus {b}",
        "offset_ms": {
            "mean": float(delta.mean()), "std": float(delta.std(ddof=1)),
            "min": float(delta.min()), "max": float(delta.max())},
    }
    ## Drift: is the offset CHANGING? A constant offset is calibratable;
    ## a trend is two clocks running at different rates.
    ##
    ## Only worth fitting over a long enough window.  Extrapolating a slope
    ## from a fraction of a second to "ms per minute" is arithmetic, not
    ## measurement -- a burst spans ~0.1 s, and the fit there is dominated
    ## by timestamp quantisation.
    elapsed = float(t[-1] - t[0])
    MIN_DRIFT_WINDOW_S = 10.0
    if elapsed >= MIN_DRIFT_WINDOW_S:
        slope = float(np.polyfit(t - t[0], delta, 1)[0])   # ms per second
        out["offset_drift_ms_per_s"] = slope
        out["offset_drift_ms_per_min"] = slope * 60.0
        out["offset_drift_window_s"] = elapsed
        out["offset_drift_note"] = (
            "ms of offset change per second of wall clock. A constant "
            "offset can be calibrated out; a nonzero slope means the two "
            "device clocks run at different rates and any fixed correction "
            "goes stale.")
    else:
        out["offset_drift_ms_per_s"] = None
        out["offset_drift_window_s"] = elapsed
        out["offset_drift_note"] = (
            f"NOT MEASURED: the session spans {elapsed:.3f} s, below the "
            f"{MIN_DRIFT_WINDOW_S:.0f} s needed for a meaningful slope. Use "
            f"--mode drift with a long --interval to measure clock drift.")
    return out


def write_summary(path: Path, pose, intr, stats, shots, names) -> None:
    """A flat, readable dump for checking numbers by hand.

    The JSON is the machine record; this is the one you read at the bench
    with a calculator, so every quantity is spelled out in the units you
    would measure in."""
    L = []
    L.append("=" * 74)
    L.append("  CAPTURE SESSION SUMMARY")
    L.append("=" * 74)
    L.append("")
    L.append("  World frame = giava.urdf root link 'base'")
    L.append("      +x operator's LEFT   +y operator's BACKWARD   +z UP")
    L.append("  T_a_b maps points from frame b into frame a,")
    L.append("      and IS the pose of b expressed in a.")
    L.append("")

    L.append("-" * 74)
    L.append("  CAMERA INTRINSICS (factory, read live at capture)")
    L.append("-" * 74)
    for cam, i in (intr or {}).items():
        L.append(f"\n  {cam}   serial {i.get('identity', {}).get('serial')}")
        L.append(f"      {i['width']}x{i['height']} @ "
                 f"{i.get('identity', {}).get('fps')} fps")
        L.append(f"      fx {i['fx']:.5f}   fy {i['fy']:.5f}")
        L.append(f"      cx {i['cx']:.5f}   cy {i['cy']:.5f}")
        L.append(f"      model {i['model']}")
        L.append(f"      coeffs {[round(c, 8) for c in i['coeffs']]}")

    if pose:
        L.append("")
        L.append("-" * 74)
        L.append("  ARM JOINT CONFIGURATION (at session start)")
        L.append("-" * 74)
        for arm, a in pose["arms"].items():
            L.append(f"\n  {arm}  ({a['ee_link']} -- {a['ee_link_meaning']})")
            for n, qd, qu in zip(a["joint_names"], a["q_driver"], a["q_urdf"]):
                flag = "" if abs(qd - qu) < 1e-9 else "   <- driver != urdf"
                L.append(f"      {n:24s} driver {qd:+9.5f}  urdf {qu:+9.5f}"
                         f"{flag}")

        L.append("")
        L.append("-" * 74)
        L.append("  END-EFFECTOR POSES IN WORLD")
        L.append("-" * 74)
        for arm, a in pose["arms"].items():
            t = a["T_world_flange"]
            L.append(f"\n  {arm} FLANGE  ({a['ee_link']})")
            L.append(f"      xyz [m]  {[round(v, 6) for v in t['translation_xyz_m']]}")
            L.append(f"      rpy [deg]{[round(v, 4) for v in t['rpy_deg']]}")
            L.append(f"      quat wxyz {[round(v, 6) for v in t['quaternion_wxyz']]}")
            if a.get("T_world_tcp"):
                tt = a["T_world_tcp"]
                L.append(f"    {arm} TCP (grasp point, "
                         f"{a.get('tcp_provenance')}, "
                         f"measured={a.get('tcp_measured')})")
                L.append(f"      xyz [m]  {[round(v, 6) for v in tt['translation_xyz_m']]}")

        L.append("")
        L.append("-" * 74)
        L.append("  CAMERA POSES IN WORLD")
        L.append("-" * 74)
        for cam, c in pose["cameras"].items():
            L.append(f"\n  {cam}   rigid to {c['rigid_to']}   "
                     f"[{c['mount_provenance']}, validated="
                     f"{c['mount_validated']}]")
            if not c.get("T_world_camera"):
                L.append(f"      NOT AVAILABLE -- {c.get('why_not')}")
                continue
            t = c["T_world_camera"]
            L.append(f"      xyz [m]   {[round(v, 6) for v in t['translation_xyz_m']]}")
            L.append(f"      rpy [deg] {[round(v, 4) for v in t['rpy_deg']]}")
            L.append(f"      quat wxyz {[round(v, 6) for v in t['quaternion_wxyz']]}")
            L.append("      rotation (row-major):")
            for row in t["matrix_4x4_row_major"][:3]:
                L.append("          [" + "  ".join(f"{v:+.6f}"
                                                   for v in row[:3]) + "]")

        pairs = pose.get("relative_extrinsics_all_pairs") or {}
        if pairs:
            L.append("")
            L.append("-" * 74)
            L.append("  CAMERA-TO-CAMERA (relative extrinsics)")
            L.append("-" * 74)
            for key, e in pairs.items():
                L.append(f"\n  {e['name']}   baseline {e['baseline_mm']:.3f} mm")
                L.append(f"      translation [m] "
                         f"{[round(v, 6) for v in e['translation_m']]}")
                L.append(f"      rpy [deg]       "
                         f"{[round(v, 4) for v in e['rpy_deg']]}")
                L.append("      rotation (row-major):")
                for row in e["rotation_3x3_row_major"]:
                    L.append("          [" + "  ".join(f"{v:+.6f}"
                                                       for v in row) + "]")
            L.append("")
            L.append("  NOTE: these inherit the mount transform, which is "
                     "NOMINAL unless")
            L.append("  camera_mount.py reports the camera as CALIBRATED. "
                     "The joint angles")
            L.append("  above are measurements -- recompute from them once "
                     "hand-eye is done.")

    if stats:
        L.append("")
        L.append("-" * 74)
        L.append("  TIMING")
        L.append("-" * 74)
        o = stats.get("offset_ms", {})
        g = stats.get("inter_shot_gap_ms", {})
        L.append(f"    shots {stats.get('n_shots')} over "
                 f"{stats.get('elapsed_s', 0):.3f} s")
        L.append(f"    inter-shot gap  mean {g.get('mean', 0):.3f} ms")
        if stats.get("achieved_rate_hz"):
            L.append(f"    achieved rate   {stats['achieved_rate_hz']:.2f} Hz "
                     f"(configured {stats.get('configured_fps')})")
        L.append(f"    offset ({stats.get('delta_definition','')}) "
                 f"mean {o.get('mean', 0):+.3f} ms  std {o.get('std', 0):.3f}")
        d = stats.get("offset_drift_ms_per_s")
        L.append(f"    clock drift     "
                 + (f"{d*60:+.4f} ms/min" if d is not None
                    else "window too short to measure"))
        L.append("")
        L.append("    Cameras free-run (D405 has no inter_cam_sync_mode);")
        L.append("    these are software clocks, not shutter alignment.")

    L.append("")
    L.append("-" * 74)
    L.append(f"  {len(shots)} shot(s); each carries its own robot_pose in "
             f"session.json")
    L.append("-" * 74)
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", nargs="+",
                    default=["left_wrist", "right_wrist"],
                    help="cameras to capture (default: both wrist cameras)")
    ap.add_argument("--mode", default="burst", choices=("burst", "drift"),
                    help="burst = consecutive frames at full speed; "
                         "drift = spaced shots to watch the offset over time")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--interval", type=float, default=None,
                    help="seconds between shots (default: 0 for burst, "
                         "5 for drift)")
    ap.add_argument("--pose", default=None,
                    help="move the arms to this named pose first "
                         "(needs ROS and torques the arms on)")
    ap.add_argument("--arms", nargs="+", default=["left", "right"],
                    choices=list(ARM_ORDER),
                    help="arms to move when --pose is given")
    ap.add_argument("--no-robot", action="store_true",
                    help="skip the robot entirely -- images and timing only, "
                         "no pose block and no recoverable extrinsics")
    ap.add_argument("--width", type=int, default=PRODUCTION_COLOR[0])
    ap.add_argument("--height", type=int, default=PRODUCTION_COLOR[1])
    ap.add_argument("--fps", type=int, default=PRODUCTION_COLOR[2])
    ap.add_argument("--no-prompt", action="store_true")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import cv2

    require_rs()
    ensure_dirs()
    interval = args.interval if args.interval is not None else (
        0.0 if args.mode == "burst" else 5.0)

    resolved = resolve(args.cameras)
    names = [n for n, _ in resolved]
    outdir = (Path(args.outdir) if args.outdir else
              DIR_VISER.parent / "sessions" /
              f"{args.mode}_{'_'.join(names)}_{timestamp()}")
    (outdir / "images").mkdir(parents=True, exist_ok=True)

    print()
    print("#" * 74)
    print(f"#  Capture session  --  {args.mode}")
    print("#" * 74)
    print(f"""
  cameras : {', '.join(f'{n} ({s})' for n, s in resolved)}
  shots   : {args.shots}, interval {interval:.2f} s
  stream  : {args.width}x{args.height} @ {args.fps} fps
  output  : {outdir}
""")

    ## ---- robot pose ---- ##
    frames = bridge = listener = None
    pose = None
    if not args.no_robot:
        ensure_ros_path()
        import rospy
        from kinematics import JointFrameBridge

        frames = RobotFrames()
        if args.pose:
            from move_validation import MotionSession
            from robot_control import interpolate_to_pose
            sessions = {a: MotionSession(a, dry_run=False) for a in args.arms}
            bridge = next(iter(sessions.values())).bridge
            for a, sess in sessions.items():
                print(f"  moving {a} to '{args.pose}'...")
                interpolate_to_pose(sess.robots[a], a,
                                    frames.q_from_named_pose(
                                        args.pose, arms=(a,))[
                                        sess.joint_idx].tolist())
                sess.settle(1.0)
        else:
            if not rospy.core.is_initialized():
                rospy.init_node("giava_capture_session", anonymous=True,
                                disable_signals=True)
            bridge = JointFrameBridge(frames.robot)
        listener = JointStateListener(ARM_ORDER)
        rospy.sleep(1.0)
        if not listener.ready():
            print(f"  WARNING: no joint_states from {listener.missing()} -- "
                  f"is the driver running? Those arms will be recorded as "
                  f"NOT MEASURED.")

    if not args.no_prompt:
        print("  Set up the object and the stopwatch in view of both "
              "cameras.")
        wait_for_enter("press ENTER to start capturing... ")

    ## ---- capture ---- ##
    workers = [CameraWorker(n, args.width, args.height, args.fps,
                            global_time=True, keep_images=True,
                            history_len=12) for n in names]
    shots: List[Dict[str, Any]] = []
    last_frames: Dict[str, Optional[int]] = {n: None for n in names}
    try:
        for w in workers:
            w.start()
        time.sleep(1.5)
        _check_delivering(workers)

        if listener is not None and listener.ready():
            pose = pose_block(frames, bridge,
                              listener.q_driver(frames), names)
            print("\n  pose recorded; extrinsics recoverable for: "
                  + ", ".join(c for c, v in pose["cameras"].items()
                              if v["recoverable"]))
            rel = pose.get("relative_extrinsic")
            if rel:
                print(f"  camera-to-camera baseline: "
                      f"{rel['baseline_mm']:.1f} mm  ({rel['name']})")
        print()

        for i in range(args.shots):
            shot = take_shot(workers, require_new=(args.mode == "burst"),
                             last_frames=last_frames)
            if shot is None:
                print(f"  shot {i}: timed out waiting for fresh frames")
                continue
            for n in names:
                last_frames[n] = shot["records"][n].frame_number
            entry = {"index": i, "reference_s": shot["reference_s"],
                     "spread_s": shot["spread_s"],
                     "timestamps_s": shot["timestamps_s"], "cameras": {}}
            ## Pose PER SHOT, not once per session.  A burst spans 0.07 s
            ## and nothing moves, but a drift run spans minutes -- and for
            ## reconstruction each image needs the pose it was actually
            ## taken at, not the pose the session started in.
            if listener is not None and listener.ready():
                entry["robot_pose"] = pose_block(
                    frames, bridge, listener.q_driver(frames), names)
            for n in names:
                rec = shot["records"][n]
                fn = f"images/shot{i:04d}_{n}.png"
                cv2.imwrite(str(outdir / fn),
                            cv2.cvtColor(rec.image, cv2.COLOR_RGB2BGR))
                d = rec.timing_dict()
                d["image"] = fn
                entry["cameras"][n] = d
            shots.append(entry)
            print(f"  shot {i:3d}  spread {shot['spread_s']*1e3:6.2f} ms  "
                  + "  ".join(f"{n}#{shot['records'][n].frame_number}"
                              for n in names))
            if interval > 0:
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        for w in workers:
            w.stop()

    ## ---- report ---- ##
    stats = analyse(shots, names, args.mode, args.fps)
    if stats.get("n_shots", 0) >= 2:
        print()
        print("=" * 74)
        print(f"  {stats['n_shots']} shots over {stats['elapsed_s']:.2f} s")
        print("=" * 74)
        g = stats["inter_shot_gap_ms"]
        print(f"\n  inter-shot gap : mean {g['mean']:.2f} ms  "
              f"min {g['min']:.2f}  max {g['max']:.2f}")
        if args.mode == "burst":
            print(f"  achieved rate  : {stats['achieved_rate_hz']:.2f} Hz  "
                  f"(configured {args.fps} Hz)")
            print(f"  -> {'MATCHES' if abs(stats['achieved_rate_hz']-args.fps) < 0.15*args.fps else 'BELOW'} "
                  f"the configured frame rate")
        o = stats["offset_ms"]
        print(f"\n  offset ({stats['delta_definition']}):")
        print(f"      mean {o['mean']:+.3f} ms   std {o['std']:.3f}   "
              f"range {o['min']:+.3f} .. {o['max']:+.3f}")
        a = stats["alignment_spread_ms"]
        print(f"  alignment spread: mean {a['mean']:.3f} ms  "
              f"max {a['max']:.3f} ms")
        if stats.get("offset_drift_ms_per_s") is not None:
            print(f"\n  offset drift: {stats['offset_drift_ms_per_s']:+.6f} "
                  f"ms/s  ({stats['offset_drift_ms_per_min']:+.4f} ms/min) "
                  f"over {stats['offset_drift_window_s']:.1f} s")
            print("      A constant offset is calibratable; a nonzero slope")
            print("      means the two device clocks run at different rates.")
        else:
            print(f"\n  offset drift: not measured "
                  f"({stats['offset_drift_window_s']:.3f} s window is too "
                  f"short)")
            print("      Use --mode drift with a long --interval for this.")

    save_json({
        "metadata": provenance(
            f"capture_session_{args.mode}",
            cameras=[{"name": n, "serial": s} for n, s in resolved],
            width=args.width, height=args.height, fps=args.fps,
            mode=args.mode, interval_s=interval, pose_commanded=args.pose),
        "hardware_sync": {
            "external_sync_wired": False,
            "inter_cam_sync_mode_supported": False,
            "statement": ("D405 does not expose inter_cam_sync_mode. These "
                          "cameras free-run; all alignment is software."),
        },
        "intrinsics": intrinsics_block(workers),
        ## Session-level pose: the state at the START. Each shot carries its
        ## own 'robot_pose' too -- use THAT one per image; this is a summary.
        "robot_pose_at_session_start": pose,
        "robot_pose": pose,
        "statistics": stats,
        "shots": shots,
    }, outdir / "session.json", overwrite=args.overwrite)
    write_summary(outdir / "summary.txt", pose, intrinsics_block(workers),
                  stats, shots, names)
    print(f"\n  saved to {outdir / 'session.json'}")
    print(f"  readable summary: {outdir / 'summary.txt'}")
    print(f"  {len(shots)} shot(s), {2*len(shots)} image(s)\n")


if __name__ == "__main__":
    main()
