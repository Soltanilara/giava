"""
Code to teleoperate the arms in different modes

It used the same IK solver, camera and arm setup and control protocols as 
data collection without the additional keyboard commands to record and 
save data

We also add an option to calibrate cameras. Particularly for oak cameras
's' saves a synchronized RAW stereo pair from the OAK, at the
native sensor resolution, straight to disk.  That is the input to
`oak_stereo_calibrate.py`, so the whole checkerboard capture -> calibrate ->
rectify loop can be driven from inside the headset: park the board where you
want it with the camera arm, look at it, press 's'.

    python teleop.py                          # asks which mode
    python teleop.py --mode middle            # camera arm only
    python teleop.py --mode middle --cameras oak \
        --capture-dir ../calibration/data/oak_calib/run1

OPTIONS
    --mode left|right|middle|bimanual|av      which arms are live (default: ask)
    --camera-source head|left|right           what drives the camera arm (head)
    --cameras oak|none                        OAK stereo -> headset (default oak)
    --capture-dir PATH                        where 's' writes pairs
    --board COLSxROWS                         checkerboard INNER corners (9x6)
    --square METRES                           checkerboard square side (0.02267)
                                              MEASURE IT: a wrong value scales
                                              the baseline, and every depth,
                                              by exactly that ratio -- with no
                                              effect on any quality number
    --collision sphere|capsule|gjk            see collision_modes.py
    --table on|off                            tabletop avoidance
    --jax gpu|cpu                             solver backend, see jax_platform.py

CONTROLS
    headset  X (left one)  = left arm     A (right one) = right arm
             Y (left two)  = camera arm   triggers      = grippers
             the camera arm also follows the head whenever a gripper arm is
             being driven (GIAVA_MIDDLE_BUTTON=y to make Y the only way)

             --camera-source right  aims the camera arm with the RIGHT HAND
             instead: hold B (right two) and move your hand.  Holding your
             head still for a whole capture session is tiring, and a hand can
             be parked where a head cannot.  --camera-source left uses the
             left hand and Y, which is the button the camera arm already had.
    keyboard s  save a raw stereo pair      c  capture summary
             i  re-park the arms + resync   q  quit (parks the arms)
             e  servo health report          reboot  clear a latched fault
             profile  motion-profile check     shadow  master/shadow check
             clear  release a watchdog hold (writes nothing to the motors)
    stereo   cov  coverage map + what pose to shoot next
             cal  fit the captured pairs now and install it on the live view
             swap exchange the LEFT/RIGHT eye labels (wiring handedness)
             , . convergence   - = zoom     (headset view only)
"""

import os
import sys
import threading
import time

## Collision model selection MUST happen before study_ik and capsule_gate are
## imported below: both read their configuration from the environment at module
## import, so setting it later would silently do nothing while the startup
## banner claimed otherwise.  Same ordering constraint, same reason, as
## data_collection.py.  See collision_modes.py.
try:
    from .collision_modes import banner as _collision_banner
    from .collision_modes import select as _collision_select
    from .collision_modes import select_table as _table_select
    from .collision_modes import take_option as _take_option
    from .jax_platform import select as _jax_select
except ImportError:
    from collision_modes import banner as _collision_banner
    from collision_modes import select as _collision_select
    from collision_modes import select_table as _table_select
    from collision_modes import take_option as _take_option
    from jax_platform import select as _jax_select

COLLISION_MODE = _collision_select()
TABLE_MODE = _table_select()
JAX_PLATFORM = _jax_select()

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

# Diagnostic switches, identical in name and meaning to data_collection.py so
# a habit learned there transfers (cheap when off).
DEBUG_HEAD = os.environ.get("GIAVA_DEBUG_HEAD", "0") == "1"
DEBUG_HANDS = os.environ.get("GIAVA_DEBUG_HANDS", "0") == "1"
LOG_CLEARANCE = os.environ.get("GIAVA_LOG_CLEARANCE", "0") == "1"

# See data_collection.py: "either" couples the camera arm to whichever gripper
# arm is being driven; "y" parks it unless Y is held.
MIDDLE_BUTTON = os.environ.get("GIAVA_MIDDLE_BUTTON", "either").strip().lower()
if MIDDLE_BUTTON not in ("either", "y"):
    print(f"[config] GIAVA_MIDDLE_BUTTON={MIDDLE_BUTTON!r} is not either|y; "
          "using 'either'")
    MIDDLE_BUTTON = "either"

# See data_collection.py: OFF because on-hardware testing showed the gripper
# arms chasing head motion with composition on.
HANDS_HEAD_RELATIVE = os.environ.get("GIAVA_HANDS_HEAD_RELATIVE", "0") == "1"

## Which tracked device aims the camera arm.  "head" is the deployed behaviour
## and what the recorded datasets mean by active vision -- the gaze trajectory
## is only a record of where the operator looked if the operator's head is what
## moved it.  "left"/"right" hand it to a controller instead, which is for
## SETUP work rather than for recording: aiming the camera at a calibration
## board, or parking it somewhere a neck will not comfortably hold.
##
## Two things change with it, both further down:
##   - the activation button becomes that controller's SECOND button (B on the
##     right, Y on the left), never a gripper button, because the hand doing
##     the aiming cannot also be driving its own arm; and
##   - the head pose is still read, and still builds the session yaw remap.
##     It is the only thing that resolves the app's arbitrary per-session world
##     yaw, and a controller's heading cannot stand in for it.
CAM_SOURCE = os.environ.get("GIAVA_CAM_SOURCE", "head").strip().lower()

## Head motion is involuntary -- breathing and weight shifts move it
## continuously -- so the head defaults carry a 15 mm deadband and 0.6 scale to
## stop the camera arm chasing all of it.  A hand is deliberate: the same
## deadband makes it feel dead and the same scale makes it feel sluggish.  Only
## applied when the operator has not set the env var themselves.
CAM_CONTROLLER_DEFAULTS = {"GIAVA_CAM_POS_SCALE": "1.0",
                           "GIAVA_CAM_DEADBAND_M": "0.003"}

import cv2  # noqa: E402

from paths import ROS_DEVEL_SITE

for ros_path in (
    Path("/opt/ros/noetic/lib/python3/dist-packages"),
    ROS_DEVEL_SITE,
):
    if ros_path.is_dir():
        ros_path_str = str(ros_path)
        if ros_path_str not in sys.path:
            sys.path.append(ros_path_str)

try:
    import rospy
except ImportError:
    rospy = None

if __package__:
    from .arm_config import ARM_CONFIG, POSES, URDF_PATH
    from .camera_manager import (
        adjust_eye_view,
        install_rectify_from_npz,
        oak_stream_settings,
        oak_swap_eyes,
        request_oak_capture,
        setup_cameras,
        take_oak_capture,
    )
    from .data_col_config import (
        ARM_MODES,
        ArmTeleopState,
        CommandKinematicsState,
        LinkGuard,
        RobotCommandState,
        TeleopConfig,
        TeleopSessionState,
        anchor_arm_state,
        clamp_joint_step,
        compute_camera_arm_target,
        compute_gripper_arm_target,
        start_teleop_session,
        stop_teleop_session,
    )
    from .gripper import update_gripper, update_gripper_from_trigger
    from .headset_link import make_headset
    from .robot_control import (
        apply_profile_limits,
        build_robot_model,
        command_state_is_stale,
        compute_fk_and_ee,
        create_and_configure_robots,
        move_to_named_pose,
        move_to_named_poses,
        resolve_middle_waist_shift,
        reset_arms,
        sync_robot_state,
    )
    from .stereo_calib_live import (
        CoverageTracker,
        board_metrics,
        calibrate_now,
        disparity_sign,
        handedness_warning,
        summarise_fit,
    )
    from .servo_health import FaultRecorder, OverloadEstimator, ServoWatchdog
    from .servo_health import check_profile as servo_check_profile
    from .servo_health import decode as servo_decode
    from .servo_health import check_shadows as servo_check_shadows
    from .servo_health import read_hardware_errors as servo_read_errors
    from .servo_health import recover as servo_recover
    from .servo_health import report as servo_report
    from .study_ik import COLLISION_MARGIN, CoupledStudyIK
    from .transform_utils import pose2mat
else:
    from arm_config import ARM_CONFIG, POSES, URDF_PATH
    from camera_manager import (
        adjust_eye_view,
        install_rectify_from_npz,
        oak_stream_settings,
        oak_swap_eyes,
        request_oak_capture,
        setup_cameras,
        take_oak_capture,
    )
    from data_col_config import (
        ARM_MODES,
        ArmTeleopState,
        CommandKinematicsState,
        LinkGuard,
        RobotCommandState,
        TeleopConfig,
        TeleopSessionState,
        anchor_arm_state,
        clamp_joint_step,
        compute_camera_arm_target,
        compute_gripper_arm_target,
        start_teleop_session,
        stop_teleop_session,
    )
    from gripper import update_gripper, update_gripper_from_trigger
    from headset_link import make_headset
    from robot_control import (
        apply_profile_limits,
        build_robot_model,
        command_state_is_stale,
        compute_fk_and_ee,
        create_and_configure_robots,
        move_to_named_pose,
        move_to_named_poses,
        resolve_middle_waist_shift,
        reset_arms,
        sync_robot_state,
    )
    from stereo_calib_live import (
        CoverageTracker,
        board_metrics,
        calibrate_now,
        disparity_sign,
        handedness_warning,
        summarise_fit,
    )
    from servo_health import FaultRecorder, OverloadEstimator, ServoWatchdog
    from servo_health import check_profile as servo_check_profile
    from servo_health import decode as servo_decode
    from servo_health import check_shadows as servo_check_shadows
    from servo_health import read_hardware_errors as servo_read_errors
    from servo_health import recover as servo_recover
    from servo_health import report as servo_report
    from study_ik import COLLISION_MARGIN, CoupledStudyIK
    from transform_utils import pose2mat


HERE = Path(__file__).resolve().parent

frame_lock = threading.Lock()
camera_shutdown = threading.Event()
latest_key = None


def keyboard_listener():
    global latest_key
    while True:
        latest_key = input().strip()


def now():
    return time.monotonic()


def select_mode(default="middle"):
    """Ask which arms to drive, listing them by name rather than by index.

    Indexed menus are how you end up teleoperating 'bimanual' because the
    ordering of a dict changed; the names are what --mode takes anyway."""
    print("\nAVAILABLE MODES")
    for name, arms in ARM_MODES.items():
        print(f"  {name:<9} {', '.join(arms)}")
    while True:
        raw = input(f"\nSelect mode [{default}]: ").strip().lower()
        if not raw:
            return default
        if raw in ARM_MODES:
            return raw
        print(f"  '{raw}' is not a mode; pick one of {sorted(ARM_MODES)}")


# --------------------------------------------------------------------------- #
# Checkerboard capture
# --------------------------------------------------------------------------- #

class CaptureSession:
    """Writes the raw stereo pairs 's' asks for, and says whether they are usable.

    The detection feedback is the point.  A stereo calibration is only as good
    as the pairs where the board was found in BOTH eyes, and that is not
    something the operator can judge from inside the headset -- the board is
    perfectly visible to a human at an angle where cornerSubPix will not
    converge.  Reporting per-capture means a bad batch is discovered while the
    board is still in your hand, not forty pairs later when the calibrator
    rejects most of them.

    Detection runs on a worker thread: findChessboardCorners on a 1280x800
    pair takes ~100-300 ms, which is 5-15 control ticks at 50 Hz, and the arms
    must not stutter because the operator pressed 's'."""

    def __init__(self, root: Path, board=(9, 6)):
        self.root = root
        self.board = board
        self.root.mkdir(parents=True, exist_ok=True)
        self.saved = 0
        self.usable = 0
        self.pending = 0
        self._lock = threading.Lock()
        self.coverage = CoverageTracker()
        ## Handedness is checked ONCE, on the first pair that has the board in
        ## both eyes, and then never again: it is a property of the wiring,
        ## not of the pose, so repeating it would only add noise to a prompt
        ## the operator is reading through a headset.
        self.handedness = None          # None until measurable, then the sign
        self._handedness_done = False

    def save(self, left_bgr, right_bgr, ts):
        idx = self.saved
        self.saved += 1
        left_path = self.root / f"pair_{idx:04d}_left.png"
        right_path = self.root / f"pair_{idx:04d}_right.png"
        with self._lock:
            self.pending += 1
        threading.Thread(
            target=self._write_and_check,
            args=(idx, left_path, right_path, left_bgr, right_bgr, ts),
            daemon=True,
        ).start()
        return idx

    def _write_and_check(self, idx, left_path, right_path, left_bgr, right_bgr, ts):
        try:
            cv2.imwrite(str(left_path), left_bgr)
            cv2.imwrite(str(right_path), right_bgr)
            ok_l = self._find(left_bgr)
            ok_r = self._find(right_bgr)

            ## Coverage is judged on the LEFT eye alone.  A pair only enters
            ## the fit when the board is in both, but where the board WAS is
            ## what the operator needs told, and the left frame answers that
            ## whether or not the right one happened to detect.
            if ok_l:
                m = board_metrics(left_bgr, self.board)
                if m is not None:
                    self.coverage.add(m)

            if ok_l and ok_r and not self._handedness_done:
                self._handedness_done = True
                disp = disparity_sign(left_bgr, right_bgr, self.board)
                if disp is not None:
                    self.handedness = disp
                    if disp < 0:
                        print(handedness_warning(disp))
                    else:
                        print(f"[handedness] eye labels OK "
                              f"(board disparity {disp:+.1f} px, must be "
                              f"positive)")

            with self._lock:
                self.pending -= 1
                if ok_l and ok_r:
                    self.usable += 1
                    verdict = "OK   board in both eyes"
                elif ok_l or ok_r:
                    verdict = ("MISS board only in the LEFT eye"
                               if ok_l else
                               "MISS board only in the RIGHT eye")
                else:
                    verdict = "MISS board in neither eye"
                usable, saved = self.usable, self.saved
            h, w = left_bgr.shape[:2]
            print(f"\n[capture] pair_{idx:04d} {w}x{h}  {verdict}   "
                  f"({usable}/{saved} usable)")
            print(f"[capture] NEXT: {self.coverage.instruction()}")
        except Exception as exc:
            with self._lock:
                self.pending -= 1
            print(f"\n[capture] pair_{idx:04d} FAILED: {exc}")

    def _find(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                 | cv2.CALIB_CB_NORMALIZE_IMAGE
                 | cv2.CALIB_CB_FAST_CHECK)
        found, _ = cv2.findChessboardCorners(gray, self.board, flags)
        return bool(found)

    def summary(self):
        with self._lock:
            return (f"[capture] {self.usable}/{self.saved} usable pairs"
                    + (f", {self.pending} still checking" if self.pending else "")
                    + f"  ->  {self.root}")


# --------------------------------------------------------------------------- #
# Arms that are not being driven
# --------------------------------------------------------------------------- #

def read_parked_joints(arm_name, timeout=1.0):
    """Measured joints of an arm this session is NOT driving, or None.

    Read-only: subscribes to the driver's joint_states topic instead of
    constructing an InterbotixManipulatorXS, because constructing one torques
    the arm on -- energising an arm the operator deliberately left out of the
    mode is not something a read should do.

    Why bother.  CoupledStudyIK always solves all three arms, and the arms
    without a target hold the configuration in prev_q.  For an arm no robot
    object exists for, that configuration is whatever the vector was
    initialised to -- zeros, which is a pose the real arm is never in.  The
    self-collision cost and the capsule gate then keep the live arm away from
    a phantom while ignoring the arm that is actually there.  One joint_states
    read at startup makes the model agree with the room."""
    if rospy is None:
        return None
    try:
        from sensor_msgs.msg import JointState
        topic = f"/{ARM_CONFIG[arm_name]['robot_name']}/joint_states"
        msg = rospy.wait_for_message(topic, JointState, timeout=timeout)
        n = ARM_CONFIG[arm_name]["num_joints"]
        return np.asarray(msg.position[:n], dtype=float)
    except Exception:
        return None


def seed_inactive_arms(q, robot, arm_names):
    """Fill q for arms outside the mode, so collision geometry is honest."""
    for arm in ("left", "right", "middle"):
        if arm in arm_names:
            continue
        idx = [robot.joints.actuated_names.index(n)
               for n in ARM_CONFIG[arm]["joint_names"]]
        measured = read_parked_joints(arm)
        if measured is not None and len(measured) == len(idx):
            q[idx] = measured
            print(f"[hold] {arm} arm is not in this mode; read its measured "
                  f"pose for the collision model")
        else:
            q[idx] = np.asarray(POSES[arm]["forward"], dtype=float)
            print(f"[hold] {arm} arm is not in this mode and its driver did "
                  f"not answer; ASSUMING the forward pose for the collision "
                  f"model. If it is parked somewhere else, the gates are "
                  f"guarding the wrong geometry.")
    return q


# --------------------------------------------------------------------------- #

def main():
    if rospy is None:
        raise ImportError("rospy is required for teleoperation.")

    ## Flags are consumed before the positional check, exactly as in
    ## data_collection.py, so an unknown option is reported by name instead of
    ## surfacing later as a confusing parse error.
    mode = _take_option("mode", default=None)
    global CAM_SOURCE
    CAM_SOURCE = str(_take_option("camera-source",
                                  default=CAM_SOURCE)).strip().lower()
    cameras_opt = str(_take_option("cameras", default="oak")).strip().lower()
    capture_dir = _take_option("capture-dir", default=None)
    board_opt = str(_take_option("board", default="9x6")).strip().lower()
    ## Needed by the live `cal` command, which runs the real calibrator.
    ##
    ## MEASURE THIS, do not assume it.  The square size is the only physical
    ## scale in the entire fit: state it wrong by a factor and the recovered
    ## baseline -- and therefore every depth the operator sees -- is wrong by
    ## exactly that factor, while the RMS and the vertical-disparity verdict
    ## stay perfect.  Demonstrated on this rig 2026-08-27: the same 93 pairs
    ## fitted to 62.63 mm at --square 0.02267 and 69.07 mm at 0.025, a 10.3%
    ## error that no quality number in the output moved by.  0.02267 is the
    ## value whose baseline matches the ruler measurement behind
    ## camera_manager.OAK_EXPECTED_BASELINE_M.
    square_m = float(_take_option("square", default=0.02267))

    leftover = [a for a in sys.argv[1:] if a.startswith("-")]
    if leftover:
        raise SystemExit(
            f"unrecognised option(s): {' '.join(leftover)}\n"
            f"  usage: python teleop.py [--mode av|bimanual|left|right|middle]\n"
            f"         [--camera-source head|left|right]\n"
            f"         [--cameras oak|none] [--capture-dir PATH] "
            f"[--board COLSxROWS]\n"
            f"         [--collision sphere|capsule|gjk] [--table on|off] "
            f"[--jax gpu|cpu]")

    if cameras_opt not in ("oak", "none"):
        raise SystemExit(f"--cameras must be oak|none, got '{cameras_opt}'")
    if CAM_SOURCE not in ("head", "left", "right"):
        raise SystemExit(
            f"--camera-source must be head|left|right, got '{CAM_SOURCE}'")
    ## Applied before TeleopConfig() is built, since its scale/deadband come
    ## from the environment through default_factory.
    if CAM_SOURCE != "head":
        for k, v in CAM_CONTROLLER_DEFAULTS.items():
            os.environ.setdefault(k, v)
    try:
        board = tuple(int(v) for v in board_opt.split("x"))
        assert len(board) == 2
    except Exception:
        raise SystemExit(
            f"--board must look like 9x6 (INNER corners, i.e. squares-1 in "
            f"each direction), got '{board_opt}'")

    rospy.init_node("teleop")

    global latest_key
    cfg = TeleopConfig()

    if mode is None:
        mode = select_mode()
    mode = str(mode).strip().lower()
    if mode not in ARM_MODES:
        raise SystemExit(f"--mode must be one of {sorted(ARM_MODES)}, got '{mode}'")
    arm_names = ARM_MODES[mode]
    print(f"\nMode: {mode}  ({', '.join(arm_names)})")

    ## The button that drives the camera arm.  With a controller source it is
    ## that controller's SECOND button, never button one: button one already
    ## drives that hand's own gripper arm, and one hand cannot aim the camera
    ## and drive an arm through the same motion.
    cam_button = {"head": None, "left": "l_button_two",
                  "right": "r_button_two"}[CAM_SOURCE]
    if "middle" in arm_names:
        if CAM_SOURCE == "head":
            print(f"Camera arm: follows the HEAD "
                  f"({'Y only' if MIDDLE_BUTTON == 'y' else 'Y, or whenever a gripper arm is driven'})")
        else:
            print(f"Camera arm: follows the {CAM_SOURCE.upper()} controller "
                  f"— hold {'B' if CAM_SOURCE == 'right' else 'Y'} and move "
                  f"your hand  (scale {cfg.cam_position_scale:g}, deadband "
                  f"{cfg.cam_deadband_m * 1000:.0f} mm)")
            if CAM_SOURCE in arm_names:
                print(f"  NOTE: the {CAM_SOURCE} ARM is also live in this mode, "
                      f"on the same hand. "
                      f"{'A' if CAM_SOURCE == 'right' else 'X'} drives the arm, "
                      f"{'B' if CAM_SOURCE == 'right' else 'Y'} drives the "
                      f"camera; holding both sends one hand motion to two arms.")
    elif CAM_SOURCE != "head":
        print(f"--camera-source {CAM_SOURCE} has no effect: the middle arm is "
              f"not in mode '{mode}'.")

    if capture_dir is None:
        ## OAK checkerboard captures live with the other calibration
        ## data, not in the scripts tree.
        capture_dir = (HERE.parent / "calibration" / "data" / "oak_calib"
                       / time.strftime("%Y%m%d_%H%M%S"))
    captures = CaptureSession(Path(capture_dir), board=board)
    print(f"Capture dir: {captures.root}   board: {board[0]}x{board[1]} inner "
          f"corners, {square_m * 1000:.1f} mm squares")
    OAK_W, OAK_H, _ = oak_stream_settings()
    ## Whether a rectification is currently applied to the live stream --
    ## read by `swap`, which needs to say that an installed calibration has
    ## just been invalidated.
    OAK_RECTIFY_ACTIVE = [False]

    threading.Thread(target=keyboard_listener, daemon=True).start()

    headset = make_headset()
    headset.run_in_thread()

    ## OAK only.  The RealSense scene cameras exist to be RECORDED; nothing in
    ## a teleop session reads them, and opening them costs USB bandwidth the
    ## OAK's native 1280x800 pair needs.
    pipelines = {}
    if cameras_opt == "oak":
        pipelines = setup_cameras(
            ["oak_left", "oak_right"],
            camera_shutdown,
            frame_lock,
            {},              # no dataset frames are kept
            {},
            headset=headset,
        )
    else:
        print("[cameras] --cameras none: no video to the headset "
              "(pose tracking still works).")

    if cameras_opt == "oak":
        import camera_manager as _cm
        OAK_RECTIFY_ACTIVE[0] = _cm.OAK_RECTIFY.get("maps") is not None
        if oak_swap_eyes():
            print("[handedness] GIAVA_OAK_SWAP_EYES=1: CAM_B is being used as "
                  "the RIGHT eye.")

    robots = create_and_configure_robots(arm_names)

    ## Explain the interbotix ERROR that just scrolled past, rather than
    ## leaving it to be learned-ignored.
    ##
    ##     [ERROR] Please set the group's 'profile type' to 'time'.
    ##     ...
    ##     Drive Mode: Time-Based-Profile          <-- a hardcoded print, not a
    ##                                                 reading; arm.py always
    ##                                                 says this
    ##
    ## The mode yamls configure a VELOCITY-based profile on purpose: it is a
    ## speed/acceleration ceiling enforced in servo firmware, which an IK
    ## discontinuity cannot exceed.  arm.py only logs its objection and then
    ## writes moving_time*1000 into Profile_Velocity regardless, so those
    ## registers hold caps, not durations.  Type `profile` to read back what
    ## the motors are actually running.
    print("\n[profile] The interbotix \"set the group's profile type to "
          "'time'\" ERROR above is EXPECTED: this robot runs a velocity-based "
          "profile deliberately (a firmware speed/accel ceiling). "
          "`moving_time`/`accel_time` are therefore CAPS, not durations.")
    ## Ask the motors what they are running and make the clamp agree.
    ## Read-only; with GIAVA_PROFILE_MODE unset it reports and changes nothing.
    apply_profile_limits(robots, cfg, arm_names)

    ## Middle-waist frame, same handling and same refusal-to-guess as
    ## data_collection.py: in ext_position mode the servo ignores
    ## Homing_Offset, so a nonzero register would desynchronize the pose
    ## tables from the servo's actual frame.
    waist_shift = 0.0
    if "middle" in arm_names:
        waist_shift = resolve_middle_waist_shift(robots["middle"])

    ## The MODEL always carries all three arms even when the mode does not:
    ## the coupled solver's collision cost and both gates are only meaningful
    ## if the arms that are physically present are in the problem.
    robot, arm_data = build_robot_model("all")

    print()
    print(_collision_banner(COLLISION_MODE))
    print()

    coupled_ik = CoupledStudyIK(
        robot,
        URDF_PATH,
        ee_links={a: ARM_CONFIG[a]["ee_link"] for a in ("left", "right", "middle")},
        control_dt=cfg.control_dt,
        waist_driver_shift=waist_shift,
    )
    print("Compiling coupled IK solver (a few seconds)...")
    _t0 = now()
    coupled_ik.warmup(
        np.asarray(robot.joint_var_cls(0).default_factory(), dtype=np.float32))
    print(f"Coupled IK ready in {now() - _t0:.1f} s")

    ## Both hard gates run on the FINAL clamped command, after the solver and
    ## after every clamp -- see capsule_gate.py / table_gate.py.  They are the
    ## reason a teleop-only script is still safe to leave running.
    try:
        from .capsule_gate import build_gate as _build_capsule_gate
        from .table_gate import build_gate as _build_table_gate
    except ImportError:
        from capsule_gate import build_gate as _build_capsule_gate
        from table_gate import build_gate as _build_table_gate
    from yourdfpy import URDF as _URDF_for_gate
    _urdf_for_gates = _URDF_for_gate.load(URDF_PATH)
    capsule_gate = _build_capsule_gate(robot, _urdf_for_gates)
    table_gate = _build_table_gate(robot, _urdf_for_gates)

    ## All arms on ONE ramp rather than one arm at a time.  The old loop was
    ## serial (`blocking=True` sleeps the caller) AND pulsed (every waypoint
    ## its own accel/decel) -- see robot_control.move_arms_together.
    move_to_named_poses({a: robots[a] for a in arm_names}, "forward")

    q = np.zeros(robot.joints.num_actuated_joints, dtype=float)
    for arm_name in arm_names:
        n = ARM_CONFIG[arm_name]["num_joints"]
        q[arm_data[arm_name]["joint_indices"]] = np.asarray(
            robots[arm_name].dxl.joint_states.position[:n], dtype=float)
    seed_inactive_arms(q, robot, arm_names)

    _, ee = compute_fk_and_ee(robot, coupled_ik.driver_to_urdf(q), arm_data)

    cmd_kin = CommandKinematicsState(
        q_cmd=q.copy(),
        T_cmd={arm: ee[arm].copy() for arm in arm_data},
    )
    cmd_state = RobotCommandState(
        last_arm_cmd_time={arm: 0.0 for arm in arm_names},
        last_cmds={arm: q[arm_data[arm]["joint_indices"]].copy()
                   for arm in arm_names},
    )
    teleop_state = TeleopSessionState()

    ## Free, per-tick "is this arm still following?" check -- see
    ## servo_health.py.  A servo that latches a hardware error torques itself
    ## off and keeps accepting commands, so nothing upstream of the measured
    ## joint states can tell the difference.
    watchdog = ServoWatchdog(arm_names)
    ## Host-side model of the servo's own overload integral, plus the
    ## seconds of history that answer "what motion caused it?".  Neither
    ## costs anything per tick: both read joint_states, which is already
    ## subscribed.
    overload = OverloadEstimator(arm_names)
    faultlog = FaultRecorder(arm_names, seconds=20.0,
                             rate_hz=1.0 / cfg.control_dt)
    _overload_last_print = {arm: -10 ** 9 for arm in arm_names}
    _faulted = {arm: False for arm in arm_names}
    ## Set by the 'reboot' diagnosis, consumed by 'reboot!'.  The two-step
    ## split exists so nobody can torque a motor off without first being
    ## shown which one and getting a chance to hold the arm.
    _pending_reboot = {}

    tick_counter = 0
    _gate_last_print = [-10 ** 9]
    _table_gate_last_print = [-10 ** 9]
    _frozen_prev = {"left": None, "right": None}
    _frozen_ticks = {"left": 0, "right": 0}
    pending_capture = False
    ## Turns a gap in the headset uplink into a release rather than a resume.
    ## That matters far more over Tailscale than on a LAN -- see LinkGuard in
    ## data_col_config.py.
    link_guard = LinkGuard()

    next_tick = now()

    print()
    print("READY  --  hold X / A / Y on the controllers to drive; "
          "'s' saves a stereo pair, 'q' quits")

    while not rospy.is_shutdown():
        key = latest_key
        latest_key = None

        if key == "s":
            if cameras_opt != "oak":
                print("\n[capture] --cameras none: there is no OAK stream to "
                      "capture from.")
            elif pending_capture:
                print("\n[capture] a request is already in flight.")
            else:
                request_oak_capture()
                pending_capture = True
            continue

        if key == "c":
            print("\n" + captures.summary())
            continue

        if key == "cov":
            print("\n" + captures.coverage.report())
            continue

        if key == "swap":
            ## Exchange which OAK socket is called "left", at the source.
            ## Everything downstream -- capture files, rectify maps, dataset
            ## streams, headset eyes -- follows from that one point, so this
            ## is a single toggle rather than four consistent edits.
            now_on = oak_swap_eyes(not oak_swap_eyes())
            print(f"\n[handedness] eye labels {'SWAPPED' if now_on else 'NORMAL'} "
                  f"(CAM_B is now the {'RIGHT' if now_on else 'LEFT'} eye).")
            print("  Pairs captured BEFORE this toggle have the old labels -- "
                  "start a fresh capture dir, or fit the old ones with "
                  "`oak_stereo_calibrate.py --swap`.")
            print("  Set GIAVA_OAK_SWAP_EYES=1 to start this way next run.")
            if OAK_RECTIFY_ACTIVE[0]:
                print("  The INSTALLED rectification was fitted for the old "
                      "labels and is now on the wrong images -- recalibrate "
                      "(`cal`) before trusting the headset view.")
            continue

        if key == "cal":
            ## Fit what has been captured so far and put it on the live
            ## stream.  Runs the real calibrator in a subprocess (see
            ## stereo_calib_live.calibrate_now) so a 1-3 s fit cannot stall
            ## the 50 Hz control loop with two arms driving.
            if captures.pending:
                print(f"\n[cal] {captures.pending} pairs still being checked "
                      f"-- try again in a moment.")
                continue
            if captures.usable < 12:
                print(f"\n[cal] only {captures.usable} usable pairs. Below "
                      f"~12 the distortion coefficients are fitted from "
                      f"noise and the result is worse than no calibration.")
                print(f"[cal] {captures.coverage.instruction()}")
                continue
            print(f"\n[cal] fitting {captures.usable} usable pairs "
                  f"(subprocess; the arms keep running)...")
            ok, npz, out = calibrate_now(captures.root, board, square_m)
            if not ok:
                print("[cal] FAILED:")
                print("\n".join("    " + l for l in out.strip().splitlines()[-12:]))
                continue
            print(summarise_fit(out))
            try:
                install_rectify_from_npz(npz, OAK_W, OAK_H)
                OAK_RECTIFY_ACTIVE[0] = True
                print(f"[cal] INSTALLED live -> {npz}")
                print("[cal] look at a corner of the room: the two eyes "
                      "should fuse without vertical offset. Keep capturing "
                      "and `cal` again to compare, or run "
                      "oak_stereo_calibrate.py --install to make it permanent.")
            except Exception as exc:
                print(f"[cal] fitted, but REFUSED for the live stream: {exc}")
            continue

        if key == "i":
            stop_teleop_session(teleop_state)
            move_to_named_poses({a: robots[a] for a in arm_names}, "forward")
            # Let the final motion's joint-state message arrive before reading.
            rospy.sleep(0.05)
            sync_robot_state(robots=robots, robot=robot, arm_data=arm_data,
                             arm_names=arm_names, cmd_kin=cmd_kin,
                             cmd_state=cmd_state,
                             to_urdf=coupled_ik.driver_to_urdf)
            print("\nREADY")
            continue

        # Live stereo-comfort tuning (headset view only).
        if key == ",":
            adjust_eye_view(dinward=+0.005)
            continue
        if key == ".":
            adjust_eye_view(dinward=-0.005)
            continue
        if key == "-":
            adjust_eye_view(dscale=-0.025)
            continue
        if key == "=":
            adjust_eye_view(dscale=+0.025)
            continue

        if key == "e":
            ## Ground truth, one blocking service call per motor -- never in
            ## the hot path, only when asked or after a watchdog trip.
            for arm in arm_names:
                servo_report(robots[arm], arm)
            continue

        if key == "clear":
            ## Clears the WATCHDOG's fault flags only -- it writes nothing to
            ## any motor.  For the case the watchdog tripped on a mechanical
            ## block or a bad command reference rather than a dead servo:
            ## the registers were clean, so there is no latch to reboot, and
            ## the arm just needs to be let back into the command set.
            watchdog.clear()
            overload.energy.clear()
            for _a in arm_names:
                _faulted[_a] = False
            sync_robot_state(robots=robots, robot=robot, arm_data=arm_data,
                             arm_names=arm_names, cmd_kin=cmd_kin,
                             cmd_state=cmd_state,
                             to_urdf=coupled_ik.driver_to_urdf)
            print("\n[servo health] fault flags cleared, state resynced.")
            continue

        if key == "profile":
            ## What motion profile the servos are ACTUALLY running.  Read-only,
            ## no motion.  Drive_Mode bit 2 decides whether moving_time and
            ## accel_time mean milliseconds or velocity-profile units, and
            ## nothing upstream notices when it disagrees with the code.
            for arm in arm_names:
                servo_check_profile(robots[arm], arm)
            continue

        if key == "shadow":
            ## Master vs shadow: position agreement and whether the pair is
            ## fighting.  Read-only, no motion.  Best run with the arms at
            ## rest, which is when a fighting pair is unambiguous.
            for arm in arm_names:
                servo_check_shadows(robots[arm], arm)
            continue

        if key == "reboot":
            ## STEP 1 of 2 -- diagnose only.  Nothing is written here.
            ##
            ## A reboot takes the faulted motors' torque OFF for about half a
            ## second.  A motor CANNOT hold position while it reboots: the arm
            ## sags, and if a shoulder is involved it drops.  So this prints
            ## exactly which motors go limp and waits to be told again.
            _pending_reboot = {}
            for arm in arm_names:
                _errs = servo_read_errors(robots[arm], arm)
                _bad = sorted(n for n, v in _errs.items() if v)
                if _bad:
                    _pending_reboot[arm] = {n: _errs[n] for n in _bad}
            if not _pending_reboot:
                print("\n[servo health] no latched faults on any arm -- "
                      "nothing to reboot.")
                continue
            print("\n[servo health] THESE MOTORS WILL LOSE TORQUE for ~0.5 s:")
            for arm, _bad in _pending_reboot.items():
                for name, _bits in _bad.items():
                    ## Shadows are rebooted with their master: two motors on
                    ## one output must not come back from different states.
                    _mate = ""
                    for _sh in ("shoulder_shadow", "elbow_shadow"):
                        _ms = _sh.replace("_shadow", "")
                        if name in (_sh, _ms):
                            _mate = f"  (+ its pair: {_sh if name == _ms else _ms})"
                    print(f"    {arm:<8s} {name:<18s} "
                          f"{servo_decode(_bits)}{_mate}")
            print("  HOLD THE ARM NOW if any of these carries its weight "
                  "(shoulder, elbow, waist).  The healthy motors stay torqued "
                  "and carry what they can; for a shoulder fault that is "
                  "nothing.\n"
                  "  Type  reboot!  to proceed, anything else to cancel.")
            continue

        if key == "reboot!":
            ## STEP 2 of 2 -- the write.
            if not _pending_reboot:
                print("\n[servo health] run  reboot  first: it names the "
                      "motors that will go limp so you can hold the arm.")
                continue
            _pending_reboot = {}
            stop_teleop_session(teleop_state)
            for arm in arm_names:
                if servo_recover(robots[arm], arm, cfg.moving_time,
                                 cfg.accel_time):
                    _faulted[arm] = False
                    watchdog.clear(arm)
            rospy.sleep(0.1)
            sync_robot_state(robots=robots, robot=robot, arm_data=arm_data,
                             arm_names=arm_names, cmd_kin=cmd_kin,
                             cmd_state=cmd_state,
                             to_urdf=coupled_ik.driver_to_urdf)
            print("\nREADY  (state resynced to the recovered pose; press 'i' "
                  "to re-park)")
            continue

        if key == "q":
            print("\nEXITING")
            stop_teleop_session(teleop_state)
            camera_shutdown.set()
            try:
                reset_arms(robots)
            except Exception as exc:
                print(f"[shutdown] could not park the arms: {exc}")
            ## setup_cameras returns {"oak": (pipeline, q_left, q_right),
            ## "realsense": {name: pipeline}} -- neither value is itself
            ## stoppable, which is why this walks into them rather than
            ## calling .stop() on whatever comes out of .values().
            oak = (pipelines or {}).get("oak")
            if oak is not None:
                try:
                    oak[0].stop()
                except Exception:
                    pass
            for rs_pipeline in ((pipelines or {}).get("realsense") or {}).values():
                try:
                    rs_pipeline.stop()
                except Exception:
                    pass
            # Detection threads may still be running; they hold the only
            # report on whether the last pairs are usable.
            for _ in range(50):
                if captures.pending == 0:
                    break
                time.sleep(0.1)
            print(captures.summary())
            break

        ## Serve a capture the moment the worker has filled it.  Polled here
        ## rather than written from the camera thread so the file I/O and the
        ## printout stay on this side of the fence -- the OAK worker's only
        ## job is to hand over the pair it already had in hand.
        if pending_capture:
            got = take_oak_capture(frame_lock)
            if got is not None:
                pending_capture = False
                captures.save(*got)

        tick_counter += 1

        ## Costs nothing: joint_states already carries position AND effort.
        ## A trip means the arm stopped following its command -- the only
        ## symptom a torqued-off servo has, since the driver keeps accepting
        ## and publishing commands to a limp motor exactly as it would to a
        ## live one.
        ## Feed the overload model and the pre-fault buffer.  Both live off
        ## joint_states, so this is arithmetic on data already in hand.
        for _arm in arm_names:
            try:
                _js = robots[_arm].dxl.joint_states
                _n = ARM_CONFIG[_arm]["num_joints"]
                _meas = np.asarray(_js.position[:_n], dtype=float)
                _eff = np.asarray(_js.effort[:_n], dtype=float)
            except Exception:
                continue
            if watchdog.dead_bus.get(_arm):
                continue          # the readings are the failed-read sentinel
            _cmd = np.asarray(cmd_state.last_cmds.get(_arm, _meas), dtype=float)
            faultlog.push(_arm, now(), _cmd, _meas, _eff)
            _e = overload.update(_arm, _eff, cfg.control_dt)
            ## Say it ONCE per crossing, not per tick: the derate is
            ## continuous and silent, and a line every 20 ms would bury the
            ## thing it is warning about.
            if _e > 0.35 and (tick_counter - _overload_last_print[_arm]) > 250:
                _overload_last_print[_arm] = tick_counter
                print(f"[overload] {overload.describe(_arm)}")

        for _arm in watchdog.update(robots, cmd_state, now()):
            _faulted[_arm] = True
            print(f"\n[servo health] {watchdog.describe(_arm)}")
            if watchdog.dead_bus[_arm]:
                ## Nothing to read, nothing to reboot.  Say so once and stop:
                ## the register poll below would only print the same thing
                ## again, one service timeout per motor.
                print(f"  Holding '{_arm}'.  Check 12 V power, the U2D2/USB "
                      f"cable, and `ls -l /dev/ttyDXL*`, then power-cycle "
                      f"that arm and restart.  A software reboot cannot "
                      f"reach a motor that is not on the bus.")
                continue
            print(f"  Holding '{_arm}' -- no further commands will be sent to "
                  f"it.  Reading its error registers...")
            _p = faultlog.dump(_arm, reason="watchdog trip",
                               extra={"overload": overload.worst(_arm)[0]})
            if _p is not None:
                print(f"  wrote the {20.0:.0f} s BEFORE the fault to {_p}")
                print(FaultRecorder.summarise(_p))
            _errs, _bad = servo_report(robots[_arm], _arm)
            if not _bad:
                print(f"  No latched fault: the registers are clean, so this "
                      f"is a MECHANICAL block or a bad command reference, not "
                      f"a dead servo.  Press 'i' to re-park, or type 'clear' "
                      f"to release the hold.")
        if tick_counter % 50 == 0:
            for _arm, _held in watchdog.hot_arms(now()):
                ## Overload shutdown is a latched integral of current over
                ## time, so this is the only warning that arrives BEFORE the
                ## motor torques itself off.  Backed off exponentially by
                ## hot_arms(); a standing condition should not scroll the
                ## terminal.
                print(f"[servo health] {_arm} has been drawing "
                      f"{watchdog.effort_warn_ma:.0f}+ mA for {_held:.0f} s -- "
                      f"it is working toward an overload latch. Ease off, or "
                      f"check what it is pushing against.")

        headset_data = headset.receive_data()
        if headset_data is None:
            ## Holding the arms is right; leaving the session anchored to a
            ## pre-gap reference is not.  LinkGuard remembers when this started.
            link_guard.missed(now())
            next_tick += cfg.control_dt
            sleep_time = next_tick - now()
            if sleep_time > 0:
                time.sleep(sleep_time)
            continue

        link_guard.recovered(now(), teleop_state)

        head_mat_raw = pose2mat(headset_data.h_pos, headset_data.h_quat)
        head_mat = head_mat_raw
        if HANDS_HEAD_RELATIVE:
            # See data_collection.py: compose with the RAW head pose; the
            # y-up basis fix measured up to 810 mm of fake hand motion.
            controller_poses = {
                "right": head_mat_raw @ pose2mat(headset_data.r_pos, headset_data.r_quat),
                "left": head_mat_raw @ pose2mat(headset_data.l_pos, headset_data.l_quat),
                "middle": head_mat,
            }
        else:
            controller_poses = {
                "right": pose2mat(headset_data.r_pos, headset_data.r_quat),
                "left": pose2mat(headset_data.l_pos, headset_data.l_quat),
                "middle": head_mat,
            }

        ## The camera arm reads its target from controller_poses["middle"], so
        ## redirecting it is exactly this one assignment.  head_mat is kept
        ## separately and still passed to every anchor below: it is what builds
        ## the session yaw remap, and no controller can substitute for it.
        if CAM_SOURCE != "head":
            controller_poses["middle"] = controller_poses[CAM_SOURCE]

        if DEBUG_HANDS and (tick_counter % 25 == 0):
            print(f"[hands] h={np.round(np.asarray(headset_data.h_pos), 3)} "
                  f"l={np.round(np.asarray(headset_data.l_pos), 3)} "
                  f"r={np.round(np.asarray(headset_data.r_pos), 3)}")

        ## A controller that has fallen asleep keeps reporting its last pose
        ## bit-for-bit; the probe measured 0.5 m of drift when it re-acquires.
        for _side, _p in (("left", headset_data.l_pos),
                          ("right", headset_data.r_pos)):
            _pv = np.asarray(_p, dtype=float).copy()
            if _frozen_prev[_side] is not None and np.array_equal(_pv, _frozen_prev[_side]):
                _frozen_ticks[_side] += 1
            else:
                _frozen_ticks[_side] = 0
            _frozen_prev[_side] = _pv
            if _frozen_ticks[_side] == 25:  # ~0.5 s bit-identical
                print(f"[tracking] {_side} controller pose is FROZEN -- asleep "
                      "or untracked. Wake it before enabling that arm.")

        cam_button_held = (headset_data.l_button_two if cam_button == "l_button_two"
                           else headset_data.r_button_two if cam_button == "r_button_two"
                           else headset_data.l_button_two)
        button_pressed = (headset_data.r_button_one or headset_data.l_button_one
                          or cam_button_held)

        arm_active = {
            "left": headset_data.l_button_one,
            "right": headset_data.r_button_one,
        }
        for _arm, _side, _pos in (("left", "left", headset_data.l_pos),
                                  ("right", "right", headset_data.r_pos)):
            # exact zeros = the headset lost this controller -- hold the arm
            # rather than chase (0,0,0)
            if arm_active.get(_arm) and float(np.linalg.norm(
                    np.asarray(_pos, dtype=float))) < 1e-6:
                arm_active[_arm] = False
                if tick_counter % 50 == 0:
                    print(f"[tracking] {_side} controller reads zeros (LOST) -- "
                          f"{_arm} arm held")
                continue
            if arm_active.get(_arm) and _frozen_ticks.get(_side, 0) > 25:
                arm_active[_arm] = False
                if tick_counter % 50 == 0:
                    print(f"[tracking] {_side} controller frozen -- {_arm} arm "
                          "held (wiggle the controller)")

        ## Decided from the arms that actually ended up driving, not the raw
        ## buttons: a dead controller must not keep the camera following the
        ## head after it has lost the right to drive its own arm.
        if CAM_SOURCE == "head":
            arm_active["middle"] = (
                cam_button_held if MIDDLE_BUTTON == "y" else
                (arm_active["left"] or arm_active["right"] or cam_button_held))
        else:
            ## No coupling to the gripper arms here.  "the camera follows
            ## whenever a hand is working" is a statement about the operator's
            ## HEAD already looking where the hands are; it is meaningless when
            ## a hand is doing the aiming, and it would hand the camera arm a
            ## pose the operator is using to drive a gripper.
            arm_active["middle"] = cam_button_held
            ## Same frozen/lost-tracking refusal the gripper arms get: a
            ## controller asleep or out of volume keeps reporting its last pose
            ## bit-for-bit, and the probe measured 0.5 m of drift on
            ## re-acquire.  A camera arm that lurches half a metre while the
            ## operator is aiming at a calibration board is the same fault as a
            ## gripper doing it, and previously only the gripper arms were
            ## guarded because only they were controller-driven.
            _cam_pos = (headset_data.r_pos if CAM_SOURCE == "right"
                        else headset_data.l_pos)
            if arm_active["middle"] and (
                    float(np.linalg.norm(np.asarray(_cam_pos, dtype=float))) < 1e-6
                    or _frozen_ticks.get(CAM_SOURCE, 0) > 25):
                arm_active["middle"] = False
                if tick_counter % 50 == 0:
                    print(f"[tracking] {CAM_SOURCE} controller frozen or lost -- "
                          "camera arm held (wiggle the controller)")

        ## Last word on whether anything may move this tick: the latch a long
        ## uplink gap leaves behind.  Placed AFTER the frozen
        ## and lost-controller checks so a veto here cannot be undone below.
        button_pressed, arm_active = link_guard.apply(
            now(), headset_data, button_pressed, arm_active)

        for arm in arm_names:
            if arm not in teleop_state.arms:
                teleop_state.arms[arm] = ArmTeleopState()
            was_active = teleop_state.arms[arm].active
            if arm_active[arm] and not was_active and teleop_state.active:
                # Activated mid-session: re-anchor, else the arm jumps by
                # everything the controller/head moved since session start.
                anchor_arm_state(
                    teleop_state.arms[arm], controller_poses[arm],
                    cmd_kin.T_cmd[arm],
                    ## head_mat, NOT controller_poses["middle"] -- with
                    ## --camera-source those differ, and the session yaw remap
                    ## is only meaningful from the head's gaze.
                    head_pose=head_mat,
                    base_remap=(cfg.R_cam_remap if arm == "middle"
                                else cfg.R_arm_remap),
                )
            teleop_state.arms[arm].active = arm_active[arm]

        if button_pressed and not teleop_state.active:
            ## Re-anchor unconditionally at every enable: any motion outside
            ## this loop desynchronizes three references at once (our
            ## last_cmds, our T_cmd, and the DRIVER's joint_commands, whose
            ## velocity check compares against its last ACCEPTED command).
            if command_state_is_stale(robots=robots, arm_data=arm_data,
                                      arm_names=arm_names, cmd_state=cmd_state,
                                      tolerance=0.08):
                print("Resynchronizing teleoperation state with measured joints.")
            sync_robot_state(robots=robots, robot=robot, arm_data=arm_data,
                             arm_names=arm_names, cmd_kin=cmd_kin,
                             cmd_state=cmd_state,
                             to_urdf=coupled_ik.driver_to_urdf)
            # Report only: starting with the grippers close is normal, and the
            # gate holds the arms until they separate.
            if capsule_gate is not None:
                capsule_gate.validate(coupled_ik.driver_to_urdf(cmd_kin.q_cmd),
                                      where="teleop enable")
            if table_gate is not None:
                table_gate.validate(coupled_ik.driver_to_urdf(cmd_kin.q_cmd),
                                    where="teleop enable")
            start_teleop_session(teleop_state, mode, controller_poses, cmd_kin,
                                 cfg=cfg, head_pose=head_mat)
            print("\nTeleop ENABLED")
        elif (not button_pressed) and teleop_state.active:
            stop_teleop_session(teleop_state)
            print("\nTeleop DISABLED")

        ## Same dispatcher as data_collection.py so the debug tool FEELS like
        ## a recording session: GIAVA_GRIPPER_MODE=analog maps the trigger's
        ## deflection to aperture, default stays binary.
        if "left" in arm_names:
            update_gripper_from_trigger(robots["left"], headset_data.l_index_trigger)
        if "right" in arm_names:
            update_gripper_from_trigger(robots["right"], headset_data.r_index_trigger)

        targets = {}
        for arm in arm_names:
            arm_state = teleop_state.arms[arm]
            if not arm_state.active:
                continue
            controller_pose = controller_poses[arm]
            if arm == "middle":
                target_pos, target_wxyz = compute_camera_arm_target(
                    cfg, arm_state, controller_pose, cmd_kin.T_cmd[arm],
                    cfg.R_cam_remap)
                if DEBUG_HEAD and (tick_counter % 25 == 0):
                    _dh = controller_pose[:3, 3] - arm_state.start_controller_pos
                    _rm = (arm_state.session_remap
                           if arm_state.session_remap is not None
                           else cfg.R_cam_remap)
                    _dr = _rm @ _dh
                    _cmd = np.asarray(target_pos) - arm_state.start_robot_pos
                    _ach = (np.asarray(cmd_kin.T_cmd[arm][4:])
                            - arm_state.start_robot_pos)
                    print(f"[head] d_raw={np.round(_dh, 3)} "
                          f"d_robot={np.round(_dr, 3)} cmd={np.round(_cmd, 3)} "
                          f"achieved={np.round(_ach, 3)} "
                          f"(x=op-left, y=op-back, z=up; cmd<<d_robot = starved)")
            else:
                target_pos, target_wxyz = compute_gripper_arm_target(
                    cfg, arm_state, controller_pose, cmd_kin.T_cmd[arm],
                    cfg.R_arm_remap)
            targets[arm] = (target_pos, target_wxyz)

        if teleop_state.active:
            t_solve_start = now()
            due_arms = [
                arm for arm in arm_names
                if teleop_state.arms[arm].active
                and not _faulted[arm]
                and (t_solve_start - cmd_state.last_arm_cmd_time[arm]) >= cfg.arm_cmd_dt
                and arm in targets
            ]

            if due_arms:
                q_new = coupled_ik.solve(
                    cmd_kin.q_cmd, {arm: targets[arm] for arm in due_arms})

                if LOG_CLEARANCE and (tick_counter % 25 == 0):
                    _clr = coupled_ik.min_clearance(q_new)
                    if _clr < COLLISION_MARGIN + 0.02:
                        state = ("INSIDE MARGIN" if _clr < COLLISION_MARGIN
                                 else "near")
                        print(f"[collision] min sphere clearance "
                              f"{_clr * 1e3:+.1f} mm ({state}, margin "
                              f"{COLLISION_MARGIN * 1e3:.0f} mm)")

                pending_cmds = []  # (arm, joint_idx, q_arm_cmd, prev_cmd)
                for arm in due_arms:
                    joint_idx = arm_data[arm]["joint_indices"]
                    target_q = np.asarray(q_new[joint_idx], dtype=float)
                    prev_cmd = cmd_state.last_cmds.get(arm,
                                                       cmd_kin.q_cmd[joint_idx])

                    # Post-solve step clamp -- off by default, to match the
                    # ik_study conditions (TeleopConfig.enable_joint_clamp).
                    if cfg.enable_joint_clamp:
                        max_step = (cfg.max_joint_step_middle if arm == "middle"
                                    else cfg.max_joint_step)
                        q_arm_cmd = clamp_joint_step(prev_cmd, target_q, max_step)
                    else:
                        q_arm_cmd = target_q

                    ## Driver-feasibility clamp.  interbotix rejects the ENTIRE
                    ## group command if any joint would exceed its position or
                    ## velocity limit, so one infeasible joint freezes the whole
                    ## arm -- it refuses, it does not clip.
                    if cfg.enable_driver_clamp:
                        lo = np.asarray(robots[arm].arm.group_info.joint_lower_limits,
                                        dtype=float)[:len(prev_cmd)]
                        hi = np.asarray(robots[arm].arm.group_info.joint_upper_limits,
                                        dtype=float)[:len(prev_cmd)]
                        ## Scaled by the overload model: a joint climbing
                        ## toward the servo's own latch gets a smaller step,
                        ## which bleeds energy out of exactly the joint that
                        ## is accumulating it while leaving the operator in
                        ## control.  1.0 unless the arm is genuinely working;
                        ## it decays back on its own once eased off, so
                        ## nothing has to be reset.
                        step = float(cfg.driver_max_step) * overload.derate(arm)
                        # Clamp against the DRIVER's reference, not ours: it
                        # validates against its last ACCEPTED command, and if
                        # one of ours was rejected the two have diverged.
                        ref = prev_cmd
                        getter = getattr(robots[arm].arm, "get_joint_commands", None)
                        if getter is not None:
                            try:
                                ref = np.asarray(getter(), dtype=float)[:len(prev_cmd)]
                            except Exception:
                                ref = prev_cmd
                        q_arm_cmd = np.clip(q_arm_cmd, ref - step, ref + step)
                        q_arm_cmd = np.clip(q_arm_cmd,
                                            lo + cfg.driver_limit_margin,
                                            hi - cfg.driver_limit_margin)

                    ## COMPUTE phase ends here -- nothing has been sent.  The
                    ## gates below need the assembled command of ALL due arms:
                    ## the clamps just modified what the solver certified, and
                    ## two arms each individually fine can still be about to
                    ## meet each other.
                    pending_cmds.append((arm, joint_idx, q_arm_cmd, prev_cmd))

                ## HARD INTER-ARM GATE.  The step is SCALED, not refused:
                ## send the largest fraction of this tick's motion whose whole
                ## swept segment clears the margin, so the arms slide up to the
                ## boundary and stop there however hard the operator pushes.
                ## Uniform across arms on purpose -- scaling only the offending
                ## pair would redirect the motion into a path nobody commanded.
                if capsule_gate is not None and pending_cmds:
                    q_prev_full = cmd_kin.q_cmd.copy()
                    q_target_full = q_prev_full.copy()
                    for _arm, _idx, _q, _ in pending_cmds:
                        q_target_full[_idx] = _q
                    _alpha, _dist, _pair = capsule_gate.largest_safe_fraction(
                        coupled_ik.driver_to_urdf(q_prev_full),
                        coupled_ik.driver_to_urdf(q_target_full))
                    if _alpha >= 1.0:
                        pass
                    elif _alpha <= 0.0:
                        if (tick_counter - _gate_last_print[0]) >= 25:  # ~2 Hz
                            ## Two different situations, and the operator
                            ## needs to be told which: "you are driving into
                            ## the margin" vs "you are INSIDE it and this
                            ## step goes deeper" (escape mode -- any step
                            ## that opens the pair up WILL be sent).
                            _why = ("already inside it -- this step would "
                                    "close the gap further; move the "
                                    "controllers APART and the arms will move"
                                    if capsule_gate.last_escape
                                    else "move the controllers apart")
                            _gate_last_print[0] = tick_counter
                            ## The pair's OWN margin -- fingertip, gripper
                            ## and camera pairs do not use the structural
                            ## one, and printing a distance against the
                            ## wrong threshold is worse than printing none.
                            _m = capsule_gate.margin_for_pair(_pair)
                            print(f"[capsule gate] HOLDING: {_pair[0]} <-> "
                                  f"{_pair[1]} at {_dist * 1e3:+.1f} mm "
                                  f"(margin {_m * 1e3:.0f} mm). "
                                  f"{_why}.")
                        pending_cmds = []
                    else:
                        ## Interpolate in DRIVER space by the same alpha:
                        ## driver->urdf is affine per joint, so the fraction
                        ## checked is exactly the fraction sent.
                        pending_cmds = [
                            (_arm, _idx,
                             q_prev_full[_idx] + _alpha * (_q - q_prev_full[_idx]),
                             _prev)
                            for _arm, _idx, _q, _prev in pending_cmds]
                        if (tick_counter - _gate_last_print[0]) >= 25:
                            _gate_last_print[0] = tick_counter
                            print(f"[capsule gate] limiting step to "
                                  f"{_alpha * 100:.0f}%: {_pair[0]} <-> "
                                  f"{_pair[1]} stopping at {_dist * 1e3:+.1f} mm "
                                  f"(margin "
                                  f"{capsule_gate.margin_for_pair(_pair) * 1e3:.0f} mm)")

                ## HARD TABLE-FLOOR GATE, run on whatever the inter-arm gate
                ## left standing -- chaining two shrink-only scalers can only
                ## leave the command more conservative than either alone.
                if table_gate is not None and pending_cmds:
                    q_prev_full = cmd_kin.q_cmd.copy()
                    q_target_full = q_prev_full.copy()
                    for _arm, _idx, _q, _ in pending_cmds:
                        q_target_full[_idx] = _q
                    _alpha, _dist, _link = table_gate.largest_safe_fraction(
                        coupled_ik.driver_to_urdf(q_prev_full),
                        coupled_ik.driver_to_urdf(q_target_full))
                    if _alpha >= 1.0:
                        pass
                    elif _alpha <= 0.0:
                        if (tick_counter - _table_gate_last_print[0]) >= 25:
                            _table_gate_last_print[0] = tick_counter
                            _why = ("already BELOW it -- this step goes "
                                    "lower; command it UP and the arms will "
                                    "move"
                                    if table_gate.last_escape
                                    else "move away from the table")
                            print(f"[table gate] HOLDING: {_link} at "
                                  f"{_dist * 1e3:+.1f} mm above the table "
                                  f"(margin {table_gate.margin * 1e3:.0f} mm). "
                                  f"{_why}.")
                        pending_cmds = []
                    else:
                        pending_cmds = [
                            (_arm, _idx,
                             q_prev_full[_idx] + _alpha * (_q - q_prev_full[_idx]),
                             _prev)
                            for _arm, _idx, _q, _prev in pending_cmds]
                        if (tick_counter - _table_gate_last_print[0]) >= 25:
                            _table_gate_last_print[0] = tick_counter
                            _mode = ("ESCAPING" if table_gate.last_escape
                                     else "limiting step to")
                            print(f"[table gate] {_mode} "
                                  f"{_alpha * 100:.0f}%: {_link} at "
                                  f"{_dist * 1e3:+.1f} mm above the table")

                for arm, joint_idx, q_arm_cmd, prev_cmd in pending_cmds:
                    robots[arm].arm.set_joint_positions(
                        q_arm_cmd.tolist(),
                        moving_time=cfg.moving_time,
                        accel_time=cfg.accel_time,
                        blocking=False,
                    )
                    cmd_kin.q_cmd[joint_idx] = q_arm_cmd
                    cmd_state.last_cmds[arm] = q_arm_cmd.copy()
                    cmd_state.last_arm_cmd_time[arm] = now()

            _, ee_cmd = compute_fk_and_ee(
                robot, coupled_ik.driver_to_urdf(cmd_kin.q_cmd), arm_data)
            for arm in arm_names:
                cmd_kin.T_cmd[arm] = ee_cmd[arm]

        # sleep to reduce drift
        next_tick += cfg.control_dt
        sleep_time = next_tick - now()
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
