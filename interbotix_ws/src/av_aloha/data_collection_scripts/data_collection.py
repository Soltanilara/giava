import os
import json

import sys
import time
import threading

## Collision model selection MUST happen before study_ik and capsule_gate
## are imported below: both read their configuration from the environment
## at module import, so setting it later would silently do nothing while
## the startup banner claimed otherwise. See collision_modes.py.
try:
    from .collision_modes import banner as _collision_banner
    from .collision_modes import select as _collision_select
    from .collision_modes import select_table as _table_select
    from .jax_platform import select as _jax_select
    from .oak_preview import select as _preview_select
    from .collision_modes import take_option as _take_option
except ImportError:
    from collision_modes import banner as _collision_banner
    from collision_modes import select as _collision_select
    from collision_modes import select_table as _table_select
    from jax_platform import select as _jax_select
    from oak_preview import select as _preview_select
    from collision_modes import take_option as _take_option
COLLISION_MODE = _collision_select()
TABLE_MODE = _table_select()
## `--preview [left|right|both]` / GIAVA_OAK_PREVIEW, resolved HERE for the
## same reason as the switches above: the flag has to be consumed before the
## positional episode-index parse below, which rejects anything flag-shaped.
## Off by default; see oak_preview.py for what it costs when it is on.
PREVIEW_MODE = _preview_select()
## Which backend the coupled solver runs on.  GPU-first: the study's CPU
## finding was measured on the BARE pose solver, and the deployed one carries
## the 180-sphere collision cost and the table term on top.  `--jax cpu`
## restores the studied configuration.  See jax_platform.py.
JAX_PLATFORM = _jax_select()
from pathlib import Path

import numpy as np

# Diagnostic switches (cheap when off).
LOG_CLEARANCE = os.environ.get("GIAVA_LOG_CLEARANCE", "0") == "1"
DEBUG_HEAD = os.environ.get("GIAVA_DEBUG_HEAD", "0") == "1"
DEBUG_HANDS = os.environ.get("GIAVA_DEBUG_HANDS", "0") == "1"
TRACK_LOG = os.environ.get("GIAVA_TRACK_LOG", "0") == "1"
# Middle-arm activation.
#   "either" (default) -- the camera arm follows the head whenever EITHER
#       gripper arm is being driven (X or A held), and also on its own with Y.
#       Active vision is a property of the task, not a fourth thing to hold:
#       the operator's head is already looking where the hands are working, so
#       coupling the camera to "am I teleoperating at all" is what makes the
#       recorded gaze trajectory match the manipulation it belongs to.
#   "y" -- camera arm ONLY while Y (left button two) is held.  Use this when
#       deliberately parking the camera, or to isolate head-motion effects
#       while debugging the gripper arms.
# When a frame starts being written into the episode.
#
#   "teleop" (default) -- not until teleop has been enabled at least once in
#       this episode.  Recording begins on the 'r' key, but the operator then
#       has to put the headset back on and squeeze a button, and every frame
#       in between is the arm PARKED AT THE RESET POSE with action == reset
#       pose.  A few seconds of that per episode makes the reset pose by far
#       the most common action in the dataset, at near-zero variance, paired
#       with observations of a static scene -- which is exactly the recipe for
#       a policy that snaps precisely back to the reset pose whenever the
#       scene stops changing.  Nothing "leaked" from the reset AFTER the save;
#       the reset pose was in the episodes from the front all along.
#   "all" -- record every frame from 'r' onward (the historical behaviour).
#
# The tail is deliberately NOT trimmed: frames after the operator releases the
# buttons are a hold at the FINAL pose, which is a reasonable thing for a
# policy to learn, and cutting them would need lookahead this loop does not
# have.
RECORD_GATE = os.environ.get("GIAVA_RECORD_GATE", "teleop").strip().lower()
if RECORD_GATE not in ("teleop", "all"):
    print(f"[config] GIAVA_RECORD_GATE={RECORD_GATE!r} is not teleop|all; "
          "using 'teleop'")
    RECORD_GATE = "teleop"
MIDDLE_BUTTON = os.environ.get("GIAVA_MIDDLE_BUTTON", "either").strip().lower()
if MIDDLE_BUTTON not in ("either", "y"):
    print(f"[config] GIAVA_MIDDLE_BUTTON={MIDDLE_BUTTON!r} is not either|y; "
          "using 'either'")
    MIDDLE_BUTTON = "either"
# Default OFF (2026-08, revised): on-hardware testing with the debug tool
# showed the gripper arms HIGHLY affected by head motion with composition ON,
# and behaving correctly with raw streams.  (An earlier stand-up test pointed
# the opposite way -- the evidence conflicts, so the switch remains:
# GIAVA_HANDS_HEAD_RELATIVE=1 restores composition if arms ever track head
# height again.)
HANDS_HEAD_RELATIVE = os.environ.get("GIAVA_HANDS_HEAD_RELATIVE", "0") == "1"

from scipy.spatial.transform import Rotation as R

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
    import jaxlie
except ImportError:
    jaxlie = None
try:
    import rospy
except ImportError:
    rospy = None

if __package__:
    from .headset_link import make_headset
    from .arm_config import ARM_CONFIG, POSES, URDF_PATH
    try:
        from .study_ik import COLLISION_MARGIN, CoupledStudyIK
    except ImportError:
        CoupledStudyIK = None
    from .oak_preview import OakPreview
    from .waist_travel import WaistTravel
    from .camera_manager import (
        adjust_eye_view,
        SYNC_FRAMES,
        select_synchronized_frames,
        CameraConfig,
        CAMERA_SERIALS,
        CAMERA_INTRINSICS,
        REQUIRE_CALIBRATION,
        save_camera_intrinsics,
        setup_cameras,
        get_active_cameras,
        join_camera_workers,
    )
    from .robot_control import (
        apply_profile_limits,
        create_and_configure_robots,
        build_robot_model,
        quiet_solver_logs,
        solver_log_summary,
        get_pose,
        move_to_named_pose,
        move_to_named_poses,
        resolve_middle_waist_shift,
        reset_arms,
        compute_fk_and_ee,
        sync_robot_state,
        command_state_is_stale
    )
    from .servo_health import (
        FaultRecorder,
        OverloadEstimator,
        StallGate,
        ServoWatchdog,
    )
    from .servo_health import check_profile as servo_check_profile
    from .servo_health import check_shadows as servo_check_shadows
    from .servo_health import decode as servo_decode
    from .servo_health import read_hardware_errors as servo_read_errors
    from .servo_health import recover as servo_recover
    from .servo_health import report as servo_report
    from .gripper import (GRIPPER_MODE, GRIPPER_CLOSED, GRIPPER_OPEN,
                              command_gripper, update_gripper,
                              update_gripper_from_trigger)
    from .dataset import (
        BackgroundEpisodeSaver,
        quiet_libav,
        create_dataset,
        resolve_resume_run,
        build_frame,
    )
    from .data_col_config import (
        anchor_arm_state,
        session_yaw_remap,
        TASKS,
        ARM_MODES,
        ArmTeleopState,
        TeleopConfig,
        TeleopSessionState,
        RobotCommandState,
        CommandKinematicsState,
        LinkGuard,
        compute_camera_arm_target,
        compute_gripper_arm_target,
        start_teleop_session,
        stop_teleop_session,
        clamp_joint_step,
        ControllerJumpGate,
        MAX_ENGAGE_S,
    )
    from .log import (
        SessionStats,
        reset_episode_log,
        log_episode_info,
        write_episode_robustness,
        write_session_config,
    )
else:
    from headset_link import make_headset

    from arm_config import ARM_CONFIG, POSES, URDF_PATH
    try:
        from study_ik import COLLISION_MARGIN, CoupledStudyIK
    except ImportError:
        CoupledStudyIK = None
    from oak_preview import OakPreview
    from waist_travel import WaistTravel
    from camera_manager import (
        adjust_eye_view,
        SYNC_FRAMES,
        select_synchronized_frames,
        CameraConfig,
        CAMERA_SERIALS,
        CAMERA_INTRINSICS,
        REQUIRE_CALIBRATION,
        save_camera_intrinsics,
        setup_cameras,
        get_active_cameras,
        join_camera_workers,
    )
    from robot_control import (
        apply_profile_limits,
        create_and_configure_robots,
        build_robot_model,
        quiet_solver_logs,
        solver_log_summary,
        get_pose,
        move_to_named_pose,
        move_to_named_poses,
        resolve_middle_waist_shift,
        reset_arms,
        compute_fk_and_ee,
        sync_robot_state,
        command_state_is_stale
    )
    from servo_health import (
        FaultRecorder,
        OverloadEstimator,
        StallGate,
        ServoWatchdog,
    )
    from servo_health import check_profile as servo_check_profile
    from servo_health import check_shadows as servo_check_shadows
    from servo_health import decode as servo_decode
    from servo_health import read_hardware_errors as servo_read_errors
    from servo_health import recover as servo_recover
    from servo_health import report as servo_report
    from gripper import (GRIPPER_MODE, GRIPPER_CLOSED, GRIPPER_OPEN,
                              command_gripper, update_gripper,
                              update_gripper_from_trigger)
    from dataset import (
        BackgroundEpisodeSaver,
        quiet_libav,
        create_dataset,
        resolve_resume_run,
        build_frame,
    )
    from data_col_config import (
        anchor_arm_state,
        session_yaw_remap,
        TASKS,
        ARM_MODES,
        ArmTeleopState,
        TeleopConfig,
        TeleopSessionState,
        RobotCommandState,
        CommandKinematicsState,
        LinkGuard,
        compute_camera_arm_target,
        compute_gripper_arm_target,
        start_teleop_session,
        stop_teleop_session,
        clamp_joint_step,
        ControllerJumpGate,
        MAX_ENGAGE_S,
    )
    from log import (
        SessionStats,
        reset_episode_log,
        log_episode_info,
        write_episode_robustness,
        write_session_config,
    )


# quat2mat / pose2mat come from transform_utils (same math; one home).
try:
    from .transform_utils import pose2mat, quat2mat  # noqa: F401
except ImportError:
    from transform_utils import pose2mat, quat2mat  # noqa: F401

latest_frames = {
    name: {"color": None, "depth": None} for name in CAMERA_SERIALS
}
latest_timestamps = {
    name: {"color": None, "depth": None} for name in CAMERA_SERIALS
}
frame_lock = threading.Lock()
camera_shutdown = threading.Event()
latest_key = None

## NOTE: scene cameras are chosen in main() from --scene-cams / GIAVA_SCENE_CAMS.
## A module-level CameraConfig used to sit here hardcoded to both-on; it was dead
## (main() rebinds the name before get_active_cameras ever sees it) but it read
## like the live setting, which is worse than not being there.

def keyboard_listener():
    global latest_key
    while True:
        latest_key = input().strip()

def select_mode():
    mode_names = list(ARM_MODES.keys())

    print("\nAVAILABLE MODES\n")

    for mode, arm_names in enumerate(mode_names):
        print(f"{mode}: {arm_names}")

    while True:
        try:
            idx = int(input("\nSelect mode: "))
            return mode_names[idx]
        except:
            print("Invalid selection")

def now():
    return time.monotonic()

def mark_episode_discarded(dataset_root, episode_idx):
    discard_file = dataset_root / "discarded_episodes.txt"
    with open(discard_file, "a") as f:
        f.write(f"{episode_idx}\n")

## CANONICAL POSE SHORTCUTS, deliberately FIXED rather than derived from
## whichever poses the current --mode happens to share.
##
## The alternative -- shortest-unique-prefix over the available names -- makes
## the same keystroke mean different things in different modes: 'f' is
## unambiguous in --mode right (forward/high/rest/low) but ambiguous in --mode
## middle, which also has far_scene.  These keys are typed at a live robot
## with the arms energised; a key whose meaning depends on a flag typed twenty
## minutes ago is a key that will eventually drive an arm somewhere the
## operator did not intend.  So f/h/r/l always mean the same four poses, and
## everything else is spelled out.
POSE_SHORTCUTS = {"f": "forward", "h": "high", "r": "rest", "l": "low",
                  "m": "med"}


def resolve_pose_key(selector, available):
    """Map an 'i<selector>' suffix to a pose name.

    Returns (pose_name, None) or (None, message).  `available` is the set of
    poses EVERY active arm has, so a name that comes back here is safe to send
    to every arm without a mid-move failure.
    """
    sel = (selector or "").strip().lower()
    avail = sorted(available)
    short = {k: v for k, v in POSE_SHORTCUTS.items() if v in available}

    def _menu():
        keys = ", ".join(f"i{k} -> {v}" for k, v in sorted(short.items()))
        extra = [a for a in avail if a not in short.values()]
        line = f"  poses: {keys}"
        if extra:
            line += f"\n  also (spell out): {', '.join('i' + e for e in extra)}"
        return line

    if sel in available:
        return sel, None
    if sel in short:
        return short[sel], None
    hits = [a for a in avail if a.startswith(sel)]
    if len(hits) == 1:
        return hits[0], None
    if len(hits) > 1:
        return None, (f"\n[pose] 'i{sel}' is ambiguous -- matches "
                      f"{', '.join(hits)}. Type more of the name.\n{_menu()}")
    return None, (f"\n[pose] no pose named '{sel}' is available on every "
                  f"active arm.\n{_menu()}")


def parse_pose_spec(spec, arm_names):
    """'forward' -> every arm to forward; 'right=forward,middle=far_scene' ->
    per arm.  Every active arm must be named in the per-arm form."""
    spec = str(spec).strip().lower()
    if "=" not in spec:
        return {a: spec for a in arm_names}
    out = {}
    for part in spec.split(","):
        arm, _, pose = part.partition("=")
        out[arm.strip()] = pose.strip()
    missing = [a for a in arm_names if a not in out]
    extra = [a for a in out if a not in arm_names]
    if missing or extra:
        raise SystemExit(f"pose spec {spec!r}: missing arms {missing}, "
                         f"unknown arms {extra}; active arms are {arm_names}")
    return out


def park_arms(robots, arm_names, spec):
    """All active arms to the pose(s) named by `spec`, one simultaneous ramp
    (robot_control.move_arms_together), so a per-arm spec is as smooth as a
    shared pose name."""
    try:
        from .robot_control import get_pose, move_arms_together
    except ImportError:
        from robot_control import get_pose, move_arms_together
    poses = parse_pose_spec(spec, arm_names)
    targets = {a: get_pose(a, poses[a]) for a in arm_names}
    return move_arms_together({a: robots[a] for a in arm_names}, targets)


def _unblock_cameras(pipelines):
    """Wake any camera worker blocked in a driver call, so it can see the
    shutdown flag.  Safe to call twice; every step is best-effort."""
    for name, p in (pipelines or {}).items():
        if isinstance(p, (tuple, list)) and p:
            for q in p[1:]:                      # oak output queues
                try:
                    q.close()
                except Exception:
                    pass
            stop = getattr(p[0], "stop", None)   # dai.Pipeline.stop()
            if callable(stop):
                try:
                    stop()
                except Exception as exc:
                    print(f"[camera] {name}.stop(): {exc}")


_CAMERAS_DOWN = {"done": False}


def shutdown_cameras(pipelines):
    """Stop the camera workers, THEN their devices -- in that order.

    Split out of safe_shutdown and registered with atexit so it also runs on
    Ctrl-C and on a traceback, not only on the 'q' path.  Skipping it is what
    produced `terminate called without an active exception` and a core dump on
    every abnormal exit: the interpreter tore down while a worker was still
    inside a cv2 / librealsense / depthai call.  The recorded data was already
    safe (episode_saver.close runs first, also via atexit), but the cameras
    were left wedged for the next run.  Idempotent -- the 'q' path calls this
    directly and the atexit hook then does nothing.
    """
    if _CAMERAS_DOWN["done"]:
        return
    _CAMERAS_DOWN["done"] = True

    try:
        camera_shutdown.set()
        ## UNBLOCK BEFORE JOINING.  oak_worker sits in a blocking
        ## MessageQueue.get(); it only re-checks camera_shutdown at the top of
        ## its loop, so setting the event alone does not wake it.  Closing the
        ## queues makes the pending get() return, and stopping the pipeline
        ## shuts the device's own pipeline down while its consumer is still
        ## alive to notice -- which is the graceful order depthai wants.
        _unblock_cameras(pipelines)
        stragglers = join_camera_workers(timeout=2.0)
        if stragglers:
            ## Not fatal and not fixable from here: a worker blocked on a
            ## camera that has stopped delivering frames only unblocks when
            ## the pipeline below is stopped.  Say so rather than appearing
            ## to hang.
            print(f"[camera] still running after 2 s: {', '.join(stragglers)} "
                  f"-- stopping their devices now (they will exit with it).")
    except Exception as exc:
        print(f"[camera] worker shutdown: {exc}")

    try:
        for name, p in (pipelines or {}).items():
            for method in ("stop", "close"):
                fn = getattr(p, method, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception as exc:
                        print(f"[camera] {name}.{method}(): {exc}")
            ## setup_cameras returns {"oak": (device, q_left, q_right),
            ## "realsense": {name: pipeline}} -- neither value has .stop(), so
            ## the getattr loop above walks straight past BOTH of them and no
            ## camera would ever be shut down.  Reach into the two shapes it
            ## actually returns.
            if isinstance(p, dict):
                for cam_name, pipe in p.items():
                    try:
                        pipe.stop()
                    except Exception as exc:
                        print(f"[camera] {cam_name}.stop(): {exc}")
            elif isinstance(p, (tuple, list)) and p:
                ## setup_oak_stereo returns (dai.Pipeline, q_left, q_right).
                ## dai.Pipeline has NO close() -- the old getattr(dev,
                ## "close") found nothing, so the OAK was never shut down at
                ## all and only died in the process destructor, which is what
                ## printed "[depthai] Device ... has crashed" and dumped a
                ## crash log on every exit.  stop() is the real call; it has
                ## already run in _unblock_cameras, so this only waits.
                _wait = getattr(p[0], "wait", None)
                if callable(_wait):
                    try:
                        _wait()
                    except Exception as exc:
                        print(f"[camera] oak wait(): {exc}")
    except Exception as exc:
        print(f"[camera] device shutdown: {exc}")


def safe_shutdown(robots, pipelines, dataset, episode_saver,
                  collecting_episode, dataset_root, episode_idx,
                  quit_pose=None, hold=False, preview=None):
    """Minimal safe shutdown helper to avoid undefined-symbol crashes.

    hold=True ('qh') leaves the arms exactly where they are, torque on, so the
    next launch -- same --start-pose, or none -- starts from here instead of
    ramping down to rest and back up.  Only for a short gap between sessions:
    the servos hold the arm's weight the whole time."""
    try:
        ## Idle FIRST: while the writer thread is still draining an episode
        ## the buffer it is closing out is still in place, so a
        ## has_pending_frames() check here would see that episode and queue it
        ## a SECOND time.
        if episode_saver is not None:
            episode_saver.wait_until_idle(announce=True)
        ## Save whatever was in progress rather than losing it on exit.
        ## has_pending_frames() rather than collecting_episode alone, so a
        ## half-recorded episode is written on the way out however the session
        ## ended.  The verdict is unknown by definition here.
        if dataset is not None and (collecting_episode
                                    or dataset.has_pending_frames()):
            episode_saver.save_episode_async("unknown")
            collecting_episode = False
    except Exception:
        pass

    try:
        ## close() -> dataset.finalize(), which flushes the buffered episode
        ## metadata and writes the parquet footers.  WITHOUT IT THE DATASET ON
        ## DISK CANNOT BE LOADED BACK -- quit with 'q', never Ctrl-C.
        if episode_saver is not None:
            episode_saver.close()
    except Exception:
        pass

    try:
        ## --quit-pose, honoured here rather than falling through to
        ## reset_arms' own DEFAULT_RESET_POSE.  The pose the arms end a
        ## session in is not cosmetic: "rest" lets them down onto their own
        ## stops so the servos are not holding the arm's weight against
        ## gravity all night, which is what actually cooks the shoulder.
        if hold:
            print("[quit] holding position (torque on) -- arms did NOT park")
        elif quit_pose:
            reset_arms(robots, quit_pose)
        else:
            reset_arms(robots)
    except Exception:
        pass

    ## Before the arms park, not after: parking takes seconds, and a window
    ## still showing a live feed of arms that are being stowed reads as if
    ## the session were still running.  Idempotent, and the atexit hook
    ## covers every other way out.
    try:
        if preview is not None:
            preview.close()
    except Exception:
        pass

    ## ORDER MATTERS: tell the workers to stop, WAIT for them, and only then
    ## tear the devices down.  Stopping a librealsense pipeline or destroying
    ## the OAK device while its worker is still inside wait_for_frames() /
    ## queue.get() / cv2 kills the process with glibc's "FATAL: exception not
    ## rethrown" and a core dump -- after finalize(), so the data survives, but
    ## it hides real crashes and leaves the cameras wedged for the next run.
    shutdown_cameras(pipelines)

    return episode_idx, True

def main():
    if rospy is None:
        raise ImportError("rospy is required for data collection.")

    ## Before ANY encoder exists -- the dataset's, and the headset's.  See
    ## quiet_libav in dataset.py: the spam is enabled by lerobot restoring
    ## libav's default callback after each encode, and the fix has to be in
    ## place before the first one.
    quiet_libav()
    ## Before the URDF is parsed: pyroki logs through loguru at INFO and its
    ## default sink writes coloured lines straight to stderr.  Warnings still
    ## print; solver_log_summary() below says how much was hidden.
    quiet_solver_logs()

    rospy.init_node("data_collection")

    global latest_key
    collecting_episode = False
    ## True once teleop has been enabled during the CURRENT episode; see
    ## RECORD_GATE above.  Reset with every r / s / d.
    episode_armed = False
    cfg = TeleopConfig()

    stats = SessionStats()

    # # Print tasks and get selection
    # print("\nAVAILABLE TASKS:\n")
    # for idx, name in TASKS.items():
    #     print(f"{idx}: {name}")

    # while True:
    #     try:
    #         task_idx = int(input("\nSelect task number: ").strip())
    #         if task_idx not in TASKS:
    #             print("Invalid task number. Try again.")
    #             continue
    #         break
    #     except ValueError:
    #         print("Please enter an integer task number.")

    ## --task picks the dataset directory (a name from TASKS).  For the
    ## shape_sorter task, --target names the piece being inserted; it becomes
    ## the per-FRAME task string (shape_sorter.task_string), which is how one
    ## run holds cube, triangle and flower episodes side by side and how the
    ## env-state builder later knows which piece each episode is about.
    ## `target <piece>` at the prompt switches it between episodes.
    try:
        from .collision_modes import take_option as _take_option
        from . import shape_sorter as _ss
    except ImportError:
        from collision_modes import take_option as _take_option
        import shape_sorter as _ss
    task_name = str(_take_option("task", default=TASKS[7])).strip()
    if task_name not in TASKS.values():
        raise SystemExit(
            f"--task must be one of {sorted(TASKS.values())}, got '{task_name}'")
    _target = _take_option("target")
    ## --episodes N: a soft per-target goal.  Nothing stops at N; the save
    ## message just says "12/30 cube" and, at N, reminds you to switch target
    ## or quit.  Counts are per target and per session (saved, any verdict).
    _goal = _take_option("episodes")
    ## --resume <timestamp>|latest: keep adding episodes to an EXISTING run
    ## instead of starting a new one, so a session can be split across
    ## restarts (and across days) without splitting the dataset.  Episode
    ## numbering continues where the run left off.
    _resume = _take_option("resume")
    ## --no-cycle: keep --target fixed across saves instead of stepping the
    ## round-robin.  For DAgger on one piece, or any single-piece session.
    _no_cycle = _take_option("no-cycle", default=None) is not None
    ## --policy <checkpoint>: HUMAN-GATED DAgger.  The policy drives every arm
    ## the operator is NOT gripping; grip a controller and that arm reverts to
    ## teleop mid-episode.  The correction is recorded as part of an ordinary
    ## episode, so verdicts, gates and the dataset writer are unchanged.
    ## Everything below is inert without this flag.
    _policy_ckpt = _take_option("policy")
    _policy_steps = _take_option("policy-n-action-steps")
    ## --replicate DIR [--replicate-episodes 0,2,7]: REPRODUCE the scene
    ## positions of an earlier rollout run before each episode.  DIR is a
    ## rollout's snapshots_* folder; the live scene cameras are shown ghosted
    ## against its episode-N frames until the object sits where it did then.
    ## This is what makes a TARGETED DAgger session possible: the positions
    ## the policy failed at are known by number, and a correction recorded
    ## anywhere else teaches the policy about a different scene.  Every
    ## episode's own starting scene is saved to <run>/snapshots/ regardless,
    ## so the corrected positions can be replicated again by the evaluation.
    _replicate = _take_option("replicate")
    _replicate_eps = _take_option("replicate-episodes")
    ## --place-grid: overlay place_grid.py's lattice on top_scene between
    ## episodes and name the nearest cell.  Lives HERE rather than in
    ## place_grid.py's own window because the two would open the same
    ## RealSense device, and it refuses a second owner.  Same window model
    ## as --replicate: ~10 Hz, only while not recording, above the headset
    ## guard so it shows while the operator is at the table.
    _place_grid = _take_option("place-grid", default=None) is not None
    task_ctx = {"task": task_name, "target": None, "saved": {},
                "goal": int(_goal) if _goal else None,
                "perms": [], "round": 0, "slot": 0, "cycle": not _no_cycle}
    if task_name == _ss.TASK_NAME:
        ## THE ORDER IS DRIVEN BY THE CODE, not remembered by the operator.
        ## Every permutation of the pieces, cycled in order: across one full
        ## cycle each piece is recorded once in every slot of the round, so
        ## "how many distractors were still on the table" is uncorrelated with
        ## which piece was the target.  Ordering by hand drifts -- and a
        ## correlation there is a shortcut the policy WILL learn instead of
        ## the task ("two objects left means fetch the cube").
        import itertools
        task_ctx["perms"] = [list(x) for x in
                             itertools.permutations(_ss.CLASS_NAMES)]
        if _target is not None:
            if _target not in _ss.CLASSES:
                raise SystemExit(f"--target must be one of {_ss.CLASS_NAMES}, "
                                 f"got '{_target}'")
            ## Start the cycle on the piece asked for, keeping the balance.
            _k = next(i for i, pm in enumerate(task_ctx["perms"])
                      if pm[0] == _target)
            task_ctx["perms"] = (task_ctx["perms"][_k:] + task_ctx["perms"][:_k])
        _first = task_ctx["perms"][0][0]
        task_ctx["target"] = _first
        task_ctx["task"] = _ss.task_string(_first)
    elif _target is not None:
        raise SystemExit(f"--target only applies to --task {_ss.TASK_NAME}")
    print(f"\nSelected task: {task_name}"
          + (f"   target={task_ctx['target']}   task string: "
             f"'{task_ctx['task']}'" if task_ctx["target"] else "")
          + (f"   goal={task_ctx['goal']}/target" if task_ctx["goal"] else ""))
    if task_ctx["perms"] and not task_ctx["cycle"]:
        print(f"  --no-cycle: target stays '{task_ctx['target']}' for the "
              f"whole session")
    elif task_ctx["perms"]:
        print(f"  round order cycles through {len(task_ctx['perms'])} "
              f"permutations, then repeats:")
        for _i, _pm in enumerate(task_ctx["perms"]):
            print(f"    round {_i + 1}: " + " -> ".join(_pm))

    # # Print modes and get selection
    # print("\nAVAILABLE MODES:\n")
    # for mode, arm_names in ARM_MODES.items():
    #     print(f"{mode}: {arm_names}")

    # while True:
    #     try:
    #         mode = input("\nSelect mode: ").strip()
    #         if mode not in ARM_MODES:
    #             print("Invalid mode. Try again.")
    #             continue
    #         break
    #     except ValueError:
    #         print("Please enter one of the listed modes.")

    ## --mode selects which arms are active.  Default "all" (all three) is
    ## the coupled-IK study deployment; "right_av"/"left_av" are one
    ## manipulator plus the camera arm.  Consumed from argv the same way
    ## --collision is, so the positional episode index still works.
    try:
        from .collision_modes import take_option as _take_option
    except ImportError:
        from collision_modes import take_option as _take_option
    mode = str(_take_option("mode", default="all")).strip().lower()
    if mode not in ARM_MODES:
        raise SystemExit(
            f"--mode must be one of {sorted(ARM_MODES)}, got '{mode}'")

    ## --start-pose / --quit-pose: which named pose the arms park at when the
    ## session starts (and on the 'i' re-park key), and which they end at.
    ##
    ## The start pose is recorded in teleop_config.json, because it is part of
    ## what shaped the data: a policy trained from demonstrations that all
    ## begin at "forward" has only ever seen that starting distribution, and a
    ## rollout launched from a different pose is out of distribution before it
    ## takes a single step.
    ##
    ## Validated against POSES for EVERY active arm up front, not at the moment
    ## of the move -- the middle arm carries poses the gripper arms do not
    ## (far_scene, looking_left/right), so a name valid for one is not
    ## necessarily valid for another, and discovering that after the servos are
    ## energised is the wrong time.
    _pose_names = sorted(set.intersection(
        *(set(POSES[a]) for a in ARM_MODES[mode])))
    start_pose = str(_take_option("start-pose", default="forward")).strip().lower()
    quit_pose = str(_take_option("quit-pose", default="rest")).strip().lower()
    ## Where the arms this mode does NOT drive are assumed to be parked; see
    ## the q vector built below.  Validated against every inactive arm.
    inactive_pose = str(_take_option("inactive-pose", default="rest")).strip().lower()
    for _a in ("left", "right", "middle"):
        if _a not in ARM_MODES[mode] and inactive_pose not in POSES[_a]:
            raise SystemExit(
                f"--inactive-pose={inactive_pose!r} is not a pose {_a} has. "
                f"Available for {_a}: {sorted(POSES[_a])}")

    if quit_pose not in _pose_names:
        raise SystemExit(
            f"--quit-pose={quit_pose!r} is not a pose every arm in --mode "
            f"{mode} has. Available for {ARM_MODES[mode]}: {_pose_names}")
    ## --start-pose takes a shared name OR a per-arm spec
    ## (right=forward,middle=far_scene), so the camera arm can start where it
    ## sees the scene while the manipulator starts at forward.
    for _a, _p in parse_pose_spec(start_pose, ARM_MODES[mode]).items():
        if _p not in POSES[_a]:
            raise SystemExit(
                f"--start-pose: {_a} has no pose {_p!r}. Available for {_a}: "
                f"{sorted(POSES[_a])}")

    # # Print modes and get selection
    # print("\nWhich scene cameras to activate?\nEnter l for low and t for top (e.g. 'lt' for both, 'l' for low only, 't' for top only):")

    # while True:
    #     try:
    #         camera_selection = input().strip()
    #         if not camera_selection:
    #             print("Invalid selection. Try again.")
    #             continue
    #         break
    #     except ValueError:
    #         print("Please enter a valid selection.")

    ## WHICH SCENE CAMERAS ARE RECORDED, from the environment rather than from
    ## an edit made before the session and forgotten.  GIAVA_* variables are
    ## snapshotted into the run's teleop_config.json, so this way the camera
    ## set is recorded WITH the dataset; a hardcoded value is not.
    ##
    ##   GIAVA_SCENE_CAMS=none      (default) wrist camera(s) + OAK only
    ##   GIAVA_SCENE_CAMS=top,low   both scene cameras -- the historical default
    ##   GIAVA_SCENE_CAMS=low       low scene only
    ##
    ## Wrist cameras are deliberately NOT selectable here: get_active_cameras
    ## ties them to the arm mode, because a wrist camera without its arm is
    ## meaningless.  The OAKs likewise follow the middle arm.
    ##
    ## WHY THE DEFAULT IS NOW 'none' (2026-09-09).  Each scene camera is a
    ## 640x480x60 fps RealSense with a capture thread, a per-frame copy and a
    ## history deque in THIS process, and every recorded tick copies its frame
    ## again under frame_lock.  Two of them are pure overhead on a control loop
    ## that is already over budget, and they contribute nothing to a session
    ## driven from the headset through the OAK.  Turned off, the loop has two
    ## fewer producers competing for the GIL and the lock.
    ##
    ## WHAT THIS COSTS, so it is a decision and not a surprise: episodes
    ## recorded without top_scene CANNOT feed the object-centric / env-state
    ## pipeline -- build_envstate_dataset.py exits with "no top_scene video"
    ## and scene_features.py reads top_scene -- nor train any 3-camera ACT
    ## config.  Recording for those, put the cameras back for the run:
    ##     GIAVA_SCENE_CAMS=top,low python data_collection.py
    ## `--scene-cams none|top|low|top,low` overrides the variable for one run.
    ## Consumed here, before the positional episode-index parse below, which
    ## rejects anything flag-shaped.
    _scene_sel = str(_take_option(
        "scene-cams", None,
        os.environ.get("GIAVA_SCENE_CAMS", "none"))).strip().lower()
    os.environ["GIAVA_SCENE_CAMS"] = _scene_sel   # so the run's config snapshot records it
    _scene_cams = {c.strip() for c in _scene_sel.split(",") if c.strip()}
    _scene_cams.discard("none")
    _unknown = _scene_cams - {"top", "low"}
    if _unknown:
        raise SystemExit(
            f"GIAVA_SCENE_CAMS={_scene_sel!r} names unknown camera(s): "
            f"{sorted(_unknown)}. Valid entries are 'top', 'low', or 'none'.")

    camera_config = CameraConfig(
        top_active="top" in _scene_cams,
        low_active="low" in _scene_cams,
    )
    ## Said out loud every session.  Which cameras a dataset contains is not
    ## recoverable from the frames by anyone who did not record it, and the
    ## default changing under a run is exactly the kind of thing that gets
    ## noticed a week later at training time.
    if _scene_cams:
        print(f"[cameras] scene cameras: {', '.join(sorted(_scene_cams))}")
    else:
        print("[cameras] scene cameras OFF (wrist + OAK only). "
              "GIAVA_SCENE_CAMS=top,low to record them -- needed for "
              "env-state / object-centric builds and 3-camera ACT.")

    # background keyboard thread to get user input
    threading.Thread(target=keyboard_listener, daemon=True).start()

    # headset thread
    ## Startup profile: one line per phase, so "startup is slow" has a number
    ## attached to it.  Hardware phases (cameras, servos, headset) cannot be
    ## cached; the IK compile is cached by jax (see jax_platform.apply).
    _startup_t0 = now()
    _phase_t = [_startup_t0]
    def _phase(label):
        t = now()
        print(f"[startup] {label:<32s} +{t - _phase_t[0]:5.1f}s   (t={t - _startup_t0:5.1f}s)")
        _phase_t[0] = t

    ## HEADSET STREAM DEFAULTS, measured on this rig.  setdefault, so an
    ## explicit environment or a flag still wins -- these only decide what
    ## happens when data_collection.py is run with nothing set, which is how it
    ## is usually run.
    ##
    ## Every one of these replaces a default that is wrong HERE, and each was
    ## measured rather than guessed (see gvlink video.py / ratecontrol.py):
    ##
    ##   preset veryfast   `ultrafast` sets --no-deblock, which is what reads as
    ##                     a grainy, blocky picture.  It is not even cheaper at
    ##                     this canvas: 1280x1280 costs 5.10 ms/frame/eye on
    ##                     ultrafast and 4.20 on veryfast, because CABAC and
    ##                     real prediction leave fewer bits to write.
    ##   threads 4         libx264 SLICE threading, hardcoded to 2 upstream.
    ##                     Halves per-frame encode time on a machine with cores
    ##                     to spare, which is the headroom that keeps the
    ##                     picture together while three arms solve coupled IK.
    ##   vbv 400 ms        the 100 ms default cannot lend bits to a detailed
    ##                     frame, so detail is exactly what it starves.
    ##   25 fps            the OAK runs at 25 (oak_stream_settings).  The 30
    ##                     default makes the rate controller budget bits across
    ##                     frames that do not exist.
    ##   25000 kbps/eye    12000 is thin for a 1280-wide atlas.
    ##   rate delay 150ms  THE IMPORTANT ONE.  BitrateController treats 40 ms of
    ##                     queueing as congestion, and 40 ms IS one frame period
    ##                     at 25 fps -- so ordinary pacing jitter reads as a
    ##                     congested link.  Cuts are x0.75 every 0.35 s and
    ##                     raises x1.08 every 1 s, so it loses by construction:
    ##                     measured on this rig at 800 kbps/eye (the floor)
    ##                     against a requested 25000, with 43 cuts and reason
    ##                     `delay` every time, on an idle LAN.  Raising the
    ##                     threshold above a frame period keeps adaptation for
    ##                     a link that is genuinely bad without firing on one
    ##                     that is merely discrete.
    for _k, _v in (("GIAVA_X264_PRESET", "veryfast"),
                   ("GIAVA_X264_THREADS", "4"),
                   ("GIAVA_X264_VBV_MS", "400"),
                   ("GIAVA_HEADSET_FPS", "25"),
                   ("GIAVA_HEADSET_BITRATE_KBPS", "25000"),
                   ("GIAVA_RATE_DELAY_MS", "150")):
        os.environ.setdefault(_k, _v)

    ## FOVEATION, per run.  The headset streams a two-layer atlas: a
    ## downscaled full field (the "coarse" band) with a 1:1 crop pasted at
    ## the gaze point (the "fovea" band).  Two independent knobs, and they do
    ## opposite-sounding things, so they are named after what they scale:
    ##
    ##   --coarse-scale  how much of the coarse band the downscaled full field
    ##                   occupies.  LOWER = blurrier periphery = stronger
    ##                   foveation.  RAISE IT to make foveation less
    ##                   aggressive.  Default 0.35.
    ##   --fovea-scale   how much of the fovea band the 1:1 crop occupies,
    ##                   i.e. how big the sharp patch is.  LOWER = smaller,
    ##                   more eye-like patch.  Default 0.5.
    ##   --fovea off     no foveation at all -- one flat image.
    ##
    ## Both live in [0.05, 1.0]; checked here rather than letting AtlasLayout
    ## raise, because it does so from inside the headset thread at startup.
    ## Written back into the environment so the run's teleop_config.json
    ## records what the operator actually looked through -- a recorded gaze
    ## trajectory means something different under a different atlas.
    _hs_kwargs = {}
    _fov_sel = str(_take_option(
        "fovea", None, os.environ.get("GIAVA_FOVEATION", "1"))).strip().lower()
    if _fov_sel in ("0", "off", "no", "false"):
        _hs_kwargs["foveation"] = False
        os.environ["GIAVA_FOVEATION"] = "0"
    elif _fov_sel not in ("1", "on", "yes", "true"):
        raise SystemExit(f"--fovea must be on|off, got '{_fov_sel}'")
    for _flag, _kw, _env, _dflt in (
            ("coarse-scale", "coarse_scale", "GIAVA_COARSE_SCALE", 0.35),
            ("fovea-scale", "fovea_scale", "GIAVA_FOVEA_SCALE", 0.5)):
        _raw = _take_option(_flag, None, os.environ.get(_env))
        if _raw is None:
            continue
        try:
            _val = float(_raw)
        except ValueError:
            raise SystemExit(f"--{_flag} must be a number, got '{_raw}'")
        if not 0.05 <= _val <= 1.0:
            raise SystemExit(
                f"--{_flag} must be in [0.05, 1.0], got {_val} "
                f"(default {_dflt})")
        _hs_kwargs[_kw] = _val
        os.environ[_env] = repr(_val)
    if _hs_kwargs:
        print(f"[headset] foveation: "
              + ", ".join(f"{k}={v}" for k, v in sorted(_hs_kwargs.items())))
        ## The viewer gets the last word when it asks for its own atlas: see
        ## GvLinkHeadset._layout_for, which uses these only as the fallback
        ## for a scale the viewer did not specify.  `--fovea off` is the
        ## exception -- want_fovea ANDs both ends, so the robot can always
        ## turn foveation OFF, it just cannot force it back on.
        print("[headset] (a viewer that requests its own atlas overrides the "
              "scales; --fovea off always wins)")

    headset = make_headset(**_hs_kwargs)
    headset.run_in_thread()

    # camera pipelines
    # determine active cameras (use camera manager helper if present; fallback to configured serials)
    active_cameras = get_active_cameras(mode, camera_config)

    print(f"active cam: {active_cameras}")

    # initialize latest frame/timestamp containers now that active_cameras is known
    latest_frames = {cam: None for cam in active_cameras}
    latest_timestamps = {cam: None for cam in active_cameras}

    # setup cameras (headset passed so the OAK worker streams the stereo
    # feed to the VR display while also filling latest_frames for the dataset)
    _phase("headset transport")
    pipelines = setup_cameras(
        active_cameras,
        camera_shutdown,
        frame_lock,
        latest_frames,
        latest_timestamps,
        headset=headset,
    )

    ## Registered THE MOMENT the devices exist, so Ctrl-C and any traceback
    ## tear the cameras down the same way 'q' does.  atexit is LIFO and
    ## episode_saver.close registers later, so the dataset is still finalized
    ## before the cameras go -- the order safe_shutdown uses.
    import atexit
    atexit.register(shutdown_cameras, pipelines)

    ## On-screen OAK preview.  Constructed even when off (it is off unless
    ## --preview / GIAVA_OAK_PREVIEW says otherwise), so the loop below can
    ## publish unconditionally.  It reads the same latest_frames the episode
    ## writer does, so the window keeps streaming whether or not an episode
    ## is being recorded -- teleop and recording are one code path, with the
    ## recorder either running or not.  See oak_preview.py.
    preview = OakPreview(PREVIEW_MODE, active_cameras)
    atexit.register(preview.close)

    arm_names = ARM_MODES[mode]

    episode_stats = reset_episode_log(active_cameras, arm_names)
    ## Servo health: the same stack teleop runs.  A data-collection session
    ## is the WORST place to lose an arm silently -- a limp servo keeps
    ## accepting commands, so the episode records confident actions against a
    ## fallen arm and looks valid on disk.  All three read joint_states,
    ## which is already subscribed, so this costs nothing per tick.
    watchdog = ServoWatchdog(arm_names)
    overload = OverloadEstimator(arm_names)
    stallgate = StallGate(arm_names)
    ## Driver-space travel limit for the camera arm's waist.  Inert unless
    ## configured or learning -- see waist_travel.py for why the physical stop
    ## is invisible to the URDF, the driver and the solver alike.
    waist_travel = WaistTravel("middle")
    if "middle" in arm_names:
        print(waist_travel.describe())
    faultlog = FaultRecorder(arm_names, seconds=20.0,
                             rate_hz=1.0 / cfg.control_dt)
    _overload_last_print = {arm: -10 ** 9 for arm in arm_names}
    _faulted = {arm: False for arm in arm_names}
    _pending_reboot = {}

    tick_counter = 0
    _gate_last_print = [-10**9]
    ## DAgger bookkeeping: where each arm's joints sit in the policy's action
    ## vector, and a throttle for the "holding" message.
    try:
        from .policy_driver import arm_slices as _arm_slices
    except ImportError:
        from policy_driver import arm_slices as _arm_slices
    _dagger_slices = _arm_slices(arm_names)
    _dagger_last_print = [-10**9]
    _dagger_prev_human = [frozenset()]
    _dagger_frames = {"policy": 0, "human": 0}
    ## 'p' toggles this.  The policy drives ONLY while it is True (and an
    ## episode is recording); 'r' alone leaves the arm parked, so the operator
    ## decides when autonomy starts, not the recorder.
    dagger_play = False
    _dagger_last_grip = {}
    _table_gate_last_print = [-10**9]
    _waist_last_print = [-10**9]
    _frozen_prev = {"left": None, "right": None}
    _frozen_ticks = {"left": 0, "right": 0}
    ## Rejects controller/head poses that moved faster than a hand can move.
    ## See ControllerJumpGate in data_col_config.py for what the older
    ## zeros/frozen guards below do NOT catch, and why a bad delta shows up as
    ## a confident slew rather than an obvious jump.
    jump_gate = ControllerJumpGate()
    ## Gate counters are session-cumulative; the per-episode robustness record
    ## wants per-episode numbers, so snapshot the counters when an episode
    ## starts and diff at save time.
    gate_baseline = ({}, {})
    if not jump_gate.enabled:
        print("[tracking] controller jump gate DISABLED "
              "(GIAVA_CTRL_JUMP_GATE=0)")

    _phase("cameras")
    robots = create_and_configure_robots(arm_names)
    ## Ask the motors which profile they run and make the clamp agree.
    ## Read-only; with GIAVA_PROFILE_MODE unset it reports and changes nothing,
    ## so an existing recording session is unaffected.
    apply_profile_limits(robots, cfg, arm_names)

    # Middle-waist driver frame = physical motor re-clock + Homing_Offset.
    # Resolved and reported in ONE place so this and teleop.py cannot drift
    # apart; see robot_control.resolve_middle_waist_shift for why the register
    # is ignored under ext_position.
    waist_shift = 0.0
    if "middle" in arm_names:
        waist_shift = resolve_middle_waist_shift(robots["middle"])

    _phase("arms + servo profiles")
    robot, arm_data = build_robot_model(mode)

    # Coupled three-arm IK (ik_study winner: smoothing 0.05 + centering 0.5 +
    # sphere self-collision, margin 20 mm / weight 100; pose 50/10).  One
    # solve per tick for all arms — replaces per-arm solve_single_arm_ik.
    if CoupledStudyIK is None:
        raise ImportError("study_ik.CoupledStudyIK unavailable — check ik/study/.")
    print()
    print(_collision_banner(COLLISION_MODE))
    print()

    _phase("pyroki/urdf robot model")
    coupled_ik = CoupledStudyIK(
        robot,
        URDF_PATH,
        ee_links={a: ARM_CONFIG[a]["ee_link"] for a in ("left", "right", "middle")},
        # Smoothing is scaled against the *actual* control period, so changing
        # the loop rate keeps the study's physical velocity budget.
        control_dt=cfg.control_dt,
        waist_driver_shift=waist_shift,
    )
    print("Compiling coupled IK solver (a few seconds)...")
    _t0 = now()
    coupled_ik.warmup(np.asarray(robot.joint_var_cls(0).default_factory(), dtype=np.float32))
    print(f"Coupled IK ready in {now() - _t0:.1f} s")
    _phase("IK build + compile (jax cache)")

    ## TUBE-MPC REFERENCE FILTER, opt-in: GIAVA_TUBE_MPC=1.  One filter per
    ## arm, built against the DRIVER's limit arrays (the ones that reject
    ## commands), pulled in by driver_limit_margin.  Off = today's path
    ## exactly.  Everything it does is visible: the env flag lands in
    ## teleop_config.json, per-arm solve/intervention stats in
    ## robustness.jsonl (controller_gate.tube_mpc), fallbacks in the episode
    ## line.  See tube_mpc_hook.py for the state-source choice.
    tube_filters = None
    if os.environ.get("GIAVA_TUBE_MPC", "0") == "1":
        from tube_mpc_hook import build_arm_filter
        ## Which arms get filtered.  Default left,right: the middle arm's
        ## waist runs in ext_position with a driver frame that is shifted
        ## from the URDF's (see waist_driver_shift / driver_to_urdf), so a
        ## box built from URDF limits is in the wrong frame for it until the
        ## hook learns the shift.  Opt in with GIAVA_TUBE_ARMS=left,right,middle.
        _tube_arms = [a.strip() for a in os.environ.get(
            "GIAVA_TUBE_ARMS", "left,right").split(",") if a.strip()]
        tube_filters = {}
        for _a in arm_names:
            if _a not in _tube_arms:
                print(f"[tube_mpc] {_a}: bypassed (GIAVA_TUBE_ARMS={','.join(_tube_arms)})")
                continue
            _gi = robots[_a].arm.group_info
            tube_filters[_a] = build_arm_filter(
                _a, _gi.joint_lower_limits, _gi.joint_upper_limits,
                cfg.driver_limit_margin)
            _f = tube_filters[_a]
            print(f"[tube_mpc] {_a}: {_f.n} joints, state={_f.state_source}, "
                  f"lookahead={_f.lookahead}, dt={_f.cfg.dt}, horizon={_f.cfg.horizon}")
            if abs(_f.cfg.dt - cfg.control_dt) > 1e-6:
                print(f"[tube_mpc] !! config dt {_f.cfg.dt} != control_dt "
                      f"{cfg.control_dt}: the filter's model is wrong for this "
                      f"loop rate. Fix dt in the yaml or GIAVA_CONTROL_HZ.")

    ## TOUCH RECORDING for calibration/touch_calibrate.py (GIAVA_TOUCH_RECORD=1).
    ## The headset link is exclusive to this process, so a controller button
    ## can only be read here: one press of GIAVA_TOUCH_BUTTON (default B --
    ## Y drives the camera arm, A/X drive the grippers) appends both arms'
    ## MEASURED joints and flange poses to a touch_<stamp>.json in the same
    ## schema the standalone recorder writes.  Press with both grippers
    ## closed and the closed tips touching.
    _touch = None
    if os.environ.get("GIAVA_TOUCH_RECORD", "0") == "1":

        _touch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "calibration", "data", "robot")
        os.makedirs(_touch_dir, exist_ok=True)
        _touch = {
            "button": os.environ.get("GIAVA_TOUCH_BUTTON", "b").strip().lower(),
            "path": os.path.join(_touch_dir, f"touch_{time.strftime('%Y%m%d_%H%M%S')}.json"),
            "touches": [], "prev": False, "last": -1e9,
        }
        print(f"[touch] recording on button '{_touch['button'].upper()}' -> {_touch['path']}")
        print("[touch] close BOTH grippers, bring the closed tips together, press the button.")

    ## Hard inter-arm collision gate on the FINAL clamped command (the
    ## solver's collision term is a soft cost checked BEFORE the clamps;
    ## this is a circumscribed-capsule proof-of-separation checked after
    ## them, immediately before set_joint_positions).  GIAVA_CAPSULE_GATE=0
    ## disables; see capsule_gate.py for margins and scope.
    try:
        from .capsule_gate import build_gate as _build_capsule_gate
    except ImportError:
        from capsule_gate import build_gate as _build_capsule_gate
    from yourdfpy import URDF as _URDF_for_gate
    capsule_gate = _build_capsule_gate(robot, _URDF_for_gate.load(URDF_PATH))
    stats.collision_mode = COLLISION_MODE

    ## Hard tabletop-floor gate, same call site and reasoning as the
    ## inter-arm gate above but against the table plane instead of the other
    ## arms (table_gate.py). Hardware-validated 2026-09-09 but never swept,
    ## unlike capsule_gate: disable with --table off / GIAVA_TABLE_GATE=0.
    try:
        from .table_gate import build_gate as _build_table_gate
    except ImportError:
        from table_gate import build_gate as _build_table_gate
    table_gate = _build_table_gate(robot, _URDF_for_gate.load(URDF_PATH))
    stats.table_mode = TABLE_MODE

    ## All arms on ONE ramp: `move_to_named_pose` in a loop is serial
    ## (blocking=True sleeps the caller) and each of its waypoints was its own
    ## accelerate-and-stop.  See robot_control.move_arms_together.
    _phase("gates + tube-mpc + headset link")
    park_arms(robots, arm_names, start_pose)
    _phase("park to start pose")

    ## Where the arms are parked RIGHT NOW.  Tracks the 'i' keys so the pose
    ## each episode actually started from is recorded per episode -- once the
    ## operator can change it mid-session, a single session-level value in
    ## teleop_config.json is no longer the truth for every episode.
    current_start_pose = start_pose

    ## INACTIVE ARMS ARE NOT AT URDF ZERO.  This vector is what the capsule
    ## gate checks against, and an arm the mode does not drive kept its zeros
    ## here -- for the left arm URDF zero is straight out over the table,
    ## right through the right arm's workspace, so the gate blocked perfectly
    ## legal --mode right_av motion ("HOLDING: right_wrist_link <->
    ## left_right_finger_link") against an arm that was parked at rest.
    ## An unmodelled arm is not a safe default in EITHER direction: zero
    ## invents an obstacle here, and would hide a real one if the arm were
    ## somewhere else.  Assume --inactive-pose (default 'rest', where q/quit
    ## leaves them); pass the pose they are actually parked in if it differs.
    q = np.zeros(robot.joints.num_actuated_joints, dtype=float)

    ## arm_data only holds the arms THIS MODE drives (build_robot_model
    ## builds it from ARM_MODES[mode]), so the inactive arms' joint indices
    ## have to come from the full robot model -- an `a in arm_data` test here
    ## silently skipped every one of them and left the zeros in place.
    _inactive = [a for a in ("left", "right", "middle") if a not in arm_names]
    for arm_name in _inactive:
        n = ARM_CONFIG[arm_name]["num_joints"]
        _idx = [robot.joints.actuated_names.index(j)
                for j in ARM_CONFIG[arm_name]["joint_names"]]
        q[_idx] = np.asarray(get_pose(arm_name, inactive_pose), dtype=float)[:n]
    if _inactive:
        print(f"[gate] inactive arm(s) {_inactive} assumed at "
              f"'{inactive_pose}' for collision checking "
              f"(--inactive-pose to change). THEY ARE NOT SENSED -- if one is "
              f"parked elsewhere, the gate is checking the wrong geometry.")

    for arm_name in arm_names:
        n = ARM_CONFIG[arm_name]["num_joints"]
        q[arm_data[arm_name]["joint_indices"]] = np.asarray(
            robots[arm_name].dxl.joint_states.position[:n],
            dtype=float,
        )

    fk, ee = compute_fk_and_ee(robot, coupled_ik.driver_to_urdf(q), arm_data)

    # for arm in arm_names:
    #     print(
    #         arm,
    #         type(ee[arm]),
    #         np.shape(ee[arm]),
    #     )

    cmd_kin = CommandKinematicsState(q_cmd=q.copy(),
        T_cmd={
            arm: ee[arm].copy() for arm in arm_names
        },
    )

    # print(type(cmd_kin.T_cmd["left"]))
    # print(np.shape(cmd_kin.T_cmd["left"]))
    # print(cmd_kin.T_cmd["left"])

    # cmd_state = RobotCommandState(
    #     last_cmds={
    #         arm: q[arm_data[arm]["joint_indices"]].copy() for arm in arm_names
    #     }
    # )

    cmd_state = RobotCommandState(
        last_arm_cmd_time={arm: 0.0 for arm in arm_names},
        last_cmds={
            arm: q[arm_data[arm]["joint_indices"]].copy()
            for arm in arm_names
        },
    )

    teleop_state = TeleopSessionState()

    full_joint_velocity_limits = np.ones(robot.joints.num_actuated_joints) * cfg.full_joint_velocity_limits_value

    ## Episode index is positional.  Skip anything flag-shaped so an
    ## unrecognised option produces a clear message rather than an
    ## int('--flag') traceback after the arms have already been energised.
    _pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    _flags = [a for a in sys.argv[1:] if a.startswith("-")]
    if _flags:
        raise SystemExit(
            f"unrecognised option(s): {' '.join(_flags)}\n"
            f"  usage: python data_collection.py [episode_index]\n"
            f"         [--collision sphere|capsule|gjk]   see collision_modes.py\n"
            f"         [--table on|off]                   tabletop avoidance\n"
            f"         [--jax gpu|cpu|cuda]               solver backend, see "
            f"jax_platform.py\n"
            f"         [--mode av|bimanual|left|right|middle]\n"
            f"         [--place-grid]                     hull-study lattice on top_scene "
            f"between episodes, see place_grid.py\n"
            f"         [--preview [left|right|both]]       OAK on the screen, "
            f"see oak_preview.py\n"
            f"         [--scene-cams none|top|low|top,low] scene cameras "
            f"(default none)\n"
            f"         [--fovea on|off] [--coarse-scale F] [--fovea-scale F]\n"
            f"         [--policy CKPT] [--policy-n-action-steps N]  human-gated "
            f"DAgger, see policy_driver.py\n"
            f"         [--replicate SNAPDIR] [--replicate-episodes 0,2,7]  "
            f"reproduce a rollout's scene positions, see scene_snapshots.py\n"
            f"  (every flag above is consumed before startup, which is why an "
            f"unknown one lands here rather than in the episode index)")
    ## A starting hint only: create_dataset() always opens a FRESH run folder,
    ## so the dataset's own episode counter starts at 0 regardless, and 'r'
    ## below replaces this with that authoritative index.
    try:
        episode_idx = int(_pos[0]) if _pos else 0
    except ValueError:
        raise SystemExit(
            f"episode index must be an integer, got '{_pos[0]}'")
    gripper_actions = {arm: 0.1 for arm in ARM_MODES[mode]}
    
    _resume_run = (resolve_resume_run(task_name, _resume.strip())
                   if _resume else None)
    ## The checkpoint records which dataset it trained on; that dataset's
    ## meta.json carries the waist branch the policy expects.
    _policy_ds_root = None
    if _policy_ckpt:
        ## lerobot writes train_config.json INSIDE pretrained_model/ (that is
        ## what train_real.py --resume points --config_path at).
        try:
            _tc = json.loads((Path(_policy_ckpt) / "train_config.json").read_text())
            _policy_ds_root = _tc["dataset"]["root"]
        except Exception as _exc:
            print(f"[dagger] could not find the policy's training dataset "
                  f"({_exc}); waist canonicalization will be SKIPPED")

    dataset, dataset_root = create_dataset(task_name, mode, active_cameras,
                                           cfg.control_dt,
                                           resume_run=_resume_run)
    ## Appending to a SINGLE-task run: stamp the new frames with the string
    ## it already uses, so the dataset stays one task.  (The flower run was
    ## recorded under the stale name 'active_vision_data_collection'; a
    ## second string would just split tasks.parquet for nothing.)  Never
    ## overrides an explicit --target.
    if _resume_run is not None and task_ctx["target"] is None:
        try:
            import pandas as _pd
            _t = _pd.read_parquet(Path(dataset_root) / "meta" / "tasks.parquet").reset_index()
            _names = [str(x) for x in _t["task"]]
            if len(_names) == 1 and _names[0] != task_ctx["task"]:
                print(f"[resume] frames will carry the run's existing task "
                      f"string {_names[0]!r} (was going to write {task_ctx['task']!r})")
                task_ctx["task"] = _names[0]
        except Exception as _exc:
            print(f"[resume] could not read tasks.parquet ({_exc}); "
                  f"writing {task_ctx['task']!r}")

    # Camera calibration travels WITH the dataset.  Intrinsics are specific to
    # the device and the resolution, so a recording that does not carry its own
    # is not undistortable afterwards -- you can no longer tell which camera at
    # which settings produced it.  Written once, at session start.
    try:
        intr_path = save_camera_intrinsics(dataset_root / "camera_intrinsics.json")
        print(f"[calib] camera intrinsics -> {intr_path} "
              f"({len(CAMERA_INTRINSICS)} camera(s))")
    except Exception as exc:
        print(f"[calib] could not save camera intrinsics: {exc}")
        if REQUIRE_CALIBRATION:
            raise

    ## The RESOLVED config travels with the dataset, same reasoning as the
    ## intrinsics above: control rate, scales, clamps and every GIAVA_* env
    ## override shaped this data, and none of them are recoverable from the
    ## frames afterwards.
    _extra = {"gripper_mode": GRIPPER_MODE,
              "start_pose": start_pose,
              "quit_pose": quit_pose,
              ## The session DEFAULT.  Per-episode start poses are in
              ## meta/robustness.jsonl, because the 'i' keys can change it
              ## between episodes.
              "available_poses": _pose_names}
    if tube_filters is not None:
        from tube_mpc_hook import stats_summary as _tube_summary
        _extra["tube_mpc"] = {_a: _tube_summary(_f) for _a, _f in tube_filters.items()}
    _cfg_path = write_session_config(dataset_root, cfg, extra=_extra)
    if _cfg_path:
        print(f"[log] resolved teleop config -> {_cfg_path} "
              f"(gripper mode: {GRIPPER_MODE})")

    ## Thin adapter over lerobot's own streaming encoder (dataset.py): frames
    ## reach the per-camera encoder threads as they are recorded, so saving an
    ## episode does not block the control loop.
    episode_saver = BackgroundEpisodeSaver(dataset)

    ## finalize() is what makes the recorded episodes LOADABLE -- without it
    ## the parquet footers are never written and nothing can read the dataset
    ## back, replay included.  It used to run only on the 'q' path, so any
    ## other way out of a session (a traceback, Ctrl-C, roscore going away)
    ## silently cost every episode recorded in that run.  close() is
    ## idempotent, so the 'q' path still works exactly as before.
    import atexit
    atexit.register(episode_saver.close)

    # print(f"Dataset: {dataset}")
    # print(f"Dataset root: {dataset_root}")

    ## Turns a gap in the headset uplink into a release rather than a resume.
    ## That matters far more over Tailscale than on a LAN -- see LinkGuard in
    ## data_col_config.py.
    link_guard = LinkGuard()

    next_tick = now()

    _slog = solver_log_summary()
    if _slog:
        print(_slog)

    _phase("dataset + config snapshot")
    print(f"[startup] total {now() - _startup_t0:.1f}s to READY")
    ## --- shape_sorter piece helpers ------------------------------------
    _PIECE_KEYS = {pc[0] + "r": pc for pc in _ss.CLASS_NAMES} if task_ctx["perms"] else {}

    def _resolve_piece(tok):
        """'triangle', 'tri' or 't' -> 'triangle'; None if ambiguous/unknown."""
        tok = (tok or "").strip().lower()
        if tok in _ss.CLASSES:
            return tok
        hits = [pc for pc in _ss.CLASS_NAMES if pc.startswith(tok)] if tok else []
        return hits[0] if len(hits) == 1 else None

    def _announce_target():
        n = len(task_ctx["perms"][0])
        counts = "  ".join(f"{pc}:{task_ctx['saved'].get(pc, 0)}"
                           for pc in _ss.CLASS_NAMES)
        print(f"\n>>> NEXT: {task_ctx['target'].upper()}   "
              f"(round {task_ctx['round'] + 1}, piece {task_ctx['slot'] + 1}/{n})"
              f"   press r   [{counts}]")

    def _advance_order():
        """Step to the next piece of the round; call for a SAVED episode only
        (a discard should repeat the same piece, not skip it)."""
        if not task_ctx["perms"] or not task_ctx.get("cycle", True):
            return
        n = len(task_ctx["perms"][0])
        task_ctx["slot"] += 1
        if task_ctx["slot"] >= n:
            task_ctx["slot"] = 0
            task_ctx["round"] += 1
            print("\n" + "=" * 62)
            print("  ROUND COMPLETE -- reset the scene:")
            print("    retrieve the pieces from the box")
            print("    place all of them on NEW grid cells")
            print("=" * 62)
        pm = task_ctx["perms"][task_ctx["round"] % len(task_ctx["perms"])]
        task_ctx["target"] = pm[task_ctx["slot"]]
        task_ctx["task"] = _ss.task_string(task_ctx["target"])
        _announce_target()

    ## Built here: after the cameras and arms are up, before the loop.
    policy_driver = None
    _pd_extractor = None
    if _policy_ckpt:
        try:
            from .policy_driver import PolicyDriver
        except ImportError:
            from policy_driver import PolicyDriver
        _pd_task = task_ctx["task"] if task_ctx["target"] else None
        policy_driver = PolicyDriver(
            _policy_ckpt, mode, arm_names, device="cuda",
            dataset_root=_policy_ds_root, task_string=_pd_task,
            n_action_steps=(int(_policy_steps) if _policy_steps else None))
        _missing = [c for c in policy_driver.cameras if c not in active_cameras]
        if _missing:
            raise SystemExit(
                f"--policy needs camera(s) {_missing}, which --mode {mode} "
                f"does not open. Collect in the mode the policy was trained in.")
        print(f"[dagger] policy loaded: {policy_driver.describe()}")

        ## OBJECT-CENTRIC FEATURES FOR THE DAGGER LOOP.  A checkpoint trained
        ## with --env-state takes observation.environment_state as a real
        ## input; policy_driver.act() refuses to act without it.  The rollout
        ## computes it live from top_scene (rollout_policy.py, make_extractor)
        ## and DAgger has to compute it THE SAME WAY -- otherwise the policy
        ## you are correcting is not the policy you evaluated, and the
        ## corrections teach the wrong thing.  Note the recorder still does
        ## NOT write the column: run build_envstate_dataset.py on the result
        ## before training on it.
        _pd_env_scene = None
        if policy_driver.env_key is not None:
            _pd_env_scene = "shape_sorter" if task_ctx["target"] else "flower"
            _pd_want = int(np.prod(
                policy_driver.cfg.input_features[policy_driver.env_key].shape))
            try:
                from .rollout_policy import make_extractor as _make_extractor
            except ImportError:
                from rollout_policy import make_extractor as _make_extractor
            _pd_extractor = _make_extractor(
                _pd_env_scene, task_ctx["target"], _pd_want)
            if _pd_want != _pd_extractor.FEATURE_DIM:
                raise SystemExit(
                    f"--policy takes a {_pd_want}-d {policy_driver.env_key} "
                    f"but the {_pd_env_scene} extractor emits "
                    f"{_pd_extractor.FEATURE_DIM}.  Wrong scene for this "
                    f"checkpoint.")
            if "top_scene" not in active_cameras:
                raise SystemExit(
                    "--policy needs env-state, which is computed from "
                    "top_scene -- and this session did not open that "
                    "camera.  top_scene is gated by camera_config."
                    "top_active, not by --mode.")
            print(f"[dagger] env-state: feeding {policy_driver.env_key} "
                  f"({_pd_want}-d, scene={_pd_env_scene}"
                  + (f", target={task_ctx['target']}"
                     if task_ctx["target"] else "") + ")")

        def _pd_env_vec(frames):
            """The env-state vector for this tick, or (None, why).  Prefers
            the tick's own top_scene frame so the features describe the same
            instant as the image tokens."""
            if _pd_extractor is None:
                return None, None
            img = frames.get("top_scene")
            if img is None:
                with frame_lock:
                    img = latest_frames.get("top_scene")
                    img = img.copy() if img is not None else None
            if img is None:
                return None, "no top_scene frame for env-state"
            return _pd_extractor(img), None
        ## One throwaway inference now, so the first REAL tick does not pay
        ## the ~180 ms CUDA warmup (measured: policy_ms max 179 ms on the
        ## first tick, 0.8 ms mean after).  Waits briefly for the cameras.
        _t_wait = now()
        while (now() - _t_wait) < 3.0:
            with frame_lock:
                _wf = {c: latest_frames.get(c) for c in policy_driver.cameras}
            if all(v is not None for v in _wf.values()):
                break
            time.sleep(0.05)
        if all(v is not None for v in _wf.values()):
            _ws = np.concatenate([
                np.concatenate([np.asarray(robots[a].dxl.joint_states.position[
                    :ARM_CONFIG[a]["num_joints"] + 1], dtype=np.float32)])
                if ARM_CONFIG[a]["has_gripper"] else
                np.asarray(robots[a].dxl.joint_states.position[
                    :ARM_CONFIG[a]["num_joints"]], dtype=np.float32)
                for a in arm_names])
            _t0w = now()
            _wv, _ = _pd_env_vec({c: v.copy() for c, v in _wf.items()})
            policy_driver.act(_ws, {c: v.copy() for c, v in _wf.items()},
                              env_vec=_wv)
            policy_driver.reset()
            print(f"[dagger] warmed up ({(now() - _t0w) * 1e3:.0f} ms, excluded)")
        else:
            print("[dagger] cameras not up yet -- first policy tick will be slow")
        print("[dagger] press p to PLAY (policy drives); hold A/X to take an "
              "arm over, release to hand it back; p again to PAUSE.")

    ## Scene replication state (see --replicate above).  `cams` are the
    ## fixed scene cameras -- the wrist camera moves with the arm and is no
    ## alignment reference.  `i` is the cursor into `eps`; it advances on a
    ## SAVED episode only, so a discarded or repeated attempt stays on the
    ## same position.  `rep <n>` / `rep next` / `rep list` move it by hand.
    try:
        from .scene_snapshots import (save_snapshot, load_snapshot,
                                      alignment_view, list_episodes)
    except ImportError:
        from scene_snapshots import (save_snapshot, load_snapshot,
                                     alignment_view, list_episodes)
    _pg = {"on": _place_grid, "tick": 0, "window": False, "warned": False}
    _rep = {"dir": None, "eps": [], "i": 0, "cams": [], "tick": 0,
            "cache": (None, {}), "window": False}
    _rep["cams"] = [c for c in active_cameras if not c.endswith("_wrist")]
    _own_snap_dir = Path(dataset_root) / "snapshots"
    if _replicate:
        _rep["dir"] = Path(_replicate)
        if not _rep["dir"].exists():
            raise SystemExit(f"--replicate: {_rep['dir']} does not exist")
        _all = list_episodes(_rep["dir"])
        if _replicate_eps:
            _rep["eps"] = [int(x) for x in str(_replicate_eps).replace(" ", ",").split(",") if x]
            _bad = [e for e in _rep["eps"] if e not in _all]
            if _bad:
                raise SystemExit(f"--replicate-episodes: {_bad} have no snapshot in "
                                 f"{_rep['dir']} (available: {_all})")
        else:
            _rep["eps"] = _all
        if not _rep["eps"]:
            raise SystemExit(f"--replicate: no ep*_<camera>.png in {_rep['dir']}")
        if not _rep["cams"]:
            raise SystemExit("--replicate needs a scene camera open: add "
                             "--scene-cams top,low")
        print(f"[replicate] {len(_rep['eps'])} reference position(s) from "
              f"{_rep['dir']}: {_rep['eps']}")
        print(f"[replicate] ghost window shows {_rep['cams']} vs reference; "
              f"starts at ep{_rep['eps'][0]:02d}.  rep <n> / rep next / rep list")
    print(f"[snapshot] each episode's starting scene -> {_own_snap_dir}")

    def _rep_current():
        """(reference dir, reference episode) or (None, None)."""
        if _rep["dir"] is None or not _rep["eps"]:
            return None, None
        return _rep["dir"], _rep["eps"][min(_rep["i"], len(_rep["eps"]) - 1)]

    def _rep_close_window():
        if _rep["window"]:
            try:
                import cv2
                cv2.destroyWindow("setup")
                cv2.pollKey()
            except Exception:
                pass
            _rep["window"] = False

    print("READY!")
    print("  keys: r record   ss/sf save success/failure   d discard   "
          f"i re-park ({start_pose})   sync re-read arms   q quit ({quit_pose})"
          f"   qh quit holding position"
          + ("   cr/tr/fr set piece + record   target <piece> switch   "
             "next skip a piece" if task_ctx["target"] else ""))
    if task_ctx["perms"]:
        print("  ss/sf take an optional piece to RELABEL the episode being "
              "saved, e.g. 'ss triangle' or 'ss t'")
    _short = ", ".join(f"i{k} {v}" for k, v in sorted(POSE_SHORTCUTS.items())
                       if v in _pose_names)
    _spelled = [a for a in _pose_names if a not in POSE_SHORTCUTS.values()]
    print(f"  park at:  {_short}"
          + (f"   |  spell out: {', '.join('i' + a for a in _spelled)}"
             if _spelled else ""))
    print("  servo: e health report   profile   shadow   clear release a hold"
          "   reboot -> reboot! clear a latched fault")

    ## Dead-man timer state (GIAVA_MAX_ENGAGE_S); inert when it is 0.
    _teleop_since = now()
    _engage_latched = False

    while not rospy.is_shutdown():
        loop_start = now()

        key = latest_key
        latest_key = None

        ## RE-PARK, AND CHOOSE WHERE.  'i' alone goes to the session's
        ## --start-pose; 'i' plus a pose selector goes anywhere every active
        ## arm can reach:
        ##
        ##   if -> forward    ih -> high    ir -> rest    il -> low
        ##   ifar_scene / ilooking_left / ilooking_right   (middle arm only)
        ##
        ## WHY THIS IS A KEY AND NOT JUST A FLAG: the pose an episode starts
        ## from IS part of the training distribution -- a policy whose
        ## demonstrations all begin at "forward" has never seen any other
        ## starting state, and a rollout launched from one is out of
        ## distribution before it takes a step.  Deliberately varying the
        ## start pose across a session is how that gets fixed, and it has to
        ## be doable between episodes, without restarting the session and
        ## losing the run.  Which pose each episode actually started from is
        ## recorded per episode in robustness.jsonl.
        ##
        ## No other key starts with 'i', so any 'i...' token lands here.
        if key and key[0] == "i":
            ## GUARD FIRST, before touching the drive session.  This used to
            ## call stop_teleop_session() and only then notice an episode was
            ## in progress -- so a mistyped 'i' mid-episode cut teleop out and
            ## left the operator holding a half-recorded demonstration.  Now a
            ## re-park request during an episode changes nothing at all.
            if collecting_episode:
                print("\nIn the middle of collecting episode. Press ss / sf to "
                      "save and stop before re-parking.")
                continue

            _sel = key[1:].strip()
            if _sel:
                _target, _err = resolve_pose_key(_sel, _pose_names)
                if _err:
                    print(_err)
                    continue
            else:
                _target = start_pose

            stop_teleop_session(teleop_state)

            ## One ramp for every arm, not a loop -- see
            ## robot_control.move_arms_together.
            park_arms(robots, arm_names, _target)
            current_start_pose = _target

            # Allow the latest ROS joint-state message to arrive after the final motion.
            rospy.sleep(0.05)

            sync_robot_state(robots=robots, robot=robot, arm_data=arm_data,
                             arm_names=arm_names, cmd_kin=cmd_kin, cmd_state=cmd_state,
                             to_urdf=coupled_ik.driver_to_urdf)

            print(f"\nREADY  (parked at '{_target}')")
            continue

        ## The piece shortcuts exist only for shape_sorter, and typing one in
        ## a session started without --task is the easy mistake: every handler
        ## below just ignores an unknown key, so it looked like the feature was
        ## broken rather than absent.  Say which flag is missing.
        if key in ("cr", "tr", "fr", "next") and not task_ctx["perms"]:
            print(f"\n'{key}' needs --task {_ss.TASK_NAME}; this session is "
                  f"'{task_name}'. Restart with:\n"
                  f"    python data_collection.py --mode {mode} "
                  f"--task {_ss.TASK_NAME} --target cube --episodes 50")
            continue

        ## PIECE SHORTCUTS (shape_sorter only).  'cr' / 'tr' / 'fr' set the
        ## target AND start recording in one keystroke, because the target has
        ## to be right BEFORE 'r' and the two-step (target x, then r) is one
        ## more thing to forget when the piece order changes every round.
        if key and task_ctx["target"] and key in _PIECE_KEYS:
            _piece = _PIECE_KEYS[key]
            if collecting_episode:
                print(f"\nAlready recording. Press ss / sf to save (add a "
                      f"piece name to relabel) or d to discard.")
                continue
            task_ctx["target"] = _piece
            task_ctx["task"] = _ss.task_string(_piece)
            key = "r"          # fall through to the record handler below

        ## 'p': PLAY / PAUSE the policy (DAgger).  Play arms the recording
        ## (frames record from here, as they do from a teleop enable) and
        ## drops any queued chunk so the first motion is planned from the
        ## scene as it is now.  Pause leaves every arm holding.
        if key == "p" and policy_driver is not None:
            if not collecting_episode:
                print("\nPress r first -- the policy only drives inside an episode.")
                continue
            dagger_play = not dagger_play
            if dagger_play:
                policy_driver.reset()
                if _pd_extractor is not None:
                    _pd_extractor.reset()
                if not episode_armed:
                    episode_armed = True
                print("\n[dagger] >>> PLAY: POLICY driving.  Hold A (right) / X "
                      "(left) to take over; release to hand back; p to pause.")
            else:
                print("\n[dagger] || PAUSE: arms holding.  p to resume, ss/sf "
                      "to save, d to discard.")
            continue

        ## 'rep <n>' / 'rep next' / 'rep list': move the scene-replication
        ## cursor by hand (a position to redo, or one to skip).
        if key and key.split()[0] == "rep":
            if _rep["dir"] is None:
                print("\nNo --replicate directory for this session.")
                continue
            _arg = key.split()[1] if len(key.split()) > 1 else "list"
            if _arg == "list":
                print("\n[replicate] " + "  ".join(
                    f"{'>' if k == _rep['i'] else ' '}ep{e:02d}"
                    for k, e in enumerate(_rep["eps"])))
            elif _arg == "next":
                _rep["i"] = min(_rep["i"] + 1, len(_rep["eps"]) - 1)
                print(f"\n[replicate] now ep{_rep['eps'][_rep['i']]:02d}")
            else:
                try:
                    _e = int(_arg)
                except ValueError:
                    print(f"\n[replicate] rep <n>|next|list, got {_arg!r}")
                    continue
                if _e in _rep["eps"]:
                    _rep["i"] = _rep["eps"].index(_e)
                elif _e in list_episodes(_rep["dir"]):
                    _rep["eps"].insert(_rep["i"], _e)
                    print(f"[replicate] ep{_e:02d} was not in the list; inserted")
                else:
                    print(f"\n[replicate] no snapshot for ep{_e:02d} in {_rep['dir']}")
                    continue
                print(f"\n[replicate] now ep{_rep['eps'][_rep['i']]:02d}")
            continue

        ## 'next': step the round order on WITHOUT recording -- a piece is
        ## out of reach, already in the box, or the round got out of step.
        if key == "next" and task_ctx["perms"]:
            if collecting_episode:
                print("\nFinish or discard the current episode first.")
                continue
            if policy_driver is not None:
                _tot = _dagger_frames["policy"] + _dagger_frames["human"]
                if _tot:
                    print(f"  [dagger] {_dagger_frames['human']}/{_tot} frames "
                          f"({_dagger_frames['human'] / _tot:.0%}) were YOUR "
                          f"correction; the rest the policy drove")
            _advance_order()
            continue

        ## 'sync': re-seed the IK / command state from the MEASURED joints
        ## without moving anything -- after the arms were driven from another
        ## terminal (move_arms.py) or nudged by hand between episodes.
        if key == "sync":
            if collecting_episode:
                print("\nFinish or discard the current episode first.")
                continue
            stop_teleop_session(teleop_state)
            rospy.sleep(0.05)
            sync_robot_state(robots=robots, robot=robot, arm_data=arm_data,
                             arm_names=arm_names, cmd_kin=cmd_kin, cmd_state=cmd_state,
                             to_urdf=coupled_ik.driver_to_urdf)
            print("\nREADY  (synced to measured joint state, arms not moved)")
            continue

        ## 'target <piece>' switches which piece the NEXT episodes are about
        ## (shape_sorter only).  Refused mid-episode: the task string is
        ## attached to every frame, so it must not change under a recording.
        if key and key.split()[:1] == ["target"]:
            if task_ctx["target"] is None:
                print(f"\n'target' only applies to --task {_ss.TASK_NAME}.")
                continue
            if collecting_episode:
                print("\nFinish or discard the current episode first "
                      "(ss / sf / d), then switch target.")
                continue
            _parts = key.split()
            if len(_parts) != 2 or _parts[1] not in _ss.CLASSES:
                print(f"\nUsage: target <{'|'.join(_ss.CLASS_NAMES)}>"
                      f"   (now: {task_ctx['target']})")
                continue
            task_ctx["target"] = _parts[1]
            task_ctx["task"] = _ss.task_string(_parts[1])
            print(f"\nTARGET -> {_parts[1]}   task string: '{task_ctx['task']}'")
            continue

        ## ONE COMMAND, VERDICT INCLUDED: 'ss' saves the episode as a
        ## SUCCESS, 'sf' saves it as a FAILURE.  Both are kept -- a failed
        ## demonstration is still data, and episode_outcomes.jsonl carries the
        ## label so the loader can filter or weight them at training time.
        ## 'd' still throws an episode away.  No prompt to answer: the operator
        ## already knows how it went, and a modal question in the middle of a
        ## live session is one more thing to get wrong.
        _save_parts = key.split() if key else []
        if _save_parts and _save_parts[0] in ("s", "ss", "sf"):
            if _save_parts[0] == "s":
                print("\nAmbiguous -- ss = save as success, sf = save as "
                      "failure, d = discard.")
                continue
            if not collecting_episode:
                print("\nNo active episode to save.")
                continue
            outcome = "success" if _save_parts[0] == "ss" else "failure"

            ## LAST CHANCE TO FIX THE LABEL: 'ss triangle' (or 'ss t') saves
            ## this episode as a triangle episode whatever the target was
            ## while it recorded.  The task string is stamped per frame, so
            ## without this a wrong target means re-doing the episode -- and a
            ## mislabelled one that slips through teaches the policy to fetch
            ## the wrong object on command.
            _task_override = None
            if len(_save_parts) > 1:
                _fix = _resolve_piece(_save_parts[1])
                if _fix is None:
                    print(f"\nUnknown piece {_save_parts[1]!r} -- one of "
                          f"{_ss.CLASS_NAMES}. Episode NOT saved.")
                    continue
                if _fix != task_ctx["target"]:
                    _task_override = _ss.task_string(_fix)
                    print(f"\nRELABEL: this episode -> {_fix} "
                          f"(was {task_ctx['target']})")
                    task_ctx["target"] = _fix
                    task_ctx["task"] = _task_override
            stop_teleop_session(teleop_state)
            collecting_episode = False
            episode_armed = False
            ## save_episode_async() detaches this episode's buffer AND its
            ## encoder, hands them to the writer thread and returns in
            ## microseconds, having already installed a fresh buffer and a
            ## fresh encoder for the next episode.  See dataset.py: that swap
            ## is what lets 'r' come straight back while the previous episode
            ## is still being written, without the two ending up in one file.
            ## It still raises on an empty buffer (r then ss with no frames in
            ## between); that must not take the session down, the arms are
            ## live.
            try:
                saved_idx = episode_saver.save_episode_async(
                    outcome, task_override=_task_override)
            except Exception as exc:
                print(f"\nNOTHING TO SAVE ({exc}) -- episode dropped.")
                episode_stats = reset_episode_log(active_cameras, arm_names)
                continue
            episode_idx = saved_idx
            log_episode_info(episode_idx, episode_stats)
            ## Tracking quality belongs WITH the episode: an episode recorded
            ## through a dozen controller discontinuities is a different kind
            ## of data from a clean one, and this is the only place that fact
            ## is ever visible.
            _tsum = jump_gate.summary()
            if _tsum:
                print(_tsum)
            ## Per-episode gate deltas (counters are session-cumulative).
            _gate_deltas = {
                "discontinuities": {
                    k: v - gate_baseline[0].get(k, 0)
                    for k, v in jump_gate.events.items()
                    if v - gate_baseline[0].get(k, 0) > 0},
                "ticks_held": {
                    k: v - gate_baseline[1].get(k, 0)
                    for k, v in jump_gate.holds.items()
                    if v - gate_baseline[1].get(k, 0) > 0},
            }
            if tube_filters is not None:
                from tube_mpc_hook import stats_summary as _tube_summary
                _gate_deltas["tube_mpc"] = {
                    _a: _tube_summary(_f) for _a, _f in tube_filters.items()}
                for _a, _f in tube_filters.items():
                    ## per-episode: the hook's lists are session-cumulative
                    _f.stats.solve_ms.clear()
                    _f.stats.intervention_rad.clear()
            write_episode_robustness(dataset_root, saved_idx, outcome,
                                     episode_stats, gate_deltas=_gate_deltas,
                                     extra={"start_pose": current_start_pose,
                                            "snapshot_dir": str(_own_snap_dir),
                                            "replicate_of": (
                                                f"{_rep_current()[0]}/ep{_rep_current()[1]:02d}"
                                                if _rep_current()[0] is not None else None)})
            if TRACK_LOG:
                _ld = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "trajectories", "logs")
                os.makedirs(_ld, exist_ok=True)
                try:
                    from study_ik import describe_weights as _dw
                    _w = _dw()
                except Exception:
                    _w = "unknown"
                for _a in arm_names:
                    _rows = getattr(episode_stats.arms[_a], "track_rows", [])
                    if len(_rows) > 5:
                        _lp = os.path.join(
                            _ld, f"dc_ep{episode_idx:04d}_{_a}_"
                            f"{1.0 / cfg.arm_cmd_dt:g}hz_"
                            f"{time.strftime('%H%M%S')}.npz")
                        np.savez(_lp, rows=np.asarray(_rows),
                                 rate=1.0 / cfg.arm_cmd_dt,
                                 scale=cfg.position_scale, arm=_a,
                                 traj=f"episode_{episode_idx}", weights=_w)
                        print(f"[track] wrote {os.path.basename(_lp)}")
            ## The DATASET decides the next index, not a local counter.
            ## They diverge the first time an episode is discarded --
            ## a discard reuses the index, a counter does not -- and from then
            ## on every logged episode number, every track-log filename and
            ## anything the operator writes down names a different episode
            ## than `--episode-idx` will replay.
            episode_idx = episode_saver.next_episode_index
            _tgt = task_ctx["target"] or task_name
            task_ctx["saved"][_tgt] = task_ctx["saved"].get(_tgt, 0) + 1
            _n = task_ctx["saved"][_tgt]
            _progress = (f"  [{_n}/{task_ctx['goal']} {_tgt}]" if task_ctx["goal"]
                         else f"  [{_n} {_tgt} this session]")
            print(f"\nepisode_{saved_idx:04d} HANDED TO WRITER ({outcome}) -- "
                  f"ready for episode_{episode_idx:04d}, press r{_progress}")
            if policy_driver is not None:
                _tot = _dagger_frames["policy"] + _dagger_frames["human"]
                if _tot:
                    print(f"  [dagger] {_dagger_frames['human']}/{_tot} frames "
                          f"({_dagger_frames['human'] / _tot:.0%}) were YOUR "
                          f"correction; the rest the policy drove")
            _advance_order()
            if _rep["dir"] is not None:
                if _rep["i"] + 1 < len(_rep["eps"]):
                    _rep["i"] += 1
                    print(f"  [replicate] next: set the scene to "
                          f"ep{_rep['eps'][_rep['i']]:02d} "
                          f"({_rep['i'] + 1}/{len(_rep['eps'])})")
                else:
                    print(f"  [replicate] all {len(_rep['eps'])} reference "
                          f"positions done; staying on ep{_rep['eps'][-1]:02d} "
                          f"(rep <n> to revisit)")
            if task_ctx["goal"] and _n >= task_ctx["goal"]:
                print(f"  goal reached for {_tgt}: "
                      + (f"`target <piece>` to switch, or " if task_ctx["target"] else "")
                      + "q to finish (arms park at "
                      f"'{quit_pose}')   session so far: {task_ctx['saved']}")
            continue

        if key == "d":
            if not collecting_episode:
                print("\nNo active episode to discard.")
            else:
                print("\nDiscarding the current episode.")
                stop_teleop_session(teleop_state)
                collecting_episode = False
                episode_armed = False
                ## Cancels the in-flight streaming encode and drops the
                ## buffer.  Episodes already saved are untouched.
                episode_saver.discard_current_episode()
                ## Do NOT advance the index: clear_episode_buffer() reuses it,
                ## so the next save really is this same episode number.
                ## And KEEP the return value -- this used to set episode_stats
                ## to None and drop the fresh log on the floor, so the next
                ## tick's episode_stats.headset.append() killed the session
                ## every time an episode was discarded.
                episode_stats = reset_episode_log(active_cameras, arm_names)
                print(f"\nEPISODE DISCARDED (episode_{episode_idx:04d} will be "
                      "reused)")
            continue

        if key == "r":
            if collecting_episode:
                print("\nAlready recording episode. Press ss / sf to save or "
                      "d to discard.")
            else:
                ## begin_episode() installs a private buffer and a private
                ## encoder for this episode, so recording it overlaps with
                ## whatever the writer thread is still finishing.  It returns
                ## immediately unless the writer has fallen
                ## GIAVA_MAX_PENDING_SAVES episodes behind, which is a memory
                ## bound, not a correctness one -- see dataset.py.
                episode_idx = episode_saver.begin_episode()
                ## The scene the operator committed to IS the episode's
                ## start; snapshot it now so this position can be replicated
                ## later (by an evaluation or another correction pass).
                _rep_close_window()
                if _rep["cams"]:
                    with frame_lock:
                        _snap = {c: (latest_frames.get(c).copy()
                                     if latest_frames.get(c) is not None else None)
                                 for c in _rep["cams"]}
                    try:
                        save_snapshot(_own_snap_dir, episode_idx, _snap)
                    except Exception as _exc:
                        print(f"    [snapshot] not saved: {_exc}")
                _rd, _re = _rep_current()
                if _rd is not None:
                    print(f"    [replicate] this episode reproduces {_rd.name}/ep{_re:02d}")
                ## A queued action chunk describes the PREVIOUS episode's
                ## scene; carrying it across a boundary would drive the arm
                ## from a stale observation for up to a full chunk.
                if policy_driver is not None:
                    policy_driver.reset()
                    if _pd_extractor is not None:
                        _pd_extractor.reset()
                    _dagger_frames["policy"] = _dagger_frames["human"] = 0
                    _dagger_prev_human[0] = frozenset()
                    dagger_play = False
                    print("    [dagger] recording; press p when you want the "
                          "policy to start")
                collecting_episode = True
                episode_armed = False

                episode_stats = reset_episode_log(active_cameras, arm_names)
                gate_baseline = (dict(jump_gate.events), dict(jump_gate.holds))

                ## Announce the dataset's index, so what the operator writes in
                ## the notebook matches what --episode-idx replays.  The TARGET
                ## goes on the same line: it is stamped onto every frame from
                ## here on and cannot be changed once recording starts, so a
                ## forgotten `target <piece>` would silently mislabel the whole
                ## episode -- and a mislabelled episode teaches the policy to
                ## fetch the wrong object on command.
                _tgt_note = (f"   TARGET: {task_ctx['target']}"
                             if task_ctx["target"] else "")
                print(f"\nCOLLECTING: episode_{episode_idx:04d}{_tgt_note}")

        ## Ask whether anything is still being written.  Cheap, always
        ## available, and the answer is the one fact the operator cannot get
        ## any other way once saving stopped blocking the loop.
        if key in ("?", "status"):
            print("\n" + episode_saver.status_line())
            continue

        # Live stereo-comfort tuning (headset view only; dataset unaffected).
        #   , / .  : pull the eye images together / apart (convergence)
        #   - / =  : shrink / grow the per-eye image (zoom comfort)
        # Values print on every press; paste the final ones into
        # camera_manager.py EYE_VIEW defaults.
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
            for arm in arm_names:
                servo_report(robots[arm], arm)
            continue

        if key == "profile":
            for arm in arm_names:
                servo_check_profile(robots[arm], arm)
            continue

        if key == "shadow":
            for arm in arm_names:
                servo_check_shadows(robots[arm], arm)
            continue

        if key == "clear":
            ## Releases the WATCHDOG's hold only; writes nothing to any motor.
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

        if key == "sync":
            ## Adopt the encoders as truth, on demand -- for an arm moved by
            ## something OUTSIDE this loop (move_arms.py, hand-guiding with
            ## torque off, a named-pose script) while this session sat idle.
            ## The button-press path a few hundred lines down already does
            ## this unconditionally on every re-enable; this is the same
            ## call exposed directly, for when re-checking without a
            ## press/release cycle is more convenient -- and unlike 'clear',
            ## it does not touch servo fault flags.
            if collecting_episode:
                print("\n[sync] finish or discard the episode first (ss / "
                      "sf / d) -- resyncing mid-episode would record a "
                      "position jump as demonstration data.")
                continue
            sync_robot_state(robots=robots, robot=robot, arm_data=arm_data,
                             arm_names=arm_names, cmd_kin=cmd_kin,
                             cmd_state=cmd_state,
                             to_urdf=coupled_ik.driver_to_urdf)
            print("\n[sync] state resynced from measured joints.")
            continue

        if key == "reboot":
            ## STEP 1 of 2 -- diagnose only, nothing is written.  A reboot
            ## takes the faulted motors' torque OFF for ~0.5 s; a motor
            ## CANNOT hold position while it reboots, so the arm sags.
            if collecting_episode:
                print("\n[servo health] finish or discard the episode first "
                      "(ss / sf / d) -- rebooting mid-episode would record a "
                      "torque-off sag as demonstration data.")
                continue
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
                    _mate = ""
                    for _sh in ("shoulder_shadow", "elbow_shadow"):
                        _ms = _sh.replace("_shadow", "")
                        if name in (_sh, _ms):
                            _mate = f"  (+ its pair: {_sh if name == _ms else _ms})"
                    print(f"    {arm:<8s} {name:<18s} "
                          f"{servo_decode(_bits)}{_mate}")
            print("  HOLD THE ARM NOW if any of these carries its weight.\n"
                  "  Type  reboot!  to proceed, anything else to cancel.")
            continue

        if key == "reboot!":
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
                    overload.energy.pop(arm, None)
            rospy.sleep(0.1)
            sync_robot_state(robots=robots, robot=robot, arm_data=arm_data,
                             arm_names=arm_names, cmd_kin=cmd_kin,
                             cmd_state=cmd_state,
                             to_urdf=coupled_ik.driver_to_urdf)
            print("\nREADY  (state resynced; press 'i' to re-park)")
            continue

        if key in ("q", "qh"):
            print("\nEXITING" + ("  (qh: arms hold position)" if key == "qh" else ""))

            episode_idx, did_shutdown = safe_shutdown(
                robots, pipelines, dataset, episode_saver, collecting_episode,
                dataset_root, episode_idx, quit_pose=quit_pose,
                hold=(key == "qh"), preview=preview)

            if did_shutdown:
                break

        # Teleoperating
        tick_counter += 1

        ## ---- servo health, every tick, off joint_states ---------------- ##
        for _arm in arm_names:
            try:
                _js = robots[_arm].dxl.joint_states
                _n = ARM_CONFIG[_arm]["num_joints"]
                _meas = np.asarray(_js.position[:_n], dtype=float)
                _eff = np.asarray(_js.effort[:_n], dtype=float)
            except Exception:
                continue
            if watchdog.dead_bus.get(_arm):
                continue          # readings are the failed-read sentinel
            _cmd = np.asarray(cmd_state.last_cmds.get(_arm, _meas), dtype=float)
            faultlog.push(_arm, now(), _cmd, _meas, _eff)
            _e = overload.update(_arm, _eff, cfg.control_dt)
            if _e > 0.35 and (tick_counter - _overload_last_print[_arm]) > 250:
                _overload_last_print[_arm] = tick_counter
                print(f"[overload] {overload.describe(_arm)}")
            ## STALL GATE.  The I^2-t model above describes the servo's
            ## thermal latch; this catches the failure that actually keeps
            ## killing the right arm, which is mechanical and which that
            ## model reads as barely half its budget until it is far too
            ## late.  Cheap: it reuses the readings already in hand.
            _was_held = stallgate.held.get(_arm, False)
            _now_held = stallgate.update(_arm, _cmd, _meas, _eff, cfg.control_dt)
            if _now_held and not _was_held:
                print(f"\n[stall] {stallgate.describe(_arm, _cmd, _meas, _eff)}")
                ## A waist stall is the one stall that carries information
                ## worth keeping: it locates the mechanical stop, which
                ## nothing else in the stack knows about.
                _wt = waist_travel.note_stall(
                    _arm, stallgate.culprit.get(_arm, -1), _cmd, _meas)
                if _wt:
                    print(f"  {_wt}")
                if collecting_episode:
                    print(f"  *** '{_arm}' stopped following. Everything "
                          f"recorded from here is commanded motion against a "
                          f"jammed arm -- press 'd' to DISCARD.")
            elif _was_held and not _now_held:
                print(f"[stall] {_arm} released -- the command is back where "
                      f"the arm is.")

        for _arm in watchdog.update(robots, cmd_state, now()):
            _faulted[_arm] = True
            print(f"\n[servo health] {watchdog.describe(_arm)}")
            ## An episode recorded against a limp or absent arm is worse than
            ## no episode: it looks valid on disk and teaches the wrong
            ## action-to-state mapping.  Say so loudly, at the moment it
            ## becomes true, while the operator can still discard it.
            if collecting_episode:
                print(f"  *** THIS EPISODE IS COMPROMISED. '{_arm}' stopped "
                      f"following partway through, so everything recorded "
                      f"since is commanded motion against an arm that was "
                      f"not doing it. Press 'd' to DISCARD.")
            if watchdog.dead_bus[_arm]:
                print(f"  Holding '{_arm}'.  Check 12 V power, the U2D2/USB "
                      f"cable, and `ls -l /dev/ttyDXL*`, then power-cycle "
                      f"that arm and restart. A software reboot cannot reach "
                      f"a motor that is not on the bus.")
                continue
            _p = faultlog.dump(_arm, reason="watchdog trip",
                               extra={"overload": overload.worst(_arm)[0],
                                      "in_episode": float(collecting_episode)})
            if _p is not None:
                print(f"  wrote the 20 s BEFORE the fault to {_p}")
                print(FaultRecorder.summarise(_p))
            _errs, _bad = servo_report(robots[_arm], _arm)
            if not _bad:
                print(f"  No latched fault: the registers are clean, so this "
                      f"is a MECHANICAL block or a bad command reference, not "
                      f"a dead servo. Type 'clear' to release the hold.")
        if tick_counter % 50 == 0:
            for _arm, _held in watchdog.hot_arms(now()):
                print(f"[servo health] {_arm} has been drawing "
                      f"{watchdog.effort_warn_ma:.0f}+ mA for {_held:.0f} s -- "
                      f"it is working toward an overload latch. Ease off, or "
                      f"check what it is pushing against.")
        ## ---------------------------------------------------------------- ##

        ## GHOST WINDOW between episodes: the live scene cameras blended
        ## 50/50 with the reference episode's frames, so the operator moves
        ## the object until its two images coincide.  ~10 Hz, and only while
        ## no episode is being recorded -- then the loop's 20 ms budget is
        ## not under pressure and the scene is not supposed to change.
        ##
        ## ABOVE THE HEADSET GUARD ON PURPOSE.  `if headset_data is None:
        ## continue` a few lines down skips the whole rest of the tick
        ## whenever the headset is not streaming -- which is exactly the
        ## situation this window exists for: the operator is at the table
        ## with the flower in hand and the headset on the bench.  Below that
        ## guard the window never appeared at all.
        if _pg["on"] and not collecting_episode:
            _pg["tick"] += 1
            if _pg["tick"] % 5 == 0:
                with frame_lock:
                    _ts = latest_frames.get("top_scene")
                    _ts = _ts.copy() if _ts is not None else None
                if _ts is None:
                    if _pg["tick"] % 250 == 0:
                        print("[place-grid] no top_scene frame yet")
                else:
                    try:
                        import cv2
                        try:
                            from . import place_grid as _pgm
                            from . import scene_features as _sfm
                        except ImportError:
                            import place_grid as _pgm
                            import scene_features as _sfm
                        _img = _pgm.draw(cv2.cvtColor(_ts, cv2.COLOR_RGB2BGR),
                                         _pgm.lattice(), _sfm.detect_object(_ts),
                                         8.0)
                        if not _pg["window"]:
                            cv2.namedWindow("placement", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("placement", _img.shape[1], _img.shape[0])
                            _pg["window"] = True
                            print("[place-grid] window open -- place on a green "
                                  "cell, never a purple one, then press r")
                        cv2.imshow("placement", _img)
                        cv2.waitKey(1)
                    except Exception as _exc:
                        if not _pg["warned"]:
                            _pg["warned"] = True
                            print(f"[place-grid] preview failed: {_exc!r} -- "
                                  f"the session still records")
        if _rep["dir"] is not None and not collecting_episode:
            _rep["tick"] += 1
            if _rep["tick"] % 5 == 0:
                _rd, _re = _rep_current()
                if _rep["cache"][0] != _re:
                    _rep["cache"] = (_re, load_snapshot(_rd, _re, _rep["cams"]))
                with frame_lock:
                    _live = {c: (latest_frames.get(c).copy()
                                 if latest_frames.get(c) is not None else None)
                             for c in _rep["cams"]}
                try:
                    import cv2
                    _img = alignment_view(
                        _live, _rep["cache"][1], _rep["cams"],
                        banner=f"replicate {_rd.name}/ep{_re:02d}   "
                               f"({_rep['i'] + 1}/{len(_rep['eps'])})   "
                               f"r when aligned   rep <n> to change")
                    if _img is None:
                        if _rep["tick"] % 250 == 0:
                            print(f"[replicate] no frames yet from {_rep['cams']}")
                    else:
                        if not _rep["window"]:
                            ## namedWindow first, as oak_preview does: on this
                            ## Qt build a bare imshow can come up unmapped.
                            cv2.namedWindow("setup", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("setup", _img.shape[1], _img.shape[0])
                            _rep["window"] = True
                            print(f"[replicate] ghost window open -- align to "
                                  f"ep{_re:02d}, then press r")
                        cv2.imshow("setup", _img)
                        cv2.waitKey(1)
                except Exception as _exc:
                    ## Loud the FIRST time.  A silent preview that is merely
                    ## broken looks exactly like one that is switched off.
                    if not _rep.get("warned"):
                        _rep["warned"] = True
                        print(f"[replicate] preview failed: {_exc!r} -- the "
                              f"session still records, you just have no ghost")

        t_headset_start = now()
        headset_data = headset.receive_data()
        episode_stats.headset.append(now() - t_headset_start)

        if headset_data is None:
            ## Holding the arms is right; leaving the session anchored to a
            ## pre-gap reference is not.  LinkGuard remembers when this started.
            link_guard.missed(now())
            continue

        link_guard.recovered(now(), teleop_state)

        head_mat_raw = pose2mat(headset_data.h_pos, headset_data.h_quat)
        # The probe (2026-08) measured the app world as z-up with an arbitrary
        # per-session yaw; the old fixed basis conversion here was wrong and is
        # gone.  All frame handling now happens through the session-yaw
        # calibration built at anchor time (data_col_config.session_yaw_remap).
        head_mat = head_mat_raw
        if HANDS_HEAD_RELATIVE:
            # The headset app reports CONTROLLER poses in the HEAD's frame, not
            # the world frame (symptom: standing up moves the gripper arms DOWN
            # -- the hands' head-relative height drops when the head rises).
            # Compose through the head pose to recover world-frame hands.
            # Never noticed while seated: with the head still, head-relative
            # deltas equal world deltas exactly.
            # Compose with the RAW head pose.  The probe (2026-08) measured the
            # app's world frame as z-up, so HEAD_BASIS_FIX (built on the y-up
            # hypothesis) must NOT be applied here -- with it, the step-3 leak
            # test showed up to 810 mm of fake hand motion from pure head
            # rotation, which was the gripper-arm drift.  Raw composition is
            # also the configuration the stand-up test validated.
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
        # GIAVA_DEBUG_HANDS=1: move ONLY your head (hands still), then ONLY one
        # hand, and read which numbers change.  If head-only motion changes
        # l_pos/r_pos, the app is reporting head-relative hands -- enable
        # GIAVA_HANDS_HEAD_RELATIVE=1.
        if DEBUG_HANDS and (tick_counter % 25 == 0):
            print(f"[hands] h={np.round(np.asarray(headset_data.h_pos), 3)} "
                  f"l={np.round(np.asarray(headset_data.l_pos), 3)} "
                  f"r={np.round(np.asarray(headset_data.r_pos), 3)}")

        for _side, _p in (("left", headset_data.l_pos), ("right", headset_data.r_pos)):
            _pv = np.asarray(_p, dtype=float).copy()
            if _frozen_prev[_side] is not None and np.array_equal(_pv, _frozen_prev[_side]):
                _frozen_ticks[_side] += 1
            else:
                _frozen_ticks[_side] = 0
            _frozen_prev[_side] = _pv
            if _frozen_ticks[_side] == 25:  # ~0.5 s bit-identical = asleep/untracked
                print(f"[tracking] {_side} controller pose is FROZEN -- asleep or "
                      "untracked. Wake it / bring it into view before enabling "
                      "that arm (probe measured 0.5 m of drift on re-acquire).")

        button_pressed = (headset_data.r_button_one or headset_data.l_button_one
                          or headset_data.l_button_two)

        # Per-arm buttons: X (left one) = left arm, A (right one) = right arm.
        # The camera arm follows the head whenever EITHER gripper arm is being
        # driven, and Y (left two) drives it alone -- so the camera can be
        # aimed before a grasp without holding a gripper button.  With
        # GIAVA_MIDDLE_BUTTON=y only the Y button counts and the camera stays
        # parked through incidental head motion.
        arm_active = {
            "left": headset_data.l_button_one,
            "right": headset_data.r_button_one,
        }
        # Tool parity: a frozen device (asleep / untracked, probe-measured 0.5 m
        # drift on re-acquire) must not drive an arm.  Refuse activation and
        # hold mid-session rather than only warning.
        for _arm, _side, _pos in (("left", "left", headset_data.l_pos),
                                  ("right", "right", headset_data.r_pos)):
            # exact zeros = the headset lost this controller (hands above the
            # head / out of volume) -- hold the arm, do not chase (0,0,0)
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

        ## DISCONTINUITY GATE.  Checked on the RAW controller pose -- before
        ## the head-relative composition and before any remap -- so what is
        ## judged is what the headset reported, not something this loop derived
        ## from it.  Every side is checked every tick, active or not: the gate
        ## needs an unbroken history to tell a jump from a gap.
        _now_s = now()
        _gate_ok = {}
        for _arm, _side, _pos, _quat in (
                ("left", "left", headset_data.l_pos, headset_data.l_quat),
                ("right", "right", headset_data.r_pos, headset_data.r_quat),
                ("head", "head", headset_data.h_pos, headset_data.h_quat)):
            _ok, _why = jump_gate.check(_side, _pos, _quat, _now_s)
            _gate_ok[_arm] = _ok
            if not _ok and _why != "settling" and _arm in arm_active and arm_active[_arm]:
                print(f"[tracking] {_side} pose jumped ({_why}) -- {_arm} arm "
                      f"HELD until it settles")
            if _arm in arm_active and not _ok:
                arm_active[_arm] = False
        if not _gate_ok.get("head", True):
            ## The head drives the camera arm.  A head-tracking glitch swings
            ## the camera exactly the way a controller glitch swings a gripper
            ## arm, and it is worse to debug because the operator is looking
            ## through it.
            if _gate_ok.get("head") is False and tick_counter % 50 == 0:
                print("[tracking] head pose jumped -- camera arm HELD")

        ## Middle is decided AFTER the tracking guard, from the arms that
        ## actually ended up driving -- not from the raw buttons.  A controller
        ## that falls asleep keeps reporting whatever it was holding, and
        ## reading the raw button would let a dead device that is no longer
        ## allowed to drive its own arm keep the camera following the head.
        arm_active["middle"] = _gate_ok.get("head", True) and (
            headset_data.l_button_two if MIDDLE_BUTTON == "y" else
            (arm_active["left"] or arm_active["right"]
             or headset_data.l_button_two))

        ## WHO THE OPERATOR IS CLAIMING, from the RAW buttons -- before the
        ## frozen / lost-tracking / jump overrides above.  Those overrides
        ## turn a claimed arm into a HELD arm; they must never turn it into a
        ## policy-driven one.  Holding A while standing still is a valid human
        ## action ("wait"), and the policy taking over at that moment is the
        ## ambiguity that made it impossible to tell who was driving.
        _human_claim = {
            "left": bool(headset_data.l_button_one),
            "right": bool(headset_data.r_button_one),
            "middle": bool(headset_data.l_button_two if MIDDLE_BUTTON == "y" else
                           (headset_data.l_button_one or headset_data.r_button_one
                            or headset_data.l_button_two)),
        }

        ## Last word on whether anything may move this tick: the latch a long
        ## uplink gap leaves behind.  Placed AFTER the gate and
        ## tracking checks so a veto here cannot be undone below.
        button_pressed, arm_active = link_guard.apply(
            now(), headset_data, button_pressed, arm_active)

        for arm in arm_names:
            if arm not in teleop_state.arms:
                teleop_state.arms[arm] = ArmTeleopState()
            was_active = teleop_state.arms[arm].active
            if arm_active[arm] and not was_active and teleop_state.active:
                # Arm activated MID-session: re-anchor its reference frames to
                # now, else it would jump by everything the controller/head
                # moved since the session started.
                anchor_arm_state(
                    teleop_state.arms[arm], controller_poses[arm], cmd_kin.T_cmd[arm],
                    head_pose=controller_poses.get("middle"),
                    base_remap=(cfg.R_cam_remap if arm == "middle" else cfg.R_arm_remap),
                )
            teleop_state.arms[arm].active = arm_active[arm]

        # teleop_state.active = any(
        #     arm_state.active
        #     for arm_state in teleop_state.arms.values()
        # )

        ## DEAD-MAN TIMER (GIAVA_MAX_ENGAGE_S).  The latch is what makes it a
        ## limit: without it the still-held button re-enables teleop on the
        ## very next tick and the timeout accomplishes nothing.  Cleared only
        ## by actually letting go.
        if _engage_latched and not button_pressed:
            _engage_latched = False
            print("  [engage] button released -- teleop can be re-enabled")
        if _engage_latched:
            button_pressed = False

        if button_pressed and not teleop_state.active:
            # Re-anchor UNCONDITIONALLY at every teleop enable, not only when
            # the stale heuristic fires.  Any motion outside the loop (named
            # poses, scripts, hand-moving a torqued-off arm) desynchronizes
            # three references at once -- our last_cmds, our T_cmd, and the
            # DRIVER's joint_commands (its velocity check compares against its
            # last accepted command, not the encoders).  One sync per enable
            # costs a joint-state read and one FK; missing one costs a stuck
            # arm rejecting every command.
            if command_state_is_stale(robots=robots, arm_data=arm_data, arm_names=arm_names, cmd_state=cmd_state, tolerance=0.08):
                print("Resynchronizing teleoperation state with measured joints.")
            sync_robot_state(
                robots=robots,
                robot=robot,
                arm_data=arm_data,
                arm_names=arm_names,
                cmd_kin=cmd_kin,
                cmd_state=cmd_state,
                to_urdf=coupled_ik.driver_to_urdf,
            )
            ## Same reason, same moment: the tube filter's virtual plant is
            ## a fourth reference that can be stale after off-loop motion.
            if tube_filters is not None:
                for _a in tube_filters:
                    tube_filters[_a].reset(np.asarray(
                        robots[_a].dxl.joint_states.position[:ARM_CONFIG[_a]["num_joints"]],
                        dtype=float))
                cmd_kin.q_ik = None      # IK re-seeds from the synced command

            ## The gate must not be armed already-violated -- it would hold
            ## every command and read as "teleop is broken".  Checked at the
            ## just-synced commanded pose; the error names the pair to move.
            if capsule_gate is not None:
                ## Report only. Starting a session with the grippers close
                ## is normal -- after a handover, for instance -- and the
                ## gate holds the arms until they separate. It must never
                ## take the session down with it.
                capsule_gate.validate(
                    coupled_ik.driver_to_urdf(cmd_kin.q_cmd),
                    where="teleop enable")
            if table_gate is not None:
                table_gate.validate(
                    coupled_ik.driver_to_urdf(cmd_kin.q_cmd),
                    where="teleop enable")

            start_teleop_session(
                teleop_state,
                mode,
                controller_poses,
                cmd_kin,
                cfg=cfg,
            )

            print("\nTeleop ENABLED")
            if collecting_episode and not episode_armed:
                episode_armed = True
                if episode_stats.frames_skipped_pre_teleop:
                    print(f"[record] episode starts here; skipped "
                          f"{episode_stats.frames_skipped_pre_teleop} parked "
                          f"frames since 'r' "
                          f"({episode_stats.frames_skipped_pre_teleop * cfg.control_dt:.1f} s "
                          f"at the reset pose). GIAVA_RECORD_GATE=all to keep "
                          f"them.")
            episode_stats.teleop_enable_count += 1
            _teleop_since = now()
        elif (not button_pressed) and teleop_state.active:
            stop_teleop_session(teleop_state)
            print("\nTeleop DISABLED")
            episode_stats.teleop_disable_count += 1

        ## Engaged too long: disengage and latch until the button is released.
        if (MAX_ENGAGE_S > 0 and teleop_state.active
                and (now() - _teleop_since) > MAX_ENGAGE_S):
            stop_teleop_session(teleop_state)
            episode_stats.teleop_disable_count += 1
            _engage_latched = True
            print(f"\nTeleop AUTO-DISABLED after {MAX_ENGAGE_S:g}s engaged "
                  f"(GIAVA_MAX_ENGAGE_S). Arms are holding. Release the "
                  f"button and grip again to carry on.")

        gripper_actions = {}
        joint_positions = {}
        joint_velocities = {}
        joint_efforts = {}
        gripper_states = {}
        gripper_velocities = {}
        gripper_efforts = {}

        for arm in arm_names:
            joint_state_msg = robots[arm].dxl.joint_states

            # joint_positions[arm] = np.asarray(joint_state_msg.position[:6], dtype=np.float32)
            n = ARM_CONFIG[arm]["num_joints"]

            joint_positions[arm] = np.asarray(
                joint_state_msg.position[:n],
                dtype=np.float32,
            )
            ## Present_Velocity off the register (0.024 rad/s resolution) --
            ## a finite difference of the 4096-count encoder is a 0.077 rad/s
            ## staircase and useless to the tube filter's measured mode.
            joint_velocities[arm] = (
                np.asarray(joint_state_msg.velocity[:n], dtype=np.float32)
                if len(joint_state_msg.velocity) >= n else np.zeros(n, dtype=np.float32))
            ## Present_Current, in the same message and previously read only
            ## by the servo watchdog and then dropped.  Recorded because it is
            ## the contact sense this rig already has: gripper current rising
            ## as the fingers close IS the grasp event, and a joint current
            ## spike IS a collision -- both otherwise need object pose
            ## estimation to infer from vision.
            joint_efforts[arm] = (
                np.asarray(joint_state_msg.effort[:n], dtype=np.float32)
                if len(joint_state_msg.effort) >= n else np.zeros(n, dtype=np.float32))

            if ARM_CONFIG[arm]["has_gripper"]:
                gripper_states[arm] = np.asarray([joint_state_msg.position[6]], dtype=np.float32)
                gripper_velocities[arm] = (
                    float(joint_state_msg.velocity[6])
                    if len(joint_state_msg.velocity) > 6 else 0.0)
                gripper_efforts[arm] = (
                    float(joint_state_msg.effort[6])
                    if len(joint_state_msg.effort) > 6 else 0.0)

        if _touch is not None:
            _tb = {"a": headset_data.r_button_one, "b": headset_data.r_button_two,
                   "x": headset_data.l_button_one, "y": headset_data.l_button_two,
                   "lstick": headset_data.l_button_thumbstick,
                   "rstick": headset_data.r_button_thumbstick}.get(_touch["button"], False)
            if _tb and not _touch["prev"] and (now() - _touch["last"]) > 1.0:
                _q_full = cmd_kin.q_cmd.copy()
                for _a in arm_names:
                    if _a in joint_positions:
                        _q_full[arm_data[_a]["joint_indices"]] = joint_positions[_a]
                _fk_t, _ee_t = compute_fk_and_ee(robot, coupled_ik.driver_to_urdf(_q_full), arm_data)
                _entry = {"point": f"t{len(_touch['touches'])}", "arms_touching": "both",
                          "time": time.time(), "q_driver": {},
                          "T_world_flange": {}, "gripper_closed": {}}
                for _a in ("left", "right"):
                    if _a not in joint_positions:
                        continue
                    _e = np.asarray(_ee_t[_a], dtype=float)
                    _T = np.eye(4)
                    ## pyroki FK returns wxyz; quat2mat (transform_utils) takes
                    ## xyzw.  Feeding wxyz straight in gave every recorded
                    ## touch a wrong orientation on 2026-09-03 -- reorder.
                    _T[:3, :3] = quat2mat(_e[[1, 2, 3, 0]])
                    _T[:3, 3] = _e[4:]
                    _entry["q_driver"][_a] = joint_positions[_a].astype(float).tolist()
                    _entry["T_world_flange"][_a] = _T.tolist()
                    _g = float(gripper_states[_a][0]) if _a in gripper_states else None
                    ## gripper motor angle: 0.0 open .. -1.5 closed (gripper.py)
                    _entry["gripper_closed"][_a] = bool(_g is not None and _g < -1.05)
                _touch["touches"].append(_entry)
                _touch["last"] = now()
                try:
                    with open(_touch["path"], "w") as _fh:
                        json.dump({"metadata": {"method": "touch_calibrate_record",
                                                "source": "data_collection.py button hook",
                                                "urdf": URDF_PATH, "units": "metres, radians"},
                                   "convention": "World frame is giava.urdf's root link 'base'. "
                                                 "T_world_flange are 4x4 poses of *_gripper_base from "
                                                 "FK of MEASURED joints through the URDF at record time.",
                                   "touches": _touch["touches"]}, _fh, indent=1)
                except Exception as _exc:
                    print(f"[touch] could not write {_touch['path']}: {_exc}")
                if "left" in _entry["T_world_flange"] and "right" in _entry["T_world_flange"]:
                    _tcp = np.array([0.0, 0.00006, 0.0722, 1.0])
                    _gap = (np.asarray(_entry["T_world_flange"]["left"]) @ _tcp
                            - np.asarray(_entry["T_world_flange"]["right"]) @ _tcp)[:3] * 1e3
                    _warn = "" if all(_entry["gripper_closed"].values()) else \
                        "   !! a gripper is NOT closed -- was this a real tip-to-tip touch?"
                    print(f"\a[touch] #{len(_touch['touches'])} recorded; FK says the tips are "
                          f"{np.linalg.norm(_gap):.1f} mm apart (dx {_gap[0]:+.1f}, dy {_gap[1]:+.1f}, "
                          f"dz {_gap[2]:+.1f}){_warn}")
                else:
                    print(f"\a[touch] #{len(_touch['touches'])} recorded (one arm only -- "
                          f"needs --mode bimanual or all to be useful)")
            _touch["prev"] = bool(_tb)

        ## The raw analog trigger goes in; GIAVA_GRIPPER_MODE decides whether
        ## it is thresholded (binary, the historical behaviour) or mapped to a
        ## continuous aperture (analog).  Either way the return value is the
        ## COMMANDED position, and that is what the dataset records as the
        ## gripper action -- the measured aperture is recorded separately in
        ## observation.state, so a grasp stores both "close hard" (intent) and
        ## the object-width the fingers actually stalled at (effect).
        ## DAgger: an arm the policy drives this tick takes its gripper from
        ## the action vector (commanded in the block below), not from the
        ## trigger.  Without this the released trigger published OPEN every
        ## tick and silently overrode -- and misrecorded -- the policy's
        ## gripper.  A claimed arm keeps the trigger, exactly as in teleop.
        _dagger_arm_live = {
            a: (policy_driver is not None and collecting_episode and dagger_play
                and not _human_claim.get(a))
            for a in arm_names}

        if "left" in arm_names and not _dagger_arm_live["left"]:
            gripper_actions["left"] = update_gripper_from_trigger(
                robots["left"], headset_data.l_index_trigger)

        if "right" in arm_names and not _dagger_arm_live["right"]:
            gripper_actions["right"] = update_gripper_from_trigger(
                robots["right"], headset_data.r_index_trigger)

        gripper_actions["middle"] = None

        targets = {}

        for arm in arm_names:

            arm_state = teleop_state.arms[arm]

            if not arm_state.active:
                continue

            controller_pose = controller_poses[arm]

            if arm == "middle":
                # print("target_pos", target_pos)
                # print("target_wxyz", target_wxyz)
                # print("middle active:", teleop_state.arms["middle"].active)
                # print("camera target")
                # print(target_pos)
                target_pos, target_wxyz = compute_camera_arm_target(
                    cfg,
                    arm_state,
                    controller_pose,
                    cmd_kin.T_cmd[arm],
                    cfg.R_cam_remap,
                )
                # GIAVA_DEBUG_HEAD=1: print the raw headset delta and the
                # remapped robot delta (~2 Hz) so the axis mapping can be
                # verified empirically: nod / shake / lean in one direction at
                # a time and read off which robot axis moves.
                if DEBUG_HEAD and (tick_counter % 25 == 0):
                    _dh = controller_pose[:3, 3] - arm_state.start_controller_pos
                    _rm = (arm_state.session_remap
                           if arm_state.session_remap is not None else cfg.R_cam_remap)
                    _dr = _rm @ _dh
                    _cmd = np.asarray(target_pos) - arm_state.start_robot_pos
                    _ach = np.asarray(cmd_kin.T_cmd[arm][4:]) - arm_state.start_robot_pos
                    print(f"[head] d_raw={np.round(_dh, 3)} d_robot={np.round(_dr, 3)} "
                          f"cmd={np.round(_cmd, 3)} achieved={np.round(_ach, 3)} "
                          f"(x=op-left, y=op-back, z=up; cmd<<d_robot = starved)")
            else:
                target_pos, target_wxyz = compute_gripper_arm_target(
                    cfg,
                    arm_state,
                    controller_pose,
                    cmd_kin.T_cmd[arm],
                    cfg.R_arm_remap,
                )

            targets[arm] = (target_pos, target_wxyz)

        _dagger_live = (policy_driver is not None and collecting_episode
                        and dagger_play)
        if teleop_state.active or _dagger_live:
            # start_teleop_session(teleop_state, mode, controller_poses, cmd_kin)
            t_solve_start = now()

            # Coupled three-arm IK (ik_study winner): ONE solve for all due
            # arms per tick, replacing the previous per-arm solve loop.  Arms
            # without a fresh target hold their commanded pose inside the
            # same problem, which is what makes the inter-arm collision term
            # meaningful.
            due_arms = [
                arm for arm in arm_names
                if teleop_state.arms[arm].active
                and not _faulted[arm]
                and (t_solve_start - cmd_state.last_arm_cmd_time[arm]) >= cfg.arm_cmd_dt
                and arm in targets
            ]

            ## DAgger: arms the operator is NOT driving are driven by the
            ## policy, but ONLY while an episode is recording -- outside one
            ## the arms stay still, exactly as they do without --policy.
            policy_arms, policy_action = [], None
            _t_policy = now()
            if _dagger_live:
                ## HANDOFFS, both directions, said out loud.  Back to the
                ## policy also drops the queued chunk: it was planned before
                ## the operator intervened, from a scene that no longer
                ## exists, and resuming it would drive the arm from a stale
                ## observation for up to n_action_steps ticks.
                _human_now = frozenset(a for a in arm_names if _human_claim.get(a))
                if _human_now != _dagger_prev_human[0]:
                    _gained = sorted(_human_now - _dagger_prev_human[0])
                    _lost = sorted(_dagger_prev_human[0] - _human_now)
                    if _gained:
                        print(f"[dagger] >>> YOU have {_gained}")
                    if _lost:
                        ## The chunk goes stale on a handoff; the
                        ## extractor's hold-last does NOT -- it is the
                        ## last place the object was actually seen, and
                        ## that is still true after the operator lets go.
                        policy_driver.reset()
                        print(f"[dagger] POLICY resumes {_lost} (re-planning)")
                    _dagger_prev_human[0] = _human_now
                _pstate = np.concatenate([
                    np.concatenate([joint_positions[a].astype(np.float32),
                                    [np.float32(gripper_states[a][0]
                                if a in gripper_states else 0.0)]])
                    if ARM_CONFIG[a]["has_gripper"]
                    else joint_positions[a].astype(np.float32)
                    for a in arm_names])
                with frame_lock:
                    _pframes = {c: (latest_frames.get(c).copy()
                                    if latest_frames.get(c) is not None else None)
                                for c in policy_driver.cameras}
                _penv, _pewhy = _pd_env_vec(_pframes)
                if _pewhy is not None:
                    policy_action, _pwhy = None, _pewhy
                else:
                    policy_action, _pwhy = policy_driver.act(
                        _pstate, _pframes, env_vec=_penv)
                if policy_action is None:
                    ## Never command on a partial observation -- hold.
                    if (tick_counter - _dagger_last_print[0]) >= 250:
                        _dagger_last_print[0] = tick_counter
                        print(f"[dagger] holding: {_pwhy}")
                else:
                    ## Not claimed by the operator, not faulted, not already
                    ## being teleoperated.  A claimed-but-frozen arm is in
                    ## none of these lists and simply holds.
                    policy_arms = [a for a in arm_names
                                   if a not in due_arms and not _faulted[a]
                                   and not _human_claim.get(a)]
                ## Gripper for policy-driven arms.  The action vector carries
                ## the COMMANDED aperture the demos recorded (0 open .. -1.5
                ## closed), so it is clamped to that range and published the
                ## same way the trigger path publishes.  gripper_actions is
                ## what the frame records, so the dataset sees the policy's
                ## intent, not the idle trigger.  With no action this tick
                ## (missing frame) the last command is held and recorded.
                for _a in arm_names:
                    if not _dagger_arm_live.get(_a) or not ARM_CONFIG[_a]["has_gripper"]:
                        continue
                    _, _gi = _dagger_slices[_a]
                    if policy_action is not None and _gi is not None:
                        _g = float(np.clip(policy_action[_gi],
                                           min(GRIPPER_CLOSED, GRIPPER_OPEN),
                                           max(GRIPPER_CLOSED, GRIPPER_OPEN)))
                        _dagger_last_grip[_a] = _g
                    _g = _dagger_last_grip.get(_a, GRIPPER_OPEN)
                    command_gripper(robots[_a], _g)
                    gripper_actions[_a] = _g
                episode_stats.policy_ms.append((now() - _t_policy) * 1e3)

            if due_arms or policy_arms:
                for arm in due_arms:
                    episode_stats.arms[arm].ik_attempts += 1

                _t_ik = now()
                ## IK warm start.  Without the tube filter, q_cmd IS the
                ## solver's previous answer (the clamps rarely touch it), so
                ## the solver re-converges to the same branch every tick.
                ## With the filter, q_cmd is the FILTERED command, which lags
                ## the solver's answer whenever the arm is moving -- seeding
                ## from it restarts the solve from a different point every
                ## tick, the smoothing cost drags the solution toward that
                ## lagging point, and with ori_weight << pos_weight the wrist
                ## and elbow wander while position converges: a slow arc
                ## through every joint that ends at the exact pose (seen
                ## 2026-09-03).  So seed the solver from its OWN last
                ## solution and let the filter be a pure post-filter -- unless
                ## that solution has run more than 0.5 rad from what was
                ## actually sent (filter capped, stall, re-anchor), in which
                ## case start again from the command.
                _seed = cmd_kin.q_cmd
                if not due_arms:
                    ## Policy-only tick: nothing to solve, and q_new is only
                    ## read for teleop arms below.
                    q_new = cmd_kin.q_cmd
                    ee_sol = {}
                if due_arms and tube_filters is not None:
                    _q_ik = getattr(cmd_kin, "q_ik", None)
                    if _q_ik is not None and np.max(np.abs(_q_ik - cmd_kin.q_cmd)) <= 0.5:
                        _seed = _q_ik
                if due_arms:
                    q_new = coupled_ik.solve(
                        _seed,
                        {arm: targets[arm] for arm in due_arms},
                    )
                if due_arms and tube_filters is not None:
                    cmd_kin.q_ik = np.asarray(q_new, dtype=float).copy()

                # Solve-time accounting: with sphere collision active the
                # study measured 5.2 ms median but 64 ms worst-trajectory p95,
                # against a 20 ms budget at 50 Hz.  Overruns are the first thing
                # to check when the motion feels bad.
                if due_arms:
                    _ik_ms = (now() - _t_ik) * 1e3
                    episode_stats.ik_solve_ms.append(_ik_ms)
                    if _ik_ms > cfg.control_dt * 1e3:
                        episode_stats.ik_overrun_ticks += 1

                # Optional: prove the collision terms see what the operator
                # sees.  GIAVA_LOG_CLEARANCE=1 prints min sphere clearance
                # whenever it drops near the margin (throttled to ~2 Hz).
                if LOG_CLEARANCE and (tick_counter % 25 == 0):
                    _clr = coupled_ik.min_clearance(q_new)
                    if _clr < COLLISION_MARGIN + 0.02:
                        state = "INSIDE MARGIN" if _clr < COLLISION_MARGIN else "near"
                        print(f"[collision] min sphere clearance "
                              f"{_clr * 1e3:+.1f} mm ({state}, margin "
                              f"{COLLISION_MARGIN * 1e3:.0f} mm)")

                if due_arms:
                    fk_sol, ee_sol = compute_fk_and_ee(
                        robot, coupled_ik.driver_to_urdf(q_new), arm_data)

                pending_cmds = []  # (arm, joint_idx, q_arm_cmd, prev_cmd)
                for arm in due_arms + policy_arms:
                    joint_idx = arm_data[arm]["joint_indices"]
                    if arm in policy_arms:
                        ## The policy's own joint targets, in driver
                        ## coordinates.  They go through EVERY clamp and both
                        ## gates below, exactly as a teleoperated command
                        ## does -- a policy command is not trusted more than
                        ## a human one.
                        _sl, _ = _dagger_slices[arm]
                        target_q = np.asarray(policy_action[_sl], dtype=float)
                        prev_cmd = (cmd_state.last_cmds[arm]
                                    if arm in cmd_state.last_cmds
                                    else cmd_kin.q_cmd[joint_idx])
                    else:
                        target_pos, target_wxyz = targets[arm]

                        episode_stats.arms[arm].ik_successes += 1

                        pos_err = np.linalg.norm(ee_sol[arm][4:] - target_pos)
                        episode_stats.arms[arm].ik_position_errors.append(pos_err)

                        quat = np.asarray(ee_sol[arm][:4], dtype=np.float64).copy()
                        ## quat2mat (transform_utils, robosuite) takes xyzw;
                        ## both of these are wxyz (pyroki FK / the IK target).
                        ## Fed straight in, the logged orientation error was
                        ## garbage (found 2026-09-03 via the touch hook).
                        R_sol = quat2mat(np.asarray(quat, dtype=float)[[1, 2, 3, 0]])
                        R_target = quat2mat(np.asarray(target_wxyz, dtype=float)[[1, 2, 3, 0]])
                        R_err = R_sol.T @ R_target
                        trace = np.clip(np.trace(R_err), -1.0, 3.0)
                        angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
                        episode_stats.arms[arm].ik_orientation_errors.append(angle_rad)

                        target_q = np.asarray(q_new[joint_idx], dtype=float)

                        prev_cmd = (cmd_state.last_cmds[arm]
                                    if arm in cmd_state.last_cmds
                                    else cmd_kin.q_cmd[joint_idx])

                    ## TUBE MPC (GIAVA_TUBE_MPC=1): the reference filter sits
                    ## HERE, on the raw IK target, before any clamp -- it
                    ## replaces the step clamps' job (the driver clamp below
                    ## stays as a proof: in tube mode it must never fire).
                    ## The gates further down are collision gates the filter
                    ## does not model, so they still run on its output.
                    if tube_filters is not None and arm in tube_filters:
                        target_q, _tinfo = tube_filters[arm].step(
                            joint_positions[arm].astype(float),
                            joint_velocities[arm].astype(float),
                            target_q)
                        if _tinfo["fallback"]:
                            episode_stats.tube_fallback_ticks += 1

                    # Post-solve step clamp -- off by default to match the
                    # ik_study conditions (see TeleopConfig.enable_joint_clamp).
                    if cfg.enable_joint_clamp:
                        max_step = (cfg.max_joint_step_middle if arm == "middle"
                                    else cfg.max_joint_step)
                        q_arm_cmd = clamp_joint_step(prev_cmd, target_q, max_step)
                        if np.any(
                            np.abs(target_q - prev_cmd) > np.asarray(max_step) * 0.99
                        ):
                            episode_stats.clamp_saturated_ticks += 1
                    else:
                        q_arm_cmd = target_q

                    # Driver-feasibility clamp: interbotix rejects the ENTIRE
                    # group command if any joint would exceed its position or
                    # velocity limit, so one infeasible joint freezes the whole
                    # arm.  Clamp into the feasible set so every command is
                    # accepted; the operator feels a slower joint rather than a
                    # dead arm.
                    if cfg.enable_driver_clamp:
                        # The URDF and the driver disagree slightly (e.g. right
                        # wrist_angle: URDF +2.234, driver rejected +2.243), and
                        # it is the DRIVER that refuses commands -- so clamp
                        # against ITS limit arrays, pulled in by the cushion.
                        lo = np.asarray(
                            robots[arm].arm.group_info.joint_lower_limits, dtype=float
                        )[: len(prev_cmd)]
                        hi = np.asarray(
                            robots[arm].arm.group_info.joint_upper_limits, dtype=float
                        )[: len(prev_cmd)]
                        ## Scaled by the overload model: a joint climbing
                        ## toward the servo's own latch gets a smaller step,
                        ## which bleeds energy out of exactly the joint
                        ## accumulating it while the operator keeps control.
                        step = float(cfg.driver_max_step) * overload.derate(arm)
                        # Clamp against the DRIVER's reference, not ours.  The
                        # driver validates against its last *accepted* command;
                        # if one of ours was rejected the two references have
                        # already diverged, and clamping against our own would
                        # keep proposing steps it will keep refusing.
                        ref = prev_cmd
                        getter = getattr(robots[arm].arm, "get_joint_commands", None)
                        if getter is not None:
                            try:
                                ref = np.asarray(getter(), dtype=float)[:len(prev_cmd)]
                            except Exception:
                                ref = prev_cmd
                        clamped = np.clip(q_arm_cmd, ref - step, ref + step)
                        clamped = np.clip(
                            clamped,
                            lo + cfg.driver_limit_margin,
                            hi - cfg.driver_limit_margin,
                        )
                        if np.any(np.abs(clamped - q_arm_cmd) > 1e-6):
                            episode_stats.driver_clamp_ticks += 1
                            hit = int(np.argmax(np.abs(clamped - q_arm_cmd)))
                            episode_stats.driver_clamp_joints[f"{arm}{hit}"] = (
                                episode_stats.driver_clamp_joints.get(f"{arm}{hit}", 0) + 1
                            )
                        q_arm_cmd = clamped

                    ## STALL HOLD, after every clamp: a jammed joint's goal
                    ## goes back to where the joint actually is, so the motor
                    ## stops pushing into whatever is blocking it.  Placed
                    ## here rather than in the solver because it must survive
                    ## the clamps -- the same reason the capsule gate runs on
                    ## the final assembled command.
                    if stallgate.held.get(arm):
                        q_arm_cmd = stallgate.apply(
                            arm, q_arm_cmd,
                            np.asarray(robots[arm].dxl.joint_states.position[
                                :len(q_arm_cmd)], dtype=float))
                        ## The filter's virtual state must not march on
                        ## while the arm is held: re-anchor it on the
                        ## encoders so release starts from where the arm is.
                        if tube_filters is not None and arm in tube_filters:
                            tube_filters[arm].reset(joint_positions[arm].astype(float))
                            cmd_kin.q_ik = None

                    ## WAIST TRAVEL, last of all: the mechanical stop is not a
                    ## preference the solver can trade against pose error, and
                    ## every clamp above works in a frame that cannot see it
                    ## (waist_travel.py).  Refusing the last few degrees costs
                    ## camera yaw the arm could not have delivered anyway; not
                    ## refusing them costs a motor pushing into a stop.
                    if arm == "middle":
                        q_arm_cmd, _wclamped = waist_travel.clamp(q_arm_cmd)
                        if _wclamped and (tick_counter
                                          - _waist_last_print[0]) >= 250:
                            _waist_last_print[0] = tick_counter
                            print(f"[waist] holding at the travel limit "
                                  f"({waist_travel.clamps} clamped ticks)")

                    ## COMPUTE phase ends here -- nothing has been sent.
                    ## Sends happen below, after the capsule gate has seen
                    ## the assembled command of ALL due arms: the clamps
                    ## just modified what the solver certified, and two
                    ## arms each individually fine can still be about to
                    ## meet each other.
                    pending_cmds.append((arm, joint_idx, q_arm_cmd, prev_cmd))

                ## HARD INTER-ARM GATE on the final command.  Circumscribed
                ## capsules: distance > margin PROVES mesh separation.  On a
                ## violation the whole tick is refused -- every arm holds its
                ## previous command (a partial send could still create the
                ## very pair the gate saw).  Nothing is modified: a gate that
                ## edits commands is a second, unstudied IK solver.
                if capsule_gate is not None and pending_cmds:
                    ## The step is SCALED, not refused.  Find the largest
                    ## fraction of this tick's motion whose whole swept
                    ## segment clears the margin, and send that -- so the
                    ## arms slide up to the boundary and stop there however
                    ## hard the operator pushes, instead of freezing at
                    ## whatever the last accepted tick happened to reach.
                    ##
                    ## Scaling is UNIFORM across arms on purpose.  Scaling
                    ## only the arms in the offending pair would change the
                    ## relative geometry mid-step -- i.e. redirect the
                    ## motion into a path nobody commanded.  A uniform
                    ## factor preserves the shape of the commanded motion
                    ## and only slows it down.
                    q_prev_full = cmd_kin.q_cmd.copy()
                    q_target_full = q_prev_full.copy()
                    for _arm, _idx, _q, _ in pending_cmds:
                        q_target_full[_idx] = _q
                    _alpha, _dist, _pair = capsule_gate.largest_safe_fraction(
                        coupled_ik.driver_to_urdf(q_prev_full),
                        coupled_ik.driver_to_urdf(q_target_full))

                    if _alpha >= 1.0:
                        pass                      # full step is clear
                    elif _alpha <= 0.0:
                        episode_stats.capsule_gate_blocks += 1
                        if (tick_counter - _gate_last_print[0]) >= 25:  # ~2 Hz
                            _gate_last_print[0] = tick_counter
                            ## margin_for_pair, NOT capsule_gate.margin:
                            ## `margin` is only the STRUCTURAL default, and
                            ## fingertip / gripper / camera pairs each carry
                            ## their own.  Printing the default next to a
                            ## fingertip pair's distance reported 25 mm for a
                            ## pair actually held at 8 mm -- which reads as
                            ## "the gate is far stricter than it is" and sends
                            ## you tuning the wrong number.
                            _pm = capsule_gate.margin_for_pair(_pair)
                            print(f"[capsule gate] HOLDING: {_pair[0]} <-> "
                                  f"{_pair[1]} at {_dist * 1e3:+.1f} mm "
                                  f"(margin {_pm * 1e3:.0f} mm). "
                                  f"Move the controllers apart.")
                        pending_cmds = []
                    else:
                        ## Partial step: interpolate in DRIVER space by the
                        ## same alpha.  driver->urdf is affine per joint, so
                        ## interpolating either side gives the identical
                        ## configuration -- the fraction checked is exactly
                        ## the fraction sent.
                        episode_stats.capsule_gate_scaled += 1
                        episode_stats.capsule_gate_min_alpha = min(
                            episode_stats.capsule_gate_min_alpha, float(_alpha))
                        scaled = []
                        for _arm, _idx, _q, _prev in pending_cmds:
                            _qs = (q_prev_full[_idx]
                                   + _alpha * (_q - q_prev_full[_idx]))
                            scaled.append((_arm, _idx, _qs, _prev))
                        pending_cmds = scaled
                        if (tick_counter - _gate_last_print[0]) >= 25:
                            _gate_last_print[0] = tick_counter
                            print(f"[capsule gate] limiting step to "
                                  f"{_alpha * 100:.0f}%: {_pair[0]} <-> "
                                  f"{_pair[1]} stopping at "
                                  f"{_dist * 1e3:+.1f} mm")

                ## HARD TABLE-FLOOR GATE, same call site and shrink-only
                ## logic as the inter-arm gate above, run on whatever the
                ## inter-arm gate left standing -- so the two compose
                ## (chaining two shrink-only scalers can only ever leave the
                ## command more conservative than either alone). See
                ## table_gate.py; hardware-validated, disable with
                ## --table off.
                if table_gate is not None and pending_cmds:
                    q_prev_full = cmd_kin.q_cmd.copy()
                    q_target_full = q_prev_full.copy()
                    for _arm, _idx, _q, _ in pending_cmds:
                        q_target_full[_idx] = _q
                    _alpha, _dist, _link = table_gate.largest_safe_fraction(
                        coupled_ik.driver_to_urdf(q_prev_full),
                        coupled_ik.driver_to_urdf(q_target_full))

                    if _alpha >= 1.0:
                        pass                      # full step is clear
                    elif _alpha <= 0.0:
                        episode_stats.table_gate_blocks += 1
                        if (tick_counter - _table_gate_last_print[0]) >= 25:
                            _table_gate_last_print[0] = tick_counter
                            print(f"[table gate] HOLDING: {_link} at "
                                  f"{_dist * 1e3:+.1f} mm above the table "
                                  f"(margin {table_gate.margin * 1e3:.0f} "
                                  f"mm). Move away from the table.")
                        pending_cmds = []
                    else:
                        episode_stats.table_gate_scaled += 1
                        episode_stats.table_gate_min_alpha = min(
                            episode_stats.table_gate_min_alpha, float(_alpha))
                        scaled = []
                        for _arm, _idx, _q, _prev in pending_cmds:
                            _qs = (q_prev_full[_idx]
                                   + _alpha * (_q - q_prev_full[_idx]))
                            scaled.append((_arm, _idx, _qs, _prev))
                        pending_cmds = scaled
                        if (tick_counter - _table_gate_last_print[0]) >= 25:
                            _table_gate_last_print[0] = tick_counter
                            print(f"[table gate] limiting step to "
                                  f"{_alpha * 100:.0f}%: {_link} stopping "
                                  f"at {_dist * 1e3:+.1f} mm above the table")

                for arm, joint_idx, q_arm_cmd, prev_cmd in pending_cmds:
                    cmd_step = np.linalg.norm(q_arm_cmd - prev_cmd)
                    episode_stats.arms[arm].joint_step_norms.append(cmd_step)

                    t_cmd_start = now()

                    robots[arm].arm.set_joint_positions(
                        q_arm_cmd.tolist(),
                        moving_time=cfg.moving_time,
                        accel_time=cfg.accel_time,
                        blocking=False,
                    )

                    episode_stats.cmd.append(now() - t_cmd_start)

                    episode_stats.arms[arm].waist_cmds.append(float(q_arm_cmd[0]))
                    if arm in targets:
                        episode_stats.arms[arm].target_positions.append(
                            np.asarray(targets[arm][0], dtype=float).copy()
                        )

                    cmd_kin.q_cmd[joint_idx] = q_arm_cmd
                    cmd_state.last_cmds[arm] = q_arm_cmd.copy()
                    cmd_state.last_arm_cmd_time[arm] = now()

            # GIAVA_TRACK_LOG=1: per-tick expected-vs-measured EE, per arm, in
            # the same row format the debug tool logs -- so plot_tracking.py
            # renders data-collection episodes identically (per-axis expected
            # vs achieved, % diff, per-rate overlays).
            if TRACK_LOG and due_arms:
                q_meas_full = cmd_kin.q_cmd.copy()
                for _a in arm_names:
                    if _a in joint_positions:
                        _n = len(arm_data[_a]["joint_indices"])
                        q_meas_full[arm_data[_a]["joint_indices"]] =                             np.asarray(joint_positions[_a], dtype=float)[:_n]
                _, ee_meas = compute_fk_and_ee(
                    robot, coupled_ik.driver_to_urdf(q_meas_full), arm_data)
                for _a in due_arms:
                    _st = teleop_state.arms.get(_a)
                    if _st is None or _st.start_robot_pos is None:
                        continue
                    _tp, _tw = targets[_a]
                    _mp = np.asarray(ee_meas[_a][4:], dtype=float)
                    _mq = np.asarray(ee_meas[_a][:4], dtype=float)  # wxyz
                    _R0 = _st.start_robot_rot
                    _Rt = R.from_quat(np.roll(np.asarray(_tw), -1)).as_matrix()
                    _Rm = R.from_quat(np.roll(_mq, -1)).as_matrix()
                    _rv = lambda Ra: np.degrees(
                        R.from_matrix(Ra @ _R0.T).as_rotvec())
                    if not hasattr(episode_stats.arms[_a], "track_rows"):
                        episode_stats.arms[_a].track_rows = []
                    episode_stats.arms[_a].track_rows.append(np.concatenate([
                        [now(), tick_counter],
                        np.asarray(_tp) - _st.start_robot_pos,
                        _mp - _st.start_robot_pos,
                        _rv(_Rt), _rv(_Rm)]))

            fk_cmd, ee_cmd = compute_fk_and_ee(robot, coupled_ik.driver_to_urdf(cmd_kin.q_cmd), arm_data)

            for arm in arm_names:
                cmd_kin.T_cmd[arm] = ee_cmd[arm]

            episode_stats.ik_solve.append(now() - t_solve_start)

        ## See RECORD_GATE: frames before the first teleop enable are the arm
        ## sitting at the reset pose, and writing them is what put thousands of
        ## identical reset-pose actions into every recorded dataset.
        if collecting_episode and RECORD_GATE == "teleop" and not episode_armed:
            episode_stats.frames_skipped_pre_teleop += 1
            collecting_episode_now = False
        else:
            collecting_episode_now = collecting_episode

        if collecting_episode_now:
            ## WALL clock, not monotonic: the camera timestamps stored beside
            ## this one are epoch-domain (librealsense global_time and the OAK
            ## worker's time.time()), and now() is time.monotonic() -- a boot-
            ## relative clock.  Recording the arm/obs timestamp from now() put
            ## two incomparable clocks in one dataset, so "how old was this
            ## image when the observation was assembled" (arm_ts - cam_ts, the
            ## number inference latency-matching needs) was uncomputable.  One
            ## epoch domain for every recorded timestamp fixes that; the loop
            ## keeps scheduling itself on the monotonic clock as before.
            obs_ts = time.time()

            images = {}
            camera_timestamps = {}

            

            # Camera frames.
            #
            # SYNC_FRAMES picks, per camera, the frame nearest a shared
            # reference instant instead of whatever is newest.  The cameras
            # free-run (the D405 has no inter_cam_sync_mode), so without this
            # the frames in one timestep can be a full frame period apart --
            # measured at 5.2 ms vs 12.6 ms old on this rig.  sync_info
            # carries the residual spread so alignment quality is recorded
            # rather than assumed.  GIAVA_SYNC_FRAMES=0 restores
            # latest-frame-wins.
            sync_info = None
            with frame_lock:
                if SYNC_FRAMES:
                    sel_frames, sel_ts, sync_info = select_synchronized_frames(
                        active_cameras)
                    for camera in active_cameras:
                        frame = sel_frames.get(camera)
                        ts = sel_ts.get(camera)
                        if frame is None or ts is None:
                            episode_stats.cameras[camera].frames_missing += 1
                            continue
                        images[camera] = frame.copy()
                        camera_timestamps[camera] = ts
                else:
                    for camera in active_cameras:
                        frame = latest_frames.get(camera)
                        ts = latest_timestamps.get(camera)

                        if frame is None or ts is None:
                            episode_stats.cameras[camera].frames_missing += 1
                            continue

                        images[camera] = frame.copy()
                        camera_timestamps[camera] = ts

            if sync_info is not None and sync_info["spread_s"] is not None:
                episode_stats.sync_spreads.append(sync_info["spread_s"])


            # Robot states
            robot_states = {}

            for arm in arm_names:

                joints = joint_positions[arm].astype(np.float32)
                cmd_track_err = np.linalg.norm(joint_positions[arm] - cmd_state.last_cmds[arm])
                episode_stats.arms[arm].cmd_track_err.append(cmd_track_err)

                robot_states[arm] = {
                    "joints": joints,
                    "gripper": float(gripper_states[arm][0])
                    if arm in gripper_states
                    else 0.0,
                    ## Same tick, same message as `joints` above -- so the
                    ## position, velocity and effort rows of a frame describe
                    ## one instant rather than three.
                    "velocity": joint_velocities.get(arm),
                    "gripper_velocity": gripper_velocities.get(arm, 0.0),
                    "effort": joint_efforts.get(arm),
                    "gripper_effort": gripper_efforts.get(arm, 0.0),
                }

            robot_actions = {}

            for arm in arm_names:

                joint_idx = arm_data[arm]["joint_indices"]

                q_cmd = (
                    cmd_state.last_cmds[arm]
                    if arm in cmd_state.last_cmds
                    else cmd_kin.q_cmd[joint_idx]
                )

                robot_actions[arm] = {
                    "joints": np.asarray(q_cmd, dtype=np.float32),
                    "gripper": (gripper_actions[arm] if arm in gripper_actions else 0.0),
                }

            ee_poses = {}

            fk, ee = compute_fk_and_ee(
                robot,
                coupled_ik.driver_to_urdf(cmd_kin.q_cmd),
                arm_data,
            )

            for arm in arm_names:
                ee_poses[arm] = np.asarray(ee[arm], dtype=np.float32)

            timestamps = {
                **camera_timestamps,
            }

            for arm in arm_names:
                timestamps[arm] = obs_ts

            frame = build_frame(
                mode=mode,
                active_cameras=active_cameras,
                robot_states=robot_states,
                robot_actions=robot_actions,
                ee_poses=ee_poses,
                timestamps=timestamps,
                images=images,
            )

            ## Who was driving this frame.  Counted rather than added to the
            ## dataset schema on purpose: a new feature would make these
            ## episodes unpoolable with the 138 already recorded.
            if policy_driver is not None:
                _human = [a for a in arm_names if _human_claim.get(a)]
                _dagger_frames["human" if _human else "policy"] += 1
                if _human:
                    episode_stats.dagger_human_frames += 1
                else:
                    episode_stats.dagger_policy_frames += 1

            dataset.add_frame(frame, task_ctx["task"])

            episode_stats.frames_added += 1

        ## AFTER the recording block and OUTSIDE it: the preview streams on
        ## every tick, recording or not.  `collecting_episode_now` rather
        ## than `collecting_episode` is deliberate -- it is the flag the
        ## frame writer above actually keys off, so the window's REC badge
        ## reports what is being written rather than what was pressed (under
        ## GIAVA_RECORD_GATE=teleop those differ for the seconds between 'r'
        ## and the first teleop enable, which is exactly when an operator
        ## most wants to know).  A no-op when the preview is off, and it
        ## never raises.
        preview.publish(frame_lock, latest_frames,
                        recording=collecting_episode_now,
                        episode=episode_idx if collecting_episode_now else None,
                        label=task_ctx["task"] if collecting_episode_now
                        else None)

        # sleep to reduce drift
        episode_stats.loop.append(now() - loop_start)
        next_tick += cfg.control_dt
        sleep_time = next_tick - now()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            episode_stats.overruns += 1
            ## After a stall longer than one period (a solver hiccup, an 'i'
            ## re-park, a save pause) the old behaviour was to free-run with
            ## no sleep until next_tick caught back up -- a burst of ticks at
            ## whatever rate the machine can manage, recorded as if they were
            ## control_dt apart.  Lost time cannot be recovered, only
            ## misrepresented: drop it and resume the schedule from now, so
            ## every recorded tick really is ~control_dt from its neighbour.
            if -sleep_time > cfg.control_dt:
                next_tick = now()

if __name__ == "__main__":
    main()
