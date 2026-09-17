"""
Configuration for teleoperation and dataset collection.
"""

from dataclasses import dataclass, field
from typing import Optional
from scipy.spatial.transform import Rotation as R
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATASET_ROOT as _DATASET_ROOT, PYROKI_EXAMPLES

sys.path.append(str(PYROKI_EXAMPLES))

import os

import numpy as np

DATASET_ROOT = str(_DATASET_ROOT)

TASKS = {
    1: "screwdriver_insertion",
    2: "block_square",
    3: "grasp_cube",
    4: "transfer_flower",
    5: "bimanual",
    6: "active_vision",
    7: "active_vision_data_collection",  # coupled-IK study deployment
    ## Separate task name, NOT a second run under "transfer_flower": every
    ## pre-July-2026 run of that task was recorded at 0.8-2.6 Hz while stamped
    ## 15/30/50 fps (see dataset/lerobot/DEFECT_pre_2026_07_datasets.md).
    ## Sharing a task directory would put defective and good runs side by side
    ## under one repo_id, which is exactly how they end up pooled by accident.
    8: "transfer_flower_v2",
    ## Pick-and-insert into the sorter box; the per-episode TARGET piece is
    ## carried in the frame task string (shape_sorter.task_string), so one run
    ## can hold cube, triangle and flower episodes side by side.  Pass
    ## --task shape_sorter --target <piece> to data_collection.py.
    9: "shape_sorter",
}

# Defines which arms are active in each mode, and the corresponding action layout.

ARM_MODES = {
    "left": ["left"],
    "right": ["right"],
    "middle": ["middle"],
    "bimanual": ["left", "right"],
    "all": ["left", "right", "middle"],
    "right_av": ["right", "middle"],
    "left_av": ["left", "middle"],
}

## ACTION_LAYOUTS maps a mode to WHERE each arm's command sits in the recorded
## action vector.  It is GENERATED, not written out, because the recorder builds
## that vector in dataset.build_action_names() -- per arm in ARM_MODES order,
## joints then gripper if the arm has one -- and a second hand-maintained copy
## of the same fact is a silent replay-into-the-wrong-joints bug waiting to
## happen.  "middle" had no entry here at all, so replaying a middle-only
## dataset raised KeyError; generating the table means every mode in ARM_MODES
## gets a layout by construction, including any mode added later.
def _arm_config():
    """ARM_CONFIG, however this module was imported (package or flat).

    Local + dual-path for the same reason every other module here does it:
    data_collection.py is run both as a script and as part of the av_aloha
    package, and a bare absolute import breaks the package case.
    """
    try:
        from .arm_config import ARM_CONFIG
    except ImportError:
        from arm_config import ARM_CONFIG
    return ARM_CONFIG


def _build_action_layout(arms):
    ARM_CONFIG = _arm_config()

    layout, i = {}, 0
    for arm in arms:
        n = ARM_CONFIG[arm]["num_joints"]
        layout[f"{arm}_arm"] = slice(i, i + n)
        i += n
        if ARM_CONFIG[arm]["has_gripper"]:
            layout[f"{arm}_gripper"] = i
            i += 1
    return layout


ACTION_LAYOUTS = {mode: _build_action_layout(arms) for mode, arms in ARM_MODES.items()}


def action_dim(mode):
    """Length of the recorded action vector for `mode`."""
    ARM_CONFIG = _arm_config()

    return sum(ARM_CONFIG[a]["num_joints"] + int(ARM_CONFIG[a]["has_gripper"])
               for a in ARM_MODES[mode])


## Frozen expectation for the layouts that existed before generation (2026-08).
## If ARM_CONFIG ever changes shape, this fails loudly at import rather than
## quietly re-indexing every dataset ever recorded.
_FROZEN = {
    "left":     {"left_arm": slice(0, 6), "left_gripper": 6},
    "right":    {"right_arm": slice(0, 6), "right_gripper": 6},
    "bimanual": {"left_arm": slice(0, 6), "left_gripper": 6,
                 "right_arm": slice(7, 13), "right_gripper": 13},
    "all":      {"left_arm": slice(0, 6), "left_gripper": 6,
                 "right_arm": slice(7, 13), "right_gripper": 13,
                 "middle_arm": slice(14, 21)},
    "right_av": {"right_arm": slice(0, 6), "right_gripper": 6,
                 "middle_arm": slice(7, 14)},
    "left_av":  {"left_arm": slice(0, 6), "left_gripper": 6,
                 "middle_arm": slice(7, 14)},
}
for _m, _exp in _FROZEN.items():
    assert ACTION_LAYOUTS[_m] == _exp, (
        f"ACTION_LAYOUTS['{_m}'] changed: {ACTION_LAYOUTS[_m]} != {_exp}. "
        f"Existing datasets are indexed by the frozen layout; fix ARM_CONFIG "
        f"or migrate the data -- do not relax this assert.")
del _m, _exp

def clamp_joint_step(q_curr, q_target, max_step):
    dq = np.clip(q_target - q_curr, -max_step, max_step)
    return q_curr + dq

def clamp_cartesian_step(target, prev_target, max_step):
    delta = target - prev_target
    norm = np.linalg.norm(delta)
    if norm > max_step and norm > 1e-9:
        delta *= (max_step / norm)
    return prev_target + delta


def clamp_angular_step(target_rot, prev_target_rot, max_step):
    """`target_rot`, but no more than `max_step` radians from the last target.

    The rotational twin of clamp_cartesian_step, and deliberately the same
    shape: clamp against the PREVIOUS TARGET, not the measured pose, so a
    sustained turn keeps advancing max_step per tick and the target converges
    on the hand once it stops.  Clamping against the measurement instead
    would let tracking error accumulate into a permanent offset."""
    if max_step <= 0 or prev_target_rot is None:
        return target_rot
    step = R.from_matrix(target_rot @ prev_target_rot.T).as_rotvec()
    ang = float(np.linalg.norm(step))
    if ang <= max_step or ang < 1e-9:
        return target_rot
    return R.from_rotvec(step * (max_step / ang)).as_matrix() @ prev_target_rot


def quat_xyzw_to_wxyz(q_xyzw):
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)

@dataclass
class TeleopConfig:
    # Control rate + scales, env-tunable (2026-08) so data_collection can be
    # tuned to match teleop_debug_tool behavior without code edits:
    #   GIAVA_CONTROL_HZ=20  GIAVA_POS_SCALE=1.0  GIAVA_CAM_POS_SCALE=0.6
    #   GIAVA_MOVING_TIME=0.14
    # Changing the rate also rescales the IK smoothing automatically
    # (CoupledStudyIK receives control_dt).
    control_dt: float = field(
        default_factory=lambda: 1.0 / float(os.environ.get("GIAVA_CONTROL_HZ", "50"))
    )
    position_scale: float = field(
        default_factory=lambda: float(os.environ.get("GIAVA_POS_SCALE", "1.35"))
    )
    ## ROTATION GAIN, the twin of position_scale.  Until 2026-09-11 there was
    ## no such thing: translation was scaled and orientation was applied 1:1,
    ## so turning GIAVA_POS_SCALE down to tame the arm made translation
    ## sluggish while leaving the wrist as fast as the operator's hand --
    ## slow where you want authority, quick where you want care.
    ##
    ## Scaling a rotation means scaling its ROTATION VECTOR: the axis is kept
    ## and the angle multiplied, which is the geodesic interpolation from
    ## identity.  Default 1.0, so nothing changes unless asked.
    rotation_scale: float = field(
        default_factory=lambda: float(os.environ.get("GIAVA_ROT_SCALE", "1.0"))
    )
    alpha: float = 0.3
    arm_cmd_dt: float = field(
        default_factory=lambda: 1.0 / float(os.environ.get("GIAVA_CONTROL_HZ", "50"))
    )
    moving_time: float = field(
        default_factory=lambda: float(os.environ.get("GIAVA_MOVING_TIME", "0.14"))
    )
    accel_time: float = 0.04
    max_ee_step: float = 0.02
    ## Per-tick ANGULAR step ceiling, the twin of max_ee_step (0.02 m/tick,
    ## i.e. 1 m/s at 50 Hz).  Nothing bounded angular rate at all: the only
    ## thing standing between a flicked wrist and the servos was the driver
    ## clamp.  In rad/tick -- 0.04 at 50 Hz is 2 rad/s, about 115 deg/s.
    ##
    ## DEFAULT 0 = OFF, deliberately.  A clamp that silently trimmed fast
    ## motions would make new demonstrations differ from the ones already
    ## recorded, so this is opt-in via GIAVA_MAX_EE_ANG_STEP.
    max_ee_ang_step: float = field(
        default_factory=lambda: float(os.environ.get("GIAVA_MAX_EE_ANG_STEP", "0"))
    )
    pos_weight: float = 40.0
    ori_weight: float = 0.25
    dq_weight: float = 0.18
    joint_reached_tol: float = 0.03
    ee_reached_tol: float = 0.01
    cmd_timeout: float = 0.25
    full_joint_velocity_limits_value: float = 2.3

    # Post-solve per-joint step clamp.  DISABLED to match the ik_study
    # conditions: the study's winner was validated with no clamp, no LPF and no
    # reseed, and a saturating clamp creates the acceleration discontinuities it
    # is meant to prevent (the solver never learns its command was truncated).
    #
    # WARNING: nothing else bounds joint velocity.  The smoothing cost is a soft
    # penalty on deviation from the previous configuration -- it makes large
    # steps expensive, it does not make them impossible.  With this off, an IK
    # discontinuity goes straight to the motors.  Re-enable with
    # GIAVA_ENABLE_JOINT_CLAMP=1 if the arms move harder than you expect.
    enable_joint_clamp: bool = field(
        default_factory=lambda: os.environ.get("GIAVA_ENABLE_JOINT_CLAMP", "0") == "1"
    )

    # Driver-feasibility clamp (separate from the tuning clamp above, and ON by
    # default).  interbotix's `check_joint_limits` rejects the WHOLE group
    # command if any joint fails, so a single infeasible joint freezes the arm
    # completely -- it does not clip, it refuses.  Its test is
    #     |goal - last_command| / moving_time  >  joint_velocity_limit
    # so the largest command it will accept is  vl * moving_time  per tick.
    # Clamping to just under that keeps every command executable while leaving
    # far more headroom than the old tuning clamp (0.05-0.10 rad).
    enable_driver_clamp: bool = True

    ## WHICH PROFILE THE SERVOS ARE RUNNING, and what that makes the clamp
    ## mean.  DEFAULT IS "legacy": today's numbers exactly, so nothing changes
    ## under a recording session.
    ##
    ##   GIAVA_PROFILE_MODE=velocity  (default) driver_max_step = v_cap *
    ##                                control_dt, i.e. how far the servo can
    ##                                actually travel in ONE TICK at its
    ##                                firmware speed cap.  With v_cap =
    ##                                3.36 rad/s and dt = 0.02 s that is
    ##                                0.067 rad.  The goal then never sits more
    ##                                than one tick ahead of the arm, so
    ##                                tracking error stays small and bounded.
    ##
    ##   GIAVA_PROFILE_MODE=legacy    driver_max_step = driver_clamp_safety *
    ##                                driver_velocity_limit * moving_time =
    ##                                0.9*3.14159*0.14 = 0.396 rad.  TIME-BASED
    ##                                arithmetic: "how far can the arm travel in
    ##                                one moving_time at the velocity limit".
    ##                                Only correct if the servos really are
    ##                                running a time-based profile.
    ##
    ##   GIAVA_PROFILE_MODE=auto      read Drive_Mode off the motors at startup
    ##                                and pick. It needs the bus up before the
    ##                                config is built, so the control loop calls
    ##                                apply_profile_limits().
    ##
    ## WHY THE DEFAULT CHANGED (2026-09-09).  It was "legacy" -- time-based
    ## arithmetic -- while all three puppet_modes_*.yaml set
    ## `profile_type: velocity`.  Under a velocity-based profile Drive_Mode
    ## bit 2 is clear, so the Profile_Velocity the interbotix layer writes
    ## (moving_time*1000 = 140) is NOT 140 ms; it is 140 * 0.229 rev/min, a
    ## 3.36 rad/s SPEED CAP.  The loop was therefore allowed 0.396 rad/tick
    ## against a servo that can execute 0.067 -- a 5.9x gap, with the
    ## commanded goal outrunning the arm by ~16 rad/s.
    ##
    ## That is not a tuning preference, it is a runaway: tracking error grows
    ## without bound until the watchdog trips at TRIP_RAD (0.7 rad), while the
    ## joint chases its goal at stall current until the motor latches an
    ## overload and shuts its own torque off.  Recorded on hardware as
    ## middle_base peaking at 2095 mA (limit ~2300) and then sitting at zero
    ## effort, frozen, while the command walked 48 degrees away -- and the
    ## wedged bus stalling the 50 Hz loop for up to 1.65 s, which is what took
    ## the arms, the headset control and the camera stream down together.
    ## Confirmed the other way on 2026-09-09: a session run with "velocity"
    ## behaved markedly better.
    ##
    ## The cost of the default is that the largest single-tick command is ~6x
    ## smaller, which IS a real change in feel.  GIAVA_PROFILE_MODE=legacy
    ## restores the old numbers exactly if a session needs to match older
    ## recordings -- but note the old numbers are the ones that latched the
    ## overloads.
    ##
    ## If the registers cannot be read, driver_max_step below falls back to
    ## driver_velocity_limit (0.9*3.14159*0.02 = 0.057 rad/tick) rather than
    ## the measured cap.  That is tighter than the truth, not looser, which is
    ## the right direction to fail in.
    profile_mode: str = field(
        default_factory=lambda: os.environ.get(
            "GIAVA_PROFILE_MODE", "velocity").strip().lower()
    )
    ## Measured firmware speed cap [rad/s], filled in by apply_profile_limits()
    ## when the registers are readable.  None = never read.
    servo_v_cap: Optional[float] = None
    ## Set by apply_profile_limits() to override the computed driver_max_step.
    driver_max_step_override: Optional[float] = None
    # Headroom fraction of the driver-feasible step (env-tunable: this is the
    # safety knob to slow the arms down globally WITHOUT the old per-joint
    # tuning clamp and its saturate-release jerk).
    #   0.9 -> max ~2.8 rad/s equivalent; 0.45 -> ~1.4 rad/s.
    driver_velocity_limit: float = 3.141593  # rad/s, from group_info
    driver_clamp_safety: float = field(
        default_factory=lambda: float(os.environ.get("GIAVA_DRIVER_CLAMP_SAFETY", "0.9"))
    )
    driver_limit_margin: float = 0.03        # rad cushion inside each DRIVER
    # position limit.  0.005 proved too thin: commands landing exactly on the
    # limit get rejected outright (right wrist_angle at +2.243 vs +2.234), and a
    # rejected command freezes the driver's reference.  ~1.7 deg of range given
    # up per joint end buys never hitting the wall.

    @property
    def driver_max_step(self) -> float:
        """Largest per-tick joint delta this config will command.

        Three sources, in precedence order:
          1. an explicit override from apply_profile_limits()
          2. profile_mode "velocity": one TICK of travel at the servo's own
             speed cap -- the physically meaningful bound when the servo is
             enforcing a velocity ceiling
          3. legacy: one moving_time of travel at the URDF velocity limit,
             which is time-based-profile arithmetic (see profile_mode)
        """
        if self.driver_max_step_override is not None:
            return float(self.driver_max_step_override)
        if self.profile_mode == "velocity":
            v = self.servo_v_cap if self.servo_v_cap else self.driver_velocity_limit
            return self.driver_clamp_safety * float(v) * self.control_dt
        return self.driver_clamp_safety * self.driver_velocity_limit * self.moving_time

    # Per-instance NumPy arrays via default_factory
    max_joint_step: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.05, 0.05, 0.06, 0.08, 0.08, 0.10],
            dtype=float,
        )
    )
    # 7-dof middle arm (wx250s_7dof: extra wrist joint); same per-joint
    # progression with a wrist-class limit for the 7th
    max_joint_step_middle: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.05, 0.05, 0.06, 0.08, 0.08, 0.10, 0.10],
            dtype=float,
        )
    )
    R_arm_remap: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            dtype=float,
        )
    )
    # Head pose (HPosition/HRotation) arrives from the SAME headset runtime and
    # world frame as the hand controllers, so the camera arm needs the SAME
    # remap as the gripper arms.  This was identity (i.e. never filled in),
    # which fed raw headset axes to the camera target: lateral head motion
    # mapped to robot +y (operator-backward) instead of +x (operator-left), and
    # head yaw (about headset up) was applied about the robot's backward axis --
    # producing the observed pitch/yaw swaps.  With the shared remap, the
    # world-frame rotation composition (R @ dR @ R^T) sends head yaw -> robot
    # yaw (about +z), pitch -> pitch (about x), roll -> roll (about y).
    # Remap for the middle (head) target.  The head pose is converted from its
    # native Unity-style basis into the CONTROLLER convention at parse time
    # (HEAD_BASIS_FIX in data_collection.py), so the same remap as the gripper
    # arms applies here.  Net head->robot mapping is identical to the previous
    # dedicated matrix (M @ C == B, verified); doing the conversion once at the
    # source also fixes head motion leaking into the gripper arms through the
    # head-relative hand composition.
    R_cam_remap: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            dtype=float,
        )
    )

    # Camera-arm sensitivity.  The head is never still -- breathing and weight
    # shifts move it by millimetres continuously, and at position_scale 1.0 the
    # camera arm chases all of it.  The deadband zeroes deltas below ~1.5 cm
    # (soft-edged, so there is no snap at the threshold) and the separate scale
    # lets head motion map sub-unity to arm motion.
    cam_position_scale: float = field(
        default_factory=lambda: float(os.environ.get("GIAVA_CAM_POS_SCALE", "0.6"))
    )
    cam_deadband_m: float = field(
        default_factory=lambda: float(os.environ.get("GIAVA_CAM_DEADBAND_M", "0.015"))
    )
    # Camera-arm step-clamp window.  The shared max_ee_step (0.02) clamps the
    # target to within 2 cm of the arm's CURRENT pose; at position weight 10
    # that caps the IK's position cost at (10*0.02)^2 = 0.04 -- a whisper next
    # to the orientation terms, so the position channel starves (x/y appear
    # dead).  A 5 cm window gives the position cost a signal worth acting on
    # while keeping the anti-windup property.  See TELEOP_MATH.md section 4.
    cam_max_ee_step: float = field(
        default_factory=lambda: float(os.environ.get("GIAVA_CAM_MAX_EE_STEP", "0.05"))
    )

# State for each arm during teleoperation, tracking the initial controller and robot poses, as well as a filtered target position for smooth motion.
@dataclass
class ArmTeleopState:
    active: bool = False
    start_controller_pos: Optional[np.ndarray] = None
    start_controller_rot: Optional[np.ndarray] = None
    start_robot_pos: Optional[np.ndarray] = None
    start_robot_rot: Optional[np.ndarray] = None
    filtered_target_pos: Optional[np.ndarray] = None
    # Session-calibrated remap (raw app world -> robot), built at anchor time
    # from the head's horizontal gaze.  The probe (2026-08) measured the app's
    # world frame as z-up but with an ARBITRARY per-session yaw (+43 deg that
    # run) -- set by where the headset faced at app start -- so a fixed remap
    # matrix is wrong by a different angle every session.  None = fall back to
    # the static matrix (pre-calibration behavior).
    session_remap: Optional[np.ndarray] = None

# Overall teleoperation session state, including whether it's active and the state for each arm.
@dataclass
class TeleopSessionState:
    active: bool = False
    arms: dict[str, ArmTeleopState] = field(default_factory=dict)

# State for tracking the last command times and values for each arm, used to implement command timeouts and ensure smooth control.
@dataclass
class RobotCommandState:
    last_arm_cmd_time: float = 0.0
    last_cmds: dict[str, np.ndarray] = field(default_factory=dict)

# Kinematics state for the commanded target poses of each arm, used to compute the desired end-effector positions and orientations based on the controller input and initial poses.
@dataclass
class CommandKinematicsState:
    q_cmd: Optional[np.ndarray] = None
    T_cmd: dict[str, np.ndarray] = field(default_factory=dict)

# Initializes the teleoperation session state for the active arms based on the current controller poses and commanded kinematics.
## session_yaw_remap and HEAD_LOCAL_FWD live in transform_utils (the shared
## frame-math helper) and are re-exported here for the teleop call sites.
try:
    from .transform_utils import HEAD_LOCAL_FWD, session_yaw_remap  # noqa: F401
except ImportError:
    from transform_utils import HEAD_LOCAL_FWD, session_yaw_remap  # noqa: F401


def anchor_arm_state(state_arm, controller_pose, cmd_pose, head_pose=None, base_remap=None):
    """(Re)anchor one arm's teleop reference frames to NOW.

    Called at session start for every arm, and again whenever an individual
    arm's button is pressed mid-session: with per-arm buttons (Y = middle),
    an arm can become active long after the session started, and its anchor
    from session start would be stale -- the arm would jump by however far
    the controller/head moved in between."""
    quat_wxyz = cmd_pose[:4]
    pos = cmd_pose[4:]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot = R.from_quat(quat_xyzw).as_matrix()
    state_arm.start_controller_pos = controller_pose[:3, 3].copy()
    state_arm.start_controller_rot = controller_pose[:3, :3].copy()
    state_arm.start_robot_pos = np.asarray(pos).copy()
    state_arm.start_robot_rot = rot.copy()
    state_arm.filtered_target_pos = np.asarray(pos).copy()
    ## Anchor the angular clamp too, or the first tick after an enable would
    ## measure its step against a rotation from the previous session.
    state_arm.prev_target_rot = rot.copy()
    if head_pose is not None and base_remap is not None:
        remap = session_yaw_remap(head_pose, base_remap)
        if remap is not None:
            state_arm.session_remap = remap
        elif state_arm.session_remap is None:
            print("[frame] head gaze too vertical at anchor -- using static remap")


def start_teleop_session(state, mode, controller_poses, cmd_kin, cfg=None,
                         head_pose=None):
    """Anchor every arm in `mode` to the current controller/head poses.

    `head_pose` is the RAW head pose, used only to build the session yaw remap
    (session_yaw_remap): the app's world frame is z-up with an arbitrary
    per-session yaw set by wherever the headset faced at app start, and the
    head's horizontal gaze is what resolves it.  It defaults to
    controller_poses["middle"] because the camera arm normally FOLLOWS the
    head, so the two are the same pose -- but they stop being the same the
    moment the camera arm is driven from a hand controller instead
    (teleop.py --camera-source right).  A controller's forward axis is wherever
    the operator happens to be pointing and says nothing about world yaw, so
    deriving the remap from it would rotate every arm's mapping by a random
    angle.  Callers that redirect the camera arm must pass the real head pose
    here."""
    state.active = True

    if head_pose is None:
        head_pose = controller_poses.get("middle")
    for arm in ARM_MODES[mode]:
        state.arms[arm] = ArmTeleopState()
        base = None
        if cfg is not None:
            base = cfg.R_cam_remap if arm == "middle" else cfg.R_arm_remap
        anchor_arm_state(state.arms[arm], controller_poses[arm], cmd_kin.T_cmd[arm],
                         head_pose=head_pose, base_remap=base)

## Escape hatch for the 2026-08-27 orientation fix below: the old path is one
## env var away, so the two can be compared on hardware in one session instead
## of by editing between runs.
ARM_ORI_LEGACY = os.environ.get("GIAVA_ARM_ORI_LEGACY", "0") == "1"


## ---------------------------------------------------------------------------
## CONTROLLER DISCONTINUITY GATE
##
## The existing tracking guards in data_collection.py catch two failure shapes:
## an exactly-zero pose (the headset says "lost"), and a BIT-IDENTICAL pose held
## for half a second (asleep / out of volume).  Neither catches the shape that
## actually moves the arms: a pose that is nonzero, changing, and WRONG.
##
##   - a dropout shorter than the 0.5 s freeze threshold, where the pose comes
##     back somewhere else -- no guard fires, so the arm is never held, never
##     re-anchored, and the whole re-acquire offset is applied as a delta
##   - occlusion (hand behind the other arm, controller edge-of-volume), where
##     the runtime keeps predicting from IMU and the pose slides metres away
##   - orientation-only corruption: position stays plausible, the quaternion
##     jumps, and the gripper rolls over while the hand is still
##
## And because the target path is EMA-filtered and then step-clamped
## (compute_gripper_arm_target below), a bad delta does not show up as a single
## violent jump that is obviously wrong -- it shows up as the arm SLEWING
## smoothly and confidently to the wrong place over a second or two, which is
## exactly what "the arms moved a lot while my hands were still" looks like.
##
## This gate works on the raw controller pose, before any remap or scaling, and
## asks one question: could a human hand have done that in the time available?
## A hand tops out around 2 m/s in normal teleoperation and perhaps 10 rad/s in
## a wrist flick; tracking failures produce tens of m/s.  There is a wide gap
## between the two, which is what makes this checkable without tuning.
##
## When a pose fails, the arm is HELD (not moved to the bad pose, not moved back)
## and stays held until the pose has been plausible for CTRL_SETTLE_TICKS in a
## row.  Releasing the hold is a rising edge on arm_active, which re-anchors the
## arm through the existing path -- so teleop resumes from wherever the hand
## really is, with no accumulated offset to catch up.
##
## Elapsed time is MEASURED, not assumed: a late tick (the IK solver overran,
## the loop stalled) legitimately carries a bigger displacement, and comparing
## it against the nominal period would fire the gate on our own jitter.
CTRL_MAX_SPEED = float(os.environ.get("GIAVA_CTRL_MAX_SPEED", "3.0"))        # m/s
CTRL_MAX_ANG_SPEED = float(os.environ.get("GIAVA_CTRL_MAX_ANG_SPEED", "12.0"))  # rad/s
CTRL_SETTLE_TICKS = int(os.environ.get("GIAVA_CTRL_SETTLE_TICKS", "5"))

## DEAD-MAN TIMER.  Seconds of CONTINUOUS teleop engagement after which the
## session disengages itself and the arms hold.  0 = off (the default), so
## nothing changes for anyone who does not ask for it.
##
## Holding a grip button is the only thing that keeps the arms following, but
## "holding" is easy to keep doing while attention is elsewhere -- and with a
## headset on, the operator cannot see the arm's real surroundings.  This
## bounds how long one grip can drive the robot.
##
## Re-engaging requires RELEASING the button first: after a timeout the same
## held button would otherwise re-enable on the very next tick, which is no
## limit at all.  data_collection.py holds that latch.
##
## Choose it above your longest episode.  A timeout mid-demonstration
## disengages while recording, which does not corrupt the episode -- the arms
## simply hold and the frames keep being written -- but it does spoil the take.
MAX_ENGAGE_S = float(os.environ.get("GIAVA_MAX_ENGAGE_S", "0"))
CTRL_GATE_ON = os.environ.get("GIAVA_CTRL_JUMP_GATE", "1").strip() not in ("0", "false", "no")


class ControllerJumpGate:
    """Rejects controller poses that moved faster than a hand can move."""

    def __init__(self, max_speed=CTRL_MAX_SPEED, max_ang_speed=CTRL_MAX_ANG_SPEED,
                 settle_ticks=CTRL_SETTLE_TICKS, enabled=CTRL_GATE_ON):
        self.max_speed = max_speed
        self.max_ang_speed = max_ang_speed
        self.settle_ticks = settle_ticks
        self.enabled = enabled
        self._prev_pos = {}
        self._prev_rot = {}
        self._prev_t = {}
        self._settle = {}
        self.holds = {}          # name -> ticks held, this session
        self.events = {}         # name -> discontinuities seen

    def reset(self, name=None):
        """Forget history (use on teleop enable, after a deliberate reset)."""
        for d in (self._prev_pos, self._prev_rot, self._prev_t, self._settle):
            d.pop(name, None) if name is not None else d.clear()

    def check(self, name, pos, quat_xyzw, now_s):
        """(usable, reason).  `usable` False means HOLD this arm."""
        if not self.enabled:
            return True, None

        pos = np.asarray(pos, dtype=float).reshape(3)
        rot = None
        if quat_xyzw is not None:
            q = np.asarray(quat_xyzw, dtype=float).reshape(4)
            ## A zero or non-unit quaternion is corruption, not a pose.
            if np.isfinite(q).all() and abs(np.linalg.norm(q) - 1.0) < 0.1:
                rot = R.from_quat(q)

        prev_pos = self._prev_pos.get(name)
        prev_rot = self._prev_rot.get(name)
        prev_t = self._prev_t.get(name)
        self._prev_pos[name] = pos
        self._prev_rot[name] = rot
        self._prev_t[name] = now_s

        if prev_pos is None or prev_t is None:
            self._settle[name] = self.settle_ticks     # first sample: trust it
            return True, None

        ## Clamped: a very long gap (teleop was off, the process stalled) makes
        ## any displacement "plausible", which is the safe direction -- the
        ## rising edge that follows re-anchors anyway.
        dt = min(max(now_s - prev_t, 1e-3), 0.5)

        reason = None
        speed = float(np.linalg.norm(pos - prev_pos)) / dt
        if speed > self.max_speed:
            reason = f"{speed:.1f} m/s"

        if reason is None and rot is not None and prev_rot is not None:
            ## Relative rotation angle; scipy returns the shortest one, so a
            ## quaternion sign flip does not read as a 2 pi rotation.
            ang = float(np.linalg.norm((rot * prev_rot.inv()).as_rotvec()))
            if ang / dt > self.max_ang_speed:
                reason = f"{ang / dt:.1f} rad/s"
        elif reason is None and quat_xyzw is not None and rot is None:
            reason = "bad quaternion"

        if reason is not None:
            self._settle[name] = 0
            self.events[name] = self.events.get(name, 0) + 1
            self.holds[name] = self.holds.get(name, 0) + 1
            return False, reason

        settled = self._settle.get(name, 0) + 1
        self._settle[name] = settled
        if settled < self.settle_ticks:
            self.holds[name] = self.holds.get(name, 0) + 1
            return False, "settling"
        return True, None

    def summary(self):
        """One line, or None when nothing was ever held."""
        if not self.events:
            return None
        parts = [f"{k}: {v} discontinuit{'y' if v == 1 else 'ies'}, "
                 f"{self.holds.get(k, 0)} ticks held" for k, v in
                 sorted(self.events.items())]
        return "[tracking] " + "; ".join(parts)


# Computes the target end-effector position and orientation for a given arm based on the current controller pose, the initial poses, and the configuration parameters.
def compute_gripper_arm_target(cfg, arm_state, controller_pose, current_pose, remap_matrix):
    if arm_state.session_remap is not None:
        remap_matrix = arm_state.session_remap
    delta_ctrl = controller_pose[:3, 3] - arm_state.start_controller_pos
    delta_robot = remap_matrix @ delta_ctrl
    raw_target = arm_state.start_robot_pos + cfg.position_scale * delta_robot
    arm_state.filtered_target_pos = cfg.alpha * raw_target + (1.0 - cfg.alpha) * arm_state.filtered_target_pos
    target_pos = clamp_cartesian_step(arm_state.filtered_target_pos, np.asarray(current_pose[4:], dtype=float), cfg.max_ee_step)

    ## ORIENTATION: world-frame delta, conjugated by the SAME remap the
    ## position path uses.  Identical in form to compute_camera_arm_target
    ## below, which is the version that works.
    ##
    ## WHAT THIS REPLACES, AND WHY IT WAS WRONG (hardware, 2026-08-27:
    ## "pitch up causes yaw right, yaw left causes pitch up, and roll is
    ## backwards on the right arm but correct on the left")
    ##
    ##     R_delta = start_controller_rot.T @ controller_pose[:3,:3]   # LOCAL
    ##     rotvec  = as_rotvec(R_delta)
    ##     rotvec_ee = [-rotvec[1], rotvec[0], rotvec[2]]              # by hand
    ##     target_rot = start_robot_rot @ from_rotvec(rotvec_ee)       # LOCAL
    ##
    ## Three faults, compounding:
    ##
    ## 1. THE HAND PERMUTATION IS THE REMAP INVERTED.  Conjugating a rotation
    ##    by a rotation M sends its axis to M @ axis, so a component swap IS
    ##    a frame change -- but [x,y,z] -> [-y,x,z] is exactly M.T for
    ##    M = R_arm_remap, not M.  The orientation path was rotating the axis
    ##    90 deg the opposite way from the position path, which turns pitch
    ##    into yaw and yaw into -pitch.  That is the reported swap, exactly.
    ##
    ## 2. THE DELTA WAS LOCAL, THE POSITION DELTA IS WORLD.  `start.T @ now`
    ##    means "about the controller's own axes as they were at anchor";
    ##    position uses `remap @ (p_now - p_start)`, a world displacement.
    ##    One hand motion was being interpreted in two different frames.
    ##
    ## 3. LOCAL COMPOSITION MADE THE TWO ARMS DISAGREE.  `start_robot_rot @
    ##    delta` rotates about the GRIPPER's own axes, and the left arm's
    ##    base carries rpy yaw = pi relative to the right (giava.urdf), so
    ##    its link frame is turned 180 deg about z.  The same commanded roll
    ##    therefore comes out mirrored between the arms -- which is why roll
    ##    read correct on the left and backwards on the right, from ONE code
    ##    path with no per-arm branch.  World composition removes the link
    ##    frame from the answer entirely; this is the same reasoning already
    ##    written out for the camera arm.
    ##
    ## Set GIAVA_ARM_ORI_LEGACY=1 to restore the old path for comparison.
    if ARM_ORI_LEGACY:
        R_delta = arm_state.start_controller_rot.T @ controller_pose[:3, :3]
        rotvec = R.from_matrix(R_delta).as_rotvec()
        rotvec_ee = np.array([-rotvec[1], rotvec[0], rotvec[2]])
        target_rot = arm_state.start_robot_rot @ R.from_rotvec(rotvec_ee).as_matrix()
    else:
        R_delta_world = controller_pose[:3, :3] @ arm_state.start_controller_rot.T
        R_delta_robot = remap_matrix @ R_delta_world @ remap_matrix.T
        ## Gain, applied to the rotation VECTOR: same axis, scaled angle.
        if cfg.rotation_scale != 1.0:
            _rv = R.from_matrix(R_delta_robot).as_rotvec()
            R_delta_robot = R.from_rotvec(cfg.rotation_scale * _rv).as_matrix()
        target_rot = R_delta_robot @ arm_state.start_robot_rot
    ## Rate limit, against the previous target (see clamp_angular_step).
    target_rot = clamp_angular_step(
        target_rot, getattr(arm_state, "prev_target_rot", None),
        cfg.max_ee_ang_step)
    arm_state.prev_target_rot = target_rot
    target_wxyz = quat_xyzw_to_wxyz(R.from_matrix(target_rot).as_quat())

    return target_pos, target_wxyz

def compute_camera_arm_target(cfg, arm_state, controller_pose, current_pose, remap_matrix):
    if arm_state.session_remap is not None:
        remap_matrix = arm_state.session_remap
    delta_ctrl = controller_pose[:3, 3] - arm_state.start_controller_pos
    delta_robot = remap_matrix @ delta_ctrl
    # Soft deadband: ignore the head's constant millimetre-level wander without
    # a snap when crossing the threshold (magnitude shrinks by the deadband).
    mag = float(np.linalg.norm(delta_robot))
    if mag <= cfg.cam_deadband_m:
        delta_robot = np.zeros(3)
    else:
        delta_robot = delta_robot * ((mag - cfg.cam_deadband_m) / mag)
    raw_target = arm_state.start_robot_pos + cfg.cam_position_scale * delta_robot
    arm_state.filtered_target_pos = cfg.alpha * raw_target + (1.0 - cfg.alpha) * arm_state.filtered_target_pos
    target_pos = clamp_cartesian_step(arm_state.filtered_target_pos, np.asarray(current_pose[4:], dtype=float), cfg.cam_max_ee_step)

    # Head rotation delta composed in the WORLD frame (delta @ start), not
    # the EE-local frame (start @ delta): local-frame application rotates
    # about the camera link's own axes, which do NOT line up with the
    # world's — at the forward pose the link z is horizontal, so head yaw
    # became camera pitch.  World-frame composition keeps yaw = yaw and
    # pitch = pitch regardless of the link's local frame convention.
    R_delta_world = controller_pose[:3, :3] @ arm_state.start_controller_rot.T
    R_delta_robot = remap_matrix @ R_delta_world @ remap_matrix.T
    target_rot = R_delta_robot @ arm_state.start_robot_rot
    target_wxyz = quat_xyzw_to_wxyz(R.from_matrix(target_rot).as_quat())

    return target_pos, target_wxyz

def stop_teleop_session(state):
    state.active = False


## ------------------------------------------------------------------ link loss

## Remote operation over Tailscale changes what a gap in the uplink means.  On a
## LAN a missed packet is sub-frame and invisible; over a WireGuard tunnel that
## may be relayed through DERP, the stream can stall for whole seconds.
##
## Both control loops already `continue` past a stale frame, which holds the
## arms -- correct -- but they left teleop_state.active set.  On recovery the
## arm therefore resumed against an anchor taken BEFORE the gap and swept to
## wherever the controller had drifted meanwhile.  anchor_arm_state's docstring
## describes exactly that failure for the mid-session activation case; a link
## gap is the same fault arriving by a different route, and it was the one route
## with no guard on it.
##
## Note what this deliberately reuses: clearing each arm's `active` flag makes
## the next fresh frame an inactive->active edge, which is already wired to
## re-anchor.  That is the same path a frozen or lost controller takes, so
## recovery from a short gap is seamless rather than a jump, with no new
## anchoring logic to keep in step with the old.
LINK_DISARM_S = float(os.environ.get("GIAVA_LINK_DISARM_S", 2.0))


class LinkGuard:
    """Refuses motion for the one reason a remote link creates: a gap in it.

    Short gap: every arm is de-activated so the existing re-anchor edge fires
    on the next frame.  Long gap (past LINK_DISARM_S): the session ends as
    well, and stays latched until the operator RELEASES the drive button.
    Past a couple of seconds the operator has been flying blind -- the video
    died with the input -- and a button they never let go of is not consent to
    start moving again.

    This class also used to gate motion on the headset's deadman bit (input
    protocol v3, `INPUT_DEADMAN`), removed 2026-09-01.  Requiring a held grip
    made teleoperation harder to read, because an arm that will not move looks
    identical whether the deadman is released, the link is latched, or the
    drive button simply is not pressed.  The viewer still sends the bit and
    gvlink still decodes it as `pkt.deadman`; nothing in this repo reads it,
    so no headset build change was needed to turn it off.

    `apply` still takes `headset_data` although it no longer reads it -- it is
    the natural place for a future per-tick veto that does need headset state.
    """

    def __init__(self, disarm_s=None, log=print):
        self.disarm_s = LINK_DISARM_S if disarm_s is None else disarm_s
        self._log = log
        self._stale_since = None
        self._rearm_block = False
        self._last_rearm_log = -1e9

    def missed(self, t):
        """Call on every tick where receive_data() returned None."""
        if self._stale_since is None:
            self._stale_since = t

    def recovered(self, t, teleop_state):
        """Call on the first fresh frame after a gap.

        Returns the gap in seconds if there was one, else None.
        """
        if self._stale_since is None:
            return None
        gap = t - self._stale_since
        self._stale_since = None
        if not teleop_state.active:
            return gap
        ## Force the re-anchor edge on every arm, whatever the gap length.
        for arm_state in teleop_state.arms.values():
            arm_state.active = False
        if gap >= self.disarm_s:
            stop_teleop_session(teleop_state)
            self._rearm_block = True
            self._log(f"\n[link] uplink gap {gap:.1f}s -- teleop DISARMED. "
                      f"Release the drive button, then press again to re-enable.")
        else:
            self._log(f"[link] uplink gap {gap * 1000:.0f} ms -- arms re-anchored")
        return gap

    def apply(self, t, headset_data, button_pressed, arm_active):
        """Veto motion for this tick. Returns (button_pressed, arm_active)."""
        if self._rearm_block:
            if not button_pressed:
                self._rearm_block = False
                self._log("[link] drive button released -- ready to re-enable")
            else:
                if t - self._last_rearm_log > 2.0:
                    self._last_rearm_log = t
                    self._log("[link] still latched after the uplink gap -- "
                              "release the drive button first")
                return False, {arm: False for arm in arm_active}

        return button_pressed, arm_active


## Cached per robot model: CoupledStudyIK JIT-compiles on construction (a few
## seconds), so building one per call inside a control loop would be a
## multi-second stall every tick.  Keyed by id(robot) because that is what the
## compiled solver is specialised to.
_STUDY_IK_CACHE = {}
_STUDY_IK_NOTICE = [False]


def solve_single_arm_ik(
    robot,
    target_link_name,
    target_position,
    target_wxyz,
    prev_q,
    dt,
    joint_velocity_limits=None,
    pos_weight=None,
    ori_weight=None,
    dq_weight=None,
):
    """One arm's target, solved by the DEPLOYED study solver.

    Was: a bare pyroki_snippets pose solve at pos 40 / ori 0.25 / dq 0.18, with
    no collision term of any kind.  That configuration predates the ik_study
    and disagrees with what data_collection.py actually runs, so a script that
    reached for it got measurably different motion from the one the study
    validated -- and no self-collision avoidance at all, which on a three-arm
    rig is not a tuning difference.

    Now: `study_ik.build_study_ik`, i.e. pos 50 / ori 10 (10/25 for the camera
    arm), smoothing 0.05, centering 0.5, 180-sphere self-collision at margin
    20 mm / weight 100, iteration cap 20.  No manipulability term (the study
    found none needed) and no post-solve smoothing or reseeding (the winner
    was validated without them, and a saturating post-filter creates the
    acceleration discontinuities it is meant to remove).

    COORDINATES: prev_q and the return value are DRIVER joints, matching
    CoupledStudyIK.  For left/right-only callers that is identical to the URDF
    frame; only the middle waist differs, and a middle-arm caller that has a
    nonzero Homing_Offset should build the solver itself and pass
    waist_driver_shift.

    THE OTHER TWO ARMS still take part -- they hold whatever `prev_q` says they
    are at, and the collision cost sees them.  Seed prev_q with the real
    measured joints of every arm, not just the one being driven, or the solver
    avoids a phantom and ignores the arm that is there.

    `joint_velocity_limits`, `pos_weight`, `ori_weight` and `dq_weight` are
    accepted and IGNORED: the study's weights are the whole point of routing
    through it, and silently honouring a caller's pos 40 would reintroduce
    exactly the divergence this replaced.  They stay in the signature so
    existing call sites keep working; a warning names any that were passed.

    Prefer `study_ik.build_study_ik(robot)` directly in new code -- holding the
    solver makes its one-off compile visible where it belongs (at startup)
    rather than inside the first control tick."""
    try:
        from .study_ik import build_study_ik
    except ImportError:
        from study_ik import build_study_ik

    if not _STUDY_IK_NOTICE[0]:
        _STUDY_IK_NOTICE[0] = True
        passed = [n for n, v in (("pos_weight", pos_weight),
                                 ("ori_weight", ori_weight),
                                 ("dq_weight", dq_weight),
                                 ("joint_velocity_limits", joint_velocity_limits))
                  if v is not None]
        print("[ik] solve_single_arm_ik now routes through the deployed study "
              "solver (collision included).")
        if passed:
            print(f"[ik]   ignoring caller weights: {', '.join(passed)} -- see "
                  "study_ik.py for the deployed values and their env overrides")

    key = id(robot)
    ik = _STUDY_IK_CACHE.get(key)
    if ik is None:
        print("[ik] compiling the study solver (a few seconds; build it at "
              "startup with study_ik.build_study_ik to avoid stalling a "
              "control loop)...")
        ik = build_study_ik(robot, control_dt=float(dt))
        _STUDY_IK_CACHE[key] = ik

    ## Which arm the target belongs to is decided by the EE LINK, not by an
    ## extra argument, so a caller cannot name one arm and pass another's link.
    try:
        from .arm_config import ARM_CONFIG as _AC
    except ImportError:
        from arm_config import ARM_CONFIG as _AC
    arm = next((a for a in ("left", "right", "middle")
                if _AC[a]["ee_link"] == target_link_name), None)
    if arm is None:
        raise ValueError(
            f"'{target_link_name}' is not any arm's ee_link "
            f"({[_AC[a]['ee_link'] for a in ('left', 'right', 'middle')]}). "
            f"The study solver tracks the configured end effectors; to track "
            f"a different link, change arm_config.ARM_CONFIG.")

    return ik.solve(np.asarray(prev_q, dtype=float),
                    {arm: (np.asarray(target_position, dtype=float),
                           np.asarray(target_wxyz, dtype=float))})
