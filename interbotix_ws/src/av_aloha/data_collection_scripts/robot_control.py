"""
Robot creation, configuration, state queries, and motion helpers.
"""
import warnings

# Ignore the ROS syntax warning where python2-era docstrings ("\p", "\*") that 
# ros noetic ships is flagged by python 3.12 on every run. Harmless.
warnings.filterwarnings("ignore", message=r"invalid escape sequence",
                        category=SyntaxWarning)

import contextlib
import io
import logging
import math
import os
import sys
import threading
import time

import numpy as np
from interbotix_xs_modules.arm import InterbotixManipulatorXS

from arm_config import ARM_CONFIG, DEFAULT_RESET_POSE, POSES, URDF_PATH
from robot_kinematics import build_robot_model, compute_fk_and_ee  # noqa: F401
from data_col_config import ARM_MODES
from gripper import configure_gripper, command_gripper


def quiet_solver_logs():
    """Drop pyroki/jaxls INFO chatter; keep warnings."""
    if os.environ.get("GIAVA_QUIET_SOLVER", "1").strip() in ("0", "false", "no"):
        return False
    try:
        from loguru import logger as _loguru
    except ImportError:
        return False

    ## remove() with no argument drops loguru's default stderr sink, which is
    ## the one printing in colour.
    _loguru.remove()
    _loguru.add(lambda m: sys.stderr.write(f"[solver] {m.record['message']}\n"),
                level="WARNING")
    return True


## MOVE SPEED.  `cruise_speed` is a peak JOINT speed in rad/s and 1.2 was a
## hand-picked constant, chosen to be safe with no knowledge of the hardware.
## The real ceiling is the servos' own Profile_Velocity cap, which
## apply_profile_limits() already READS off the live motors -- so the honest
## "as fast as possible" is that measured cap with a margin, not a bigger
## guess.  Commanding faster than the cap does not go faster: the servo paces
## the move and the settle loop simply waits longer, which is the failure mode
## that looks like "the stream finished but the arm is still travelling".
##
##   default            0.8 x the measured cap, once it has been measured
##   before measurement MOVE_SPEED_FALLBACK (the historical 1.2)
##   GIAVA_MOVE_SPEED   an explicit rad/s that overrides both
##
## Callers may still pass cruise_speed= explicitly; nothing here overrides an
## argument that was actually given.
MOVE_SPEED_FALLBACK = 1.2
MOVE_SPEED_FRAC = float(os.environ.get("GIAVA_MOVE_SPEED_FRAC", "0.8"))
_MOVE_SPEED_ENV = os.environ.get("GIAVA_MOVE_SPEED", "").strip()
MOVE_SPEED_OVERRIDE = float(_MOVE_SPEED_ENV) if _MOVE_SPEED_ENV else None
_SERVO_V_CAP = None


def set_move_speed_cap(v_cap):
    """Record the servos' measured velocity cap for move planning."""
    global _SERVO_V_CAP
    _SERVO_V_CAP = float(v_cap) if v_cap else None


def resolve_cruise_speed(explicit=None):
    """rad/s to plan a move at, and where the number came from."""
    if explicit is not None:
        return float(explicit), "caller"
    if MOVE_SPEED_OVERRIDE is not None:
        return MOVE_SPEED_OVERRIDE, "GIAVA_MOVE_SPEED"
    if _SERVO_V_CAP:
        return _SERVO_V_CAP * MOVE_SPEED_FRAC, f"{MOVE_SPEED_FRAC:.0%} of the measured servo cap"
    return MOVE_SPEED_FALLBACK, "default (servo cap not measured yet)"


REPLAY_MAX_JOINT_STEP_LR = np.array([0.06, 0.06, 0.08, 0.12, 0.12, 0.14], dtype=float)
REPLAY_MAX_JOINT_STEP_M = np.array([0.06, 0.06, 0.08, 0.12, 0.12, 0.14, 0.16], dtype=float)

## THE SDK BANNER.  InterbotixRobotXSCore and InterbotixArmXSInterface print
## six unconditional lines per arm from their constructors (core.py:60,
## arm.py:61 in interbotix_xs_modules) -- robot name, model, group name, moving
## time, accel time, drive mode -- none of which this code uses: moving_time
## and accel_time are passed and then never applied (see move_arms_together's
## note on Profile_Velocity), and "Drive Mode: Time-Based-Profile" is the
## SDK's assumption, not what these servos are running.
##
## Two rospy.logerr lines come with them and are EXPECTED on this rig, not
## faults:
##
##   "Please set the group's 'profile type' to 'time'"  -- the arms run a
##       VELOCITY-based profile on purpose; servo_health.check_profile and
##       apply_profile_limits are both built around that choice.
##   "Please set the gripper's 'operating mode' to 'pwm' or 'current'" -- the
##       gripper is put into current_based_position by configure_gripper,
##       which runs immediately AFTER this constructor.
##
## Eighteen lines of that at every start buries the messages that do matter
## (the [SAFETY] and [profile] lines right below it).  Captured, replaced with
## one line per arm, and REPRINTED IN FULL if construction raises -- the noise
## is only noise when it works.  GIAVA_QUIET_SDK=0 restores it.
QUIET_SDK = os.environ.get("GIAVA_QUIET_SDK", "1").strip() not in ("0", "false", "no")

_SDK_EXPECTED_LOGS = (
    "profile type",
    "operating mode",
)


class _DropExpectedSdkLogs(logging.Filter):
    def filter(self, record):
        return not any(t in record.getMessage() for t in _SDK_EXPECTED_LOGS)


@contextlib.contextmanager
def _quiet_sdk():
    """Swallow the constructor banner; hand it back if something goes wrong."""
    if not QUIET_SDK:
        yield None
        return
    buf = io.StringIO()
    ## rospy.logerr goes through the 'rosout' logger, whose handlers hold
    ## their own reference to stderr -- redirecting sys.stderr would not catch
    ## it, so the two expected messages are filtered at the logger instead.
    log_filter = _DropExpectedSdkLogs()
    loggers = [logging.getLogger("rosout"), logging.getLogger()]
    for lg in loggers:
        lg.addFilter(log_filter)
    try:
        with contextlib.redirect_stdout(buf):
            yield buf
    except Exception:
        ## It failed: the banner may be the only clue about how far it got.
        sys.stdout.write(buf.getvalue())
        raise
    finally:
        for lg in loggers:
            lg.removeFilter(log_filter)


# Robot creation
def create_robot(arm_name, moving_time=0.14, accel_time=0.04):
    cfg = ARM_CONFIG[arm_name]

    with _quiet_sdk():
        return _create_robot_inner(arm_name, cfg, moving_time, accel_time)


def _create_robot_inner(arm_name, cfg, moving_time, accel_time):
    return InterbotixManipulatorXS(
        robot_model=cfg["robot_model"],
        group_name="arm",
        gripper_name="gripper" if cfg["has_gripper"] else None,
        robot_name=cfg["robot_name"],
        moving_time=moving_time,
        accel_time=accel_time,
        init_node=False,
    )
    

def create_and_configure_robot(arm_name, moving_time=0.14, accel_time=0.04):
    bot = create_robot(arm_name, moving_time, accel_time)

    if ARM_CONFIG[arm_name]["has_gripper"]:
        configure_gripper(bot, ARM_CONFIG[arm_name]["robot_name"])
    return bot


def create_and_configure_robots(arm_names=("left", "right", "middle")):
    return {arm_name: create_and_configure_robot(arm_name) for arm_name in arm_names}


# State query helper: get current joint positions as a NumPy array.
def apply_profile_limits(robots, cfg, arm_names=None, verbose=True):
    """Reconcile the control-loop clamp with the profile the servos RUN.

    Reads Drive_Mode / Profile_Velocity / Profile_Acceleration off the live
    motors and reports what they mean, then -- only when cfg.profile_mode is
    "velocity" or "auto" -- sets cfg.servo_v_cap so driver_max_step becomes
    "one tick of travel at the firmware speed cap" instead of "one moving_time
    at the URDF velocity limit".

    WHY THIS IS A READ, NOT A WRITE
    ================================
    Switching the servos between profiles means writing Drive_Mode, which is
    EEPROM: the xs_sdk torques every motor OFF to do it and back on afterwards
    (xs_sdk_obj.cpp robot_set_joint_operating_mode).  Under gravity that is a
    visible sag at every program start -- the same reason
    create_and_configure_robot stopped calling robot_set_operating_modes.  So
    this never writes.  It asks the hardware what it is doing and makes the
    SOFTWARE agree, which removes the entire class of bug where the two
    silently disagree, at zero mechanical cost.

    Returns {arm: (mode, pv, pa)}.
    """
    from servo_health import (profile_limits, read_profile_modes, summarize_profiles)

    arm_names = list(arm_names or robots.keys())
    found = {}
    caps = []
    for arm in arm_names:
        try:
            per = read_profile_modes(robots[arm], arm)
            mode, pv, pa = summarize_profiles(per)
        except Exception as exc:
            if verbose:
                print(f"[profile] {arm}: could not read registers ({exc})")
            continue
        if mode is None:
            if verbose:
                print(f"[profile] {arm}: no motor answered -- profile unknown")
            continue
        found[arm] = (mode, pv, pa)
        ## Name a group whose joints run DIFFERENT profiles.  Real, not
        ## hypothetical: puppet_modes_middle.yaml puts `waist` and `camera_yaw`
        ## on a time-based profile while the `arm` group stays velocity-based.
        ## summarize_profiles() reports the binding (velocity) case, so the
        ## yaml and the clamp can be describing different things.
        if verbose and len({m for m, _, _ in per.values()}) > 1:
            _by = {}
            for _j, (_m, _, _) in sorted(per.items()):
                _by.setdefault(_m, []).append(_j)
            print(f"[profile] {arm}: joints DISAGREE -- "
                  + "; ".join(f"{_m}: {', '.join(_js)}"
                              for _m, _js in sorted(_by.items())))
            print(f"[profile] {arm}: clamping as '{mode}' -- the "
                  f"velocity-based joints are the binding constraint "
                  f"(a time-based joint can always go gentler, a "
                  f"velocity-based one cannot go faster)")
        lim = profile_limits(mode, pv, pa, cfg.moving_time)
        if verbose:
            if mode == "time":
                print(f"[profile] {arm}: TIME-based, move {pv} ms, accel "
                      f"{pa} ms -- peak speed is UNBOUNDED (it scales with "
                      f"how far the goal jumped)")
            else:
                v, a = lim["v_max"], lim["a_max"]
                print(f"[profile] {arm}: VELOCITY-based, cap "
                      f"{v:.2f} rad/s, accel {a:.1f} rad/s^2"
                      if v and a else
                      f"[profile] {arm}: VELOCITY-based, PV={pv} PA={pa} "
                      f"(0 = NO PROFILE, i.e. maximum speed)")
        if lim.get("v_max"):
            caps.append(lim["v_max"])

    modes = {m for m, _, _ in found.values()}
    if len(modes) > 1 and verbose:
        print(f"[profile] WARNING: the arms disagree ({sorted(modes)}). One "
              f"clamp cannot be correct for both.")

    want = cfg.profile_mode
    if want == "auto":
        want = "velocity" if modes == {"velocity"} else "legacy"
        if verbose:
            print(f"[profile] auto -> '{want}'")
        cfg.profile_mode = want

    if want == "velocity":
        if caps:
            cfg.servo_v_cap = float(min(caps))
            ## Same number the control-loop clamp uses now also paces
            ## move_arms_together -- see resolve_cruise_speed.
            set_move_speed_cap(cfg.servo_v_cap)
        if verbose:
            print(f"[profile] clamp set from the SERVO cap: "
                  f"driver_max_step = {cfg.driver_max_step:.4f} rad/tick "
                  f"({cfg.driver_max_step / cfg.control_dt:.2f} rad/s "
                  f"equivalent, was "
                  f"{cfg.driver_clamp_safety * cfg.driver_velocity_limit * cfg.moving_time:.4f})")
    elif verbose:
        print(f"[profile] clamp left on LEGACY arithmetic "
              f"(driver_max_step = {cfg.driver_max_step:.4f} rad/tick). "
              f"GIAVA_PROFILE_MODE=velocity to match the servo instead.")
    return found


def get_joint_positions(bot):
    return np.array(bot.arm.core.joint_states.position[:len(bot.arm.group_info.joint_names)], dtype=float)

def sync_robot_state(robots,
    robot,
    arm_data,
    arm_names,
    cmd_kin,
    cmd_state,
    to_urdf=None,
):
    """
    Synchronize software command state with the robots' measured joint states.

    Call this after any motion that occurs outside the normal teleoperation
    command loop, such as resetting the arms or moving to a named pose.
    """
    measured_q_full = cmd_kin.q_cmd.copy()

    for arm in arm_names:
        joint_idx = arm_data[arm]["joint_indices"]
        num_joints = ARM_CONFIG[arm]["num_joints"]

        measured_q_arm = np.asarray(
            robots[arm].dxl.joint_states.position[:num_joints],
            dtype=float,
        ).copy()

        if measured_q_arm.shape[0] != len(joint_idx):
            raise RuntimeError(
                f"{arm}: measured {measured_q_arm.shape[0]} joints, "
                f"but robot model expects {len(joint_idx)}."
            )

        measured_q_full[joint_idx] = measured_q_arm
        cmd_state.last_cmds[arm] = measured_q_arm.copy()

        # Reset the DRIVER's own command reference too.
        #
        # interbotix validates every command against `arm.joint_commands` --
        # its last *accepted* command -- not against the measured position:
        #     speed = |goal - joint_commands| / moving_time  >  velocity_limit
        # When a command is rejected, `joint_commands` is never updated, so it
        # stays frozen wherever it was.  Resyncing only our own bookkeeping
        # therefore does nothing: the driver keeps measuring every new goal
        # against a stale reference that may be radians away, rejects it, and
        # the arm is stuck permanently.
        #
        # interbotix's own `capture_joint_positions()` does this, but it is dead
        # code in this workspace: `import modern_robotics as mr` is commented
        # out at the top of arm.py (as is `self.robot_des`), so its final
        # `mr.FKinSpace(...)` line raises NameError.  We therefore set
        # `joint_commands` directly -- that FK line only maintains `T_sb`,
        # which nothing in this pipeline reads.
        arm_iface = robots[arm].arm
        try:
            core = arm_iface.core
            arm_iface.joint_commands = [
                core.joint_states.position[core.js_index_map[name]]
                for name in arm_iface.group_info.joint_names
            ]
        except Exception as exc:  # never let bookkeeping kill the session
            print(f"[{arm}] could not reset driver joint_commands: {exc}")

        # Permit the next command immediately.
        cmd_state.last_arm_cmd_time[arm] = 0.0

    cmd_kin.q_cmd[:] = measured_q_full

    # Recompute end-effector poses from the synchronized joint configuration.
    #
    # q_cmd is in DRIVER coordinates, and the middle arm's waist is offset by pi
    # between the driver and URDF frames.  Forward kinematics must therefore run
    # on the converted vector -- the main control loop already does this via
    # coupled_ik.driver_to_urdf().  Without it, T_cmd["middle"] lands pi away
    # from where the camera arm actually is, so the first teleop command after a
    # reset or a stale-state resync is computed against a bogus start pose.
    q_for_fk = cmd_kin.q_cmd if to_urdf is None else to_urdf(cmd_kin.q_cmd)
    _, measured_ee = compute_fk_and_ee(
        robot,
        q_for_fk,
        arm_data,
    )

    for arm in arm_names:
        cmd_kin.T_cmd[arm] = measured_ee[arm].copy()

def command_state_is_stale(
    robots,
    arm_data,
    arm_names,
    cmd_state,
    tolerance=0.08,
):
    """
    Return True when measured joints differ substantially from the command
    state stored by the teleoperation loop.
    """
    for arm in arm_names:
        num_joints = ARM_CONFIG[arm]["num_joints"]

        measured_q = np.asarray(
            robots[arm].dxl.joint_states.position[:num_joints],
            dtype=float,
        )

        expected_q = np.asarray(
            cmd_state.last_cmds[arm],
            dtype=float,
        )

        err = np.abs(measured_q - expected_q)
        k = int(np.argmax(err))
        max_error = float(err[k])

        if max_error > tolerance:
            # 
            names = ARM_CONFIG[arm]["joint_names"]
            joint = names[k] if k < len(names) else f"joint {k}"
            print(
                f"\n[{arm}] Command state is stale: {joint} commanded "
                f"{expected_q[k]:+.3f} rad, measured {measured_q[k]:+.3f} rad "
                f"(difference {max_error:.3f} rad; "
                f"largest of {len(err)} joints)."
            )
            # Will print a warning if arm configuration is far from expected configuration
            print(f"        (expected if {arm} was just moved externally -- "
                  f"move_arms.py, hand-guiding, or a reboot; re-anchoring "
                  f"from measured joints, no snap-back. If NOBODY moved it, "
                  f"suspect torque loss or an encoder wrap.)")
            return True

    return False

def interpolate_to_pose(bot, arm, pose, moving_time=0.2, accel_time=0.1,
                        blocking=True, cruise_speed=None, accel_frac=0.25):
    """ 
    Previous implementation split the motion in to waypoints and sent each one as a complete move
    This means the motors will accelerate and decelerate to get to each waypoint
    Now it establishes a single start and delta and hands it to stream profile which sets one
    acceleration at the start, one deceleration at the end, and constant velocity in between

    I also changed the arms to use the velocity based profile where maximum velocity is the limit
    as oposed to having duration as the limit. Moreover, blocking can be set to False runs the 
    stream on a daemon thread and returns at once.
    """
    del moving_time, accel_time                 # see the docstring
    prepared = _prepare_move_target(bot, arm, pose)
    if prepared is None:
        return
    start_q, target_q = prepared
    delta = target_q - start_q
    if float(np.max(np.abs(delta))) < 1e-4:
        return
    robots = {arm: bot}
    plans = {arm: (start_q, delta)}
    kw = dict(cruise_speed=cruise_speed, accel_frac=accel_frac, rate_hz=50.0,
              settle_s=0.5, settle_tol=0.05, verbose=True)
    if blocking:
        return _stream_profile(robots, plans, **kw)
    ## Daemon thread dies WITH the process, wherever the stream happens to
    ## be.  Any caller whose process may exit before the move ends must join
    ## the returned thread, or the arm is left mid-trajectory. It now uses
    ## move_arms_together instead.
    t = threading.Thread(target=_stream_profile, args=(robots, plans),
                         kwargs=kw, daemon=True)
    t.start()
    return t


def move_to_named_pose(bot, arm_name, pose_name, moving_time=0.2, accel_time=0.1, blocking=True):
    interpolate_to_pose(bot, arm_name, get_pose(arm_name, pose_name), moving_time, accel_time, blocking)

def reset_arm(bot, arm_name, pose_name=DEFAULT_RESET_POSE):
    move_to_named_pose(bot, arm_name, pose_name)

def reset_arms(robots, pose_name=DEFAULT_RESET_POSE, together=True, **kwargs):
    """Move every arm to rest pose.  Simultaneous and continuous by default.

    `together=False` restores the old behavior in cases where an arm must 
    be watched alone."""
    if together:
        return move_to_named_poses(robots, pose_name, **kwargs)
    for arm_name, bot in robots.items():
        reset_arm(bot, arm_name, pose_name)

def _prepare_move_target(bot, arm_name, pose):
    """(start_q, target_q) for one arm, or None if this arm must not move.

    Runs the same guards `interpolate_to_pose` does -- dead bus, wrapped
    encoder, middle-waist 2pi frame -- but returns the pair instead of
    driving, so a caller can move several arms off one clock."""
    n = len(pose)
    current_q = np.asarray(bot.dxl.joint_states.position[:n], dtype=float)
    target_q = np.asarray(pose, dtype=float)

    if len(current_q) >= 3 and float(np.ptp(current_q)) < 1e-6:
        print(f"[SAFETY] {arm_name}: all joints read an identical "
              f"{current_q[0]:+.4f} rad -- the motors are not answering on "
              f"the bus. SKIPPING this arm.")
        return None
    try:
        lower = np.asarray(bot.arm.group_info.joint_lower_limits, dtype=float)
        upper = np.asarray(bot.arm.group_info.joint_upper_limits, dtype=float)
        excess = np.maximum(lower - current_q, current_q - upper)
        if (excess > 0.5).any():
            names = list(bot.arm.group_info.joint_names)
            for i in np.nonzero(excess > 0.5)[0]:
                print(f"[SAFETY] {arm_name}: joint '{names[i]}' reads "
                      f"{current_q[i]:+.3f}, {excess[i]:.3f} rad outside "
                      f"[{lower[i]:+.3f}, {upper[i]:+.3f}] -- encoder likely "
                      f"wrapped. SKIPPING this arm.")
            return None
    except AttributeError:
        pass

    if arm_name == "middle":
        k = np.round((current_q[0] - target_q[0]) / (2 * np.pi))
        if k != 0:
            target_q = target_q.copy()
            target_q[0] = target_q[0] + 2 * np.pi * k
            print(f"[frame] middle waist target -> {target_q[0]:+.3f} "
                  f"(nearest 2pi-equivalent to current {current_q[0]:+.3f})")
    return current_q, target_q


def move_arms_together(robots, targets, cruise_speed=None, accel_frac=0.25,
                       rate_hz=50.0, settle_s=0.5, settle_tol=0.05,
                       verbose=True):
    """Drive every arm to its target simultaneously, on one continuous ramp.

    Every arm is commanded each tick (`blocking=False`) along one shared path
    parameter s(t) with a trapezoidal velocity profile, so they start, cruise
    and arrive together and no joint finishes early.  Duration comes from the
    largest joint motion anywhere in the robot, making `cruise_speed` [rad/s] a
    real peak-joint-speed bound rather than a fixed duration.

    Passes moving_time=None deliberately: the group runs a VELOCITY-based
    profile where Profile_Velocity is a speed CAP, not a duration, so writing
    one per call would silently re-cap the servos (see
    servo_health.check_profile).  Keep `cruise_speed` below that cap.

    Returns the set of arms that actually moved.
    """
    plans = {}
    for arm_name, target in targets.items():
        bot = robots.get(arm_name)
        if bot is None:
            continue
        prepared = _prepare_move_target(bot, arm_name, target)
        if prepared is None:
            continue
        start_q, target_q = prepared
        delta = target_q - start_q
        if float(np.max(np.abs(delta))) < 1e-4:
            continue          # already there
        plans[arm_name] = (start_q, delta)

    if not plans:
        if verbose:
            print("[move] nothing to move.")
        return set()

    return _stream_profile(robots, plans, cruise_speed=cruise_speed,
                           accel_frac=accel_frac, rate_hz=rate_hz,
                           settle_s=settle_s, settle_tol=settle_tol,
                           verbose=verbose)


def _stream_profile(robots, plans, cruise_speed, accel_frac, rate_hz,
                    settle_s, settle_tol, verbose):
    """Drive {arm: (start_q, delta)} along one shared trapezoid.

    The single motion primitive: `move_arms_together` and `interpolate_to_pose`
    both funnel here, so one-arm and three-arm moves share one code path.
    """
    if not plans:
        return set()
    cruise_speed, _speed_src = resolve_cruise_speed(cruise_speed)
    max_delta = max(float(np.max(np.abs(d))) for _, d in plans.values())
    accel_frac = float(np.clip(accel_frac, 0.0, 1.0))
    ## Duration such that the peak joint speed is EXACTLY cruise_speed.
    ## A trapezoid covers its distance in an equivalent (T - t_a) at cruise
    ## speed -- t_a, not accel_frac*T, because the two ramps each contribute
    ## half their span.
    T = max_delta / max(cruise_speed * (1.0 - 0.5 * accel_frac), 1e-6)
    t_a = 0.5 * accel_frac * T          # one ramp
    v_c = 1.0 / max(T - t_a, 1e-6)      # cruise ds/dt, so that s(T) == 1

    if verbose:
        print(f"[move] {', '.join(sorted(plans))} together: "
              f"max delta {max_delta:.3f} rad, {T:.2f} s "
              f"({cruise_speed:.2f} rad/s cruise from {_speed_src}, "
              f"{accel_frac * 100:.0f}% ramping)")

    def s_of(t):
        """Trapezoidal-velocity path parameter, s(0)=0, s(T)=1."""
        if t <= 0.0:
            return 0.0
        if t >= T:
            return 1.0
        if t_a <= 0.0:
            return t / T
        if t < t_a:
            return 0.5 * (v_c / t_a) * t * t
        if t <= T - t_a:
            return 0.5 * v_c * t_a + v_c * (t - t_a)
        r = T - t
        return 1.0 - 0.5 * (v_c / t_a) * r * r

    dt = 1.0 / float(rate_hz)
    t0 = time.monotonic()
    refused = {a: 0 for a in plans}
    next_tick = t0
    while True:
        t = time.monotonic() - t0
        s = s_of(t)
        for arm_name, (start_q, delta) in plans.items():
            q = start_q + s * delta
            ok = robots[arm_name].arm.set_joint_positions(
                q.tolist(), moving_time=None, accel_time=None, blocking=False)
            if ok is False:
                refused[arm_name] += 1
        if t >= T:
            break
        next_tick += dt
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)

    ## Arrival is CHECKED, not assumed, and the check WAITS WHILE PROGRESS
    ## IS BEING MADE rather than against a fixed deadline.
    ##
    ## A fixed `settle_s` is wrong for the reason hardware just showed: the
    ## servo enforces its own velocity and acceleration caps, so for a large
    ## move it lags the setpoint stream and is still travelling when the
    ## stream ends.  The goal register already holds the target, so it WILL
    ## arrive -- just later than any constant anyone picks.  Timing out then
    ## returned mid-flight, and the caller moved on to the next thing while
    ## the arm was still going: "move_arms does not reach the pose when it is
    ## far".
    ##
    ## So: keep waiting as long as the worst joint is measurably closing on
    ## the target, and give up only when it STALLS -- no progress beyond
    ## `settle_tol / 4` for `stall_s`.  That is unbounded in the good case
    ## (a long move takes as long as it takes) and still terminates in the
    ## bad one (blocked joint, dead bus, refused commands), which a plain
    ## "wait forever" would not.
    def _worst_error():
        w = 0.0
        for arm_name, (start_q, delta) in plans.items():
            n = len(start_q)
            meas = np.asarray(
                robots[arm_name].dxl.joint_states.position[:n], dtype=float)
            w = max(w, float(np.max(np.abs(meas - (start_q + delta)))))
        return w

    stall_s = max(settle_s, 0.5)
    progress_eps = max(settle_tol / 4.0, 1e-3)
    worst = _worst_error()
    best = worst
    last_progress = time.monotonic()
    t_settle0 = last_progress
    while worst > settle_tol:
        time.sleep(0.02)
        worst = _worst_error()
        if worst < best - progress_eps:
            best = worst
            last_progress = time.monotonic()
        elif (time.monotonic() - last_progress) > stall_s:
            if verbose:
                print(f"[move] STALLED {worst:.3f} rad from target after "
                      f"{time.monotonic() - t_settle0:.1f} s with no progress "
                      f"for {stall_s:.1f} s. The arm is not moving: check for "
                      f"a mechanical block, a latched servo fault ('e'), or "
                      f"refused commands below.")
            break
    else:
        if verbose and (time.monotonic() - t_settle0) > 0.1:
            print(f"[move] settled in {time.monotonic() - t_settle0:.1f} s "
                  f"after the ramp (the servo profile was the limit, not the "
                  f"stream)")

    for arm_name, count in refused.items():
        if count:
            print(f"[move] {arm_name}: the driver REFUSED {count} of the "
                  f"commands (check_joint_limits). Lower cruise_speed.")
    return set(plans)


def move_to_named_poses(robots, pose_name=DEFAULT_RESET_POSE, **kwargs):
    """Every arm to the same named pose, simultaneously and smoothly."""
    targets = {a: get_pose(a, pose_name) for a in robots}
    return move_arms_together(robots, targets, **kwargs)


def stop_arm(bot):
    """Halt an arm in place. Torque stays on, so nothing drops.

    moving_time is sized from the ACTUAL distance rather than fixed at
    0.05 s.  interbotix validates every command (arm.py:check_joint_limits)
    as

        speed = |goal - self.joint_commands| / moving_time
        if speed > joint_velocity_limits: return False

    where joint_commands is the driver's last ACCEPTED command, not the
    measured position -- so mid-motion that difference is the whole
    remaining travel.  A fixed 0.05 s therefore asks for 4 rad/s to cancel
    0.2 rad, gets refused against the 3.14 rad/s limit, and
    set_joint_positions returns False rather than raising.  Nothing checked
    that return value, so the stop silently did nothing in exactly the case
    it was needed: a fast, large motion.

    Sizing moving_time to the distance keeps every halt inside the limit.
    The return value is checked, and the command escalated, so a refusal
    can no longer pass unnoticed."""
    n = len(bot.arm.group_info.joint_names)
    measured = np.asarray(bot.arm.core.joint_states.position[:n], dtype=float)

    ref = measured
    getter = getattr(bot.arm, "get_joint_commands", None)
    if getter is not None:
        try:
            ref = np.asarray(getter(), dtype=float)[:n]
        except Exception:
            pass

    try:
        vel = np.asarray(bot.arm.group_info.joint_velocity_limits,
                         dtype=float)[:n]
        vel = np.where(vel > 1e-6, vel, np.pi)
    except Exception:
        vel = np.full(n, np.pi)

    # 0.7 of the limit leaves room for the driver's rounding to 3 decimals.
    moving_time = max(
        0.05,
        float(np.max(np.abs(measured - ref) / np.maximum(0.7 * vel, 1e-6))))

    for _ in range(4):
        ## Same ordering trap as replay_arm_command: the velocity check reads
        ## self.moving_time, and a moving_time argument only updates that
        ## AFTER a command is accepted.  Halting usually passes anyway
        ## (measured sits near the frozen reference), but mid-flight -- the
        ## one case this function exists for -- the sized timing must be in
        ## place BEFORE the check runs.
        bot.arm.set_trajectory_time(moving_time,
                                    min(0.02, 0.5 * moving_time))
        if bot.arm.set_joint_positions(
                measured.tolist(), moving_time=None,
                accel_time=None, blocking=False):
            return True
        moving_time *= 2.0

    print("[SAFETY] stop_arm: the driver refused every halt command. "
          "Kill the roslaunch or use the physical power switch.")
    return False

def stop_robots(robots):
    for bot in robots.values():
        stop_arm(bot)

# Set gripper position directly.
def set_gripper(bot, position):
    if not hasattr(bot, "gripper"):
        return

    command_gripper(bot, float(position))

# Send a joint command during dataset replay.
# Returns the commanded joint vector so replay.py can compute tracking error.
def replay_arm_command(bot, target_q, moving_time=0.03, accel_time=0.01):
    """Stream one recorded action frame, riding through recorded transients.

    The return value of set_joint_positions was ignored here, which wedged
    whole replays (hardware, 2026-09-01): arm.py validates every command as
    |goal - self.joint_commands| / moving_time against the velocity limit and
    a REFUSED command never updates self.joint_commands.  One recorded
    transient over pi*0.03 = 0.094 rad/tick (a re-anchor jump in the episode:
    wrist_angle stepped 0.101 rad at frame 334 of the 20260901_163329 set) is
    therefore permanent: the reference freezes, the trajectory walks away
    from it, and every remaining frame is refused at 50 Hz while the arm
    stands still -- the "[WARN] Would exceed velocity limits" spam names
    whichever joint has drifted furthest from the FROZEN reference, not the
    joint that jumped.

    Datasets legitimately contain such steps (data_collection records the
    commanded stream, and an operator re-enable re-anchors it), so replay
    must tolerate what recording produced: on a refusal, resend with
    moving_time sized from the actual gap to the driver's reference, exactly
    stop_arm's escalation above.  The stretched profile chases through the
    transient in one tick, the reference re-syncs, and the next frame is
    back on the fast path."""
    target_q = np.asarray(target_q, dtype=float)

    if bot.arm.set_joint_positions(
            target_q.tolist(),
            moving_time=moving_time,
            accel_time=accel_time,
            blocking=False,
    ):
        return target_q

    n = len(bot.arm.group_info.joint_names)
    ref = np.asarray(bot.arm.core.joint_states.position[:n], dtype=float)
    getter = getattr(bot.arm, "get_joint_commands", None)
    if getter is not None:
        try:
            ref = np.asarray(getter(), dtype=float)[:n]
        except Exception:
            pass
    try:
        vel = np.asarray(bot.arm.group_info.joint_velocity_limits,
                         dtype=float)[:n]
        vel = np.where(vel > 1e-6, vel, np.pi)
    except Exception:
        vel = np.full(n, np.pi)

    # 0.7 of the limit leaves room for the driver's rounding to 3 decimals.
    mt = max(moving_time,
             float(np.max(np.abs(target_q[:n] - ref) / np.maximum(0.7 * vel,
                                                                  1e-6))))
    for _ in range(4):
        bot.arm.set_trajectory_time(mt, min(accel_time, 0.5 * mt))
        if bot.arm.set_joint_positions(
                target_q.tolist(),
                moving_time=None,
                accel_time=None,
                blocking=False,
        ):
            print(f"[replay] recorded transient exceeded the driver rate "
                  f"check; stretched one command to {mt * 1000:.0f} ms and "
                  f"resynchronized.")
            return target_q
        mt *= 2.0

    print("[replay] the driver refused a command 4 times even with a "
          "stretched profile -- this frame is lost; check for a [SAFETY] "
          "or position-limit message above.")
    return target_q

# -----------------------------------------------------------------------------
# Torque
# -----------------------------------------------------------------------------

def torque_on(bot):
    bot.dxl.robot_torque_enable("group", "arm", True)

    if hasattr(bot, "gripper"):
        bot.dxl.robot_torque_enable("single", "gripper", True)


def torque_off(bot):
    bot.dxl.robot_torque_enable("group", "arm", False)

    if hasattr(bot, "gripper"):
        bot.dxl.robot_torque_enable("single", "gripper", False)

def safe_move_arm_joints(bot, target_q, total_time=3.0, step_time=0.25, accel_ratio=0.35):
    n = len(bot.arm.group_info.joint_names)
    current_q = np.array(bot.dxl.joint_states.position[:n], dtype=float)
    target_q = np.array(target_q, dtype=float)
    max_step = np.array([0.05, 0.05, 0.06, 0.10, 0.10, 0.12], dtype=float)

    delta = target_q - current_q
    n_steps = int(np.ceil(np.max(np.abs(delta) / max_step)))
    n_steps = max(n_steps, int(np.ceil(total_time / step_time)), 1)

    waypoints = np.linspace(current_q, target_q, n_steps + 1)[1:]
    move_t = step_time
    accel_t = min(accel_ratio * move_t, 0.5 * move_t)

    for q_cmd in waypoints:
        bot.arm.set_joint_positions(q_cmd.tolist(), moving_time=move_t, accel_time=accel_t, blocking=True)

# Middle-waist frame shift, in radians, mirroring the servo's Homing_Offset
# register (reported = actual + offset).  The pose tables in arm_config.py
# store LEGACY driver values recorded with offset 0; when a Homing_Offset has
# been written (set_waist_homing_offset.py), every commanded waist value must
# shift by the same amount.  data_collection reads the register at startup and
# calls set_middle_waist_shift(); 0.0 keeps historical behavior exactly.
## THE MIDDLE WAIST'S DRIVER FRAME.
##
## Two different things can shift it, and they are NOT interchangeable:
##
##   Homing_Offset      a REGISTER offset (reported = actual + offset).  Inert
##                      while the joint runs in ext_position mode, and capped
##                      at +/-1024 ticks (+/-90 deg) in position mode.
##   physical re-clock  the motor bolted to the arm at a different angle.
##                      Unlimited, works in any operating mode, and invisible
##                      to every register -- nothing on the bus can report it,
##                      so it has to be configured here.
##
## Both land in the same place: `theta` such that
##     driver_new = driver_old + theta
## which `get_pose` adds to every middle pose and `study_ik` / `kinematics`
## turn into  waist_urdf_offset = pi - theta.
##
## 2026-09-11: the motor was re-clocked by roughly a half turn to move the
## encoder seam out of the working range (rest went from driver ~+3.13 rad,
## six ticks off the 0/4096 wrap, to ~0 rad).  MEASURE the exact value and put
## it here -- do not infer it by differencing two approximate rest poses, and
## do not guess.  giava.urdf's middle chain was once "corrected" from an
## assumed relationship that looked right geometrically and sent the real arm
## to a completely wrong pose (see _2026_08_28_attempted_revert in
## middle_joint_offsets.json).  Verify with  python verify_middle_urdf.py.
##
## MEASURED 2026-09-11: the motor was remounted rotated a HALF TURN, so this
## is exactly -pi, not an approximation to it.  A half turn is an exact
## mechanical quantity -- the horn went back on 180 deg round -- so the value
## is known a priori and does not inherit any pose-measurement error.  (The
## -3.1309 a rest-pose subtraction suggests is NOT independent evidence: the
## waist had been jogged to ~0 while the other joints held their old goals, so
## that difference is just 0.0015 - 3.1324 and means nothing about theta.)
##
## Conveniently exact: waist_urdf_offset = pi - (-pi) = 2*pi, and
## driver_to_urdf wraps into [-pi, pi], so the driver frame and the URDF frame
## now COINCIDE.  urdf == driver for middle_base, for the first time.
##
## 0.0 would be the pre-re-clock frame, i.e. the pose tables in arm_config.py
## as written.  Override without editing:  GIAVA_MIDDLE_WAIST_RECLOCK=<radians>
MIDDLE_WAIST_RECLOCK_RAD = -math.pi

## DEFAULTS TO THE PHYSICAL RE-CLOCK, NOT ZERO.
##
## This used to be 0.0 and was only corrected when a caller remembered to call
## set_middle_waist_shift().  data_collection.py and teleop.py did; rollout_policy.py
## and move_arms.py did not -- so every pose they fetched through get_pose() came
## back in the PRE-re-clock frame and parking swung the camera waist 177 degrees
## (commanded -3.1140 at an arm sitting at -0.023).
##
## A physical re-clock is a static property of the hardware, not something a
## session discovers, so it applies unconditionally.  The REGISTER half
## (Homing_Offset) still needs a live read, and resolve_middle_waist_shift()
## overwrites this with reclock+register when a driver is up.
##
## This is the same lesson as the middle_joint_offsets.json fold: a correction
## that every caller must remember to apply is a correction that some caller
## will not apply.
MIDDLE_WAIST_DRIVER_SHIFT = MIDDLE_WAIST_RECLOCK_RAD


def set_middle_waist_shift(shift_rad):
    global MIDDLE_WAIST_DRIVER_SHIFT
    MIDDLE_WAIST_DRIVER_SHIFT = float(shift_rad)
    if abs(MIDDLE_WAIST_DRIVER_SHIFT) > 1e-9:
        ## NOT "Homing_Offset shift", which is what this used to say: since
        ## the 2026-09-11 re-clock the dominant term is physical and the
        ## register is 0, so naming the register sends the next reader to look
        ## at something that will tell them nothing.  resolve_middle_waist_shift
        ## prints the breakdown; this only confirms it was applied.
        print(f"[frame] middle waist driver-frame shift "
              f"{MIDDLE_WAIST_DRIVER_SHIFT:+.3f} rad applied -- pose tables "
              "and URDF offset adjusted to match")


def middle_waist_reclock():
    """The configured PHYSICAL re-clock of the waist motor, in radians."""
    raw = os.environ.get("GIAVA_MIDDLE_WAIST_RECLOCK")
    if raw is None or not str(raw).strip():
        return float(MIDDLE_WAIST_RECLOCK_RAD)
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(
            f"GIAVA_MIDDLE_WAIST_RECLOCK must be a number (radians), got {raw!r}")


def resolve_middle_waist_shift(bot):
    """Total driver-frame shift for the middle waist, and report it.

    Single source of truth for a decision data_collection.py and teleop.py
    used to make with identical copy-pasted blocks.  Combines the physical
    re-clock (which no register can report) with Homing_Offset (which only
    counts when the servo actually applies it), and calls
    set_middle_waist_shift so the pose tables follow.

    THE REGISTER IS IGNORED IN ext_position MODE.  That is a measurement, not
    a guess: the servo does not apply Homing_Offset there, so honouring a
    nonzero value would desynchronize the pose tables from the frame the arm
    is really in.  Refusing to guess is the point -- a silently wrong waist
    frame is how the camera arm ends up pointing somewhere else entirely.
    """
    reclock = middle_waist_reclock()
    register = read_middle_waist_shift(bot)

    mode = None
    try:
        resp = bot.dxl.robot_get_motor_registers("single", "waist", "Operating_Mode")
        mode = int(resp.values[0]) if resp.values else None
    except Exception:
        pass
    ext = (mode == 4)

    if abs(register) > 1e-6 and ext:
        print("=" * 60)
        print(f"[frame] waist Homing_Offset is {register:+.3f} rad but the waist")
        print("        runs in ext_position mode, which ignores it. Treating it")
        print("        as 0. Revert the register so the two agree:")
        print("            python set_waist_homing_offset.py --degrees 0")
        print("=" * 60)
        register = 0.0

    shift = reclock + register
    print(f"[frame] middle waist: re-clock {reclock:+.4f} rad + register "
          f"{register:+.4f} rad = shift {shift:+.4f} rad")
    if abs(reclock) < 1e-9:
        print("        (no physical re-clock configured -- pose tables used as "
              "written. If the motor HAS been re-clocked, set "
              "MIDDLE_WAIST_RECLOCK_RAD or GIAVA_MIDDLE_WAIST_RECLOCK.)")

    ## The seam is what the re-clock exists to avoid, so say where it is now
    ## rather than making the operator work it out from a tick count.
    try:
        pos = bot.arm.core.joint_states.position[
            bot.arm.core.js_index_map["waist"]]
        seam = np.pi - abs((pos + np.pi) % (2 * np.pi) - np.pi)
        print(f"        waist reads {pos:+.4f} rad, {np.degrees(seam):.1f} deg "
              f"from the encoder seam")
    except Exception:
        pass

    set_middle_waist_shift(shift)
    return shift


def read_middle_waist_shift(bot):
    """Read the waist Homing_Offset from the servo, in radians (0.0 if unset)."""
    try:
        resp = bot.dxl.robot_get_motor_registers("single", "waist", "Homing_Offset")
        ticks = int(resp.values[0]) if resp.values else 0
        if ticks >= (1 << 31):
            ticks -= 1 << 32
        return ticks * 2.0 * np.pi / 4096.0
    except Exception as exc:
        print(f"[frame] could not read waist Homing_Offset ({exc}); assuming 0")
        return 0.0


# Return a named pose for an arm; raises ValueError if not defined.
def get_pose(arm_name, pose_name):
    if pose_name not in POSES[arm_name]:
        raise ValueError(
            f"Pose '{pose_name}' not defined for arm '{arm_name}'"
        )

    pose = POSES[arm_name][pose_name]
    if arm_name == "middle" and abs(MIDDLE_WAIST_DRIVER_SHIFT) > 1e-9:
        pose = np.asarray(pose, dtype=float).copy()
        pose[0] += MIDDLE_WAIST_DRIVER_SHIFT  # waist is joint 0
        ## ...then back onto the principal branch.  The tables predate the
        ## re-clock and some entries were already multi-turn (looking_left
        ## +4.179, witness -2.333); adding theta pushes those past -2*pi, and
        ## ext_position would honour the literal value -- commanding 'witness'
        ## as -5.475 instead of the physically identical +0.808 unwinds most
        ## of a turn to reach a pose the arm is already next to.  A multiple
        ## of 2*pi is the same angle, so this changes nothing physical.
        ##
        ## Only reached when a shift is configured, so the legacy theta = 0
        ## behaviour -- including the deliberate multi-turn entries -- is
        ## untouched.
        pose[0] = (pose[0] + np.pi) % (2 * np.pi) - np.pi
    return pose

