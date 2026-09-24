"""Dynamixel fault detection, diagnosis, and in-place recovery.

WHY THIS EXISTS
================
2026-08-27, three-arm teleop: the right arm stopped following, its command
state drifted 4.071 rad from measured, and the enable-time resync faithfully
adopted a pose whose forward kinematics put `right_gripper_base` 307 mm BELOW
the table.

That pose is NOT physically reachable on this rig -- the tabletop is in the
way -- which is the sharper version of the problem.  It means the resync did
not adopt a real fallen pose; it adopted joint angles that no real arm was in.
Whether those came from a limp motor still reporting stale counts, a bad read,
or an encoder that had wrapped, the failure is the same: something upstream
handed the loop a measurement that could not be true, and the loop believed it.

Nothing noticed: `set_joint_positions`' return value is discarded, and the only
measured-vs-commanded check ran at teleop enable.  A guard that asks "is this
pose physically possible" catches this class of fault regardless of its cause,
which a drift threshold alone does not.

A servo that latches `Hardware_Error_Status` shuts its own torque off and
stays that way.  Every command after that is accepted by the driver, published
onto the bus, and silently ignored by a limp motor.  From inside the control
loop it looks exactly like a healthy arm -- which is why this has to be
detected explicitly rather than inferred.

TWO SIGNALS, DIFFERENT COSTS
=============================
`joint_states` carries POSITION and EFFORT (Present_Current, mA -- see
xs_sdk_obj.cpp:1229) at 100 Hz, already subscribed, free to read.  So:

  * tracking error (free, every tick) -- a limp arm's measured position walks
    away from the command without bound.  This is the TRIP.
  * effort (free, every tick) -- overload shutdown is a LATCHED INTEGRAL of
    current over time, so sustained high current precedes the fault by
    seconds.  This is the WARNING, and the only signal that gives any chance
    of backing off before the motor latches.
  * `Hardware_Error_Status` (a blocking ROS service round-trip) -- the ground
    truth, and the only thing that says WHICH motor and WHY.  Read only after
    a trip, never in the hot path.

TRACKING THRESHOLD, AND WHY IT IS NOT SMALL
============================================
The driver runs a TIME-based profile: a command is a goal to reach in
`moving_time` (0.14 s default), so the arm legitimately trails the command by
up to one profile's worth of motion -- at the driver clamp's ~2.8 rad/s that
is ~0.40 rad of perfectly healthy lag.  A threshold under that false-trips on
every fast move.  Hence 0.7 rad SUSTAINED for 0.75 s: a live arm closes the
gap within one profile, a limp one never does.

RECOVERY IS A REBOOT, NOT A POWER CYCLE
========================================
The DYNAMIXEL `REBOOT` instruction clears the latched error, and the SDK
already exposes it (`robot_reboot_motors`, xs_sdk_obj.cpp:189) with
`smart_reboot` to touch only the motors actually in error.  Two things the
SDK does NOT do for us, both handled in `recover()`:

  1. `Profile_Velocity` / `Profile_Acceleration` are RAM and reset to 0 on
     reboot.  `InterbotixArmXSInterface.set_trajectory_time` only writes them
     when its CACHED value changes (arm.py:82), so after a reboot it never
     rewrites them -- and profile 0 means "no profile", i.e. the next command
     executes at maximum speed.  They are rewritten here unconditionally.
  2. `smart_reboot` reads `Hardware_Error_Status` on the GROUP's joints only;
     shadow motors (shoulder_shadow, elbow_shadow) are not group members, so a
     shadow that faults is invisible to it and never rebooted.  They are polled
     and rebooted by name here.

Recovery reboots with torque OFF, republishes the MEASURED position as the
goal, and only then torques on -- so the motor holds where the arm actually
is instead of wherever its goal register happened to land.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from arm_config import ARM_CONFIG


## Control-table bits, X-series (Hardware_Error_Status, address 70).
ERROR_BITS: Tuple[Tuple[int, str], ...] = (
    (0x01, "input voltage out of range"),
    (0x04, "overheating"),
    (0x08, "motor encoder error"),
    (0x10, "electrical shock / short"),
    (0x20, "OVERLOAD -- sustained torque past the limit"),
)

## Not group members, so `smart_reboot` cannot see them.  Present on both
## vx300s and wx250s_7dof (see their motor config yamls).
SHADOW_MOTORS: Tuple[str, ...] = ("shoulder_shadow", "elbow_shadow")

## Sustained tracking error that means "this arm is not following" -- see the
## module docstring for why it is larger than one profile's worth of lag.
TRIP_RAD = float(os.environ.get("GIAVA_SERVO_TRIP_RAD", "0.7"))
TRIP_HOLD_S = float(os.environ.get("GIAVA_SERVO_TRIP_HOLD_S", "0.75"))

## Present_Current [mA] that, held this long, is walking toward an overload
## latch.  Warning only -- it never stops the arm on its own.
EFFORT_WARN_MA = float(os.environ.get("GIAVA_SERVO_EFFORT_WARN_MA", "1400"))
EFFORT_WARN_HOLD_S = float(os.environ.get("GIAVA_SERVO_EFFORT_HOLD_S", "1.0"))


## The reading a DEAD BUS produces, and why it is not obvious.
##
## When the xs_sdk's syncRead fails (motors unpowered, cable out, U2D2 gone),
## the default `read_failure_behavior` is PUB_DXL_WB (xs_sdk_obj.cpp:621) --
## it publishes whatever the workbench produced, which for a failed read is
## the ZERO-INITIALISED raw buffer.  `convertValue2Radian(id, 0)` on an
## X-series maps raw 0 to -3.1431 rad, so EVERY joint of a disconnected arm
## reports -pi, at the full 100 Hz, with a fresh header stamp.  Present_Current
## may instead come back with the workbench's last buffered values, frozen at
## whatever the arm was drawing when it died.
##
## So a dead bus looks like: a live topic, a plausible timestamp, a physically
## impossible pose, and a constant current.  Nothing downstream can tell it
## from a real reading without this check -- which is how a watchdog ended up
## warning for 64 s about the overload risk of an arm that had been off the
## bus the whole time (hardware, 2026-08-27).
DEAD_BUS_RAD = -3.1431247288715194
DEAD_BUS_TOL = 2e-3


def bus_reading_is_dead(measured: np.ndarray) -> bool:
    """True when every joint reports the failed-read sentinel (-pi)."""
    if measured.size == 0:
        return True
    if not np.all(np.isfinite(measured)):
        return True     # the PUB_NAN behaviour, if it is ever configured
    return bool(np.all(np.abs(measured - DEAD_BUS_RAD) < DEAD_BUS_TOL))


def decode(bits: int) -> str:
    """Human-readable `Hardware_Error_Status`."""
    if bits == 0:
        return "clean"
    named = [text for mask, text in ERROR_BITS if bits & mask]
    unknown = bits & ~sum(mask for mask, _ in ERROR_BITS)
    if unknown:
        named.append(f"unknown bits 0x{unknown:02x}")
    return f"0x{bits:02x} ({', '.join(named)})"


def motor_names(bot, arm: str) -> List[str]:
    """Every motor on this arm's bus: group joints, shadows, gripper.

    Taken from the DRIVER (`group_info.joint_names`), never from
    `ARM_CONFIG`.  The two disagree: the URDF calls the camera arm's joints
    `middle_base`/`middle_upper_arm`/..., the driver calls them
    `waist`/`elbow`/`camera_roll`/... (wx250s_7dof.yaml:125).  Register
    services take the driver's names."""
    names = list(bot.arm.group_info.joint_names)
    names += list(SHADOW_MOTORS)
    if ARM_CONFIG[arm].get("has_gripper"):
        names.append("gripper")
    return names


def read_hardware_errors(bot, arm: str,
                         timeout_s: float = 2.0) -> Dict[str, int]:
    """{motor_name: Hardware_Error_Status} -- one blocking service call per
    motor.  NEVER raises: a diagnostic that can kill the session is worse
    than no diagnostic.  Motors that could not be read are omitted."""
    out: Dict[str, int] = {}
    all_names = motor_names(bot, arm)
    deadline = time.monotonic() + timeout_s
    for name in all_names:
        if time.monotonic() > deadline:
            print(f"[servo health] {arm}: register read timed out; "
                  f"{len(out)} of {len(all_names)} motors checked")
            break
        try:
            resp = bot.dxl.robot_get_motor_registers(
                "single", name, "Hardware_Error_Status")
            values = list(getattr(resp, "values", []) or [])
            if values:
                out[name] = int(values[0])
        except Exception as exc:
            ## A motor absent from this model's config answers with an error,
            ## which is information, not a failure: skip it quietly unless it
            ## is one we expected.
            if name not in SHADOW_MOTORS:
                print(f"[servo health] {arm}: could not read "
                      f"'{name}': {exc}")
    return out


def report(bot, arm: str) -> Tuple[Dict[str, int], bool]:
    """Print a per-motor health report.  Returns (errors, any_faulted)."""
    errs = read_hardware_errors(bot, arm)
    if not errs:
        ## The service ANSWERED (no exception) but returned nothing.  The SDK
        ## returns early without filling `values` when `itemRead` fails on the
        ## bus (xs_sdk_obj.cpp:395), so the node is alive and the MOTORS are
        ## not answering.  That distinction is the whole diagnosis: a software
        ## reboot cannot reach a motor that is not on the bus.
        print(f"[servo health] {arm}: the xs_sdk node answered but NO MOTOR "
              f"did.\n"
              f"  This is not a latched fault -- it is a dead bus, and no "
              f"software reboot can reach it.\n"
              f"  Check, in this order: 12 V power to the {arm} arm; the "
              f"U2D2/USB cable; that its serial port still exists\n"
              f"  (ls -l /dev/ttyDXL*); then the daisy-chain connector at the "
              f"first motor.  Power-cycle the arm.")
        return errs, False
    faulted = {n: v for n, v in errs.items() if v}
    print(f"\n[servo health] {arm}: {len(errs)} motors polled, "
          f"{len(faulted)} in error")
    for name, value in errs.items():
        flag = "  << FAULT" if value else ""
        print(f"    {name:<18s} {decode(value)}{flag}")
    if faulted:
        print(f"  '{arm}' has latched a hardware error: its torque is OFF and "
              f"every command is being ignored.\n"
              f"  Type  reboot  to clear it in place (no power cycle needed) "
              f"-- the motors are rebooted torqued OFF, re-given their "
              f"MEASURED position as the goal, then torqued back on, so the "
              f"arm holds where it is instead of jumping.")
    return errs, bool(faulted)


def recover(bot, arm: str, moving_time: float, accel_time: float) -> bool:
    """Clear latched faults on `arm` in place.  Returns True on success.

    Sequence, and why each step is here:
      1. reboot the faulted motors AND their shadows, torque OFF
      2. rewrite Profile_Velocity / Profile_Acceleration -- RAM, reset to 0
         by the reboot, and `set_trajectory_time` will not rewrite them
         because its cache says they are already correct.  Profile 0 = no
         profile = the next command runs at maximum speed.
      3. republish the MEASURED position as the goal, then torque on, so the
         motor holds the pose the arm is actually in.
    """
    errs = read_hardware_errors(bot, arm)
    faulted = sorted(n for n, v in errs.items() if v)
    if not faulted:
        print(f"[servo health] {arm}: nothing latched; nothing to reboot.")
        return True

    ## A shadow shares its joint's load: rebooting one of a pair and not the
    ## other leaves two motors driving the same output from different states.
    to_reboot = set(faulted)
    for shadow in SHADOW_MOTORS:
        master = shadow.replace("_shadow", "")
        if shadow in to_reboot or master in to_reboot:
            to_reboot.update({shadow, master})
    to_reboot &= set(errs)

    print(f"[servo health] {arm}: rebooting {', '.join(sorted(to_reboot))} "
          f"(torque off)")
    for name in sorted(to_reboot):
        try:
            bot.dxl.robot_reboot_motors("single", name, False)
        except Exception as exc:
            print(f"[servo health] {arm}: reboot of '{name}' failed: {exc}")
            return False
    time.sleep(0.5)   # the bus needs a moment before the motor answers again

    ## Step 2 -- see the docstring.  Written through the service directly
    ## rather than via set_trajectory_time(), whose cache would skip it.
    try:
        group = bot.arm.group_name
        bot.dxl.robot_set_motor_registers(
            "group", group, "Profile_Velocity", int(moving_time * 1000))
        bot.dxl.robot_set_motor_registers(
            "group", group, "Profile_Acceleration", int(accel_time * 1000))
        bot.arm.moving_time = moving_time
        bot.arm.accel_time = accel_time
    except Exception as exc:
        print(f"[servo health] {arm}: could NOT restore the motion profile "
              f"({exc}).  Do not command this arm -- with Profile_Velocity 0 "
              f"the next command executes at maximum speed.")
        return False

    ## Step 3 -- goal := measured, THEN torque on.
    n = ARM_CONFIG[arm]["num_joints"]
    measured = np.asarray(bot.dxl.joint_states.position[:n], dtype=float)
    try:
        bot.arm.joint_commands = measured.tolist()
        bot.arm.publish_positions(measured.tolist(),
                                  moving_time=moving_time,
                                  accel_time=accel_time,
                                  blocking=False)
        bot.dxl.robot_torque_enable("group", bot.arm.group_name, True)
    except Exception as exc:
        print(f"[servo health] {arm}: torque-on failed: {exc}")
        return False

    time.sleep(0.2)
    after = read_hardware_errors(bot, arm)
    still = sorted(n_ for n_, v in after.items() if v)
    if still:
        print(f"[servo health] {arm}: STILL FAULTED after reboot: "
              f"{', '.join(still)}.  A fault that survives a reboot is the "
              f"condition persisting (a stalled joint against a hard stop, an "
              f"overheated motor that has not cooled, a wiring fault) -- fix "
              f"the cause, then power cycle.")
        return False

    print(f"[servo health] {arm}: recovered.  Command state is stale by "
          f"construction -- press 'i' to re-park and resync before driving.")
    return True


class ServoWatchdog:
    """Free, per-tick 'is this arm still following?' check.

    Reads only what `joint_states` already carries.  Trips on SUSTAINED
    tracking error (a limp arm), warns on SUSTAINED current (an arm walking
    toward an overload latch).  Never commands anything itself -- it reports,
    and the control loop decides."""

    def __init__(self, arm_names, trip_rad: float = TRIP_RAD,
                 trip_hold_s: float = TRIP_HOLD_S,
                 effort_warn_ma: float = EFFORT_WARN_MA,
                 effort_hold_s: float = EFFORT_WARN_HOLD_S):
        self.arm_names = list(arm_names)
        self.trip_rad = float(trip_rad)
        self.trip_hold_s = float(trip_hold_s)
        self.effort_warn_ma = float(effort_warn_ma)
        self.effort_hold_s = float(effort_hold_s)
        self._bad_since: Dict[str, Optional[float]] = {a: None for a in self.arm_names}
        self._hot_since: Dict[str, Optional[float]] = {a: None for a in self.arm_names}
        self.tripped: Dict[str, bool] = {a: False for a in self.arm_names}
        self.dead_bus: Dict[str, bool] = {a: False for a in self.arm_names}
        ## Next elapsed-time at which the "drawing high current" warning may
        ## print again, per arm.  It BACKS OFF (1 s, 2, 4, 8, ... capped at
        ## 60) rather than repeating on a fixed period: the first few tell
        ## you something, and sixty identical lines a minute apart tell you
        ## nothing while burying everything else in the terminal.
        self._hot_next: Dict[str, float] = {a: 1.0 for a in self.arm_names}
        self.last_error: Dict[str, float] = {a: 0.0 for a in self.arm_names}
        self.last_joint: Dict[str, int] = {a: -1 for a in self.arm_names}

    def clear(self, arm: Optional[str] = None) -> None:
        for a in ([arm] if arm else self.arm_names):
            self._bad_since[a] = None
            self._hot_since[a] = None
            self.tripped[a] = False
            self.dead_bus[a] = False
            self._hot_next[a] = 1.0

    def update(self, robots, cmd_state, t: float) -> List[str]:
        """One tick.  Returns the arms that JUST tripped (newly, once each).

        VALIDITY BEFORE ANY VERDICT.  A disconnected arm keeps publishing
        joint_states -- see `bus_reading_is_dead` -- so both signals this
        class reads are still there, still fresh-looking, and both are lies:
        the tracking error is ~0 (the command state was synced FROM the dead
        reading) and the current is frozen wherever the arm died.  Judging
        either one first is how a dead bus gets reported as an impending
        overload.  So the bus is checked first and, when it is dead, nothing
        else about that arm is evaluated at all."""
        newly: List[str] = []
        for arm in self.arm_names:
            n = ARM_CONFIG[arm]["num_joints"]
            try:
                js = robots[arm].dxl.joint_states
                measured = np.asarray(js.position[:n], dtype=float)
                effort = np.asarray(js.effort[:n], dtype=float)
            except Exception:
                continue

            ## ---- validity gate ------------------------------------- ##
            if bus_reading_is_dead(measured):
                self._bad_since[arm] = None
                self._hot_since[arm] = None
                if not self.dead_bus[arm]:
                    self.dead_bus[arm] = True
                    self.tripped[arm] = True
                    newly.append(arm)
                continue
            if self.dead_bus[arm]:
                self.dead_bus[arm] = False
                print(f"[servo health] {arm}: the bus is answering again.")

            commanded = np.asarray(cmd_state.last_cmds.get(arm, measured),
                                   dtype=float)
            if commanded.shape != measured.shape:
                continue

            err = np.abs(measured - commanded)
            k = int(np.argmax(err))
            self.last_error[arm] = float(err[k])
            self.last_joint[arm] = k

            if err[k] > self.trip_rad:
                if self._bad_since[arm] is None:
                    self._bad_since[arm] = t
                elif (t - self._bad_since[arm]) >= self.trip_hold_s \
                        and not self.tripped[arm]:
                    self.tripped[arm] = True
                    newly.append(arm)
            else:
                self._bad_since[arm] = None

            if effort.size and float(np.max(np.abs(effort))) > self.effort_warn_ma:
                if self._hot_since[arm] is None:
                    self._hot_since[arm] = t
            else:
                self._hot_since[arm] = None
        return newly

    def hot_arms(self, t: float) -> List[Tuple[str, float]]:
        """Arms drawing warning-level current for longer than the hold time.

        Reports each arm on an EXPONENTIAL BACKOFF, so a condition that
        persists says so a handful of times instead of once every two
        seconds forever."""
        out = []
        for arm in self.arm_names:
            since = self._hot_since[arm]
            if since is None or self.dead_bus[arm]:
                self._hot_next[arm] = 1.0
                continue
            held = t - since
            if held >= self.effort_hold_s and held >= self._hot_next[arm]:
                ## Next allowed time, not next interval: capping the INTERVAL
                ## at 60 s leaves `held >= next` true forever once the cap is
                ## reached, which fires every tick -- the exact flood this is
                ## meant to prevent.  Doubles up to a 60 s spacing, then holds
                ## that spacing.
                self._hot_next[arm] = held + min(max(held, 1.0), 60.0)
                out.append((arm, held))
        return out

    def describe(self, arm: str) -> str:
        if self.dead_bus[arm]:
            return (f"{arm}: NO MOTOR ON THE BUS -- every joint reads the "
                    f"failed-read sentinel (-3.143 rad). The topic is live; "
                    f"the arm is not.")
        j = self.last_joint[arm]
        names = ARM_CONFIG[arm]["joint_names"]
        joint = names[j] if 0 <= j < len(names) else "?"
        return (f"{arm}: not following -- {joint} is {self.last_error[arm]:.3f} "
                f"rad from its command for over {self.trip_hold_s:.2f} s")


## ------------------------------------------------------------------ ##
## Register-level diagnostics: motion profile, and master/shadow health
## ------------------------------------------------------------------ ##
##
## Both are READ-ONLY and need no motion.  They exist because the two
## failures they catch are invisible from the joint_states topic:
##   * the group can be running a profile whose UNITS are not the ones the
##     control loop's clamp math assumes (see PROFILE_UNITS below), and
##   * a master and its shadow can be fighting each other at high current
##     while the joint sits perfectly still.

## Register width in BYTES.  Needed because `robot_get_motor_registers`
## returns the raw little-endian word widened into an int32 WITHOUT sign
## extension (dynamixel_driver.cpp readRegister: DXL_MAKEWORD / MAKEDWORD
## into a uint32).  A -200 mA current therefore arrives as 65336, and an
## extended-mode negative position as ~4.29e9.  Every signed register has
## to be folded back by hand.
REG_BYTES = {
    "Present_Position": 4, "Present_Current": 2, "Present_Velocity": 4,
    "Homing_Offset": 4, "Goal_Position": 4,
    "Profile_Velocity": 4, "Profile_Acceleration": 4,
    "Drive_Mode": 1, "Operating_Mode": 1, "Torque_Enable": 1,
    "Hardware_Error_Status": 1, "Shutdown": 1,
}
SIGNED_REGS = {"Present_Position", "Present_Current", "Present_Velocity",
               "Homing_Offset", "Goal_Position"}

## X-series control-table scalings.
COUNTS_PER_REV = 4096.0
CURRENT_UNIT_MA = 2.69          # XM430 / XM540; XL-series differs
PROFILE_VEL_UNIT_RPM = 0.229    # velocity-based profile only
PROFILE_ACC_UNIT_RPM2 = 214.577  # velocity-based profile only

## Master/shadow verdict thresholds.
SHADOW_POS_TOL_COUNTS = 20      # ~1.76 deg -- calibration should be exact
SHADOW_IDLE_CURRENT_MA = 300.0  # while stationary; above this they are working


def read_register(bot, motor: str, reg: str) -> Optional[int]:
    """One register, sign-corrected.  None if it could not be read."""
    try:
        resp = bot.dxl.robot_get_motor_registers("single", motor, reg)
        values = list(getattr(resp, "values", []) or [])
        if not values:
            return None
        v = int(values[0])
    except Exception:
        return None
    if reg in SIGNED_REGS:
        bits = 8 * REG_BYTES.get(reg, 4)
        if v >= (1 << (bits - 1)):
            v -= (1 << bits)
    return v


def counts_to_rad(counts: int) -> float:
    return counts * 2.0 * np.pi / COUNTS_PER_REV


def read_profile_modes(bot, arm: str):
    """{joint: (mode, pv, pa)} for every joint the ARM GROUP command drives.

    mode is "time" | "velocity".  Joints that did not answer are omitted; an
    empty dict means nothing on the bus replied.

    GROUP JOINTS ONLY -- `bot.arm.group_info.joint_names`, not motor_names().
    Shadows mirror whatever is written to their master (Secondary_ID), and
    the gripper is commanded on its own path, so neither takes part in
    deciding whether a GROUP position command is executable.
    """
    out = {}
    try:
        names = list(bot.arm.group_info.joint_names)
    except Exception:
        return out
    for name in names:
        drive = read_register(bot, name, "Drive_Mode")
        if drive is None:
            continue
        out[name] = (
            "time" if (drive & 0x04) else "velocity",
            read_register(bot, name, "Profile_Velocity"),
            read_register(bot, name, "Profile_Acceleration"),
        )
    return out


def summarize_profiles(per):
    """Collapse {joint: (mode, pv, pa)} to the ONE profile a group command
    has to satisfy.  Returns (mode, pv, pa), or (None, None, None) if empty.

    WHY "ANY VELOCITY-BASED JOINT MAKES THE GROUP VELOCITY-BASED"
    =============================================================
    A group position command is one write to joints that may be running
    different profiles, and the clamp on it has to be executable by ALL of
    them.  The two profiles bound completely different things:

      velocity-based   the servo will not exceed pv * 0.229 rev/min, so a
                       per-tick step larger than v_cap * dt CANNOT be
                       executed -- the goal simply runs away from the arm.
      time-based       the servo covers whatever distance it is given in
                       pv milliseconds, so there is no speed bound at all
                       and any step is "executable" (at whatever speed that
                       takes).

    So the velocity-based joints are the binding constraint, and a mixed
    group must be clamped as if it were velocity-based.  Doing the reverse --
    which is what reading only the first joint used to do here -- lets the
    clamp hand the velocity-based joints a step they physically cannot
    perform.  Clamping the time-based joints to the velocity joints' cap
    only makes them move more gently, which is always executable.

    THE MIDDLE ARM IS EXACTLY THIS CASE.  puppet_modes_middle.yaml puts the
    whole `arm` group in a velocity-based profile and then overrides `waist`
    and `camera_yaw` under `singles:` with `profile_type: time`.  Both are
    group joints, so whether the old first-joint read reported "time" or
    "velocity" depended on the driver's joint ordering -- and reporting
    "time" is what let GIAVA_PROFILE_MODE=auto pick the legacy clamp for an
    arm whose other five joints cap out six times lower.

    pv is the TIGHTEST cap among the velocity-based joints, because the
    group can only go as fast as its slowest joint.  pv == 0 means NO
    PROFILE (maximum speed) rather than "the tightest possible", so it is
    excluded from that minimum and only survives when every joint has it.
    """
    if not per:
        return None, None, None
    vel = [(pv, pa) for m, pv, pa in per.values() if m == "velocity"]
    if not vel:
        ## Every joint time-based.  The group is done when its SLOWEST joint
        ## is, so the largest move time is the honest one to report.  Nothing
        ## downstream clamps on it (profile_limits gives a time-based profile
        ## no v_max at all); it is for the operator reading the banner.
        pvs = [pv for _, pv, _ in per.values() if pv]
        pas = [pa for _, _, pa in per.values() if pa]
        return "time", (max(pvs) if pvs else 0), (max(pas) if pas else 0)
    caps = [pv for pv, _ in vel if pv]
    accs = [pa for _, pa in vel if pa]
    return "velocity", (min(caps) if caps else 0), (min(accs) if accs else 0)


def read_profile_mode(bot, arm: str):
    """(mode, profile_velocity, profile_acceleration) a GROUP command must
    satisfy.  mode is "time" | "velocity" | None (nothing answered).

    Reads EVERY joint in the group and collapses them with
    summarize_profiles(); see there for why a mixed group is velocity-based.

    This exists so the control loop can ASK the hardware what profile it is
    running instead of assuming.  The assumption was wrong on this robot for
    months: the mode yamls request a velocity-based profile, while
    `InterbotixArmXSInterface` only logs an objection and then writes
    `moving_time * 1000` into Profile_Velocity as if it were milliseconds.
    """
    return summarize_profiles(read_profile_modes(bot, arm))


def profile_limits(mode: str, pv: int, pa: int, moving_time: float):
    """What the servo's profile actually bounds, in SI units.

    Returns {"v_max": rad/s or None, "a_max": rad/s^2 or None}.

    Under a TIME-based profile there is no speed bound at all -- the servo
    computes whatever velocity is needed to cover the commanded distance in
    Profile_Velocity milliseconds, so peak speed scales with how far the goal
    jumped.  That is the property that makes an IK discontinuity dangerous
    there and bounded here.
    """
    if mode == "time":
        return {"v_max": None, "a_max": None,
                "move_s": (pv or 0) / 1000.0, "accel_s": (pa or 0) / 1000.0}
    v = (pv or 0) * PROFILE_VEL_UNIT_RPM * 2.0 * np.pi / 60.0
    a = (pa or 0) * PROFILE_ACC_UNIT_RPM2 * 2.0 * np.pi / 3600.0
    return {"v_max": (v if pv else None), "a_max": (a if pa else None)}


def check_profile(bot, arm: str) -> bool:
    """Print what motion profile the motors are ACTUALLY running.

    Returns True when the group is time-based (what the interbotix Python
    layer and this repo's driver clamp both assume).

    WHY THIS IS WORTH A COMMAND OF ITS OWN
    =======================================
    `Drive_Mode` bit 2 decides what Profile_Velocity / Profile_Acceleration
    MEAN, and nothing upstream notices when it disagrees with the code:

      bit 2 SET   time-based     PV = total move time [ms]
                                 PA = accel ramp time [ms]
      bit 2 CLEAR velocity-based PV = speed cap [0.229 rev/min]
                                 PA = accel cap [214.577 rev/min^2]

    `InterbotixArmXSInterface.__init__` only LOGS when the group is not
    time-based (arm.py:42) and then writes `int(moving_time * 1000)` into
    PV regardless -- so under a velocity-based profile, `moving_time=0.14`
    silently becomes a 3.36 rad/s speed cap instead of a 140 ms move, and
    `cfg.driver_max_step` (which is derived from moving_time) is measuring
    something the servo is not doing.  A value of 0 means NO PROFILE AT
    ALL: the servo drives at maximum speed toward the goal.
    """
    names = motor_names(bot, arm)
    print(f"\n[servo profile] {arm}")
    all_time = True
    for name in names:
        mode = read_register(bot, name, "Operating_Mode")
        drive = read_register(bot, name, "Drive_Mode")
        pv = read_register(bot, name, "Profile_Velocity")
        pa = read_register(bot, name, "Profile_Acceleration")
        if drive is None:
            continue
        time_based = bool(drive & 0x04)
        all_time = all_time and time_based
        reverse = "reverse" if (drive & 0x01) else "normal "
        mode_name = {0: "current", 1: "velocity", 3: "position",
                     4: "ext_position", 5: "current_pos",
                     16: "pwm"}.get(mode, str(mode))
        if time_based:
            profile = (f"time-based: move {pv} ms, accel {pa} ms"
                       if pv else "time-based: PV=0 -> NO PROFILE (max speed)")
        else:
            v = (pv or 0) * PROFILE_VEL_UNIT_RPM * 2 * np.pi / 60.0
            a = (pa or 0) * PROFILE_ACC_UNIT_RPM2 * 2 * np.pi / 3600.0
            profile = (f"velocity-based: {v:.2f} rad/s cap, {a:.1f} rad/s^2"
                       if pv else
                       "velocity-based: PV=0 -> NO PROFILE (max speed)")
        print(f"    {name:<18s} {mode_name:<12s} {reverse}  {profile}")
    if not all_time:
        print("  NOT time-based.  `moving_time` / `accel_time` / "
              "`driver_max_step` are all being interpreted by the servo in "
              "velocity-profile units -- see this function's docstring.")
    return all_time


def check_shadows(bot, arm: str) -> bool:
    """Compare each master motor with its shadow.  True when all pairs agree.

    HOW THE PAIR IS SUPPOSED TO WORK
    =================================
    Master and shadow drive the same joint from opposite sides.  The shadow
    carries `Secondary_ID` = the master's ID, so every WRITE addressed to
    the master (Goal_Position, Profile_*) also reaches the shadow; it never
    answers reads on that ID, so reads must use its own.  Its `Drive_Mode`
    reverse bit is set opposite to the master's, since they face each other.

    At xs_sdk STARTUP ONLY (xs_sdk_obj.cpp:801), the SDK reads both
    Present_Positions and writes the difference into the shadow's
    Homing_Offset.  After that both motors report the SAME Present_Position.
    So a disagreement here means the calibration instant was bad -- the arm
    was being pushed, mid-sag, or one motor had just wrapped -- and the two
    motors have been pulling against each other ever since.  That is a
    stationary, sustained, high-current condition, which is exactly what
    latches an overload.

    THE TWO SIGNALS
    ================
      position   should match within a few counts.  Tens of counts apart =
                 mis-calibrated; relaunch the xs_sdk with the arm at rest.
      current    while the joint is STATIONARY, both should be small and of
                 the same sign (sharing the gravity load).  Large and
                 OPPOSITE = fighting.  Confirm the sign convention once
                 against a known-good pair before trusting the sign alone;
                 the magnitudes are unambiguous on their own.
    """
    print(f"\n[servo shadows] {arm}")
    ok = True
    found = False
    for shadow in SHADOW_MOTORS:
        master = shadow.replace("_shadow", "")
        m_pos = read_register(bot, master, "Present_Position")
        s_pos = read_register(bot, shadow, "Present_Position")
        if m_pos is None or s_pos is None:
            continue
        found = True
        m_cur = read_register(bot, master, "Present_Current") or 0
        s_cur = read_register(bot, shadow, "Present_Current") or 0
        m_ma, s_ma = m_cur * CURRENT_UNIT_MA, s_cur * CURRENT_UNIT_MA
        d = s_pos - m_pos
        off = read_register(bot, shadow, "Homing_Offset")

        flags = []
        if abs(d) > SHADOW_POS_TOL_COUNTS:
            flags.append(f"POSITION MISMATCH {abs(d)} counts "
                         f"({np.degrees(counts_to_rad(abs(d))):.2f} deg)")
        if (m_ma * s_ma) < 0 and max(abs(m_ma), abs(s_ma)) > SHADOW_IDLE_CURRENT_MA:
            flags.append("OPPOSITE CURRENTS -- the pair may be fighting")
        elif max(abs(m_ma), abs(s_ma)) > SHADOW_IDLE_CURRENT_MA:
            flags.append("high current (fine if the joint is loaded/moving)")
        ok = ok and not flags

        print(f"    {master:<16s} pos {m_pos:>8d}   current {m_ma:>+8.0f} mA")
        print(f"    {shadow:<16s} pos {s_pos:>8d}   current {s_ma:>+8.0f} mA"
              f"   (homing offset {off})")
        print(f"      delta {d:+d} counts"
              + ("   << " + "; ".join(flags) if flags else "   ok"))
    if not found:
        print("    no shadow motors answered -- this arm may not have any")
    elif not ok:
        print("  Shadow calibration happens ONCE, at xs_sdk launch, from a "
              "single instant's readings.  Relaunch the driver with the arms "
              "at rest and untouched, then re-check.")
    return ok


## ------------------------------------------------------------------ ##
## Seeing a fault coming, and recording what caused it
## ------------------------------------------------------------------ ##
##
## The overload latch is an integral of current over time INSIDE the servo,
## and the only thing published about it is `Present_Current` in
## joint_states.  Two consequences shape everything below:
##
##   * A fault is never instantaneous.  By the time Hardware_Error_Status
##     latches, the motion that caused it is seconds in the past -- which is
##     why "the right arm keeps faulting" is hard to diagnose from the
##     terminal: whatever is on screen at that moment is the aftermath.
##     `FaultRecorder` keeps the seconds BEFORE.
##
##   * The same integral can be estimated on the host from the same signal.
##     `OverloadEstimator` runs a simple I^2-t accumulator per joint, which
##     rises while a joint is working hard and decays while it is not.  That
##     gives a graduated warning, and something to act on, instead of a
##     binary trip after the fact.

## Current a joint can hold indefinitely without heating [mA].  Below this
## the accumulator decays.  XM430/XM540-class; deliberately conservative.
OVERLOAD_I_CONT_MA = float(os.environ.get("GIAVA_OVERLOAD_I_CONT_MA", "900"))
## How long at DOUBLE that current should count as "at the limit" [s].  This
## sets the accumulator's scale: E is normalised so 1.0 == that condition.
## It is a HEURISTIC, not a datasheet figure -- the servo's real thresholds
## are not published per-model.  Calibrate it from a recorded fault: run the
## arm until it latches, open the dump, and see what E had reached.
OVERLOAD_T_LIMIT_S = float(os.environ.get("GIAVA_OVERLOAD_T_LIMIT_S", "6.0"))
## Where the graduated response starts and where it saturates.
OVERLOAD_WARN = float(os.environ.get("GIAVA_OVERLOAD_WARN", "0.35"))
OVERLOAD_MAX_DERATE = float(os.environ.get("GIAVA_OVERLOAD_MAX_DERATE", "0.25"))


class OverloadEstimator:
    """Per-joint I^2-t accumulator: a host-side model of the servo's latch.

    E_j grows as (i^2 - i_cont^2) while a joint pulls more than it can hold
    and decays at the same rate while it pulls less, normalised so E = 1.0
    is "2 x continuous current for OVERLOAD_T_LIMIT_S seconds".

    WHY A DERATE AND NOT A TRIP
    ============================
    A binary cut-off at a threshold is the wrong shape twice over: it does
    nothing at all until it does everything, and the thing it does (stop) is
    itself a hazard mid-motion.  Scaling the per-tick step down as E rises
    bleeds energy out of exactly the joint that is accumulating it, while
    the operator keeps control -- the arm gets heavy rather than dying.  And
    because the accumulator decays, easing off restores full speed on its
    own without anyone resetting anything.

    This is a MODEL, not a measurement of the servo's internal state.  It
    cannot be exact; its job is to move the failure from "latched, torque
    off, arm falls" to "that joint felt sluggish for a few seconds".
    """

    def __init__(self, arm_names, i_cont_ma: float = OVERLOAD_I_CONT_MA,
                 t_limit_s: float = OVERLOAD_T_LIMIT_S):
        self.arm_names = list(arm_names)
        self.i_cont = float(i_cont_ma)
        ## Normaliser: (2*i_cont)^2 - i_cont^2 = 3*i_cont^2, held t_limit.
        self._scale = 3.0 * self.i_cont ** 2 * float(t_limit_s)
        self.energy: Dict[str, np.ndarray] = {}
        self.peak: Dict[str, float] = {a: 0.0 for a in self.arm_names}

    def update(self, arm: str, effort_ma: np.ndarray, dt: float) -> float:
        """Advance one tick.  Returns this arm's worst normalised energy."""
        i2 = np.asarray(effort_ma, dtype=float) ** 2
        e = self.energy.get(arm)
        if e is None or e.shape != i2.shape:
            e = np.zeros_like(i2)
        ## Symmetric: the same expression charges above i_cont and discharges
        ## below it, so a joint that works hard then rests nets out near zero
        ## -- which is what duty cycle means physically.
        e = np.maximum(e + (i2 - self.i_cont ** 2) * dt / self._scale, 0.0)
        self.energy[arm] = e
        w = float(e.max()) if e.size else 0.0
        self.peak[arm] = max(self.peak[arm], w)
        return w

    def worst(self, arm: str):
        e = self.energy.get(arm)
        if e is None or not e.size:
            return 0.0, -1
        k = int(np.argmax(e))
        return float(e[k]), k

    def derate(self, arm: str) -> float:
        """Multiplier for this arm's per-tick step, in [MAX_DERATE, 1.0]."""
        w, _ = self.worst(arm)
        if w <= OVERLOAD_WARN:
            return 1.0
        span = max(1.0 - OVERLOAD_WARN, 1e-6)
        f = 1.0 - (1.0 - OVERLOAD_MAX_DERATE) * min((w - OVERLOAD_WARN) / span,
                                                    1.0)
        return float(max(f, OVERLOAD_MAX_DERATE))

    def describe(self, arm: str) -> str:
        w, k = self.worst(arm)
        names = ARM_CONFIG[arm]["joint_names"]
        joint = names[k] if 0 <= k < len(names) else "?"
        return (f"{arm}: {joint} is at {w * 100:.0f}% of the modelled "
                f"overload budget (step scaled to {self.derate(arm) * 100:.0f}%)")


## ------------------------------------------------------------------ ##
## Commanded but not moving: the stall gate
## ------------------------------------------------------------------ ##
##
## WHY THIS EXISTS ALONGSIDE OverloadEstimator
## ===========================================
## The I^2-t model above is a good description of the servo's THERMAL latch
## and a bad detector of the failure that actually keeps happening on this
## rig.  Read fault_right_20260828_142703.npz: three wrist joints
## (forearm_roll, wrist_angle, wrist_rotate) stop dead at the same instant,
## their measured positions pinned to within one encoder count for 2.5 s,
## while the commands walk away from them -- wrist_rotate ends 1.30 rad from
## its goal -- and the motors hold 2.0-3.6 A the whole time.  That is a
## MECHANICAL block (a hard stop, or the gripper cable winding up as
## forearm_roll runs out past 2.8 rad), not heat.
##
## The thermal model saw it only as E = 0.54 by the time the watchdog
## tripped, so a threshold anywhere near "90-95%" would never have fired.
## The signal that WAS unambiguous, 2.3 s earlier and at a third of the
## current, is much simpler:
##
##      commanded somewhere it is not   AND   not moving   AND   pulling hard
##
## No model, no calibration, no per-model datasheet.  A joint being asked to
## move, drawing current, and not moving is jammed -- there is no benign
## reading of it.
##
## WHAT THE GATE DOES
## ==================
## It does NOT stop the arm dead: that is its own hazard, and the operator
## is holding the other end.  It CLAMPS THE GOAL TO WHERE THE JOINT ACTUALLY
## IS.  The motor stops fighting the block immediately (this is the same
## move `smart_reboot` makes on recovery: goal := measured, so the servo
## holds where the arm is rather than where something wanted it), current
## falls, and the accumulated energy bleeds off -- while the operator keeps
## commanding, sees the arm has stopped following, and backs off.
##
## The hold releases on its own once the commanded target comes back within
## RELEASE_ERR of the measured position, i.e. once the operator has undone
## whatever drove into the block.  Nothing to reset, nothing to acknowledge.
STALL_ERR_RAD = float(os.environ.get("GIAVA_STALL_ERR_RAD", "0.05"))
## One encoder count is 2*pi/4096 = 1.53 mrad, and a jammed joint in the
## recorded fault jitters by exactly one.  2.5 mrad/tick is ~0.12 rad/s at
## 50 Hz: comfortably below any real motion, comfortably above the jitter.
STALL_MOVE_RAD = float(os.environ.get("GIAVA_STALL_MOVE_RAD", "0.0025"))
## Above the ~900 mA these joints hold against gravity, below the 2000 mA
## the recorded stall sat at for seconds.
STALL_I_MA = float(os.environ.get("GIAVA_STALL_I_MA", "1000"))
STALL_HOLD_S = float(os.environ.get("GIAVA_STALL_HOLD_S", "0.30"))
STALL_RELEASE_ERR_RAD = float(os.environ.get("GIAVA_STALL_RELEASE_ERR_RAD", "0.03"))
STALL_ENABLED = os.environ.get("GIAVA_STALL_GATE", "1") == "1"


class StallGate:
    """Per-arm hold for 'commanded, drawing current, and not moving'.

    Usage, once per tick, off the same joint_states the health loop already
    reads:

        gate.update(arm, cmd, meas, effort, dt)

    then, where the command is assembled:

        q = gate.apply(arm, q, meas)

    `apply` returns the measured position for a held arm and `q` unchanged
    for a healthy one, so the call site does not branch.
    """

    def __init__(self, arm_names, enabled: bool = STALL_ENABLED):
        self.arm_names = list(arm_names)
        self.enabled = bool(enabled)
        self._ticks: Dict[str, np.ndarray] = {}
        self._prev: Dict[str, np.ndarray] = {}
        self.held: Dict[str, bool] = {a: False for a in self.arm_names}
        self.culprit: Dict[str, int] = {a: -1 for a in self.arm_names}
        self.since: Dict[str, float] = {a: 0.0 for a in self.arm_names}
        self.trips: Dict[str, int] = {a: 0 for a in self.arm_names}

    def update(self, arm: str, cmd, meas, effort, dt: float) -> bool:
        """Advance one tick.  Returns True while this arm is held.

        Transitions are NOT printed here -- `newly_held` reports them, so a
        caller can log once per event instead of once per tick."""
        if not self.enabled:
            return False
        cmd = np.asarray(cmd, dtype=float)
        meas = np.asarray(meas, dtype=float)
        eff = np.abs(np.asarray(effort, dtype=float))
        n = min(len(cmd), len(meas), len(eff))
        if n == 0:
            return self.held.get(arm, False)
        cmd, meas, eff = cmd[:n], meas[:n], eff[:n]

        prev = self._prev.get(arm)
        self._prev[arm] = meas.copy()
        ticks = self._ticks.get(arm)
        if ticks is None or ticks.shape != meas.shape:
            ticks = np.zeros(n, dtype=float)
        if prev is None or prev.shape != meas.shape:
            self._ticks[arm] = ticks
            return self.held.get(arm, False)

        err = np.abs(cmd - meas)
        moved = np.abs(meas - prev)
        jammed = (err > STALL_ERR_RAD) & (moved < STALL_MOVE_RAD) & (eff > STALL_I_MA)
        ## Consecutive-tick counter per joint: one clean tick resets it, so a
        ## joint that is merely slow (moving, just not fast enough) never
        ## accumulates.  Only a joint that is genuinely not moving does.
        ticks = np.where(jammed, ticks + max(dt, 1e-6), 0.0)
        self._ticks[arm] = ticks

        if self.held.get(arm):
            ## Release when the operator has brought the target back to
            ## where the arm actually is.  Checked on the WORST joint, so a
            ## second jammed joint keeps the hold.
            if float(err.max()) <= STALL_RELEASE_ERR_RAD:
                self.held[arm] = False
                self.culprit[arm] = -1
                self._ticks[arm] = np.zeros(n, dtype=float)
            return self.held[arm]

        k = int(np.argmax(ticks))
        if ticks[k] >= STALL_HOLD_S:
            self.held[arm] = True
            self.culprit[arm] = k
            self.since[arm] = float(ticks[k])
            self.trips[arm] += 1
        return self.held[arm]

    def apply(self, arm: str, q_cmd, meas):
        """Goal for this tick: where the arm IS, while it is held."""
        if not self.enabled or not self.held.get(arm):
            return q_cmd
        meas = np.asarray(meas, dtype=float)
        q = np.asarray(q_cmd, dtype=float).copy()
        n = min(len(q), len(meas))
        q[:n] = meas[:n]
        return q

    def describe(self, arm: str, cmd=None, meas=None, effort=None) -> str:
        k = self.culprit.get(arm, -1)
        names = ARM_CONFIG[arm]["joint_names"]
        joint = names[k] if 0 <= k < len(names) else "?"
        extra = ""
        if cmd is not None and meas is not None and 0 <= k < len(names):
            extra = (f" commanded {float(cmd[k]):+.3f}, measured "
                     f"{float(meas[k]):+.3f} ({abs(float(cmd[k]) - float(meas[k])):.3f} "
                     f"rad away)")
            if effort is not None:
                extra += f" at {abs(float(effort[k])):.0f} mA"
        return (f"{arm}: '{joint}' is being commanded but is not moving{extra}. "
                f"Goal clamped to the measured position so the motor stops "
                f"pushing; back the command off to release.")


class FaultRecorder:
    """Rolling pre-fault buffer, written to disk when something trips.

    Keeps the last `seconds` of commanded joints, measured joints and
    current for every arm.  On a fault it dumps them, so the QUESTION
    "what motion causes this?" becomes a file rather than a memory.

    Sized in ticks, not bytes: at 50 Hz with three arms and 7 joints, 20 s
    is ~25k floats per channel -- nothing.  The cost of keeping it is zero
    and the cost of not having it is another session spent guessing.
    """

    def __init__(self, arm_names, seconds: float = 20.0, rate_hz: float = 50.0,
                 out_dir: Optional[str] = None):
        self.arm_names = list(arm_names)
        self.n = max(int(seconds * rate_hz), 10)
        self.out_dir = out_dir or str(Path(__file__).resolve().parent
                                      / "fault_logs")
        self.buf: Dict[str, List] = {a: [] for a in self.arm_names}

    def push(self, arm: str, t: float, cmd, meas, effort) -> None:
        b = self.buf.setdefault(arm, [])
        b.append((float(t), np.asarray(cmd, dtype=np.float32).copy(),
                  np.asarray(meas, dtype=np.float32).copy(),
                  np.asarray(effort, dtype=np.float32).copy()))
        if len(b) > self.n:
            del b[:len(b) - self.n]

    def dump(self, arm: str, reason: str, extra: Optional[Dict] = None):
        """Write this arm's buffer.  Returns the path, or None."""
        b = self.buf.get(arm) or []
        if not b:
            return None
        try:
            d = Path(self.out_dir)
            d.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = d / f"fault_{arm}_{stamp}.npz"
            np.savez_compressed(
                path,
                t=np.array([r[0] for r in b], dtype=np.float64),
                cmd=np.stack([r[1] for r in b]),
                meas=np.stack([r[2] for r in b]),
                effort=np.stack([r[3] for r in b]),
                joint_names=np.array(ARM_CONFIG[arm]["joint_names"]),
                reason=np.array(reason),
                **{k: np.array(v) for k, v in (extra or {}).items()},
            )
            return path
        except Exception as exc:
            print(f"[fault log] could not write the pre-fault buffer: {exc}")
            return None

    @staticmethod
    def summarise(path) -> str:
        """The few facts worth reading without opening the file."""
        try:
            z = np.load(path, allow_pickle=False)
        except Exception as exc:
            return f"    (could not re-read {path}: {exc})"
        eff = np.abs(np.asarray(z["effort"], dtype=float))
        names = [str(n) for n in z["joint_names"]]
        t = np.asarray(z["t"], dtype=float)
        span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        k = int(np.argmax(eff.max(axis=0))) if eff.size else -1
        lines = [f"    {span:.1f} s of history, {eff.shape[0]} ticks"]
        if 0 <= k < len(names):
            col = eff[:, k]
            hot = float((col > OVERLOAD_I_CONT_MA).mean()) * 100.0
            lines.append(f"    hardest-working joint: {names[k]} -- peak "
                         f"{col.max():.0f} mA, above {OVERLOAD_I_CONT_MA:.0f} "
                         f"mA for {hot:.0f}% of the window")
            err = np.abs(np.asarray(z["meas"], dtype=float)[:, k]
                         - np.asarray(z["cmd"], dtype=float)[:, k])
            lines.append(f"    its tracking error went "
                         f"{err[0]:.3f} -> {err[-1]:.3f} rad")
        return "\n".join(lines)
