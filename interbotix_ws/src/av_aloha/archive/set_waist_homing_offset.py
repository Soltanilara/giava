"""One-time fix: move the middle waist's encoder seam away from the rest pose.

Why
---
The Dynamixel's absolute encoder is single-turn: at boot it reports the shaft
angle in (-pi, +pi].  The middle arm was assembled with the camera-forward rest
pose sitting essentially AT that seam (pose tables record waist ~ +3.10), so a
hair of mechanical settling at power-off decides whether the next boot reads
+3.1 or -3.1 -- and with it, which yaw direction runs into a limit wall.

The software equivalent of re-clocking the horn is the servo's Homing_Offset
register:  reported = actual + offset.  Shifting by -90 deg makes both former
boot readings collapse to one deterministic value (~ +1.57) and moves the seam
90 deg away from camera-forward.

In Position Control Mode the servo accepts offsets only within +/-1024 ticks
(+/-90 deg); larger values are silently ignored, which is why 180 deg is not an
option without physical re-assembly.

Usage (with the 3-arm driver running, data_collection NOT running):
    python set_waist_homing_offset.py            # apply -90 deg
    python set_waist_homing_offset.py --degrees 90
    python set_waist_homing_offset.py --degrees 0   # revert to stock

No code constants need editing afterwards: data_collection reads this register
at startup and derives the URDF offset and pose-table shift from it.
"""

from __future__ import annotations

import argparse
import time

TICKS_PER_REV = 4096
LIMIT_TICKS = 1024  # Position Control Mode accepts only +/-1024


def _signed(v: int) -> int:
    return v - (1 << 32) if v >= (1 << 31) else v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degrees", type=float, default=-90.0,
                    help="new Homing_Offset in degrees (clamped to +/-90)")
    ap.add_argument("--robot-name", default="puppet_middle")
    ap.add_argument("--robot-model", default="wx250s_7dof")
    args = ap.parse_args()

    from interbotix_xs_modules.core import InterbotixRobotXSCore

    deg = max(-90.0, min(90.0, args.degrees))
    if deg != args.degrees:
        print(f"clamped {args.degrees} -> {deg} deg (Position Mode limit)")
    ticks = int(round(deg / 360.0 * TICKS_PER_REV))
    assert -LIMIT_TICKS <= ticks <= LIMIT_TICKS

    core = InterbotixRobotXSCore(
        robot_model=args.robot_model, robot_name=args.robot_name, init_node=True
    )

    old = core.robot_get_motor_registers("single", "waist", "Homing_Offset")
    old_ticks = _signed(int(old.values[0])) if old.values else 0
    pos_before = core.robot_get_motor_registers("single", "waist", "Present_Position")
    print(f"current Homing_Offset: {old_ticks} ticks "
          f"({old_ticks / TICKS_PER_REV * 360.0:+.1f} deg)")

    # EEPROM register: requires torque off.  The waist is a vertical axis, so
    # nothing sags while it is briefly limp.
    core.robot_torque_enable("single", "waist", False)
    time.sleep(0.2)
    core.robot_set_motor_registers("single", "waist", "Homing_Offset", ticks)
    time.sleep(0.2)
    new = core.robot_get_motor_registers("single", "waist", "Homing_Offset")
    new_ticks = _signed(int(new.values[0])) if new.values else None

    # Re-seat the goal BEFORE torque comes back, or the arm swings.
    #
    # Writing Homing_Offset moves the REPORTED frame (reported = actual +
    # offset) without moving the shaft.  Goal_Position is a raw tick value in
    # RAM and is not rewritten by the shift, so after a -90 deg write the servo
    # believes it is 1024 ticks from goal.  Re-enabling torque then drives the
    # whole camera arm 90 deg at profile_velocity 200 -- with the arm in its
    # rest cradle and the operator's hands nearby.  Pinning Goal_Position to
    # the freshly reported Present_Position makes the error zero across the
    # transition, so torque comes back on a stationary arm.
    settle = core.robot_get_motor_registers("single", "waist", "Present_Position")
    if settle.values:
        core.robot_set_motor_registers(
            "single", "waist", "Goal_Position", _signed(int(settle.values[0])))
        time.sleep(0.1)
    else:
        print("!! could not read Present_Position -- NOT torquing back on.")
        print("   Hold the arm, then re-enable manually once the goal is sane.")
        return

    core.robot_torque_enable("single", "waist", True)

    if new_ticks != ticks:
        print(f"!! readback mismatch: wrote {ticks}, servo reports {new_ticks}")
        print("   (offset beyond +/-1024 is ignored in Position Control Mode)")
        return

    pos_after = core.robot_get_motor_registers("single", "waist", "Present_Position")
    print(f"new Homing_Offset: {new_ticks} ticks ({deg:+.1f} deg) -- WRITTEN")
    if pos_before.values and pos_after.values:
        print(f"waist Present_Position: {_signed(int(pos_before.values[0]))} -> "
              f"{_signed(int(pos_after.values[0]))} ticks (reported frame shifted; "
              "the shaft did not move)")
    print()
    print("Expected from now on:")
    print(f"  - boot reading at the rest pose is deterministic (~{3.14 + deg / 180 * 3.14:+.2f} rad "
          "regardless of which side the gearbox settled)")
    print(f"  - the encoder seam now sits {abs(deg):.0f} deg "
          f"{'clockwise' if deg < 0 else 'counter-clockwise'} of camera-forward")
    print("  - restart the driver + data_collection; they read this register at")
    print("    startup, so pose tables and the URDF offset adjust automatically")


if __name__ == "__main__":
    main()
