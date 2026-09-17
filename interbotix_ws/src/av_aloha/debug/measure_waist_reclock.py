"""Measure the middle waist's physical re-clock (theta) against the live arm.

    python measure_waist_reclock.py --pose rest
    python measure_waist_reclock.py --pose forward --reading 0.0015

THE ONE THING THAT CANNOT BE READ OFF THE BUS
=============================================
A physical re-clock -- the motor bolted to the arm at a different angle -- is
invisible to every register.  Homing_Offset does not see it (and is inert in
ext_position anyway); Present_Position reports the shaft, which is exactly the
thing that moved.  The ONLY way to recover theta is to observe one physical
pose in both frames:

    theta = driver_reading_now - driver_reading_that_pose_used_to_have

So: put the arm PHYSICALLY in a pose whose old driver value is recorded in
arm_config.POSES, then run this.  It reads the live waist, differences it
against the table, and reports what that theta does to every other pose.

"PHYSICALLY in a pose" means the camera is actually pointing where that pose
points -- not that the servo reports the pose's numbers.  Differencing two
approximate rest poses is NOT a measurement: the non-waist joints can sit at
their old goals while the waist has been jogged anywhere, which is precisely
the situation after a re-clock.  Sight the camera against something you can
check (the workspace centreline, the view a stored pose is documented to give
in arm_config.py) before trusting the number.

Nothing here commands the arm or writes any register.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_config import POSES  # noqa: E402

TWO_PI = 2.0 * math.pi


def seam_distance(rad: float) -> float:
    """Radians from `rad` to the nearest encoder seam (an odd multiple of pi).

    The seam is where a BOOT reading jumps +pi -> -pi.  In ext_position the
    joint runs multi-turn and never wraps mid-session, so this only matters
    for where the arm is parked at power-off -- which is the whole point."""
    return math.pi - abs((rad + math.pi) % TWO_PI - math.pi)


def live_waist() -> float:
    from interbotix_xs_modules.core import InterbotixRobotXSCore
    core = InterbotixRobotXSCore(
        robot_model="wx250s_7dof", robot_name="puppet_middle", init_node=True)
    r = core.robot_get_motor_registers("single", "waist", "Present_Position")
    if not r.values:
        raise SystemExit("could not read waist Present_Position -- driver up?")
    ticks = int(r.values[0])
    if ticks >= (1 << 31):
        ticks -= 1 << 32
    ## xs_sdk publishes (ticks - 2048) * 2*pi/4096; matching it here keeps this
    ## number comparable with joint_states and with the pose tables.
    return (ticks - 2048) * TWO_PI / 4096.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pose", required=True,
                    help=f"which middle pose the arm is PHYSICALLY in "
                         f"({', '.join(sorted(POSES['middle']))})")
    ap.add_argument("--reading", type=float, default=None,
                    help="driver waist [rad]; omit to read the live arm")
    args = ap.parse_args()

    if args.pose not in POSES["middle"]:
        raise SystemExit(f"unknown pose {args.pose!r}; have "
                         f"{sorted(POSES['middle'])}")

    old = float(POSES["middle"][args.pose][0])
    now = args.reading if args.reading is not None else live_waist()
    theta = now - old

    print(f"\npose '{args.pose}' waist was {old:+.4f} rad in the table")
    print(f"arm reads                  {now:+.4f} rad now")
    print(f"\n  theta = {theta:+.4f} rad  ({math.degrees(theta):+.2f} deg)")

    ## A re-clock done by flipping the horn is exactly a half turn, and lands
    ## the arithmetic somewhere much tidier (waist_urdf_offset becomes 2*pi,
    ## i.e. the driver frame and the URDF frame coincide).  Worth noticing.
    for k in (-2, -1, 1, 2):
        if abs(theta - k * math.pi) < math.radians(3):
            print(f"  -- within {math.degrees(abs(theta - k*math.pi)):.2f} deg "
                  f"of exactly {k}*pi. If the horn was flipped a half turn, the "
                  f"true value IS {k*math.pi:+.4f} and the remainder is pose "
                  f"error; confirm before rounding to it.")

    print(f"\n  waist_urdf_offset = pi - theta = {math.pi - theta:+.4f} rad")
    print(f"\n  MIDDLE_WAIST_RECLOCK_RAD = {theta:.6f}")
    print(f"  (or GIAVA_MIDDLE_WAIST_RECLOCK={theta:.6f})")

    print(f"\n{'pose':<16s} {'old':>9s} {'new':>9s} {'seam':>8s}")
    worst, worst_name = 999.0, ""
    for name, q in sorted(POSES["middle"].items()):
        o = float(q[0])
        n = o + theta
        d = seam_distance(n)
        flag = "  << NEAR SEAM" if d < math.radians(20) else ""
        print(f"{name:<16s} {o:+9.3f} {n:+9.3f} {math.degrees(d):7.1f} deg{flag}")
        if d < worst:
            worst, worst_name = d, name

    print(f"\ntightest seam clearance: {math.degrees(worst):.1f} deg ({worst_name})")
    if worst < math.radians(20):
        print("!! Something still parks close to the seam. A pose within ~20 deg")
        print("   of it can boot on either branch, which is the failure this")
        print("   re-clock exists to remove.")
    else:
        print("   Comfortable. Single-turn `position` mode is viable at this theta.")

    ## The recorded datasets are the other half of the frame change, and the
    ## span they cover is wider than any single pose.
    print(f"\nRecorded middle_base span (573,776 frames, mod 2*pi) was "
          f"[3.037, 4.426] rad;")
    lo, hi = 3.037 + theta, 4.426 + theta
    print(f"  under this theta that becomes [{lo:+.3f}, {hi:+.3f}], clearing the "
          f"seam by {math.degrees(min(seam_distance(lo), seam_distance(hi))):.1f} deg.")
    print("\nNothing was written. Put theta in robot_control."
          "MIDDLE_WAIST_RECLOCK_RAD once you trust it.")


if __name__ == "__main__":
    main()
