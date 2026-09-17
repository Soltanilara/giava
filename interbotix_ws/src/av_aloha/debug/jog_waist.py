"""Keyboard jog for the middle arm's waist -- a re-clocking aid.

Usage (3-arm driver running, data_collection NOT running):
    python jog_waist.py                 # 1 deg fine / 10 deg coarse
    python jog_waist.py --fine 0.5 --coarse 5
    python jog_waist.py --joint camera_yaw

KEYS
    j / k     step CW / CCW by the fine step
    h / l     step CW / CCW by the coarse step
    - / =     halve / double the fine step
    t         torque off/on -- torque OFF lets you turn the joint BY HAND,
              which is what you want while the arm is apart
    m         mark here as the reference zero (deltas are measured from it)
    p         reprint status
    q         quit (leaves torque as you left it)

WHY IT GUARDS THE CURRENT
    This joint's fault history is 8 of 9 middle-arm faults, every one within
    0.2 rad of the +/-pi seam, twice at 2095-2289 mA against a ~2300 mA
    overload latch.  A jog that walks into the mechanical stop is exactly how
    that happens, so every step checks Present_Current afterwards and backs
    the goal off if the joint is pushing rather than turning.  STALL_I_MA
    matches servo_health.py's own constant.

    Torque comes back on ONLY after Goal_Position is seated at the present
    reading.  Skipping that is what makes the arm swing when torque returns
    (see set_waist_homing_offset.py).
"""

from __future__ import annotations

import argparse
import math
import sys
import termios
import time
import tty

TICKS_PER_REV = 4096
ZERO_TICK = 2048            # xs_sdk reports (ticks - 2048) * 2*pi/4096
CURRENT_UNIT_MA = 2.69          # XM430 / XM540, as in servo_health.py
STALL_I_MA = 1000.0             # servo_health.py's STALL_I_MA
SETTLE_S = 1.2                  # how long to let a step finish


def _signed(v: int, bits: int = 32) -> int:
    return v - (1 << bits) if v >= (1 << (bits - 1)) else v


class Waist:
    def __init__(self, core, joint: str):
        self.core, self.joint = core, joint

    def _get(self, reg: str, bits: int = 32):
        r = self.core.robot_get_motor_registers("single", self.joint, reg)
        return _signed(int(r.values[0]), bits) if r.values else None

    @property
    def pos(self) -> int:
        return self._get("Present_Position")

    @property
    def current_ma(self) -> float:
        raw = self._get("Present_Current", bits=16)
        return abs(raw) * CURRENT_UNIT_MA if raw is not None else 0.0

    def set_goal(self, ticks: int) -> None:
        self.core.robot_set_motor_registers(
            "single", self.joint, "Goal_Position", int(ticks))

    def torque(self, on: bool) -> None:
        ## Seat the goal at where the joint actually IS before energising, or
        ## the servo drives to a stale goal the moment torque returns.
        if on:
            here = self.pos
            if here is None:
                print("\r!! cannot read Present_Position -- NOT torquing on.")
                return
            self.set_goal(here)
            time.sleep(0.1)
        self.core.robot_torque_enable("single", self.joint, on)


def fmt(ticks: int, mark: int | None) -> str:
    """One status line, in the DRIVER frame that joint_states publishes.

    THE TICK ORIGIN IS 2048, NOT 0.  The xs_sdk reports
    (ticks - 2048) * 2*pi/4096, so tick 2048 is 0 rad and the encoder seam --
    where a boot reading jumps +pi -> -pi -- sits at tick 0/4096.  Verified on
    hardware: tick 1995 reads -0.0813 rad in /puppet_middle/joint_states, and
    the pre-rebuild rest pose of +3.1324 rad is tick 4090, six ticks off the
    wrap.  Getting this origin wrong inverts the seam distance, which is the
    one number this tool exists to show.
    """
    rad = (ticks - ZERO_TICK) * 2.0 * math.pi / TICKS_PER_REV
    deg = math.degrees(rad)
    ## Seam = the nearest whole turn away from tick 0, in tick space, so this
    ## stays correct for the multi-turn values ext_position reports.
    seam_tk = abs(ticks - round(ticks / TICKS_PER_REV) * TICKS_PER_REV)
    seam = seam_tk / TICKS_PER_REV * 360.0
    out = f"{ticks:+7d} tk  {deg:+8.2f} deg  {rad:+7.4f} rad  seam {seam:5.1f} deg"
    if mark is not None:
        d = (ticks - mark) / TICKS_PER_REV * 360.0
        out += f"  |  from mark {d:+7.2f} deg ({math.radians(d):+.4f} rad)"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fine", type=float, default=1.0, help="fine step [deg]")
    ap.add_argument("--coarse", type=float, default=10.0, help="coarse step [deg]")
    ap.add_argument("--joint", default="waist")
    ap.add_argument("--robot-name", default="puppet_middle")
    ap.add_argument("--robot-model", default="wx250s_7dof")
    ap.add_argument("--max-ma", type=float, default=STALL_I_MA)
    args = ap.parse_args()

    from interbotix_xs_modules.core import InterbotixRobotXSCore

    core = InterbotixRobotXSCore(
        robot_model=args.robot_model, robot_name=args.robot_name, init_node=True)
    w = Waist(core, args.joint)

    here = w.pos
    if here is None:
        sys.exit(f"cannot read {args.joint} Present_Position -- is the driver up?")
    w.set_goal(here)                      # never inherit a stale goal
    time.sleep(0.1)

    fine, coarse = args.fine, args.coarse
    mark, torque_on = here, True

    print(__doc__.split("KEYS")[1].split("WHY")[0].rstrip())
    print(f"\n{args.joint} on {args.robot_name}   guard {args.max_ma:.0f} mA")
    print(fmt(here, None) + "   [marked]")

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            key = sys.stdin.read(1)
            if key == "q":
                break

            if key == "t":
                torque_on = not torque_on
                w.torque(torque_on)
                state = "ON" if torque_on else "OFF -- turn it by hand"
                print(f"\rtorque {state}\n  {fmt(w.pos, mark)}")
                continue
            if key == "m":
                mark = w.pos
                print(f"\rmarked\n  {fmt(mark, mark)}")
                continue
            if key == "p":
                print(f"\r  {fmt(w.pos, mark)}   {w.current_ma:.0f} mA")
                continue
            if key in "-=":
                fine = max(0.05, fine / 2) if key == "-" else min(45.0, fine * 2)
                print(f"\rfine step {fine:.2f} deg")
                continue

            deg = {"j": -fine, "k": +fine, "h": -coarse, "l": +coarse}.get(key)
            if deg is None:
                continue
            if not torque_on:
                print("\rtorque is OFF -- press t to energise before jogging")
                continue

            start = w.pos
            goal = start + int(round(deg / 360.0 * TICKS_PER_REV))
            w.set_goal(goal)

            ## Watch the step instead of assuming it. A joint that is pushing
            ## rather than moving gets its goal handed straight back, which
            ## ends the push before the overload accumulator cares.
            peak, t0 = 0.0, time.time()
            while time.time() - t0 < SETTLE_S:
                time.sleep(0.05)
                peak = max(peak, w.current_ma)
                if peak > args.max_ma:
                    break
            now = w.pos
            if peak > args.max_ma:
                w.set_goal(now)
                print(f"\r!! {peak:.0f} mA at {fmt(now, mark)}")
                print(f"   backed off -- that direction is against a stop. "
                      f"Moved {abs(now - start)} tk of "
                      f"{abs(goal - start)} tk asked.")
                continue
            print(f"\r  {fmt(now, mark)}   {peak:.0f} mA")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        final = w.pos
        print(f"\n{args.joint} left at {fmt(final, mark)}")
        print(f"torque {'ON' if torque_on else 'OFF'}")
        if mark is not None and final is not None and final != mark:
            d = (final - mark) / TICKS_PER_REV * 360.0
            print(f"\nNet re-clock from mark: {d:+.2f} deg "
                  f"({d * 3.141592653589793 / 180.0:+.4f} rad)")
            print("That is the theta to apply to the waist driver frame.")


if __name__ == "__main__":
    main()
