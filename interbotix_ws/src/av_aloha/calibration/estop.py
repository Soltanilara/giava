"""Emergency stop for the calibration motion tools.

WHY CTRL-C ALONE IS NOT ENOUGH
==============================
`set_joint_positions(..., blocking=False)` hands a goal to the interbotix
driver and returns immediately.  The driver then executes that trajectory on
its own.  Killing the Python process does NOT cancel it -- the script dies,
the arm keeps going to wherever it was last told.

So an emergency stop cannot just exit.  It has to actively command the arm
to hold where it is right now.  Torque stays on throughout (torque-off would
drop the arm under gravity); this freezes it in place.

WHY THE OBVIOUS HALT SILENTLY FAILS
===================================
`robot_control.stop_arm` commands the measured position with
`moving_time=0.05`.  But interbotix validates every command
(`arm.py:check_joint_limits`):

    speed = |goal - self.joint_commands| / moving_time
    if speed > joint_velocity_limits: return False        # REJECTED

Two traps in there.  `joint_commands` is the driver's last *accepted
command*, not the measured position -- so mid-motion the difference is the
whole remaining travel.  And `set_joint_positions` returns False rather than
raising, and nothing in the codebase checks the return value.

Concretely: 0.2 rad of remaining travel halted with moving_time=0.05 asks
for 4 rad/s against a 3.14 rad/s limit.  Rejected.  The stop silently does
nothing, and the arm continues -- precisely in the situation where you
pressed stop because it was moving too far, too fast.

This module sizes `moving_time` from the ACTUAL distance so the halt is
always inside the velocity limit, checks the return value, and escalates if
a command is still refused.
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

## Keep the halt inside this fraction of each joint's velocity limit, so a
## rounding difference between our arithmetic and the driver's cannot push
## it over the line into a rejection.
SPEED_SAFETY = 0.7

## Never command a halt slower than this, however small the excursion.
MIN_MOVING_TIME = 0.05

## Give up escalating after this many attempts per arm.
MAX_ATTEMPTS = 4

DEFAULT_VELOCITY_LIMIT = float(np.pi)


class EmergencyStop:
    """Halts every registered arm in place, on demand or on a signal."""

    def __init__(self) -> None:
        self._arms: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._installed = False
        self._prev_handlers: Dict[int, Any] = {}
        self.triggered = False
        self.last_report: List[str] = []

    # -------------------------------------------------------------- #
    def register(self, name: str, bot: Any) -> None:
        with self._lock:
            self._arms[name] = bot

    def unregister(self, name: str) -> None:
        with self._lock:
            self._arms.pop(name, None)

    @property
    def arm_names(self) -> List[str]:
        with self._lock:
            return sorted(self._arms)

    # -------------------------------------------------------------- #
    def halt(self, reason: str = "") -> List[str]:
        """Command every registered arm to hold its current position.

        Safe to call repeatedly and from a signal handler.  Returns a
        human-readable line per arm describing what happened -- including
        failures, which must never be silent."""
        with self._lock:
            arms = dict(self._arms)
        self.triggered = True
        report: List[str] = []

        for name, bot in arms.items():
            try:
                report.append(self._halt_one(name, bot))
            except Exception as exc:  # never let one arm block the others
                report.append(f"{name}: HALT FAILED ({type(exc).__name__}: "
                              f"{exc})")

        self.last_report = report
        if arms:
            print("\n" + "!" * 70)
            print(f"  EMERGENCY STOP{(' -- ' + reason) if reason else ''}")
            for line in report:
                print(f"    {line}")
            print("  Torque is still ON; the arms are holding position.")
            print("!" * 70, flush=True)
        return report

    # -------------------------------------------------------------- #
    def _halt_one(self, name: str, bot: Any) -> str:
        arm = bot.arm
        n = len(arm.group_info.joint_names)
        measured = np.asarray(arm.core.joint_states.position[:n], dtype=float)

        # The driver validates against its last ACCEPTED command, so that is
        # what sets the distance the halt has to cover -- not the measured
        # position.
        ref = measured
        getter = getattr(arm, "get_joint_commands", None)
        if getter is not None:
            try:
                ref = np.asarray(getter(), dtype=float)[:n]
            except Exception:
                pass

        vel = _velocity_limits(arm, n)
        delta = np.abs(measured - ref)
        # moving_time such that every joint's implied speed sits inside its
        # own limit, with margin.
        needed = float(np.max(delta / np.maximum(vel * SPEED_SAFETY, 1e-6)))
        mt = max(MIN_MOVING_TIME, needed)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            ok = arm.set_joint_positions(
                measured.tolist(), moving_time=mt,
                accel_time=min(0.02, 0.5 * mt), blocking=False)
            if ok:
                return (f"{name}: holding at measured position "
                        f"(moving_time {mt * 1e3:.0f} ms, "
                        f"max travel to cancel {np.max(delta) * 1e3:.0f} mrad"
                        + (f", attempt {attempt}" if attempt > 1 else "")
                        + ")")
            # Refused: the only lever is a longer moving_time.
            mt *= 2.0

        return (f"{name}: *** COULD NOT HALT *** the driver refused "
                f"{MAX_ATTEMPTS} commands (last moving_time "
                f"{mt / 2:.2f} s). Kill the roslaunch or hit the physical "
                f"power switch.")

    # -------------------------------------------------------------- #
    def install_signal_handlers(self) -> None:
        """Make Ctrl-C (and SIGTERM) halt the arms before anything else.

        The halt runs INSIDE the handler rather than by raising and
        unwinding, so it happens immediately instead of waiting for
        whatever the main thread is doing to reach a try/except."""
        if self._installed:
            return

        def _handler(signum, frame):
            name = {signal.SIGINT: "Ctrl-C", getattr(signal, "SIGTERM", None):
                    "SIGTERM"}.get(signum, str(signum))
            self.halt(reason=name)
            prev = self._prev_handlers.get(signum)
            if signum == signal.SIGINT:
                # Re-raise as KeyboardInterrupt so callers still get their
                # normal cleanup and save paths -- but only AFTER the arms
                # have been told to stop.
                raise KeyboardInterrupt
            if callable(prev):
                prev(signum, frame)
            else:
                sys.exit(1)

        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                self._prev_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # Not the main thread, or the platform refuses -- the
                # explicit halt() path still works.
                pass
        self._installed = True

    def describe(self) -> str:
        arms = self.arm_names
        if not arms:
            return "emergency stop: no arms registered (nothing to halt)"
        return (f"emergency stop ARMED for {', '.join(arms)} -- "
                f"Ctrl-C halts them in place, torque stays on")


def _velocity_limits(arm: Any, n: int) -> np.ndarray:
    try:
        v = np.asarray(arm.group_info.joint_velocity_limits, dtype=float)[:n]
        v = np.where(v > 1e-6, v, DEFAULT_VELOCITY_LIMIT)
        if v.shape[0] == n:
            return v
    except Exception:
        pass
    return np.full(n, DEFAULT_VELOCITY_LIMIT)


## One shared instance: every motion tool in this package registers its arms
## with it, so a single Ctrl-C halts whatever is moving.
ESTOP = EmergencyStop()
