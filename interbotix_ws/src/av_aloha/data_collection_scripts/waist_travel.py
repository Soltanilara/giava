"""Driver-space travel limits for the camera arm's waist.

WHY THE PHYSICAL STOP IS INVISIBLE TO EVERYTHING ELSE
=====================================================
The middle arm's waist has a real mechanical limit -- the arm was rebuilt
with the motor clocked so that its working range sits near the encoder seam,
and past a certain angle it simply cannot turn.  Nothing in the software
knows that, and each of the three places that could have known is looking at
a different frame:

  giava.urdf            middle_base lower/upper = +-3.1416.  That is a FULL
                        2*pi of range, i.e. "the waist may point at any
                        yaw" -- no constraint at all in practice.
  wx250s_7dof.urdf      waist lower/upper = +-4.90, deliberately widened for
  (the driver's)        extended-position multi-turn.  Wider than physical.
  CoupledStudyIK        driver_to_urdf() WRAPS the waist into [-pi, pi]
                        before the solver sees it, and urdf_to_driver() maps
                        the answer back to whichever 2*pi-equivalent is
                        nearest the current driver reading.

That last one is the important one.  The wrap means the solver can never see
that the arm is sitting near a stop: whatever the driver reads, the solver
gets a legal angle in [-pi, pi] and treats the waist as free.  The inverse
map then hands the command back out near the current position, so a target
that needs a few more degrees of yaw sails straight past the stop.  The
motor takes it, pushes, and the current climbs -- measured on hardware at
1256, 1547 and 1719 mA against a ~2300 mA overload latch, on the same motor
that latched at 2095 mA earlier the same day.

So the limit has to be expressed HERE, in DRIVER radians, because that is
the only frame in which the stop holds still.  In URDF space it moves,
because the wrap moves it.

WHAT THIS DOES NOT DO
=====================
It clamps the COMMAND; it does not tell the solver.  The solver keeps asking
for the yaw it wants and the clamp keeps refusing the last few degrees, so
at the extreme the camera stops following the operator's head.  That is the
honest outcome -- the arm physically cannot go there, and the alternative is
a motor pushing into a stop until it latches.  Teaching the solver to spend
`camera_yaw` (middle_pan, +-[-0.76, 3.3]) instead would be the better fix
and is a bigger change: pyroki bakes joint limits at build time, and the
usable waist range is only knowable in driver space at run time.

CONFIGURING IT
==============
    GIAVA_MIDDLE_WAIST_MIN / _MAX   driver radians.  Unset = no limit.
    GIAVA_MIDDLE_WAIST_LEARN=1      tighten the limit when the waist stalls
                                    (session-local, only ever tightens)
    GIAVA_MIDDLE_WAIST_MARGIN       cushion inside a learned stop, default
                                    0.03 rad (~1.7 deg)

The numbers are NOT guessed here.  With nothing configured this module is
inert and only reports: on a waist stall it prints the angle the arm jammed
at and the exact variable to set.  Run once, read the line, set it.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"{name} must be a number (driver radians), got {raw!r}")


class WaistTravel:
    """Clamp one joint of one arm to a driver-space range.

    Inert unless a bound is configured or learned, so constructing it always
    costs nothing and the call sites need no branch."""

    def __init__(self, arm: str = "middle", waist_idx: int = 0,
                 lo: Optional[float] = None, hi: Optional[float] = None):
        self.arm = arm
        self.waist_idx = int(waist_idx)
        self.lo = lo if lo is not None else _env_float("GIAVA_MIDDLE_WAIST_MIN")
        self.hi = hi if hi is not None else _env_float("GIAVA_MIDDLE_WAIST_MAX")
        self.learn = os.environ.get("GIAVA_MIDDLE_WAIST_LEARN", "0") == "1"
        self.margin = float(os.environ.get("GIAVA_MIDDLE_WAIST_MARGIN", "0.03"))
        self.clamps = 0
        self.learned = 0
        if self.lo is not None and self.hi is not None and self.lo >= self.hi:
            raise SystemExit(
                f"GIAVA_MIDDLE_WAIST_MIN ({self.lo}) must be below "
                f"GIAVA_MIDDLE_WAIST_MAX ({self.hi})")

    @property
    def active(self) -> bool:
        return self.lo is not None or self.hi is not None

    def describe(self) -> str:
        if not self.active:
            return (f"[waist] {self.arm} waist travel limit: none set "
                    + ("(learning from stalls)" if self.learn else
                       "(GIAVA_MIDDLE_WAIST_MIN/_MAX to set one)"))
        lo = "-inf" if self.lo is None else f"{self.lo:+.3f}"
        hi = "+inf" if self.hi is None else f"{self.hi:+.3f}"
        return (f"[waist] {self.arm} waist clamped to [{lo}, {hi}] rad "
                f"(driver frame)"
                + (f", learning" if self.learn else ""))

    def clamp(self, q_arm_cmd) -> Tuple[np.ndarray, bool]:
        """Clamp the waist element of one arm's DRIVER-space command.

        Returns (command, was_clamped).  The array is copied only when it
        actually changes, so the common path allocates nothing."""
        if not self.active:
            return q_arm_cmd, False
        q = np.asarray(q_arm_cmd, dtype=float)
        if self.waist_idx >= q.shape[0]:
            return q_arm_cmd, False
        w = float(q[self.waist_idx])
        lim = w
        if self.lo is not None:
            lim = max(lim, self.lo)
        if self.hi is not None:
            lim = min(lim, self.hi)
        if lim == w:
            return q_arm_cmd, False
        out = q.copy()
        out[self.waist_idx] = lim
        self.clamps += 1
        return out, True

    def note_stall(self, arm: str, culprit_idx: int, commanded, measured
                   ) -> Optional[str]:
        """A stall was just detected.  Returns a line to print, or None.

        Direction comes from command-vs-measured: a command asking to go UP
        that does not move locates an UPPER stop, and one asking to go DOWN
        locates a LOWER stop.  Both readings are individually valid.

        WHEN BOTH FIRE, THE JOINT IS WEDGED, NOT AT A LIMIT.  A travel stop
        blocks one direction and leaves the other free; a joint that will not
        move either way is jammed on something, and the two "limits" it
        reports will cross (observed on hardware: stalled going up at +3.504,
        then going down at +3.447, which would learn lower +3.477 above upper
        +3.474).  Applying that would forbid the entire range and pin the arm
        exactly where it is stuck -- the opposite of the intent -- so a
        crossed pair is detected, reported, and discarded rather than
        applied."""
        if arm != self.arm or int(culprit_idx) != self.waist_idx:
            return None
        cmd = float(np.asarray(commanded, dtype=float)[self.waist_idx])
        meas = float(np.asarray(measured, dtype=float)[self.waist_idx])
        if cmd == meas:
            return None

        up = cmd > meas
        var = "GIAVA_MIDDLE_WAIST_MAX" if up else "GIAVA_MIDDLE_WAIST_MIN"
        bound = "upper" if up else "lower"
        suggested = meas - self.margin if up else meas + self.margin

        if not self.learn:
            return (f"[waist] jammed at {meas:+.3f} rad with the command "
                    f"{cmd:+.3f} ({bound} travel). To stop commanding past "
                    f"it: {var}={suggested:.3f}  (or "
                    f"GIAVA_MIDDLE_WAIST_LEARN=1 to pick it up "
                    f"automatically)")

        ## Only ever tighten.  A stall further out than a bound already
        ## learned says nothing new.
        if up:
            if self.hi is not None and suggested >= self.hi:
                return None
            new_lo, new_hi = self.lo, suggested
        else:
            if self.lo is not None and suggested <= self.lo:
                return None
            new_lo, new_hi = suggested, self.hi

        if new_lo is not None and new_hi is not None and new_lo >= new_hi:
            ## Wedged, not limited -- see the docstring.  Drop BOTH learned
            ## bounds: whichever one was learned first is now unreliable too,
            ## because the evidence for it was the same jam.
            self.lo = self.hi = None
            self.learn = False
            return (f"[waist] stalled in BOTH directions around "
                    f"{meas:+.3f} rad -- that is a WEDGED joint, not a travel "
                    f"limit (a real stop leaves the other direction free). "
                    f"Learned limits discarded and learning disabled; clear "
                    f"whatever is blocking the waist, or set "
                    f"GIAVA_MIDDLE_WAIST_MIN/_MAX by hand.")

        self.lo, self.hi = new_lo, new_hi
        self.learned += 1
        return (f"[waist] jammed going {'UP' if up else 'DOWN'} at "
                f"{meas:+.3f} rad -- {bound} limit now {suggested:+.3f} "
                f"(learned). Bake it in with {var}={suggested:.3f}")
