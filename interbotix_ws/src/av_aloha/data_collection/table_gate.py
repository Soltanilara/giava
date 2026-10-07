"""Hard tabletop-floor gate on commanded joint configurations.

WHY THIS EXISTS, GIVEN THE SOLVER ALREADY HAS A TABLE COST
============================================================
`ik/table_collision.py` gives the solver a SOFT cost against the
tabletop plane, on the INSCRIBED sphere model (optimistic by up to ~18 mm on
true contact) -- exactly the same two reasons `capsule_gate.py` exists for
the inter-arm case: a soft cost can be traded away under a hard-enough
command, and an inscribed reading of "clear" can be real-world touching or
worse. Same fix, same shape: check the FINAL clamped command against
CIRCUMSCRIBED geometry, at the same call site as the inter-arm gate,
immediately before `set_joint_positions`.

WHAT "DON'T ALLOW NEGATIVE Z" MEANS HERE
=========================================
The request was literally "no part of the robot below table height" -- that
IS `pk.collision.HalfSpace` distance vs the tabletop plane, computed on
every link's circumscribed capsule (`collision_models.tight_capsule_collision`,
the same corrected long-OBB-axis fit `capsule_gate.py` uses).  Because the
capsule contains its link's full mesh,

    capsule_to_plane_distance(link, q)  <=  true_mesh_to_plane_distance(link, q)

so "every kept link's capsule distance >= margin" PROVES no mesh point is
within margin of the table, exactly the inter-arm gate's argument transposed
onto a plane.

SIMPLER THAN THE INTER-ARM GATE, ON PURPOSE
============================================
capsule-vs-capsule distance needs the coarse/GJK escalation in
`capsule_gate.py` because the coarse capsule fit pads BOTH bodies.
capsule-vs-PLANE distance has no such approximation to escalate away from --
it is `(closest point on the capsule's centerline to the plane) - radius`,
exact in closed form.  So this gate is coarse-only and does not need a fine
tier at all.

WHICH LINKS ARE CHECKED
========================
Every link's capsule, MINUS the six links kinematically height-invariant
under their own actuated joint (waist yaw doesn't change height): the
*_base_link and *_shoulder_link links (see table_collision.py's docstring
for the corpus measurement).  Measured with THIS gate's own circumscribed
capsules (2026-08-23): those six read a CONSTANT, often negative, distance
at every configuration (left_base_link -47.5 mm, right_base_link -47.3 mm,
middle_base_link -29.8 mm -- the corrected capsule fit pads the base mount
plate well past the table already at rest) -- a permanent violation with no
discriminative signal, the same "structural" failure mode C3 pruned for
self-collision.  Excluding them is a kinematic fact, not a corpus-coverage
choice, unlike everything else here.

MARGIN: FINGERS GET THEIR OWN, SAME REASONING AS THE INTER-ARM GATE
=====================================================================
The corrected single-capsule fit still can't represent a box well
(COLLISION_STUDY.md Part 2): it doubles a finger's thin dimension, so
finger capsules read close to the table well before the real fingertip
does, exactly when picking something up off it -- a task, not an accident.
Finger links get a small margin (default 8 mm, matching the inter-arm
gate's fingertip pairs); every other link keeps the full structural margin.

STATUS: HARDWARE-VALIDATED, NOT SWEPT
=====================================
Validated 2026-09-09 on the real arms -- the tabletop behaviour was confirmed
to be what was intended.  No margin sweep. Enabled by default (a floor gate is a coarse,
conservative backstop -- the failure mode of a wrong margin is "stops early",
not "fails to stop"), but watch `[table gate]` prints on first hardware use
and raise the margin if it holds somewhere unexpected.

CONFIG (env, same convention as capsule_gate)
==============================================
    GIAVA_TABLE_GATE            1 (default) | 0 -- master switch
    GIAVA_TABLE_GATE_MARGIN     metres, default 0.020
    GIAVA_TABLE_GATE_MARGIN_FINGER  metres, default 0.008
    GIAVA_TABLE_GATE_Z          metres, default 0.0 (world frame, same
                                 convention as base_validation.py --table-z)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent / "ik")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GATE_ENABLED = os.environ.get("GIAVA_TABLE_GATE", "1") == "1"
## 25 -> 20 mm on 2026-08-28.  Hardware: fingertips reach the table nicely
## (they run on GATE_MARGIN_FINGER, untouched), but a gripper YAWED flat to
## the table is governed by this structural margin and stopped ~30 mm up,
## which is more standoff than the task wants.
## 20 -> 15 mm on 2026-09-01: still too much standoff in practice -- the
## camera arm working low over the scene is also held here (the table gate
## has no camera tier; camera links run on this structural margin).  15 mm
## is the floor the inter-arm gate documents for trusting servo tracking,
## and the plane check is EXACT (closed-form capsule-vs-plane, no coarse
## pad on the plane side), so 15 mm here is not the gamble 15 mm would be
## between two coarse capsules.  Do not go lower without a hardware sweep.
GATE_MARGIN = float(os.environ.get("GIAVA_TABLE_GATE_MARGIN", "0.015"))
GATE_MARGIN_FINGER = float(os.environ.get("GIAVA_TABLE_GATE_MARGIN_FINGER", "0.008"))
GATE_TABLE_Z = float(os.environ.get("GIAVA_TABLE_GATE_Z", "0.0"))

## Kinematically height-invariant under their own joint, and (measured with
## the circumscribed capsule fit) permanently at or below the table --
## see module docstring. Same six links table_collision.py excludes from
## the soft cost, for the same underlying kinematic reason, plus the URDF
## root link itself: "base" has no collision mesh (a zero-radius, zero-height
## phantom capsule sitting exactly at the origin -- verified 2026-08-23), so
## it reads a constant 0.0 mm from the table plane by construction. It is a
## coordinate frame, not robot material.
TABLE_GATE_EXCLUDED_LINKS: Tuple[str, ...] = (
    "base",
    "left_base_link", "right_base_link", "middle_base_link",
    "left_shoulder_link", "right_shoulder_link", "middle_shoulder_link",
)


def _is_finger(link_name: str) -> bool:
    return link_name.endswith("_finger_link")


class TableGate:
    """Circumscribed-capsule-vs-tabletop-plane check, jitted once."""

    def __init__(self, robot, urdf, table_z: float = GATE_TABLE_Z,
                 margin: float = GATE_MARGIN,
                 margin_finger: float = GATE_MARGIN_FINGER,
                 excluded_links: Tuple[str, ...] = TABLE_GATE_EXCLUDED_LINKS):
        import jax
        import jax.numpy as jnp

        from collision_models import tight_capsule_collision
        from pyroki.collision import HalfSpace

        self.robot = robot
        self.urdf = urdf
        self.margin = float(margin)
        self.margin_finger = float(margin_finger)
        self.table_z = float(table_z)

        self.coll = tight_capsule_collision(urdf)
        names = self.coll.link_names
        kept = [i for i, n in enumerate(names) if n not in excluded_links]
        if not kept:
            raise RuntimeError(
                "table gate ended up with zero kept links -- the link "
                "naming convention must have changed")
        self._kept_idx = np.asarray(kept, dtype=np.int32)
        self.link_names: List[str] = [names[i] for i in kept]
        self._margin_per_kept = np.array(
            [self.margin_finger if _is_finger(n) else self.margin
             for n in self.link_names])

        self.plane = HalfSpace.from_point_and_normal(
            point=jnp.array([0.0, 0.0, self.table_z], dtype=jnp.float32),
            normal=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
        )

        ## `compute_world_collision_distance`'s own batch-cfg path mis-shapes
        ## its output on this robot's kinematic tree (verified 2026-08-23:
        ## an (n, 31)-shaped cfg comes back (31, n) instead of (n, 31, 1)).
        ## The single-config path is correct, so batching is done here with
        ## our own vmap instead of relying on the library's internal one.
        def _dists_one(cfg):
            d = self.coll.compute_world_collision_distance(
                self.robot, cfg, self.plane)
            return d[self._kept_idx, 0]

        self._dists_one = jax.jit(_dists_one)
        self._dists = jax.jit(jax.vmap(_dists_one))

        t0 = time.perf_counter()
        d = np.asarray(self._dists_one(np.zeros(
            robot.joints.num_actuated_joints, dtype=np.float32)))
        self._n_links = int(d.size)
        self.compile_ms = (time.perf_counter() - t0) * 1e3
        ## True when the last largest_safe_fraction() answered from inside
        ## the margin, i.e. the fraction returned is an ESCAPE step rather
        ## than a proven-clear one.  The control loop reads it only to say
        ## the right thing to the operator.
        self.last_escape = False

    # ------------------------------------------------------------------ #
    def check(self, q_urdf: np.ndarray) -> Tuple[bool, float, Optional[str]]:
        """(ok, min_distance_m, offending_link) at a URDF-coordinate config.

        ok=False means: this command could not be PROVEN clear of the table
        -- some link's circumscribed capsule comes within its margin of the
        plane.  Because the capsule contains the mesh, ok=True proves
        separation; ok=False is conservative, not a certainty of contact."""
        d = np.atleast_1d(np.asarray(self._dists_one(
            np.asarray(q_urdf, dtype=np.float32))))
        slack = d - self._margin_per_kept
        k = int(np.argmin(slack))
        dist = float(d[k])
        if slack[k] >= 0.0:
            return True, dist, None
        return False, dist, self.link_names[k]

    def validate(self, q_urdf: np.ndarray, where: str = "startup") -> bool:
        """Report whether the gate is already violated. NEVER raises --
        same reasoning as CapsuleGate.validate: the gate's job is to stop
        MOTION, not the program."""
        ok, dist, link = self.check(q_urdf)
        if not ok:
            print(f"\n[table gate] INSIDE THE MARGIN at {where}: "
                  f"{link} at {dist * 1e3:+.1f} mm above the table "
                  f"(table z={self.table_z * 1e3:+.1f} mm).\n"
                  f"  ESCAPE MODE: only motion that does not lower this "
                  f"link further will be sent, until it clears the margin. "
                  f"Teleop is running; nothing has crashed.\n"
                  + self.report(q_urdf, worst=5))
        return ok

    def largest_safe_fraction(self, q_prev: np.ndarray, q_target: np.ndarray,
                              n_samples: int = 16, refine: int = 16
                              ) -> Tuple[float, float, Optional[str]]:
        """Largest fraction of the step from q_prev toward q_target whose
        WHOLE swept segment stays clear of the table -- same swept-and-scale
        shape as CapsuleGate.largest_safe_fraction (see that docstring for
        why scaling beats refusing), minus the fine-tier escalation, which
        this gate never needs (capsule-vs-plane is already exact).

        ESCAPE MODE (2026-08-27).  When q_prev ITSELF is inside the margin,
        this used to return 0.0 -- which is a deadlock, not a safety
        property: every step is refused, including the ones that climb out,
        so the arm can never reach a state where motion is allowed again.
        Hardware found it: an arm lost torque, fell over the front edge of
        the table, and the gate then held all three arms forever at
        `right_gripper_base -307.2 mm`.  From inside the margin the rule
        becomes MONOTONE IMPROVEMENT -- send the longest prefix of the step
        whose worst link never reads lower than it does right now -- so
        motion out is allowed and motion further in is still refused.  It
        degenerates to the old behaviour exactly when nothing improves."""
        self.last_escape = False
        q_prev = np.asarray(q_prev, dtype=np.float32)
        q_target = np.asarray(q_target, dtype=np.float32)
        delta = q_target - q_prev
        if not np.any(np.abs(delta) > 1e-12):
            ok, dist, link = self.check(q_prev)
            self.last_escape = not ok
            return (1.0 if ok else 0.0), dist, link

        alphas = np.linspace(0.0, 1.0, int(n_samples) + 1, dtype=np.float32)
        cfgs = q_prev[None, :] + alphas[:, None] * delta[None, :]
        d = np.atleast_2d(np.asarray(self._dists(cfgs)))            # (n+1, links)
        slack = d - self._margin_per_kept[None, :]
        safe = slack.min(axis=1) >= 0.0
        which = slack.argmin(axis=1)
        per_sample = d[np.arange(len(d)), which]

        if bool(safe.all()):
            return 1.0, float(per_sample[-1]), None
        if not safe[0]:
            ## Already inside the margin -- monotone-improvement rule.  The
            ## comparison is against the START's own worst reading, so a
            ## step is admissible only while it makes nothing worse; the
            ## 1 um tolerance is float32 noise, not a budget to sink by.
            self.last_escape = True
            ## Two rules, because one is not enough and the strict
            ## elementwise version deadlocks all over again.
            ##
            ##  (a) NO NEW VIOLATIONS.  Anything clear at the start must
            ##      still be clear -- escaping one violation by creating a
            ##      second is not an escape.
            ##  (b) THE DEEPEST EXISTING VIOLATION MAY NOT GET DEEPER.
            ##      Reduced over the already-violated set rather than held
            ##      elementwise: with several links inside the margin at
            ##      once, "no violated links may worsen at all" is refused
            ##      by almost every direction, which is the deadlock this
            ##      whole branch exists to remove.  Trading depth among
            ##      already-violated links is bounded by the state we are
            ##      already in; creating a new one is not.
            viol = slack[0] < 0.0
            clear_ok = (slack[:, ~viol] >= -1e-6).all(axis=1) \
                if bool((~viol).any()) else np.ones(len(slack), dtype=bool)
            worst_viol = slack[:, viol].min(axis=1)
            viol_ok = worst_viol >= (float(worst_viol[0]) - 1e-6)
            improving = clear_ok & viol_ok
            k = (int(np.argmax(~improving)) if not bool(improving.all())
                 else int(improving.size))
            j = max(k - 1, 0)     # last index of the improving prefix
            return (float(alphas[j]), float(per_sample[j]),
                    self.link_names[int(which[j])])

        first_bad = int(np.argmax(~safe))
        lo_a, hi_a = float(alphas[first_bad - 1]), float(alphas[first_bad])
        fine = np.linspace(lo_a, hi_a, int(refine) + 2, dtype=np.float32)[1:-1]
        if fine.size:
            cf = q_prev[None, :] + fine[:, None] * delta[None, :]
            df = np.atleast_2d(np.asarray(self._dists(cf)))
            slack_f = df - self._margin_per_kept[None, :]
            ok_f = slack_f.min(axis=1) >= 0.0
            if bool(ok_f.any()):
                first_f = int(np.argmax(~ok_f)) if not bool(ok_f.all()) else int(ok_f.size)
                if first_f > 0:
                    lo_a = float(fine[first_f - 1])

        cl = np.atleast_2d(q_prev + lo_a * delta)
        dl = np.atleast_2d(np.asarray(self._dists(cl)))
        slack_l = dl - self._margin_per_kept[None, :]
        wl = int(slack_l.argmin(axis=1)[0])
        return lo_a, float(dl[0, wl]), self.link_names[wl]

    def report(self, q_urdf: np.ndarray, worst: int = 8) -> str:
        d = np.asarray(self._dists_one(np.asarray(q_urdf, dtype=np.float32)))
        order = np.argsort(d - self._margin_per_kept)[:worst]
        lines = [f"  table gate (margin {self.margin * 1e3:.0f} mm / "
                 f"{self.margin_finger * 1e3:.0f} mm fingertips, "
                 f"{self._n_links} links checked, table z="
                 f"{self.table_z * 1e3:+.1f} mm)."]
        for k in order:
            m = self._margin_per_kept[int(k)]
            flag = "  << inside margin" if d[k] < m else ""
            lines.append(f"    {d[k] * 1e3:+8.1f} mm  (margin {m * 1e3:.0f}) "
                         f" {self.link_names[int(k)]}{flag}")
        return "\n".join(lines)

    def describe(self) -> str:
        return (f"table gate ON: margin={self.margin * 1e3:.0f} mm "
                f"({self.margin_finger * 1e3:.0f} mm on finger links), "
                f"{self._n_links} links checked (of {len(self.coll.link_names)}; "
                f"{len(self.coll.link_names) - self._n_links} excluded as "
                f"kinematically height-invariant), table z="
                f"{self.table_z * 1e3:+.1f} mm (compile {self.compile_ms:.0f} ms). "
                f"Capsules are circumscribed, so every distance is a LOWER "
                f"BOUND on the true mesh-to-table distance -- a pass proves "
                f"clearance. Hardware-validated 2026-09-09 on the real "
                f"arms; margin not swept.")


def build_gate(robot, urdf) -> Optional[TableGate]:
    """The gate, or None when disabled -- never half-built."""
    if not GATE_ENABLED:
        print("[table_gate] DISABLED via GIAVA_TABLE_GATE=0 -- table safety "
              "rests on the solver's soft table cost alone (if enabled)")
        return None
    gate = TableGate(robot, urdf)
    print(f"[table_gate] {gate.describe()}")
    return gate
