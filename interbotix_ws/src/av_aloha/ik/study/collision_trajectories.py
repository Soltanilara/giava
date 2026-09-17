"""Collision-study Phase 8 — collision-stress trajectory subset.

Deliberately a SEPARATE module from the frozen suite (trajectories.py, hash
3815490046334af4): these are additive collision-specific tests, per the
study protocol.  Both are designed to the user's fine-bimanual requirement:
the *correct* behaviour is close-quarters tracking with **zero deviation** —
any deviation measures false repulsion.

bimanual_parallel
    Hands reorient to a downward grasp pose side by side at 0.16 m
    centre-to-centre (fork surfaces ~4–6 cm apart), translate together
    ±8 cm along y, then separate.  Truth: collision-free throughout with
    small positive clearance.  Categories: (2) near-boundary tracking.

bimanual_handoff
    The left hand holds a downward-pointing pose; the right hand approaches
    until the grippers are ~4 cm apart (as if handing an object over),
    holds 1.5 s, retreats.  Truth: collision-free, clearance dips to the
    margin's edge.  Categories: (2)/(4) — the solver may deviate only if the
    commanded pose truly enters the unsafe band.

Both validated by the multi-seed oracle (reachability) and mesh-truth spot
checks at the apex (see phase8 pilot output).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trajectories import (
    DT,
    Trajectory,
    _hold,
    grasp_orientation,
    minjerk,
    quat_slerp,
)


def bimanual_parallel(base_pos, base_wxyz, sep: float = 0.16,
                      travel: float = 0.08) -> Trajectory:
    """Side-by-side close-quarters translation, wide-transition version.

    At home the fingertips start only 14 mm apart, so a direct inward+rotating
    transition swings the fingers through each other's space (measured
    −23 mm).  The transition therefore goes WIDE first:
      A1 (1.5 s): home → (±0.24, −0.16, 0.42), slerp to the 20°-down grip
      A2 (1.0 s): wide → (±sep/2, −0.16, 0.40), orientation already set
      track (8 s): Δy = travel·sin(2π t/4)
      retreat mirrors A2 then A1."""
    d_hat = np.array([0.0, -np.sin(np.radians(20.0)), -np.cos(np.radians(20.0))])
    q_work = grasp_orientation(d_hat)

    n_a1 = int(round(1.5 / DT))
    n_a2 = int(round(1.0 / DT))
    n_track = int(round(8.0 / DT))
    steps = 2 * (n_a1 + n_a2) + n_track
    pos, wxyz = _hold(base_pos, base_wxyz, steps)

    s1 = minjerk(np.arange(n_a1) / (n_a1 - 1))
    s2 = minjerk(np.arange(n_a2) / (n_a2 - 1))
    for arm, sign in ((0, 1.0), (1, -1.0)):
        p_wide = np.array([sign * 0.24, -0.16, 0.42])
        p_goal = np.array([sign * sep / 2, -0.16, 0.40])
        k = 0
        for i in range(n_a1):
            pos[k + i, arm] = base_pos[arm] + s1[i] * (p_wide - base_pos[arm])
            wxyz[k + i, arm] = quat_slerp(base_wxyz[arm], q_work, float(s1[i]))
        k += n_a1
        for i in range(n_a2):
            pos[k + i, arm] = p_wide + s2[i] * (p_goal - p_wide)
            wxyz[k + i, arm] = q_work
        k += n_a2
        t = np.arange(n_track) * DT
        dy = travel * np.sin(2 * np.pi * t / 4.0)
        for i in range(n_track):
            pos[k + i, arm] = p_goal + np.array([0.0, dy[i], 0.0])
            wxyz[k + i, arm] = q_work
        k += n_track
        for i in range(n_a2):
            pos[k + i, arm] = p_goal + s2[i] * (p_wide - p_goal)
            wxyz[k + i, arm] = q_work
        k += n_a2
        for i in range(n_a1):
            pos[k + i, arm] = p_wide + s1[i] * (base_pos[arm] - p_wide)
            wxyz[k + i, arm] = quat_slerp(q_work, base_wxyz[arm], float(s1[i]))
    return Trajectory(
        name="bimanual_parallel",
        category="X",  # collision subset
        description=f"Hands work side by side at {sep*100:.0f} cm centres "
        f"(~4–6 cm surface gap), tracking ±{travel*100:.0f} cm together.",
        purpose="Fine-bimanual acceptance test: the correct collision-aware "
        "behaviour is ZERO deviation — deviation here measures false "
        "repulsion (user requirement).",
        expected_motion="Hands drop to a side-by-side downward pose, sway in "
        "y together, return.",
        failure_modes="Repulsive gradients pushing the hands apart before "
        "contact; tracking error growing with proximity; jerk from "
        "margin-boundary chatter.",
        definition=(
            f"approach 2.5 s min-jerk to (±{sep/2:.2f}, −0.10, 0.40), grip "
            "20°-from-vertical; Δy = 0.08·sin(2πt/4) for 8 s; return 2.5 s "
            f"({steps} steps)"
        ),
        dt=DT,
        positions=pos,
        wxyzs=wxyz,
        feasible="reachable; truly collision-free (mesh-verified at apex)",
    )


def bimanual_handoff(base_pos, base_wxyz, gap: float = 0.04) -> Trajectory:
    """Hand-over approach: right hand closes to `gap` from the left gripper.

    Left holds (0.06, −0.10, 0.40) pointing down; right approaches along −x
    from its side until fork surfaces are ~`gap` apart (centres at
    0.06 − 0.15 − gap), holds 1.5 s, retreats."""
    d_down = np.array([0.0, -np.sin(np.radians(20.0)), -np.cos(np.radians(20.0))])
    q_down = grasp_orientation(d_down)
    p_left = np.array([0.06, -0.10, 0.40])
    # fork half-width ≈ 0.075 each → centre distance for `gap` surface gap
    centre_gap = 2 * 0.075 + gap
    p_right_far = np.array([0.06 - centre_gap - 0.10, -0.10, 0.40])
    p_right_near = np.array([0.06 - centre_gap, -0.10, 0.40])

    n_in = int(round(2.5 / DT))
    n_appr = int(round(2.0 / DT))
    n_hold = int(round(1.5 / DT))
    steps = n_in + n_appr + n_hold + n_appr + n_in
    pos, wxyz = _hold(base_pos, base_wxyz, steps)

    s = minjerk(np.arange(n_in) / (n_in - 1))
    sa = minjerk(np.arange(n_appr) / (n_appr - 1))
    for i in range(n_in):  # both hands to working poses
        pos[i, 0] = base_pos[0] + s[i] * (p_left - base_pos[0])
        wxyz[i, 0] = quat_slerp(base_wxyz[0], q_down, float(s[i]))
        pos[i, 1] = base_pos[1] + s[i] * (p_right_far - base_pos[1])
        wxyz[i, 1] = quat_slerp(base_wxyz[1], q_down, float(s[i]))
    pos[n_in:, 0] = p_left
    wxyz[n_in:, 0] = q_down
    k = n_in
    for i in range(n_appr):  # approach
        pos[k + i, 1] = p_right_far + sa[i] * (p_right_near - p_right_far)
        wxyz[k + i, 1] = q_down
    k += n_appr
    pos[k:k + n_hold, 1] = p_right_near
    wxyz[k:k + n_hold, 1] = q_down
    k += n_hold
    for i in range(n_appr):  # retreat
        pos[k + i, 1] = p_right_near + sa[i] * (p_right_far - p_right_near)
        wxyz[k + i, 1] = q_down
    k += n_appr
    for i in range(n_in):  # both home
        pos[k + i, 0] = p_left + s[i] * (base_pos[0] - p_left)
        wxyz[k + i, 0] = quat_slerp(q_down, base_wxyz[0], float(s[i]))
        pos[k + i, 1] = p_right_far + s[i] * (base_pos[1] - p_right_far)
        wxyz[k + i, 1] = quat_slerp(q_down, base_wxyz[1], float(s[i]))
    return Trajectory(
        name="bimanual_handoff",
        category="X",
        description=f"Right hand approaches the holding left hand to a "
        f"{gap*100:.0f} cm fork gap, holds 1.5 s, retreats.",
        purpose="Margin-edge behaviour: at a 25 mm margin the commanded hold "
        "sits just outside/at the activation band — measures 'aware but not "
        "repelled' vs premature repulsion.",
        expected_motion="Left hand parks pointing down; right hand slides in "
        "until the grippers nearly touch, pauses, slides out.",
        failure_modes="Right hand refuses the last centimetres (over-"
        "repulsion); left hand pushed away; oscillation at the margin edge.",
        definition=(
            f"left holds (0.06, −0.10, 0.40); right: x from "
            f"{p_right_far[0]:.3f} to {p_right_near[0]:.3f} (surface gap "
            f"≈ {gap*100:.0f} cm), min-jerk 2 s in / 1.5 s hold / 2 s out "
            f"({steps} steps)"
        ),
        dt=DT,
        positions=pos,
        wxyzs=wxyz,
        feasible="reachable; clearance dips to ≈ margin edge by design",
    )


def build_collision_subset(base_pos, base_wxyz):
    """Three tests spanning the user's taxonomy.

    Measured commanded-path sphere clearances (baseline rollout, after the
    wide-transition + y=−0.16 fixes):
      bimanual_parallel (sep 0.20): min +17.1 mm, track med +27 — category 2
          (near boundary, truly clear; correct behaviour = zero deviation)
      bimanual_tight    (sep 0.16): min −5.4 mm throughout      — category 3
          (boundary-riding: perfect tracking implies mild model contact,
          ≈ −13 mm true; correct behaviour = deviate JUST enough to hold
          ≈0 clearance)
      bimanual_handoff  (gap 4 cm): min +4.1 mm                 — category 2
          (margin-edge hold; correct behaviour = zero deviation)
    """
    tight = bimanual_parallel(base_pos, base_wxyz, sep=0.16)
    tight.name = "bimanual_tight"
    tight.purpose = (
        "Boundary-riding test: the commanded track phase sits at −5.4 mm "
        "model clearance; the collision-aware solver must deviate just "
        "enough to hold ≈0 clearance — graceful minimal deviation, not "
        "repulsion."
    )
    tight.feasible = "commanded track model-penetrating by design (−5.4 mm)"
    return [
        bimanual_parallel(base_pos, base_wxyz, sep=0.20),
        tight,
        bimanual_handoff(base_pos, base_wxyz),
    ]
