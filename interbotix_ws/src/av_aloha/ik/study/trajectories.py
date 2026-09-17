"""Phase 2 — the standardized trajectory benchmark suite (rev 2.1, user-reviewed).

Rev 2.1 (2026-08-08, user-approved, pre-ablation): rot_{home,low,forward}
hold positions get a mirrored ±6 cm outward x-spread — baseline measurement
showed the fingertips (57 mm apart at home) swing into capsule contact during
±30° rotations, which would have confounded category C with collision
avoidance.  See rot_at(spread=...).

Every trajectory is defined analytically here, in world frame, at the
deployment control rate (50 Hz, `DT = 0.02 s` — matches
`data_col_config.control_dt`).  Targets are *commanded poses*, generated
independently of any solver.

Rev 2 changes after user review of rev 1 in Viser (2026-08-08):
- Translational trajectories are longer (15–18 s) and larger (±12 cm x/y,
  ±9 cm z; diag 13 cm; Lissajous 10/10/7 cm) — re-validated by the oracle.
- Three reach-and-grasp variants: top-front 35°-pitch (`teleop_grasp`),
  side approach = pitch to horizontal (`teleop_grasp_side`), and a combined
  pitch+yaw diagonal approach (`teleop_grasp_yaw`).
- A middle-arm mirror suite (`mid_*`): the camera arm performs the same
  motion families (translations, general translation, pure rotation,
  coupled approach, reach-limit) while the hands park out of the way.
  The camera has no gripper, so its "grasp" analog is a coupled
  look-at + approach toward the object (`mid_approach`).

Categories:
    A  pure translation        trans_{x,y,z}            (hands)
    B  general translation     trans_diag, trans_lissajous
    C  pure rotation           rot_{home,low,forward}
    D  teleop reach-and-grasp  teleop_grasp{,_side,_yaw}
    E  stress                  arms_converge, self_fold, reach_limit,
                               wrist_twist, near_singular, middle_gaze_sweep,
                               jitter_teleop
    M  middle-arm mirror       mid_trans_{x,y,z}, mid_trans_diag,
                               mid_trans_lissajous, mid_rot_{home,low},
                               mid_approach, mid_reach_limit

Conventions
-----------
- Arm order everywhere: (left, right, middle); positions (T,3,3), unit
  quaternions wxyz (T,3,4).
- Non-moving arms hold their home pose as actively tracked targets.
  For `mid_*` trajectories the hands first *park* out of the way
  (min-jerk, 2 s intro; `eval_start` marks the end of the intro so
  Phase 3 aggregates exclude it), then hold the park pose.
- Gripper frame: local +z = approach axis, ±x = finger opening.
  Camera frame: local +x = optical axis.
- s(u) = 10u³ − 15u⁴ + 6u⁵ (min-jerk; peak velocity 1.875·D/T for a
  segment of length D over T seconds).

Reachability of every trajectory is established by
`validate_trajectories.py` (multi-seed cold-start oracle) and recorded in
TRAJECTORIES.md.  Stress-test targets are *deliberately* infeasible where
documented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

DT = 0.02  # s — 50 Hz, matches deployment control_dt
NUM_TARGETS = 3
HANDS: Tuple[int, ...] = (0, 1)
MIDDLE: Tuple[int, ...] = (2,)

## Shoulder positions (world, at home; FK constants of giava.urdf — see
## robot_model.py).  reach_limit aims *away from the arm's own shoulder*:
## ±x "outward from the midline" would move *toward* the bases at x = ±0.469
## and stay comfortably reachable (verified by the oracle in rev 1).
## FROZEN with the suite (hash 3815490046334af4). These were the shoulder
## positions when the suite was defined; the URDF bases moved to +/-0.520 on
## 2026-08-19 (measured), but editing these would change every trajectory
## and invalidate every comparison ever run against the suite. They are
## world-space anchor points of a frozen benchmark, not live geometry.
SHOULDER_L = np.array([0.469, -0.019, 0.099])
SHOULDER_R = np.array([-0.469, -0.019, 0.099])
SHOULDER_M = np.array([0.0, 0.400, 0.089])

## Where the hands go during middle-arm trajectories: retreat toward the own
## base and down, clearing the central corridor the camera works in.
PARK_OFFSET = np.array([0.20, 0.0, -0.15])  # +x for left, −x for right
PARK_INTRO_S = 2.0


# --------------------------------------------------------------------------- #
# math helpers (pure numpy, explicit)
# --------------------------------------------------------------------------- #
def minjerk(u: np.ndarray) -> np.ndarray:
    """Minimum-jerk profile s(u) = 10u³ − 15u⁴ + 6u⁵, clipped to [0, 1]."""
    u = np.clip(u, 0.0, 1.0)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    return np.array([np.cos(angle / 2.0), *(np.sin(angle / 2.0) * axis)])


def quat_slerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    """Geodesic interpolation between wxyz quaternions (shortest arc)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    if d > 1.0 - 1e-10:
        out = a + u * (b - a)
        return out / np.linalg.norm(out)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    return (np.sin((1 - u) * th) * a + np.sin(u * th) * b) / np.sin(th)


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """wxyz quaternion from a rotation matrix (Shepperd's method)."""
    m = np.asarray(R, dtype=np.float64)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_local(q_base: np.ndarray, axis_local: np.ndarray, angle: float) -> np.ndarray:
    """R = R_base · exp(angle · axis_local): rotate about a base-local axis."""
    return quat_mul(q_base, quat_from_axis_angle(axis_local, angle))


def grasp_orientation(d_hat: np.ndarray) -> np.ndarray:
    """Gripper orientation with approach axis (local +z) = d̂ and the finger
    axis (local +x) horizontal: x̂ = normalize(d̂ × ẑ_world) (world x for a
    vertical approach); ŷ = d̂ × x̂ completes the right-handed frame."""
    d = np.asarray(d_hat, dtype=np.float64)
    d = d / np.linalg.norm(d)
    x = np.cross(d, np.array([0.0, 0.0, 1.0]))
    n = np.linalg.norm(x)
    x = np.array([1.0, 0.0, 0.0]) if n < 1e-9 else x / n
    y = np.cross(d, x)
    return quat_from_matrix(np.stack([x, y, d], axis=1))


# --------------------------------------------------------------------------- #
# container
# --------------------------------------------------------------------------- #
@dataclass
class Trajectory:
    name: str
    category: str  # "A".."E", "M"
    description: str
    purpose: str
    expected_motion: str
    failure_modes: str
    definition: str  # explicit math, human-readable
    dt: float
    positions: np.ndarray  # (T, 3, 3)
    wxyzs: np.ndarray  # (T, 3, 4)
    markers: Dict[str, np.ndarray] = field(default_factory=dict)
    feasible: str = "reachable"
    eval_start: int = 0  # metrics aggregate from this step (skips park intro)

    def __len__(self) -> int:
        return self.positions.shape[0]

    @property
    def duration(self) -> float:
        return len(self) * self.dt


def _hold(base_pos: np.ndarray, base_wxyz: np.ndarray, steps: int):
    pos = np.repeat(base_pos[None], steps, axis=0).astype(np.float64)
    wxyz = np.repeat(base_wxyz[None], steps, axis=0).astype(np.float64)
    return pos, wxyz


def _with_park_intro(base_pos, base_wxyz, motion_steps: int):
    """Arrays for a middle-arm trajectory: 2 s min-jerk hand-park intro, then
    `motion_steps` of holds (caller fills in the middle arm's motion).

    Returns (pos, wxyz, intro_steps).  Hands end at home + (±0.20, 0, −0.15),
    orientation unchanged."""
    intro = int(round(PARK_INTRO_S / DT))
    steps = intro + motion_steps
    pos, wxyz = _hold(base_pos, base_wxyz, steps)
    park_l = base_pos[0] + PARK_OFFSET * np.array([1.0, 1.0, 1.0])
    park_r = base_pos[1] + PARK_OFFSET * np.array([-1.0, 1.0, 1.0])
    s = minjerk(np.arange(intro) / (intro - 1))
    pos[:intro, 0] = base_pos[0] + np.outer(s, park_l - base_pos[0])
    pos[:intro, 1] = base_pos[1] + np.outer(s, park_r - base_pos[1])
    pos[intro:, 0] = park_l
    pos[intro:, 1] = park_r
    return pos, wxyz, intro


# --------------------------------------------------------------------------- #
# A / M — pure translation
# --------------------------------------------------------------------------- #
def trans_axis(base_pos, base_wxyz, axis: int, amp: float, freq: float = 0.2,
               duration: float = 15.0, arms: Sequence[int] = HANDS) -> Trajectory:
    """p_a(t) = p_home_a + A·sin(2π f t)·ê_axis for the moving arms;
    everything else holds (hands park first when the middle arm moves)."""
    mid = tuple(arms) == MIDDLE
    steps = int(round(duration / DT))
    if mid:
        pos, wxyz, intro = _with_park_intro(base_pos, base_wxyz, steps)
    else:
        pos, wxyz = _hold(base_pos, base_wxyz, steps)
        intro = 0
    t = np.arange(steps) * DT
    offset = amp * np.sin(2 * np.pi * freq * t)
    for a in arms:
        pos[intro:, a, axis] += offset
    name = ("mid_" if mid else "") + "trans_" + "xyz"[axis]
    who = "camera" if mid else "both hands"
    vpeak = 2 * np.pi * freq * amp
    return Trajectory(
        name=name,
        category="M" if mid else "A",
        description=f"{who} oscillate ±{amp*100:.0f} cm along {'xyz'[axis]} "
        f"({duration:.0f} s, {freq} Hz, peak {vpeak:.2f} m/s).",
        purpose="Isolate translational tracking along a single world axis"
        + (" for the 7-DoF camera arm" if mid else "")
        + f"; peak speed 2πfA = {vpeak:.2f} m/s, peak accel "
        f"{(2*np.pi*freq)**2*amp:.2f} m/s².",
        expected_motion=(
            ("Hands park outward-down, then the camera translates; " if mid else
             "Hands translate together (constant separation); ")
            + "orientations fixed."
        ),
        failure_modes="Lag at sine peaks; orientation drift while translating; "
        + ("null-space drift of the extra DoF." if mid else "left/right asymmetry."),
        definition=(
            f"p_a(t) = p_home_a + {amp}·sin(2π·{freq}·t)·ê_{'xyz'[axis]}, "
            f"a ∈ {'{middle}' if mid else '{left,right}'}; "
            + (f"hands park at home + (±{PARK_OFFSET[0]}, {PARK_OFFSET[1]}, "
               f"{PARK_OFFSET[2]}) during a {PARK_INTRO_S:.0f} s min-jerk intro; "
               if mid else "")
            + f"{duration:.0f} s @ {1/DT:.0f} Hz ({steps + intro} steps)"
        ),
        dt=DT,
        positions=pos,
        wxyzs=wxyz,
        eval_start=intro,
    )


# --------------------------------------------------------------------------- #
# B / M — general translation
# --------------------------------------------------------------------------- #
def trans_diag(base_pos, base_wxyz, amp: float = 0.13, half_period: float = 2.5,
               cycles: int = 3, arms: Sequence[int] = HANDS) -> Trajectory:
    """Straight-line diagonal along d̂ = (1,1,1)/√3, min-jerk there-and-back."""
    mid = tuple(arms) == MIDDLE
    d = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    steps_half = int(round(half_period / DT))
    s_up = minjerk(np.arange(steps_half) / (steps_half - 1))
    s = np.tile(np.concatenate([s_up, s_up[::-1]]), cycles)
    steps = len(s)
    if mid:
        pos, wxyz, intro = _with_park_intro(base_pos, base_wxyz, steps)
    else:
        pos, wxyz = _hold(base_pos, base_wxyz, steps)
        intro = 0
    for a in arms:
        pos[intro:, a, :] += np.outer(amp * s, d)
    vpeak = 1.875 * amp / half_period
    name = ("mid_" if mid else "") + "trans_diag"
    return Trajectory(
        name=name,
        category="M" if mid else "B",
        description=f"Straight diagonal along (1,1,1)/√3, ±{amp*100:.0f} cm, "
        f"min-jerk, peak {vpeak:.2f} m/s.",
        purpose="Tracking when all three position components change "
        "proportionally (motion not aligned with any axis).",
        expected_motion=("Camera moves" if mid else "Hands move together")
        + " diagonally up-forward-left and back, orientation fixed.",
        failure_modes="Curved actual path when the commanded path is straight; "
        "axis-dependent error anisotropy.",
        definition=(
            f"p_a(t) = p_home_a + {amp}·s(t)·(1,1,1)/√3, min-jerk there-and-back, "
            f"half-period {half_period} s × {cycles} cycles "
            f"({steps + intro} steps); peak vel 1.875·A/T = {vpeak:.2f} m/s"
        ),
        dt=DT,
        positions=pos,
        wxyzs=wxyz,
        eval_start=intro,
    )


def trans_lissajous(base_pos, base_wxyz, duration: float = 18.0,
                    amps: Tuple[float, float, float] = (0.10, 0.10, 0.07),
                    arms: Sequence[int] = HANDS) -> Trajectory:
    """3-D Lissajous — all axes change simultaneously at incommensurate rates."""
    mid = tuple(arms) == MIDDLE
    steps = int(round(duration / DT))
    if mid:
        pos, wxyz, intro = _with_park_intro(base_pos, base_wxyz, steps)
    else:
        pos, wxyz = _hold(base_pos, base_wxyz, steps)
        intro = 0
    t = np.arange(steps) * DT
    ax, ay, az = amps
    fx, fy, fz = 0.20, 0.10, 0.26
    dxyz = np.stack(
        [
            ax * np.sin(2 * np.pi * fx * t),
            ay * np.sin(2 * np.pi * fy * t),
            az * np.sin(2 * np.pi * fz * t),
        ],
        axis=1,
    )
    for a in arms:
        pos[intro:, a, :] += dxyz
    name = ("mid_" if mid else "") + "trans_lissajous"
    return Trajectory(
        name=name,
        category="M" if mid else "B",
        description=f"3-D Lissajous ({ax*100:.0f}/{ay*100:.0f}/{az*100:.0f} cm "
        f"at {fx}/{fy}/{fz} Hz), {duration:.0f} s.",
        purpose="General simultaneous XYZ motion with continuously rotating "
        "velocity direction — no axis is ever privileged.",
        expected_motion="Smooth wandering figure-eight-like path, fixed orientation.",
        failure_modes="Error spikes when the velocity direction rotates fastest; "
        "coupling of position error into orientation error.",
        definition=(
            f"Δp(t) = ({ax}·sin(2π·{fx}t), {ay}·sin(2π·{fy}t), "
            f"{az}·sin(2π·{fz}t)); peak per-axis speed "
            f"{2*np.pi*max(ax*fx, ay*fy, az*fz):.2f} m/s; "
            f"{duration:.0f} s ({steps + intro} steps)"
        ),
        dt=DT,
        positions=pos,
        wxyzs=wxyz,
        eval_start=intro,
    )


# --------------------------------------------------------------------------- #
# C / M — pure rotation at several workspace positions
# --------------------------------------------------------------------------- #
def rot_at(base_pos, base_wxyz, name: str, pos_offset: np.ndarray,
           theta_deg: float = 30.0, per_axis: float = 4.0,
           arms: Sequence[int] = HANDS, spread: float = 0.06) -> Trajectory:
    """Hold position at home + offset; oscillate orientation ±θ about each
    end-effector-local axis in sequence.

    Hands: x = finger axis, y = palm normal, z = approach axis.
    Camera: x = optical axis (roll), y, z (transverse).
    R_a(t) = R_home_a · exp(θ·sin(2π t′/T_ax)·ê_j), j cycling x, y, z.

    `spread` (rev 2.1): mirrored outward x-offset added to each *hand*'s held
    position (+x left, −x right).  At home the fingertips are only 57 mm
    apart, and ±30° rotations were measured (baseline rev 2) to swing the
    finger capsules into contact (−19 to −45 mm clearance), which would have
    confounded category C with collision avoidance once the collision ablation
    runs.  6 cm per hand keeps every rotation clear of contact; near-contact
    behaviour stays covered by arms_converge / self_fold / the grasp family.
    User-approved amendment before any ablation ran (2026-08-08)."""
    mid = tuple(arms) == MIDDLE
    steps_ax = int(round(per_axis / DT))
    steps = 3 * steps_ax
    if mid:
        pos, wxyz, intro = _with_park_intro(base_pos, base_wxyz, steps)
    else:
        pos, wxyz = _hold(base_pos, base_wxyz, steps)
        intro = 0
    for a in arms:
        pos[intro:, a, :] += pos_offset
    if not mid and spread:
        pos[intro:, 0, 0] += spread  # left outward (+x)
        pos[intro:, 1, 0] -= spread  # right outward (−x)
    th = np.radians(theta_deg)
    axes = (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))
    for j, axis in enumerate(axes):
        for k in range(steps_ax):
            ang = th * np.sin(2 * np.pi * k / steps_ax)
            i = intro + j * steps_ax + k
            for a in arms:
                wxyz[i, a] = rot_local(base_wxyz[a], axis, ang)
    where = "home" if not np.any(pos_offset) else f"home + {pos_offset.tolist()}"
    if not mid and spread:
        where += f" ± {spread} m outward (x)"
    return Trajectory(
        name=name,
        category="M" if mid else "C",
        description=f"Position fixed at {where}; ±{theta_deg:.0f}° about each "
        "EE-local axis in sequence"
        + (" (camera: roll about optical axis, then transverse axes)"
           if mid else "") + ".",
        purpose="Isolate orientation tracking and its dependence on workspace "
        "position (compare the rot_* / mid_rot_* variants).",
        expected_motion=("Camera pivots in place" if mid else
                         "Grippers pivot in place") + ": one local axis at a time.",
        failure_modes="Position drift during pure rotation (weight coupling); "
        "wrist-limit saturation; posture-dependent error.",
        definition=(
            f"p_a(t) = p_home_a + {pos_offset.tolist()};  "
            f"R_a(t) = R_home_a · exp({theta_deg}°·sin(2π t′/{per_axis} s)·ê_j), "
            f"j = x,y,z sequentially ({steps + intro} steps); peak angular rate "
            f"{np.radians(theta_deg) * 2 * np.pi / per_axis:.2f} rad/s"
        ),
        dt=DT,
        positions=pos,
        wxyzs=wxyz,
        eval_start=intro,
    )


# --------------------------------------------------------------------------- #
# D — teleoperation-like reach and grasp (three approach styles)
# --------------------------------------------------------------------------- #
def _grasp_traj(base_pos, base_wxyz, name: str, d_hat_l: np.ndarray,
                d_hat_r: np.ndarray, style: str,
                noise_sigma_pos: float = 0.0, noise_sigma_rot_deg: float = 0.0,
                seed: int = 0) -> Trajectory:
    """Shared machinery for the reach-and-grasp family.

    Keyframes (per hand):
      position: t=0: home →(min-jerk 2.5 s) obj − 0.10·d̂ →(min-jerk 2 s)
                obj − 0.02·d̂ → hold to 6 s.
      orientation: slerp(R_home → R_grasp) time-warped by min-jerk over
                t ∈ [1.2, 4.5] s — overlapping the position motion, so
                position and orientation change *simultaneously*.
      R_grasp: approach axis ẑ = d̂; finger axis = horizontal ⟂ d̂.
    """
    obj_l = np.array([0.20, -0.12, 0.12])
    obj_r = np.array([-0.20, -0.12, 0.12])
    q_grasp_l = grasp_orientation(d_hat_l)
    q_grasp_r = grasp_orientation(d_hat_r)

    duration = 6.0
    steps = int(round(duration / DT))
    t = np.arange(steps) * DT
    pos, wxyz = _hold(base_pos, base_wxyz, steps)

    for arm, obj, d_hat, q_g in (
        (0, obj_l, d_hat_l, q_grasp_l),
        (1, obj_r, d_hat_r, q_grasp_r),
    ):
        p0 = base_pos[arm]
        p_pre = obj - 0.10 * d_hat
        p_grasp = obj - 0.02 * d_hat
        for i, ti in enumerate(t):
            if ti < 2.5:
                p = p0 + (p_pre - p0) * minjerk(ti / 2.5)
            elif ti < 4.5:
                p = p_pre + (p_grasp - p_pre) * minjerk((ti - 2.5) / 2.0)
            else:
                p = p_grasp
            pos[i, arm] = p
            u = minjerk((ti - 1.2) / 3.3)
            wxyz[i, arm] = quat_slerp(base_wxyz[arm], q_g, float(u))

    fdef = (
        "objects at (±0.20, −0.12, 0.12); "
        f"d̂_left = {np.round(d_hat_l, 3).tolist()}, "
        f"d̂_right = {np.round(d_hat_r, 3).tolist()}; "
        "pos: home →(min-jerk 2.5 s) obj−0.10d̂ →(min-jerk 2 s) obj−0.02d̂ → hold; "
        "ori: slerp(R_home → R_grasp) min-jerk-warped over t∈[1.2,4.5] s; "
        "R_grasp: ẑ=d̂, x̂ = horizontal ⟂ d̂; 6 s, 300 steps"
    )
    traj = Trajectory(
        name=name,
        category="D",
        description=f"Teleop-like reach-and-grasp, {style} approach.",
        purpose="Measure the practical position-vs-orientation tradeoff during "
        "realistic coupled motion (Phase 5 weighting testbed); the three "
        "approach styles vary the orientation demand.",
        expected_motion=f"Hands sweep to the objects while the grippers "
        f"reorient into the {style} grasp; 1.5 s settle at the grasp pose.",
        failure_modes="Position error spike when orientation starts moving "
        "(weight competition); wrist reconfiguration mid-reach; grasp-pose "
        "orientation bias.",
        definition=fdef,
        dt=DT,
        positions=pos,
        wxyzs=wxyz,
        markers={"object_left": obj_l, "object_right": obj_r},
    )
    if noise_sigma_pos > 0.0 or noise_sigma_rot_deg > 0.0:
        rng = np.random.default_rng(seed)
        traj.positions = traj.positions + rng.normal(
            0.0, noise_sigma_pos, traj.positions.shape
        )
        for i in range(len(traj)):
            for arm in HANDS:
                axis = rng.normal(size=3)
                ang = rng.normal(0.0, np.radians(noise_sigma_rot_deg))
                traj.wxyzs[i, arm] = quat_mul(
                    traj.wxyzs[i, arm], quat_from_axis_angle(axis, ang)
                )
        traj.category = "E"
        traj.description += (
            f" + tracker noise (σ_pos {noise_sigma_pos*1e3:.0f} mm, "
            f"σ_rot {noise_sigma_rot_deg:.0f}°, seed {seed})."
        )
        traj.purpose = (
            "Stress: solver behaviour under VR-tracker-like noise — jerk "
            "amplification without smoothing, over-filtering with it."
        )
        traj.definition = fdef + (
            f"; + iid noise σ_pos={noise_sigma_pos}, σ_rot={noise_sigma_rot_deg}°"
        )
    return traj


def teleop_grasp(base_pos, base_wxyz, name: str = "teleop_grasp",
                 **noise) -> Trajectory:
    """Top-front grasp: approach tilted 35° from vertical toward −y."""
    d = np.array([0.0, -np.sin(np.radians(35.0)), -np.cos(np.radians(35.0))])
    return _grasp_traj(base_pos, base_wxyz, name, d, d, "top-front 35°-pitch",
                       **noise)


def teleop_grasp_side(base_pos, base_wxyz) -> Trajectory:
    """Side grasp: pitch increased all the way to horizontal — approach along
    −y, gripper level with the object, fingers opening horizontally."""
    d = np.array([0.0, -1.0, 0.0])
    return _grasp_traj(base_pos, base_wxyz, "teleop_grasp_side", d, d,
                       "horizontal side (pitch 90°)")


def teleop_grasp_yaw(base_pos, base_wxyz) -> Trajectory:
    """Diagonal grasp: pitch 55° from vertical *and* yaw 40° — each hand
    approaches its object from its own outer side (mirrored)."""
    pitch = np.radians(55.0)
    yaw = np.radians(40.0)
    d0 = np.array([0.0, -np.sin(pitch), -np.cos(pitch)])
    c, s = np.cos(-yaw), np.sin(-yaw)
    Rz_neg = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    d_l = Rz_neg @ d0  # left approaches from outside-left
    d_r = Rz_neg.T @ d0  # mirrored for the right hand
    return _grasp_traj(base_pos, base_wxyz, "teleop_grasp_yaw", d_l, d_r,
                       "55°-pitch + 40°-yaw diagonal")


# --------------------------------------------------------------------------- #
# M — middle-arm coupled approach + reach limit
# --------------------------------------------------------------------------- #
def mid_approach(base_pos, base_wxyz) -> Trajectory:
    """Camera-arm analog of the grasp: coupled look-at + approach.

    The camera moves from home toward an inspection point (stopping 0.15 m
    away) while its gaze pans from the home gaze point onto that point —
    position and orientation change simultaneously.  (The camera has no
    gripper; 'grasp' translates to 'move in and look'.)

    position runs t∈[0, 5] s; gaze interpolation runs t∈[1.0, 4.5] s."""
    R_home_cam = quat_to_matrix(base_wxyz[2])
    optical = R_home_cam[:, 0]
    C0 = base_pos[2] + 0.25 * optical  # home gaze point
    obj = np.array([0.0, -0.18, 0.12])
    to_obj = obj - base_pos[2]
    p_final = obj - 0.15 * to_obj / np.linalg.norm(to_obj)

    duration = 6.0
    steps = int(round(duration / DT))
    t = np.arange(steps) * DT
    pos, wxyz, intro = _with_park_intro(base_pos, base_wxyz, steps)
    for i, ti in enumerate(t):
        u_p = minjerk(ti / 5.0)
        p = base_pos[2] + (p_final - base_pos[2]) * u_p
        u_g = minjerk((ti - 1.0) / 3.5)
        gaze = C0 + (obj - C0) * u_g
        x_cam = gaze - p
        x_cam = x_cam / np.linalg.norm(x_cam)
        y_cam = np.cross(np.array([0.0, 0, 1.0]), x_cam)
        y_cam = y_cam / np.linalg.norm(y_cam)
        z_cam = np.cross(x_cam, y_cam)
        pos[intro + i, 2] = p
        wxyz[intro + i, 2] = quat_from_matrix(np.stack([x_cam, y_cam, z_cam], axis=1))
    return Trajectory(
        name="mid_approach",
        category="M",
        description="Camera zooms from home to 15 cm from an inspection point "
        "while panning its gaze onto it (coupled pos+ori).",
        purpose="Middle-arm analog of the reach-and-grasp: coupled "
        "position/orientation tracking for the 7-DoF camera arm.",
        expected_motion="Hands park; camera leans in toward the point between "
        "the hand workspaces, converging its view onto it.",
        failure_modes="Gaze error during the coupled phase; elbow/null-space "
        "swing; wrist limits near the close viewpoint.",
        definition=(
            f"p(t): home → obj − 0.15·(obj − p_home)/‖·‖, obj = {obj.tolist()}, "
            "min-jerk over [0,5] s; R(t) = look-at(gaze(t)), gaze: C0 → obj "
            "min-jerk over [1,4.5] s, C0 = p_home + 0.25·x̂_optical; "
            f"hold to 6 s ({steps + intro} steps incl. 2 s park intro)"
        ),
        dt=DT,
        positions=pos,
        wxyzs=wxyz,
        markers={"inspect_point": obj},
        eval_start=intro,
    )


def mid_reach_limit(base_pos, base_wxyz, overreach: float = 0.40) -> Trajectory:
    """Camera arm driven past its reachable boundary along the shoulder→EE ray
    and back (same protocol as the hands' reach_limit)."""
    u = (base_pos[2] - SHOULDER_M) / np.linalg.norm(base_pos[2] - SHOULDER_M)
    d = overreach * u
    seg = lambda n: minjerk(np.arange(n) / (n - 1))  # noqa: E731
    n4, n2 = int(round(4.0 / DT)), int(round(2.0 / DT))
    s = np.concatenate([seg(n4), np.ones(n2), seg(n4)[::-1], np.zeros(n2)])
    steps = len(s)
    pos, wxyz, intro = _with_park_intro(base_pos, base_wxyz, steps)
    pos[intro:, 2, :] = base_pos[2] + np.outer(s, d)
    return Trajectory(
        name="mid_reach_limit",
        category="M",
        description=f"Camera commanded {overreach*100:.0f} cm along its "
        "shoulder→EE ray — past max reach — then back.",
        purpose="Workspace-boundary behaviour and recovery for the camera arm.",
        expected_motion="Camera stretches down-forward to full extension, "
        "holds, returns.",
        failure_modes="Stuck stretched configuration on return (local-minimum "
        "trap); extension singularity of the 7-DoF chain.",
        definition=(
            f"Δp_mid = s(t)·{overreach}·û, û = {np.round(u, 3).tolist()}; "
            "s: min-jerk 0→1 (4 s), hold (2 s), 1→0 (4 s), hold (2 s) "
            f"({steps + intro} steps incl. park intro); orientation holds home"
        ),
        dt=DT,
        positions=pos,
        wxyzs=wxyz,
        feasible="deliberately-infeasible during the hold",
        eval_start=intro,
    )


# --------------------------------------------------------------------------- #
# E — stress tests (hands unless noted)
# --------------------------------------------------------------------------- #
def arms_converge(base_pos, base_wxyz, cross: float = 0.032,
                  stagger: float = 0.03) -> Trajectory:
    """Mirrored inward sweep: each hand crosses `cross` m past the midline
    (targets overlap → physically infeasible), ±`stagger` y offset so the two
    targets are never the identical point.  In 5 s, hold 1 s, out 5 s."""
    steps_in = int(round(5.0 / DT))
    steps_hold = int(round(1.0 / DT))
    s = np.concatenate(
        [
            minjerk(np.arange(steps_in) / (steps_in - 1)),
            np.ones(steps_hold),
            minjerk(np.arange(steps_in) / (steps_in - 1))[::-1],
        ]
    )
    steps = len(s)
    pos, wxyz = _hold(base_pos, base_wxyz, steps)
    sweep_l = base_pos[0, 0] + cross
    sweep_r = -base_pos[1, 0] + cross
    pos[:, 0, 0] = base_pos[0, 0] - sweep_l * s
    pos[:, 1, 0] = base_pos[1, 0] + sweep_r * s
    pos[:, 0, 1] = base_pos[0, 1] + stagger * s
    pos[:, 1, 1] = base_pos[1, 1] - stagger * s
    return Trajectory(
        name="arms_converge",
        category="E",
        description=f"Hands sweep toward and {cross*100:.0f} cm past each other "
        f"(±{stagger*100:.0f} cm y-stagger).",
        purpose="Two arms approaching one another / arm-arm collision: what "
        "does the solver do when targets demand interpenetration?",
        expected_motion="Hands converge at the midline, targets cross, part.",
        failure_modes="Physical interpenetration (baseline has no collision "
        "term); oscillation between competing solutions; recovery transient.",
        definition=(
            f"left x: {base_pos[0,0]:.3f} → {-cross:.3f}, right mirrored, "
            f"min-jerk in 5 s, hold 1 s, out 5 s ({steps} steps); "
            f"y staggered ±{stagger}·s(t); z, orientations, middle hold"
        ),
        dt=DT,
        positions=pos,
        wxyzs=wxyz,
        feasible="kinematically reachable per-arm; physically overlapping at "
        "apex (grippers would interpenetrate — invisible to the kinematic "
        "oracle, measured by the collision metric)",
    )


def self_fold(base_pos, base_wxyz) -> Trajectory:
    """Right arm folds back toward its own base column; left+middle hold."""
    goal = np.array([-0.42, -0.019, 0.20])
    steps_half = int(round(6.0 / DT))
    s = np.concatenate(
        [
            minjerk(np.arange(steps_half) / (steps_half - 1)),
            minjerk(np.arange(steps_half) / (steps_half - 1))[::-1],
        ]
    )
    steps = len(s)
    pos, wxyz = _hold(base_pos, base_wxyz, steps)
    pos[:, 1, :] = base_pos[1] + np.outer(s, goal - base_pos[1])
    return Trajectory(
        name="self_fold",
        category="E",
        description="Right hand target dives to 5 cm from its own base column.",
        purpose="Arm folding toward itself: self-collision risk with its own "
        "base/shoulder, elbow near limits, wrist flips.",
        expected_motion="Right arm folds down/backwards toward its mount, returns.",
        failure_modes="Forearm–base interpenetration; elbow-limit saturation; "
        "configuration flip on the way back out.",
        definition=(
            f"p_right(t) = p_home_r + s(t)·({goal.tolist()} − p_home_r), "
            f"min-jerk in 6 s / out 6 s ({steps} steps); orientation holds home"
        ),
        dt=DT,
        positions=pos,
        wxyzs=wxyz,
        feasible="partially (end point near self-collision; worst sampled "
        "waypoint ~20–30 mm unreachable by the oracle, seed-dependent — intended)",
    )


def reach_limit(base_pos, base_wxyz, overreach: float = 0.35) -> Trajectory:
    """Both hands driven past the reachable boundary along the shoulder→EE ray
    (home reach 0.597 m of measured 0.758 m max; +0.35 m demands ~0.95 m).

    min-jerk out 4 s → hold 2 s (infeasible) → back 4 s → settle 2 s.
    The return leg probes the warm-started local solver's recovery from the
    stretched configuration (the deployed solver's documented trap)."""
    u_l = (base_pos[0] - SHOULDER_L) / np.linalg.norm(base_pos[0] - SHOULDER_L)
    u_r = (base_pos[1] - SHOULDER_R) / np.linalg.norm(base_pos[1] - SHOULDER_R)
    seg = lambda n: minjerk(np.arange(n) / (n - 1))  # noqa: E731
    n4, n2 = int(round(4.0 / DT)), int(round(2.0 / DT))
    s = np.concatenate([seg(n4), np.ones(n2), seg(n4)[::-1], np.zeros(n2)])
    steps = len(s)
    pos, wxyz = _hold(base_pos, base_wxyz, steps)
    pos[:, 0, :] = base_pos[0] + np.outer(s, overreach * u_l)
    pos[:, 1, :] = base_pos[1] + np.outer(s, overreach * u_r)
    return Trajectory(
        name="reach_limit",
        category="E",
        description=f"Hands commanded {overreach*100:.0f} cm along the "
        "shoulder→EE ray — past max reach — then back home.",
        purpose="Workspace-boundary behaviour and, critically, *recovery*: "
        "does the solver return cleanly after chasing an unreachable target?",
        expected_motion="Arms stretch up-inward to full extension, hold, return.",
        failure_modes="Stuck stretched configuration on the return leg "
        "(local-minimum trap); joint-limit dwell; extension singularity "
        "(doubles as the low-manipulability stress).",
        definition=(
            f"Δp_a = s(t)·{overreach}·û_a, û_a = (p_home_a − p_shoulder_a)/‖·‖ "
            f"≈ {np.round(u_l, 3).tolist()} (left; right mirrored); "
            "s: min-jerk 0→1 (4 s), hold (2 s), 1→0 (4 s), hold (2 s); "
            f"{steps} steps; orientations, middle hold"
        ),
        dt=DT,
        positions=pos,
        wxyzs=wxyz,
        feasible="deliberately-infeasible during the hold (worst sampled "
        "waypoint 211 mm unreachable)",
    )


def wrist_twist(base_pos, base_wxyz, max_deg: float = 200.0) -> Trajectory:
    """Roll about the gripper approach axis: 0 → +200° (3 s) → −200° (6 s)
    → 0 (3 s), min-jerk.

    Every commanded pose is individually reachable (+200° ≡ −160° as an
    absolute orientation, inside the ±180° wrist_rotate range).  The stress is
    the *path*: a warm-started tracker winds wrist_rotate toward +180°,
    saturates, then must unwind ~320° — a guaranteed configuration
    discontinuity."""
    th = np.radians(max_deg)
    seg = lambda n: minjerk(np.arange(n) / (n - 1))  # noqa: E731
    n3, n6 = int(round(3.0 / DT)), int(round(6.0 / DT))
    ang = np.concatenate([th * seg(n3), th - 2 * th * seg(n6), -th + th * seg(n3)])
    steps = len(ang)
    pos, wxyz = _hold(base_pos, base_wxyz, steps)
    for i in range(steps):
        for arm in HANDS:
            wxyz[i, arm] = rot_local(base_wxyz[arm], np.array([0.0, 0, 1.0]), ang[i])
    return Trajectory(
        name="wrist_twist",
        category="E",
        description=f"±{max_deg:.0f}° roll about the approach axis, positions fixed.",
        purpose="Joint-limit stress: the commanded roll path exceeds the "
        "wrist_rotate range, forcing limit dwell then a large unwind.",
        expected_motion="Grippers spin about their pointing axis, stall near "
        "±180°, and unwind.",
        failure_modes="Limit dwell with growing orientation error near +180°; "
        "a ~320° unwind jump (configuration discontinuity); position "
        "disturbance during the flip.",
        definition=(
            f"R_a(t) = R_home_a · exp(θ(t)·ẑ_local); θ: 0→+{max_deg}° (3 s) "
            f"→ −{max_deg}° (6 s) → 0 (3 s), min-jerk segments ({steps} steps)"
        ),
        dt=DT,
        positions=pos,
        wxyzs=wxyz,
        feasible="every pose reachable; the path forces a ~320° unwind at ±180°",
    )


def near_singular(base_pos, base_wxyz) -> Trajectory:
    """Targets pass directly above each arm's base column (waist-axis
    singularity: x-y position becomes insensitive to the waist joint)."""
    goal_l = np.array([0.469, -0.019, 0.55])
    goal_r = np.array([-0.469, -0.019, 0.55])
    steps_half = int(round(5.0 / DT))
    s = np.concatenate(
        [
            minjerk(np.arange(steps_half) / (steps_half - 1)),
            np.ones(int(round(2.0 / DT))),
            minjerk(np.arange(steps_half) / (steps_half - 1))[::-1],
        ]
    )
    steps = len(s)
    pos, wxyz = _hold(base_pos, base_wxyz, steps)
    pos[:, 0, :] = base_pos[0] + np.outer(s, goal_l - base_pos[0])
    pos[:, 1, :] = base_pos[1] + np.outer(s, goal_r - base_pos[1])
    return Trajectory(
        name="near_singular",
        category="E",
        description="Targets sweep to points directly above each arm's own "
        "base (waist singularity) and hold 2 s.",
        purpose="Poor-manipulability region: on the waist axis the Jacobian "
        "loses rank.  The candidate showcase for the manipulability residual.",
        expected_motion="Arms rear up over their mounts, hold, return.",
        failure_modes="Waist indeterminacy near the axis; large condition "
        "number; erratic wrist compensation; slow convergence.",
        definition=(
            f"p_a(t): home → 0.55 m above own base ({goal_l.tolist()} / "
            "mirrored), min-jerk 5 s, hold 2 s, back 5 s; orientation holds "
            f"home ({steps} steps)"
        ),
        dt=DT,
        positions=pos,
        wxyzs=wxyz,
        feasible="reachable (position and orientation, oracle-verified); the "
        "stress is conditioning, not reachability",
    )


def middle_gaze_sweep(base_pos, base_wxyz, sweep_deg: float = 35.0) -> Trajectory:
    """Camera arm orbits its home gaze point; hands hold home (this is the
    original rev-1 camera trajectory — kept alongside the mid_* mirror suite
    because the hands-present case stresses the coupled problem differently).

    C = p_cam_home + 0.25 m along the home optical axis;
    p(t) = C + Rz(35°·sin(2πt/12))·(p_cam_home − C); R(t) = look-at(C)."""
    R_home_cam = quat_to_matrix(base_wxyz[2])
    optical = R_home_cam[:, 0]
    C = base_pos[2] + 0.25 * optical
    u0 = base_pos[2] - C
    duration = 12.0
    steps = int(round(duration / DT))
    t = np.arange(steps) * DT
    phi = np.radians(sweep_deg) * np.sin(2 * np.pi * t / duration)
    pos, wxyz = _hold(base_pos, base_wxyz, steps)
    for i in range(steps):
        c, s_ = np.cos(phi[i]), np.sin(phi[i])
        Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
        p = C + Rz @ u0
        x_cam = C - p
        x_cam = x_cam / np.linalg.norm(x_cam)
        y_cam = np.cross(np.array([0.0, 0, 1.0]), x_cam)
        y_cam = y_cam / np.linalg.norm(y_cam)
        z_cam = np.cross(x_cam, y_cam)
        wxyz[i, 2] = quat_from_matrix(np.stack([x_cam, y_cam, z_cam], axis=1))
        pos[i, 2] = p
    return Trajectory(
        name="middle_gaze_sweep",
        category="E",
        description=f"Camera orbits its gaze point ±{sweep_deg:.0f}° about "
        "world z; hands hold home.",
        purpose="7-DoF redundancy under coupled look-at tracking, with the "
        "hands present as static targets (unlike the parked mid_* suite).",
        expected_motion="Camera swings side to side, always pointing at the "
        "same spot between the hands.",
        failure_modes="Null-space drift; gaze-point error; elbow swing.",
        definition=(
            "C = p_cam_home + 0.25·x̂_optical; p(t) = C + Rz(35°·sin(2πt/12))·"
            "(p_cam_home − C); R(t) = look-at(C), horizontal image axis; "
            f"12 s ({steps} steps)"
        ),
        dt=DT,
        positions=pos,
        wxyzs=wxyz,
        markers={"gaze_center": C},
    )


# --------------------------------------------------------------------------- #
# suite assembly
# --------------------------------------------------------------------------- #
def build_suite(base_pos: np.ndarray, base_wxyz: np.ndarray) -> List[Trajectory]:
    """The full Phase 2 suite (rev 2), in evaluation order."""
    return [
        # A — pure translation (hands)
        trans_axis(base_pos, base_wxyz, 0, amp=0.12),
        trans_axis(base_pos, base_wxyz, 1, amp=0.12),
        trans_axis(base_pos, base_wxyz, 2, amp=0.09),
        # B — general translation (hands)
        trans_diag(base_pos, base_wxyz),
        trans_lissajous(base_pos, base_wxyz),
        # C — pure rotation at three workspace positions (hands)
        rot_at(base_pos, base_wxyz, "rot_home", np.array([0.0, 0.0, 0.0])),
        # rot_low: originally −0.18 z, which puts the grippers at the camera
        # body's height and the ±30° finger sweeps into it (measured −17 to
        # −45 mm regardless of x-spread / y-shift).  −0.12 z clears with
        # +5.7 mm margin (offset grid-searched with the baseline rollout) and
        # keeps the "purely lower posture" intent (rev 2.1 contact-purity
        # amendment).
        rot_at(base_pos, base_wxyz, "rot_low", np.array([0.0, 0.0, -0.12])),
        rot_at(base_pos, base_wxyz, "rot_forward", np.array([0.0, -0.10, -0.08])),
        # D — teleop reach-and-grasp, three approach styles
        teleop_grasp(base_pos, base_wxyz),
        teleop_grasp_side(base_pos, base_wxyz),
        teleop_grasp_yaw(base_pos, base_wxyz),
        # E — stress
        arms_converge(base_pos, base_wxyz),
        self_fold(base_pos, base_wxyz),
        reach_limit(base_pos, base_wxyz),
        wrist_twist(base_pos, base_wxyz),
        near_singular(base_pos, base_wxyz),
        middle_gaze_sweep(base_pos, base_wxyz),
        teleop_grasp(base_pos, base_wxyz, name="jitter_teleop",
                     noise_sigma_pos=0.003, noise_sigma_rot_deg=1.0, seed=0),
        # M — middle-arm mirror suite (hands parked)
        trans_axis(base_pos, base_wxyz, 0, amp=0.10, arms=MIDDLE),
        trans_axis(base_pos, base_wxyz, 1, amp=0.10, arms=MIDDLE),
        trans_axis(base_pos, base_wxyz, 2, amp=0.08, arms=MIDDLE),
        trans_diag(base_pos, base_wxyz, amp=0.10, arms=MIDDLE),
        trans_lissajous(base_pos, base_wxyz, amps=(0.08, 0.08, 0.05), arms=MIDDLE),
        rot_at(base_pos, base_wxyz, "mid_rot_home", np.array([0.0, 0.0, 0.0]),
               arms=MIDDLE),
        rot_at(base_pos, base_wxyz, "mid_rot_low", np.array([0.0, -0.10, -0.05]),
               arms=MIDDLE),
        mid_approach(base_pos, base_wxyz),
        mid_reach_limit(base_pos, base_wxyz),
    ]


## Subset used for expensive weight sweeps (Phases 5/7): one representative
## per category, chosen *before* any ablation results are known
## (pre-registered to avoid cherry-picking).
CORE_SUBSET = (
    "trans_lissajous",
    "rot_home",
    "teleop_grasp",
    "arms_converge",
    "reach_limit",
    "mid_approach",
)
