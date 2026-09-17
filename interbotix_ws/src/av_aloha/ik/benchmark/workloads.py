"""Target-trajectory generators used to stress the IK solver.

Each workload yields, per control step, three target poses (left, right, middle)
plus an optional moving world obstacle.  Targets are built by forward-kinematics
of perturbed configurations wherever possible, so they stay reachable unless the
workload deliberately asks for the opposite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import jaxlie
import numpy as np
import pyroki as pk
from pyroki.collision import CollGeom, Sphere

NUM_TARGETS = 3


@dataclass
class Workload:
    """A scripted teleoperation episode."""

    name: str
    positions: np.ndarray  # (T, 3, 3)
    wxyzs: np.ndarray  # (T, 3, 4)
    obstacles: Optional[List[Optional[CollGeom]]] = None
    description: str = ""

    def __len__(self) -> int:
        return self.positions.shape[0]

    def obstacle_at(self, t: int) -> Optional[CollGeom]:
        if self.obstacles is None:
            return None
        return self.obstacles[t]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def home_poses(robot: pk.Robot, q0: np.ndarray, link_indices: np.ndarray):
    fk = robot.forward_kinematics(np.asarray(q0, dtype=np.float32))
    pos, wxyz = [], []
    for idx in link_indices:
        se3 = jaxlie.SE3(fk[idx])
        pos.append(np.asarray(se3.translation(), dtype=np.float64))
        wxyz.append(np.asarray(se3.rotation().wxyz, dtype=np.float64))
    return np.stack(pos), np.stack(wxyz)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), *(axis * s)])


def _minimum_jerk(t: np.ndarray) -> np.ndarray:
    """Normalized minimum-jerk profile s(t) on t in [0, 1]."""
    return 10 * t**3 - 15 * t**4 + 6 * t**5


# --------------------------------------------------------------------------- #
# workloads
# --------------------------------------------------------------------------- #
def smooth_track(
    base_pos: np.ndarray,
    base_wxyz: np.ndarray,
    steps: int = 300,
    dt: float = 0.05,
    amp: float = 0.10,
    rot_amp_deg: float = 25.0,
    freq: float = 0.25,
) -> Workload:
    """Slow Lissajous reach + wrist rotation: the nominal teleoperation regime."""
    t = np.arange(steps) * dt
    pos = np.repeat(base_pos[None], steps, axis=0)
    wxyz = np.repeat(base_wxyz[None], steps, axis=0)
    for arm in range(NUM_TARGETS):
        phase = arm * 0.7
        pos[:, arm, 0] += amp * np.sin(2 * np.pi * freq * t + phase)
        pos[:, arm, 1] += amp * np.sin(2 * np.pi * freq * 0.5 * t + phase)
        pos[:, arm, 2] += 0.6 * amp * np.sin(2 * np.pi * freq * 1.3 * t)
        ang = np.radians(rot_amp_deg) * np.sin(2 * np.pi * freq * 0.7 * t + phase)
        for i in range(steps):
            wxyz[i, arm] = _quat_mul(
                base_wxyz[arm], _quat_from_axis_angle(np.array([0.0, 1.0, 0.3]), ang[i])
            )
    return Workload("smooth_track", pos, wxyz, description="slow Lissajous + wrist roll")


def fast_track(base_pos, base_wxyz, steps: int = 300, dt: float = 0.05) -> Workload:
    """Same shape as smooth_track but 4x faster: probes the velocity budget."""
    wl = smooth_track(base_pos, base_wxyz, steps, dt, amp=0.14, rot_amp_deg=45.0, freq=1.0)
    wl.name = "fast_track"
    wl.description = "aggressive hand motion, stresses velocity limits and lag"
    return wl


def step_response(
    base_pos, base_wxyz, steps: int = 240, dt: float = 0.05, jump: float = 0.15
) -> Workload:
    """Instantaneous target jumps: measures rise time, overshoot and jerk."""
    pos = np.repeat(base_pos[None], steps, axis=0)
    wxyz = np.repeat(base_wxyz[None], steps, axis=0)
    period = steps // 4
    for k in range(1, 4):
        offset = np.zeros(3)
        offset[k % 3] = jump * (1 if k % 2 else -1)
        pos[k * period :, :, :] += offset
    return Workload("step_response", pos, wxyz, description="step target jumps")


def jitter_track(
    base_pos, base_wxyz, steps: int = 300, dt: float = 0.05, sigma: float = 0.004,
    rot_sigma_deg: float = 1.0, seed: int = 0,
) -> Workload:
    """Smooth motion contaminated with VR tracker noise: probes jerk amplification."""
    wl = smooth_track(base_pos, base_wxyz, steps, dt)
    rng = np.random.default_rng(seed)
    pos = wl.positions + rng.normal(0.0, sigma, wl.positions.shape)
    wxyz = wl.wxyzs.copy()
    for i in range(steps):
        for arm in range(NUM_TARGETS):
            axis = rng.normal(size=3)
            ang = rng.normal(0.0, np.radians(rot_sigma_deg))
            wxyz[i, arm] = _quat_mul(wxyz[i, arm], _quat_from_axis_angle(axis, ang))
    return Workload(
        "jitter_track", pos, wxyz, description="smooth motion + tracker noise"
    )


def crossing_arms(
    base_pos, base_wxyz, steps: int = 240, dt: float = 0.05, overlap: float = 0.12
) -> Workload:
    """Both hands sweep *past* each other so their targets genuinely overlap.

    The previous version hardcoded the y axis and a fixed 0.22m reach. On this
    robot the hands are separated in *x* (0.236m) with identical y, so
    ``np.sign(midline_y - base_pos[arm, 1])`` evaluated to ``sign(0.0) == 0``,
    the ``or 1.0`` fallback fired for both arms, and they swept in the *same*
    direction. The hands never approached each other, so the workload named
    "crossing_arms" exercised no self-collision whatsoever -- any benchmark
    number it produced for a collision variant was measuring nothing.

    Now the separation axis is detected from the measured home poses and the
    reach is derived from that separation, so a genuine crossing happens on any
    robot layout. At the apex the two targets sit ``2 * overlap`` past each
    other. Those commanded poses are deliberately infeasible: that is the
    point. With self-collision active the solver should refuse them and hold a
    positive clearance; with it off the arms should interpenetrate.

    The perpendicular stagger keeps the two hands from being commanded to the
    identical point, which is a degenerate case rather than a realistic one.
    """
    pos = np.repeat(base_pos[None], steps, axis=0)
    wxyz = np.repeat(base_wxyz[None], steps, axis=0)
    s = _minimum_jerk(np.clip(np.arange(steps) / (steps * 0.5), 0.0, 1.0))
    s = np.concatenate([s[: steps // 2], s[: steps - steps // 2][::-1]])[:steps]

    delta = np.asarray(base_pos[0, :3]) - np.asarray(base_pos[1, :3])
    axis = int(np.argmax(np.abs(delta)))  # the axis the hands actually sit apart on
    stagger_axis = (axis + 1) % 3
    separation = abs(float(delta[axis]))
    reach = 0.5 * separation + overlap

    for arm in (0, 1):
        # Aim each hand at where the *other* hand is, not at a fixed world axis.
        toward_other = -np.sign(delta[axis]) if arm == 0 else np.sign(delta[axis])
        pos[:, arm, axis] = base_pos[arm, axis] + toward_other * reach * s
        pos[:, arm, stagger_axis] = base_pos[arm, stagger_axis] + (
            0.06 if arm == 0 else -0.06
        ) * s
    return Workload(
        "crossing_arms",
        pos,
        wxyz,
        description=f"hands cross {2 * overlap:.2f}m past each other",
    )


def obstacle_sweep(
    base_pos,
    base_wxyz,
    steps: int = 240,
    dt: float = 0.05,
    radius: float = 0.10,
) -> Workload:
    """A sphere sweeps through the workspace while the hands track a line."""
    wl = smooth_track(base_pos, base_wxyz, steps, dt, amp=0.06, rot_amp_deg=10.0)
    center_start = base_pos[:2].mean(axis=0) + np.array([0.12, -0.30, 0.05])
    center_end = base_pos[:2].mean(axis=0) + np.array([0.12, 0.30, 0.05])
    s = _minimum_jerk(np.linspace(0.0, 1.0, steps))
    obstacles = [
        Sphere.from_center_and_radius(
            np.asarray(center_start + (center_end - center_start) * s[i], dtype=np.float32),
            np.asarray([radius], dtype=np.float32),
        )
        for i in range(steps)
    ]
    return Workload(
        "obstacle_sweep",
        wl.positions,
        wl.wxyzs,
        obstacles=obstacles,
        description="moving sphere crosses the workspace",
    )


def workspace_edge(
    base_pos, base_wxyz, steps: int = 240, dt: float = 0.05, overreach: float = 0.45
) -> Workload:
    """Targets pushed past the reachable set: joint-limit / singularity behaviour."""
    pos = np.repeat(base_pos[None], steps, axis=0)
    wxyz = np.repeat(base_wxyz[None], steps, axis=0)
    s = _minimum_jerk(np.clip(np.arange(steps) / (steps * 0.6), 0.0, 1.0))
    for arm in (0, 1):
        outward = base_pos[arm] - base_pos[:2].mean(axis=0)
        outward = outward / max(np.linalg.norm(outward), 1e-6)
        pos[:, arm] = base_pos[arm] + np.outer(s, outward * overreach + np.array([0.25, 0, 0.1]))
    return Workload(
        "workspace_edge", pos, wxyz, description="targets driven outside the workspace"
    )


def replay(path: str, dt: float = 0.05) -> Workload:
    """Replay recorded teleop targets.

    Expects an .npz with `positions` (T, 3, 3) and `wxyzs` (T, 3, 4), which is the
    format to dump from the real VR loop for a like-for-like comparison.
    """
    data = np.load(path)
    return Workload(
        f"replay:{path.split('/')[-1]}",
        np.asarray(data["positions"], dtype=np.float64),
        np.asarray(data["wxyzs"], dtype=np.float64),
        description="recorded VR teleoperation targets",
    )


BUILDERS: Dict[str, Callable[..., Workload]] = {
    "smooth_track": smooth_track,
    "fast_track": fast_track,
    "step_response": step_response,
    "jitter_track": jitter_track,
    "crossing_arms": crossing_arms,
    "obstacle_sweep": obstacle_sweep,
    "workspace_edge": workspace_edge,
}

DEFAULT_WORKLOADS = (
    "smooth_track",
    "fast_track",
    "step_response",
    "jitter_track",
    "crossing_arms",
    "obstacle_sweep",
    "workspace_edge",
)


def build(names, base_pos, base_wxyz, steps: int, dt: float) -> List[Workload]:
    out = []
    for name in names:
        if name.startswith("replay:"):
            out.append(replay(name.split(":", 1)[1], dt))
        else:
            out.append(BUILDERS[name](base_pos, base_wxyz, steps=steps, dt=dt))
    return out
