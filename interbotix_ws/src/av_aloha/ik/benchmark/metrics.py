"""State evaluation and trajectory-level metrics for the IK benchmark.

Everything here is measurement only: it never influences the solve, so the same
metric set is comparable across every cost configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
from pyroki.collision import CollGeom, RobotCollision

ARMS = ("left", "right", "middle")


# --------------------------------------------------------------------------- #
# Per-configuration evaluators (jitted)
# --------------------------------------------------------------------------- #
class StateEvaluator:
    """Evaluates pose error, manipulability, clearances and limit margins."""

    def __init__(
        self,
        robot: pk.Robot,
        robot_coll: Optional[RobotCollision],
        target_link_indices: np.ndarray,
    ) -> None:
        self.robot = robot
        self.robot_coll = robot_coll
        idx = jnp.asarray(target_link_indices)
        # Metrics on joint limits and velocity only make sense for arm joints.
        self.arm_joint_mask = np.asarray(
            ["finger" not in n for n in robot.joints.actuated_names], dtype=bool
        )

        @jax.jit
        def pose_of(q):
            fk = robot.forward_kinematics(q)
            se3 = jaxlie.SE3(fk[idx])
            return se3.translation(), se3.rotation().wxyz

        @jax.jit
        def manipulability(q):
            """Yoshikawa index per target link: translational (3x3) and full 6D."""

            def per_link(link_index):
                jac_full = jax.jacfwd(
                    lambda qq: jaxlie.SE3(robot.forward_kinematics(qq)).log()
                )(q)[link_index]
                jac_t = jax.jacfwd(
                    lambda qq: jaxlie.SE3(robot.forward_kinematics(qq)).translation()
                )(q)[link_index]
                w_t = jnp.sqrt(jnp.maximum(0.0, jnp.linalg.det(jac_t @ jac_t.T)))
                w_6 = jnp.sqrt(
                    jnp.maximum(0.0, jnp.linalg.det(jac_full @ jac_full.T + 1e-12 * jnp.eye(6)))
                )
                sv = jnp.linalg.svd(jac_full, compute_uv=False)
                cond = sv[0] / jnp.maximum(sv[-1], 1e-9)
                return w_t, w_6, cond

            return jax.vmap(per_link)(idx)

        @jax.jit
        def limit_margin(q):
            lower = robot.joints.lower_limits
            upper = robot.joints.upper_limits
            return jnp.minimum(q - lower, upper - q)

        self._pose_of = pose_of
        self._manip = manipulability
        self._limit_margin = limit_margin

        if robot_coll is not None:

            @jax.jit
            def self_clearance(q):
                return jnp.min(robot_coll.compute_self_collision_distance(robot, q))

            @jax.jit
            def world_clearance(q, geom: CollGeom):
                return jnp.min(robot_coll.compute_world_collision_distance(robot, q, geom))

            self._self_clearance = self_clearance
            self._world_clearance = world_clearance
        else:
            self._self_clearance = None
            self._world_clearance = None

    # -- helpers ------------------------------------------------------------ #
    def ee_poses(self, q: np.ndarray):
        pos, wxyz = self._pose_of(jnp.asarray(q, dtype=jnp.float32))
        return np.asarray(pos), np.asarray(wxyz)

    def pose_error(self, q: np.ndarray, target_pos: np.ndarray, target_wxyz: np.ndarray):
        """Returns (position error [m] per arm, orientation error [deg] per arm)."""
        pos, wxyz = self.ee_poses(q)
        pos_err = np.linalg.norm(pos - np.asarray(target_pos), axis=-1)
        ori_err = np.asarray(
            [
                _geodesic_deg(wxyz[i], np.asarray(target_wxyz)[i])
                for i in range(wxyz.shape[0])
            ]
        )
        return pos_err, ori_err

    def manipulability(self, q: np.ndarray):
        w_t, w_6, cond = self._manip(jnp.asarray(q, dtype=jnp.float32))
        return np.asarray(w_t), np.asarray(w_6), np.asarray(cond)

    def limit_margin(self, q: np.ndarray) -> np.ndarray:
        """Distance to the nearest joint limit, arm joints only."""
        margin = np.asarray(self._limit_margin(jnp.asarray(q, dtype=jnp.float32)))
        return margin[self.arm_joint_mask]

    def self_clearance(self, q: np.ndarray) -> float:
        if self._self_clearance is None:
            return float("nan")
        return float(self._self_clearance(jnp.asarray(q, dtype=jnp.float32)))

    def world_clearance(self, q: np.ndarray, geom: Optional[CollGeom]) -> float:
        if self._world_clearance is None or geom is None:
            return float("nan")
        return float(self._world_clearance(jnp.asarray(q, dtype=jnp.float32), geom))


def _geodesic_deg(wxyz_a: np.ndarray, wxyz_b: np.ndarray) -> float:
    """Angle of the relative rotation, in degrees (sign-invariant)."""
    dot = float(np.clip(abs(np.dot(wxyz_a, wxyz_b)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


# --------------------------------------------------------------------------- #
# Rollout recording
# --------------------------------------------------------------------------- #
@dataclass
class Rollout:
    """Raw per-step logs of one workload run."""

    dt: float
    q: List[np.ndarray] = field(default_factory=list)
    target_pos: List[np.ndarray] = field(default_factory=list)
    target_wxyz: List[np.ndarray] = field(default_factory=list)
    ee_pos: List[np.ndarray] = field(default_factory=list)
    pos_err: List[np.ndarray] = field(default_factory=list)
    ori_err: List[np.ndarray] = field(default_factory=list)
    manip_t: List[np.ndarray] = field(default_factory=list)
    manip_6: List[np.ndarray] = field(default_factory=list)
    cond: List[np.ndarray] = field(default_factory=list)
    limit_margin: List[np.ndarray] = field(default_factory=list)
    self_clear: List[float] = field(default_factory=list)
    world_clear: List[float] = field(default_factory=list)
    solve_ms: List[float] = field(default_factory=list)
    iterations: List[int] = field(default_factory=list)

    def array(self, name: str) -> np.ndarray:
        return np.asarray(getattr(self, name), dtype=np.float64)


def _pct(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p)) if x.size else float("nan")


def summarize(
    rollout: Rollout,
    velocity_limit: float,
    arm_joint_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Condense a rollout into scalar metrics.

    `arm_joint_mask` selects the joints that count for motion-quality metrics
    (i.e. excludes the gripper fingers).
    """
    dt = rollout.dt
    q = rollout.array("q")  # (T, n)
    if arm_joint_mask is not None:
        q = q[:, np.asarray(arm_joint_mask, dtype=bool)]
    out: Dict[str, float] = {"steps": float(q.shape[0])}

    # --- tracking ---------------------------------------------------------- #
    pos_err = rollout.array("pos_err")  # (T, 3 arms)
    ori_err = rollout.array("ori_err")
    for i, arm in enumerate(ARMS):
        out[f"pos_err_mean_{arm}_mm"] = float(np.mean(pos_err[:, i])) * 1e3
        out[f"pos_err_p95_{arm}_mm"] = _pct(pos_err[:, i], 95) * 1e3
        out[f"pos_err_max_{arm}_mm"] = float(np.max(pos_err[:, i])) * 1e3
        out[f"ori_err_mean_{arm}_deg"] = float(np.mean(ori_err[:, i]))
        out[f"ori_err_p95_{arm}_deg"] = _pct(ori_err[:, i], 95)
    out["pos_err_mean_mm"] = float(np.mean(pos_err[:, :2])) * 1e3  # arms only
    out["ori_err_mean_deg"] = float(np.mean(ori_err[:, :2]))
    out["pos_err_p95_mm"] = _pct(pos_err[:, :2].ravel(), 95) * 1e3
    # Both arms: a single-arm orientation metric leaves the other arm's weight
    # unconstrained, which is how an unjustified left/right asymmetry appeared
    # in the first search.
    out["ori_err_p95_deg"] = _pct(ori_err[:, :2].ravel(), 95)

    # --- smoothness (joint space) ------------------------------------------ #
    if q.shape[0] >= 4:
        dq = np.diff(q, axis=0) / dt
        ddq = np.diff(dq, axis=0) / dt
        dddq = np.diff(ddq, axis=0) / dt
        out["joint_vel_rms"] = float(np.sqrt(np.mean(np.sum(dq**2, axis=1))))
        out["joint_acc_rms"] = float(np.sqrt(np.mean(np.sum(ddq**2, axis=1))))
        out["joint_jerk_rms"] = float(np.sqrt(np.mean(np.sum(dddq**2, axis=1))))
        out["joint_vel_max"] = float(np.max(np.abs(dq)))
        # Motor-stress proxies.  Joint torque tracks acceleration, so peak and
        # RMS acceleration are the quantities that predict overload; RMS stands
        # in for thermal load, peak for instantaneous current.  These are
        # kinematic proxies -- there is no dynamics model here, so they rank
        # configurations against each other rather than predicting amps.
        out["joint_acc_max"] = float(np.max(np.abs(ddq)))
        out["joint_acc_p95"] = _pct(np.abs(ddq).ravel(), 95)
        # fraction of joint-steps pinned at the velocity cap
        cap = velocity_limit * 0.99
        out["vel_saturation_frac"] = float(np.mean(np.abs(dq) >= cap))
        # discontinuity / elbow-flip detector: joint-space jumps that are large
        # relative to the median step of this run
        step_norm = np.linalg.norm(np.diff(q, axis=0), axis=1)
        med = float(np.median(step_norm)) if step_norm.size else 0.0
        thresh = max(5.0 * med, 0.15)
        out["config_jumps"] = float(np.sum(step_norm > thresh))
        out["config_jump_max_rad"] = float(np.max(step_norm))
    else:
        for k in (
            "joint_vel_rms",
            "joint_acc_rms",
            "joint_jerk_rms",
            "joint_vel_max",
            "joint_acc_max",
            "joint_acc_p95",
            "vel_saturation_frac",
            "config_jumps",
            "config_jump_max_rad",
        ):
            out[k] = float("nan")

    # --- smoothness (task space, what the operator sees) ------------------- #
    ee = rollout.array("ee_pos")  # (T, 3 arms, 3)
    if ee.shape[0] >= 4:
        d3 = np.diff(ee, n=3, axis=0) / dt**3
        for i, arm in enumerate(ARMS):
            out[f"ee_jerk_rms_{arm}"] = float(
                np.sqrt(np.mean(np.sum(d3[:, i] ** 2, axis=-1)))
            )
        out["ee_jerk_rms"] = float(np.mean([out[f"ee_jerk_rms_{a}"] for a in ARMS[:2]]))

    # --- configuration quality --------------------------------------------- #
    manip_t = rollout.array("manip_t")
    manip_6 = rollout.array("manip_6")
    cond = rollout.array("cond")
    for i, arm in enumerate(ARMS):
        out[f"manip_min_{arm}"] = float(np.min(manip_t[:, i]))
        out[f"manip_mean_{arm}"] = float(np.mean(manip_t[:, i]))
        out[f"manip6_mean_{arm}"] = float(np.mean(manip_6[:, i]))
        out[f"cond_p95_{arm}"] = _pct(cond[:, i], 95)
    out["manip_min"] = float(np.min(manip_t[:, :2]))
    out["manip_mean"] = float(np.mean(manip_t[:, :2]))

    # --- safety ------------------------------------------------------------ #
    lm = rollout.array("limit_margin")
    out["limit_margin_min_rad"] = float(np.min(lm))
    out["limit_violation_frac"] = float(np.mean(lm < 0.0))
    out["near_limit_frac"] = float(np.mean(lm < 0.05))
    sc = rollout.array("self_clear")
    if sc.size and np.isfinite(sc).any():
        out["self_clear_min_m"] = float(np.nanmin(sc))
        out["self_collision_frac"] = float(np.nanmean(sc < 0.0))
    wc = rollout.array("world_clear")
    if wc.size and np.isfinite(wc).any():
        out["world_clear_min_m"] = float(np.nanmin(wc))
        out["world_collision_frac"] = float(np.nanmean(wc < 0.0))

    # --- compute ------------------------------------------------------------ #
    ms = rollout.array("solve_ms")
    if ms.size:
        out["solve_ms_mean"] = float(np.mean(ms))
        out["solve_ms_p95"] = _pct(ms, 95)
        out["solve_ms_max"] = float(np.max(ms))
        out["realtime_factor"] = float(np.mean(ms) / (dt * 1e3))
    it = rollout.array("iterations")
    if it.size:
        out["iters_mean"] = float(np.mean(it))

    # --- responsiveness / lag ---------------------------------------------- #
    tgt = rollout.array("target_pos")  # (T, 3 arms, 3)
    if tgt.shape[0] > 8:
        lags = []
        for i in range(2):  # arms only
            lags.append(_lag_steps(tgt[:, i], ee[:, i]))
        out["lag_steps"] = float(np.mean(lags))
        out["lag_ms"] = out["lag_steps"] * dt * 1e3
    return out


def _lag_steps(target: np.ndarray, actual: np.ndarray, max_lag: int = 20) -> float:
    """Delay (in control steps) that best aligns actual with target motion."""
    t = target - target.mean(axis=0)
    a = actual - actual.mean(axis=0)
    scale = np.linalg.norm(t) * np.linalg.norm(a)
    if scale < 1e-9:
        return 0.0
    best, best_score = 0, -np.inf
    for lag in range(0, max_lag + 1):
        if lag >= t.shape[0]:
            break
        score = float(np.sum(t[: t.shape[0] - lag] * a[lag:]) / scale)
        if score > best_score:
            best_score, best = score, lag
    return float(best)


# --------------------------------------------------------------------------- #
# Teleoperation score
# --------------------------------------------------------------------------- #
DEFAULT_SCORE_TERMS = {
    # metric: (target "good" value, weight).  Each term contributes
    # weight * min(1, value / target); lower total is better.
    "pos_err_p95_mm": (5.0, 1.0),
    "ori_err_mean_deg": (2.0, 0.7),
    "ee_jerk_rms": (5.0, 1.0),
    "joint_jerk_rms": (200.0, 0.5),
    "lag_ms": (100.0, 0.8),
    "config_jumps": (1.0, 1.0),
    "vel_saturation_frac": (0.05, 0.5),
    "near_limit_frac": (0.05, 0.5),
    "solve_ms_mean": (20.0, 0.6),
}
PENALTY_TERMS = {
    # metric: (weight) -- fraction-of-time constraint violations, heavily punished
    "limit_violation_frac": 3.0,
    "self_collision_frac": 3.0,
    "world_collision_frac": 2.0,
}
BONUS_TERMS = {
    # metric: (reference value, weight) -- higher is better
    "manip_min": (0.02, 0.6),
}


def teleop_score(metrics: Dict[str, float]) -> float:
    """Lower is better.  A single number for ranking configurations.

    Each cost term is normalized by an operator-relevant target value so the
    units (mm, deg, ms, m/s^3) become comparable, then linearly weighted.
    """
    score = 0.0
    for key, (target, weight) in DEFAULT_SCORE_TERMS.items():
        val = metrics.get(key, np.nan)
        if not np.isfinite(val):
            continue
        score += weight * (val / target)
    for key, weight in PENALTY_TERMS.items():
        val = metrics.get(key, 0.0)
        if np.isfinite(val):
            score += weight * 10.0 * val
    for key, (ref, weight) in BONUS_TERMS.items():
        val = metrics.get(key, np.nan)
        if np.isfinite(val):
            score -= weight * min(1.0, val / ref)
    return float(score)


# --------------------------------------------------------------------------- #
# Priority-structured objective
# --------------------------------------------------------------------------- #
@dataclass
class Priority:
    """Encodes 'pose accuracy first, everything else important but secondary'.

    The primary objective (position and orientation error) is minimized
    *continuously* -- every millimetre counts, always.

    The secondary objectives are hinge terms: they contribute exactly zero while
    the metric is inside its acceptable band, and grow once it leaves.  That is
    what 'secondary but important' means operationally -- they never trade away
    accuracy while behaviour is acceptable, but they dominate the objective the
    moment it is not.  A weighted-sum score (the earlier `teleop_score`) cannot
    express this: it always trades, so it will happily accept a jerky solution
    for a fraction of a millimetre.
    """

    # primary: error at which the term contributes 1.0
    pos_scale_mm: float = 5.0
    ori_scale_deg: float = 2.0
    ori_weight: float = 1.0

    # secondary: acceptable band; zero cost inside, hinge outside
    max_ee_jerk: float = 80.0
    max_config_jumps: float = 0.5
    max_lag_ms: float = 60.0
    max_vel_saturation: float = 0.10
    # Calibrated against measured baselines, not picked a priori: every variant
    # sits near 0.39-0.46 self-collision fraction because the capsule model
    # overlaps even at the home pose, and ~0.10 of arm-joint samples sit within
    # 0.05 rad of a limit in normal operation.  Thresholds tighter than the
    # baseline would penalize every candidate equally and just add a constant.
    max_near_limit_frac: float = 0.10
    max_limit_violation_frac: float = 0.0
    max_self_collision_frac: float = 0.45
    max_world_collision_frac: float = 0.30
    min_manip: float = 0.025
    max_iters: float = 25.0

    # how hard the secondary hinges bite once violated
    secondary_weight: float = 4.0
    safety_weight: float = 20.0


def _hinge_over(value: float, limit: float, scale: Optional[float] = None) -> float:
    """0 while value <= limit, then grows linearly. `scale` normalizes the slope."""
    if not np.isfinite(value):
        return 0.0
    denom = scale if scale is not None else max(abs(limit), 1e-6)
    return max(0.0, (value - limit) / denom)


def _hinge_under(value: float, floor: float) -> float:
    """0 while value >= floor, then grows as the value falls below it."""
    if not np.isfinite(value):
        return 0.0
    return max(0.0, (floor - value) / max(abs(floor), 1e-6))


def objective(metrics: Dict[str, float], priority: Optional[Priority] = None) -> Dict[str, float]:
    """Returns {'objective', 'primary', 'secondary', 'safety'}; lower is better."""
    p = priority or Priority()

    pos = metrics.get("pos_err_p95_mm", np.nan)
    ori = metrics.get("ori_err_p95_deg", np.nan)
    if not np.isfinite(ori):
        ori = metrics.get("ori_err_mean_deg", np.nan)
    primary = 0.0
    if np.isfinite(pos):
        primary += pos / p.pos_scale_mm
    if np.isfinite(ori):
        primary += p.ori_weight * ori / p.ori_scale_deg

    secondary = (
        _hinge_over(metrics.get("ee_jerk_rms", np.nan), p.max_ee_jerk)
        + _hinge_over(metrics.get("config_jumps", np.nan), p.max_config_jumps, scale=1.0)
        + _hinge_over(metrics.get("lag_ms", np.nan), p.max_lag_ms)
        + _hinge_over(metrics.get("vel_saturation_frac", np.nan), p.max_vel_saturation)
        + _hinge_over(metrics.get("iters_mean", np.nan), p.max_iters)
    )

    safety = (
        _hinge_over(metrics.get("near_limit_frac", np.nan), p.max_near_limit_frac, scale=0.10)
        + _hinge_over(metrics.get("limit_violation_frac", 0.0), p.max_limit_violation_frac, scale=0.05)
        + _hinge_over(metrics.get("self_collision_frac", np.nan), p.max_self_collision_frac)
        + _hinge_over(metrics.get("world_collision_frac", np.nan), p.max_world_collision_frac)
        + _hinge_under(metrics.get("manip_min", np.nan), p.min_manip)
    )

    return {
        "objective": primary + p.secondary_weight * secondary + p.safety_weight * safety,
        "primary": primary,
        "secondary": secondary,
        "safety": safety,
    }


def aggregate(per_workload: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Mean of each metric across workloads (missing metrics ignored)."""
    keys = sorted({k for m in per_workload.values() for k in m})
    out: Dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in per_workload.values() if k in m and np.isfinite(m[k])]
        if vals:
            out[k] = float(np.mean(vals))
    return out
