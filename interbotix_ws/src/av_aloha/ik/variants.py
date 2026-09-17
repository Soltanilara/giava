"""Phase 4 — one-residual-at-a-time ablation variants.

Each variant is the *frozen baseline problem* (BASELINE.md: pose ×3 with
weights 50/10 + joint-limit constraint, LM ≤100 iters, dense Cholesky,
lambda_initial 1.0, warm start = previous commanded configuration) plus
EXACTLY ONE additional residual.  baseline.py is not modified; nothing else
about the solver changes between variants.

Default weights are the user's own deployed / preset values (the point of
Phase 4 is to evaluate *their* additions against the clean baseline; Phase 5
sweeps the weights):

variant          extra residual                                    weight
--------------------------------------------------------------------------
smoothing        (w/(v_nom·dt)) · (q − q_prev), all 23 joints      w=0.5,
                 [custom, from three_arm_ik.py / ik_benchmark]     v_nom=2.0
centering        w · (q − mid)/half_range, arm joints only         w=0.05
                 [custom, from ik_benchmark solver.py]
collision        pyroki self_collision_cost, margin 3 cm           w=5.0
                 [pyroki-native; world collision deferred — no
                 obstacle in the suite; deeper collision study
                 planned separately]
manipulability   pyroki manipulability_cost = w/(Yoshikawa_t+1e-6),
                 hand arms only (deployed preset zeroed the
                 middle arm, and a zero-weight term would still
                 cost compute)                                     w=0.02

Notes on formulations:
- smoothing's anchor is `q_init`, which under the tracking protocol IS the
  previously commanded configuration — the residual turns the warm start's
  implicit basin preference into an explicit cost.  At 50 Hz the scale is
  w/(v_nom·dt) = 0.5/(2.0·0.02) = 12.5 (the deployed 20 Hz loop had 5.0):
  same formulation, same physical meaning (fraction of the per-tick velocity
  budget), different tick length.
- centering normalizes by each joint's half-range, so the residual reads as
  "fraction of available travel used"; fingers are excluded (their 0–0.041
  range would dominate).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
from pyroki.collision import RobotCollision

sys.path.insert(0, str(Path(__file__).resolve().parent))

import robot_model as rm
from baseline import (
    DEFAULT_ORI_WEIGHT,
    DEFAULT_POS_WEIGHT,
    NUM_TARGETS,
    SolveResult,
)

EPS = 1e-6

# The user's deployed / preset weights (see module docstring).
SMOOTHING_WEIGHT = 0.5
SMOOTHING_V_NOMINAL = 2.0  # rad/s — deployed ControllerConfig.nominal_velocity
CENTERING_WEIGHT = 0.05
COLLISION_WEIGHT = 5.0
COLLISION_MARGIN = 0.03  # m
MANIPULABILITY_WEIGHT = 0.02

DT = 0.02  # must match trajectories.DT


@jaxls.Cost.create_factory
def _smoothing_residual(vals, joint_var, prev_q, scale):
    """User's velocity-budget-scaled previous-configuration regularization."""
    return (scale * (vals[joint_var] - prev_q)).flatten()


@jaxls.Cost.create_factory
def _centering_residual(vals, robot, joint_var, weight, joint_mask):
    """User's range-normalized joint centering (dimensionless residual)."""
    q = vals[joint_var]
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits
    mid = 0.5 * (lower + upper)
    half = jnp.maximum(0.5 * (upper - lower), EPS)
    return (weight * joint_mask * (q - mid) / half).flatten()


class VariantIK:
    """Baseline + exactly one extra residual (`extra` ∈ smoothing / centering /
    collision / manipulability).  Same interface as BaselineIK."""

    def __init__(
        self,
        robot: pk.Robot,
        target_link_names: Tuple[str, str, str],
        extra: str,
        robot_coll: Optional[RobotCollision] = None,
        max_iterations: int = 100,
    ) -> None:
        assert extra in ("smoothing", "centering", "collision", "manipulability")
        if extra == "collision" and robot_coll is None:
            raise ValueError("collision variant needs robot_coll")
        self.robot = robot
        self.extra = extra
        self.max_iterations = max_iterations
        self.target_indices = np.asarray(
            [robot.links.names.index(n) for n in target_link_names], dtype=np.int32
        )
        target_idx_jax = jnp.asarray(self.target_indices)
        arm_mask = jnp.asarray(
            (~rm.finger_joint_mask(robot)).astype(np.float32)
        )

        @jdc.jit
        def _solve(
            q_init: jax.Array,
            target_positions: jax.Array,
            target_wxyzs: jax.Array,
            pos_weights: jax.Array,
            ori_weights: jax.Array,
            extra_weight: jax.Array,  # meaning depends on `extra`
            collision_margin: jax.Array,
        ):
            joint_var = robot.joint_var_cls(0)
            costs = []
            for i in range(NUM_TARGETS):
                costs.append(
                    pk.costs.pose_cost_analytic_jac(
                        robot,
                        joint_var,
                        jaxlie.SE3.from_rotation_and_translation(
                            jaxlie.SO3(target_wxyzs[i]), target_positions[i]
                        ),
                        target_idx_jax[i],
                        pos_weight=pos_weights[i],
                        ori_weight=ori_weights[i],
                    )
                )
            costs.append(pk.costs.limit_constraint(robot, joint_var))

            # -- the single ablation term (static branch: compiled once) ---- #
            if extra == "smoothing":
                costs.append(
                    _smoothing_residual(
                        joint_var=joint_var,
                        prev_q=q_init,
                        scale=extra_weight,  # = w / (v_nom · dt)
                    )
                )
            elif extra == "centering":
                costs.append(
                    _centering_residual(
                        robot=robot,
                        joint_var=joint_var,
                        weight=extra_weight,
                        joint_mask=arm_mask,
                    )
                )
            elif extra == "collision":
                costs.append(
                    pk.costs.self_collision_cost(
                        robot=robot,
                        robot_coll=robot_coll,
                        joint_var=joint_var,
                        margin=collision_margin,
                        weight=extra_weight,
                    )
                )
            elif extra == "manipulability":
                for i in range(2):  # hand arms only (deployed preset)
                    costs.append(
                        pk.costs.manipulability_cost(
                            robot=robot,
                            joint_var=joint_var,
                            target_link_indices=target_idx_jax[i],
                            weight=extra_weight,
                        )
                    )

            sol, summary = (
                jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make([joint_var.with_value(q_init)]),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
                    termination=jaxls.TerminationConfig(max_iterations=max_iterations),
                    return_summary=True,
                )
            )
            return sol[joint_var], summary.iterations, summary.termination_criteria

        self._solve_jax = _solve
        self.extra_weight = {
            "smoothing": SMOOTHING_WEIGHT / (SMOOTHING_V_NOMINAL * DT),
            "centering": CENTERING_WEIGHT,
            "collision": COLLISION_WEIGHT,
            "manipulability": MANIPULABILITY_WEIGHT,
        }[extra]
        self.collision_margin = COLLISION_MARGIN

    def solve(
        self,
        q_init: np.ndarray,
        target_positions: np.ndarray,
        target_wxyzs: np.ndarray,
        pos_weights: Optional[np.ndarray] = None,
        ori_weights: Optional[np.ndarray] = None,
        extra_weight: Optional[float] = None,
    ) -> SolveResult:
        pos = np.asarray(target_positions, dtype=np.float32).reshape(NUM_TARGETS, 3)
        wxyz = np.asarray(target_wxyzs, dtype=np.float32).reshape(NUM_TARGETS, 4)
        wxyz = wxyz / np.linalg.norm(wxyz, axis=-1, keepdims=True)
        pw = (
            np.full(NUM_TARGETS, DEFAULT_POS_WEIGHT, dtype=np.float32)
            if pos_weights is None
            else np.asarray(pos_weights, dtype=np.float32)
        )
        ow = (
            np.full(NUM_TARGETS, DEFAULT_ORI_WEIGHT, dtype=np.float32)
            if ori_weights is None
            else np.asarray(ori_weights, dtype=np.float32)
        )
        ew = self.extra_weight if extra_weight is None else float(extra_weight)

        t0 = time.perf_counter()
        q, iters, term = self._solve_jax(
            q_init=jnp.asarray(np.asarray(q_init, dtype=np.float32)),
            target_positions=jnp.asarray(pos),
            target_wxyzs=jnp.asarray(wxyz),
            pos_weights=jnp.asarray(pw),
            ori_weights=jnp.asarray(ow),
            extra_weight=jnp.float32(ew),
            collision_margin=jnp.float32(self.collision_margin),
        )
        q = q.block_until_ready()
        solve_ms = (time.perf_counter() - t0) * 1e3
        return SolveResult(
            q=np.asarray(q, dtype=np.float32),
            iterations=int(iters),
            termination=np.asarray(term, dtype=bool).reshape(-1),
            solve_ms=solve_ms,
        )

    def warmup(self, q0: np.ndarray) -> None:
        fk = self.robot.forward_kinematics(np.asarray(q0, dtype=np.float32))
        pos = np.stack(
            [np.asarray(jaxlie.SE3(fk[i]).translation()) for i in self.target_indices]
        )
        wxyz = np.stack(
            [
                np.asarray(jaxlie.SE3(fk[i]).rotation().wxyz)
                for i in self.target_indices
            ]
        )
        self.solve(q0, pos, wxyz)


class ComboIK:
    """Baseline + a *set* of extra residuals with fixed weights (Phase 7).

    Weights are baked per-combo (no sweeps planned at combo level); pos/ori
    weights remain runtime arguments as in BaselineIK.  Phase 5 operating
    points: smoothing w=0.05 (scale 1.25), centering w=0.5,
    collision (5.0, 3 cm — flagged broken-as-configured, included for
    interaction data only), manipulability w=0.02 (least-bad; no good point
    exists per the sweep)."""

    def __init__(
        self,
        robot: pk.Robot,
        target_link_names: Tuple[str, str, str],
        extras: dict,  # name -> weight (smoothing weight given as w, not scale)
        robot_coll: Optional[RobotCollision] = None,
        max_iterations: int = 100,
        collision_margin: float = COLLISION_MARGIN,
        table_coll: Optional[RobotCollision] = None,
        table_geom=None,  # pk.collision.CollGeom, e.g. table_collision.table_halfspace()
        table_margin: float = 0.020,
    ) -> None:
        assert set(extras) <= {
            "smoothing", "centering", "collision", "manipulability", "table",
        }
        if "collision" in extras and robot_coll is None:
            raise ValueError("collision combo needs robot_coll")
        if "table" in extras and (table_coll is None or table_geom is None):
            raise ValueError("table combo needs table_coll and table_geom "
                              "(see table_collision.py)")
        self.robot = robot
        self.extras = dict(extras)
        self.max_iterations = max_iterations
        self.target_indices = np.asarray(
            [robot.links.names.index(n) for n in target_link_names], dtype=np.int32
        )
        target_idx_jax = jnp.asarray(self.target_indices)
        arm_mask = jnp.asarray((~rm.finger_joint_mask(robot)).astype(np.float32))
        smoothing_scale = (
            extras.get("smoothing", 0.0) / (SMOOTHING_V_NOMINAL * DT)
        )

        @jdc.jit
        def _solve(
            q_init: jax.Array,
            target_positions: jax.Array,
            target_wxyzs: jax.Array,
            pos_weights: jax.Array,
            ori_weights: jax.Array,
        ):
            joint_var = robot.joint_var_cls(0)
            costs = []
            for i in range(NUM_TARGETS):
                costs.append(
                    pk.costs.pose_cost_analytic_jac(
                        robot,
                        joint_var,
                        jaxlie.SE3.from_rotation_and_translation(
                            jaxlie.SO3(target_wxyzs[i]), target_positions[i]
                        ),
                        target_idx_jax[i],
                        pos_weight=pos_weights[i],
                        ori_weight=ori_weights[i],
                    )
                )
            costs.append(pk.costs.limit_constraint(robot, joint_var))
            if "smoothing" in extras:
                costs.append(
                    _smoothing_residual(
                        joint_var=joint_var, prev_q=q_init,
                        scale=jnp.float32(smoothing_scale),
                    )
                )
            if "centering" in extras:
                costs.append(
                    _centering_residual(
                        robot=robot, joint_var=joint_var,
                        weight=jnp.float32(extras["centering"]),
                        joint_mask=arm_mask,
                    )
                )
            if "collision" in extras:
                costs.append(
                    pk.costs.self_collision_cost(
                        robot=robot, robot_coll=robot_coll, joint_var=joint_var,
                        margin=collision_margin,
                        weight=extras["collision"],
                    )
                )
            if "manipulability" in extras:
                for i in range(2):
                    costs.append(
                        pk.costs.manipulability_cost(
                            robot=robot, joint_var=joint_var,
                            target_link_indices=target_idx_jax[i],
                            weight=extras["manipulability"],
                        )
                    )
            if "table" in extras:
                costs.append(
                    pk.costs.world_collision_cost(
                        robot=robot, robot_coll=table_coll, joint_var=joint_var,
                        world_geom=table_geom,
                        margin=table_margin,
                        weight=extras["table"],
                    )
                )
            sol, summary = (
                jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make([joint_var.with_value(q_init)]),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
                    termination=jaxls.TerminationConfig(max_iterations=max_iterations),
                    return_summary=True,
                )
            )
            return sol[joint_var], summary.iterations, summary.termination_criteria

        self._solve_jax = _solve

    def solve(self, q_init, target_positions, target_wxyzs,
              pos_weights=None, ori_weights=None) -> SolveResult:
        pos = np.asarray(target_positions, dtype=np.float32).reshape(NUM_TARGETS, 3)
        wxyz = np.asarray(target_wxyzs, dtype=np.float32).reshape(NUM_TARGETS, 4)
        wxyz = wxyz / np.linalg.norm(wxyz, axis=-1, keepdims=True)
        pw = (
            np.full(NUM_TARGETS, DEFAULT_POS_WEIGHT, dtype=np.float32)
            if pos_weights is None else np.asarray(pos_weights, dtype=np.float32)
        )
        ow = (
            np.full(NUM_TARGETS, DEFAULT_ORI_WEIGHT, dtype=np.float32)
            if ori_weights is None else np.asarray(ori_weights, dtype=np.float32)
        )
        t0 = time.perf_counter()
        q, iters, term = self._solve_jax(
            q_init=jnp.asarray(np.asarray(q_init, dtype=np.float32)),
            target_positions=jnp.asarray(pos),
            target_wxyzs=jnp.asarray(wxyz),
            pos_weights=jnp.asarray(pw),
            ori_weights=jnp.asarray(ow),
        )
        q = q.block_until_ready()
        return SolveResult(
            q=np.asarray(q, dtype=np.float32),
            iterations=int(iters),
            termination=np.asarray(term, dtype=bool).reshape(-1),
            solve_ms=(time.perf_counter() - t0) * 1e3,
        )

    def warmup(self, q0: np.ndarray) -> None:
        fk = self.robot.forward_kinematics(np.asarray(q0, dtype=np.float32))
        pos = np.stack(
            [np.asarray(jaxlie.SE3(fk[i]).translation()) for i in self.target_indices]
        )
        wxyz = np.stack(
            [np.asarray(jaxlie.SE3(fk[i]).rotation().wxyz)
             for i in self.target_indices]
        )
        self.solve(q0, pos, wxyz)


## Phase 5 operating points used for every Phase 7 combination.
COMBO_WEIGHTS = {
    "smoothing": 0.05,
    "centering": 0.5,
    "collision": COLLISION_WEIGHT,
    "manipulability": MANIPULABILITY_WEIGHT,
}
COMBOS = {
    "smooth_center": ("smoothing", "centering"),
    "smooth_collision": ("smoothing", "collision"),
    "smooth_manip": ("smoothing", "manipulability"),
    "center_collision": ("centering", "collision"),
    "center_manip": ("centering", "manipulability"),
    "collision_manip": ("collision", "manipulability"),
    "smooth_center_collision": ("smoothing", "centering", "collision"),
    "smooth_center_manip": ("smoothing", "centering", "manipulability"),
}

## Deferred combos, now with the VALIDATED collision model (pruned 180-sphere,
## margin from the Phase-9 sweep).  Names end in S to distinguish from the
## broken-capsule-era combos above.
SPHERE_COLLISION_WEIGHT = 100.0  # finalized from the Phase-9 sweep
SPHERE_COLLISION_MARGIN = 0.020
COMBOS_SPHERE = {
    "smooth_collisionS": ("smoothing", "collision"),
    "center_collisionS": ("centering", "collision"),
    "collisionS_manip": ("collision", "manipulability"),
    "smooth_center_collisionS": ("smoothing", "centering", "collision"),
}


## --------------------------------------------------------------------- ##
## Collision-study variants (Phase 8): validated sphere geometry
## --------------------------------------------------------------------- ##
## Custom (flagged): pyroki ships no self-collision *constraint*; this wraps
## the pyroki-native residual in jaxls's augmented-Lagrangian machinery,
## exactly as pyroki does for world collision.
import jaxls as _jaxls
from pyroki._residuals import self_collision_residual as _self_coll_res

self_collision_constraint = _jaxls.Cost.factory(kind="constraint_leq_zero")(
    _self_coll_res
)

COLLISION_MARGIN_SPHERE = 0.020  # m — Phase-9 winner (window 20–25 mm)
COLLISION_WEIGHT_SPHERE = 100.0  # Phase-9: first weight that stops the
                                 # designed crossing; zero false repulsion


class CollisionIK:
    """Baseline + self-collision on a supplied collision model.

    formulation: 'soft'  — pyroki self_collision_cost (margin hinge, weight)
                 'al'    — augmented-Lagrangian constraint at the margin
    margin/weight are runtime arguments (no recompile across the sweep)."""

    def __init__(self, robot, target_link_names, robot_coll,
                 formulation: str = "soft", max_iterations: int = 100):
        assert formulation in ("soft", "al")
        self.robot = robot
        self.max_iterations = max_iterations
        self.target_indices = np.asarray(
            [robot.links.names.index(n) for n in target_link_names], dtype=np.int32
        )
        target_idx_jax = jnp.asarray(self.target_indices)

        @jdc.jit
        def _solve(q_init, target_positions, target_wxyzs,
                   pos_weights, ori_weights, margin, weight):
            joint_var = robot.joint_var_cls(0)
            costs = []
            for i in range(NUM_TARGETS):
                costs.append(
                    pk.costs.pose_cost_analytic_jac(
                        robot, joint_var,
                        jaxlie.SE3.from_rotation_and_translation(
                            jaxlie.SO3(target_wxyzs[i]), target_positions[i]
                        ),
                        target_idx_jax[i],
                        pos_weight=pos_weights[i], ori_weight=ori_weights[i],
                    )
                )
            costs.append(pk.costs.limit_constraint(robot, joint_var))
            if formulation == "soft":
                costs.append(
                    pk.costs.self_collision_cost(
                        robot=robot, robot_coll=robot_coll,
                        joint_var=joint_var, margin=margin, weight=weight,
                    )
                )
            else:
                costs.append(
                    self_collision_constraint(
                        robot=robot, robot_coll=robot_coll,
                        joint_var=joint_var, margin=margin, weight=1.0,
                    )
                )
            sol, summary = (
                jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make([joint_var.with_value(q_init)]),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
                    termination=jaxls.TerminationConfig(max_iterations=max_iterations),
                    return_summary=True,
                )
            )
            return sol[joint_var], summary.iterations, summary.termination_criteria

        self._solve_jax = _solve
        self.margin = COLLISION_MARGIN_SPHERE
        self.weight = COLLISION_WEIGHT_SPHERE

    def solve(self, q_init, target_positions, target_wxyzs,
              pos_weights=None, ori_weights=None,
              margin=None, weight=None) -> SolveResult:
        pos = np.asarray(target_positions, dtype=np.float32).reshape(NUM_TARGETS, 3)
        wxyz = np.asarray(target_wxyzs, dtype=np.float32).reshape(NUM_TARGETS, 4)
        wxyz = wxyz / np.linalg.norm(wxyz, axis=-1, keepdims=True)
        pw = (np.full(NUM_TARGETS, DEFAULT_POS_WEIGHT, dtype=np.float32)
              if pos_weights is None else np.asarray(pos_weights, dtype=np.float32))
        ow = (np.full(NUM_TARGETS, DEFAULT_ORI_WEIGHT, dtype=np.float32)
              if ori_weights is None else np.asarray(ori_weights, dtype=np.float32))
        t0 = time.perf_counter()
        q, iters, term = self._solve_jax(
            q_init=jnp.asarray(np.asarray(q_init, dtype=np.float32)),
            target_positions=jnp.asarray(pos),
            target_wxyzs=jnp.asarray(wxyz),
            pos_weights=jnp.asarray(pw),
            ori_weights=jnp.asarray(ow),
            margin=jnp.float32(self.margin if margin is None else margin),
            weight=jnp.float32(self.weight if weight is None else weight),
        )
        q = q.block_until_ready()
        return SolveResult(
            q=np.asarray(q, dtype=np.float32),
            iterations=int(iters),
            termination=np.asarray(term, dtype=bool).reshape(-1),
            solve_ms=(time.perf_counter() - t0) * 1e3,
        )

    def warmup(self, q0):
        fk = self.robot.forward_kinematics(np.asarray(q0, dtype=np.float32))
        pos = np.stack([np.asarray(jaxlie.SE3(fk[i]).translation())
                        for i in self.target_indices])
        wxyz = np.stack([np.asarray(jaxlie.SE3(fk[i]).rotation().wxyz)
                         for i in self.target_indices])
        self.solve(q0, pos, wxyz)


def make(name: str, robot):
    """Registry hook for run_study.py."""
    from pathlib import Path as _Path

    _results = _Path(__file__).resolve().parent / "results"
    if name in ("collision_sphere", "collision_sphere_al", "collision_capsule"):
        from collision_models import (
            pruned_sphere_collision,
            pruned_tight_capsule_collision,
        )

        _, urdf = rm.load(with_urdf=True)
        if name == "collision_capsule":
            rc = pruned_tight_capsule_collision(urdf)
            geom = "tight capsules (421 link pairs, Part-1 pruning)"
        else:
            rc = pruned_sphere_collision(urdf, _results)
            geom = "180 spheres, 5884 corpus-pruned pairs"
        form = "al" if name.endswith("_al") else "soft"
        ik = CollisionIK(robot, rm.TARGET_LINKS, rc, formulation=form)
        return ik, {
            "pos_weight": DEFAULT_POS_WEIGHT,
            "ori_weight": DEFAULT_ORI_WEIGHT,
            "max_iterations": 100,
            "lambda_initial": 1.0,
            "linear_solver": "dense_cholesky",
            "warm_start": "previous commanded configuration",
            "solver": f"baseline + self-collision [{form}] on {geom}",
            "collision_margin": COLLISION_MARGIN_SPHERE,
            "collision_weight": COLLISION_WEIGHT_SPHERE if form == "soft" else None,
            "num_pairs": len(rc.active_idx_i),
        }
    if name in COMBOS_SPHERE:
        from collision_models import pruned_sphere_collision

        _, urdf = rm.load(with_urdf=True)
        rc = pruned_sphere_collision(urdf, _results)
        extras = {e: COMBO_WEIGHTS[e] for e in COMBOS_SPHERE[name]}
        extras["collision"] = SPHERE_COLLISION_WEIGHT
        ik = ComboIK(robot, rm.TARGET_LINKS, extras, robot_coll=rc,
                     collision_margin=SPHERE_COLLISION_MARGIN)
        return ik, {
            "pos_weight": DEFAULT_POS_WEIGHT,
            "ori_weight": DEFAULT_ORI_WEIGHT,
            "max_iterations": 100,
            "lambda_initial": 1.0,
            "linear_solver": "dense_cholesky",
            "warm_start": "previous commanded configuration",
            "solver": f"baseline + {'+'.join(COMBOS_SPHERE[name])} "
                      "(validated sphere collision)",
            "extras": extras,
            "collision_margin": SPHERE_COLLISION_MARGIN,
            "num_pairs": len(rc.active_idx_i),
        }
    if name in COMBOS:
        extras = {e: COMBO_WEIGHTS[e] for e in COMBOS[name]}
        robot_coll = None
        if "collision" in extras:
            _, urdf = rm.load(with_urdf=True)
            robot_coll = RobotCollision.from_urdf(urdf)
        ik = ComboIK(robot, rm.TARGET_LINKS, extras, robot_coll=robot_coll)
        return ik, {
            "pos_weight": DEFAULT_POS_WEIGHT,
            "ori_weight": DEFAULT_ORI_WEIGHT,
            "max_iterations": 100,
            "lambda_initial": 1.0,
            "linear_solver": "dense_cholesky",
            "warm_start": "previous commanded configuration",
            "solver": f"baseline + {'+'.join(COMBOS[name])} (Phase 5 weights)",
            "extras": extras,
            "collision_margin": COLLISION_MARGIN if "collision" in extras else None,
        }
    common = {
        "pos_weight": DEFAULT_POS_WEIGHT,
        "ori_weight": DEFAULT_ORI_WEIGHT,
        "max_iterations": 100,
        "lambda_initial": 1.0,
        "linear_solver": "dense_cholesky",
        "warm_start": "previous commanded configuration",
    }
    if name == "smoothing":
        ik = VariantIK(robot, rm.TARGET_LINKS, "smoothing")
        return ik, dict(
            common,
            solver="baseline + smoothing (user's velocity-scaled prev-q residual)",
            smoothing_weight=SMOOTHING_WEIGHT,
            nominal_velocity=SMOOTHING_V_NOMINAL,
            effective_scale=SMOOTHING_WEIGHT / (SMOOTHING_V_NOMINAL * DT),
        )
    if name == "centering":
        ik = VariantIK(robot, rm.TARGET_LINKS, "centering")
        return ik, dict(
            common,
            solver="baseline + joint centering (user's range-normalized residual)",
            centering_weight=CENTERING_WEIGHT,
        )
    if name == "collision":
        _, urdf = rm.load(with_urdf=True)
        robot_coll = RobotCollision.from_urdf(urdf)
        ik = VariantIK(robot, rm.TARGET_LINKS, "collision", robot_coll=robot_coll)
        return ik, dict(
            common,
            solver="baseline + self-collision (pyroki self_collision_cost)",
            collision_weight=COLLISION_WEIGHT,
            collision_margin=COLLISION_MARGIN,
            num_pairs=int(np.asarray(robot_coll.active_idx_i).shape[0]),
        )
    if name == "manipulability":
        ik = VariantIK(robot, rm.TARGET_LINKS, "manipulability")
        return ik, dict(
            common,
            solver="baseline + manipulability (pyroki Yoshikawa cost, hand arms)",
            manipulability_weight=MANIPULABILITY_WEIGHT,
        )
    raise SystemExit(f"unknown variant {name!r}")
