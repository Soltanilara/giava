"""Configurable three-arm PyRoki IK solver used by the benchmark.

The solver is built once per *cost structure* (which cost terms are enabled) and
JIT-compiled.  All weights, margins and controller gains stay runtime arguments,
so a weight sweep re-uses a single compilation.

Cost terms available
--------------------
pose            : position + orientation tracking, per arm (analytic Jacobian).
limit_constraint: hard-ish joint limit via augmented Lagrangian (pyroki builtin).
manipulability  : 1 / Yoshikawa (translational) per arm, keeps arms away from
                  singular configurations.
self_collision  : robot-vs-robot signed-distance penalty.
world_collision : robot-vs-obstacle signed-distance penalty.
smoothness      : previous-configuration regularization, scaled by the per-joint
                  velocity budget (dq_weight / (v_max * dt)).  This is what makes
                  the solution continuous across control ticks.
rest            : posture bias towards a nominal configuration; resolves the
                  redundancy of the 6-DoF arms + 7-DoF camera arm consistently.
limit_barrier   : custom soft barrier that starts pushing back *before* a joint
                  reaches its limit, unlike `limit_constraint` which only reacts
                  on violation.  Prevents the "arm parks on a limit and stops
                  responding" failure mode during teleoperation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
from pyroki.collision import CollGeom, RobotCollision, Sphere

EPS = 1e-6
NUM_TARGETS = 3  # left, right, middle (camera)


# --------------------------------------------------------------------------- #
# Custom residuals
# --------------------------------------------------------------------------- #
@jaxls.Cost.create_factory
def _smoothness_scaled_residual(vals, joint_var, prev_q, scales):
    """Previous-configuration regularization normalized by the velocity budget."""
    return scales * (vals[joint_var] - prev_q)


@jaxls.Cost.create_factory
def _centering_residual(vals, robot, joint_var, weight, joint_mask):
    """Bias each joint towards the centre of its own range.

    residual_i = weight * (q_i - mid_i) / half_range_i

    Normalizing by each joint's half-range is what makes this different from a
    plain rest-pose cost: it is dimensionless, so a joint with a narrow range and
    a joint with a wide one feel equal pressure, and the cost reads directly as
    "fraction of the available travel used up".

    Compared with the manipulability cost this is cheaper (no Jacobian, no
    determinant), better conditioned (linear residual, no 1/(w+eps) blow-up at
    singularities) and targets the quantity that actually matters for continued
    teleoperation: how much joint travel is left in every direction.
    """
    q = vals[joint_var]
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits
    mid = 0.5 * (lower + upper)
    half_range = jnp.maximum(0.5 * (upper - lower), EPS)
    return (weight * joint_mask * (q - mid) / half_range).flatten()


@jaxls.Cost.create_factory
def _limit_barrier_residual(vals, robot, joint_var, margin, weight, joint_mask):
    """Soft barrier that activates within `margin` (rad) of a joint limit.

    residual_i = weight * relu(margin - d_i) / margin, with d_i the distance of
    joint i to its nearest limit.  Zero in the interior, growing linearly (and
    quadratically in the cost) as the joint approaches its limit.
    """
    q = vals[joint_var]
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits
    dist = jnp.minimum(q - lower, upper - q)
    violation = jnp.maximum(0.0, margin - dist) / jnp.maximum(margin, EPS)
    return (weight * joint_mask * violation).flatten()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostStructure:
    """Which cost terms exist in the problem (static: changes trigger a recompile)."""

    pose: bool = True
    limit_constraint: bool = True
    manipulability: bool = False
    self_collision: bool = False
    world_collision: bool = False
    world_collision_hard: bool = False
    """Enforce world collision as an augmented-Lagrangian constraint rather
    than a soft cost.

    A soft cost cannot guarantee separation: it is one term in a weighted sum,
    so a pose target that demands penetration gets penetration once the pose
    weight outweighs it.  Measured on `obstacle_sweep`, a soft world-collision
    weight of 50 against pose weights of 48-81 still left -67mm of penetration.
    The hard form keeps a Lagrange multiplier that ratchets up until the
    constraint holds, so the obstacle actually stops the arm.

    The cost is outer-loop iterations: each augmented-Lagrangian update needs
    its own inner LM solve.  Worth it for a rigid obstacle, which is why pyroki
    uses the constraint form in its single-shot IK example and the soft form in
    trajectory optimization.
    """
    smoothness: bool = True
    rest: bool = False
    limit_barrier: bool = False
    centering: bool = False

    def key(self) -> Tuple:
        return (
            self.pose,
            self.limit_constraint,
            self.manipulability,
            self.self_collision,
            self.world_collision,
            self.world_collision_hard,
            self.smoothness,
            self.rest,
            self.limit_barrier,
            self.centering,
        )


@dataclass
class Weights:
    """Runtime weights.  Sweeping these never recompiles."""

    # per-arm (left, right, middle)
    position: Tuple[float, float, float] = (50.0, 50.0, 40.0)
    orientation: Tuple[float, float, float] = (5.0, 5.0, 1.0)
    manipulability: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    active_mask: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    # scalars
    smoothness: float = 0.5
    rest: float = 0.0
    self_collision: float = 0.0
    world_collision: float = 0.0
    limit_barrier: float = 0.0
    centering: float = 0.0
    # margins
    collision_margin: float = 0.03
    limit_barrier_margin: float = 0.20  # rad

    def replace(self, **kw) -> "Weights":
        return replace(self, **kw)

    def flat(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for arm_i, arm in enumerate(("left", "right", "middle")):
            out[f"w_pos_{arm}"] = self.position[arm_i]
            out[f"w_ori_{arm}"] = self.orientation[arm_i]
            out[f"w_manip_{arm}"] = self.manipulability[arm_i]
        out.update(
            w_smooth=self.smoothness,
            w_rest=self.rest,
            w_self_coll=self.self_collision,
            w_world_coll=self.world_collision,
            w_limit_barrier=self.limit_barrier,
            w_centering=self.centering,
            collision_margin=self.collision_margin,
            limit_barrier_margin=self.limit_barrier_margin,
        )
        return out


@dataclass
class ControllerConfig:
    """Loop-level knobs that sit outside the optimizer but shape felt behaviour.

    On the safety limits: motor overload is a torque problem, and joint torque
    tracks *acceleration*, not velocity.  A velocity cap bounds acceleration only
    indirectly and costs responsiveness across the whole workspace.  An explicit
    acceleration limit bounds the quantity that actually overloads the motor,
    which allows a much higher velocity ceiling for the same peak torque.

    Note also that collision costs do nothing for this failure mode: they prevent
    geometric contact, not electrical or thermal overload.  Loosening the
    velocity limit because collision avoidance is now active would be trading
    against the wrong risk.
    """

    dt: float = 0.05
    velocity_limit: float = 2.0  # rad/s, uniform cap applied to every joint
    clamp_velocity: bool = True  # hard clip of dq after the solve
    nominal_velocity: float = 2.0
    """Velocity scale used to normalize the smoothness cost and the saturation
    metric.  Deliberately separate from `velocity_limit`: the clamp is a
    post-solve safety action, and changing or disabling it must not silently
    rescale the cost function, or a limits sweep would be comparing different
    optimization problems rather than different limits."""
    accel_limit: float = 0.0  # rad/s^2; 0 disables. The motor-protection knob.
    target_lpf: float = 0.0  # EMA on the incoming target (0 = off, 1 = frozen)
    output_lpf: float = 0.0  # EMA on the commanded joints (0 = off)

    # --- escape from local minima ----------------------------------------- #
    reseed_error_m: float = 0.0
    """Position error (m) above which the solver is considered stuck. 0 = off.

    Levenberg-Marquardt is a *local* method warm-started from the previous
    configuration.  After chasing an unreachable target the arm ends up in a
    stretched configuration, and returning to a reachable target can require
    passing through a higher-cost region (re-articulating the elbow) that a
    descent method will not climb.  No choice of weights fixes this -- measured
    directly: lowering smoothness, disabling manipulability and adding a rest
    cost all stayed stuck at ~86-95 mm, while re-solving from the home
    configuration recovered to 6.5 mm.

    When the error stays above this threshold for `reseed_after_ticks`, the tick
    is solved a second time from `rest`/home as the initial guess and the better
    of the two solutions is kept.  This is a one-extra-solve multi-start, not a
    different cost function.
    """
    reseed_after_ticks: int = 5
    """Consecutive stuck ticks before a re-seed is attempted."""
    recovery_max_ticks: int = 40
    """Give up on a latched recovery goal after this many ticks."""

    # --- workspace-boundary clutch ---------------------------------------- #
    freeze_error_m: float = 0.0
    """Freeze at the workspace boundary when tracking error exceeds this. 0 = off.

    Prevention rather than cure.  Chasing an unreachable target is what drags the
    arm into the stretched, near-singular configuration that later traps the
    solver, so the cheapest fix is not to chase it: hold the last reachable
    target, leave the arm parked at the boundary, and wait for the operator to
    bring their hand back.  This is the usual teleoperation clutch, and unlike a
    recovery scheme it never puts the robot somewhere it has to escape from.
    """
    freeze_after_ticks: int = 5
    """Consecutive over-threshold ticks before freezing.  Instantaneous error is
    not a reachability test: fast hand motion transiently exceeds any sensible
    threshold while the arm catches up, so freezing on a single tick would clutch
    constantly during normal use."""
    reengage_m: float = 0.05
    """Re-engage once the commanded position comes back within this distance of
    where the end effector actually is.  Motion resumes from where the operator
    picks it up, so control is continuous rather than jumping."""


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class ThreeArmIK:
    """Stateful three-arm IK controller with a configurable cost structure."""

    ARM_NAMES = ("left", "right", "middle")

    def __init__(
        self,
        robot: pk.Robot,
        robot_coll: Optional[RobotCollision],
        target_link_names: Tuple[str, str, str],
        structure: CostStructure,
        linear_solver: str = "dense_cholesky",
        lambda_initial: float = 1.0,
        max_iterations: Optional[int] = None,
    ) -> None:
        """`max_iterations` bounds the Levenberg-Marquardt budget.

        Two reasons to set it.  A 20 Hz teleoperation loop cannot afford an
        unbounded solve anyway, so a budget is realistic rather than a
        compromise.  And under `jax.vmap` the iteration loop cannot exit until
        *every* configuration in the batch has converged, so one slow-converging
        weight set makes the whole population pay its cost -- an unbounded solve
        makes batched search far slower than the per-config timing suggests.
        """
        self.robot = robot
        self.robot_coll = robot_coll
        self.structure = structure
        self.num_joints = robot.joints.num_actuated_joints
        self.target_indices = np.asarray(
            [robot.links.names.index(n) for n in target_link_names], dtype=np.int32
        )
        if (structure.self_collision or structure.world_collision) and robot_coll is None:
            raise ValueError("robot_coll is required for collision costs.")

        target_idx_jax = jnp.asarray(self.target_indices)
        # Gripper fingers are actuated but not part of the arm kinematics: they
        # live inside a 0..0.041 rad range, so a limit barrier would fire on them
        # permanently.  Exclude them from the barrier.
        self.arm_joint_mask = np.asarray(
            [
                0.0 if "finger" in name else 1.0
                for name in robot.joints.actuated_names
            ],
            dtype=np.float32,
        )
        arm_mask_jax = jnp.asarray(self.arm_joint_mask)
        rest_default = jnp.asarray(
            np.asarray(robot.joint_var_cls(0).default_factory(), dtype=np.float32)
        )

        @jdc.jit
        def _solve(
            prev_q: jax.Array,
            rest_q: jax.Array,
            target_positions: jax.Array,
            target_wxyzs: jax.Array,
            max_dq: jax.Array,
            pos_w: jax.Array,
            ori_w: jax.Array,
            manip_w: jax.Array,
            mask: jax.Array,
            smooth_w: jax.Array,
            rest_w: jax.Array,
            self_coll_w: jax.Array,
            world_coll_w: jax.Array,
            barrier_w: jax.Array,
            centering_w: jax.Array,
            coll_margin: jax.Array,
            barrier_margin: jax.Array,
            world_obstacle: CollGeom,
        ) -> jax.Array:
            joint_var = robot.joint_var_cls(0)
            costs = []

            for i in range(NUM_TARGETS):
                if structure.pose:
                    costs.append(
                        pk.costs.pose_cost_analytic_jac(
                            robot,
                            joint_var,
                            jaxlie.SE3.from_rotation_and_translation(
                                jaxlie.SO3(target_wxyzs[i]), target_positions[i]
                            ),
                            target_idx_jax[i],
                            pos_weight=pos_w[i] * mask[i],
                            ori_weight=ori_w[i] * mask[i],
                        )
                    )
                if structure.manipulability:
                    costs.append(
                        pk.costs.manipulability_cost(
                            robot=robot,
                            joint_var=joint_var,
                            target_link_indices=target_idx_jax[i],
                            weight=manip_w[i] * mask[i],
                        )
                    )

            if structure.limit_constraint:
                costs.append(pk.costs.limit_constraint(robot, joint_var))
            if structure.limit_barrier:
                costs.append(
                    _limit_barrier_residual(
                        robot=robot,
                        joint_var=joint_var,
                        margin=barrier_margin,
                        weight=barrier_w,
                        joint_mask=arm_mask_jax,
                    )
                )
            if structure.centering:
                costs.append(
                    _centering_residual(
                        robot=robot,
                        joint_var=joint_var,
                        weight=centering_w,
                        joint_mask=arm_mask_jax,
                    )
                )
            if structure.self_collision:
                costs.append(
                    pk.costs.self_collision_cost(
                        robot=robot,
                        robot_coll=robot_coll,
                        joint_var=joint_var,
                        margin=coll_margin,
                        weight=self_coll_w,
                    )
                )
            if structure.world_collision:
                if structure.world_collision_hard:
                    # No weight: a constraint is not traded off against the
                    # other terms, so `world collision` in the GUI has no
                    # effect in this mode. The margin still sets how early the
                    # constraint starts pushing.
                    costs.append(
                        pk.costs.world_collision_constraint(
                            robot,
                            robot_coll,
                            joint_var,
                            world_obstacle,
                            coll_margin,
                        )
                    )
                else:
                    costs.append(
                        pk.costs.world_collision_cost(
                            robot=robot,
                            robot_coll=robot_coll,
                            joint_var=joint_var,
                            world_geom=world_obstacle,
                            margin=coll_margin,
                            weight=world_coll_w,
                        )
                    )
            if structure.smoothness:
                costs.append(
                    _smoothness_scaled_residual(
                        joint_var=joint_var,
                        prev_q=prev_q,
                        scales=smooth_w / jnp.maximum(max_dq, EPS),
                    )
                )
            if structure.rest:
                costs.append(
                    pk.costs.rest_cost(
                        joint_var,
                        rest_pose=rest_q,
                        weight=rest_w,
                    )
                )

            sol, summary = (
                jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_q)]),
                    verbose=False,
                    linear_solver=linear_solver,
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=lambda_initial),
                    return_summary=True,
                    **(
                        {}
                        if max_iterations is None
                        else {
                            "termination": jaxls.TerminationConfig(
                                max_iterations=max_iterations
                            )
                        }
                    ),
                )
            )
            return sol[joint_var], summary.iterations

        self._solve_jax = _solve
        self._rest_default = rest_default
        self._dummy_obstacle = Sphere.from_center_and_radius(
            np.zeros(3, dtype=np.float32), np.asarray([1e-3], dtype=np.float32)
        )
        # controller state
        self.reset()

    # -- state ------------------------------------------------------------- #
    def reset(self, q0: Optional[np.ndarray] = None) -> None:
        if q0 is None:
            q0 = np.asarray(
                self.robot.joint_var_cls(0).default_factory(), dtype=np.float32
            )
        self.q = np.asarray(q0, dtype=np.float32).copy()
        self.v = np.zeros_like(self.q)
        self._stuck_ticks = 0
        self._recovery_goal: Optional[np.ndarray] = None
        self._recovery_ticks = 0
        self.last_reseeded = False
        self._held_target: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._over_ticks = 0
        self.frozen = False
        self._filtered_targets: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # -- one control tick --------------------------------------------------- #
    def step(
        self,
        target_positions: np.ndarray,  # (3, 3)
        target_wxyzs: np.ndarray,  # (3, 4)
        weights: Weights,
        controller: ControllerConfig,
        rest_q: Optional[np.ndarray] = None,
        world_obstacle: Optional[CollGeom] = None,
    ) -> np.ndarray:
        pos = np.asarray(target_positions, dtype=np.float32).reshape(NUM_TARGETS, 3)
        wxyz = np.asarray(target_wxyzs, dtype=np.float32).reshape(NUM_TARGETS, 4)
        wxyz = wxyz / np.linalg.norm(wxyz, axis=-1, keepdims=True)

        if controller.target_lpf > 0.0:
            a = float(controller.target_lpf)
            if self._filtered_targets is None:
                self._filtered_targets = (pos.copy(), wxyz.copy())
            fp, fw = self._filtered_targets
            pos = a * fp + (1.0 - a) * pos
            # slerp-free quaternion EMA + renormalize; sign-align first.
            sign = np.sign(np.sum(fw * wxyz, axis=-1, keepdims=True))
            sign[sign == 0] = 1.0
            wxyz = a * fw + (1.0 - a) * (sign * wxyz)
            wxyz = wxyz / np.linalg.norm(wxyz, axis=-1, keepdims=True)
            self._filtered_targets = (pos.copy(), wxyz.copy())

        # Smoothness normalization uses the nominal velocity, never the clamp.
        # --- workspace-boundary clutch ------------------------------------- #
        if controller.freeze_error_m > 0.0:
            ee_now = self._ee_positions(self.q)
            if self.frozen:
                back_in_range = bool(
                    np.max(np.linalg.norm(pos[:2] - ee_now[:2], axis=-1))
                    < controller.reengage_m
                )
                if back_in_range:
                    self.frozen = False
                    self._held_target = None
                elif self._held_target is not None:
                    pos, wxyz = self._held_target  # keep commanding the boundary
            # Freeze decision happens after the solve, below, where the achieved
            # error is known.

        max_dq = np.full(
            self.num_joints,
            max(controller.nominal_velocity * controller.dt, EPS),
            dtype=np.float32,
        )
        rest = (
            self._rest_default
            if rest_q is None
            else jnp.asarray(np.asarray(rest_q, dtype=np.float32))
        )
        obstacle = world_obstacle if world_obstacle is not None else self._dummy_obstacle

        q_sol, iters = self._solve_jax(
            prev_q=jnp.asarray(self.q),
            rest_q=rest,
            target_positions=jnp.asarray(pos),
            target_wxyzs=jnp.asarray(wxyz),
            max_dq=jnp.asarray(max_dq),
            pos_w=jnp.asarray(np.asarray(weights.position, dtype=np.float32)),
            ori_w=jnp.asarray(np.asarray(weights.orientation, dtype=np.float32)),
            manip_w=jnp.asarray(np.asarray(weights.manipulability, dtype=np.float32)),
            mask=jnp.asarray(np.asarray(weights.active_mask, dtype=np.float32)),
            smooth_w=jnp.float32(weights.smoothness),
            rest_w=jnp.float32(weights.rest),
            self_coll_w=jnp.float32(weights.self_collision),
            world_coll_w=jnp.float32(weights.world_collision),
            barrier_w=jnp.float32(weights.limit_barrier),
            centering_w=jnp.float32(weights.centering),
            coll_margin=jnp.float32(weights.collision_margin),
            barrier_margin=jnp.float32(weights.limit_barrier_margin),
            world_obstacle=obstacle,
        )
        q_sol = q_sol.block_until_ready()
        self.last_iterations = int(iters)
        self.last_reseeded = False

        # --- multi-start escape from a local minimum ----------------------- #
        if controller.reseed_error_m > 0.0:
            err = self._position_error(np.asarray(q_sol, dtype=np.float32), pos)
            if err > controller.reseed_error_m:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = 0
            if self._stuck_ticks >= controller.reseed_after_ticks:
                q_alt, iters_alt = self._solve_jax(
                    prev_q=rest,  # second start: the nominal configuration
                    rest_q=rest,
                    target_positions=jnp.asarray(pos),
                    target_wxyzs=jnp.asarray(wxyz),
                    max_dq=jnp.asarray(max_dq),
                    pos_w=jnp.asarray(np.asarray(weights.position, dtype=np.float32)),
                    ori_w=jnp.asarray(np.asarray(weights.orientation, dtype=np.float32)),
                    manip_w=jnp.asarray(np.asarray(weights.manipulability, dtype=np.float32)),
                    mask=jnp.asarray(np.asarray(weights.active_mask, dtype=np.float32)),
                    smooth_w=jnp.float32(0.0),  # the spring to prev_q is what traps it
                    rest_w=jnp.float32(weights.rest),
                    self_coll_w=jnp.float32(weights.self_collision),
                    world_coll_w=jnp.float32(weights.world_collision),
                    barrier_w=jnp.float32(weights.limit_barrier),
                    centering_w=jnp.float32(weights.centering),
                    coll_margin=jnp.float32(weights.collision_margin),
                    barrier_margin=jnp.float32(weights.limit_barrier_margin),
                    world_obstacle=obstacle,
                )
                q_alt = np.asarray(q_alt.block_until_ready(), dtype=np.float32)
                if self._position_error(q_alt, pos) < err:
                    # Latch it as a recovery goal rather than applying it for a
                    # single tick.  Applying it once does not work: the rate
                    # limiter only allows a fraction of the way there, and the
                    # next tick's local solve -- warm-started from the
                    # barely-moved configuration -- falls straight back into the
                    # original basin.  Measured: single-tick re-seeding fired 15
                    # times and left the arm worse (159 mm vs 86 mm).
                    self._recovery_goal = q_alt
                    self._recovery_ticks = 0
                    self._stuck_ticks = 0

        q_next = np.asarray(q_sol, dtype=np.float32)

        if controller.freeze_error_m > 0.0 and not self.frozen:
            if self._position_error(q_next, pos) > controller.freeze_error_m:
                self._over_ticks += 1
            else:
                self._over_ticks = 0
                self._held_target = (pos.copy(), wxyz.copy())  # last reachable
            # Freeze only on a *persistent* miss, and only once there is a
            # reachable target to fall back to.
            if (
                self._over_ticks >= controller.freeze_after_ticks
                and self._held_target is not None
            ):
                self.frozen = True
                self._over_ticks = 0
                q_next = self.q.copy()  # stop stretching outwards

        # While a recovery goal is latched, steer towards it instead of towards
        # the local solution.  The rate limits below still apply, so the motion
        # is exactly as gentle as any other -- it just does not get re-decided
        # (and undone) every tick.
        if self._recovery_goal is not None:
            self._recovery_ticks += 1
            reached = float(np.max(np.abs(self._recovery_goal - self.q))) < 0.05
            expired = self._recovery_ticks > controller.recovery_max_ticks
            if reached or expired:
                self._recovery_goal = None
                self._recovery_ticks = 0
            else:
                q_next = self._recovery_goal
                self.last_reseeded = True

        # Rate limiting, applied on velocity: acceleration first (the torque
        # bound), then the velocity ceiling).
        vel = (q_next - self.q) / controller.dt
        if controller.accel_limit > 0.0:
            dv = controller.accel_limit * controller.dt
            vel = np.clip(self.v, -np.inf, np.inf) + np.clip(vel - self.v, -dv, dv)
        if controller.clamp_velocity:
            vel = np.clip(vel, -controller.velocity_limit, controller.velocity_limit)
        if controller.accel_limit > 0.0 or controller.clamp_velocity:
            q_next = self.q + vel * controller.dt

        if controller.output_lpf > 0.0:
            b = float(controller.output_lpf)
            q_next = b * self.q + (1.0 - b) * q_next

        self.v = (q_next - self.q) / controller.dt
        self.q = q_next
        return self.q.copy()

    def _ee_positions(self, q: np.ndarray) -> np.ndarray:
        """End-effector positions for all three targets. Shape (3, 3)."""
        fk = self.robot.forward_kinematics(np.asarray(q, dtype=np.float32))
        return np.stack(
            [np.asarray(jaxlie.SE3(fk[i]).translation()) for i in self.target_indices]
        )

    def _position_error(self, q: np.ndarray, target_pos: np.ndarray) -> float:
        """Worst arm position error [m] for a configuration (fingers excluded)."""
        fk = self.robot.forward_kinematics(np.asarray(q, dtype=np.float32))
        errs = []
        for k, link_index in enumerate(self.target_indices[:2]):  # the two arms
            p = np.asarray(jaxlie.SE3(fk[link_index]).translation())
            errs.append(float(np.linalg.norm(p - np.asarray(target_pos)[k])))
        return max(errs)

    def warmup(
        self,
        weights: Weights,
        controller: ControllerConfig,
        world_obstacle: Optional[CollGeom] = None,
    ) -> None:
        """Trigger JIT compilation so timings exclude it."""
        q0 = self.q.copy()
        fk = self.robot.forward_kinematics(q0)
        pos = np.stack(
            [np.asarray(jaxlie.SE3(fk[i]).translation()) for i in self.target_indices]
        )
        wxyz = np.stack(
            [
                np.asarray(jaxlie.SE3(fk[i]).rotation().wxyz)
                for i in self.target_indices
            ]
        )
        self.step(pos, wxyz, weights, controller, world_obstacle=world_obstacle)
        self.reset(q0)


# --------------------------------------------------------------------------- #
# Presets: the four families the study compares
# --------------------------------------------------------------------------- #
def preset(name: str) -> Tuple[CostStructure, Weights]:
    """Return (structure, default weights) for a named IK variant."""
    base_pos = (50.0, 50.0, 40.0)
    base_ori = (5.0, 5.0, 1.0)

    if name == "pure":
        # Pose + joint limits only.  No temporal coupling at all: the reference
        # for "what does raw absolute-pose IK feel like".
        return (
            CostStructure(smoothness=False),
            Weights(position=base_pos, orientation=base_ori, smoothness=0.0),
        )
    if name == "pure_smooth":
        # Pose + limits + previous-configuration regularization.
        return (
            CostStructure(smoothness=True),
            Weights(position=base_pos, orientation=base_ori, smoothness=0.5),
        )
    if name == "collision":
        return (
            CostStructure(smoothness=True, self_collision=True, world_collision=True),
            Weights(
                position=base_pos,
                orientation=base_ori,
                smoothness=0.5,
                self_collision=5.0,
                world_collision=5.0,
            ),
        )
    if name == "manipulability":
        return (
            CostStructure(smoothness=True, manipulability=True),
            Weights(
                position=base_pos,
                orientation=base_ori,
                smoothness=0.5,
                manipulability=(0.02, 0.02, 0.0),
            ),
        )
    if name == "custom":
        # Custom terms only (rest posture + soft limit barrier), no collision or
        # manipulability, to isolate their effect.
        return (
            CostStructure(smoothness=True, rest=True, limit_barrier=True),
            Weights(
                position=base_pos,
                orientation=base_ori,
                smoothness=0.5,
                rest=0.01,
                limit_barrier=1.0,
            ),
        )
    if name == "collision_manip":
        return (
            CostStructure(
                smoothness=True, self_collision=True, world_collision=True,
                manipulability=True,
            ),
            Weights(
                position=base_pos,
                orientation=base_ori,
                smoothness=0.5,
                manipulability=(0.02, 0.02, 0.0),
                self_collision=5.0,
                world_collision=5.0,
            ),
        )
    if name == "collision_custom":
        return (
            CostStructure(
                smoothness=True, self_collision=True, world_collision=True,
                rest=True, limit_barrier=True,
            ),
            Weights(
                position=base_pos,
                orientation=base_ori,
                smoothness=0.5,
                self_collision=5.0,
                world_collision=5.0,
                rest=0.01,
                limit_barrier=1.0,
            ),
        )
    if name == "manip_custom":
        return (
            CostStructure(
                smoothness=True, manipulability=True, rest=True, limit_barrier=True
            ),
            Weights(
                position=base_pos,
                orientation=base_ori,
                smoothness=0.5,
                manipulability=(0.02, 0.02, 0.0),
                rest=0.01,
                limit_barrier=1.0,
            ),
        )
    if name == "centering":
        return (
            CostStructure(smoothness=True, centering=True),
            Weights(
                position=base_pos, orientation=base_ori,
                smoothness=0.5, centering=0.05,
            ),
        )
    if name == "full":
        return (
            CostStructure(
                smoothness=True,
                manipulability=True,
                self_collision=True,
                world_collision=True,
                rest=True,
                limit_barrier=True,
                centering=True,
            ),
            Weights(
                position=base_pos,
                orientation=base_ori,
                smoothness=0.5,
                manipulability=(0.02, 0.02, 0.0),
                self_collision=5.0,
                world_collision=5.0,
                rest=0.01,
                limit_barrier=1.0,
                centering=0.05,
            ),
        )
    raise KeyError(f"Unknown preset: {name}")


PRESETS = (
    "pure",
    "pure_smooth",
    "collision",
    "manipulability",
    "custom",
    "collision_manip",
    "collision_custom",
    "manip_custom",
    "centering",
    "full",
)
