"""Batched evaluation: roll out N weight configurations simultaneously.

The solver's weights are already runtime arguments, so `jax.vmap` can evaluate a
whole population of weight sets against the same target trajectory in one pass.
Measured on an RTX 3090 with the `full` cost structure: 74 ms/tick sequentially
vs 1.48 ms/tick/config at N=96 -- a 50x throughput gain, which is what makes a
real weight search affordable.

Everything stays on device during a rollout; results transfer once at the end.

Timing caveat: per-configuration wall-clock is meaningless inside a batch (all
configurations share one kernel). Solver *iterations* are recorded as the
compute proxy, and the final chosen configuration should be re-timed
sequentially with `microbench.py` or a `compare` run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
from pyroki.collision import CollGeom, RobotCollision

import metrics as M
from solver import EPS, ControllerConfig, CostStructure, ThreeArmIK, Weights
from workloads import Workload

# The weight fields that vary across a population.  Order is fixed and is the
# canonical ordering used by the sampler, the CEM update and the reports.
WEIGHT_FIELDS = (
    "pos_left", "pos_right", "pos_middle",
    "ori_left", "ori_right", "ori_middle",
    "manip_left", "manip_right", "manip_middle",
    "smoothness", "rest", "self_collision", "world_collision",
    "limit_barrier", "collision_margin", "limit_barrier_margin",
)


def weights_to_vector(w: Weights) -> np.ndarray:
    return np.asarray(
        [
            *w.position, *w.orientation, *w.manipulability,
            w.smoothness, w.rest, w.self_collision, w.world_collision,
            w.limit_barrier, w.collision_margin, w.limit_barrier_margin,
        ],
        dtype=np.float32,
    )


def vector_to_weights(v: np.ndarray) -> Weights:
    v = np.asarray(v, dtype=np.float32)
    return Weights(
        position=tuple(float(x) for x in v[0:3]),
        orientation=tuple(float(x) for x in v[3:6]),
        manipulability=tuple(float(x) for x in v[6:9]),
        smoothness=float(v[9]),
        rest=float(v[10]),
        self_collision=float(v[11]),
        world_collision=float(v[12]),
        limit_barrier=float(v[13]),
        collision_margin=float(v[14]),
        limit_barrier_margin=float(v[15]),
    )


class BatchedIK:
    """Runs N weight configurations through the same workload in lockstep."""

    def __init__(
        self,
        robot: pk.Robot,
        robot_coll: Optional[RobotCollision],
        target_link_names: Sequence[str],
        structure: CostStructure,
        max_iterations: Optional[int] = 20,
    ) -> None:
        self.ik = ThreeArmIK(
            robot, robot_coll, tuple(target_link_names), structure,
            max_iterations=max_iterations,
        )
        self.robot = robot
        self.robot_coll = robot_coll
        self.num_joints = self.ik.num_joints
        self.target_indices = self.ik.target_indices
        self.arm_joint_mask = np.asarray(
            ["finger" not in n for n in robot.joints.actuated_names], dtype=bool
        )

        solve = self.ik._solve_jax
        inner = getattr(solve, "__wrapped__", solve)
        idx = jnp.asarray(self.target_indices)

        def one(prev_q, vel, weight_vec, targets_pos, targets_wxyz,
                max_dq, rest_q, obstacle, dt, vel_limit, acc_limit, clamp):
            q_sol, iters = inner(
                prev_q=prev_q,
                rest_q=rest_q,
                target_positions=targets_pos,
                target_wxyzs=targets_wxyz,
                max_dq=max_dq,
                pos_w=weight_vec[0:3],
                ori_w=weight_vec[3:6],
                manip_w=weight_vec[6:9],
                mask=jnp.ones(3, dtype=jnp.float32),
                smooth_w=weight_vec[9],
                rest_w=weight_vec[10],
                self_coll_w=weight_vec[11],
                world_coll_w=weight_vec[12],
                barrier_w=weight_vec[13],
                coll_margin=weight_vec[14],
                barrier_margin=weight_vec[15],
                world_obstacle=obstacle,
            )
            # Rate limiting, identical to the sequential controller.
            v_des = (q_sol - prev_q) / dt
            v_new = jnp.where(
                acc_limit > 0.0,
                vel + jnp.clip(v_des - vel, -acc_limit * dt, acc_limit * dt),
                v_des,
            )
            v_new = jnp.where(clamp, jnp.clip(v_new, -vel_limit, vel_limit), v_new)
            q_next = prev_q + v_new * dt
            return q_next, (q_next - prev_q) / dt, iters

        self._step_batch = jax.jit(
            jax.vmap(one, in_axes=(0, 0, 0, None, None, None, None, None,
                                   None, None, None, None))
        )

        # --- batched measurement, split by cost ----------------------------- #
        # Cheap (every step): forward kinematics, tracking error, limit margin.
        # These feed the jerk/lag/tracking metrics, which need every sample.
        def measure_cheap(q, targets_pos, targets_wxyz):
            fk = robot.forward_kinematics(q)
            se3 = jaxlie.SE3(fk[idx])
            ee_pos = se3.translation()
            ee_wxyz = se3.rotation().wxyz
            pos_err = jnp.linalg.norm(ee_pos - targets_pos, axis=-1)
            dot = jnp.abs(jnp.sum(ee_wxyz * targets_wxyz, axis=-1))
            ori_err = jnp.degrees(2.0 * jnp.arccos(jnp.clip(dot, 0.0, 1.0)))
            limit_margin = jnp.minimum(
                q - robot.joints.lower_limits, robot.joints.upper_limits - q
            )
            return ee_pos, pos_err, ori_err, limit_margin

        # Expensive (subsampled): 3 autodiff Jacobians plus every collision pair.
        # These feed min/mean statistics, which subsample without material loss.
        def measure_expensive(q, obstacle):
            def manip(link_index):
                jac_t = jax.jacfwd(
                    lambda qq: jaxlie.SE3(robot.forward_kinematics(qq)).translation()
                )(q)[link_index]
                return jnp.sqrt(jnp.maximum(0.0, jnp.linalg.det(jac_t @ jac_t.T)))

            manip_t = jax.vmap(manip)(idx)
            if robot_coll is not None:
                self_clear = jnp.min(robot_coll.compute_self_collision_distance(robot, q))
                world_clear = jnp.min(
                    robot_coll.compute_world_collision_distance(robot, q, obstacle)
                )
            else:
                self_clear = jnp.float32(jnp.nan)
                world_clear = jnp.float32(jnp.nan)
            return manip_t, self_clear, world_clear

        self._measure_cheap = jax.jit(jax.vmap(measure_cheap, in_axes=(0, None, None)))
        self._measure_expensive = jax.jit(jax.vmap(measure_expensive, in_axes=(0, None)))

    # ---------------------------------------------------------------------- #
    def rollout(
        self,
        weight_matrix: np.ndarray,  # (N, len(WEIGHT_FIELDS))
        workload: Workload,
        controller: ControllerConfig,
        q0: np.ndarray,
        measure_every: int = 5,
    ) -> List[M.Rollout]:
        """Roll the whole population through one workload; one Rollout each.

        `measure_every` subsamples only the expensive metrics (manipulability,
        collision clearance).  Tracking, jerk and lag are still measured every
        step, so nothing that depends on consecutive samples is degraded.
        """
        n = weight_matrix.shape[0]
        w = jnp.asarray(np.asarray(weight_matrix, dtype=np.float32))
        q = jnp.tile(jnp.asarray(np.asarray(q0, dtype=np.float32)), (n, 1))
        vel = jnp.zeros_like(q)
        rest_q = jnp.asarray(np.asarray(q0, dtype=np.float32))
        max_dq = jnp.full(
            self.num_joints,
            max(controller.nominal_velocity * controller.dt, EPS),
            dtype=jnp.float32,
        )
        dt = jnp.float32(controller.dt)
        vel_limit = jnp.float32(
            controller.velocity_limit if np.isfinite(controller.velocity_limit) else 1e6
        )
        acc_limit = jnp.float32(controller.accel_limit)
        clamp = jnp.bool_(controller.clamp_velocity)

        buf: Dict[str, list] = {
            k: [] for k in
            ("ee_pos", "pos_err", "ori_err", "manip_t", "limit_margin",
             "self_clear", "world_clear", "q", "iters")
        }
        for t in range(len(workload)):
            tp = jnp.asarray(np.asarray(workload.positions[t], dtype=np.float32))
            tw = jnp.asarray(np.asarray(workload.wxyzs[t], dtype=np.float32))
            obstacle = workload.obstacle_at(t) or self.ik._dummy_obstacle
            q, vel, iters = self._step_batch(
                q, vel, w, tp, tw, max_dq, rest_q, obstacle,
                dt, vel_limit, acc_limit, clamp,
            )
            ee_pos, pos_err, ori_err, limit_margin = self._measure_cheap(q, tp, tw)
            buf["ee_pos"].append(ee_pos)
            buf["pos_err"].append(pos_err)
            buf["ori_err"].append(ori_err)
            buf["limit_margin"].append(limit_margin)
            buf["q"].append(q)
            buf["iters"].append(iters)

            if t % measure_every == 0 or t == len(workload) - 1:
                manip_t, self_clear, world_clear = self._measure_expensive(q, obstacle)
                buf["manip_t"].append(manip_t)
                buf["self_clear"].append(self_clear)
                buf["world_clear"].append(world_clear)

        # One device->host transfer for the whole rollout.
        stacked = {k: np.asarray(jnp.stack(v)) for k, v in buf.items()}

        rollouts: List[M.Rollout] = []
        for i in range(n):
            r = M.Rollout(dt=controller.dt)
            r.q = list(stacked["q"][:, i])
            r.ee_pos = list(stacked["ee_pos"][:, i])
            r.pos_err = list(stacked["pos_err"][:, i])
            r.ori_err = list(stacked["ori_err"][:, i])
            # Subsampled series: shorter than the rollout, which is fine because
            # summarize() only takes min/mean over them.
            r.manip_t = list(stacked["manip_t"][:, i])
            r.manip_6 = list(stacked["manip_t"][:, i])  # 6D omitted in batch mode
            r.cond = list(np.zeros_like(stacked["manip_t"][:, i]))
            r.limit_margin = list(stacked["limit_margin"][:, i][:, self.arm_joint_mask])
            r.self_clear = list(stacked["self_clear"][:, i])
            r.world_clear = list(stacked["world_clear"][:, i])
            r.target_pos = list(np.asarray(workload.positions, dtype=np.float64))
            r.target_wxyz = list(np.asarray(workload.wxyzs, dtype=np.float64))
            r.iterations = list(stacked["iters"][:, i])
            r.solve_ms = []  # meaningless inside a batch; see module docstring
            rollouts.append(r)
        return rollouts

    def evaluate(
        self,
        weight_matrix: np.ndarray,
        workload_list: Sequence[Workload],
        controller: ControllerConfig,
        q0: np.ndarray,
        measure_every: int = 5,
    ) -> List[Dict[str, Dict[str, float]]]:
        """Per configuration: {workload_name: metrics}."""
        n = weight_matrix.shape[0]
        out: List[Dict[str, Dict[str, float]]] = [{} for _ in range(n)]
        for workload in workload_list:
            rollouts = self.rollout(weight_matrix, workload, controller, q0, measure_every)
            for i, r in enumerate(rollouts):
                out[i][workload.name] = M.summarize(
                    r,
                    velocity_limit=controller.nominal_velocity,
                    arm_joint_mask=self.arm_joint_mask,
                )
        return out
