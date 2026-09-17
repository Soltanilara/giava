"""Phase 3 — evaluation metrics, computed independently of the solver.

Everything here is a *measurement* on a rollout history (the sequence of
commanded configurations), never part of any optimization:

Accuracy
    - position error: Euclidean ‖p_actual − p_target‖ per arm [m]
    - orientation error: geodesic angle ‖log(R_actualᵀ R_target)‖ per arm [rad]
      (NOT the SE(3)-log residual the solver minimizes — see BASELINE.md §2)
Compute
    - per-step solve wall-clock (mean/median/p95/max), LM iterations
Robustness
    - non-convergence: steps where no jaxls termination criterion fired and
      the iteration budget was exhausted
    - tracking miss: feasible steps with position error > 10 mm
    - joint-limit violations (any joint outside its range) and minimum margin
      to the nearest limit (fingers excluded — their 0–0.041 range would
      register as permanently near-limit)
Behaviour
    - joint velocity / acceleration / jerk RMS and max (finite differences at
      50 Hz, fingers excluded)
    - configuration jumps: steps with ‖Δq‖_∞ > max(5 × median ‖Δq‖_∞, 0.05 rad)
    - Yoshikawa translational manipulability w = √det(J Jᵀ) per hand arm
      (min / mean) and for the camera arm
    - minimum self-collision clearance (capsule model; pairs closer than 5 mm
      at the *home* configuration are ignored as approximation artifacts —
      the home pose is collision-free in reality)

Feasibility handling (pre-registered in TRAJECTORIES.md):
    - Steps whose commanded target lies beyond (max_reach − 5 mm) of the
      arm's shoulder are excluded from tracking-accuracy aggregates and
      scored as boundary/recovery behaviour instead:
      `recovery_ms` = time from the end of the infeasible segment until the
      arm's position error drops below 10 mm (NaN if it never does);
      `end_pos_err_mm` = mean position error over the final second.
    - A trajectory's `eval_start` (hand-park intro) is excluded everywhere.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
from pyroki.collision import RobotCollision

import robot_model as rm
from trajectories import SHOULDER_L, SHOULDER_M, SHOULDER_R, Trajectory

MAX_REACH_HAND = 0.758  # m, measured (validate_trajectories.py, 20k samples)
MAX_REACH_MID = 0.658
REACH_MARGIN = 0.005
MISS_THRESHOLD = 0.010  # m — "tracking miss" on feasible steps
JUMP_FLOOR = 0.05  # rad — configuration-jump absolute floor
SHOULDERS = (SHOULDER_L, SHOULDER_R, SHOULDER_M)
MAX_REACHES = (MAX_REACH_HAND, MAX_REACH_HAND, MAX_REACH_MID)


class Evaluator:
    """Batched (vmapped) metric computation for rollout histories.

    coll_model: None → pyroki capsule model (backwards-compatible with all
    earlier runs' `self_clearance_min_mm`); or a RobotCollision to use
    instead (e.g. the validated pruned sphere model — collision-study runs).
    """

    def __init__(self, robot: pk.Robot, urdf, coll_model=None) -> None:
        self._coll_override = coll_model
        self.robot = robot
        self.indices = rm.target_link_indices(robot)
        self.finger_mask = rm.finger_joint_mask(robot)
        self.arm_mask = ~self.finger_mask  # joints included in motion metrics
        self.lower = np.asarray(robot.joints.lower_limits)
        self.upper = np.asarray(robot.joints.upper_limits)

        idx = jnp.asarray(self.indices)

        @jax.jit
        def _pose_batch(qs, tgt_pos, tgt_wxyz):
            """qs (T,23), tgt_pos (T,3,3), tgt_wxyz (T,3,4) ->
            pos_err (T,3), ori_err (T,3)."""

            def one(q, tp, tw):
                fk = robot.forward_kinematics(q)
                T_links = jaxlie.SE3(fk[idx])  # batched over 3 targets
                p = T_links.translation()
                perr = jnp.linalg.norm(p - tp, axis=-1)
                R = T_links.rotation()
                rel = R.inverse() @ jaxlie.SO3(tw)
                oerr = jnp.linalg.norm(rel.log(), axis=-1)
                return perr, oerr

            return jax.vmap(one)(qs, tgt_pos, tgt_wxyz)

        @jax.jit
        def _manip_batch(qs):
            """Yoshikawa translational manipulability per arm, (T,3)."""

            def one_link(q, link_index):
                J = jax.jacfwd(
                    lambda qq: jaxlie.SE3(
                        robot.forward_kinematics(qq)
                    ).translation()[link_index]
                )(q)
                JJT = J @ J.T
                return jnp.sqrt(jnp.maximum(0.0, jnp.linalg.det(JJT)))

            def one(q):
                return jnp.stack([one_link(q, i) for i in self.indices])

            return jax.vmap(one)(qs)

        self._pose_batch = _pose_batch
        self._manip_batch = _manip_batch

        # --- self-collision clearance ---------------------------------- #
        self.robot_coll: Optional[RobotCollision] = None
        self._coll_keep: Optional[np.ndarray] = None
        try:
            self.robot_coll = (
                self._coll_override
                if self._coll_override is not None
                else RobotCollision.from_urdf(urdf)
            )
            q0 = rm.home_config(robot)
            d_home = np.asarray(
                self.robot_coll.compute_self_collision_distance(
                    robot, jnp.asarray(q0)
                )
            ).reshape(-1)
            # Ignore pairs already closer than 5 mm at the collision-free home
            # pose: capsule-approximation artifacts (adjacent geometry).
            self._coll_keep = d_home > 0.005
            # Pairs involving a middle-arm link, reported separately: during
            # mid_* trajectories the parked hands graze their own base
            # (−1.9 mm, constant), which would otherwise mask the camera
            # arm's clearance; and during grasp reaches the hands pass the
            # camera body — a deployment-relevant contact of its own.
            # active_idx are geometry-level in sphere mode: map to links
            g2l = np.asarray(self.robot_coll._geom_to_link_idx)
            pair_names = [
                (self.robot_coll.link_names[g2l[i]],
                 self.robot_coll.link_names[g2l[j]])
                for i, j in zip(
                    np.asarray(self.robot_coll.active_idx_i),
                    np.asarray(self.robot_coll.active_idx_j),
                )
            ]
            self._coll_mid = np.asarray(
                [a.startswith("middle") or b.startswith("middle")
                 for a, b in pair_names]
            )
            robot_coll = self.robot_coll

            @jax.jit
            def _clearance_batch(qs):
                def one(q):
                    d = robot_coll.compute_self_collision_distance(robot, q)
                    return d.reshape(-1)

                return jax.vmap(one)(qs)

            self._clearance_batch = _clearance_batch
        except Exception as e:  # pragma: no cover — collision model optional
            print(f"[metrics] self-collision model unavailable: {e}")
            self._clearance_batch = None

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        traj: Trajectory,
        qs: np.ndarray,  # (T, 23) commanded configurations
        solve_ms: np.ndarray,  # (T,)
        iterations: np.ndarray,  # (T,)
        terminations: np.ndarray,  # (T, 3) bool — [cost, gradient, param]
        max_iterations: int,
        with_collision: bool = True,
    ) -> Dict[str, float]:
        T = len(traj)
        assert qs.shape[0] == T
        s = traj.eval_start  # skip hand-park intro
        dt = traj.dt

        qs_j = jnp.asarray(qs.astype(np.float32))
        perr, oerr = self._pose_batch(
            qs_j,
            jnp.asarray(traj.positions.astype(np.float32)),
            jnp.asarray(traj.wxyzs.astype(np.float32)),
        )
        perr = np.asarray(perr)[s:]  # (T', 3)
        oerr = np.asarray(oerr)[s:]
        manip = np.asarray(self._manip_batch(qs_j))[s:]  # (T', 3)

        # --- feasibility mask per arm (commanded-target reach test) ----- #
        tgt = traj.positions[s:]  # (T', 3, 3)
        feas = np.stack(
            [
                np.linalg.norm(tgt[:, a] - SHOULDERS[a], axis=-1)
                < (MAX_REACHES[a] - REACH_MARGIN)
                for a in range(3)
            ],
            axis=1,
        )  # (T', 3) bool

        out: Dict[str, float] = {
            "trajectory": traj.name,
            "steps": T - s,
            "feasible_frac": float(feas.all(axis=1).mean()),
        }

        # --- accuracy on feasible steps -------------------------------- #
        hands_p = np.where(feas[:, :2], perr[:, :2], np.nan)
        hands_o = np.where(feas[:, :2], oerr[:, :2], np.nan)
        mid_p = np.where(feas[:, 2], perr[:, 2], np.nan)
        mid_o = np.where(feas[:, 2], oerr[:, 2], np.nan)

        def stats(x, scale, prefix):
            x = x[~np.isnan(x)]
            if x.size == 0:
                return {f"{prefix}_mean": np.nan}
            return {
                f"{prefix}_mean": float(np.mean(x)) * scale,
                f"{prefix}_median": float(np.median(x)) * scale,
                f"{prefix}_p95": float(np.percentile(x, 95)) * scale,
                f"{prefix}_max": float(np.max(x)) * scale,
            }

        out.update(stats(hands_p, 1e3, "pos_mm"))
        out.update(stats(np.degrees(hands_o), 1.0, "ori_deg"))
        out.update(stats(mid_p, 1e3, "mid_pos_mm"))
        out.update(stats(np.degrees(mid_o), 1.0, "mid_ori_deg"))
        out["miss_frac"] = float(np.nanmean(hands_p > MISS_THRESHOLD))

        # --- boundary / recovery for infeasible trajectories ------------ #
        infeas_any = ~feas.all(axis=1)
        if infeas_any.any():
            last_bad = int(np.max(np.nonzero(infeas_any)[0]))
            worst_after = perr[last_bad + 1 :].max(axis=1) if last_bad + 1 < len(perr) else np.array([np.nan])
            rec = np.nonzero(worst_after < MISS_THRESHOLD)[0]
            out["recovery_ms"] = float(rec[0] * dt * 1e3) if rec.size else float("nan")
            n_end = max(1, int(round(1.0 / dt)))
            out["end_pos_err_mm"] = float(perr[-n_end:].max(axis=1).mean() * 1e3)

        # --- compute ---------------------------------------------------- #
        tms = np.asarray(solve_ms)[s:]
        out.update(
            solve_ms_mean=float(tms.mean()),
            solve_ms_median=float(np.median(tms)),
            solve_ms_p95=float(np.percentile(tms, 95)),
            solve_ms_max=float(tms.max()),
            iters_mean=float(np.mean(iterations[s:])),
            iters_max=int(np.max(iterations[s:])),
        )

        # --- robustness -------------------------------------------------- #
        term = np.asarray(terminations)[s:]
        budget_exhausted = (~term.any(axis=1)) & (
            np.asarray(iterations)[s:] >= max_iterations
        )
        out["nonconverged_frac"] = float(budget_exhausted.mean())
        viol = (qs[s:] < self.lower - 1e-6) | (qs[s:] > self.upper + 1e-6)
        out["limit_violation_frac"] = float(viol.any(axis=1).mean())
        margins = np.minimum(qs[s:] - self.lower, self.upper - qs[s:])
        out["limit_margin_min"] = float(margins[:, self.arm_mask].min())

        # --- behaviour --------------------------------------------------- #
        dq = np.diff(qs[s:, self.arm_mask], axis=0)
        if len(dq) > 1:
            vel = dq / dt
            acc = np.diff(vel, axis=0) / dt
            jerk = np.diff(acc, axis=0) / dt
            step_inf = np.abs(dq).max(axis=1)
            jump_thresh = max(5.0 * float(np.median(step_inf)), JUMP_FLOOR)
            out.update(
                vel_rms=float(np.sqrt(np.mean(vel**2))),
                vel_max=float(np.abs(vel).max()),
                acc_rms=float(np.sqrt(np.mean(acc**2))),
                acc_max=float(np.abs(acc).max()),
                jerk_rms=float(np.sqrt(np.mean(jerk**2))),
                config_jumps=int((step_inf > jump_thresh).sum()),
            )
        out.update(
            manip_hands_min=float(manip[:, :2].min()),
            manip_hands_mean=float(manip[:, :2].mean()),
            manip_mid_min=float(manip[:, 2].min()),
            manip_mid_mean=float(manip[:, 2].mean()),
        )

        # --- self-collision clearance ------------------------------------ #
        if with_collision and self._clearance_batch is not None:
            d_all = np.asarray(self._clearance_batch(qs_j))[s:]  # (T', P)
            d = d_all[:, self._coll_keep]
            out["self_clearance_min_mm"] = float(d.min() * 1e3)
            out["self_collision_frac"] = float((d.min(axis=1) < 0.0).mean())
            mid_keep = self._coll_keep & self._coll_mid
            if mid_keep.any():
                d_mid = d_all[:, mid_keep]
                out["mid_clearance_min_mm"] = float(d_mid.min() * 1e3)

        return out
