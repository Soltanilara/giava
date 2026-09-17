"""User-proposed variant: REST-POSE-CALIBRATED capsules.

Idea (user): compute each capsule pair's intersection at explicitly-allowed
poses (home, self-fold rest, parked hands) and permit that much intersection
everywhere — so the arm may fold into itself as it does at rest, without
per-pair pruning.

Formalization: per-pair allowance
    α_ij = min(0, min_{q ∈ allowed} d_ij(q))
and the collision residual uses the shifted distance d̃ = d − α:
    h_ij = −colldist_from_sdf(d_ij − α_ij, margin)
Pairs clear at every allowed pose get α = 0 (normal behaviour); pairs
penetrating at an allowed pose are allowed that depth everywhere.

Known theoretical flaw (measured here): penetration depth is a scalar with
no direction, so a pair allowed −150 mm at rest is blind to real contact up
to that depth in *any other* approach geometry.  The experiment quantifies
whether that matters in practice on this robot.

Model: corrected (tight) capsules, adjacency-ignored + functional finger
pairs ignored, NO structural pruning — calibration replaces it.
Settings: margin 20 mm, weight 100 (identical to the sphere winner).

Run (background): JAX_PLATFORMS=cpu python calibrated_capsule_eval.py
Writes results/calibrated_capsule_eval.csv
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import pyroki as pk
from pyroki.collision import colldist_from_sdf

import robot_model as rm
import trajectories as T
from baseline import DEFAULT_ORI_WEIGHT, DEFAULT_POS_WEIGHT, NUM_TARGETS, SolveResult
from collision_models import pruned_sphere_collision, tight_capsule_collision
from collision_trajectories import build_collision_subset
from variants import CollisionIK

HERE = Path(__file__).resolve().parent
MARGIN = 0.020
WEIGHT = 100.0
FUNCTIONAL = (
    ("left_left_finger_link", "left_right_finger_link"),
    ("right_left_finger_link", "right_right_finger_link"),
)


def allowed_poses(robot):
    q0 = rm.home_config(robot)
    poses = [q0]
    for fname, idx in (("self_fold.npz", 300), ("mid_trans_x.npz", 150)):
        f = HERE / "results" / "baseline" / fname
        q = np.load(f)["q"]
        poses.append(q[min(idx, len(q) - 1)].astype(np.float32))
    return poses


class CalibratedCapsuleIK(CollisionIK):
    def __init__(self, robot, links, robot_coll, alpha, max_iterations=100):
        self.robot = robot
        self.max_iterations = max_iterations
        self.target_indices = np.asarray(
            [robot.links.names.index(n) for n in links], dtype=np.int32)
        tgt = jnp.asarray(self.target_indices)
        alpha_j = jnp.asarray(alpha.astype(np.float32))

        @jaxls.Cost.create_factory
        def _calibrated_cost(vals, joint_var, margin, weight):
            d = robot_coll.compute_self_collision_distance(robot, vals[joint_var])
            return (-colldist_from_sdf(d - alpha_j, margin) * weight).flatten()

        @jdc.jit
        def _solve(q_init, target_positions, target_wxyzs,
                   pos_weights, ori_weights, margin, weight):
            joint_var = robot.joint_var_cls(0)
            costs = []
            for i in range(NUM_TARGETS):
                costs.append(pk.costs.pose_cost_analytic_jac(
                    robot, joint_var,
                    jaxlie.SE3.from_rotation_and_translation(
                        jaxlie.SO3(target_wxyzs[i]), target_positions[i]),
                    tgt[i], pos_weight=pos_weights[i],
                    ori_weight=ori_weights[i]))
            costs.append(pk.costs.limit_constraint(robot, joint_var))
            costs.append(_calibrated_cost(joint_var=joint_var, margin=margin,
                                          weight=weight))
            sol, summary = (
                jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make(
                        [joint_var.with_value(q_init)]),
                    verbose=False, linear_solver="dense_cholesky",
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
                    termination=jaxls.TerminationConfig(
                        max_iterations=max_iterations),
                    return_summary=True,
                )
            )
            return sol[joint_var], summary.iterations, summary.termination_criteria

        self._solve_jax = _solve
        self.margin = MARGIN
        self.weight = WEIGHT


def main() -> None:
    robot, urdf = rm.load(with_urdf=True)
    q0 = rm.home_config(robot)
    home_pos, home_wxyz = rm.home_poses(robot)

    rc = tight_capsule_collision(urdf, user_ignore_pairs=FUNCTIONAL)
    print(f"capsule pairs (functional-ignored only): {len(rc.active_idx_i)}")

    # per-pair allowance over the allowed poses
    ds = []
    for q in allowed_poses(robot):
        ds.append(np.asarray(
            rc.compute_self_collision_distance(robot, jnp.asarray(q))
        ).reshape(-1))
    alpha = np.minimum(0.0, np.min(np.stack(ds), axis=0))
    print(f"pairs with nonzero allowance: {(alpha < 0).sum()} "
          f"(deepest {alpha.min()*1e3:.0f} mm)")

    ik = CalibratedCapsuleIK(robot, rm.TARGET_LINKS, rc, alpha)
    t0 = time.time()
    ik.warmup(q0)
    print(f"compile {time.time()-t0:.0f} s", flush=True)

    rc_sphere = pruned_sphere_collision(urdf, HERE / "results")
    idx = rm.target_link_indices(robot)

    @jax.jit
    def sphere_clear(qb):
        return jax.vmap(
            lambda q: rc_sphere.compute_self_collision_distance(robot, q).min())(qb)

    @jax.jit
    def ee_batch(qb):
        def one(q):
            fk = robot.forward_kinematics(q)
            return jaxlie.SE3(fk[jnp.asarray(idx[:2])]).translation()
        return jax.vmap(one)(qb)

    suite = {t.name: t for t in T.build_suite(home_pos, home_wxyz)}
    trajs = build_collision_subset(home_pos, home_wxyz) + [
        suite["self_fold"], suite["arms_converge"], suite["trans_x"],
        suite["teleop_grasp"], suite["reach_limit"],
    ]

    rows = []
    for traj in trajs:
        q = q0.copy()
        qs, tms, its = [], [], []
        for i in range(len(traj)):
            r = ik.solve(q, traj.positions[i], traj.wxyzs[i])
            q = r.q
            qs.append(q); tms.append(r.solve_ms); its.append(r.iterations)
        qs = np.stack(qs); tms = np.asarray(tms)
        qsj = jnp.asarray(qs.astype(np.float32))
        clear = float(np.asarray(sphere_clear(qsj)).min())  # judged by the
        ee = np.asarray(ee_batch(qsj))                      # honest model
        perr = np.linalg.norm(ee - traj.positions[:, :2], axis=-1).max(axis=1)
        row = dict(
            trajectory=traj.name,
            pos_p95_mm=float(np.percentile(perr, 95) * 1e3),
            sphere_clear_min_mm=clear * 1e3,
            solve_ms_med=float(np.median(tms)),
            solve_ms_p95=float(np.percentile(tms, 95)),
            solve_ms_max=float(tms.max()),
            iters_mean=float(np.mean(its)),
        )
        rows.append(row)
        print(f"{traj.name:20s} pos p95 {row['pos_p95_mm']:7.1f}  "
              f"sphere-clear {row['sphere_clear_min_mm']:7.1f}  "
              f"med {row['solve_ms_med']:6.2f} ms  iters {row['iters_mean']:5.1f}",
              flush=True)

    with open(HERE / "results" / "calibrated_capsule_eval.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote results/calibrated_capsule_eval.csv")


if __name__ == "__main__":
    main()
