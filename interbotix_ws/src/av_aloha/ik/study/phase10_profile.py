"""Collision-study Phase 10 — cost anatomy.

Where does collision solve time go?  Measured pieces (jitted, CPU, n=100):

  residual eval   R(q): FK → transform 180 spheres → P pair distances →
                  margin hinge.  P ∈ {5884 pruned, 14491 unpruned} spheres,
                  421 pruned tight capsules.
  Jacobian        what jaxls computes per LM iteration.  jac_mode='auto'
                  picks jacfwd for P > 23 → 23 JVPs through R.
  aggregation     r = w·√(Σ hᵢ² + ε) — IDENTICAL least-squares cost
                  (Σ w²hᵢ² + w²ε), but scalar output → jacrev, 1 VJP.
                  The Gauss-Newton approximation changes (rank-1 vs rank-P),
                  so iterations may differ — measured at solve level.

Solve-level: CollisionIK(soft, pruned spheres) vs an aggregated-cost solver
on bimanual_handoff + arms_converge: time, clearance, tracking equivalence.

Run AFTER other heavy jobs finish (timing-sensitive):
    JAX_PLATFORMS=cpu python phase10_profile.py
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

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
from baseline import DEFAULT_ORI_WEIGHT, DEFAULT_POS_WEIGHT, NUM_TARGETS, SolveResult
from collision_models import (
    pruned_sphere_collision,
    pruned_tight_capsule_collision,
    sphere_collision,
)
from collision_trajectories import build_collision_subset
from variants import CollisionIK

HERE = Path(__file__).resolve().parent


def timeit(fn, *args, n=100, warmup=3):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n * 1e3


def main() -> None:
    robot, urdf = rm.load(with_urdf=True)
    q0 = jnp.asarray(rm.home_config(robot))
    home_pos, home_wxyz = rm.home_poses(robot)

    models = {
        "spheres-pruned (5884)": pruned_sphere_collision(urdf, HERE / "results"),
        "spheres-full (14491)": sphere_collision(
            urdf, HERE / "results" / "sphere_decomposition.json"),
        "capsules-tight (421)": pruned_tight_capsule_collision(urdf),
    }
    MARGIN = 0.025

    print("== micro-timings (ms/call, jitted, n=100) ==")
    fk = jax.jit(lambda q: robot.forward_kinematics(q))
    print(f"{'FK (all links)':34s} {timeit(fk, q0):8.3f}")
    for name, rc in models.items():
        res = jax.jit(
            lambda q, rc=rc: -colldist_from_sdf(
                rc.compute_self_collision_distance(robot, q), MARGIN
            )
        )
        jac_fwd = jax.jit(jax.jacfwd(res))
        agg = jax.jit(
            lambda q, rc=rc: jnp.sqrt(
                jnp.sum(
                    (-colldist_from_sdf(
                        rc.compute_self_collision_distance(robot, q), MARGIN
                    )) ** 2
                ) + 1e-12
            )
        )
        agg_grad = jax.jit(jax.grad(agg))
        print(f"{name:34s} residual {timeit(res, q0):8.3f}   "
              f"jacfwd {timeit(jac_fwd, q0):8.3f}   "
              f"agg {timeit(agg, q0):8.3f}   agg-grad {timeit(agg_grad, q0):8.3f}")

    # ------------------------------------------------------------------ #
    # solve-level: per-pair vs aggregated formulation
    # ------------------------------------------------------------------ #
    rc = models["spheres-pruned (5884)"]

    @jaxls.Cost.create_factory
    def _agg_collision_cost(vals, joint_var, margin, weight):
        d = rc.compute_self_collision_distance(robot, vals[joint_var])
        h = -colldist_from_sdf(d, margin)
        return (weight * jnp.sqrt(jnp.sum(h**2) + 1e-12)).reshape(1)

    class AggCollisionIK(CollisionIK):
        def __init__(self, robot_, links, robot_coll, max_iterations=100):
            self.robot = robot_
            self.max_iterations = max_iterations
            self.target_indices = np.asarray(
                [robot_.links.names.index(n) for n in links], dtype=np.int32)
            tgt = jnp.asarray(self.target_indices)

            @jdc.jit
            def _solve(q_init, target_positions, target_wxyzs,
                       pos_weights, ori_weights, margin, weight):
                joint_var = robot_.joint_var_cls(0)
                costs = []
                for i in range(NUM_TARGETS):
                    costs.append(pk.costs.pose_cost_analytic_jac(
                        robot_, joint_var,
                        jaxlie.SE3.from_rotation_and_translation(
                            jaxlie.SO3(target_wxyzs[i]), target_positions[i]),
                        tgt[i], pos_weight=pos_weights[i],
                        ori_weight=ori_weights[i]))
                costs.append(pk.costs.limit_constraint(robot_, joint_var))
                costs.append(_agg_collision_cost(
                    joint_var=joint_var, margin=margin, weight=weight))
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
            self.weight = 30.0

    q0n = rm.home_config(robot)
    trajs = build_collision_subset(home_pos, home_wxyz)
    import trajectories as T
    suite = {t.name: t for t in T.build_suite(home_pos, home_wxyz)}
    trajs.append(suite["arms_converge"])

    @jax.jit
    def clearance_batch(qb):
        return jax.vmap(
            lambda q: rc.compute_self_collision_distance(robot, q).min())(qb)

    print("\n== solve-level: per-pair vector vs aggregated scalar residual ==")
    print(f"{'solver':12s} {'traj':18s} {'med ms':>7s} {'p95 ms':>7s} "
          f"{'max ms':>7s} {'iters':>6s} {'clear min':>9s} {'pos p95':>8s}")
    for label, ik in (
        ("per-pair", CollisionIK(robot, rm.TARGET_LINKS, rc, formulation="soft")),
        ("aggregated", AggCollisionIK(robot, rm.TARGET_LINKS, rc)),
    ):
        ik.warmup(q0n)
        for traj in trajs:
            q = q0n.copy()
            qs, tms, its = [], [], []
            for i in range(len(traj)):
                r = ik.solve(q, traj.positions[i], traj.wxyzs[i],
                             margin=MARGIN, weight=30.0)
                q = r.q
                qs.append(q); tms.append(r.solve_ms); its.append(r.iterations)
            qs = np.stack(qs); tms = np.asarray(tms)
            clear = float(np.asarray(
                clearance_batch(jnp.asarray(qs.astype(np.float32)))).min())
            idx = rm.target_link_indices(robot)
            fkq = jax.vmap(lambda q_: jaxlie.SE3(
                robot.forward_kinematics(q_)[jnp.asarray(idx[:2])]).translation())
            ee = np.asarray(fkq(jnp.asarray(qs.astype(np.float32))))
            perr = np.linalg.norm(ee - traj.positions[:, :2], axis=-1).max(axis=1)
            print(f"{label:12s} {traj.name:18s} {np.median(tms):7.2f} "
                  f"{np.percentile(tms, 95):7.2f} {tms.max():7.2f} "
                  f"{np.mean(its):6.1f} {clear*1e3:9.1f} "
                  f"{np.percentile(perr, 95)*1e3:8.1f}")


if __name__ == "__main__":
    main()
