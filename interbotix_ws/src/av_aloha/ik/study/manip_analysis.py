"""Phase 6 — anatomy of the manipulability cost, and cheaper formulations.

Answers the specific questions from the study brief with measurements:

1/2. What metric, maximized or minimized?  (from source, verified numerically)
     pyroki `manipulability_residual` = weight / (Yoshikawa_translational + 1e-6),
     Yoshikawa_t = √det(J·Jᵀ) with J = ∂(EE position)/∂q — *translational only*,
     entering LM as a least-squares residual → minimizing it maximizes
     manipulability.  The 6-D (full twist) Yoshikawa is NOT used.
3.   What Jacobian?  `jax.jacfwd` of the translation of *every* link's FK,
     indexed to one link afterwards: forward-mode over all 23 inputs, full
     model, per arm, per evaluation.
4.   How is the residual's own gradient obtained?  Automatic differentiation:
     the cost provides no analytic Jacobian (unlike pose_cost_analytic_jac),
     so jaxls differentiates through the jacfwd-FK — i.e. second-order
     derivatives of the whole kinematic chain.
5.   Complexity / where the time goes: micro-timings below (value eval vs
     residual-Jacobian eval vs whole solve), plus the Phase 4/5 facts:
     iterations *decrease* with the term (4.4 vs 7.4) while solve time rises
     10–30×, and the cost persists at weight 0 → per-iteration derivative
     work, not optimizer behaviour.
6-9. Does weight change the configuration / help?  Measured in Phase 5
     (results/sweeps/manipulability.csv): below w=0.1 nothing changes but
     bias; above, tracking collapses while the Yoshikawa floor moves
     0.009→0.04.  Never fixes the trap; centering 0.5 dominates.

Cheaper formulations tested here (custom code, clearly non-pyroki):
  arm6   — same 1/w residual, but jacfwd restricted to the 6 joints of the
           arm that actually move the link (input dim 23→6).
  hinge  — w·relu(m0 − manip)/m0 (active only below m0): near-zero gradient
           work *for the optimizer* in healthy regions and bounded residual,
           fixing the 1/(w+ε) blow-up conditioning.
  arm6_hinge — both.

Each: micro-timing (residual value + LM-relevant Jacobian) and a mini-rollout
(near_singular, self_fold, trans_lissajous) to check the behavioural effect
is preserved w.r.t. the reference formulation at the same weight.

Run AFTER other heavy jobs finish (timings are contention-sensitive):
    JAX_PLATFORMS=cpu python manip_analysis.py
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import pyroki as pk

import metrics as M
import robot_model as rm
import trajectories as T
from baseline import BaselineIK, NUM_TARGETS
from variants import VariantIK

EPS = 1e-6


def timeit(fn, *args, n=200, warmup=5):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n * 1e3  # ms


# --------------------------------------------------------------------------- #
# custom residual formulations (Phase 6 candidates — custom code)
# --------------------------------------------------------------------------- #
def make_manip_fns(robot, link_index: int, arm_joint_idx: np.ndarray):
    """Return dict of jitted manipulability-value functions by formulation."""
    li = int(link_index)
    aidx = jnp.asarray(arm_joint_idx)

    def yoshikawa_full(q):
        # pyroki's computation: jacfwd of ALL links' translation, then index.
        J = jax.jacfwd(
            lambda qq: jaxlie.SE3(robot.forward_kinematics(qq)).translation()
        )(q)[li]
        return jnp.sqrt(jnp.maximum(0.0, jnp.linalg.det(J @ J.T)))

    def yoshikawa_arm6(q):
        # differentiate only w.r.t. this arm's 6 joints (input dim 23 -> 6)
        def pos_of_arm(qa):
            qq = q.at[aidx].set(qa)
            return jaxlie.SE3(robot.forward_kinematics(qq)).translation()[li]

        J = jax.jacfwd(pos_of_arm)(q[aidx])  # (3, 6)
        return jnp.sqrt(jnp.maximum(0.0, jnp.linalg.det(J @ J.T)))

    return {"full": yoshikawa_full, "arm6": yoshikawa_arm6}


def residual_inv(m, w):  # pyroki form
    return w / (m + EPS)


def residual_hinge(m, w, m0=0.02):
    return w * jnp.maximum(0.0, m0 - m) / m0


def make_custom_cost(manip_fn, form: str):
    """Cost factory with the manipulability function captured by closure
    (functions cannot be pytree arguments)."""

    @jaxls.Cost.create_factory
    def _cost(vals, joint_var, weight):
        m = manip_fn(vals[joint_var])
        if form == "inv":
            return residual_inv(m, weight).reshape(1)
        return residual_hinge(m, weight).reshape(1)

    return _cost


class CustomManipIK(BaselineIK):
    """Baseline + custom manipulability formulation on both hand arms."""

    def __init__(self, robot, target_link_names, jac: str, form: str,
                 weight: float, max_iterations: int = 100):
        # Build baseline costs by reusing BaselineIK's structure via fresh jit.
        import jax_dataclasses as jdc

        self.robot = robot
        self.max_iterations = max_iterations
        self.target_indices = np.asarray(
            [robot.links.names.index(n) for n in target_link_names],
            dtype=np.int32,
        )
        target_idx_jax = jnp.asarray(self.target_indices)
        arm_idx = rm.arm_joint_indices(robot)
        cost_factories = [
            make_custom_cost(
                make_manip_fns(robot, self.target_indices[0], arm_idx["left"])[jac],
                form,
            ),
            make_custom_cost(
                make_manip_fns(robot, self.target_indices[1], arm_idx["right"])[jac],
                form,
            ),
        ]
        w = float(weight)

        @jdc.jit
        def _solve(q_init, target_positions, target_wxyzs, pos_weights, ori_weights):
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
            for factory in cost_factories:
                costs.append(factory(joint_var=joint_var, weight=w))
            sol, summary = (
                jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make(
                        [joint_var.with_value(q_init)]
                    ),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
                    termination=jaxls.TerminationConfig(
                        max_iterations=max_iterations
                    ),
                    return_summary=True,
                )
            )
            return sol[joint_var], summary.iterations, summary.termination_criteria

        self._solve_jax = _solve


# --------------------------------------------------------------------------- #
def main() -> None:
    robot, urdf = rm.load(with_urdf=True)
    q0 = rm.home_config(robot)
    home_pos, home_wxyz = rm.home_poses(robot)
    arm_idx = rm.arm_joint_indices(robot)
    li = int(rm.target_link_indices(robot)[0])
    fns = make_manip_fns(robot, li, arm_idx["left"])
    q0j = jnp.asarray(q0)

    print("== 1/2. sanity: metric value & direction ==")
    m_home = float(jax.jit(fns["full"])(q0j))
    print(f"Yoshikawa_t(home, left) = {m_home:.4f}; residual 1/(m+1e-6) = "
          f"{1/(m_home+1e-6):.2f} → minimizing residual maximizes m ✓")
    print(f"arm6 formulation identical value: {float(jax.jit(fns['arm6'])(q0j)):.4f}")

    print("\n== 5. micro-timings (ms/call, jitted, n=200) ==")
    fk = jax.jit(lambda q: robot.forward_kinematics(q))
    print(f"FK (all links)                      : {timeit(fk, q0j):8.3f}")
    for name, fn in fns.items():
        v = jax.jit(fn)
        g = jax.jit(jax.grad(lambda q: residual_inv(fn(q), 0.02) ** 2))
        print(f"manip value  [{name:5s}]              : {timeit(v, q0j):8.3f}")
        print(f"d/dq of squared residual [{name:5s}]   : {timeit(g, q0j):8.3f}")

    print("\n== solve-level cost of formulations (mini-rollout) ==")
    suite = {t.name: t for t in T.build_suite(home_pos, home_wxyz)}
    mini = [suite[n] for n in ("near_singular", "self_fold", "trans_lissajous")]
    evaluator = M.Evaluator(robot, urdf)

    def rollout(ik, traj):
        q = q0.copy()
        qs, tms, its, terms = [], [], [], []
        for i in range(len(traj)):
            r = ik.solve(q, traj.positions[i], traj.wxyzs[i])
            q = r.q
            qs.append(q); tms.append(r.solve_ms)
            its.append(r.iterations); terms.append(r.termination)
        return np.stack(qs), np.asarray(tms), np.asarray(its), np.stack(terms)

    variants = {
        "pyroki_inv(w=.02)": VariantIK(robot, rm.TARGET_LINKS, "manipulability"),
        "full_inv(w=.02)": CustomManipIK(robot, rm.TARGET_LINKS, "full", "inv", 0.02),
        "arm6_inv(w=.02)": CustomManipIK(robot, rm.TARGET_LINKS, "arm6", "inv", 0.02),
        "arm6_hinge(w=1)": CustomManipIK(robot, rm.TARGET_LINKS, "arm6", "hinge", 1.0),
    }
    print(f"{'variant':22s} {'traj':16s} {'pos p95':>8s} {'manip min':>9s} "
          f"{'solve ms':>8s} {'max ms':>8s}")
    for vname, ik in variants.items():
        ik.warmup(q0)
        for traj in mini:
            qs, tms, its, terms = rollout(ik, traj)
            r = evaluator.evaluate(traj, qs, tms, its, terms, 100,
                                   with_collision=False)
            print(f"{vname:22s} {traj.name:16s} "
                  f"{r.get('pos_mm_p95', float('nan')):8.2f} "
                  f"{r.get('manip_hands_min', float('nan')):9.4f} "
                  f"{r['solve_ms_mean']:8.2f} {r['solve_ms_max']:8.2f}")


if __name__ == "__main__":
    main()
