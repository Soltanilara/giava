"""Phase 1 — the clean reference IK solver (three-arm GIAVA).

FROZEN once signed off: ablation variants (Phase 4+) are built in a separate
module by *adding* one residual at a time to this exact problem; this file is
never edited to accommodate them.

The optimization problem
------------------------
Decision variable:
    q ∈ R^23  — all actuated joints (2×6 arm + 2×2 finger + 7 camera-arm).

Objective (nonlinear least squares):

    min_q  Σ_{a ∈ {left,right,middle}} || W_a · log( T_a(q)^-1 · T_a* ) ||²

    subject to  lower_j ≤ q_j ≤ upper_j   for every joint j,

where T_a(q) is the forward-kinematics pose of arm a's target link,
T_a* the commanded pose, log(·) the SE(3) logarithm (a 6-vector twist
[v; ω]), and W_a = diag(pos_weight_a · I₃, ori_weight_a · I₃).

Everything below is PyRoki / jaxls **native** functionality; the only code in
this file is problem assembly and NumPy<->JAX plumbing.  There is no smoothing
term, no joint centering, no collision term, no manipulability term, no
velocity clamp, no target or output filtering, and no re-seeding logic.

Residuals (2 cost terms):
1. `pk.costs.pose_cost_analytic_jac` ×3 (one per arm) — the 6-D SE(3) log
   residual above, with an analytically derived Jacobian (no autodiff).
2. `pk.costs.limit_constraint` — joint limits, enforced by jaxls's augmented
   Lagrangian machinery (residual = limit violation, zero inside the range).

Solver: Levenberg-Marquardt (jaxls), dense Cholesky linear solves,
lambda_initial=1.0 — identical to PyRoki's canonical basic-IK example.

Initialization: warm start from the previous commanded configuration
(`q_init`), which is the standard tracking-controller setting.  Note this is
the one unavoidable form of temporal coupling in the baseline: LM is a local
method, so the warm start selects the solution basin.  It is initialization,
not a residual — documented, and identical for every variant in the study.

See BASELINE.md for the full numerical specification (weights, tolerances,
termination criteria, dtypes).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk

NUM_TARGETS = 3

# PyRoki's canonical basic-IK weights (examples/pyroki_snippets/_solve_ik.py),
# applied uniformly to all three arms.  Meaning: with pos_weight=50 [1/m] and
# ori_weight=10 [1/rad], a 0.2 m position error and a 1 rad orientation error
# contribute equally to the cost; 5 mm ≙ 1.43°.
DEFAULT_POS_WEIGHT = 50.0
DEFAULT_ORI_WEIGHT = 10.0


@dataclass
class SolveResult:
    q: np.ndarray  # (23,) solution configuration
    iterations: int  # LM iterations of the (final) inner solve
    # Which jaxls termination criterion fired: [cost_tol, gradient_tol, param_tol].
    # All-False with iterations == max_iterations means the budget was exhausted.
    termination: np.ndarray  # (3,) bool
    solve_ms: float  # wall clock incl. device sync, excl. JIT compile


class BaselineIK:
    """Minimal three-arm tracking IK.  One compiled LM problem, fixed structure.

    Weights are runtime arguments (changing them never recompiles); the cost
    *structure* is fixed at construction.
    """

    def __init__(
        self,
        robot: pk.Robot,
        target_link_names: Tuple[str, str, str],
        max_iterations: int = 100,
    ) -> None:
        self.robot = robot
        self.num_joints = robot.joints.num_actuated_joints
        self.target_indices = np.asarray(
            [robot.links.names.index(n) for n in target_link_names], dtype=np.int32
        )
        target_idx_jax = jnp.asarray(self.target_indices)
        self.max_iterations = max_iterations

        @jdc.jit
        def _solve(
            q_init: jax.Array,  # (23,) warm start
            target_positions: jax.Array,  # (3, 3)
            target_wxyzs: jax.Array,  # (3, 4), unit quaternions
            pos_weights: jax.Array,  # (3,)
            ori_weights: jax.Array,  # (3,)
        ):
            joint_var = robot.joint_var_cls(0)
            costs = []
            for i in range(NUM_TARGETS):  # fixed length; unrolled at trace time
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

            sol, summary = (
                jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make([joint_var.with_value(q_init)]),
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

    # ------------------------------------------------------------------ #
    def solve(
        self,
        q_init: np.ndarray,
        target_positions: np.ndarray,  # (3, 3) [left, right, middle]
        target_wxyzs: np.ndarray,  # (3, 4)
        pos_weights: Optional[np.ndarray] = None,
        ori_weights: Optional[np.ndarray] = None,
    ) -> SolveResult:
        """One IK solve, warm-started from `q_init`."""
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

        t0 = time.perf_counter()
        q, iters, term = self._solve_jax(
            q_init=jnp.asarray(np.asarray(q_init, dtype=np.float32)),
            target_positions=jnp.asarray(pos),
            target_wxyzs=jnp.asarray(wxyz),
            pos_weights=jnp.asarray(pw),
            ori_weights=jnp.asarray(ow),
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
        """Trigger JIT compilation (targets = FK of q0) so timings exclude it."""
        fk = self.robot.forward_kinematics(np.asarray(q0, dtype=np.float32))
        pos = np.stack(
            [
                np.asarray(jaxlie.SE3(fk[i]).translation())
                for i in self.target_indices
            ]
        )
        wxyz = np.stack(
            [
                np.asarray(jaxlie.SE3(fk[i]).rotation().wxyz)
                for i in self.target_indices
            ]
        )
        self.solve(q0, pos, wxyz)
