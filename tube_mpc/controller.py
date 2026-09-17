"""The tube-MPC reference filter: a receding-horizon QP over the exact
LTI kinematic model with Chisci-style tightened constraints and a
safe-stop terminal set.

Every control period:
    u_k = TubeMPC.solve(x_k, q_ref_traj, u_human)["u"]

Guarantee (see MATH.md, Prop. 1): if the QP is feasible at k, it stays
feasible at k+1 for every disturbance realization in W, and the closed
loop never violates the joint position/velocity/acceleration limits.
The reference can do anything -- it only enters the cost.
"""

import numpy as np
import osqp
import scipy.sparse as sp

from tube_mpc.model import KinematicModel
from tube_mpc.sets import chisci_margins, dlqr


class TubeMPC:
    def __init__(
        self,
        model: KinematicModel,
        w_box: np.ndarray,
        horizon: int = 25,
        q_weight: float = 50.0,
        v_weight: float = 0.1,
        u_weight: float = 0.01,
        u_human_weight: float = 0.0,
        lqr_q: float = 10.0,
        lqr_r: float = 1.0,
        alpha_target: float = 0.05,
        q_backoff: float = 2e-3,
        v_backoff: float = 5e-3,
    ):
        """q_backoff / v_backoff (rad, rad/s) are small extra restrictions on
        the position/velocity bounds that budget for QP solver tolerance --
        the tube margins only cover the physical disturbance W, not the fact
        that OSQP returns an approximately-feasible plan."""
        self.model = model
        self.N = int(horizon)
        n, nx = model.n, 2 * model.n
        self.n, self.nx = n, nx
        lim = model.limits

        w_box = np.asarray(w_box, dtype=float)
        if w_box.shape != (nx,):
            raise ValueError(f"w_box must have shape ({nx},): |w_q| bounds then |w_v| bounds")

        # Ancillary LQR gain and the tightening margins it induces.
        K, _ = dlqr(model.A, model.B, lqr_q * np.eye(nx), lqr_r * np.eye(n))
        self.K = K
        A_K = model.A + model.B @ K
        self.margins = chisci_margins(A_K, K, w_box, self.N, alpha_target)

        # The tightened sets must stay nonempty, otherwise W is too large
        # for these limits (or the LQR gain contracts too slowly).
        span = lim.q_max - lim.q_min
        mq_inf = self.margins.state_inf[:n]
        mv_inf = self.margins.state_inf[n:]
        mu_inf = self.margins.input_inf
        if np.any(2 * mq_inf >= span) or np.any(mv_inf >= lim.v_max) or np.any(mu_inf >= lim.a_max):
            raise ValueError(
                "tightened constraint set is empty: disturbance box w_box is too large "
                "for these limits. Shrink w_box (calibrate from data) or retune lqr_q/lqr_r.")

        self._backoff = np.concatenate(
            [np.full(n, q_backoff), np.full(n, v_backoff)])
        self._build_qp(q_weight, v_weight, u_weight, u_human_weight)
        # Fallback plan: the tail of the last successful solve. Applying its
        # next input is exactly the shifted candidate from the recursive-
        # feasibility argument, so a late/failed solve degrades gracefully.
        self._plan_u: np.ndarray | None = None
        self._plan_x: np.ndarray | None = None
        self._plan_idx = 0

    # ------------------------------------------------------------------ QP
    def _build_qp(self, q_w: float, v_w: float, u_w: float, uh_w: float):
        n, nx, N = self.n, self.nx, self.N
        m = self.model
        lim = m.limits
        n_var = nx * (N + 1) + n * N
        ix = lambda i: i * nx              # offset of x_i
        iu = lambda i: nx * (N + 1) + i * n  # offset of u_i

        # ---- cost: 0.5 y'Py + q'y  (q updated each tick from the reference)
        P = np.zeros(n_var)
        for i in range(1, N + 1):
            P[ix(i): ix(i) + n] = 2 * q_w
            P[ix(i) + n: ix(i) + nx] = 2 * v_w
        for i in range(N):
            P[iu(i): iu(i) + n] = 2 * u_w
        P[iu(0): iu(0) + n] += 2 * uh_w
        self._uh_w = uh_w
        self._q_w = q_w
        self.P = sp.diags(P, format="csc")
        self.q_lin = np.zeros(n_var)

        # ---- constraints, all as l <= A y <= u
        rows: list[sp.spmatrix] = []
        l: list[np.ndarray] = []
        u: list[np.ndarray] = []

        def add(block: sp.spmatrix, lo: np.ndarray, hi: np.ndarray) -> None:
            rows.append(block)
            l.append(lo)
            u.append(hi)

        # (1) initial condition x_0 = x_meas (bounds updated every tick)
        E0 = sp.lil_matrix((nx, n_var))
        E0[:, :nx] = sp.eye(nx)
        add(E0, np.zeros(nx), np.zeros(nx))
        self._n_init = nx

        # (2) dynamics x_{i+1} = A x_i + B u_i
        for i in range(N):
            D = sp.lil_matrix((nx, n_var))
            D[:, ix(i + 1): ix(i + 1) + nx] = sp.eye(nx)
            D[:, ix(i): ix(i) + nx] = -m.A
            D[:, iu(i): iu(i) + n] = -m.B
            add(D, np.zeros(nx), np.zeros(nx))

        # (3) tightened state bounds, i = 1..N (i = 0 is pinned to x_meas)
        for i in range(1, N + 1):
            S = sp.lil_matrix((nx, n_var))
            S[:, ix(i): ix(i) + nx] = sp.eye(nx)
            if i < N:
                mq = self.margins.state[i][:n] + self._backoff[:n]
                mv = self.margins.state[i][n:] + self._backoff[n:]
                lo = np.concatenate([lim.q_min + mq, -(lim.v_max - mv)])
                hi = np.concatenate([lim.q_max - mq, lim.v_max - mv])
            else:
                # terminal safe-stop set: zero velocity, mRPI-tightened box
                mq = self.margins.state_inf[:n] + self._backoff[:n]
                lo = np.concatenate([lim.q_min + mq, np.zeros(n)])
                hi = np.concatenate([lim.q_max - mq, np.zeros(n)])
            add(S, lo, hi)

        # (4) tightened input bounds, i = 0..N-1
        for i in range(N):
            U = sp.lil_matrix((n, n_var))
            U[:, iu(i): iu(i) + n] = sp.eye(n)
            mu = self.margins.input[i] if i < len(self.margins.input) else self.margins.input_inf
            add(U, -(lim.a_max - mu), lim.a_max - mu)

        self.A_con = sp.vstack([r.tocsc() for r in rows], format="csc")
        self.l = np.concatenate(l)
        self.u = np.concatenate(u)

        self.solver = osqp.OSQP()
        self.solver.setup(P=self.P, q=self.q_lin, A=self.A_con, l=self.l, u=self.u,
                          verbose=False, eps_abs=1e-4, eps_rel=1e-4, max_iter=50000,
                          polishing=True, time_limit=0.8 * self.model.dt)

    # ---------------------------------------------------------------- solve
    def solve(self, x: np.ndarray, q_ref: np.ndarray,
              u_human: np.ndarray | None = None) -> dict:
        """One receding-horizon step.

        x       : measured state (q, qdot), shape (2n,)
        q_ref   : reference joint positions -- shape (n,) for hold, or
                  (N+1, n) for a predicted trajectory (row 0 unused).
        u_human : optional raw human-implied acceleration for the
                  minimal-intervention term (needs u_human_weight > 0).
        """
        n, nx, N = self.n, self.nx, self.N
        q_ref = np.asarray(q_ref, dtype=float)
        if q_ref.ndim == 1:
            q_ref = np.tile(q_ref, (N + 1, 1))

        q_lin = self.q_lin
        q_lin[:] = 0.0
        for i in range(1, N + 1):
            q_lin[i * nx: i * nx + n] = -2 * self._q_w * q_ref[i]
        if u_human is not None and self._uh_w > 0:
            off = nx * (N + 1)
            q_lin[off: off + n] += -2 * self._uh_w * u_human

        self.l[: self._n_init] = x
        self.u[: self._n_init] = x
        self.solver.update(q=q_lin, l=self.l, u=self.u)
        res = self.solver.solve()

        status = res.info.status
        # NOT startswith("solved"): OSQP also returns "solved inaccurate"
        # when it stops on the time limit or a loosened tolerance, and that
        # plan can be materially infeasible (primal residual ~1e-2 measured,
        # i.e. ~10x q_backoff). The recursive-feasibility argument only
        # licenses applying a FEASIBLE plan, so anything else takes the
        # shifted-plan fallback below -- which is exactly what the proof
        # says to do, and costs ~5% of ticks on an adversarial reference.
        ok = (status == "solved")
        lim = self.model.limits
        u0_cap = lim.a_max - (
            self.margins.input[0] if len(self.margins.input) else 0.0)

        if ok:
            y = res.x
            x_pred = y[: nx * (N + 1)].reshape(N + 1, nx)
            u_pred = y[nx * (N + 1):].reshape(N, n)
            # Project the applied input onto its (hard, known) box so solver
            # tolerance can never leak past the actuation limit.
            u0 = np.clip(u_pred[0], -u0_cap, u0_cap)
            self._plan_u = u_pred.copy()
            self._plan_x = x_pred.copy()
            self._plan_idx = 1
            fallback = False
        else:
            # Late or failed solve: run the tube controller along the stored
            # plan -- nominal input plus ancillary feedback K(x - x_plan),
            # which is exactly how the tube absorbs disturbances between
            # successful replans. If no plan is left, brake to zero velocity.
            x_pred = u_pred = None
            fallback = True
            if self._plan_u is not None and self._plan_idx < len(self._plan_u):
                i = self._plan_idx
                u0 = self._plan_u[i] + self.K @ (x - self._plan_x[i])
                u0 = np.clip(u0, -lim.a_max, lim.a_max)
                self._plan_idx += 1
            else:
                v = x[self.n:]
                u0 = np.clip(-v / self.model.dt, -lim.a_max, lim.a_max)

        return {
            "ok": ok,
            "fallback": fallback,
            "status": status,
            "u": u0,
            "x_pred": x_pred,
            "u_pred": u_pred,
            "solve_time": res.info.run_time,
            "objective": res.info.obj_val,
        }
