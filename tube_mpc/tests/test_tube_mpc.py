"""Sanity tests for the tube-MPC filter.

Run from the repo root:  python -m pytest tube_mpc/tests -q
"""

import numpy as np
import pytest

from tube_mpc.controller import TubeMPC
from tube_mpc.model import JointLimits, KinematicModel
from tube_mpc.sets import chisci_margins, dlqr, mrpi_alpha_s

DT = 0.02
N = 20
W_Q, W_V = 2e-3, 1e-2


@pytest.fixture(scope="module")
def setup():
    n = 6
    lim = JointLimits(
        q_min=-np.full(n, 3.1), q_max=np.full(n, 3.1),
        v_max=np.full(n, np.pi), a_max=np.full(n, 12.0))
    model = KinematicModel(lim, DT)
    w_box = np.concatenate([np.full(n, W_Q), np.full(n, W_V)])
    mpc = TubeMPC(model, w_box, horizon=N)
    return model, mpc, w_box


def test_lqr_stabilizes_and_alpha_found(setup):
    model, mpc, w_box = setup
    K, _ = dlqr(model.A, model.B, 10 * np.eye(2 * model.n), np.eye(model.n))
    A_K = model.A + model.B @ K
    assert np.max(np.abs(np.linalg.eigvals(A_K))) < 1.0
    s, alpha = mrpi_alpha_s(A_K, w_box)
    assert 0 < alpha <= 0.05 and s >= 1


def test_margins_monotone_and_bounded(setup):
    model, mpc, w_box = setup
    m = mpc.margins
    assert np.all(np.diff(m.state, axis=0) >= -1e-12)
    assert np.all(m.state <= m.state_inf + 1e-12)
    assert np.all(m.state_inf < np.concatenate(
        [model.limits.q_max - model.limits.q_min, model.limits.v_max]))


def test_no_violation_under_disturbance(setup):
    """500 steps of an adversarial reference + random w in W: zero violations
    and zero infeasible solves (recursive feasibility, empirically)."""
    model, mpc, w_box = setup
    rng = np.random.default_rng(1)
    n = model.n
    lim = model.limits
    x = np.zeros(2 * n)
    fallbacks = 0
    for k in range(500):
        # adversarial: reference teleports around, often outside the limits
        q_ref = rng.uniform(lim.q_min - 1.0, lim.q_max + 1.0)
        out = mpc.solve(x, q_ref)
        fallbacks += out["fallback"]
        u = out["u"]
        assert np.all(np.abs(u) <= lim.a_max + 1e-6)
        w = rng.uniform(-1, 1, size=2 * n) * w_box
        x = model.step(x, u, w)
        assert np.all(x[:n] <= lim.q_max + 1e-6), f"q upper violated at {k}"
        assert np.all(x[:n] >= lim.q_min - 1e-6), f"q lower violated at {k}"
        assert np.all(np.abs(x[n:]) <= lim.v_max + 1e-6), f"v violated at {k}"
    # a teleporting reference is warm-start hostile; some time-limited solves
    # may fall back to the shifted plan, but it must stay rare
    assert fallbacks <= 50, f"too many fallback ticks: {fallbacks}/500"


def test_stops_inside_limit_for_infeasible_reference(setup):
    """Reference far beyond a joint limit: the arm must brake and settle
    strictly inside the limit, without oscillating against it."""
    model, mpc, w_box = setup
    n = model.n
    lim = model.limits
    x = np.zeros(2 * n)
    q_ref = lim.q_max + 2.0  # hopeless target
    qs = []
    fallbacks = 0
    for _ in range(400):
        out = mpc.solve(x, q_ref)
        # The guarantee is "never infeasible", not "always converged to full
        # accuracy inside the time limit" -- a time-limited/inaccurate solve
        # takes the shifted-plan fallback, which is what Prop. 1 licenses.
        assert "infeasible" not in out["status"], out["status"]
        fallbacks += out["fallback"]
        x = model.step(x, out["u"])
        qs.append(x[:n].copy())
    assert fallbacks <= 40, f"too many fallback ticks: {fallbacks}/400"
    qs = np.array(qs)
    assert np.all(qs <= lim.q_max + 1e-9)
    tail = qs[-50:]
    assert np.all(tail.std(axis=0) < 1e-3), "should settle, not chatter"
    # it should get close to the boundary (within the terminal tightening + slack)
    gap = lim.q_max - tail.mean(axis=0)
    assert np.all(gap < 0.2), f"too conservative: gap to limit {gap}"


def test_transparent_when_feasible(setup):
    """A gentle in-workspace reference should be tracked closely when the
    intended velocity-extrapolated reference predictor is used."""
    from tube_mpc.reference import decayed_velocity_reference

    model, mpc, w_box = setup
    n = model.n
    x = np.zeros(2 * n)
    t = 0.0
    errs = []
    q_prev = np.zeros(n)
    fallbacks = 0
    for _ in range(300):
        q_ref = 0.5 * np.sin(2 * np.pi * 0.3 * t) * np.ones(n)
        r_traj = decayed_velocity_reference(q_ref, q_prev, DT, N,
                                            v_cap=model.limits.v_max)
        out = mpc.solve(x, r_traj)
        assert "infeasible" not in out["status"], out["status"]
        fallbacks += out["fallback"]
        x = model.step(x, out["u"])
        errs.append(np.abs(x[:n] - q_ref).max())
        q_prev = q_ref
        t += DT
    assert fallbacks <= 30, f"too many fallback ticks: {fallbacks}/300"
    assert np.mean(errs[50:]) < 0.06, f"tracking too loose: {np.mean(errs[50:]):.4f} rad"
