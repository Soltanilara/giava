"""Set-theoretic machinery: LQR ancillary gain, support functions,
mRPI approximation, and constraint-tightening margins.

Disturbance model
-----------------
The execution error per step is w in W, a symmetric box
    W = {w : |w_j| <= w_box_j},   w_box in R^{2n}, w_box > 0.
W covers servo tracking error, discretization error, and command latency
jitter -- calibrate it from logged commanded-vs-measured joint data.

Ancillary feedback
------------------
K is the discrete LQR gain for (A, B), so the error dynamics under the
tube controller are e+ = (A + B K) e + w with A_K = A + B K Schur stable.

Tightening (Chisci et al., 2001 constraint-restriction variant)
---------------------------------------------------------------
With the nominal trajectory pinned to the measured state (z_0 = x_k), the
predicted error at step i is e_i = sum_{j=0}^{i-1} A_K^j w_j, so a state
constraint row c^T x <= d must be tightened at step i by the support

    m_i(c) = sum_{j=0}^{i-1} h_W(A_K^j^T c),   h_W(a) = sum_j |a_j| w_box_j.

Input rows g^T u <= h tighten by the support of K e_i. The margins are
monotone in i and converge; the terminal set uses the limit, upper-bounded
via the Rakovic et al. (2005) (s, alpha) outer approximation:

    m_inf(c) <= (1 - alpha)^{-1} * m_s(c),  where  A_K^s W subseteq alpha W.
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """Discrete LQR. Returns (K, P) with u = K x and A + B K Schur stable."""
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def box_support(w_box: np.ndarray, a: np.ndarray) -> float:
    """Support function of the box {|w_j| <= w_box_j} in direction a."""
    return float(np.abs(a) @ w_box)


def mrpi_alpha_s(A_K: np.ndarray, w_box: np.ndarray, alpha_target: float = 0.05,
                 s_max: int = 500) -> tuple[int, float]:
    """Find (s, alpha) with A_K^s W subseteq alpha W and alpha <= alpha_target.

    For the symmetric box W this reduces to a support-function check along
    the coordinate directions. Raises if A_K is not contractive enough to
    reach alpha_target within s_max powers (i.e. K is a bad gain).
    """
    w = np.maximum(w_box, 1e-12)  # keep W full-dimensional
    Ak_s = np.eye(A_K.shape[0])
    for s in range(1, s_max + 1):
        Ak_s = Ak_s @ A_K
        # alpha(s) = max_j h_W((A_K^s)^T e_j) / w_j
        alpha = float(np.max((np.abs(Ak_s) @ w) / w))
        if alpha <= alpha_target:
            return s, alpha
    raise RuntimeError(
        f"could not satisfy A_K^s W subseteq {alpha_target}*W within s_max={s_max}; "
        "check that the LQR gain stabilizes the model")


@dataclass
class TighteningMargins:
    """Per-step constraint margins for an N-step horizon.

    state[i] : R^{2n}, margin on |x_i| rows (i = 0..N); state[0] = 0.
    input[i] : R^{n},  margin on |u_i| rows (i = 0..N-1); input[0] = 0.
    state_inf, input_inf : outer bounds on the limits as i -> infinity
        (used for the terminal set).
    """

    state: np.ndarray
    input: np.ndarray
    state_inf: np.ndarray
    input_inf: np.ndarray


def chisci_margins(A_K: np.ndarray, K: np.ndarray, w_box: np.ndarray, N: int,
                   alpha_target: float = 0.05) -> TighteningMargins:
    """Growing constraint restrictions along the horizon (exact, no polytopes).

    Row directions are the coordinate axes (all our constraints are boxes),
    so margins are computed for every state/input component at once:
        state margin_i = sum_{j<i} |A_K^j|^T-row support of W
        input margin_i = sum_{j<i} support of W under (K A_K^j)^T rows.
    """
    nx = A_K.shape[0]
    nu = K.shape[0]
    s, alpha = mrpi_alpha_s(A_K, w_box, alpha_target)

    horizon = max(N, s)
    state_m = np.zeros((horizon + 1, nx))
    input_m = np.zeros((horizon + 1, nu))
    Ak_j = np.eye(nx)
    acc_state = np.zeros(nx)
    acc_input = np.zeros(nu)
    for i in range(1, horizon + 1):
        # h_W((A_K^{i-1})^T e_r) for every state row r, vectorized:
        acc_state = acc_state + np.abs(Ak_j) @ w_box
        acc_input = acc_input + np.abs(K @ Ak_j) @ w_box
        state_m[i] = acc_state
        input_m[i] = acc_input
        Ak_j = A_K @ Ak_j

    scale = 1.0 / (1.0 - alpha)
    state_inf = scale * state_m[s]
    input_inf = scale * input_m[s]
    # Margins are monotone and bounded by the mRPI support; cap for safety.
    state_m = np.minimum(state_m[: N + 1], state_inf)
    input_m = np.minimum(input_m[:N], input_inf) if N > 0 else input_m[:0]
    return TighteningMargins(state=state_m, input=input_m,
                             state_inf=state_inf, input_inf=input_inf)
