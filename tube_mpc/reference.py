"""Reference extrapolators: turn the latest IK targets into an N-step
predicted reference for the MPC cost.

The reference only enters the cost, so these cannot break any guarantee;
a better predictor just makes the filter more transparent (it intervenes
less because it anticipates the operator). Start with zero-order hold,
graduate to decayed-velocity, and see MATH.md for the minimum-jerk upgrade.
"""

import numpy as np


def hold_reference(q_ref: np.ndarray, N: int) -> np.ndarray:
    """Zero-order hold: assume the operator's target stays where it is."""
    return np.tile(np.asarray(q_ref, dtype=float), (N + 1, 1))


def decayed_velocity_reference(q_ref: np.ndarray, q_ref_prev: np.ndarray,
                               dt: float, N: int, decay: float = 0.85,
                               v_cap: np.ndarray | None = None) -> np.ndarray:
    """Extrapolate with the finite-difference reference velocity, decayed.

    r_{k+i} = r_k + v * dt * sum_{j=1..i} decay^j, with v optionally capped.
    decay < 1 encodes that human hands decelerate (bell-shaped speed
    profiles, Flash & Hogan 1985); it also keeps a noisy difference from
    launching the predicted reference far away.
    """
    r = np.asarray(q_ref, dtype=float)
    v = (r - np.asarray(q_ref_prev, dtype=float)) / dt
    if v_cap is not None:
        v = np.clip(v, -v_cap, v_cap)
    out = np.empty((N + 1, len(r)))
    out[0] = r
    step = v * dt
    gain = 0.0
    for i in range(1, N + 1):
        gain = gain * decay + decay
        out[i] = r + step * gain
    return out
