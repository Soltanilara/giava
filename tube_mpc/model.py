"""Joint-space kinematic model: an exactly-LTI double integrator.

State x = (q, qdot) in R^{2n}, input u = qddot in R^n:

    q_{k+1}    = q_k + dt * qdot_k + dt^2/2 * u_k
    qdot_{k+1} = qdot_k + dt * u_k

This is the plant the MPC plans over. The real robot tracks the commanded
joint positions through its own servo loops; everything the servos and the
discretization get wrong is lumped into an additive disturbance w bounded
by a box (see sets.py). No linearization is involved -- the model is exact.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class JointLimits:
    """Symmetric-rate joint limits. All arrays have length n."""

    q_min: np.ndarray
    q_max: np.ndarray
    v_max: np.ndarray  # |qdot| <= v_max
    a_max: np.ndarray  # |qddot| <= a_max

    def __post_init__(self) -> None:
        n = len(self.q_min)
        for name in ("q_max", "v_max", "a_max"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has wrong length (expected {n})")
        if np.any(self.q_min >= self.q_max):
            raise ValueError("q_min must be strictly below q_max")
        if np.any(self.v_max <= 0) or np.any(self.a_max <= 0):
            raise ValueError("v_max and a_max must be positive")

    @property
    def n(self) -> int:
        return len(self.q_min)


class KinematicModel:
    """LTI double integrator for n joints at sample period dt."""

    def __init__(self, limits: JointLimits, dt: float):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.limits = limits
        self.dt = float(dt)
        n = limits.n
        self.n = n
        eye = np.eye(n)
        self.A = np.block([[eye, dt * eye], [np.zeros((n, n)), eye]])
        self.B = np.vstack([0.5 * dt**2 * eye, dt * eye])

    def step(self, x: np.ndarray, u: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
        """Simulate one step, optionally with an additive disturbance w."""
        xn = self.A @ x + self.B @ u
        if w is not None:
            xn = xn + w
        return xn
