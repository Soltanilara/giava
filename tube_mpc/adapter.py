"""The adapter seam: the only code you rewrite per machine.

The filter core never imports lerobot, pyroki, interbotix, or anything
robot-specific. To port to a new computer, implement the three methods of
TeleopAdapter against whatever stack that machine runs (any lerobot
version, any IK weights, velocity- or position-commanded arms) and call
run_filter_step() from your existing teleop loop.

command_mode decides what extract_command() hands your send():
    position     -> next planned joint positions  q_plan[1]
    velocity     -> next planned joint velocities v_plan[1]
    acceleration -> the MPC input u_0 itself
All three come from the same plan, so the guarantee is unchanged; pick
whichever your driver accepts. For velocity-commanded arms, recalibrate
w_box on that machine -- the servo tracking residuals will differ.
"""

from typing import Protocol

import numpy as np

from tube_mpc.config import FilterConfig
from tube_mpc.controller import TubeMPC
from tube_mpc.model import KinematicModel
from tube_mpc.reference import decayed_velocity_reference, hold_reference


class TeleopAdapter(Protocol):
    """Implement these three methods on each machine."""

    def read_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (q, qdot) measured, each shape (n,). If your driver gives
        no velocities, finite-difference q and low-pass it."""
        ...

    def read_target(self) -> np.ndarray:
        """Return the current IK target joint configuration, shape (n,).
        This is where your PyRoki/mink/whatever solve plugs in -- any
        weights, any version."""
        ...

    def send(self, q: np.ndarray, v: np.ndarray, a: np.ndarray,
             command: np.ndarray) -> None:
        """Send the tick's command. `command` is the mode-selected array
        (see extract_command); q/v/a give the full planned point in case
        your driver wants feedforward terms."""
        ...


def extract_command(out: dict, mode: str, model: KinematicModel,
                    x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull (q, v, a, command) for this tick from a TubeMPC.solve() result.

    Falls back to propagating the model one step with the (always available)
    input u when the solve used the stored-plan fallback path.
    """
    n = model.n
    u = out["u"]
    if out["x_pred"] is not None:
        q_next = out["x_pred"][1, :n]
        v_next = out["x_pred"][1, n:]
    else:
        x_next = model.step(x, u)
        q_next, v_next = x_next[:n], x_next[n:]
    command = {"position": q_next, "velocity": v_next, "acceleration": u}[mode]
    return q_next, v_next, u, command


class FilterRunner:
    """Holds the per-tick state (previous target for the extrapolator) and
    runs one filter step. Wire this into your existing loop:

        runner = FilterRunner(config, adapter)
        while teleoperating:
            runner.step()
    """

    def __init__(self, config: FilterConfig, adapter: TeleopAdapter):
        self.cfg = config
        self.adapter = adapter
        self.names, self.model, self.mpc = config.build_mpc()
        self._prev_target: np.ndarray | None = None
        self.last: dict | None = None  # last solve result, for logging

    def step(self) -> dict:
        q, v = self.adapter.read_state()
        x = np.concatenate([q, v])
        target = np.asarray(self.adapter.read_target(), dtype=float)

        if self._prev_target is None:
            ref = hold_reference(target, self.cfg.horizon)
        else:
            ref = decayed_velocity_reference(
                target, self._prev_target, self.cfg.dt, self.cfg.horizon,
                decay=self.cfg.ref_decay, v_cap=self.model.limits.v_max)
        self._prev_target = target

        out = self.mpc.solve(x, ref)
        qc, vc, ac, command = extract_command(out, self.cfg.command_mode, self.model, x)
        self.adapter.send(qc, vc, ac, command)
        self.last = out
        return out


class SimAdapter:
    """Reference adapter: simulates the plant with the model itself plus a
    random disturbance inside the calibrated box. Useful for testing a
    config on a machine before touching the real robot."""

    def __init__(self, config: FilterConfig, seed: int = 0):
        _, self.model, self.w_box = config.build()
        lim = self.model.limits
        self.x = np.concatenate([0.5 * (lim.q_min + lim.q_max),
                                 np.zeros(self.model.n)])
        self.target = self.x[: self.model.n].copy()
        self.rng = np.random.default_rng(seed)

    def read_state(self):
        n = self.model.n
        return self.x[:n].copy(), self.x[n:].copy()

    def read_target(self):
        return self.target.copy()

    def send(self, q, v, a, command):
        w = self.rng.uniform(-1, 1, size=2 * self.model.n) * self.w_box
        self.x = self.model.step(self.x, a, w)
