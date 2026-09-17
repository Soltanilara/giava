"""Machine-specific configuration for the tube-MPC filter.

Everything that differs between computers/robot setups lives in one YAML
file; the code is identical everywhere. See config.example.yaml for a
commented template, and calibrate_w.py for filling in w_box from logs.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import yaml

from tube_mpc.model import JointLimits, KinematicModel
from tube_mpc.urdf_limits import parse_urdf_limits

COMMAND_MODES = ("position", "velocity", "acceleration")

# Where a *relative* urdf_path is resolved from. Anchoring on the package
# rather than the cwd is what lets `python -m tube_mpc.check` and pytest run
# from any directory; an ABSOLUTE urdf_path (what a machine config should
# carry) skips this entirely.
_PKG_PARENT = Path(__file__).resolve().parent.parent


def resolve_urdf(path: str | Path) -> str:
    """Absolute path to the URDF, searched cwd-first then package-relative.

    Raises FileNotFoundError naming both places it looked -- a wrong URDF is
    the single most common porting mistake, and a bare ENOENT on a relative
    name tells you nothing about which directory was assumed.
    """
    p = Path(path)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"urdf_path does not exist: {p}")
        return str(p)
    tried = [Path.cwd() / p, _PKG_PARENT / p]
    for cand in tried:
        if cand.exists():
            return str(cand.resolve())
    raise FileNotFoundError(
        f"urdf_path {path!r} not found. Looked in: "
        + ", ".join(str(t) for t in tried)
        + ". Set an absolute urdf_path in your machine config.")


@dataclass
class FilterConfig:
    # --- plant / timing (machine-specific: MUST be set per machine) ------
    urdf_path: str = "giava.urdf"
    joint_prefix: str = "right_"
    dt: float = 0.02                 # control period of YOUR teleop loop (s)
    a_max: float = 12.0              # rad/s^2; identify from logs/servo spec
    command_mode: str = "position"   # what the adapter sends: position |
    #                                  velocity | acceleration

    # --- disturbance box (machine-specific: calibrate, don't guess) ------
    # per-joint bounds; scalars are broadcast to all joints
    w_q: float | list = 2e-3         # rad, one-step position residual bound
    w_v: float | list = 1e-2         # rad/s, one-step velocity residual bound

    # --- MPC tuning (portable defaults; retune only if behavior asks) ----
    horizon: int = 25
    q_weight: float = 50.0
    v_weight: float = 0.1
    u_weight: float = 0.01
    u_human_weight: float = 0.0
    lqr_q: float = 10.0
    lqr_r: float = 1.0
    alpha_target: float = 0.05
    q_backoff: float = 2e-3
    v_backoff: float = 5e-3

    # --- reference extrapolation ----------------------------------------
    ref_decay: float = 0.85
    # Cap on the extrapolated reference velocity (rad/s).  None = v_max, which
    # is the servo's speed, not the operator's: a one-tick jump in the target
    # then extrapolates at full v_max for the whole decay window and the
    # filter chases an overshoot it has to swing back from.  ~1 rad/s is a
    # hand moving a joint quickly.
    ref_v_cap: float | None = None
    # A per-tick target change above this (rad) is a discontinuity, not a
    # velocity: hold the reference for that tick instead of extrapolating.
    ref_jump_rad: float = 0.1

    def __post_init__(self) -> None:
        if self.command_mode not in COMMAND_MODES:
            raise ValueError(f"command_mode must be one of {COMMAND_MODES}")

    # ------------------------------------------------------------- helpers
    @classmethod
    def load(cls, path: str | Path) -> "FilterConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    def build(self):
        """Instantiate (joint_names, KinematicModel, w_box) from this config."""
        names, limits = parse_urdf_limits(
            resolve_urdf(self.urdf_path), self.joint_prefix, self.a_max)
        n = limits.n
        wq = np.broadcast_to(np.asarray(self.w_q, dtype=float), (n,))
        wv = np.broadcast_to(np.asarray(self.w_v, dtype=float), (n,))
        w_box = np.concatenate([wq, wv])
        return names, KinematicModel(limits, self.dt), w_box

    def build_mpc(self):
        """Instantiate everything: (joint_names, model, TubeMPC)."""
        from tube_mpc.controller import TubeMPC

        names, model, w_box = self.build()
        mpc = TubeMPC(
            model, w_box, horizon=self.horizon,
            q_weight=self.q_weight, v_weight=self.v_weight,
            u_weight=self.u_weight, u_human_weight=self.u_human_weight,
            lqr_q=self.lqr_q, lqr_r=self.lqr_r,
            alpha_target=self.alpha_target,
            q_backoff=self.q_backoff, v_backoff=self.v_backoff)
        return names, model, mpc
