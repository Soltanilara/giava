"""Tube-MPC reference filter wired to ONE interbotix arm.

This is the adapter seam between `tube_mpc` (robot-agnostic, see
/home/devi/giava/tube_mpc/README.md) and this rig's driver.  It is shared by
`tube_mpc_bench.py` (quantitative tests on one arm) and, under
GIAVA_TUBE_MPC=1, by `data_collection.py` (the real teleop loop), so both
run the exact same filter and the numbers transfer.

What it decides that the package leaves to the adapter
------------------------------------------------------
* POSITION LIMITS = URDF limits intersected with the DRIVER's limit arrays,
  pulled in by TeleopConfig.driver_limit_margin.  It is the driver that
  refuses commands (the whole group command, on any single joint), and it
  disagrees with the URDF on some joints (right wrist_angle: URDF +2.234,
  driver rejected +2.243).  A filter that certifies against the wrong array
  produces commands the driver silently drops.

* STATE SOURCE.  The arms are POSITION-servoed: we publish a goal, the servo
  chases it with its own PID.  Two ways to close the filter's loop:

    command  (default)  x_k = the filter's OWN last planned state (a virtual
                        double integrator driven by the command stream).
                        Exactly LTI with w = 0, so the guarantee applies to
                        the COMMAND trajectory: nothing infeasible ever
                        reaches the driver.  The servo's tracking lag is
                        outside the loop -- measured separately, not fought.
    measured            x_k = (q, qdot) off the encoders.  The textbook tube,
                        but on a position servo the goal is then only ever
                        one plan step ahead of where the arm IS, so the
                        servo's P-loop sees a tiny error and crawls.
                        `lookahead` (send x_pred[lookahead] instead of
                        x_pred[1]) is the standard fix; W must then cover
                        servo lag and MUST come from calibrate_w on this rig.

  Start with `command`.  Switch to `measured` only once the bench shows the
  servo lag is small enough that calibrate_w accepts W.

* RE-ANCHOR.  The virtual state must be re-synced to the encoders whenever
  the command stream and the arm can have diverged: teleop enable, a stall
  hold, a driver-rejected command.  `reset(q_meas)` does that and drops the
  stored plan, and `step()` re-anchors itself if the command has run more
  than `reanchor_rad` from the measured position (a jammed arm with the
  command marching on is exactly the failure servo_health.StallGate exists
  for; the filter must not add to it).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from tube_mpc.config import FilterConfig, resolve_urdf
from tube_mpc.controller import TubeMPC
from tube_mpc.model import JointLimits, KinematicModel
from tube_mpc.reference import decayed_velocity_reference, hold_reference
from tube_mpc.urdf_limits import parse_urdf_limits

try:
    from .paths import TUBE_MPC_CONFIG as _TUBE_MPC_CONFIG
except ImportError:  # run as a script, not a package member
    from paths import TUBE_MPC_CONFIG as _TUBE_MPC_CONFIG

DEFAULT_CONFIG = str(_TUBE_MPC_CONFIG)
PREFIX_OF = {"left": "left_", "right": "right_", "middle": "middle_"}


@dataclass
class TubeStats:
    """Per-arm counters the teleop loop folds into robustness.jsonl."""

    ticks: int = 0
    fallbacks: int = 0
    reanchors: int = 0
    ref_jumps: int = 0        # target discontinuities the extrapolator refused to follow
    solve_ms: list = field(default_factory=list)
    # |q_cmd - q_target| per tick: how much the filter changed the operator's
    # command.  Zero = transparent; large = it was braking for a limit.
    intervention_rad: list = field(default_factory=list)


class ArmTubeFilter:
    def __init__(self, cfg: FilterConfig, names: list[str], limits: JointLimits,
                 state_source: str = "command", lookahead: int = 1,
                 reanchor_rad: float = 0.35):
        if state_source not in ("command", "measured"):
            raise ValueError("state_source must be 'command' or 'measured'")
        if not (1 <= lookahead <= cfg.horizon):
            raise ValueError(f"lookahead must be in [1, horizon={cfg.horizon}]")
        self.cfg = cfg
        self.names = names
        self.model = KinematicModel(limits, cfg.dt)
        n = self.model.n
        wq = np.broadcast_to(np.asarray(cfg.w_q, dtype=float), (n,))
        wv = np.broadcast_to(np.asarray(cfg.w_v, dtype=float), (n,))
        self.w_box = np.concatenate([wq, wv])
        self.mpc = TubeMPC(
            self.model, self.w_box, horizon=cfg.horizon,
            q_weight=cfg.q_weight, v_weight=cfg.v_weight,
            u_weight=cfg.u_weight, u_human_weight=cfg.u_human_weight,
            lqr_q=cfg.lqr_q, lqr_r=cfg.lqr_r, alpha_target=cfg.alpha_target,
            q_backoff=cfg.q_backoff, v_backoff=cfg.v_backoff)
        self.state_source = state_source
        self.lookahead = int(lookahead)
        self.reanchor_rad = float(reanchor_rad)
        self.stats = TubeStats()
        self._x: np.ndarray | None = None       # virtual state (command mode)
        self._prev_target: np.ndarray | None = None

    # ------------------------------------------------------------------
    @property
    def n(self) -> int:
        return self.model.n

    def reset(self, q_meas: np.ndarray) -> None:
        """Re-anchor on the encoders and forget the plan and the reference
        history.  Call on teleop enable and after any hold/rejection."""
        q = np.asarray(q_meas, dtype=float)
        self._x = np.concatenate([q, np.zeros(self.n)])
        self._prev_target = None
        self.mpc._plan_u = None
        self.mpc._plan_x = None
        self.mpc._plan_idx = 0

    def step(self, q_meas: np.ndarray, v_meas: np.ndarray | None,
             q_target: np.ndarray) -> tuple[np.ndarray, dict]:
        """One tick.  Returns (q_cmd to publish, info dict).

        q_meas / v_meas : encoders (v_meas may be None in command mode)
        q_target        : the IK target for this arm, BEFORE any clamp
        """
        q_meas = np.asarray(q_meas, dtype=float)
        q_target = np.asarray(q_target, dtype=float)
        n = self.n

        if self._x is None:
            self.reset(q_meas)

        if self.state_source == "measured":
            if v_meas is None:
                raise ValueError("measured state source needs v_meas")
            x = np.concatenate([q_meas, np.asarray(v_meas, dtype=float)])
        else:
            x = self._x
            # the command has run away from the arm: pull it back
            if np.max(np.abs(x[:n] - q_meas)) > self.reanchor_rad:
                self.reset(q_meas)
                x = self._x
                self.stats.reanchors += 1

        # The MPC certifies against the tightened box; a target that sits
        # outside the raw limits is fine (it only enters the cost), but the
        # extrapolator should not launch the predicted reference off a step
        # change on the first tick after a reset.
        v_cap = self.model.limits.v_max
        if getattr(self.cfg, "ref_v_cap", None):
            v_cap = np.minimum(v_cap, float(self.cfg.ref_v_cap))
        jump = getattr(self.cfg, "ref_jump_rad", 0.1)
        if (self._prev_target is None
                or np.max(np.abs(q_target - self._prev_target)) > jump):
            # First tick, or a discontinuity (teleop enable, IK branch change,
            # solver hiccup): extrapolating THROUGH it would send the arm on
            # a wide detour past the new target and back.  Hold instead; the
            # MPC still moves toward the new target at up to v_max.
            ref = hold_reference(q_target, self.cfg.horizon)
            if self._prev_target is not None:
                self.stats.ref_jumps += 1
        else:
            ref = decayed_velocity_reference(
                q_target, self._prev_target, self.cfg.dt, self.cfg.horizon,
                decay=self.cfg.ref_decay, v_cap=v_cap)
        self._prev_target = q_target

        t0 = time.perf_counter()
        out = self.mpc.solve(x, ref)
        solve_ms = (time.perf_counter() - t0) * 1e3

        if out["x_pred"] is not None:
            k = self.lookahead
            q_cmd = out["x_pred"][k, :n]
            x_next = out["x_pred"][1]
        else:
            # fallback tick: propagate the model one step with the fallback u
            x_next = self.model.step(x, out["u"])
            q_cmd = x_next[:n]

        if self.state_source == "command":
            self._x = x_next.copy()

        st = self.stats
        st.ticks += 1
        st.fallbacks += int(out["fallback"])
        st.solve_ms.append(solve_ms)
        st.intervention_rad.append(float(np.max(np.abs(q_cmd - q_target))))

        info = {
            "solve_ms": solve_ms,
            "status": out["status"],
            "fallback": out["fallback"],
            "u": out["u"],
            "v_plan": x_next[n:],
            "intervention": st.intervention_rad[-1],
        }
        return q_cmd.copy(), info


# ----------------------------------------------------------------------
def build_arm_filter(arm: str, driver_lo=None, driver_hi=None,
                     driver_margin: float = 0.03,
                     config_path: str | None = None,
                     state_source: str | None = None,
                     lookahead: int | None = None,
                     v_scale: float | None = None) -> ArmTubeFilter:
    """Construct the filter for `arm`.

    driver_lo/hi : the driver's joint limit arrays
                   (bot.arm.group_info.joint_lower_limits / _upper_limits);
                   None = URDF only (sim / offline).
    v_scale      : multiply the URDF velocity limits (0 < v_scale <= 1).  The
                   filter drives toward a far reference at v_max, so the
                   first hardware runs want 0.3-0.5 here, not 1.0.
    Env overrides, so an A/B session never edits a file:
        GIAVA_TUBE_CONFIG      path to the FilterConfig yaml
        GIAVA_TUBE_STATE       command | measured
        GIAVA_TUBE_LOOKAHEAD   plan point to send (default 3 ~ servo lag; 1 = next step)
        GIAVA_TUBE_VSCALE      velocity-limit scale (default 1.0)
    """
    config_path = config_path or os.environ.get("GIAVA_TUBE_CONFIG", DEFAULT_CONFIG)
    state_source = state_source or os.environ.get("GIAVA_TUBE_STATE", "command")
    lookahead = int(lookahead or os.environ.get("GIAVA_TUBE_LOOKAHEAD", "3"))
    v_scale = float(v_scale if v_scale is not None else os.environ.get("GIAVA_TUBE_VSCALE", "1.0"))
    if not (0.0 < v_scale <= 1.0):
        raise ValueError(f"v_scale must be in (0, 1], got {v_scale}")

    cfg = FilterConfig.load(config_path)
    cfg.joint_prefix = PREFIX_OF[arm]
    names, lim = parse_urdf_limits(resolve_urdf(cfg.urdf_path), cfg.joint_prefix, cfg.a_max)

    q_min, q_max = lim.q_min.copy(), lim.q_max.copy()
    if driver_lo is not None and driver_hi is not None:
        lo = np.asarray(driver_lo, dtype=float)[: lim.n] + driver_margin
        hi = np.asarray(driver_hi, dtype=float)[: lim.n] - driver_margin
        q_min = np.maximum(q_min, lo)
        q_max = np.minimum(q_max, hi)
    limits = JointLimits(q_min=q_min, q_max=q_max, v_max=v_scale * lim.v_max, a_max=lim.a_max)

    return ArmTubeFilter(cfg, names, limits, state_source=state_source,
                         lookahead=lookahead)


def describe(f: ArmTubeFilter) -> str:
    st = f.stats
    if not st.ticks:
        return "tube_mpc: no ticks"
    sm = np.asarray(st.solve_ms)
    iv = np.asarray(st.intervention_rad)
    return (f"tube_mpc[{f.state_source}]: {st.ticks} ticks, fallbacks "
            f"{st.fallbacks} ({100 * st.fallbacks / st.ticks:.1f}%), reanchors "
            f"{st.reanchors}, ref jumps {st.ref_jumps}, solve mean {sm.mean():.2f} ms p99 "
            f"{np.quantile(sm, .99):.2f} ms max {sm.max():.2f} ms, intervention "
            f"mean {iv.mean():.4f} rad p99 {np.quantile(iv, .99):.4f} rad")


def stats_summary(f: ArmTubeFilter) -> dict:
    """JSON-ready block for robustness.jsonl / teleop_config.json."""
    st = f.stats

    def dist(v):
        if not len(v):
            return None
        a = np.asarray(v, dtype=float)
        return {"mean": float(a.mean()), "p50": float(np.median(a)),
                "p95": float(np.quantile(a, .95)), "p99": float(np.quantile(a, .99)),
                "max": float(a.max())}

    return {
        "state_source": f.state_source,
        "lookahead": f.lookahead,
        "config": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in vars(f.cfg).items()},
        "q_min": f.model.limits.q_min.tolist(),
        "q_max": f.model.limits.q_max.tolist(),
        "ticks": st.ticks,
        "fallbacks": st.fallbacks,
        "reanchors": st.reanchors,
        "ref_jumps": st.ref_jumps,
        "solve_ms": dist(st.solve_ms),
        "intervention_rad": dist(st.intervention_rad),
    }
