"""Single-arm demo: tube MPC vs. a naive clamp-style tracker on an
adversarial reference, using the real right-arm limits from giava.urdf.

The scripted reference imitates an aggressive operator: fast in-workspace
motion, then a lunge *past* the shoulder/elbow joint limits (an infeasible
command every VR teleop system eventually receives), then a fast return.
The naive tracker -- clamp position to limits, clamp velocity, clamp
acceleration, no lookahead -- is what "IK + safety clamps" amounts to.

Run:  python -m tube_mpc.demo_single_arm
Writes metrics to stdout and a plot to tube_mpc/out/demo_single_arm.png.
"""

import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tube_mpc.controller import TubeMPC
from tube_mpc.model import KinematicModel
from tube_mpc.reference import decayed_velocity_reference
from tube_mpc.urdf_limits import parse_urdf_limits

REPO = pathlib.Path(__file__).resolve().parent.parent
DT = 0.02          # 50 Hz control loop -- confirm against the real teleop loop
A_MAX = 12.0       # rad/s^2 placeholder -- identify from logs (see README)
HORIZON = 25       # 0.5 s of preview
W_Q, W_V = 2e-3, 1e-2  # disturbance box placeholder -- calibrate from logs
T_END = 12.0
SEED = 0


def scripted_reference(t: np.ndarray, lim) -> np.ndarray:
    """(T, n) joint reference: benign -> infeasible lunge -> fast return."""
    n = lim.n
    mid = 0.5 * (lim.q_min + lim.q_max)
    amp = 0.35 * (lim.q_max - lim.q_min)
    ref = np.tile(mid, (len(t), 1))
    ref += 0.4 * amp * np.sin(2 * np.pi * 0.4 * t)[:, None] * np.ones(n)
    lunge = (t >= 4.0) & (t < 8.0)
    # push shoulder & elbow 0.4 rad past their upper limits
    for j in (1, 2):
        ref[lunge, j] = lim.q_max[j] + 0.4
    fast = t >= 8.0
    ref[fast] = mid + 0.45 * amp * np.sin(2 * np.pi * 1.2 * t[fast, None] + 1.0)
    return ref


def naive_tracker(model: KinematicModel, x: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Clamp-everything baseline: no preview, no braking-distance reasoning."""
    lim = model.limits
    q, v = x[: model.n], x[model.n:]
    q_tgt = np.clip(q_ref, lim.q_min, lim.q_max)
    v_des = np.clip((q_tgt - q) / DT, -lim.v_max, lim.v_max)
    return np.clip((v_des - v) / DT, -lim.a_max, lim.a_max)


def violations(lim, qs, vs):
    q_viol = np.maximum(qs - lim.q_max, lim.q_min - qs).max(axis=1)
    v_viol = (np.abs(vs) - lim.v_max).max(axis=1)
    return q_viol, v_viol


def main() -> None:
    rng = np.random.default_rng(SEED)
    names, lim = parse_urdf_limits(str(REPO / "giava.urdf"), "right_", A_MAX)
    n = lim.n
    print(f"joints: {names}")
    model = KinematicModel(lim, DT)
    w_box = np.concatenate([np.full(n, W_Q), np.full(n, W_V)])
    mpc = TubeMPC(model, w_box, horizon=HORIZON)

    steps = int(T_END / DT)
    t = np.arange(steps) * DT
    ref = scripted_reference(t, lim)
    x0 = np.concatenate([0.5 * (lim.q_min + lim.q_max), np.zeros(n)])
    disturb = rng.uniform(-1, 1, size=(steps, 2 * n)) * w_box

    logs = {}
    for label in ("naive", "tube_mpc"):
        x = x0.copy()
        qs, vs, us, times = [], [], [], []
        infeasible = 0
        for k in range(steps):
            if label == "naive":
                u = naive_tracker(model, x, ref[k])
            else:
                r_traj = decayed_velocity_reference(
                    ref[k], ref[max(k - 1, 0)], DT, HORIZON, v_cap=lim.v_max)
                t0 = time.perf_counter()
                out = mpc.solve(x, r_traj)
                times.append(time.perf_counter() - t0)
                if out["fallback"]:
                    infeasible += 1  # counts fallback ticks (shifted-plan applied)
                u = out["u"]
            qs.append(x[:n].copy())
            vs.append(x[n:].copy())
            us.append(u.copy())
            x = model.step(x, u, disturb[k])
        logs[label] = dict(q=np.array(qs), v=np.array(vs), u=np.array(us),
                           times=np.array(times), infeasible=infeasible)

    # ------------------------------------------------------------- metrics
    print(f"\n{'':14s}{'naive':>14s}{'tube_mpc':>14s}")
    rows = []
    for label in ("naive", "tube_mpc"):
        L = logs[label]
        qv, vv = violations(lim, L["q"], L["v"])
        track = np.linalg.norm(np.clip(ref, lim.q_min, lim.q_max) - L["q"], axis=1)
        rows.append([
            (qv > 1e-9).sum(), qv.max(), (vv > 1e-9).sum(),
            np.abs(np.diff(L["q"], axis=0)).max(), track.mean(), L["infeasible"]])
    labels = ["q-viol steps", "worst q-viol (rad)", "v-viol steps",
              "max |dq| step (rad)", "mean track err (rad)", "fallback ticks"]
    for i, name in enumerate(labels):
        a, b = rows[0][i], rows[1][i]
        print(f"{name:22s}{a:14.4g}{b:14.4g}")
    st = logs["tube_mpc"]["times"] * 1e3
    print(f"\nMPC solve time: mean {st.mean():.2f} ms | p99 {np.quantile(st, .99):.2f} ms "
          f"| max {st.max():.2f} ms  (budget {DT*1e3:.0f} ms)")

    # ---------------------------------------------------------------- plot
    out_dir = REPO / "tube_mpc" / "out"
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    j = 1  # shoulder: the joint the lunge attacks
    ax = axes[0]
    ax.plot(t, ref[:, j], "k--", lw=1, label="reference (infeasible lunge)")
    ax.plot(t, logs["naive"]["q"][:, j], label="naive clamp")
    ax.plot(t, logs["tube_mpc"]["q"][:, j], label="tube MPC")
    ax.axhline(lim.q_max[j], color="r", lw=0.8)
    ax.axhline(lim.q_min[j], color="r", lw=0.8)
    ax.set_ylabel(f"q[{j}] (rad)")
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    ax.plot(t, logs["naive"]["v"][:, j], label="naive clamp")
    ax.plot(t, logs["tube_mpc"]["v"][:, j], label="tube MPC")
    ax.axhline(lim.v_max[j], color="r", lw=0.8)
    ax.axhline(-lim.v_max[j], color="r", lw=0.8)
    ax.set_ylabel(f"qdot[{j}] (rad/s)")
    ax = axes[2]
    ax.plot(t, logs["naive"]["u"][:, j], label="naive clamp")
    ax.plot(t, logs["tube_mpc"]["u"][:, j], label="tube MPC")
    ax.axhline(lim.a_max[j], color="r", lw=0.8)
    ax.axhline(-lim.a_max[j], color="r", lw=0.8)
    ax.set_ylabel(f"u[{j}] (rad/s^2)")
    ax.set_xlabel("t (s)")
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle("Tube MPC vs naive clamping under an infeasible operator lunge (right arm, giava.urdf)")
    fig.tight_layout()
    path = out_dir / "demo_single_arm.png"
    fig.savefig(path, dpi=130)
    print(f"plot written to {path}")


if __name__ == "__main__":
    main()
