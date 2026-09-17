"""Calibrate the disturbance box W from logged robot data.

This is the ONE step you must redo on every machine (and after switching
command mode, servo gains, or firmware): the tube's guarantee is only as
strong as W is honest.

Input: an .npz file with
    q_cmd  : (T, n) commanded joint positions      [required]
    q_meas : (T, n) measured joint positions       [required]
    v_meas : (T, n) measured joint velocities      [optional; else finite diff]
    dt     : scalar sample period                  [optional; else --dt]

The model says x_{k+1} = A x_k + B u_k + w with u_k the commanded
acceleration. We reconstruct u_k from the command stream, compute the
one-step residual w_k on the measured stream, and bound it per joint by
the --percentile quantile (with a safety inflation factor).

Usage:
    python -m tube_mpc.calibrate_w log.npz --config my_config.yaml --write
    python -m tube_mpc.calibrate_w log.npz --dt 0.02          # print only
"""

import argparse

import numpy as np


def calibrate(q_cmd: np.ndarray, q_meas: np.ndarray, dt: float,
              v_meas: np.ndarray | None = None, percentile: float = 99.9,
              inflate: float = 1.25) -> tuple[np.ndarray, np.ndarray]:
    """Return (w_q, w_v): per-joint one-step residual bounds."""
    if q_cmd.shape != q_meas.shape or q_cmd.ndim != 2:
        raise ValueError("q_cmd and q_meas must both be (T, n)")
    if v_meas is None:
        v_meas = np.gradient(q_meas, dt, axis=0)
    v_cmd = np.gradient(q_cmd, dt, axis=0)
    a_cmd = np.gradient(v_cmd, dt, axis=0)

    # one-step model prediction from the measured state under commanded accel
    q_pred = q_meas[:-1] + dt * v_meas[:-1] + 0.5 * dt**2 * a_cmd[:-1]
    v_pred = v_meas[:-1] + dt * a_cmd[:-1]
    rq = np.abs(q_meas[1:] - q_pred)
    rv = np.abs(v_meas[1:] - v_pred)

    w_q = inflate * np.percentile(rq, percentile, axis=0)
    w_v = inflate * np.percentile(rv, percentile, axis=0)
    # never allow an exactly-zero bound (W must be full-dimensional)
    return np.maximum(w_q, 1e-6), np.maximum(w_v, 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help=".npz with q_cmd, q_meas [, v_meas, dt]")
    ap.add_argument("--dt", type=float, default=None,
                    help="sample period if the log has none")
    ap.add_argument("--percentile", type=float, default=99.9)
    ap.add_argument("--inflate", type=float, default=1.25,
                    help="safety factor on the percentile bound")
    ap.add_argument("--config", default=None,
                    help="FilterConfig yaml to compare against / update")
    ap.add_argument("--write", action="store_true",
                    help="write w_q/w_v back into --config")
    args = ap.parse_args()

    data = np.load(args.log)
    dt = float(data["dt"]) if "dt" in data else args.dt
    if dt is None:
        ap.error("log has no dt; pass --dt")
    v_meas = data["v_meas"] if "v_meas" in data else None
    w_q, w_v = calibrate(data["q_cmd"], data["q_meas"], dt, v_meas,
                         args.percentile, args.inflate)

    np.set_printoptions(precision=5, suppress=True)
    print(f"per-joint w_q (rad):   {w_q}")
    print(f"per-joint w_v (rad/s): {w_v}")
    print(f"\nconfig values (yaml):\nw_q: {w_q.tolist()}\nw_v: {w_v.tolist()}")

    if args.config:
        from tube_mpc.config import FilterConfig

        cfg = FilterConfig.load(args.config)
        cfg.w_q = [round(float(x), 6) for x in w_q]
        cfg.w_v = [round(float(x), 6) for x in w_v]
        # fail fast if the calibrated W leaves no room inside the limits
        try:
            cfg.build_mpc()
        except ValueError as e:
            print(f"\nCALIBRATION REJECTED: {e}")
            print(
                "A huge W usually means the double-integrator model is missing\n"
                "systematic behavior (servo lag, wrong dt, wrong command mode),\n"
                "so the residual is swallowing model error, not disturbance.\n"
                "Check dt against the log, check command_mode, and look at the\n"
                "residual time series before blaming the robot for being noisy.")
            raise SystemExit(1)
        if args.write:
            cfg.save(args.config)
            print(f"\nwrote w_q/w_v into {args.config} (margins verified nonempty)")
        else:
            print("\n(config check passed; re-run with --write to save)")


if __name__ == "__main__":
    main()
