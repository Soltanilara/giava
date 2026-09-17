"""30-second smoke test for a machine's config: run this first on any new
computer, before wiring the adapter into a real teleop loop.

    python -m tube_mpc.check --config my_config.yaml

Verifies, in order:
  1. the URDF parses and the joint set looks right
  2. the LQR gain stabilizes the model and the mRPI (s, alpha) exists
  3. the tightened constraint sets are nonempty (W fits the limits)
  4. a simulated run under worst-case-box disturbances has zero violations
  5. solve times fit inside the configured dt
Exit code 0 = ready; 1 = fix the printed item before going near the robot.
"""

import argparse
import sys
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="FilterConfig yaml (default: built-in defaults)")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    from tube_mpc.adapter import FilterRunner, SimAdapter
    from tube_mpc.config import FilterConfig

    cfg = FilterConfig.load(args.config) if args.config else FilterConfig()

    # 1-3: construction runs the URDF parse, DARE, mRPI, and margin checks
    try:
        names, model, mpc = cfg.build_mpc()
    except Exception as e:
        print(f"FAIL (setup): {e}")
        return 1
    lim = model.limits
    n = model.n
    print(f"ok  {n} joints ({cfg.joint_prefix}*): {names}")
    mq = mpc.margins.state_inf[:n]
    mv = mpc.margins.state_inf[n:]
    print(f"ok  terminal tightening: worst q margin {mq.max():.4f} rad "
          f"({100*(2*mq/(lim.q_max-lim.q_min)).max():.1f}% of narrowest range), "
          f"worst v margin {mv.max():.4f} rad/s")

    # 4-5: simulated closed loop with an adversarial target and box noise
    adapter = SimAdapter(cfg)
    runner = FilterRunner(cfg, adapter)
    rng = np.random.default_rng(7)
    times, viol, fallbacks = [], 0, 0
    for k in range(args.steps):
        t = k * cfg.dt
        adapter.target = 0.5 * (lim.q_min + lim.q_max) + \
            0.6 * (lim.q_max - lim.q_min) * np.sin(2 * np.pi * 0.5 * t + np.arange(n))
        if args.steps // 3 < k < 2 * args.steps // 3:
            adapter.target = lim.q_max + 0.5  # infeasible lunge
        t0 = time.perf_counter()
        out = runner.step()
        times.append(time.perf_counter() - t0)
        fallbacks += out["fallback"]
        q, v = adapter.read_state()
        if (np.any(q > lim.q_max + 1e-6) or np.any(q < lim.q_min - 1e-6)
                or np.any(np.abs(v) > lim.v_max + 1e-6)):
            viol += 1
    st = np.array(times) * 1e3
    budget = cfg.dt * 1e3
    print(f"{'ok ' if viol == 0 else 'FAIL'} constraint violations: {viol}/{args.steps}")
    print(f"{'ok ' if fallbacks < args.steps // 10 else 'WARN'} fallback ticks: "
          f"{fallbacks}/{args.steps}")
    fits = np.quantile(st, 0.99) < budget
    print(f"{'ok ' if fits else 'FAIL'} solve time: mean {st.mean():.2f} ms, "
          f"p99 {np.quantile(st, .99):.2f} ms, max {st.max():.2f} ms "
          f"(budget {budget:.0f} ms)")

    ready = viol == 0 and fits
    print("READY" if ready else "NOT READY -- fix the FAIL lines above")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
