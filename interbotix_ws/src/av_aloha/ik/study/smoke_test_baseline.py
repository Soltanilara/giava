"""Phase 1 smoke test: does the baseline behave sanely before any benchmarking?

Checks (not a benchmark — Phase 3 does real measurement):
1. Solving for the home poses from the home configuration stays at home.
2. A small reachable offset (2 cm) is reached to sub-mm / sub-0.1° error.
3. Warm-started tracking of a slow 200-step sine keeps errors small.
4. Timing sanity after warmup (mean/median/max over the 200 steps).

Run:  python smoke_test_baseline.py            (default JAX device)
      JAX_PLATFORMS=cpu python smoke_test_baseline.py
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jaxlie

import robot_model as rm
from baseline import BaselineIK


def pose_errors(robot, q, target_pos, target_wxyz, indices):
    """Euclidean position error [m] and geodesic orientation error [rad], per arm."""
    fk = robot.forward_kinematics(np.asarray(q, dtype=np.float32))
    pos_err, ori_err = [], []
    for k, idx in enumerate(indices):
        se3 = jaxlie.SE3(fk[idx])
        p = np.asarray(se3.translation())
        pos_err.append(np.linalg.norm(p - target_pos[k]))
        R_actual = se3.rotation()
        R_target = jaxlie.SO3(np.asarray(target_wxyz[k], dtype=np.float32))
        ori_err.append(float(np.linalg.norm((R_actual.inverse() @ R_target).log())))
    return np.asarray(pos_err), np.asarray(ori_err)


def main() -> None:
    print(f"JAX backend: {jax.default_backend()}")
    robot = rm.load()
    q0 = rm.home_config(robot)
    home_pos, home_wxyz = rm.home_poses(robot)
    indices = rm.target_link_indices(robot)

    ik = BaselineIK(robot, rm.TARGET_LINKS)
    ik.warmup(q0)

    # -- 1. identity solve --------------------------------------------------
    res = ik.solve(q0, home_pos, home_wxyz)
    p_err, o_err = pose_errors(robot, res.q, home_pos, home_wxyz, indices)
    dq = np.abs(res.q - q0).max()
    print("\n[1] home-pose solve from home config")
    print(f"    pos err  [mm]: {np.round(p_err * 1e3, 4)}")
    print(f"    ori err [deg]: {np.round(np.degrees(o_err), 4)}")
    print(f"    max |dq| [rad]: {dq:.5f}   iters: {res.iterations}   term: {res.termination}")

    # -- 2. small reachable offset ------------------------------------------
    tgt = home_pos.copy()
    tgt[:2, 2] -= 0.02  # both hands 2 cm down; camera holds
    res = ik.solve(q0, tgt, home_wxyz)
    p_err, o_err = pose_errors(robot, res.q, tgt, home_wxyz, indices)
    print("\n[2] 2 cm offset solve")
    print(f"    pos err  [mm]: {np.round(p_err * 1e3, 3)}")
    print(f"    ori err [deg]: {np.round(np.degrees(o_err), 3)}")
    print(f"    iters: {res.iterations}   term: {res.termination}")

    # -- 3./4. warm-started sine tracking + timing ---------------------------
    steps, dt, amp, freq = 200, 0.02, 0.06, 0.2
    q = q0.copy()
    perr_hist, oerr_hist, t_hist, it_hist = [], [], [], []
    for t in range(steps):
        tgt = home_pos.copy()
        tgt[:2, 0] += amp * np.sin(2 * np.pi * freq * t * dt)  # x sine, both hands
        res = ik.solve(q, tgt, home_wxyz)
        q = res.q
        p_err, o_err = pose_errors(robot, q, tgt, home_wxyz, indices)
        perr_hist.append(p_err[:2].max())
        oerr_hist.append(o_err[:2].max())
        t_hist.append(res.solve_ms)
        it_hist.append(res.iterations)
    perr = np.asarray(perr_hist) * 1e3
    oerr = np.degrees(np.asarray(oerr_hist))
    tms = np.asarray(t_hist)
    print(f"\n[3] 200-step x-sine tracking (amp {amp} m, {1/dt:.0f} Hz, f={freq} Hz)")
    print(f"    pos err  [mm]: mean {perr.mean():.3f}  median {np.median(perr):.3f}  max {perr.max():.3f}")
    print(f"    ori err [deg]: mean {oerr.mean():.3f}  median {np.median(oerr):.3f}  max {oerr.max():.3f}")
    print(f"\n[4] solve time [ms]: mean {tms.mean():.2f}  median {np.median(tms):.2f}  "
          f"p95 {np.percentile(tms, 95):.2f}  max {tms.max():.2f}")
    print(f"    iterations: mean {np.mean(it_hist):.1f}  max {max(it_hist)}")


if __name__ == "__main__":
    main()
