"""Phase 3+ — benchmark runner.

Rolls a solver variant over the frozen trajectory suite, computes metrics
(metrics.py), and records the complete experimental conditions so later
comparisons are genuinely comparable.

    JAX_PLATFORMS=cpu python run_study.py --variant baseline
    JAX_PLATFORMS=cpu python run_study.py --variant baseline --trajectories trans_x rot_home

Outputs under results/<variant>/:
    conditions.json   solver + environment + suite-hash record
    summary.csv       one row per trajectory, all metrics
    <traj>.npz        per-step history (q, solve_ms, iterations, terminations)

The variant registry starts with only `baseline`; Phase 4 adds one-residual
variants in variants.py without touching baseline.py.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax

import metrics as M
import robot_model as rm
import trajectories as T
from baseline import BaselineIK, DEFAULT_ORI_WEIGHT, DEFAULT_POS_WEIGHT


def suite_hash(suite) -> str:
    """SHA-256 over every commanded pose in the suite: proves all variants ran
    the identical benchmark."""
    h = hashlib.sha256()
    for tr in suite:
        h.update(tr.name.encode())
        h.update(np.ascontiguousarray(tr.positions).tobytes())
        h.update(np.ascontiguousarray(tr.wxyzs).tobytes())
    return h.hexdigest()[:16]


def make_variant(name: str, robot):
    """Variant registry.  Every variant must expose
    solve(q_init, positions, wxyzs) -> SolveResult and warmup(q0)."""
    if name == "baseline":
        return BaselineIK(robot, rm.TARGET_LINKS), {
            "solver": "BaselineIK (pose_cost_analytic_jac ×3 + limit_constraint)",
            "pos_weight": DEFAULT_POS_WEIGHT,
            "ori_weight": DEFAULT_ORI_WEIGHT,
            "max_iterations": 100,
            "lambda_initial": 1.0,
            "linear_solver": "dense_cholesky",
            "warm_start": "previous commanded configuration",
        }
    try:
        import variants  # Phase 4+

        return variants.make(name, robot)
    except ImportError:
        raise SystemExit(f"unknown variant {name!r} and no variants.py yet")


def rollout(ik, traj: T.Trajectory, q0: np.ndarray):
    """Warm-started tracking rollout.  Returns per-step history arrays."""
    q = q0.copy()
    qs, tms, its, terms = [], [], [], []
    for i in range(len(traj)):
        res = ik.solve(q, traj.positions[i], traj.wxyzs[i])
        q = res.q
        qs.append(q)
        tms.append(res.solve_ms)
        its.append(res.iterations)
        terms.append(res.termination)
    return (
        np.stack(qs),
        np.asarray(tms),
        np.asarray(its),
        np.stack(terms),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--trajectories", nargs="*", default=None,
                    help="subset by name (default: full suite)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-collision-metric", action="store_true",
                    help="skip the self-clearance metric (it is the slowest)")
    ap.add_argument("--clearance-model", choices=("capsule", "sphere"),
                    default="capsule",
                    help="collision model for the clearance METRIC (capsule = "
                    "backwards-comparable with earlier runs; sphere = the "
                    "validated pruned 180-sphere model)")
    args = ap.parse_args()

    robot, urdf = rm.load(with_urdf=True)
    q0 = rm.home_config(robot)
    home_pos, home_wxyz = rm.home_poses(robot)
    suite = T.build_suite(home_pos, home_wxyz)
    full_hash = suite_hash(suite)  # hash of the *full* frozen suite
    if args.trajectories:
        wanted = set(args.trajectories)
        unknown = wanted - {t.name for t in suite}
        if unknown:
            raise SystemExit(f"unknown trajectories: {sorted(unknown)}")
        suite = [t for t in suite if t.name in wanted]

    ik, solver_cfg = make_variant(args.variant, robot)
    out_dir = Path(args.out or Path(__file__).parent / "results" / args.variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    git_rev = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=Path(__file__).parent,
    ).stdout.strip()
    conditions = {
        "variant": args.variant,
        "solver": solver_cfg,
        "suite_rev": "2.1",
        "suite_hash": full_hash,
        "num_trajectories": len(suite),
        "dt": T.DT,
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "dtype": "float32",
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "git": git_rev,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if args.clearance_model == "sphere":
        conditions["clearance_model"] = "pruned-sphere-180"
    (out_dir / "conditions.json").write_text(json.dumps(conditions, indent=2))
    print(json.dumps(conditions, indent=2))

    coll_model = None
    if args.clearance_model == "sphere":
        from collision_models import pruned_sphere_collision

        coll_model = pruned_sphere_collision(
            urdf, Path(__file__).parent / "results")
    evaluator = M.Evaluator(robot, urdf, coll_model=coll_model)
    max_iter = solver_cfg.get("max_iterations", 100)

    rows = []
    t_suite = time.perf_counter()
    for traj in suite:
        ik.warmup(q0)
        t0 = time.perf_counter()
        qs, tms, its, terms = rollout(ik, traj, q0)
        wall = time.perf_counter() - t0
        row = evaluator.evaluate(
            traj, qs, tms, its, terms, max_iter,
            with_collision=not args.no_collision_metric,
        )
        row["rollout_wall_s"] = round(wall, 2)
        rows.append(row)
        np.savez_compressed(
            out_dir / f"{traj.name}.npz",
            q=qs, solve_ms=tms, iterations=its, terminations=terms,
            target_positions=traj.positions, target_wxyzs=traj.wxyzs,
            eval_start=traj.eval_start,
        )
        print(
            f"{traj.name:20s} pos p95 {row.get('pos_mm_p95', float('nan')):7.2f} mm  "
            f"ori p95 {row.get('ori_deg_p95', float('nan')):6.2f}°  "
            f"solve {row['solve_ms_mean']:5.2f} ms (max {row['solve_ms_max']:6.2f})  "
            f"iters {row['iters_mean']:4.1f}  "
            f"jumps {row.get('config_jumps', 0):3d}  "
            f"clear {row.get('self_clearance_min_mm', float('nan')):7.1f} mm"
        )

    keys: list = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nsuite wall time {time.perf_counter() - t_suite:.1f} s")
    print(f"wrote {out_dir}/summary.csv")


if __name__ == "__main__":
    main()
