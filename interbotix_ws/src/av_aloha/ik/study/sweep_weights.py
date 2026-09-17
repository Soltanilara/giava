"""Phase 5 — weight sensitivity sweeps.

Each sweep varies ONE weight over a grid, rolls the affected variant over a
fixed, pre-registered trajectory subset, and writes one CSV row per
(weight, trajectory).  Weights are runtime arguments, so a whole sweep uses a
single compilation.

    python sweep_weights.py --sweep smoothing
    python sweep_weights.py --sweep posori
    python sweep_weights.py --sweep centering
    python sweep_weights.py --sweep manipulability

Subsets (pre-registered; chosen for the failure mode each weight targets,
before sweep results were known):
- smoothing:   jump-prone + lag-sensitive: trans_y, trans_lissajous, rot_home,
               teleop_grasp, jitter_teleop, reach_limit
- posori:      the grasp family + rotation tests (the pos/ori tradeoff
               testbeds): teleop_grasp{,_side,_yaw}, rot_home, rot_forward,
               trans_lissajous (control: should be insensitive)
- centering:   limit-dwell + trap: wrist_twist, teleop_grasp_yaw, rot_forward,
               trans_y, near_singular
- manipulability: singularity + trap + control: near_singular, reach_limit,
               teleop_grasp_yaw, self_fold, trans_lissajous (control),
               rot_home (control)
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as M
import robot_model as rm
import trajectories as T
from baseline import BaselineIK
from variants import VariantIK

OUT = Path(__file__).resolve().parent / "results" / "sweeps"

GRIDS = {
    # 0.0 re-measures the baseline inside the same compiled structure
    # (sanity anchor: must reproduce results/baseline within noise).
    "smoothing": [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],  # w in w/(v_nom·dt)
    "posori": [2.0, 5.0, 10.0, 20.0, 40.0],  # ori_weight at pos_weight=50
    "centering": [0.0, 0.05, 0.2, 0.5, 1.0, 2.0],
    "manipulability": [0.0, 0.005, 0.02, 0.1, 0.5],
}
SUBSETS = {
    "smoothing": ("trans_y", "trans_lissajous", "rot_home", "teleop_grasp",
                  "jitter_teleop", "reach_limit"),
    "posori": ("teleop_grasp", "teleop_grasp_side", "teleop_grasp_yaw",
               "rot_home", "rot_forward", "trans_lissajous"),
    "centering": ("wrist_twist", "teleop_grasp_yaw", "rot_forward", "trans_y",
                  "near_singular"),
    "manipulability": ("near_singular", "reach_limit", "teleop_grasp_yaw",
                       "self_fold", "trans_lissajous", "rot_home"),
}
KEEP_COLS = (
    "pos_mm_mean", "pos_mm_p95", "pos_mm_max", "ori_deg_mean", "ori_deg_p95",
    "solve_ms_mean", "solve_ms_p95", "solve_ms_max", "iters_mean",
    "config_jumps", "jerk_rms", "vel_max", "nonconverged_frac",
    "limit_violation_frac", "limit_margin_min", "manip_hands_min",
    "manip_hands_mean", "self_clearance_min_mm", "recovery_ms",
    "end_pos_err_mm", "miss_frac",
)


def rollout(ik, traj, q0, **solve_kw):
    q = q0.copy()
    qs, tms, its, terms = [], [], [], []
    for i in range(len(traj)):
        res = ik.solve(q, traj.positions[i], traj.wxyzs[i], **solve_kw)
        q = res.q
        qs.append(q)
        tms.append(res.solve_ms)
        its.append(res.iterations)
        terms.append(res.termination)
    return np.stack(qs), np.asarray(tms), np.asarray(its), np.stack(terms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, choices=sorted(GRIDS))
    args = ap.parse_args()

    robot, urdf = rm.load(with_urdf=True)
    q0 = rm.home_config(robot)
    home_pos, home_wxyz = rm.home_poses(robot)
    suite = {t.name: t for t in T.build_suite(home_pos, home_wxyz)}
    subset = [suite[n] for n in SUBSETS[args.sweep]]
    evaluator = M.Evaluator(robot, urdf)

    if args.sweep == "posori":
        ik = BaselineIK(robot, rm.TARGET_LINKS)
    else:
        extra = args.sweep
        if extra == "smoothing":
            ik = VariantIK(robot, rm.TARGET_LINKS, "smoothing")
        elif extra == "centering":
            ik = VariantIK(robot, rm.TARGET_LINKS, "centering")
        else:
            ik = VariantIK(robot, rm.TARGET_LINKS, "manipulability")
    ik.warmup(q0)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for w in GRIDS[args.sweep]:
        for traj in subset:
            if args.sweep == "posori":
                kw = dict(ori_weights=np.full(3, w, dtype=np.float32))
            elif args.sweep == "smoothing":
                # grid value is the *weight* in w/(v_nom·dt); convert to scale
                kw = dict(extra_weight=w / (2.0 * T.DT))
            else:
                kw = dict(extra_weight=w)
            qs, tms, its, terms = rollout(ik, traj, q0, **kw)
            r = evaluator.evaluate(traj, qs, tms, its, terms, 100)
            row = {"weight": w, "trajectory": traj.name}
            row.update({k: r.get(k, "") for k in KEEP_COLS})
            rows.append(row)
            print(f"w={w:<7g} {traj.name:20s} pos p95 "
                  f"{float(r.get('pos_mm_p95') or np.nan):8.2f} mm  "
                  f"ori p95 {float(r.get('ori_deg_p95') or np.nan):6.2f}°  "
                  f"jumps {r.get('config_jumps', 0):3}  "
                  f"solve {r['solve_ms_mean']:5.2f} ms", flush=True)

    path = OUT / f"{args.sweep}.csv"
    with open(path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\n{time.perf_counter()-t0:.0f} s; wrote {path}")


if __name__ == "__main__":
    main()
