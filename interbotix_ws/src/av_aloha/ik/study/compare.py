"""Phase 4+ — variant comparison against the baseline.

Reads results/<variant>/summary.csv for every requested variant and prints
(1) a per-trajectory table for a chosen metric, (2) a suite-level aggregate
table with absolute values and deltas vs baseline.  Verifies that every run
used the identical frozen suite (conditions.json suite_hash).

    python compare.py                            # all variants found, key metrics
    python compare.py --metric ori_deg_p95
    python compare.py --variants smoothing collision
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"

## Trajectory groups for aggregation (feasible tracking vs designed-infeasible).
CLEAN_TRACKING = (
    "trans_x", "trans_y", "trans_z", "trans_diag", "trans_lissajous",
    "rot_home", "rot_low", "rot_forward",
    "teleop_grasp", "teleop_grasp_side", "teleop_grasp_yaw",
    "near_singular", "middle_gaze_sweep",
    "mid_trans_x", "mid_trans_y", "mid_trans_z", "mid_trans_diag",
    "mid_trans_lissajous", "mid_rot_home", "mid_rot_low", "mid_approach",
)
GRASP_FAMILY = ("teleop_grasp", "teleop_grasp_side", "teleop_grasp_yaw")
STRESS = ("arms_converge", "self_fold", "reach_limit", "wrist_twist",
          "jitter_teleop", "mid_reach_limit")

KEY_METRICS = (
    # (column, aggregate-fn over trajectories, format, label)
    ("pos_mm_p95", "mean", "{:8.2f}", "pos p95 mm (clean mean)"),
    ("pos_mm_max", "max", "{:8.1f}", "pos max mm (clean worst)"),
    ("ori_deg_p95", "mean", "{:8.2f}", "ori p95 ° (clean mean)"),
    ("solve_ms_mean", "mean", "{:8.2f}", "solve mean ms"),
    ("solve_ms_p95", "max", "{:8.2f}", "solve p95 ms (worst traj)"),
    ("iters_mean", "mean", "{:8.1f}", "LM iters (mean)"),
    ("config_jumps", "sum", "{:8.0f}", "config jumps (suite total)"),
    ("jerk_rms", "mean", "{:8.1f}", "jerk RMS (mean)"),
    ("nonconverged_frac", "mean", "{:8.3f}", "nonconverged (mean frac)"),
    ("limit_violation_frac", "mean", "{:8.3f}", "limit dwell (mean frac)"),
    ("self_clearance_min_mm", "min", "{:8.1f}", "worst clearance mm"),
    ("manip_hands_min", "min", "{:8.4f}", "manip hands (suite min)"),
)


def load(variant: str):
    d = RESULTS / variant
    rows = {r["trajectory"]: r for r in csv.DictReader(open(d / "summary.csv"))}
    cond = json.loads((d / "conditions.json").read_text())
    return rows, cond


def fval(row, col):
    v = row.get(col, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def aggregate(rows, col, how, subset):
    vals = np.asarray([fval(rows[t], col) for t in subset if t in rows])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan
    return {"mean": np.mean, "max": np.max, "min": np.min, "sum": np.sum}[how](vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--metric", default=None,
                    help="per-trajectory table for this column")
    args = ap.parse_args()

    variants = args.variants or sorted(
        p.name for p in RESULTS.iterdir()
        if (p / "summary.csv").exists() and p.name != "baseline"
    )
    base_rows, base_cond = load("baseline")
    print(f"suite: rev {base_cond['suite_rev']}  hash {base_cond['suite_hash']}  "
          f"backend {base_cond['backend']}")

    all_rows = {"baseline": base_rows}
    for v in variants:
        rows, cond = load(v)
        if cond["suite_hash"] != base_cond["suite_hash"]:
            print(f"!! {v}: SUITE HASH MISMATCH ({cond['suite_hash']}) — "
                  "not comparable, excluded")
            continue
        all_rows[v] = rows

    names = list(all_rows.keys())

    if args.metric:
        col = args.metric
        print(f"\nper-trajectory {col}:")
        print(f"{'trajectory':20s} " + " ".join(f"{n[:12]:>12s}" for n in names))
        for t in base_rows:
            cells = [f"{fval(all_rows[n].get(t, {}), col):12.3f}" for n in names]
            print(f"{t:20s} " + " ".join(cells))
        return

    print(f"\n{'metric':28s} " + " ".join(f"{n[:14]:>14s}" for n in names))
    print(f"{'':28s} " + " ".join(f"{'(Δ vs base)' if n != 'baseline' else '':>14s}"
                                  for n in names))
    for col, how, fmt, label in KEY_METRICS:
        subset = CLEAN_TRACKING if col.startswith(("pos_", "ori_")) else tuple(base_rows)
        base_val = aggregate(base_rows, col, how, subset)
        cells = []
        for n in names:
            val = aggregate(all_rows[n], col, how, subset)
            if n == "baseline":
                cells.append(f"{fmt.format(val):>14s}")
            else:
                delta = val - base_val
                cells.append(f"{fmt.format(val)}({delta:+.4g})"[:14].rjust(14))
        print(f"{label:28s} " + " ".join(cells))

    # grasp-family focus (the pos/ori tradeoff testbed)
    print("\ngrasp family (pos_mm_p95 / ori_deg_p95 / persistent end error):")
    for t in GRASP_FAMILY:
        cells = []
        for n in names:
            r = all_rows[n].get(t, {})
            end = fval(r, "end_pos_err_mm")
            cells.append(
                f"{fval(r,'pos_mm_p95'):6.1f}/{fval(r,'ori_deg_p95'):5.1f}"
            )
        print(f"{t:20s} " + " ".join(f"{c:>14s}" for c in cells))
    print("\nstress (recovery_ms / end_pos_err_mm):")
    for t in ("reach_limit", "mid_reach_limit"):
        cells = []
        for n in names:
            r = all_rows[n].get(t, {})
            cells.append(f"{fval(r,'recovery_ms'):6.0f}/{fval(r,'end_pos_err_mm'):6.2f}")
        print(f"{t:20s} " + " ".join(f"{c:>14s}" for c in cells))


if __name__ == "__main__":
    main()
