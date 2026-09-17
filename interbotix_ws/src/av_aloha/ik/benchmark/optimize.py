"""Find the cost weights that give the best teleoperation behaviour.

Redesign notes (why this replaces the preset comparison)
-------------------------------------------------------
1. A cost with weight zero is mathematically identical to a cost that is absent,
   so comparing hand-picked "structures" only ever tested one arbitrary weight
   point per structure -- which is why every structure scored the same.  Here a
   single `full` structure is used and *the weights decide*, with zero included
   in the search range.  Every earlier preset is a point in this space.
2. Weights are runtime arguments, so `jax.vmap` evaluates a whole population per
   pass (50x throughput).  The search is a population method for that reason.
3. The objective is priority-structured, not a weighted sum: pose accuracy is
   minimized continuously; smoothness, safety, collision and manipulability are
   hinges that cost nothing while acceptable and dominate once violated.
4. The controller's safety limits are held at the values the `limits` sweep
   found to dominate (v=6 rad/s, a=40 rad/s^2), so the search varies costs only.

Stages
------
  explore : Sobol-style log-uniform sampling over all weights, zeros included.
  refine  : cross-entropy method -- fit a Gaussian to the elite set in log space,
            resample, repeat.  Converges on the good region rather than sampling
            the whole space uniformly.
  ablate  : take the winner, zero / halve / double each term one at a time and
            re-measure.  This is what actually answers "does this cost earn its
            keep, and how sensitive is it".

Usage
-----
  python optimize.py --population 96 --explore-batches 3 --refine-rounds 4 \
      --steps 100 --out results/opt
  python optimize.py --ablate-only results/opt_best.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics as M  # noqa: E402
import workloads as W  # noqa: E402
from batched import (  # noqa: E402
    WEIGHT_FIELDS,
    BatchedIK,
    vector_to_weights,
    weights_to_vector,
)
from run_benchmark import TARGET_LINKS, load_robot, print_table, write_csv  # noqa: E402
from solver import ControllerConfig, CostStructure, preset  # noqa: E402

# (low, high, p_zero) per weight field.  p_zero > 0 means the term may be
# switched off entirely, which is how "is this cost needed at all" gets tested.
SEARCH_SPACE: Dict[str, Tuple[float, float, float]] = {
    "pos_left": (10.0, 400.0, 0.0),
    "pos_right": (10.0, 400.0, 0.0),
    "pos_middle": (5.0, 200.0, 0.0),
    "ori_left": (0.5, 100.0, 0.0),
    "ori_right": (0.5, 100.0, 0.0),
    "ori_middle": (0.1, 20.0, 0.0),
    "manip_left": (1e-4, 1.0, 0.30),
    "manip_right": (1e-4, 1.0, 0.30),
    "manip_middle": (1e-4, 0.5, 0.60),
    "smoothness": (0.02, 8.0, 0.15),
    "rest": (1e-3, 1.0, 0.30),
    "self_collision": (0.5, 300.0, 0.25),
    "world_collision": (0.5, 300.0, 0.25),
    "limit_barrier": (0.05, 50.0, 0.30),
    "collision_margin": (0.01, 0.08, 0.0),
    "limit_barrier_margin": (0.05, 0.35, 0.0),
}
# Margins are geometric parameters, not weights: never zero them.
NEVER_ZERO = ("collision_margin", "limit_barrier_margin")


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def sample_population(rng: np.random.Generator, n: int) -> np.ndarray:
    """Log-uniform samples with explicit zeros, one row per configuration."""
    cols = []
    for field in WEIGHT_FIELDS:
        lo, hi, p_zero = SEARCH_SPACE[field]
        vals = np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))
        if p_zero > 0.0 and field not in NEVER_ZERO:
            vals = np.where(rng.random(n) < p_zero, 0.0, vals)
        cols.append(vals)
    return np.stack(cols, axis=1).astype(np.float32)


def cem_resample(
    rng: np.random.Generator, elites: np.ndarray, n: int, spread: float
) -> np.ndarray:
    """Resample around the elite set in log space (zeros preserved as a rate)."""
    cols = []
    for j, field in enumerate(WEIGHT_FIELDS):
        lo, hi, p_zero = SEARCH_SPACE[field]
        col = elites[:, j]
        nonzero = col[col > 0]
        zero_rate = 1.0 - len(nonzero) / max(len(col), 1)
        if len(nonzero) == 0:
            cols.append(np.zeros(n, dtype=np.float32))
            continue
        mu = float(np.mean(np.log(nonzero)))
        sigma = float(np.std(np.log(nonzero))) if len(nonzero) > 1 else 0.5
        sigma = max(sigma, 0.15) * spread
        vals = np.exp(rng.normal(mu, sigma, size=n))
        vals = np.clip(vals, lo, hi)
        if field not in NEVER_ZERO and zero_rate > 0:
            vals = np.where(rng.random(n) < zero_rate, 0.0, vals)
        cols.append(vals)
    return np.stack(cols, axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def score_population(
    batched: BatchedIK,
    population: np.ndarray,
    workload_list: Sequence[W.Workload],
    controller: ControllerConfig,
    q0: np.ndarray,
    priority: M.Priority,
    measure_every: int = 5,
) -> List[Dict[str, float]]:
    """Evaluate every configuration; return one summary dict each."""
    per_config = batched.evaluate(
        population, workload_list, controller, q0, measure_every
    )
    rows: List[Dict[str, float]] = []
    for i, per_workload in enumerate(per_config):
        # Worst-case across workloads for the objective: a setting that is great
        # on smooth tracking and unusable on jitter is not a good setting.
        parts = [M.objective(m, priority) for m in per_workload.values()]
        agg = M.aggregate(per_workload)
        rows.append(
            {
                "objective": float(np.max([p["objective"] for p in parts])),
                "objective_mean": float(np.mean([p["objective"] for p in parts])),
                "primary": float(np.mean([p["primary"] for p in parts])),
                "secondary": float(np.mean([p["secondary"] for p in parts])),
                "safety": float(np.mean([p["safety"] for p in parts])),
                **{f: float(population[i, j]) for j, f in enumerate(WEIGHT_FIELDS)},
                **agg,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# ablation
# --------------------------------------------------------------------------- #
def ablate(
    batched: BatchedIK,
    best: np.ndarray,
    workload_list: Sequence[W.Workload],
    controller: ControllerConfig,
    q0: np.ndarray,
    priority: M.Priority,
) -> List[Dict[str, object]]:
    """Zero / halve / double each term of the winner, one at a time."""
    variants: List[Tuple[str, np.ndarray]] = [("baseline", best.copy())]
    for j, field in enumerate(WEIGHT_FIELDS):
        for label, factor in (("zeroed", 0.0), ("halved", 0.5), ("doubled", 2.0)):
            if field in NEVER_ZERO and factor == 0.0:
                continue
            v = best.copy()
            v[j] = best[j] * factor
            variants.append((f"{field} {label}", v))

    population = np.stack([v for _, v in variants])
    rows = score_population(
        batched, population, workload_list, controller, q0, priority
    )
    base_obj = rows[0]["objective"]
    out: List[Dict[str, object]] = []
    for (label, _), row in zip(variants, rows):
        out.append(
            {
                "change": label,
                "objective": row["objective"],
                "delta": row["objective"] - base_obj,
                "primary": row["primary"],
                "secondary": row["secondary"],
                "safety": row["safety"],
                "pos_err_p95_mm": row.get("pos_err_p95_mm", float("nan")),
                "ori_err_mean_deg": row.get("ori_err_mean_deg", float("nan")),
                "ee_jerk_rms": row.get("ee_jerk_rms", float("nan")),
                "config_jumps": row.get("config_jumps", float("nan")),
                "manip_min": row.get("manip_min", float("nan")),
                "self_collision_frac": row.get("self_collision_frac", float("nan")),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--urdf", default="/home/devi/giava/giava.urdf")
    p.add_argument("--population", type=int, default=96)
    p.add_argument("--explore-batches", type=int, default=3)
    p.add_argument("--refine-rounds", type=int, default=4)
    p.add_argument("--elite-frac", type=float, default=0.15)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=12)
    p.add_argument(
        "--workloads", nargs="+",
        default=["smooth_track", "fast_track", "jitter_track", "crossing_arms"],
    )
    p.add_argument("--velocity-limit", type=float, default=6.0)
    p.add_argument("--accel-limit", type=float, default=40.0)
    p.add_argument("--nominal-velocity", type=float, default=2.0)
    p.add_argument("--out", default="results/opt")
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument(
        "--max-iterations", type=int, default=20,
        help="Levenberg-Marquardt budget per tick. Bounds the batched solve "
             "(vmap waits for the slowest config) and matches what a 20 Hz "
             "control loop can actually afford.",
    )
    p.add_argument(
        "--measure-every", type=int, default=5,
        help="subsample manipulability/clearance every k steps (tracking and "
             "jerk are always measured every step)",
    )
    args = p.parse_args()

    robot, robot_coll = load_robot(args.urdf, auto_ignore=True)
    structure, _ = preset("full")
    batched = BatchedIK(
        robot, robot_coll, TARGET_LINKS, structure,
        max_iterations=args.max_iterations,
    )
    q0 = np.asarray(robot.joint_var_cls(0).default_factory(), dtype=np.float32)
    idx = np.asarray([robot.links.names.index(n) for n in TARGET_LINKS])
    base_pos, base_wxyz = W.home_poses(robot, q0, idx)
    workload_list = W.build(args.workloads, base_pos, base_wxyz, args.steps, args.dt)
    controller = ControllerConfig(
        dt=args.dt,
        velocity_limit=args.velocity_limit,
        clamp_velocity=args.velocity_limit > 0,
        accel_limit=args.accel_limit,
        nominal_velocity=args.nominal_velocity,
    )
    priority = M.Priority()
    rng = np.random.default_rng(args.seed)

    print(
        f"population={args.population} workloads={args.workloads} steps={args.steps}\n"
        f"controller: v<={args.velocity_limit} a<={args.accel_limit} dt={args.dt}\n"
        f"PRIMARY   (minimized continuously): position p95 / {priority.pos_scale_mm} mm, "
        f"orientation p95 / {priority.ori_scale_deg} deg\n"
        f"SECONDARY (zero cost until exceeded, weight {priority.secondary_weight}): "
        f"ee_jerk<={priority.max_ee_jerk}, jumps<={priority.max_config_jumps}, "
        f"lag<={priority.max_lag_ms}ms, saturation<={priority.max_vel_saturation}, "
        f"iters<={priority.max_iters}\n"
        f"SAFETY    (zero cost until exceeded, weight {priority.safety_weight}): "
        f"near_limit<={priority.max_near_limit_frac}, limit_viol<={priority.max_limit_violation_frac}, "
        f"self_coll<={priority.max_self_collision_frac}, world_coll<={priority.max_world_collision_frac}, "
        f"manip>={priority.min_manip}\n"
    )

    all_rows: List[Dict[str, float]] = []

    # -- stage 1: explore --------------------------------------------------- #
    for b in range(args.explore_batches):
        pop = sample_population(rng, args.population)
        if b == 0:
            # Seed with the old hand-picked presets so they appear in the ranking.
            for k, name in enumerate(("full", "pure_smooth", "collision", "manipulability")):
                _, w = preset(name)
                pop[k] = weights_to_vector(w)
        t0 = time.perf_counter()
        rows = score_population(
            batched, pop, workload_list, controller, q0, priority, args.measure_every
        )
        all_rows.extend(rows)
        best = min(rows, key=lambda r: r["objective"])
        print(
            f"[explore {b + 1}/{args.explore_batches}] best={best['objective']:8.2f} "
            f"(primary={best['primary']:6.2f} secondary={best['secondary']:5.2f} "
            f"safety={best['safety']:5.2f})  {time.perf_counter() - t0:5.1f}s",
            flush=True,
        )

    # -- stage 2: refine ---------------------------------------------------- #
    n_elite = max(4, int(args.elite_frac * args.population))
    for r in range(args.refine_rounds):
        ranked = sorted(all_rows, key=lambda x: x["objective"])[:n_elite]
        elites = np.stack(
            [np.asarray([row[f] for f in WEIGHT_FIELDS], dtype=np.float32) for row in ranked]
        )
        spread = max(0.35, 1.0 - 0.2 * r)
        pop = cem_resample(rng, elites, args.population, spread)
        pop[0] = elites[0]  # elitism: never lose the incumbent
        t0 = time.perf_counter()
        rows = score_population(
            batched, pop, workload_list, controller, q0, priority, args.measure_every
        )
        all_rows.extend(rows)
        best = min(all_rows, key=lambda x: x["objective"])
        print(
            f"[refine  {r + 1}/{args.refine_rounds}] best={best['objective']:8.2f} "
            f"(primary={best['primary']:6.2f} secondary={best['secondary']:5.2f} "
            f"safety={best['safety']:5.2f})  {time.perf_counter() - t0:5.1f}s",
            flush=True,
        )

    all_rows.sort(key=lambda r: r["objective"])
    print(f"\n=== top {args.top} of {len(all_rows)} configurations ===")
    print_table(
        all_rows[: args.top],
        "objective",
        columns=[
            "primary", "secondary", "safety", "pos_left", "ori_left", "smoothness",
            "manip_left", "self_collision", "rest", "limit_barrier",
            "pos_err_p95_mm", "ori_err_mean_deg", "ee_jerk_rms", "config_jumps",
            "manip_min", "self_collision_frac", "iters_mean",
        ],
    )

    best_row = all_rows[0]
    best_vec = np.asarray([best_row[f] for f in WEIGHT_FIELDS], dtype=np.float32)
    best_weights = vector_to_weights(best_vec)
    print("\n=== best weights ===")
    print(json.dumps(best_weights.flat(), indent=2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}_best.json", "w") as f:
        json.dump(
            {
                "weights": best_weights.flat(),
                "vector": [float(x) for x in best_vec],
                "objective": best_row["objective"],
                "controller": {
                    "velocity_limit": args.velocity_limit,
                    "accel_limit": args.accel_limit,
                    "nominal_velocity": args.nominal_velocity,
                    "dt": args.dt,
                },
                "workloads": args.workloads,
            },
            f,
            indent=2,
        )
    print(f"wrote {args.out}_best.json")
    write_csv(f"{args.out}_search.csv", all_rows)

    # -- stage 3: ablate ---------------------------------------------------- #
    if not args.skip_ablation:
        print("\n=== ablation around the winner (delta > 0 means worse) ===")
        rows = ablate(batched, best_vec, workload_list, controller, q0, priority)
        rows_sorted = [rows[0]] + sorted(rows[1:], key=lambda r: -abs(float(r["delta"])))
        print_table(
            rows_sorted,
            "change",
            columns=[
                "delta", "objective", "primary", "secondary", "safety",
                "pos_err_p95_mm", "ori_err_mean_deg", "ee_jerk_rms",
                "config_jumps", "manip_min", "self_collision_frac",
            ],
        )
        write_csv(f"{args.out}_ablation.csv", rows_sorted)


if __name__ == "__main__":
    main()
