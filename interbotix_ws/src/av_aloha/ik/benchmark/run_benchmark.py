"""IK performance benchmark for the three-arm GIAVA setup (PyRoki).

Two modes:

  compare : run every cost variant (pure / +collision / +manipulability /
            +custom / full) over the same workloads and metrics.
  sweep   : randomly sample weights within a given cost structure and rank them
            by the teleoperation score, to find the settings that are actually
            good for smooth, intuitive teleoperation.

Examples
--------
  python run_benchmark.py compare --steps 200 --out results/compare
  python run_benchmark.py compare --variants pure full --workloads smooth_track jitter_track
  python run_benchmark.py sweep --structure full --samples 60 \
      --workloads smooth_track fast_track jitter_track crossing_arms --out results/sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyroki as pk
from pyroki.collision import RobotCollision
from yourdfpy import URDF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics as M  # noqa: E402
import workloads as W  # noqa: E402
from solver import (  # noqa: E402
    PRESETS,
    ControllerConfig,
    CostStructure,
    ThreeArmIK,
    Weights,
    preset,
)

DEFAULT_URDF = "/home/devi/giava/giava.urdf"
LEFT_EE_LINK = "left_gripper_base"
RIGHT_EE_LINK = "right_gripper_base"
MIDDLE_EE_LINK = "middle_camera_cover"
TARGET_LINKS = (LEFT_EE_LINK, RIGHT_EE_LINK, MIDDLE_EE_LINK)


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
def run_workload(
    ik: ThreeArmIK,
    evaluator: M.StateEvaluator,
    workload: W.Workload,
    weights: Weights,
    controller: ControllerConfig,
    q0: np.ndarray,
) -> M.Rollout:
    ik.reset(q0)
    ik.warmup(weights, controller, world_obstacle=workload.obstacle_at(0))
    ik.reset(q0)

    rollout = M.Rollout(dt=controller.dt)
    for t in range(len(workload)):
        obstacle = workload.obstacle_at(t)
        t0 = time.perf_counter()
        q = ik.step(
            workload.positions[t],
            workload.wxyzs[t],
            weights,
            controller,
            world_obstacle=obstacle,
        )
        rollout.solve_ms.append((time.perf_counter() - t0) * 1e3)
        rollout.iterations.append(getattr(ik, "last_iterations", 0))

        ee_pos, _ = evaluator.ee_poses(q)
        pos_err, ori_err = evaluator.pose_error(
            q, workload.positions[t], workload.wxyzs[t]
        )
        manip_t, manip_6, cond = evaluator.manipulability(q)

        rollout.q.append(q)
        rollout.target_pos.append(np.asarray(workload.positions[t]))
        rollout.target_wxyz.append(np.asarray(workload.wxyzs[t]))
        rollout.ee_pos.append(ee_pos)
        rollout.pos_err.append(pos_err)
        rollout.ori_err.append(ori_err)
        rollout.manip_t.append(manip_t)
        rollout.manip_6.append(manip_6)
        rollout.cond.append(cond)
        rollout.limit_margin.append(evaluator.limit_margin(q))
        rollout.self_clear.append(evaluator.self_clearance(q))
        rollout.world_clear.append(evaluator.world_clearance(q, obstacle))
    return rollout


def evaluate_config(
    robot: pk.Robot,
    robot_coll: Optional[RobotCollision],
    structure: CostStructure,
    weights: Weights,
    controller: ControllerConfig,
    workload_list: Sequence[W.Workload],
    q0: np.ndarray,
    evaluator: M.StateEvaluator,
    solver_cache: Dict[tuple, ThreeArmIK],
    saturation_reference: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """Run one (structure, weights) configuration over every workload."""
    key = structure.key()
    if key not in solver_cache:
        solver_cache[key] = ThreeArmIK(
            robot=robot,
            robot_coll=robot_coll,
            target_link_names=TARGET_LINKS,
            structure=structure,
        )
    ik = solver_cache[key]

    per_workload: Dict[str, Dict[str, float]] = {}
    for workload in workload_list:
        rollout = run_workload(ik, evaluator, workload, weights, controller, q0)
        per_workload[workload.name] = M.summarize(
            rollout,
            velocity_limit=(
                saturation_reference
                if saturation_reference is not None
                else controller.velocity_limit
            ),
            arm_joint_mask=evaluator.arm_joint_mask,
        )
    return per_workload


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
REPORT_COLUMNS = [
    "score",
    "pos_err_mean_mm",
    "pos_err_p95_mm",
    "ori_err_mean_deg",
    "ee_jerk_rms",
    "joint_jerk_rms",
    "lag_ms",
    "config_jumps",
    "vel_saturation_frac",
    "manip_min",
    "near_limit_frac",
    "limit_violation_frac",
    "self_clear_min_m",
    "self_collision_frac",
    "world_clear_min_m",
    "world_collision_frac",
    "solve_ms_mean",
    "solve_ms_p95",
    "iters_mean",
]


def print_table(rows: List[Dict[str, object]], label_key: str, columns=REPORT_COLUMNS):
    cols = [label_key] + [c for c in columns if any(c in r for r in rows)]
    widths = {c: max(len(c), 8) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(_fmt(r.get(c, ""))))
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(_fmt(r.get(c, "")).ljust(widths[c]) for c in cols))


def _fmt(v) -> str:
    if isinstance(v, float):
        if not np.isfinite(v):
            return "-"
        if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0):
            return f"{v:.3g}"
        return f"{v:.3f}"
    return str(v)


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
def auto_ignore_pairs(
    robot: pk.Robot,
    robot_coll: RobotCollision,
    q_samples: np.ndarray,
    always_frac: float = 0.999,
) -> tuple:
    """Link pairs that are 'in collision' in (almost) every sampled configuration.

    The capsule approximation of neighbouring links overlaps permanently for this
    URDF.  Those pairs carry no information, and leaving them in makes the
    self-collision cost push against a violation it can never resolve, so they
    are removed from the model used for both optimization and measurement.
    """
    import jax.numpy as jnp

    dists = np.asarray(
        robot_coll.compute_self_collision_distance(robot, jnp.asarray(q_samples))
    )  # (num_samples, num_pairs); row 0 is the home configuration
    always_colliding = np.mean(dists < 0.0, axis=0) >= always_frac
    colliding_at_home = dists[0] < 0.0
    bad = always_colliding | colliding_at_home
    names = robot_coll.link_names
    return tuple(
        (names[robot_coll.active_idx_i[p]], names[robot_coll.active_idx_j[p]])
        for p in np.flatnonzero(bad)
    )


def load_robot(urdf_path: str, auto_ignore: bool = True, seed: int = 0):
    urdf = URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollision.from_urdf(urdf)
    if not auto_ignore:
        return robot, robot_coll

    rng = np.random.default_rng(seed)
    lower = np.asarray(robot.joints.lower_limits)
    upper = np.asarray(robot.joints.upper_limits)
    q_samples = np.concatenate(
        [
            np.asarray(robot.joint_var_cls(0).default_factory(), dtype=np.float32)[None],
            rng.uniform(lower, upper, size=(64, lower.shape[0])).astype(np.float32),
        ]
    )
    ignore = auto_ignore_pairs(robot, robot_coll, q_samples)
    if ignore:
        print(f"auto-ignoring {len(ignore)} always-colliding link pairs")
        robot_coll = RobotCollision.from_urdf(urdf, user_ignore_pairs=ignore)
    return robot, robot_coll


def make_context(args):
    robot, robot_coll = load_robot(args.urdf, auto_ignore=not args.no_auto_ignore)
    target_indices = np.asarray(
        [robot.links.names.index(n) for n in TARGET_LINKS], dtype=np.int32
    )
    evaluator = M.StateEvaluator(robot, robot_coll, target_indices)
    q0 = np.asarray(robot.joint_var_cls(0).default_factory(), dtype=np.float32)
    base_pos, base_wxyz = W.home_poses(robot, q0, target_indices)
    workload_list = W.build(args.workloads, base_pos, base_wxyz, args.steps, args.dt)
    controller = ControllerConfig(
        dt=args.dt,
        velocity_limit=args.velocity_limit,
        clamp_velocity=not args.no_clamp,
        accel_limit=args.accel_limit,
        target_lpf=args.target_lpf,
        output_lpf=args.output_lpf,
    )
    return robot, robot_coll, evaluator, q0, workload_list, controller


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def cmd_compare(args) -> None:
    robot, robot_coll, evaluator, q0, workload_list, controller = make_context(args)
    solver_cache: Dict[tuple, ThreeArmIK] = {}

    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    variants = list(args.variants)
    tuned: Dict[str, object] = {}
    if args.weights_json:
        import json as _json
        with open(args.weights_json) as fh:
            blob = _json.load(fh)
        from batched import vector_to_weights
        tuned = {
            "structure": preset("full")[0],
            "weights": vector_to_weights(np.asarray(blob["vector"], dtype=np.float32)),
        }
        variants.append("tuned")
        print(f"loaded tuned weights from {args.weights_json}")

    for variant in variants:
        if variant == "tuned":
            structure, weights = tuned["structure"], tuned["weights"]
        else:
            structure, weights = preset(variant)
        print(f"\n=== {variant} ===")
        print(f"structure: {asdict(structure)}")
        t0 = time.perf_counter()
        per_workload = evaluate_config(
            robot, robot_coll, structure, weights, controller,
            workload_list, q0, evaluator, solver_cache,
        )
        wall = time.perf_counter() - t0

        for wl_name, met in per_workload.items():
            row: Dict[str, object] = {"variant": variant, "workload": wl_name}
            row["score"] = M.teleop_score(met)
            row.update(met)
            detail_rows.append(row)

        agg = M.aggregate(per_workload)
        agg_row: Dict[str, object] = {"variant": variant}
        agg_row["score"] = float(
            np.mean([M.teleop_score(m) for m in per_workload.values()])
        )
        agg_row.update(agg)
        agg_row.update(weights.flat())
        agg_row["wall_s"] = wall
        summary_rows.append(agg_row)

        print_table(
            [{"workload": k, "score": M.teleop_score(v), **v} for k, v in per_workload.items()],
            "workload",
        )

    summary_rows.sort(key=lambda r: r["score"])
    print("\n=== variant summary (mean over workloads, lower score is better) ===")
    print_table(summary_rows, "variant")

    if args.out:
        write_csv(f"{args.out}_summary.csv", summary_rows)
        write_csv(f"{args.out}_per_workload.csv", detail_rows)


def cmd_limits(args) -> None:
    """Sweep the safety limits: velocity ceiling x acceleration limit.

    Answers 'how much can the velocity clamp be loosened, and what does an
    acceleration limit buy back', with peak acceleration as the motor-stress
    proxy that the limits are actually there to bound.
    """
    robot, robot_coll, evaluator, q0, workload_list, _ = make_context(args)
    solver_cache: Dict[tuple, ThreeArmIK] = {}
    structure, weights = preset(args.structure)

    rows: List[Dict[str, object]] = []
    for vel in args.velocities:
        for acc in args.accels:
            controller = ControllerConfig(
                dt=args.dt,
                velocity_limit=vel if vel > 0.0 else float("inf"),
                clamp_velocity=vel > 0.0,
                # Held fixed across the grid so every cell solves the *same*
                # optimization problem and only the safety limits differ.
                nominal_velocity=args.velocity_limit,
                accel_limit=acc,
                target_lpf=args.target_lpf,
                output_lpf=args.output_lpf,
            )
            per_workload = evaluate_config(
                robot, robot_coll, structure, weights, controller,
                workload_list, q0, evaluator, solver_cache,
                saturation_reference=args.velocity_limit,
            )
            agg = M.aggregate(per_workload)
            row: Dict[str, object] = {
                "vel_limit": vel if vel > 0 else float("inf"),
                "accel_limit": acc if acc > 0 else float("inf"),
                "score": float(np.mean([M.teleop_score(m) for m in per_workload.values()])),
            }
            row.update(agg)
            rows.append(row)
            print(
                f"v={vel:5.2f} a={acc:7.2f} -> score={row['score']:7.2f} "
                f"pos_p95={agg.get('pos_err_p95_mm', float('nan')):7.2f}mm "
                f"lag={agg.get('lag_ms', float('nan')):6.1f}ms "
                f"acc_max={agg.get('joint_acc_max', float('nan')):8.1f} "
                f"jerk={agg.get('ee_jerk_rms', float('nan')):8.2f}"
            )

    print("\n=== safety-limit sweep (sorted by score) ===")
    print_table(
        sorted(rows, key=lambda r: r["score"]),
        "vel_limit",
        columns=[
            "accel_limit", "score", "pos_err_p95_mm", "ori_err_mean_deg", "lag_ms",
            "joint_vel_max", "joint_acc_max", "joint_acc_p95", "ee_jerk_rms",
            "vel_saturation_frac", "config_jumps", "self_clear_min_m",
        ],
    )
    if args.out:
        write_csv(f"{args.out}_limits.csv", rows)


def _sample_weights(rng: np.random.Generator, structure: CostStructure, args) -> Weights:
    """Log-uniform sampling over the weights that matter for feel."""

    def loguni(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    pos = loguni(*args.range_pos)
    ori = loguni(*args.range_ori)
    return Weights(
        position=(pos, pos, pos * 0.8),
        orientation=(ori, ori, ori * 0.2),
        manipulability=(
            (loguni(*args.range_manip),) * 2 + (0.0,)
            if structure.manipulability
            else (0.0, 0.0, 0.0)
        ),
        smoothness=loguni(*args.range_smooth) if structure.smoothness else 0.0,
        rest=loguni(*args.range_rest) if structure.rest else 0.0,
        self_collision=loguni(*args.range_coll) if structure.self_collision else 0.0,
        world_collision=loguni(*args.range_coll) if structure.world_collision else 0.0,
        limit_barrier=loguni(*args.range_barrier) if structure.limit_barrier else 0.0,
        collision_margin=float(rng.uniform(0.01, 0.06)),
        limit_barrier_margin=float(rng.uniform(0.05, 0.35)),
    )


def cmd_sweep(args) -> None:
    robot, robot_coll, evaluator, q0, workload_list, controller = make_context(args)
    solver_cache: Dict[tuple, ThreeArmIK] = {}
    structure, base_weights = preset(args.structure)
    rng = np.random.default_rng(args.seed)

    rows: List[Dict[str, object]] = []
    for i in range(args.samples):
        weights = base_weights if i == 0 else _sample_weights(rng, structure, args)
        t0 = time.perf_counter()
        per_workload = evaluate_config(
            robot, robot_coll, structure, weights, controller,
            workload_list, q0, evaluator, solver_cache,
        )
        agg = M.aggregate(per_workload)
        score = float(np.mean([M.teleop_score(m) for m in per_workload.values()]))
        row: Dict[str, object] = {"sample": i, "score": score}
        row.update(weights.flat())
        row.update(agg)
        row["wall_s"] = time.perf_counter() - t0
        rows.append(row)
        print(
            f"[{i + 1}/{args.samples}] score={score:8.3f} "
            f"pos_p95={agg.get('pos_err_p95_mm', float('nan')):6.2f}mm "
            f"jerk={agg.get('ee_jerk_rms', float('nan')):8.2f} "
            f"lag={agg.get('lag_ms', float('nan')):6.1f}ms "
            f"({row['wall_s']:.1f}s)"
        )

    rows.sort(key=lambda r: r["score"])
    print(f"\n=== top {min(args.top, len(rows))} configurations ===")
    print_table(
        rows[: args.top],
        "sample",
        columns=[
            "score", "w_pos_left", "w_ori_left", "w_smooth", "w_manip_left",
            "w_self_coll", "w_rest", "w_limit_barrier", "collision_margin",
            "pos_err_p95_mm", "ori_err_mean_deg", "ee_jerk_rms", "lag_ms",
            "config_jumps", "manip_min", "self_clear_min_m", "solve_ms_mean",
        ],
    )

    best = rows[0]
    print("\nbest weights:")
    print(json.dumps({k: v for k, v in best.items() if k.startswith(("w_", "collision_", "limit_barrier_"))}, indent=2))

    if args.out:
        write_csv(f"{args.out}_sweep.csv", rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--urdf", default=DEFAULT_URDF)
        sp.add_argument("--steps", type=int, default=200, help="steps per workload")
        sp.add_argument("--dt", type=float, default=0.05, help="control period [s]")
        sp.add_argument("--velocity-limit", type=float, default=2.0, help="rad/s cap")
        sp.add_argument("--no-clamp", action="store_true", help="disable the post-solve velocity clamp")
        sp.add_argument(
            "--accel-limit",
            type=float,
            default=0.0,
            help="rad/s^2 acceleration limit (0 = off). Bounds motor torque directly.",
        )
        sp.add_argument("--target-lpf", type=float, default=0.0, help="EMA on incoming targets, 0..1")
        sp.add_argument("--output-lpf", type=float, default=0.0, help="EMA on commanded joints, 0..1")
        sp.add_argument(
            "--workloads",
            nargs="+",
            default=list(W.DEFAULT_WORKLOADS),
            help="workload names, or replay:/path/to/episode.npz",
        )
        sp.add_argument("--out", default="", help="output path prefix for CSVs")
        sp.add_argument(
            "--weights-json", default="",
            help="optimize.py output; adds a 'tuned' variant using those weights",
        )
        sp.add_argument(
            "--no-auto-ignore",
            action="store_true",
            help="keep link pairs that are always in collision in the capsule model",
        )

    sp = sub.add_parser("compare", help="compare cost variants")
    common(sp)
    sp.add_argument("--variants", nargs="+", default=list(PRESETS), choices=list(PRESETS))
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("sweep", help="random-search weights within one structure")
    common(sp)
    sp.add_argument("--structure", default="full", choices=list(PRESETS))
    sp.add_argument("--samples", type=int, default=40)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--top", type=int, default=10)
    sp.add_argument("--range-pos", nargs=2, type=float, default=(5.0, 200.0))
    sp.add_argument("--range-ori", nargs=2, type=float, default=(0.2, 50.0))
    sp.add_argument("--range-smooth", nargs=2, type=float, default=(0.02, 5.0))
    sp.add_argument("--range-manip", nargs=2, type=float, default=(1e-4, 0.5))
    sp.add_argument("--range-rest", nargs=2, type=float, default=(1e-3, 0.5))
    sp.add_argument("--range-coll", nargs=2, type=float, default=(0.5, 50.0))
    sp.add_argument("--range-barrier", nargs=2, type=float, default=(0.05, 20.0))
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("limits", help="sweep velocity ceiling x acceleration limit")
    common(sp)
    sp.add_argument("--structure", default="full", choices=list(PRESETS))
    sp.add_argument(
        "--velocities", nargs="+", type=float, default=[1.0, 2.0, 4.0, 8.0, 0.0],
        help="rad/s ceilings to test; 0 means no velocity clamp",
    )
    sp.add_argument(
        "--accels", nargs="+", type=float, default=[0.0, 40.0, 20.0, 10.0, 5.0],
        help="rad/s^2 limits to test; 0 means no acceleration limit",
    )
    sp.set_defaults(func=cmd_limits)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
