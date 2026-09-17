"""Phase 2 — trajectory reachability validation.

For every trajectory, sample waypoints at 0.5 s intervals and ask a
*multi-seed oracle* whether each commanded pose is reachable:

    oracle(waypoint) = min over seeds of pose error after a 200-iteration
                       cold-start LM solve (seeds: home + K random
                       configurations drawn uniformly in the joint box)

This is deliberately not the warm-started single solve being benchmarked: a
multi-start solve with a 200-iteration budget is a much stronger search, so
"oracle fails" ≈ "pose is not reachable", independent of the tracking
controller's local behaviour.  (A sampling-based global method would be
stronger still; multi-start LM is the standard practical oracle.)

A waypoint counts as reachable if some seed achieves < 3 mm and < 1°.

Also reports the numeric max reach of the hand arms (max ‖p_ee − p_shoulder‖
over 20k random configurations) to contextualize the translation amplitudes.

Run:  JAX_PLATFORMS=cpu python validate_trajectories.py
Writes results to results/validation.json.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import jaxlie

import robot_model as rm
import trajectories as T
from baseline import BaselineIK

SAMPLE_EVERY_S = 0.5
NUM_RANDOM_SEEDS = 6
POS_TOL = 0.003  # m
ORI_TOL = np.radians(1.0)
RNG = np.random.default_rng(1234)


def max_reach(robot, ee_link: str, shoulder_link: str, n: int = 20000) -> float:
    """Max distance from `shoulder_link` to `ee_link` over random q."""
    lower = np.asarray(robot.joints.lower_limits)
    upper = np.asarray(robot.joints.upper_limits)
    qs = RNG.uniform(lower, upper, size=(n, len(lower))).astype(np.float32)
    idx_ee = robot.links.names.index(ee_link)
    idx_sh = robot.links.names.index(shoulder_link)

    def dist(q):
        fk = robot.forward_kinematics(q)
        p_ee = jaxlie.SE3(fk[idx_ee]).translation()
        p_sh = jaxlie.SE3(fk[idx_sh]).translation()
        return jnp.linalg.norm(p_ee - p_sh)

    d = jax.vmap(dist)(jnp.asarray(qs))
    return float(jnp.max(d))


def pose_error(robot, q, indices, tgt_pos, tgt_wxyz):
    fk = robot.forward_kinematics(np.asarray(q, dtype=np.float32))
    perr, oerr = [], []
    for k, idx in enumerate(indices):
        se3 = jaxlie.SE3(fk[idx])
        perr.append(float(np.linalg.norm(np.asarray(se3.translation()) - tgt_pos[k])))
        rel = se3.rotation().inverse() @ jaxlie.SO3(
            np.asarray(tgt_wxyz[k], dtype=np.float32)
        )
        oerr.append(float(np.linalg.norm(np.asarray(rel.log()))))
    return np.asarray(perr), np.asarray(oerr)


def main() -> None:
    robot = rm.load()
    q0 = rm.home_config(robot)
    home_pos, home_wxyz = rm.home_poses(robot)
    indices = rm.target_link_indices(robot)
    lower = np.asarray(robot.joints.lower_limits)
    upper = np.asarray(robot.joints.upper_limits)

    print(f"JAX backend: {jax.default_backend()}")
    reach = max_reach(robot, "left_gripper_base", "left_shoulder_link")
    print(f"max hand reach (shoulder→gripper_base, 20k samples): {reach:.3f} m")
    d_home = np.linalg.norm(home_pos[0] - np.array([0.469, -0.019, 0.099]))
    print(f"hand home reach fraction: {d_home:.3f} m = {100*d_home/reach:.0f}% of max")
    reach_m = max_reach(robot, "middle_camera_cover", "middle_shoulder_link")
    d_home_m = np.linalg.norm(home_pos[2] - np.array([0.0, 0.400, 0.089]))
    print(f"max camera reach (shoulder→camera_cover): {reach_m:.3f} m; "
          f"home = {d_home_m:.3f} m = {100*d_home_m/reach_m:.0f}% of max\n")

    oracle = BaselineIK(robot, rm.TARGET_LINKS, max_iterations=200)
    oracle.warmup(q0)

    seeds = [q0] + [
        RNG.uniform(lower, upper).astype(np.float32) for _ in range(NUM_RANDOM_SEEDS)
    ]

    suite = T.build_suite(home_pos, home_wxyz)
    every = int(round(SAMPLE_EVERY_S / T.DT))
    report = {}
    print(f"{'trajectory':18s} {'wpts':>4s}  {'hands: max-min-err':>22s}  "
          f"{'reach%':>6s}  {'middle: max-min-err':>22s}  flagged-as")
    for traj in suite:
        sample_idx = list(range(0, len(traj), every))
        if (len(traj) - 1) not in sample_idx:
            sample_idx.append(len(traj) - 1)
        worst_hand_p, worst_hand_o = 0.0, 0.0
        worst_mid_p, worst_mid_o = 0.0, 0.0
        n_reach = 0
        for i in sample_idx:
            tgt_p, tgt_w = traj.positions[i], traj.wxyzs[i]
            best_p = np.full(3, np.inf)
            best_o = np.full(3, np.inf)
            for seed in seeds:
                res = oracle.solve(seed, tgt_p, tgt_w)
                p, o = pose_error(robot, res.q, indices, tgt_p, tgt_w)
                improve = p + 0.2 * o < best_p + 0.2 * best_o
                best_p = np.where(improve, p, best_p)
                best_o = np.where(improve, o, best_o)
            hp, ho = best_p[:2].max(), best_o[:2].max()
            worst_hand_p, worst_hand_o = max(worst_hand_p, hp), max(worst_hand_o, ho)
            worst_mid_p = max(worst_mid_p, best_p[2])
            worst_mid_o = max(worst_mid_o, best_o[2])
            if hp < POS_TOL and ho < ORI_TOL:
                n_reach += 1
        frac = 100.0 * n_reach / len(sample_idx)
        print(
            f"{traj.name:18s} {len(sample_idx):4d}  "
            f"{worst_hand_p*1e3:8.1f} mm {np.degrees(worst_hand_o):6.1f}°  "
            f"{frac:5.0f}%  "
            f"{worst_mid_p*1e3:8.1f} mm {np.degrees(worst_mid_o):6.1f}°  "
            f"{traj.feasible}"
        )
        report[traj.name] = dict(
            waypoints=len(sample_idx),
            reachable_frac=frac / 100.0,
            worst_hand_pos_mm=worst_hand_p * 1e3,
            worst_hand_ori_deg=float(np.degrees(worst_hand_o)),
            worst_middle_pos_mm=worst_mid_p * 1e3,
            worst_middle_ori_deg=float(np.degrees(worst_mid_o)),
            declared=traj.feasible,
        )

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "validation.json").write_text(json.dumps(
        {"max_reach_m": reach, "trajectories": report}, indent=2))
    print(f"\nwrote {out / 'validation.json'}")


if __name__ == "__main__":
    main()
