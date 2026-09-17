"""Collision-study Phase 8 final evaluation — collision-subset battery.

Rolls a set of solver variants over the three-trajectory collision subset
and reports the phase-9-style metrics (tracking, sphere clearance, deviation
from baseline, timing).  Complements the full frozen-suite runs done via
run_study.py (which keep suite-hash comparability).

    JAX_PLATFORMS=cpu python phase8_final_eval.py \
        --variants baseline collision_sphere collision_capsule smooth_center \
                   smooth_center_collisionS

Writes results/collision_subset_eval.csv
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jaxlie
import jax.numpy as jnp

import robot_model as rm
from baseline import BaselineIK
from collision_models import pruned_sphere_collision
from collision_trajectories import build_collision_subset

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    args = ap.parse_args()

    robot, urdf = rm.load(with_urdf=True)
    q0 = rm.home_config(robot)
    home_pos, home_wxyz = rm.home_poses(robot)
    trajs = build_collision_subset(home_pos, home_wxyz)
    rc = pruned_sphere_collision(urdf, HERE / "results")
    idx = rm.target_link_indices(robot)

    @jax.jit
    def clearance_batch(qb):
        return jax.vmap(
            lambda q: rc.compute_self_collision_distance(robot, q).min())(qb)

    @jax.jit
    def ee_batch(qb):
        def one(q):
            fk = robot.forward_kinematics(q)
            return jaxlie.SE3(fk[jnp.asarray(idx[:2])]).translation()
        return jax.vmap(one)(qb)

    def rollout(ik, traj):
        q = q0.copy()
        qs, tms, its = [], [], []
        for i in range(len(traj)):
            r = ik.solve(q, traj.positions[i], traj.wxyzs[i])
            q = r.q
            qs.append(q); tms.append(r.solve_ms); its.append(r.iterations)
        return np.stack(qs), np.asarray(tms), np.asarray(its)

    def build(name):
        if name == "baseline":
            ik = BaselineIK(robot, rm.TARGET_LINKS)
            return ik
        from variants import make

        return make(name, robot)[0]

    base_ik = build("baseline")
    base_ik.warmup(q0)
    base_ee = {}
    for traj in trajs:
        qs, *_ = rollout(base_ik, traj)
        base_ee[traj.name] = np.asarray(
            ee_batch(jnp.asarray(qs.astype(np.float32))))

    rows = []
    for vname in args.variants:
        ik = build(vname)
        ik.warmup(q0)
        for traj in trajs:
            qs, tms, its = rollout(ik, traj)
            qsj = jnp.asarray(qs.astype(np.float32))
            clear = np.asarray(clearance_batch(qsj))
            ee = np.asarray(ee_batch(qsj))
            perr = np.linalg.norm(ee - traj.positions[:, :2], axis=-1).max(axis=1)
            dev = np.linalg.norm(ee - base_ee[traj.name], axis=-1).max(axis=1)
            dq = np.abs(np.diff(qs, axis=0)).max(axis=1)
            row = dict(
                variant=vname, trajectory=traj.name,
                pos_p95_mm=float(np.percentile(perr, 95) * 1e3),
                pos_max_mm=float(perr.max() * 1e3),
                clear_min_mm=float(clear.min() * 1e3),
                viol_frac=float((clear < 0).mean()),
                dev_rms_mm=float(np.sqrt((dev**2).mean()) * 1e3),
                dev_max_mm=float(dev.max() * 1e3),
                solve_ms_med=float(np.median(tms)),
                solve_ms_p95=float(np.percentile(tms, 95)),
                solve_ms_max=float(tms.max()),
                iters_mean=float(np.mean(its)),
                jump_max_deg=float(np.degrees(dq.max())),
            )
            rows.append(row)
            print(f"{vname:26s} {traj.name:18s} pos p95 {row['pos_p95_mm']:7.1f} "
                  f"clear {row['clear_min_mm']:7.1f} dev {row['dev_max_mm']:6.1f} "
                  f"med {row['solve_ms_med']:5.1f} ms", flush=True)

    out = HERE / "results" / "collision_subset_eval.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
