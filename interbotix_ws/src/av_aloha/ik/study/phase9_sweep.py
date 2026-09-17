"""Collision-study Phase 9 — margin × weight sweep (soft) + AL reference.

Trajectory set (contact-relevant): bimanual_parallel, bimanual_handoff
(false-repulsion testbeds — correct behaviour is zero deviation),
arms_converge, self_fold (designed contact — the model must hold the
boundary), teleop_grasp (hands pass the camera).

Per (margin, weight) and per trajectory:
  pos_p95_mm      hand tracking error (the repulsion cost)
  clear_min_mm    minimum SPHERE-model clearance (calibration: mesh ≈ model
                  − 8 ± 10 mm, Part 2)
  viol_frac       fraction of steps with model clearance < 0
  dev_rms/max_mm  deviation of the EE path from the *baseline* solution of
                  the same trajectory (unnecessary-motion metric; computed
                  on feasible steps)
  solve stats, iterations, nonconverged

Soft grid: margins {15,20,25,30,40} mm × weights {10,30,100,300}.
AL: margins {10, 25} mm (hard standoff at the margin — expected to fail
bimanual; measured to prove it).

Run (background): JAX_PLATFORMS=cpu python phase9_sweep.py
Writes results/sweeps/collision_margin_weight.csv
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jaxlie
import jax.numpy as jnp

import robot_model as rm
import trajectories as T
from baseline import BaselineIK
from collision_models import pruned_sphere_collision
from collision_trajectories import build_collision_subset
from variants import CollisionIK

HERE = Path(__file__).resolve().parent
MARGINS = (0.015, 0.020, 0.025, 0.030, 0.040)
WEIGHTS = (10.0, 30.0, 100.0, 300.0)
AL_MARGINS = (0.010, 0.025)


def main() -> None:
    robot, urdf = rm.load(with_urdf=True)
    q0 = rm.home_config(robot)
    home_pos, home_wxyz = rm.home_poses(robot)
    suite = {t.name: t for t in T.build_suite(home_pos, home_wxyz)}
    trajs = build_collision_subset(home_pos, home_wxyz) + [
        suite["arms_converge"], suite["self_fold"], suite["teleop_grasp"],
    ]
    rc = pruned_sphere_collision(urdf, HERE / "results")
    idx = rm.target_link_indices(robot)

    @jax.jit
    def clearance_batch(qb):
        return jax.vmap(
            lambda q: rc.compute_self_collision_distance(robot, q).min()
        )(qb)

    @jax.jit
    def ee_batch(qb):
        def one(q):
            fk = robot.forward_kinematics(q)
            return jaxlie.SE3(fk[jnp.asarray(idx[:2])]).translation()
        return jax.vmap(one)(qb)

    def rollout(ik, traj, **kw):
        q = q0.copy()
        qs, tms, its, terms = [], [], [], []
        for i in range(len(traj)):
            r = ik.solve(q, traj.positions[i], traj.wxyzs[i], **kw)
            q = r.q
            qs.append(q); tms.append(r.solve_ms)
            its.append(r.iterations); terms.append(r.termination)
        return np.stack(qs), np.asarray(tms), np.asarray(its), np.stack(terms)

    # baseline reference rollouts (deviation metric)
    base_ik = BaselineIK(robot, rm.TARGET_LINKS)
    base_ik.warmup(q0)
    base_ee = {}
    base_qs = {}
    for traj in trajs:
        qs, *_ = rollout(base_ik, traj)
        base_qs[traj.name] = qs
        base_ee[traj.name] = np.asarray(ee_batch(jnp.asarray(qs.astype(np.float32))))

    def evaluate(traj, qs, tms, its, terms):
        qsj = jnp.asarray(qs.astype(np.float32))
        clear = np.asarray(clearance_batch(qsj))
        ee = np.asarray(ee_batch(qsj))
        # hand tracking error vs commanded
        tgt = traj.positions[:, :2]
        perr = np.linalg.norm(ee - tgt, axis=-1).max(axis=1)
        dev = np.linalg.norm(ee - base_ee[traj.name], axis=-1).max(axis=1)
        return dict(
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
            nonconv=float(((~terms.any(axis=1)) & (its >= 100)).mean()),
        )

    rows = []
    ik = CollisionIK(robot, rm.TARGET_LINKS, rc, formulation="soft")
    ik.warmup(q0)
    t0 = time.perf_counter()
    for m in MARGINS:
        for w in WEIGHTS:
            for traj in trajs:
                qs, tms, its, terms = rollout(ik, traj, margin=m, weight=w)
                r = dict(form="soft", margin=m, weight=w, trajectory=traj.name)
                r.update(evaluate(traj, qs, tms, its, terms))
                rows.append(r)
                print(f"soft m={m*1e3:.0f} w={w:<5g} {traj.name:18s} "
                      f"pos p95 {r['pos_p95_mm']:7.1f}  clear "
                      f"{r['clear_min_mm']:7.1f}  dev {r['dev_max_mm']:6.1f}  "
                      f"med {r['solve_ms_med']:5.1f} ms", flush=True)

    ik_al = CollisionIK(robot, rm.TARGET_LINKS, rc, formulation="al")
    ik_al.warmup(q0)
    for m in AL_MARGINS:
        for traj in trajs:
            qs, tms, its, terms = rollout(ik_al, traj, margin=m)
            r = dict(form="al", margin=m, weight="", trajectory=traj.name)
            r.update(evaluate(traj, qs, tms, its, terms))
            rows.append(r)
            print(f"al   m={m*1e3:.0f}        {traj.name:18s} "
                  f"pos p95 {r['pos_p95_mm']:7.1f}  clear "
                  f"{r['clear_min_mm']:7.1f}  dev {r['dev_max_mm']:6.1f}  "
                  f"med {r['solve_ms_med']:5.1f} ms", flush=True)

    out = HERE / "results" / "sweeps" / "collision_margin_weight.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\n{time.perf_counter()-t0:.0f} s; wrote {out}")


if __name__ == "__main__":
    main()
