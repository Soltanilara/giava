"""Collision-study Phase 6 — sphere-count Pareto.

Reduction operator: greedy pairwise merge.  Merging spheres (c1,r1),(c2,r2)
yields their minimal bounding sphere:
    d = ‖c2−c1‖
    if one contains the other: keep the container
    else r = (d + r1 + r2)/2,  c = c1 + (c2−c1)·(r − r1)/d
The bound contains both parents, so mesh coverage never decreases — only
conservatism (over-approximation) can grow.  We merge the *closest* pair
first (least added volume), per link, until the link hits its target count.

Variants scored against the 70 mesh-ground-truth cases:
    full-180     — the Phase-5 decomposition as fitted
    reduced-~140 — forks 20→10, bases 12→8, upper_forearm 8→6
    minimal-~100 — forks →6, bases →6, most links →⌈n/2⌉

Metrics per variant: mean/max |clearance error|, signed bias, FP collisions,
true contacts inside the 25 mm band, stolen-workspace max, sphere-pair count.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jaxlie
import jax.numpy as jnp

import robot_model as rm

HERE = Path(__file__).resolve().parent
NEAR_BAND = 0.025


def merge_two(c1, r1, c2, r2):
    d = float(np.linalg.norm(c2 - c1))
    if d + r2 <= r1:
        return c1, r1
    if d + r1 <= r2:
        return c2, r2
    r = 0.5 * (d + r1 + r2)
    c = c1 + (c2 - c1) * ((r - r1) / d)
    return c, r


def reduce_link(centers, radii, target):
    centers = [np.asarray(c, dtype=float) for c in centers]
    radii = [float(r) for r in radii]
    while len(centers) > target:
        # closest pair (surface distance) → least added volume
        best, bi, bj = np.inf, 0, 1
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = np.linalg.norm(centers[i] - centers[j]) - radii[i] - radii[j]
                if d < best:
                    best, bi, bj = d, i, j
        c, r = merge_two(centers[bi], radii[bi], centers[bj], radii[bj])
        for k in sorted((bi, bj), reverse=True):
            centers.pop(k)
            radii.pop(k)
        centers.append(c)
        radii.append(r)
    return centers, radii


def make_variant(decomp, targets):
    out = {}
    for link, d in decomp.items():
        c, r = d["centers"], d["radii"]
        t = targets.get(link, len(c))
        if t < len(c):
            c, r = reduce_link(c, r, t)
        out[link] = {"centers": [list(map(float, x)) for x in c],
                     "radii": [float(x) for x in r]}
    return out


def sphere_pair_count(decomp, link_pairs):
    counts = {k: len(v["radii"]) for k, v in decomp.items()}
    return sum(counts.get(a, 0) * counts.get(b, 0) for a, b in link_pairs)


def main() -> None:
    robot, urdf = rm.load(with_urdf=True)
    decomp = json.loads((HERE / "results" / "sphere_decomposition.json").read_text())
    from pyroki.collision import RobotCollision

    rc = RobotCollision.from_urdf(urdf)
    name_to_idx = {n: i for i, n in enumerate(rc.link_names)}
    link_pairs = [(rc.link_names[i], rc.link_names[j])
                  for i, j in zip(rc.active_idx_i, rc.active_idx_j)]

    def targets_scaled(fork, base, half_all=False):
        t = {}
        for link, d in decomp.items():
            n = len(d["radii"])
            if "gripper_base" in link:
                t[link] = fork
            elif link.endswith("base_link"):
                t[link] = base
            elif "upper_forearm" in link and n > 6:
                t[link] = 6
            elif half_all and n > 3:
                t[link] = max(3, int(np.ceil(n / 2)))
        return t

    variants = {
        "full": decomp,
        "reduced": make_variant(decomp, targets_scaled(fork=10, base=8)),
        "minimal": make_variant(decomp, targets_scaled(fork=6, base=6, half_all=True)),
    }

    # ground truth cases
    rows = list(csv.DictReader(open(HERE / "results" / "mesh_ground_truth.csv")))
    from mesh_ground_truth import CASES

    q0 = rm.home_config(robot)
    configs = {}
    for name, src in CASES.items():
        if src is None:
            configs[name] = q0
        else:
            q = np.load(HERE / "results" / "baseline" / src[0])["q"]
            configs[name] = q[min(src[1], len(q) - 1)]

    def sphere_dist(dec, q, a, b):
        fk = np.asarray(robot.forward_kinematics(np.asarray(q, dtype=np.float32)))
        Ta, Tb = (jaxlie.SE3(jnp.asarray(fk[name_to_idx[x]])) for x in (a, b))
        ca = np.asarray(dec[a]["centers"]); ra = np.asarray(dec[a]["radii"])
        cb = np.asarray(dec[b]["centers"]); rb = np.asarray(dec[b]["radii"])
        wa = ca @ np.asarray(Ta.rotation().as_matrix()).T + np.asarray(Ta.translation())
        wb = cb @ np.asarray(Tb.rotation().as_matrix()).T + np.asarray(Tb.translation())
        d = (np.linalg.norm(wa[:, None] - wb[None], axis=-1)
             - ra[:, None] - rb[None, :])
        return float(d.min())

    print(f"{'variant':10s} {'spheres':>7s} {'pairs':>7s} {'mean|e|':>8s} "
          f"{'max|e|':>7s} {'bias':>7s} {'FP':>3s} {'in-band':>7s} "
          f"{'stolenMax':>9s} {'worstContactRead':>16s}")
    for vname, dec in variants.items():
        n_spheres = sum(len(d["radii"]) for d in dec.values())
        n_pairs = sphere_pair_count(dec, link_pairs)
        errs, fp, inband, stolen, contact_reads = [], 0, 0, [], []
        n_contacts = 0
        for r in rows:
            a, b = r["pair"].split("|")
            truth = float(r["mesh_mm"]) / 1e3
            model = sphere_dist(dec, configs[r["config"]], a, b)
            errs.append(model - truth)
            if truth >= 0 and model < 0:
                fp += 1
            if truth >= 0:
                stolen.append(max(0.0, truth - model))
            if truth < 0:
                n_contacts += 1
                contact_reads.append(model)
                if model < NEAR_BAND:
                    inband += 1
        e = np.asarray(errs) * 1e3
        print(f"{vname:10s} {n_spheres:7d} {n_pairs:7d} {np.abs(e).mean():8.1f} "
              f"{np.abs(e).max():7.1f} {e.mean():+7.1f} {fp:3d} "
              f"{inband}/{n_contacts:<5d} {max(stolen)*1e3:9.1f} "
              f"{max(contact_reads)*1e3:16.1f}")
        (HERE / "results" / f"sphere_decomposition_{vname}.json").write_text(
            json.dumps(dec, indent=1))
    print("\nwrote sphere_decomposition_{full,reduced,minimal}.json")


if __name__ == "__main__":
    main()
