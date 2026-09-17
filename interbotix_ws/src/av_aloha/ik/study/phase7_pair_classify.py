"""Collision-study Phase 7 — sphere-pair classification and pruning.

Rules at margin m = 25 mm with a 25 mm activation buffer, over the
20,350-config corpus (full frozen-suite baseline rollout + 3k random):

  functional  link pairs whose contact is intended:
              same-gripper finger↔finger (grasping)          → prune
  permanent   sphere pairs with corpus MAX distance < m:
              can never satisfy the margin → zero signal,
              constant bias                                   → prune
  inactive    sphere pairs with corpus MIN distance > m+25 mm:
              never near the activation band                  → prune (speed)
  kept        everything else                                 → the model

Output: results/sphere_pair_pruning.npz (kept indices + stats) and a
summary of which link pairs survive / die and why.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp

import robot_model as rm
from collision_models import sphere_collision

HERE = Path(__file__).resolve().parent
MARGIN = 0.025
BUFFER = 0.025

FUNCTIONAL_LINK_PAIRS = {
    frozenset(("left_left_finger_link", "left_right_finger_link")),
    frozenset(("right_left_finger_link", "right_right_finger_link")),
}


def main() -> None:
    robot, urdf = rm.load(with_urdf=True)
    rc = sphere_collision(urdf, HERE / "results" / "sphere_decomposition.json")
    P = len(rc.active_idx_i)
    g2l = np.asarray(rc._geom_to_link_idx)
    pair_links = [
        (rc.link_names[g2l[i]], rc.link_names[g2l[j]])
        for i, j in zip(rc.active_idx_i, rc.active_idx_j)
    ]
    print(f"sphere pairs (adjacency-filtered): {P}")

    qs = [np.load(f)["q"] for f in sorted(glob.glob(str(HERE / "results/baseline/*.npz")))]
    suite_q = np.concatenate(qs).astype(np.float32)
    rng = np.random.default_rng(7)
    lower = np.asarray(robot.joints.lower_limits)
    upper = np.asarray(robot.joints.upper_limits)
    rand_q = rng.uniform(lower, upper, size=(3000, len(lower))).astype(np.float32)
    allq = np.concatenate([suite_q, rand_q])

    @jax.jit
    def dists(qb):
        return jax.vmap(
            lambda q: rc.compute_self_collision_distance(robot, q).reshape(-1)
        )(qb)

    d_min = np.full(P, np.inf)
    d_max = np.full(P, -np.inf)
    for s in range(0, len(allq), 1000):
        d = np.asarray(dists(jnp.asarray(allq[s:s + 1000])))
        d_min = np.minimum(d_min, d.min(axis=0))
        d_max = np.maximum(d_max, d.max(axis=0))

    functional = np.asarray(
        [frozenset(pl) in FUNCTIONAL_LINK_PAIRS for pl in pair_links]
    )
    permanent = (d_max < MARGIN) & ~functional
    inactive = (d_min > MARGIN + BUFFER) & ~functional & ~permanent
    kept = ~(functional | permanent | inactive)

    print(f"functional (grasping)     : {functional.sum():6d}")
    print(f"permanent  (max < {MARGIN*1e3:.0f} mm) : {permanent.sum():6d}")
    print(f"inactive   (min > {(MARGIN+BUFFER)*1e3:.0f} mm) : {inactive.sum():6d}")
    print(f"kept                      : {kept.sum():6d}")

    # link-pair level narrative
    from collections import defaultdict

    stats = defaultdict(lambda: [0, 0, 0, 0])  # func, perm, inact, kept
    for k, pl in enumerate(pair_links):
        key = tuple(sorted(pl))
        idx = 0 if functional[k] else 1 if permanent[k] else 2 if inactive[k] else 3
        stats[key][idx] += 1
    fully_pruned = [k for k, v in stats.items() if v[3] == 0]
    print(f"\nlink pairs fully pruned: {len(fully_pruned)} "
          f"(of {len(stats)})")
    perm_dominated = [k for k in fully_pruned
                      if stats[k][1] > 0 and stats[k][0] == 0 and stats[k][2] == 0]
    print("fully-permanent link pairs (no sphere pair can exit the margin):")
    for k in perm_dominated:
        print(f"   {k}")

    kept_idx_i = tuple(int(rc.active_idx_i[k]) for k in range(P) if kept[k])
    kept_idx_j = tuple(int(rc.active_idx_j[k]) for k in range(P) if kept[k])
    np.savez(
        HERE / "results" / "sphere_pair_pruning.npz",
        kept=kept, functional=functional, permanent=permanent,
        inactive=inactive, d_min=d_min, d_max=d_max,
        kept_idx_i=np.asarray(kept_idx_i, dtype=np.int32),
        kept_idx_j=np.asarray(kept_idx_j, dtype=np.int32),
        margin=MARGIN, buffer=BUFFER,
    )
    print(f"\nwrote results/sphere_pair_pruning.npz "
          f"({kept.sum()} kept sphere pairs)")


if __name__ == "__main__":
    main()
