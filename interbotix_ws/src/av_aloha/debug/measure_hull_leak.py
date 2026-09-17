"""Measure how far each link's true mesh pokes OUTSIDE its convex hulls.

WHY THIS SCRIPT EXISTS
=======================
`capsule_gate`'s fine tier runs exact GJK on the VHACD convex hulls, and
compensates for the decomposition being an approximation by demanding extra
separation:

    need = margin + leak_i + leak_j            # capsule_gate._gjk_pairs

That `leak` comes from `multi_capsule.decompose_link`, where it is measured
as the distance from the mesh to the union of the CAPSULES:

    caps = [fit_tight_capsule(p) for p in parts]
    leak = max(union_distance(caps, pts).max(), 0.0)      # capsules, not hulls

A capsule fitted to a convex piece CONTAINS that piece, so

    union(capsules)  ⊇  union(hulls)
    => dist(point, union(capsules))  <=  dist(point, union(hulls))
    => capsule_leak  <=  hull_leak

The number being used is therefore a LOWER bound on the error it is meant to
cover, and the gate allows less slack for hull error than the hulls need.
The direction matters: for a safety gate, erring permissive is the one
direction that is not acceptable.

This script measures the real thing -- max distance from the mesh surface to
the union of that link's hulls -- so the fix can be made against a number
rather than an argument.

METHOD
======
Exact, not sampled-and-hoped:
  * points = every mesh vertex PLUS a dense area-weighted surface sample.
    Vertices alone miss leaks through the middle of a large triangle;
    surface samples alone miss sharp corners, which is exactly where a
    convex decomposition cuts.
  * distance from a point to one convex hull is computed with trimesh's
    exact closest-point query (0 when inside).
  * distance to the UNION is the min over hulls; the leak is the max over
    points of that.

Run:  python measure_hull_leak.py [--samples 20000]
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent / "ik")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE = _HERE.parent / "ik" / "study" / "results" / "multi_capsule_decomposition.json"
URDF_PATH = "/home/devi/giava/giava.urdf"


def hull_union_distance(hulls, pts):
    """Distance from each point to the union of convex hulls (0 = inside)."""
    import trimesh
    best = np.full(len(pts), np.inf)
    for h in hulls:
        m = trimesh.Trimesh(vertices=np.asarray(h, dtype=float), process=False)
        m = m.convex_hull
        _, dist, _ = trimesh.proximity.closest_point(m, pts)
        inside = m.contains(pts)
        d = np.where(inside, 0.0, dist)
        best = np.minimum(best, d)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=20_000,
                    help="surface sample points per link (default 20000)")
    ap.add_argument("--out", type=Path,
                    default=_HERE.parent / "ik" / "study" / "results" / "hull_leak.json")
    args = ap.parse_args()

    import yourdfpy
    from pyroki.collision._robot_collision import RobotCollision
    import multi_capsule as mc

    urdf = yourdfpy.URDF.load(URDF_PATH)
    fine = mc.build(urdf)

    print(f"{'link':<30}{'hulls':>6}{'cap leak':>10}{'HULL leak':>11}"
          f"{'ratio':>8}")
    print("-" * 65)
    rows = {}
    for name, entry in sorted(fine.items()):
        hulls = entry.get("hulls", [])
        if not hulls:
            continue
        mesh = RobotCollision._get_trimesh_collision_geometries(urdf, name)
        if mesh.is_empty:
            continue
        pts = np.vstack([np.asarray(mesh.vertices),
                         mesh.sample(args.samples)])
        d = hull_union_distance(hulls, pts)
        hull_leak = float(max(d.max(), 0.0))
        cap_leak = float(entry["stats"].get("leak_m", 0.0))
        ratio = (hull_leak / cap_leak) if cap_leak > 1e-9 else float("inf")
        rows[name] = {"hull_leak_m": hull_leak, "capsule_leak_m": cap_leak,
                      "n_hulls": len(hulls), "n_points": int(len(pts))}
        print(f"{name:<30}{len(hulls):>6}{cap_leak * 1e3:>9.2f}m"
              f"{hull_leak * 1e3:>10.2f}m{ratio:>8.2f}")

    hl = np.array([r["hull_leak_m"] for r in rows.values()])
    cl = np.array([r["capsule_leak_m"] for r in rows.values()])
    print("-" * 65)
    print(f"{len(rows)} links")
    print(f"  capsule leak (IN USE): median {np.median(cl) * 1e3:.2f}  "
          f"max {cl.max() * 1e3:.2f} mm")
    print(f"  HULL leak    (TRUTH) : median {np.median(hl) * 1e3:.2f}  "
          f"max {hl.max() * 1e3:.2f} mm")
    worse = int((hl > cl + 1e-6).sum())
    print(f"  hull leak exceeds the number in use on {worse}/{len(rows)} links")
    if worse:
        gap = (hl - cl)
        k = int(np.argmax(gap))
        nm = list(rows)[k]
        print(f"  worst under-allowance: {nm} short by "
              f"{gap[k] * 1e3:.2f} mm")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
