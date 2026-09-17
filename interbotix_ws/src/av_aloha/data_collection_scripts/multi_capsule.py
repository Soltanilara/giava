"""Tight multi-capsule link models for the fine tier of the collision gate.

WHY A SECOND MODEL EXISTS AT ALL
================================
One circumscribed capsule per link is cheap and provably safe, but it is
FAT: measured against the real meshes it pads the true surface by 28 mm on
average and **61 mm on the gripper bases** -- a plate-like link wrapped in
a cylinder wastes the whole corner volume.  That padding is not a rounding
error, it is exactly the clearance the operator loses.  Two grippers that
could physically pass 10 mm apart are stopped 130 mm apart.

Decomposing each link into convex pieces (VHACD) and fitting a tight
capsule to each brings that padding down to 8.5 mm median, 6.9 mm on the
gripper bases -- 3.3x tighter overall and 9x where it matters most.

WHY IT IS THE *FINE* TIER AND NOT THE ONLY MODEL
================================================
Cost.  Eight pieces per link turns one link-pair test into 64, and the
gate's 300 link pairs into ~19,200.  Swept over a step that is ~32 ms
against a 20 ms tick.  The coarse single-capsule model rejects almost
every pair almost instantly, so the expensive model only ever runs on the
one or two pairs that are actually close -- which is where its precision
is worth paying for.

CONSERVATISM IS PRESERVED, AND IS NOT FREE
==========================================
The guarantee rests on: model distance <= true mesh distance, always.
VHACD pieces only APPROXIMATE the mesh, so a per-piece capsule can leave
part of the original surface uncovered -- measured up to 2.04 mm on
`right_gripper_base`.  That would silently break the guarantee.

So every link's radii are inflated by its own densely-measured worst leak
plus a safety factor, and containment is then re-verified against a fresh,
denser point set than the one that produced the correction.  A link that
still leaks after correction is rejected outright rather than shipped: the
build fails loudly instead of producing a model that is quietly unsafe.

The inflation is why the fine tier reads ~9 mm rather than ~7 mm of
padding.  Still far tighter than the 28-61 mm it replaces.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent / "ik")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE = _HERE.parent / "ik" / "study" / "results" / "multi_capsule_decomposition.json"

## VHACD settings.  8 pieces is where the tightness gain flattens on these
## links; more pieces cost pair-count quadratically for little benefit.
MAX_PIECES = 8
VHACD_RESOLUTION = 50_000
## Safety factor on the measured leak.  The leak is estimated from a finite
## point sample, so a bare correction would be exact only for the points
## that were looked at.
LEAK_SAFETY = 1.5
LEAK_FLOOR = 2.0e-4          # 0.2 mm, absorbs sampling and float noise
FIT_SAMPLES = 4_000          # points used to FIT the correction
VERIFY_SAMPLES = 20_000      # denser, independent points used to CHECK it


def capsule_point_distance(center, axis, radius, height, pts) -> np.ndarray:
    """Signed distance from points to one capsule (segment swept by sphere)."""
    v = np.asarray(pts, dtype=float) - np.asarray(center, dtype=float)
    ax = np.asarray(axis, dtype=float)
    t = np.clip(v @ ax, -height / 2.0, height / 2.0)
    return np.linalg.norm(v - t[:, None] * ax[None, :], axis=1) - radius


def union_distance(caps, pts) -> np.ndarray:
    """Distance to the UNION of capsules: the min over members."""
    return np.min(np.stack([capsule_point_distance(*c, pts) for c in caps]),
                  axis=0)


def decompose_link(mesh, max_pieces: int = MAX_PIECES
                   ) -> Tuple[List[Tuple], Dict]:
    """Convex-decompose one link mesh and fit a corrected capsule to each."""
    from collision_models import fit_tight_capsule

    import trimesh

    try:
        parts = mesh.convex_decomposition(maxConvexHulls=max_pieces,
                                          resolution=VHACD_RESOLUTION)
        if isinstance(parts, trimesh.Trimesh):
            parts = [parts]
        parts = [p for p in parts if not p.is_empty]
    except Exception as e:  # noqa: BLE001
        ## No VHACD -> fall back to the convex hull, which is a single
        ## convex piece and still strictly contains the mesh.  Less tight,
        ## never unsafe.
        parts = [mesh.convex_hull]
        print(f"    convex decomposition unavailable ({e}); using the hull")
    if not parts:
        parts = [mesh.convex_hull]

    caps = [list(fit_tight_capsule(p)) for p in parts]

    ## Measure the leak against the ORIGINAL mesh, not the pieces.
    fit_pts = np.vstack([np.asarray(mesh.vertices), mesh.sample(FIT_SAMPLES)])
    leak = float(max(union_distance(caps, fit_pts).max(), 0.0))
    inflate = max(leak * LEAK_SAFETY, LEAK_FLOOR)
    for c in caps:
        c[2] = float(c[2]) + inflate

    ## Re-verify on an independent, denser sample.
    ver_pts = np.vstack([np.asarray(mesh.vertices),
                         mesh.sample(VERIFY_SAMPLES)])
    residual = float(union_distance(caps, ver_pts).max())
    padding = float(-np.median(union_distance(caps, ver_pts)))
    return caps, {
        "hulls": [np.asarray(p.vertices, dtype=float) for p in parts],
        "leak_m": float(leak),
        "n_pieces": len(parts),
        "measured_leak_mm": leak * 1e3,
        "inflation_mm": inflate * 1e3,
        "residual_leak_mm": residual * 1e3,
        "median_padding_mm": padding * 1e3,
    }


def build(urdf, cache: Optional[Path] = CACHE, force: bool = False) -> Dict:
    """Decompose every link, verify containment, cache the result."""
    from pyroki.collision._robot_collision import RobotCollision

    cache = Path(cache) if cache else None
    if cache and cache.exists() and not force:
        raw = json.loads(cache.read_text())
        hulls = {}
        hf = cache.with_suffix(".hulls.npz")
        if hf.exists():
            z = np.load(hf, allow_pickle=False)
            verts = np.asarray(z["verts"])
            idx = json.loads(str(z["index"]))
            hulls = {k: [verts[a:b] for a, b in spans]
                     for k, spans in idx.items()}
        return {k: {"caps": [tuple(np.asarray(x, dtype=float) if i < 2
                                   else float(x) for i, x in enumerate(c))
                             for c in v["caps"]],
                    "hulls": hulls.get(k, []),
                    "stats": v["stats"]}
                for k, v in raw["links"].items()}

    out, failed = {}, []
    t0 = time.time()
    names = list(urdf.link_map)
    print(f"  decomposing {len(names)} links (VHACD, one-off, cached)...")
    for name in names:
        mesh = RobotCollision._get_trimesh_collision_geometries(urdf, name)
        if mesh.is_empty:
            continue
        caps, stats = decompose_link(mesh)
        ## A link whose correction did not take is NOT shipped.
        if stats["residual_leak_mm"] > 1e-3:
            failed.append((name, stats["residual_leak_mm"]))
        hulls = stats.pop("hulls")
        out[name] = {"caps": [tuple(c) for c in caps], "hulls": hulls,
                     "stats": stats}
        print(f"    {name:<28} {stats['n_pieces']} pieces  "
              f"leak {stats['measured_leak_mm']:5.2f} -> "
              f"{stats['residual_leak_mm']:+.4f} mm  "
              f"padding {stats['median_padding_mm']:5.1f} mm")

    if failed:
        raise RuntimeError(
            "these links still leak after leak correction, so the fine "
            "model would NOT be conservative and is refused:\n  "
            + "\n  ".join(f"{n}: {v:+.4f} mm" for n, v in failed))

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        flat, index = [], {}
        for k, v in out.items():
            index[k] = []
            for h in v["hulls"]:
                index[k].append((len(flat), len(flat) + len(h)))
                flat.extend(np.asarray(h, dtype=float))
        np.savez_compressed(cache.with_suffix(".hulls.npz"),
                            verts=np.asarray(flat, dtype=float),
                            index=json.dumps(index))
        cache.write_text(json.dumps({
            "_README": (
                "Multi-capsule link decomposition for the collision gate's "
                "FINE tier. Radii already include the per-link leak "
                "correction that makes the union contain the mesh; do not "
                "shrink them. Rebuild with multi_capsule.py --rebuild."),
            "params": {"max_pieces": MAX_PIECES,
                       "vhacd_resolution": VHACD_RESOLUTION,
                       "leak_safety": LEAK_SAFETY,
                       "leak_floor_mm": LEAK_FLOOR * 1e3},
            "hull_file": cache.with_suffix(".hulls.npz").name,
            "links": {k: {"caps": [[list(np.asarray(c[0])),
                                    list(np.asarray(c[1])),
                                    float(c[2]), float(c[3])]
                                   for c in v["caps"]],
                          "stats": v["stats"]} for k, v in out.items()},
        }, indent=1))
        print(f"  cached -> {cache}")
    print(f"  built in {time.time() - t0:.1f} s")
    return out


def pair_distance(caps_a, T_a, caps_b, T_b) -> float:
    """Min distance between two links' capsule unions, in world frame.

    Vectorised over the k*m piece pairs in numpy -- no jax.  The fine tier
    runs on one or two link pairs per tick, so the batched-on-device
    machinery the coarse tier needs would cost more in dispatch than it
    saves in arithmetic."""
    from pyroki.collision import _utils

    import jax.numpy as jnp

    def segs(caps, T):
        R, t = np.asarray(T)[:3, :3], np.asarray(T)[:3, 3]
        c = np.stack([R @ np.asarray(x[0]) + t for x in caps])
        a = np.stack([R @ np.asarray(x[1]) for x in caps])
        r = np.array([float(x[2]) for x in caps])
        h = np.array([float(x[3]) for x in caps])
        return c - 0.5 * h[:, None] * a, c + 0.5 * h[:, None] * a, r

    a0, a1, ra = segs(caps_a, T_a)
    b0, b1, rb = segs(caps_b, T_b)
    n, m = len(ra), len(rb)
    A0 = np.repeat(a0, m, axis=0); A1 = np.repeat(a1, m, axis=0)
    B0 = np.tile(b0, (n, 1));      B1 = np.tile(b1, (n, 1))
    p, q = _utils.closest_segment_to_segment_points(
        jnp.asarray(A0), jnp.asarray(A1), jnp.asarray(B0), jnp.asarray(B1))
    d = np.linalg.norm(np.asarray(p) - np.asarray(q), axis=1)
    return float((d - np.repeat(ra, m) - np.tile(rb, n)).min())


def main() -> None:
    import argparse

    import robot_model as rm

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    _, urdf = rm.load(with_urdf=True)
    d = build(urdf, force=args.rebuild)
    pads = [v["stats"]["median_padding_mm"] for v in d.values()]
    print(f"\n  {len(d)} links, median padding {np.median(pads):.1f} mm "
          f"(single-capsule was ~28 mm)")


if __name__ == "__main__":
    main()
