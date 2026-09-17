"""Collision-study — corrected collision models for the GIAVA robot.

`tight_capsule_collision(urdf)` builds a RobotCollision whose per-link capsule
axes follow the link's *longest* oriented-bounding-box dimension (with exact
vertex containment), instead of pyroki's minimum-bounding-cylinder fit which
minimizes cylinder volume and therefore orients plate/finger-like links along
their *thin* dimension — the caps then add a full radius on both flat sides
(user-diagnosed on the gripper fingers: a 10 cm finger became a ~13 cm-long
fat disc).

This is custom code (flagged); pair logic, distances, and costs remain
pyroki-native and unchanged.

Fit definition (per link, on the URDF collision mesh in link frame):
  axis  â    = longest axis of trimesh.bounding_box_oriented
  center     = midpoint of the vertex span along â
  radius R   = max distance of any vertex from the axis line
  height h   = smallest value such that every vertex satisfies the capsule
               containment condition  |z_i| ≤ h/2 + √(R² − r_i²)
The result contains every mesh vertex by construction (conservative), and is
never *less* tight than pyroki's fit along the two dominant dimensions.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyroki.collision import RobotCollision
from pyroki.collision._geometry import Capsule


def _fit_capsule_with_axis(pts: np.ndarray, axis: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """The containment fit's actual math, axis given rather than chosen.

    Shared by `fit_tight_capsule` (axis = longest OBB dimension) and
    `fit_capsule_along_axis` (axis = whatever the caller supplies) so the
    two can never drift into different containment guarantees -- only
    which axis they hand this function differs."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    center0 = pts.mean(axis=0)
    z = (pts - center0) @ axis
    z_mid = 0.5 * (z.max() + z.min())
    center = center0 + z_mid * axis
    z = z - z_mid

    radial = pts - center - np.outer(z, axis)
    r = np.linalg.norm(radial, axis=1)
    R = float(r.max()) + 1e-4  # containment + hair of numerical padding

    slack = np.sqrt(np.maximum(R**2 - r**2, 0.0))
    h = float(2.0 * np.maximum(np.abs(z) - slack, 0.0).max())
    return center, axis, R, h


## Surface samples MUST be deterministic.  `mesh.sample()` draws from numpy's
## global RNG, so every launch fitted slightly different capsules: the safety
## margins moved run to run, and -- because these capsules are constants
## inside the jitted table-collision cost and the gates -- every launch
## compiled a DIFFERENT program, which made jax's persistent compilation cache
## miss every time (~15 s of recompilation per data_collection startup).
## Seeded here rather than at the call sites so no future caller can forget.
SURFACE_SAMPLE_SEED = 0
SURFACE_SAMPLE_COUNT = 1024


def _surface_samples(mesh: trimesh.Trimesh) -> np.ndarray:
    return trimesh.sample.sample_surface(
        mesh, SURFACE_SAMPLE_COUNT, seed=SURFACE_SAMPLE_SEED)[0]


def fit_tight_capsule(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (center (3,), axis (3,), radius, height) of the containment fit.

    Vertices are augmented with face samples so thin meshes with sparse
    vertices are still covered. Axis is the longest OBB dimension -- for
    most links (roughly a solid of revolution about its long axis) that is
    also the direction a human would call "the link's length", but it is
    only ever the mesh's LARGEST bounding-box extent, not necessarily the
    functionally meaningful axis. See `fit_capsule_along_axis` for links
    where those two disagree (giava's custom finger: its bounding box is
    longest along the open/close SLIDE direction, not the reach direction
    the finger actually points a capsule's readers would expect)."""
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(mesh.faces) > 0:
        pts = np.vstack([pts, _surface_samples(mesh)])

    obb = mesh.bounding_box_oriented
    T = np.asarray(obb.primitive.transform, dtype=np.float64)
    extents = np.asarray(obb.primitive.extents, dtype=np.float64)
    axis = T[:3, int(np.argmax(extents))]
    return _fit_capsule_with_axis(pts, axis)


def fit_capsule_along_axis(mesh: trimesh.Trimesh, axis: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Same containment fit as `fit_tight_capsule`, but along a CHOSEN axis
    instead of the mesh's own longest OBB dimension.

    Still fully conservative -- radius and height are still computed to
    CONTAIN every sampled point, exactly like the automatic fit -- it is
    only the axis that stops being automatic. Use this when the automatic
    choice is provably wrong for what the capsule is meant to represent
    (see CAPSULE_FORCED_AXIS below), not as a way to shrink a capsule:
    a badly-chosen axis makes the fit WORSE (fatter), never tighter, so
    this is for correcting orientation, and `CAPSULE_OVERRIDE` is still
    the tool for shrinking an already-correctly-oriented fit."""
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(mesh.faces) > 0:
        pts = np.vstack([pts, _surface_samples(mesh)])
    return _fit_capsule_with_axis(pts, axis)


def _capsule_from_fit(center, axis, radius, height) -> Capsule:
    """Build a pyroki Capsule (local z = axis) at `center`."""
    z = np.asarray(axis, dtype=np.float64)
    # any orthonormal frame with z as third column
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(tmp, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rm = np.stack([x, y, z], axis=1)
    se3 = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_matrix(jnp.asarray(Rm)), jnp.asarray(center)
    )
    cap = Capsule.from_radius_height(
        position=jnp.zeros(3),
        wxyz=jnp.array([1.0, 0.0, 0.0, 0.0]),
        radius=jnp.asarray(radius),
        height=jnp.asarray(height),
    )
    return cap.transform(se3)


## ------------------------------------------------------------------ ##
## Per-link manual overrides
## ------------------------------------------------------------------ ##
##
## `fit_tight_capsule` is automatic and link-blind: it does not know that
## "right_gripper_base" carries FOUR concatenated collision meshes (gripper
## prop + bar + wrist mount + the D405 wrist camera -- see giava.urdf,
## the four <collision> blocks under that one <link>), so it fits ONE
## capsule spanning all of them, as fat and as long as the camera-to-plate
## distance demands. Splitting that properly means giving the camera its
## own URDF LINK (the way the middle arm's camera already gets its own
## middle_camera/middle_camera_body/middle_camera_cover, distinct from
## middle_wrist_link) -- a real change to giava.urdf, not a Python knob.
##
## Until/unless that split happens, this table is the escape hatch: a
## named link's fit can be adjusted or replaced WITHOUT touching
## `fit_tight_capsule` or any other link's fit. Every entry is a function
## (center, axis, R, h) -> (center, axis, R, h), applied strictly AFTER
## the automatic containment fit -- so "no entry for this link" is
## BYTE-IDENTICAL to before this table existed.
##
## WARNING ON TIGHTENING: shrinking R or h below what the automatic fit
## returned means the capsule may no longer CONTAIN the true mesh -- the
## circumscription proof ("capsule clear" implies "mesh clear") no longer
## holds for that link. This is lower-risk than it sounds ONLY because of
## the fine/GJK tier: capsule_gate.py's fine tier re-checks flagged pairs
## against the true convex hulls (mesh-exact, unaffected by anything in
## this file), so a tightened coarse capsule mostly changes which pairs
## get fast-accepted vs. escalated -- it does not, by itself, weaken what
## the gate ultimately enforces. It DOES weaken table_gate.py, which has
## no fine tier and treats the coarse capsule as authoritative (see its
## docstring) -- do not tighten a link's capsule below its true envelope
## if that link is ever checked against the table.
CAPSULE_OVERRIDE: dict = {
    ## Example (uncomment/edit to use):
    ##
    ## "right_left_finger_link": lambda center, axis, R, h: (
    ##     center, axis, R * 0.8, h),   # 20% tighter radius, same length
}


## ------------------------------------------------------------------ ##
## Per-link forced fit axis
## ------------------------------------------------------------------ ##
##
## `fit_tight_capsule`'s longest-OBB-dimension rule assumes the mesh's
## biggest bounding-box extent is the functionally meaningful "length" --
## true for most links, false for giava's custom finger. Measured
## (2026-09-02, calibration/tcp.py's own TCP direction as ground truth for
## "where the finger actually points"): the finger mesh's longest OBB
## extent (100.5 mm) runs within 16 deg of the SLIDE axis (open/close),
## and the TCP/reach direction -- where the finger actually extends toward
## a grasped object -- sits 75 deg away from that, i.e. close to
## PERPENDICULAR to the auto-fit axis and near-exactly along the gripper
## plate's surface NORMAL. A capsule fit along the auto axis is long and
## thin in the wrong direction: it points sideways, roughly parallel to
## the plate, rather than out toward whatever the fingers are closing on.
##
## Values are UNIT AXES IN THE LINK'S OWN LOCAL FRAME (the frame
## `_get_trimesh_collision_geometries` already transforms the mesh into --
## the same frame `fit_tight_capsule` receives). Derived from
## calibration/tcp.py's TCP_ALONG_GRIPPER_LINK_X, composed through the
## gripper_link->gripper_base and gripper_base->finger joint rotations in
## giava.urdf; see scratch derivation in this session's history if it
## ever needs re-deriving after a URDF change to either joint.
CAPSULE_FORCED_AXIS: Dict[str, np.ndarray] = {
    "right_left_finger_link":  np.array([0.0, -1.0, 0.0]),
    "right_right_finger_link": np.array([0.0,  1.0, 0.0]),
    "left_left_finger_link":   np.array([0.0, -1.0, 0.0]),
    "left_right_finger_link":  np.array([0.0,  1.0, 0.0]),
}


def _apply_override(name: str, center, axis, R: float, h: float):
    fn = CAPSULE_OVERRIDE.get(name)
    if fn is None:
        return center, axis, R, h
    return fn(center, axis, R, h)


## --------------------------------------------------------------------- ##
## Disk cache for fitted capsule models
## --------------------------------------------------------------------- ##
## Building one of these costs ~3.4 s (2.1 s of it inside
## RobotCollision.from_urdf), and a data_collection launch builds THREE: the
## IK's table-collision cost, the capsule gate and the table gate.  The robot
## does not change between launches, so the result is cached under
## ~/.cache/giava_capsules (GIAVA_CAPSULE_CACHE=<dir>, or =0 to disable).
##
## THIS IS SAFETY GEOMETRY, so the cache is validated, not trusted: an entry
## is used only when the URDF's XML, this module's source (every override and
## forced axis lives here), the pyroki geometry class and the exact link-name
## list all match.  Any mismatch, any unreadable file, any shape surprise --
## rebuild.  A stale entry can therefore not quietly widen a margin.
_CACHE_VERSION = 1


def _capsule_cache_dir() -> Path | None:
    val = os.environ.get("GIAVA_CAPSULE_CACHE", "").strip()
    if val in ("0", "off", "false"):
        return None
    return Path(val) if val else Path.home() / ".cache" / "giava_capsules"


def _cache_key(urdf, tag: str) -> str:
    h = hashlib.sha256()
    h.update(f"v{_CACHE_VERSION}|{tag}|".encode())
    h.update(Path(__file__).read_bytes())
    try:
        h.update(urdf.write_xml_string())
    except Exception:
        return ""          # cannot identify this URDF -> do not cache
    return h.hexdigest()


def _cache_load(key: str, urdf) -> RobotCollision | None:
    d = _capsule_cache_dir()
    if not key or d is None or not (d / f"{key}.npz").exists():
        return None
    try:
        z = np.load(d / f"{key}.npz", allow_pickle=False)
        names = tuple(str(x) for x in z["link_names"])
        ## The names are the identity check that survives a pyroki change:
        ## same robot, same links, same order, or the entry is not ours.
        if len(names) != int(z["num_links"]):
            return None
        if tuple(urdf.link_map.keys()) and not set(names) <= set(urdf.link_map):
            return None
        coll = Capsule(pose=jaxlie.SE3(jnp.asarray(z["pose"])),
                       size=jnp.asarray(z["size"]))
        return RobotCollision(
            num_links=int(z["num_links"]),
            link_names=names,
            coll=coll,
            active_idx_i=tuple(int(x) for x in z["active_idx_i"]),
            active_idx_j=tuple(int(x) for x in z["active_idx_j"]),
            _geom_to_link_idx=tuple(int(x) for x in z["geom_to_link_idx"]),
        )
    except Exception:
        return None        # corrupt / older layout -> rebuild, never guess


def _cache_store(key: str, rc: RobotCollision) -> None:
    d = _capsule_cache_dir()
    if not key or d is None:
        return
    try:
        d.mkdir(parents=True, exist_ok=True)
        ## Name it *.npz: np.savez APPENDS .npz to anything else, which
        ## would leave the rename below pointing at a file that never existed.
        tmp = d / f"{key}.tmp{os.getpid()}.npz"
        np.savez(tmp,
                 num_links=np.int64(rc.num_links),
                 link_names=np.asarray(list(rc.link_names), dtype=object).astype("U"),
                 pose=np.asarray(rc.coll.pose.wxyz_xyz),
                 size=np.asarray(rc.coll.size),
                 active_idx_i=np.asarray(rc.active_idx_i, dtype=np.int64),
                 active_idx_j=np.asarray(rc.active_idx_j, dtype=np.int64),
                 geom_to_link_idx=np.asarray(rc._geom_to_link_idx, dtype=np.int64))
        tmp.replace(d / f"{key}.npz")      # atomic: no half-written entry
    except Exception:
        pass               # a cache that cannot be written is not an error


def _cached(tag: str, urdf, build):
    key = _cache_key(urdf, tag)
    hit = _cache_load(key, urdf)
    if hit is not None:
        return hit
    rc = build()
    if isinstance(rc.coll, Capsule):
        _cache_store(key, rc)
    return rc


def tight_capsule_collision(urdf, user_ignore_pairs: Tuple[Tuple[str, str], ...] = ()
                            ) -> RobotCollision:
    """RobotCollision identical to `from_urdf` except for the capsule fits."""
    return _cached(f"tight|{sorted(user_ignore_pairs)}", urdf,
                   lambda: _tight_capsule_collision(urdf, user_ignore_pairs))


def _tight_capsule_collision(urdf, user_ignore_pairs: Tuple[Tuple[str, str], ...] = ()
                             ) -> RobotCollision:
    base = RobotCollision.from_urdf(urdf, user_ignore_pairs=user_ignore_pairs)
    caps = []
    for name in base.link_names:
        mesh = RobotCollision._get_trimesh_collision_geometries(urdf, name)
        if mesh.is_empty:
            caps.append(
                Capsule(pose=jaxlie.SE3.identity(), size=jnp.zeros(2))
            )
            continue
        forced = CAPSULE_FORCED_AXIS.get(name)
        if forced is not None:
            center, axis, R, h = fit_capsule_along_axis(mesh, forced)
        else:
            center, axis, R, h = fit_tight_capsule(mesh)
        center, axis, R, h = _apply_override(name, center, axis, R, h)
        caps.append(_capsule_from_fit(center, axis, R, h))
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *caps)
    return RobotCollision(
        num_links=base.num_links,
        link_names=base.link_names,
        coll=stacked,
        active_idx_i=base.active_idx_i,
        active_idx_j=base.active_idx_j,
        _geom_to_link_idx=base._geom_to_link_idx,
    )


## --------------------------------------------------------------------- ##
## Multi-sphere model (curobo VOXEL algorithm, mirrored dependency-free)
## --------------------------------------------------------------------- ##
## curobo's voxel_fit_mesh: uniform grid over the mesh bbox (~n cells) →
## keep interior points (SDF) → radius = largest inscribed sphere at that
## centre.  Their SDF runs on Warp; we use trimesh's (convex-hull fallback
## for non-watertight links).  Deviation flagged: same algorithm, no GPU dep.
## Under-coverage between inscribed spheres is expected and absorbed by the
## collision margin; the scorecard quantifies the signed bias.

## Per-link sphere budgets — geometry-driven (more for boxes/plates/forks,
## fewer for cylinders), tuned in Phase 6:
SPHERE_BUDGET = {
    "base_link": 12,        # 0.30×0.20×0.08 plate
    "shoulder_link": 4,
    "upper_arm_link": 6,
    "upper_forearm_link": 8,  # 0.20×0.10×0.04 plate-ish
    "lower_forearm_link": 4,
    "wrist_link": 4,
    "gripper_link": 4,
    "gripper_base": 10,     # the fork
    "right_finger_link": 6,
    "left_finger_link": 6,
}
SPHERE_BUDGET_MIDDLE = {
    "middle_base_link": 12,
    "middle_shoulder_link": 4,
    "middle_upper_arm_link": 6,
    "middle_upper_forearm_link": 4,  # slender cylinder
    "middle_lower_forearm_link": 4,
    "middle_wrist_link": 3,
    "middle_pan_link": 3,
    "middle_camera": 2,
    "middle_camera_body": 4,
    "middle_camera_cover": 6,  # thin plate
}


def _budget_for(link_name: str) -> int:
    if link_name in SPHERE_BUDGET_MIDDLE:
        return SPHERE_BUDGET_MIDDLE[link_name]
    for suffix, n in SPHERE_BUDGET.items():
        if link_name.endswith(suffix):
            return n
    return 0


def _solid_bodies(mesh: trimesh.Trimesh):
    """The solid envelope the link occupies: per-connected-body convex hulls.

    The GIAVA link STLs are mostly thin-shell housings (watertight but only
    12–23% volume fill), so inscribed spheres of the *raw* mesh live in the
    walls.  What collision cares about is the occupied envelope — per-body
    hulls (splitting first keeps separate sub-parts from being bridged).
    Negligible sub-bodies (screws, brackets: hull < 8 cm³ and < 6 cm across)
    are dropped — the neighbouring housing envelope covers them."""
    bodies = mesh.split(only_watertight=False)
    if len(bodies) == 0:
        bodies = [mesh]
    hulls = []
    for b in bodies:
        try:
            h = b.convex_hull
        except Exception:
            continue
        if h.volume <= 1e-9:
            continue
        if h.volume < 8e-6 and float(h.extents.max()) < 0.06:
            continue  # negligible hardware
        hulls.append(h)
    if not hulls:
        hulls = [mesh.convex_hull]
    return hulls


PLATE_THICKNESS = 0.025  # hulls thinner than this get the mid-plane treatment
PLATE_RADIUS = 0.012  # target sphere radius on plates (bounded protrusion)


def _fit_plate(h: trimesh.Trimesh, nb: int):
    """Flat hull: 2-D grid of spheres in the mid-plane.

    Inscribed spheres of a plate can never cover its area (r ≤ thickness/2),
    so plates get r = max(thickness/2, 12 mm) with centers on a grid in the
    two long directions — bounded out-of-plane protrusion (≤ 12 − t/2 mm,
    vs. the capsule fit's 70 mm on the camera cover)."""
    T = np.asarray(h.bounding_box_oriented.primitive.transform)
    E = np.asarray(h.bounding_box_oriented.primitive.extents)
    order = np.argsort(E)  # thin axis first
    r = max(E[order[0]] / 2, PLATE_RADIUS)
    # grid counts in the two long directions, proportional to extent
    e1, e2 = E[order[1]], E[order[2]]
    n2 = max(1, int(round(np.sqrt(nb * e2 / max(e1, 1e-6)))))
    n1 = max(1, nb // n2)
    c_local = np.zeros((n1 * n2, 3))
    axes_local = np.eye(3)
    g1 = np.linspace(-e1 / 2 + r, e1 / 2 - r, n1) if n1 > 1 else np.array([0.0])
    g2 = np.linspace(-e2 / 2 + r, e2 / 2 - r, n2) if n2 > 1 else np.array([0.0])
    k = 0
    for a in g1:
        for b in g2:
            c_local[k, order[1]] = a
            c_local[k, order[2]] = b
            k += 1
    c_local = c_local[:k]
    centers = (T[:3, :3] @ c_local.T).T + T[:3, 3]
    return centers, np.full(len(centers), r)


def fit_spheres_voxel(mesh: trimesh.Trimesh, n: int):
    """curobo voxel-fit (interior grid → inscribed radii) on per-body hulls,
    spread-aware selection, plate-aware fallback.  Exact budget total."""
    if mesh.is_empty or n <= 0:
        return np.zeros((0, 3)), np.zeros(0)
    hulls = _solid_bodies(mesh)
    vols = np.array([max(h.volume, 1e-9) for h in hulls])
    budgets = np.maximum(1, np.round(n * vols / vols.sum()).astype(int))
    while budgets.sum() > n and budgets.max() > 1:
        budgets[np.argmax(budgets)] -= 1

    all_c, all_r = [], []
    for h, nb in zip(hulls, budgets):
        E = np.asarray(h.bounding_box_oriented.primitive.extents)
        if E.min() < PLATE_THICKNESS:
            c, r = _fit_plate(h, nb)
            all_c.append(c)
            all_r.append(r)
            continue
        lo, hi = h.bounds
        extents = np.maximum(hi - lo, 1e-6)
        pitch = float((np.prod(extents) / (6 * nb)) ** (1 / 3))
        axes = [np.linspace(lo[k] + pitch / 2, hi[k] - pitch / 2,
                            max(int(np.ceil(extents[k] / pitch)), 1))
                for k in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        sd = trimesh.proximity.signed_distance(h, grid)  # positive inside
        inside = sd > 2e-3  # candidates with a non-degenerate radius
        if not inside.any():
            c, r = _fit_plate(h, nb)
            all_c.append(c)
            all_r.append(r)
            continue
        cand_c, cand_r = grid[inside], sd[inside]
        chosen = [int(np.argmax(cand_r))]
        while len(chosen) < min(nb, len(cand_c)):
            d_near = np.min(
                np.linalg.norm(
                    cand_c[:, None, :] - cand_c[chosen][None, :, :], axis=-1
                ),
                axis=1,
            )
            chosen.append(int(np.argmax(d_near + 0.5 * cand_r)))
        all_c.append(cand_c[chosen])
        all_r.append(cand_r[chosen])
    return np.concatenate(all_c), np.concatenate(all_r)


def sphere_decomposition(urdf, cache: Path | None = None) -> dict:
    """{link: {centers, radii}} for every link with geometry (JSON-cached)."""
    import json

    if cache is not None and cache.exists():
        return json.loads(cache.read_text())
    base = RobotCollision.from_urdf(urdf)
    out = {}
    for name in base.link_names:
        n = _budget_for(name)
        mesh = RobotCollision._get_trimesh_collision_geometries(urdf, name)
        centers, radii = fit_spheres_voxel(mesh, n)
        if len(centers):
            out[name] = {"centers": [list(map(float, c)) for c in centers],
                         "radii": [float(r) for r in radii]}
    if cache is not None:
        cache.write_text(json.dumps(out, indent=1))
    return out


def sphere_collision(urdf, cache: Path | None = None) -> RobotCollision:
    """RobotCollision in sphere mode from the (cached) decomposition."""
    decomp = sphere_decomposition(urdf, cache)
    return RobotCollision.from_sphere_decomposition(decomp, urdf)


def pruned_sphere_collision(urdf, results_dir: Path) -> RobotCollision:
    """The deployment model: 180-sphere decomposition with the Phase-7
    corpus-pruned pair set (5,884 of 14,491 sphere pairs kept; functional
    grasping pairs, permanently-inside-margin pairs, and never-active pairs
    removed by rule — see phase7_pair_classify.py)."""
    base = sphere_collision(urdf, results_dir / "sphere_decomposition.json")
    pruning = np.load(results_dir / "sphere_pair_pruning.npz")
    return RobotCollision(
        num_links=base.num_links,
        link_names=base.link_names,
        coll=base.coll,
        active_idx_i=tuple(int(x) for x in pruning["kept_idx_i"]),
        active_idx_j=tuple(int(x) for x in pruning["kept_idx_j"]),
        _geom_to_link_idx=base._geom_to_link_idx,
    )


def pruned_tight_capsule_collision(urdf) -> RobotCollision:
    """Head-to-head capsule reference: corrected fits + the Part-1 link-level
    pruning (12 structural + 2 functional pairs removed)."""
    STRUCTURAL = (
        ("left_base_link", "left_upper_arm_link"),
        ("right_base_link", "right_upper_arm_link"),
        ("middle_base_link", "middle_upper_arm_link"),
        ("middle_camera_body", "middle_camera_cover"),
        ("left_wrist_link", "left_gripper_base"),
        ("right_wrist_link", "right_gripper_base"),
        ("left_lower_forearm_link", "left_gripper_base"),
        ("right_lower_forearm_link", "right_gripper_base"),
        ("middle_pan_link", "middle_camera_cover"),
        ("middle_wrist_link", "middle_camera_cover"),
        ("middle_pan_link", "middle_camera_body"),
        ("middle_lower_forearm_link", "middle_pan_link"),
        ## *_gripper_camera (2026-09-02, split off *_gripper_base -- see
        ## that link's own comment in giava.urdf): a SIBLING of the wrist
        ## mount, the fingers and gripper_base itself under the same rigid
        ## assembly, so it inherits the same "always close by mounting
        ## geometry" relationship gripper_base already has with the wrist
        ## and lower-forearm links, and immediate-adjacency auto-ignore
        ## does NOT cover siblings (exactly why middle_camera_body/cover
        ## above needed a manual entry too).
        ##
        ## THE RULE IS CONSTANCY, NOT SIGN. A pair whose coarse distance
        ## does not move across a real pose sweep cannot be influenced by
        ## any command -- there is nothing for the gate to do about it
        ## either way, and leaving a NEAR-MARGIN constant active is fragile
        ## against the margin being retuned later (this file's margins
        ## have moved twice already). A pair that varies with pose, even
        ## if it happens to read positive everywhere tested, is a real
        ## gated relationship and stays active.
        ##
        ## Measured across five named poses (urdf-zero/rest/forward/
        ## high/low), against the margins active at measurement time
        ## (camera 32 mm, gripper 28 mm):
        ##   wrist_link          25.7-25.8 mm  constant, violates  -> prune
        ##   each finger         33.0-33.7 mm  constant, 1 mm clear -> prune
        ##                                     (too close to the margin to
        ##                                      trust as pose sweeps or the
        ##                                      margin itself move)
        ##   lower_forearm_link  55.3-57.8 mm  constant, comfortably clear
        ##                                     -> NOT pruned; no reason to
        ##                                        exclude a pair with this
        ##                                        much real headroom
        ("left_wrist_link", "left_gripper_camera"),
        ("right_wrist_link", "right_gripper_camera"),
        ("left_left_finger_link", "left_gripper_camera"),
        ("left_right_finger_link", "left_gripper_camera"),
        ("right_left_finger_link", "right_gripper_camera"),
        ("right_right_finger_link", "right_gripper_camera"),
        ## *_gripper_plate (2026-09-02, split off *_gripper_base -- see
        ## giava.urdf): prop+bar, i.e. the piece the fingers are actually
        ## mounted on, so it is closer to everything nearby than the
        ## wrist-mount piece gripper_base kept was. Same constancy rule:
        ##   wrist_link    -34.8 mm  constant, deep overlap    -> prune
        ##   each finger   -56.7/-57.4 mm  constant, deep overlap -> prune
        ##   camera        +12.6 mm  constant, violates (camera margin) -> prune
        ##   lower_forearm 14.3-18.8 mm  VARIES with pose -> NOT pruned,
        ##                 left for the fine/GJK tier to rescue as needed
        ("left_wrist_link", "left_gripper_plate"),
        ("right_wrist_link", "right_gripper_plate"),
        ("left_left_finger_link", "left_gripper_plate"),
        ("left_right_finger_link", "left_gripper_plate"),
        ("right_left_finger_link", "right_gripper_plate"),
        ("right_right_finger_link", "right_gripper_plate"),
        ("left_gripper_camera", "left_gripper_plate"),
        ("right_gripper_camera", "right_gripper_plate"),
    )
    FUNCTIONAL = (
        ("left_left_finger_link", "left_right_finger_link"),
        ("right_left_finger_link", "right_right_finger_link"),
    )
    return tight_capsule_collision(urdf, user_ignore_pairs=STRUCTURAL + FUNCTIONAL)


if __name__ == "__main__":
    import robot_model as rm

    robot, urdf = rm.load(with_urdf=True)
    rc_old = RobotCollision.from_urdf(urdf)
    rc_new = tight_capsule_collision(urdf)
    r_old = np.asarray(rc_old.coll.radius)
    h_old = np.asarray(rc_old.coll.height)
    r_new = np.asarray(rc_new.coll.radius)
    h_new = np.asarray(rc_new.coll.height)
    print(f"{'link':28s} {'old r':>6s} {'old h':>6s} {'old len':>8s} | "
          f"{'new r':>6s} {'new h':>6s} {'new len':>8s}")
    for i, name in enumerate(rc_old.link_names):
        if r_old[i] < 1e-6 and r_new[i] < 1e-6:
            continue
        print(
            f"{name:28s} {r_old[i]:6.3f} {h_old[i]:6.3f} "
            f"{h_old[i] + 2 * r_old[i]:8.3f} | "
            f"{r_new[i]:6.3f} {h_new[i]:6.3f} {h_new[i] + 2 * r_new[i]:8.3f}"
        )
