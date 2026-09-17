"""Collision-study C5 — mesh-distance ground truth.

For key (configuration, link-pair) cases, computes the *true* collision-mesh
distance (trimesh signed distance on dense surface samples, both directions)
and compares it against the pyroki capsule model and the tight-capsule model.

Purposes:
1. Quantify each model's conservatism (capsule-distance − mesh-distance).
2. Validate the user's requirement that the self-fold apex — a legitimate
   rest posture — is truly collision-free (any model reading it as collision
   is over-conservative there).
3. Confirm designed-contact cases (arms_converge apex) are real contacts.

Run:  JAX_PLATFORMS=cpu python mesh_ground_truth.py
Writes results/mesh_ground_truth.csv
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import csv
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jaxlie
import jax.numpy as jnp

import robot_model as rm
from collision_models import tight_capsule_collision
from pyroki.collision import RobotCollision
from pyroki.collision._collision import collide

N_SAMPLES = 4000

CASES = {
    "home": None,  # filled from home config
    "self_fold_apex": ("self_fold.npz", 300),
    "arms_converge_apex": ("arms_converge.npz", 275),
    "teleop_grasp_end": ("teleop_grasp.npz", 299),
    "parked_hands": ("mid_trans_x.npz", 150),
}

PAIRS = [
    # structural suspects (intra-chain)
    ("right_base_link", "right_upper_arm_link"),
    ("right_wrist_link", "right_gripper_base"),
    ("right_lower_forearm_link", "right_gripper_base"),
    # folding pairs (self_fold)
    ("right_base_link", "right_lower_forearm_link"),
    ("right_base_link", "right_wrist_link"),
    ("right_base_link", "right_gripper_base"),
    ("right_upper_arm_link", "right_lower_forearm_link"),
    # inter-hand pairs (bimanual)
    ("right_gripper_base", "left_gripper_base"),
    ("right_right_finger_link", "left_left_finger_link"),
    ("right_left_finger_link", "left_right_finger_link"),
    # hands vs camera
    ("right_right_finger_link", "middle_camera_body"),
    ("left_gripper_base", "middle_camera_body"),
    ("left_left_finger_link", "middle_camera_cover"),
    # parked-hands artifact pair
    ("right_base_link", "right_upper_forearm_link"),
]


def mesh_distance(m1: trimesh.Trimesh, m2: trimesh.Trimesh) -> float:
    """Approximate signed distance between two meshes.

    Separated case: min distance between dense surface samples (cKDTree);
    accuracy ~ sampling density (±3 mm at 4k samples/mesh — sufficient to
    adjudicate multi-cm model conservatism).
    Penetrating case: sign and depth from convex-hull containment — hull
    signed distance of each mesh's samples in the other (slight over-estimate
    of depth for non-convex links; sign is what matters here)."""
    p1 = np.vstack([m1.sample(N_SAMPLES), m1.vertices])
    p2 = np.vstack([m2.sample(N_SAMPLES), m2.vertices])
    from scipy.spatial import cKDTree

    d_surf = float(min(cKDTree(p1).query(p2)[0].min(),
                       cKDTree(p2).query(p1)[0].min()))
    h1, h2 = m1.convex_hull, m2.convex_hull
    inside_21 = trimesh.proximity.signed_distance(h1, p2)  # >0 inside hull
    inside_12 = trimesh.proximity.signed_distance(h2, p1)
    depth = float(max(inside_21.max(), inside_12.max()))
    if depth > 1e-4:
        return -depth
    return d_surf


def main() -> None:
    robot, urdf = rm.load(with_urdf=True)
    q0 = rm.home_config(robot)
    res = Path(__file__).parent / "results"

    configs = {}
    for name, src in CASES.items():
        if src is None:
            configs[name] = q0
        else:
            f = res / "baseline" / src[0]
            q = np.load(f)["q"]
            configs[name] = q[min(src[1], len(q) - 1)]

    rc_old = RobotCollision.from_urdf(urdf)
    rc_new = tight_capsule_collision(urdf)
    link_meshes = {
        n: RobotCollision._get_trimesh_collision_geometries(urdf, n)
        for n in {l for p in PAIRS for l in p}
    }
    name_to_idx = {n: i for i, n in enumerate(rc_old.link_names)}

    def capsule_dist(rc, q, a, b) -> float:
        fk = robot.forward_kinematics(np.asarray(q, dtype=np.float32))
        coll = rc.coll.transform(
            jaxlie.SE3(jnp.asarray(fk)[rc._geom_to_link_idx])
        )
        import jax

        ca = jax.tree.map(lambda x: x[name_to_idx[a]], coll)
        cb = jax.tree.map(lambda x: x[name_to_idx[b]], coll)
        return float(collide(ca, cb))

    rows = []
    print(f"{'config':20s} {'pair':52s} {'mesh':>8s} {'pyroki':>8s} {'tight':>8s}  (mm)")
    for cname, q in configs.items():
        fk = robot.forward_kinematics(np.asarray(q, dtype=np.float32))
        world_meshes = {}
        for lname, mesh in link_meshes.items():
            m = mesh.copy()
            se3 = jaxlie.SE3(fk[name_to_idx[lname]])
            T = np.eye(4)
            T[:3, :3] = np.asarray(se3.rotation().as_matrix())
            T[:3, 3] = np.asarray(se3.translation())
            m.apply_transform(T)
            world_meshes[lname] = m
        for a, b in PAIRS:
            d_mesh = mesh_distance(world_meshes[a], world_meshes[b])
            d_old = capsule_dist(rc_old, q, a, b)
            d_new = capsule_dist(rc_new, q, a, b)
            rows.append(
                dict(config=cname, pair=f"{a}|{b}", mesh_mm=d_mesh * 1e3,
                     pyroki_mm=d_old * 1e3, tight_mm=d_new * 1e3)
            )
            print(f"{cname:20s} {a + '|' + b:52s} {d_mesh*1e3:8.1f} "
                  f"{d_old*1e3:8.1f} {d_new*1e3:8.1f}")

    with open(res / "mesh_ground_truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {res/'mesh_ground_truth.csv'}")


if __name__ == "__main__":
    main()
