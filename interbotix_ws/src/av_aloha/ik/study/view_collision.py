"""Collision-study C3 — inspect the collision geometry in Viser.

Shows the robot with its fitted collision geometry overlaid (semi-transparent),
at selectable poses (home / parked / stress-trajectory apexes from the saved
baseline rollouts).  Toggle between the capsule model (pyroki from_urdf) and,
once built, the sphere decomposition.

Color code:
  orange  — capsule of a normal (kept) link
  red     — capsule involved in a *permanently-inside-margin* pair (the 12
            structural pairs; these carry no signal and get pruned)
  magenta — same-gripper finger capsules (functional grasp contact; pruned)

Run:  JAX_PLATFORMS=cpu python view_collision.py [--port 8091]
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jaxlie
import viser
from viser.extras import ViserUrdf

import robot_model as rm
from pyroki.collision import RobotCollision

STRUCTURAL_LINKS = {
    # links appearing in the 12 permanently-inside-margin pairs
    "left_base_link", "left_upper_arm_link", "right_base_link",
    "right_upper_arm_link", "middle_base_link", "middle_upper_arm_link",
    "left_wrist_link", "left_gripper_base", "right_wrist_link",
    "right_gripper_base", "left_lower_forearm_link", "right_lower_forearm_link",
    "middle_camera_body", "middle_camera_cover", "middle_pan_link",
    "middle_wrist_link", "middle_lower_forearm_link",
}
FINGER_LINKS = {
    "left_left_finger_link", "left_right_finger_link",
    "right_left_finger_link", "right_right_finger_link",
}


def load_poses():
    """Named configurations to inspect, from the saved baseline rollouts."""
    res = Path(__file__).parent / "results" / "baseline"
    robot = rm.load()
    q0 = rm.home_config(robot)
    poses = {"home": q0}
    picks = {
        "parked hands (mid suite)": ("mid_trans_x.npz", 150),
        "arms_converge apex": ("arms_converge.npz", 275),
        "self_fold apex": ("self_fold.npz", 300),
        "teleop_grasp end": ("teleop_grasp.npz", 299),
        "reach_limit apex": ("reach_limit.npz", 250),
    }
    for name, (fname, idx) in picks.items():
        f = res / fname
        if f.exists():
            q = np.load(f)["q"]
            poses[name] = q[min(idx, len(q) - 1)]
    return poses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()

    from collision_models import sphere_collision, tight_capsule_collision

    robot, urdf = rm.load(with_urdf=True)
    rc = RobotCollision.from_urdf(urdf)
    rc_tight = tight_capsule_collision(urdf)
    rc_sphere = sphere_collision(
        urdf, Path(__file__).parent / "results" / "sphere_decomposition.json"
    )
    models = {
        "spheres (180, voxel fit)": rc_sphere.get_link_collision_meshes(),
        "pyroki (min-cylinder fit)": rc.get_link_collision_meshes(),
        "tight (longest-axis fit)": rc_tight.get_link_collision_meshes(),
    }
    poses = load_poses()

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    with server.gui.add_folder("Inspect"):
        pose_dd = server.gui.add_dropdown("pose", tuple(poses.keys()),
                                          initial_value="home")
        model_dd = server.gui.add_dropdown("collision model", tuple(models.keys()),
                                           initial_value="spheres (180, voxel fit)")
        show_robot = server.gui.add_checkbox("show robot mesh", True)
        show_caps = server.gui.add_checkbox("show capsules", True)
        opacity = server.gui.add_slider("capsule opacity", 0.05, 0.9, 0.05, 0.35)
    info = server.gui.add_markdown(
        "**red** = link in a permanently-inside-margin pair (pruned)\n\n"
        "**magenta** = gripper fingers (functional contact; same-gripper pair pruned)\n\n"
        "**orange** = kept collision capsule"
    )

    mesh_handles = {}

    def color_for(name: str):
        if name in FINGER_LINKS:
            return (216, 82, 129)
        if name in STRUCTURAL_LINKS:
            return (217, 89, 38)
        return (237, 161, 0)

    def update(_=None) -> None:
        q = np.asarray(poses[pose_dd.value], dtype=np.float32)
        urdf_vis.update_cfg(q)
        link_meshes = models[model_dd.value]
        fk = robot.forward_kinematics(q)
        for i, name in enumerate(rc.link_names):
            mesh = link_meshes[name]
            if mesh.is_empty:
                continue
            se3 = jaxlie.SE3(fk[i])
            key = f"/capsules/{name}"
            if key in mesh_handles:
                mesh_handles[key].remove()
            mesh_handles[key] = server.scene.add_mesh_simple(
                key,
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                faces=np.asarray(mesh.faces, dtype=np.uint32),
                color=color_for(name),
                opacity=float(opacity.value),
                visible=show_caps.value,
            )
            h = mesh_handles[key]
            h.position = tuple(np.asarray(se3.translation(), dtype=float))
            h.wxyz = tuple(np.asarray(se3.rotation().wxyz, dtype=float))

    @pose_dd.on_update
    def _(_):
        update()

    @model_dd.on_update
    def _(_):
        update()

    @show_caps.on_update
    def _(_):
        for h in mesh_handles.values():
            h.visible = show_caps.value

    @opacity.on_update
    def _(_):
        for h in mesh_handles.values():
            h.opacity = float(opacity.value)

    @show_robot.on_update
    def _(_):
        # ViserUrdf exposes its frame root; toggle visibility via scene node.
        try:
            urdf_vis._joint_frames[0]  # noqa: B018 — existence probe
        except Exception:
            pass
        server.scene.add_frame("/robot", show_axes=False, visible=show_robot.value)

    update()
    print(f"\n  Collision-geometry inspector:  http://localhost:{args.port}\n")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
