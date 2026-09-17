"""Tabletop world-collision inspector (table_collision.py). Hardware-validated,
un-swept term
-- see that module's docstring. This is the "look at the geometry before you
trust it" step the self-collision study did before any hardware trial
(COLLISION_STUDY.md: "Benchmarking pending user inspection of the geometry
in Viser").

Shows the robot, the tabletop plane, and every sphere the table cost can see
(grey = excluded as structurally height-invariant, see
table_collision.TABLE_EXCLUDED_LINKS) color-coded by live clearance to the
plane. A jog panel drives the LEFT gripper's commanded target straight down
through the table using the real ComboIK table-only solve, so you can watch
the soft-hinge cost hold a boundary instead of letting the gripper punch
through -- exactly the "allow contact, resist penetration" behaviour this
term is meant to provide.

Run:  JAX_PLATFORMS=cpu python view_table_collision.py [--port 8096]
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import json
import sys
import time
from pathlib import Path

import jaxlie
import numpy as np
import viser
from viser.extras import ViserUrdf

sys.path.insert(0, str(Path(__file__).resolve().parent))

import robot_model as rm
from table_collision import (TABLE_EXCLUDED_LINKS, table_halfspace,
                             table_robot_collision)
from variants import ComboIK

EE_LINKS = ("left_gripper_base", "right_gripper_base", "middle_camera_cover")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--table-z", type=float, default=0.0,
                     help="world-frame table height, m "
                          "(same convention as base_validation.py --table-z)")
    ap.add_argument("--margin", type=float, default=0.020)
    ap.add_argument("--weight", type=float, default=100.0)
    args = ap.parse_args()

    robot, urdf = rm.load(with_urdf=True)
    results_dir = Path(__file__).parent / "results"
    decomp = json.loads((results_dir / "sphere_decomposition.json").read_text())
    table_coll = table_robot_collision(urdf, results_dir)
    table_geom = table_halfspace(args.table_z)

    ik = ComboIK(
        robot, EE_LINKS, extras={"table": args.weight},
        table_coll=table_coll, table_geom=table_geom, table_margin=args.margin,
    )

    q0 = rm.home_config(robot)
    fk0 = robot.forward_kinematics(np.asarray(q0, dtype=np.float32))
    idx = {n: robot.links.names.index(n) for n in EE_LINKS}
    home_pos = np.stack([np.asarray(jaxlie.SE3(fk0[idx[n]]).translation()) for n in EE_LINKS])
    home_wxyz = np.stack([np.asarray(jaxlie.SE3(fk0[idx[n]]).rotation().wxyz) for n in EE_LINKS])
    ik.warmup(q0)

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid(
        "/table", width=1.5, height=1.5, cell_size=0.1,
        position=(0.0, 0.0, args.table_z), plane="xy",
    )
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    with server.gui.add_folder("Jog left gripper (real table-only solve)"):
        jog_z = server.gui.add_slider(
            "commanded z, mm", -150.0, 650.0, 1.0, home_pos[0, 2] * 1e3)
        info = server.gui.add_markdown("")

    with server.gui.add_folder("Display"):
        show_spheres = server.gui.add_checkbox("clearance spheres", True)
        opacity = server.gui.add_slider("sphere opacity", 0.05, 1.0, 0.05, 0.55)

    server.gui.add_markdown(
        "**green** clear by > 2x margin  \n"
        "**amber** inside 2x margin  \n"
        "**red** inside margin (soft cost actively pushing back)  \n"
        "**grey** excluded: kinematically height-invariant "
        f"({len(TABLE_EXCLUDED_LINKS)} links -- base/shoulder mounts)"
    )

    sphere_handles: dict[str, object] = {}
    q_state = {"q": np.asarray(q0, dtype=np.float32)}

    def color_for(dist: float, margin: float) -> tuple[int, int, int]:
        if dist < margin:
            return (211, 47, 47)
        if dist < 2 * margin:
            return (245, 166, 35)
        return (67, 160, 71)

    def redraw_spheres(q: np.ndarray, dist_mat: np.ndarray) -> None:
        fk = robot.forward_kinematics(q)
        geom_idx = 0  # tracks compute_world_collision_distance's row order:
                      # table_coll.link_names, in each link's decomp order
        for link in table_coll.link_names:
            ent = decomp.get(link)
            if ent is None:
                continue
            centers = np.asarray(ent["centers"], dtype=float)
            radii = np.asarray(ent["radii"], dtype=float)
            excluded = link in TABLE_EXCLUDED_LINKS
            se3 = jaxlie.SE3(fk[robot.links.names.index(link)])
            R = np.asarray(se3.rotation().as_matrix())
            p = np.asarray(se3.translation())
            world = (R @ centers.T).T + p
            for k in range(len(radii)):
                key = f"/spheres/{link}_{k}"
                if excluded:
                    color, opac = (120, 120, 120), float(opacity.value) * 0.5
                else:
                    d = float(dist_mat[geom_idx])
                    geom_idx += 1
                    color, opac = color_for(d, args.margin), float(opacity.value)
                if key in sphere_handles:
                    sphere_handles[key].remove()
                sphere_handles[key] = server.scene.add_icosphere(
                    key, radius=float(radii[k]), position=tuple(world[k]),
                    color=color, opacity=opac, visible=show_spheres.value,
                )

    def update(_=None) -> None:
        target_pos = home_pos.copy()
        target_pos[0, 2] = float(jog_z.value) * 1e-3
        res = ik.solve(q_state["q"], target_pos, home_wxyz)
        q = np.asarray(res.q, dtype=np.float32)
        q_state["q"] = q  # warm-start the next jog, like the real control loop
        urdf_vis.update_cfg(q)

        dist_mat = np.asarray(
            table_coll.compute_world_collision_distance(robot, q, table_geom)
        )[:, 0]
        fk = robot.forward_kinematics(q)
        achieved_z = float(jaxlie.SE3(fk[idx["left_gripper_base"]]).translation()[2])
        info.content = (
            f"commanded z: **{jog_z.value:+.1f} mm**  \n"
            f"achieved z:  **{achieved_z * 1e3:+.1f} mm**  \n"
            f"min table clearance (model): **{float(dist_mat.min()) * 1e3:+.1f} mm**  \n"
            f"solve: {res.solve_ms:.1f} ms, {res.iterations} iters"
        )
        redraw_spheres(q, dist_mat)

    @jog_z.on_update
    def _(_):
        update()

    @show_spheres.on_update
    def _(_):
        for h in sphere_handles.values():
            h.visible = show_spheres.value

    @opacity.on_update
    def _(_):
        update()

    update()
    print(f"\n  Table-collision inspector:  http://localhost:{args.port}\n")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
