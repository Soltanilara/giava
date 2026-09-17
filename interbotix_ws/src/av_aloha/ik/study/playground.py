"""Interactive four-mode IK playground (user-requested).

Four solver modes, each a FUNDAMENTALLY SEPARATE compiled problem — a
distinct cost list assembled at construction and JIT-compiled on its own
(ComboIK's `extras` is static: different sets → different jaxls problems and
different XLA programs; nothing is shared or zero-weighted):

  1  smooth+center          pose(50/10) + limits + smoothing 0.05 + centering 0.5
  2  + collision            … + sphere self-collision (180 spheres, 5,884
                              pruned pairs, margin 20 mm, weight 100)
  3  + manipulability       … + Yoshikawa cost on both hand arms (w = 0.02)
  4  + collision + manip    … + both

All four are built and warmed up at startup (each compiles separately —
watch the console), so switching modes in the GUI swaps whole solvers.

Input sources:
  trajectory — every trajectory from the frozen suite + the bimanual
               collision subset, with play/pause, speed, scrub, loop.
  manual     — drag the three end-effector gizmos; the active solver tracks
               them live.

Live readouts: backend, solve time (last + EMA), per-arm position and
orientation error, min sphere-model clearance, LM iterations.
No world obstacles are drawn or modeled (self-collision only), and no
marker spheres are added to the scene.

Run ON GPU (A4000):
  cd ik_study
  CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python playground.py --port 8092
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import jaxlie
import viser
from viser.extras import ViserUrdf

import robot_model as rm
import trajectories as T
from collision_models import pruned_sphere_collision
from collision_trajectories import build_collision_subset
from variants import ComboIK

HERE = Path(__file__).resolve().parent

MODES = {
    "1: smooth+center (tuned base)": dict(
        extras={"smoothing": 0.05, "centering": 0.5},
        collision=False,
        blurb="pose 50/10 + joint limits + smoothing w=0.05 + centering w=0.5 "
              "— the Phase-5 tuned pair. No collision, no manipulability.",
    ),
    "2: base + collision": dict(
        extras={"smoothing": 0.05, "centering": 0.5, "collision": 100.0},
        collision=True,
        blurb="mode 1 + sphere self-collision (180 spheres, 5,884 pruned "
              "pairs, soft margin hinge, margin 20 mm, weight 100).",
    ),
    "3: base + manipulability": dict(
        extras={"smoothing": 0.05, "centering": 0.5, "manipulability": 0.02},
        collision=False,
        blurb="mode 1 + pyroki Yoshikawa manipulability cost on both hand "
              "arms (w=0.02, the deployed preset; the study found no better "
              "weight).",
    ),
    "4: base + collision + manipulability": dict(
        extras={"smoothing": 0.05, "centering": 0.5, "collision": 100.0,
                "manipulability": 0.02},
        collision=True,
        blurb="everything: mode 1 + sphere collision + manipulability.",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8092)
    args = ap.parse_args()

    print("BACKEND:", jax.default_backend(), jax.devices(), flush=True)
    robot, urdf = rm.load(with_urdf=True)
    q0 = rm.home_config(robot)
    home_pos, home_wxyz = rm.home_poses(robot)
    indices = rm.target_link_indices(robot)
    rc_sphere = pruned_sphere_collision(urdf, HERE / "results")

    # ---------------- build all four solvers (separate compiles) --------- #
    solvers = {}
    for name, spec in MODES.items():
        t0 = time.time()
        solvers[name] = ComboIK(
            robot, rm.TARGET_LINKS, spec["extras"],
            robot_coll=rc_sphere if spec["collision"] else None,
            collision_margin=0.020,
        )
        solvers[name].warmup(q0)
        print(f"  built+compiled [{name}] in {time.time()-t0:.1f} s", flush=True)

    # ---------------- jitted readout helpers ----------------------------- #
    idx_j = jnp.asarray(indices)

    @jax.jit
    def ee_poses(q):
        fk = robot.forward_kinematics(q)
        se3 = jaxlie.SE3(fk[idx_j])
        return se3.translation(), se3.rotation().wxyz

    @jax.jit
    def min_clearance(q):
        return rc_sphere.compute_self_collision_distance(robot, q).min()

    # ---------------- trajectories --------------------------------------- #
    suite = T.build_suite(home_pos, home_wxyz) + build_collision_subset(
        home_pos, home_wxyz)
    by_name = {t.name: t for t in suite}

    # ---------------- viser scene ---------------------------------------- #
    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.update_cfg(q0)

    with server.gui.add_folder("Solver"):
        mode_dd = server.gui.add_dropdown(
            "mode", tuple(MODES.keys()), initial_value=list(MODES.keys())[0])
        backend_txt = server.gui.add_text(
            "backend", initial_value=str(jax.default_backend()), disabled=True)
        reset_btn = server.gui.add_button("reset to home")
    mode_md = server.gui.add_markdown(MODES[mode_dd.value]["blurb"])

    with server.gui.add_folder("Input"):
        source_dd = server.gui.add_dropdown(
            "source", ("trajectory", "manual gizmos"),
            initial_value="trajectory")
        traj_dd = server.gui.add_dropdown(
            "trajectory", tuple(by_name.keys()),
            initial_value=suite[0].name)
        playing = server.gui.add_checkbox("play", True)
        loop_cb = server.gui.add_checkbox("loop", True)
        frame_sl = server.gui.add_slider("frame", 0, len(suite[0]) - 1, 1, 0)
        speed_sl = server.gui.add_slider("speed ×", 0.1, 3.0, 0.1, 1.0)

    with server.gui.add_folder("Live metrics"):
        solve_txt = server.gui.add_text("solve ms (last / EMA)", "–", disabled=True)
        iters_txt = server.gui.add_text("LM iterations", "–", disabled=True)
        perr_txt = server.gui.add_text("pos err mm (L / R / mid)", "–", disabled=True)
        oerr_txt = server.gui.add_text("ori err ° (L / R / mid)", "–", disabled=True)
        clear_txt = server.gui.add_text("min sphere clearance mm", "–", disabled=True)

    # target frames (axes only — no marker spheres per user request)
    target_handles = [
        server.scene.add_frame(f"/targets/arm{a}", axes_length=0.07,
                               axes_radius=0.0035)
        for a in range(3)
    ]
    gizmos = [None, None, None]

    state = dict(q=q0.copy(), frame=0.0, ema=None,
                 targets_pos=home_pos.copy(), targets_wxyz=home_wxyz.copy())

    def current_ee():
        p, w = ee_poses(jnp.asarray(state["q"]))
        return np.asarray(p), np.asarray(w)

    def spawn_gizmos():
        p, w = current_ee()
        for a in range(3):
            if gizmos[a] is None:
                gizmos[a] = server.scene.add_transform_controls(
                    f"/gizmos/arm{a}", scale=0.15,
                    position=tuple(p[a]), wxyz=tuple(w[a]))
            else:
                gizmos[a].position = tuple(p[a])
                gizmos[a].wxyz = tuple(w[a])

    def remove_gizmos():
        for a in range(3):
            if gizmos[a] is not None:
                gizmos[a].remove()
                gizmos[a] = None

    @source_dd.on_update
    def _(_):
        if source_dd.value == "manual gizmos":
            spawn_gizmos()
        else:
            remove_gizmos()

    @mode_dd.on_update
    def _(_):
        mode_md.content = MODES[mode_dd.value]["blurb"]
        state["q"] = q0.copy()  # fresh start per solver, comparable runs
        state["frame"] = 0.0
        state["ema"] = None
        if source_dd.value == "manual gizmos":
            spawn_gizmos()

    @reset_btn.on_click
    def _(_):
        state["q"] = q0.copy()
        state["frame"] = 0.0
        if source_dd.value == "manual gizmos":
            spawn_gizmos()

    @traj_dd.on_update
    def _(_):
        state["frame"] = 0.0
        frame_sl.max = len(by_name[traj_dd.value]) - 1

    print(f"\n  IK playground:  http://localhost:{args.port}\n", flush=True)

    clear_every, tick = 5, 0
    last_wall = time.perf_counter()
    while True:
        now = time.perf_counter()
        dt_wall = now - last_wall
        last_wall = now

        # ---- targets from the active source --------------------------- #
        if source_dd.value == "trajectory":
            traj = by_name[traj_dd.value]
            if playing.value:
                state["frame"] += dt_wall * speed_sl.value / traj.dt
                if state["frame"] >= len(traj):
                    state["frame"] = 0.0 if loop_cb.value else len(traj) - 1
                frame_sl.value = int(state["frame"])
            else:
                state["frame"] = float(frame_sl.value)
            i = int(np.clip(state["frame"], 0, len(traj) - 1))
            state["targets_pos"] = traj.positions[i].copy()
            state["targets_wxyz"] = traj.wxyzs[i].copy()
        else:
            for a in range(3):
                if gizmos[a] is not None:
                    state["targets_pos"][a] = np.asarray(gizmos[a].position)
                    state["targets_wxyz"][a] = np.asarray(gizmos[a].wxyz)

        for a in range(3):
            target_handles[a].position = tuple(state["targets_pos"][a])
            target_handles[a].wxyz = tuple(state["targets_wxyz"][a])

        # ---- one solve of the ACTIVE solver --------------------------- #
        ik = solvers[mode_dd.value]
        res = ik.solve(state["q"], state["targets_pos"], state["targets_wxyz"])
        state["q"] = res.q
        urdf_vis.update_cfg(res.q)

        # ---- readouts -------------------------------------------------- #
        ema = res.solve_ms if state["ema"] is None else (
            0.95 * state["ema"] + 0.05 * res.solve_ms)
        state["ema"] = ema
        solve_txt.value = f"{res.solve_ms:6.2f} / {ema:6.2f}"
        iters_txt.value = str(res.iterations)
        p, w = current_ee()
        perr = np.linalg.norm(p - state["targets_pos"], axis=-1) * 1e3
        oerr = []
        for a in range(3):
            rel = (jaxlie.SO3(jnp.asarray(w[a])).inverse()
                   @ jaxlie.SO3(jnp.asarray(state["targets_wxyz"][a].astype(np.float32))))
            oerr.append(float(np.degrees(np.linalg.norm(np.asarray(rel.log())))))
        perr_txt.value = " / ".join(f"{v:6.1f}" for v in perr)
        oerr_txt.value = " / ".join(f"{v:6.1f}" for v in oerr)
        tick += 1
        if tick % clear_every == 0:
            clear_txt.value = f"{float(min_clearance(jnp.asarray(state['q']))) * 1e3:6.1f}"

        time.sleep(max(0.0, 0.02 - (time.perf_counter() - now)))


if __name__ == "__main__":
    main()
