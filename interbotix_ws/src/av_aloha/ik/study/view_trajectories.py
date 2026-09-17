"""Phase 2 — Viser preview of the trajectory benchmark suite.

Purpose: visual sign-off of the *commanded targets* before any quantitative
benchmarking.  What is drawn is the trajectory definition itself — target
pose triads and paths — independent of any solver.  An optional checkbox
animates the robot along the precomputed **baseline** IK solution as a sanity
preview (clearly a preview: the benchmark in Phase 3 does the measuring).

Run:  JAX_PLATFORMS=cpu python view_trajectories.py [--port 8090]
Then open http://localhost:8090 in a browser.

Controls: trajectory dropdown · play/pause · frame slider · speed ·
"animate baseline IK" checkbox.  The info panel shows each trajectory's
definition, purpose, expected motion, and failure modes.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import viser
from viser.extras import ViserUrdf

import robot_model as rm
import trajectories as T
from baseline import BaselineIK

ARM_COLORS = {
    0: (230, 120, 40),  # left  — orange
    1: (60, 120, 230),  # right — blue
    2: (60, 180, 90),  # middle — green
}


def path_segments(positions: np.ndarray, arm: int) -> np.ndarray:
    """(T,3,3) target positions -> (T-1, 2, 3) line segments for one arm."""
    p = positions[:, arm, :]
    return np.stack([p[:-1], p[1:]], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    robot, urdf = rm.load(with_urdf=True)
    q0 = rm.home_config(robot)
    home_pos, home_wxyz = rm.home_poses(robot)
    suite = T.build_suite(home_pos, home_wxyz)
    by_name = {t.name: t for t in suite}

    # Baseline IK for the optional robot-motion preview (lazy, cached).
    ik = BaselineIK(robot, rm.TARGET_LINKS)
    ik.warmup(q0)
    ik_cache: dict = {}

    def baseline_rollout(traj: T.Trajectory) -> np.ndarray:
        if traj.name not in ik_cache:
            q = q0.copy()
            qs = []
            for i in range(len(traj)):
                q = ik.solve(q, traj.positions[i], traj.wxyzs[i]).q
                qs.append(q)
            ik_cache[traj.name] = np.stack(qs)
        return ik_cache[traj.name]

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.update_cfg(q0)

    with server.gui.add_folder("Trajectory"):
        traj_dd = server.gui.add_dropdown(
            "select", tuple(by_name.keys()), initial_value=suite[0].name
        )
        playing = server.gui.add_checkbox("Play", True)
        loop_cb = server.gui.add_checkbox("Loop", True)
        frame_sl = server.gui.add_slider("Frame", 0, len(suite[0]) - 1, 1, 0)
        speed_sl = server.gui.add_slider("Speed ×", 0.1, 3.0, 0.1, 1.0)
        animate_cb = server.gui.add_checkbox("Animate baseline IK (preview)", False)

    info_md = server.gui.add_markdown("")

    state = {"traj": suite[0], "frame": 0, "dirty": True}

    def describe(t: T.Trajectory) -> str:
        return (
            f"### {t.name}  \n"
            f"*category {t.category} — {t.duration:.0f} s, {len(t)} steps @ "
            f"{1/t.dt:.0f} Hz*  \n\n"
            f"**What:** {t.description}  \n\n"
            f"**Definition:** {t.definition}  \n\n"
            f"**Purpose:** {t.purpose}  \n\n"
            f"**Expected motion:** {t.expected_motion}  \n\n"
            f"**Failure modes:** {t.failure_modes}  \n\n"
            f"**Feasibility:** {t.feasible}"
        )

    def rebuild_scene() -> None:
        t = state["traj"]
        # target paths
        for arm in range(3):
            moved = np.abs(t.positions[:, arm] - t.positions[0, arm]).max() > 1e-6
            if moved:
                server.scene.add_line_segments(
                    f"/paths/arm{arm}",
                    path_segments(t.positions, arm),
                    colors=ARM_COLORS[arm],
                    line_width=3.0,
                )
            else:
                server.scene.add_line_segments(
                    f"/paths/arm{arm}",
                    np.zeros((0, 2, 3), dtype=np.float32),
                    colors=ARM_COLORS[arm],
                )
        # markers (objects, gaze center)
        server.scene.add_frame("/markers", show_axes=False)
        for i, (name, p) in enumerate(t.markers.items()):
            server.scene.add_icosphere(
                f"/markers/{name}",
                radius=0.02,
                color=(200, 60, 160),
                position=tuple(np.asarray(p, dtype=float)),
            )
        info_md.content = describe(t)
        frame_sl.max = len(t) - 1

    def update_frame() -> None:
        t = state["traj"]
        i = int(np.clip(state["frame"], 0, len(t) - 1))
        for arm in range(3):
            server.scene.add_frame(
                f"/targets/arm{arm}",
                wxyz=tuple(t.wxyzs[i, arm]),
                position=tuple(t.positions[i, arm]),
                axes_length=0.07,
                axes_radius=0.0035,
            )
        if animate_cb.value:
            urdf_vis.update_cfg(baseline_rollout(t)[i])

    @traj_dd.on_update
    def _(_) -> None:
        state["traj"] = by_name[traj_dd.value]
        state["frame"] = 0
        state["dirty"] = True

    @animate_cb.on_update
    def _(_) -> None:
        if not animate_cb.value:
            urdf_vis.update_cfg(q0)

    @frame_sl.on_update
    def _(_) -> None:
        if not playing.value:  # manual scrub
            state["frame"] = int(frame_sl.value)
            update_frame()

    print(f"\n  Viser trajectory preview:  http://localhost:{args.port}\n")
    last = time.perf_counter()
    acc = 0.0
    while True:
        if state["dirty"]:
            rebuild_scene()
            update_frame()
            state["dirty"] = False
        now = time.perf_counter()
        elapsed, last = now - last, now
        if playing.value:
            t = state["traj"]
            acc += elapsed * speed_sl.value / t.dt
            if acc >= 1.0:
                state["frame"] += int(acc)
                acc = 0.0
                if state["frame"] >= len(t):
                    state["frame"] = 0 if loop_cb.value else len(t) - 1
                frame_sl.value = int(state["frame"])
                update_frame()
        time.sleep(0.005)


if __name__ == "__main__":
    main()
