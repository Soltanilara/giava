"""Interactive weight playground: drag targets, tune weights, watch the metrics.

Everything the benchmark measures is computed live here with the same code, so
what you see in the viewer is what the numbers meant.

  python viser_playground.py                       # starts with tuned weights
  python viser_playground.py --weights results/opt_best.json
  python viser_playground.py --preset pure         # start from a named preset

Panels
------
  Playback   : replay any benchmark workload so you can watch the scripted
               motions the metrics were computed from, or drag the targets
               yourself when paused.
  Weights    : every cost weight, live.  Changing one never recompiles.
  Controller : velocity / acceleration limits and filters.
  Readout    : live tracking error, smoothness, manipulability, clearance and
               solve time, plus a rolling window matching the benchmark's
               definitions.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
import time
from collections import deque
from typing import Optional

import numpy as np
import viser
from viser.extras import ViserUrdf
from yourdfpy import URDF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics as M  # noqa: E402
import workloads as W  # noqa: E402
from batched import vector_to_weights, weights_to_vector  # noqa: E402
from pyroki.collision import Sphere  # noqa: E402
from run_benchmark import TARGET_LINKS, load_robot  # noqa: E402
from solver import (  # noqa: E402
    PRESETS as PRESET_NAMES,
    ControllerConfig,
    ThreeArmIK,
    Weights,
    preset,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ARMS = ("left", "right", "middle")


@dataclasses.dataclass(frozen=True)
class SpeedProfile:
    """A named point on the tracking-vs-safety tradeoff.

    Measured on this robot: the profile choice barely moves solve time, so
    slowing down is close to free in latency terms -- what it costs is
    tracking lag. Margins grow *with* speed, because stopping distance does.
    """

    name: str
    velocity_limit: float      # rad/s
    accel_limit: float         # rad/s^2 -- the motor-protection knob
    smoothness: float
    collision_margin: float
    self_collision: float
    world_collision: float


CAUTIOUS = SpeedProfile("cautious", 2.0, 15.0, 2.0, 0.08, 20.0, 20.0)
FAST = SpeedProfile("fast", 6.0, 40.0, 0.5, 0.05, 20.0, 20.0)
SPEED_PROFILES = {p.name: p for p in (CAUTIOUS, FAST)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", default="/home/devi/giava/giava.urdf")
    ap.add_argument("--weights", default="results/opt_best.json",
                    help="JSON from optimize.py; falls back to --preset if missing")
    ap.add_argument("--preset", default="full")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument(
        "--max-iterations", type=int, default=8,
        help="Levenberg-Marquardt budget. Measured: 8 matches 100 exactly on "
             "warm-started tracking at less than half the wall-clock. Static "
             "(changing it recompiles), so it is a flag rather than a slider.",
    )
    ap.add_argument(
        "--hard-world-collision", action="store_true",
        help="Enforce world collision as a constraint instead of a soft cost. "
             "Static (recompiles), so it is a flag rather than a checkbox; "
             "restart to compare against the soft form.",
    )
    args = ap.parse_args()

    robot, robot_coll = load_robot(args.urdf, auto_ignore=True)
    urdf = URDF.load(args.urdf)
    structure, weights = preset(args.preset)
    if args.hard_world_collision:
        structure = dataclasses.replace(
            structure, world_collision=True, world_collision_hard=True
        )

    controller = ControllerConfig(dt=args.dt, velocity_limit=6.0, accel_limit=40.0)
    loaded_from = f"preset '{args.preset}'"
    # Resolve relative to this file, not the cwd: the old relative default meant
    # running from any other directory silently fell back to the preset.
    if args.weights and not os.path.isabs(args.weights):
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.weights)
        if os.path.exists(candidate):
            args.weights = candidate
    if args.weights and os.path.exists(args.weights):
        with open(args.weights) as f:
            blob = json.load(f)
        weights = vector_to_weights(np.asarray(blob["vector"], dtype=np.float32))
        c = blob.get("controller", {})
        controller = ControllerConfig(
            dt=c.get("dt", args.dt),
            velocity_limit=c.get("velocity_limit", 6.0),
            clamp_velocity=c.get("velocity_limit", 6.0) > 0,
            accel_limit=c.get("accel_limit", 40.0),
            nominal_velocity=c.get("nominal_velocity", 2.0),
        )
        loaded_from = args.weights
    print(f"loaded weights from {loaded_from}")

    ik = ThreeArmIK(
        robot, robot_coll, TARGET_LINKS, structure,
        max_iterations=args.max_iterations,
    )
    target_idx = ik.target_indices
    evaluator = M.StateEvaluator(robot, robot_coll, target_idx)
    q0 = np.asarray(robot.joint_var_cls(0).default_factory(), dtype=np.float32)
    ik.reset(q0)
    base_pos, base_wxyz = W.home_poses(robot, q0, target_idx)

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.update_cfg(q0)

    handles = [
        server.scene.add_transform_controls(
            f"/targets/{name}", scale=0.15,
            position=tuple(base_pos[i]), wxyz=tuple(base_wxyz[i]),
        )
        for i, name in enumerate(ARMS)
    ]
    obstacle_radius = 0.10
    obstacle_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.12, position=(0.35, 0.0, 0.35)
    )
    server.scene.add_icosphere("/obstacle/mesh", radius=obstacle_radius, color=(230, 120, 60))

    # ---------------- playback ---------------- #
    with server.gui.add_folder("Playback"):
        workload_dd = server.gui.add_dropdown(
            "Workload", tuple(W.DEFAULT_WORKLOADS), initial_value="smooth_track"
        )
        playing = server.gui.add_checkbox("Play workload", False)
        loop_cb = server.gui.add_checkbox("Loop", True)
        progress = server.gui.add_slider("Frame", 0, 1, 1, 0, disabled=True)
        reset_btn = server.gui.add_button("Reset robot to home")

    # ---------------- speed profile ---------------- #
    with server.gui.add_folder("Speed profile"):
        speed_btn = server.gui.add_button_group("Mode", ("cautious", "fast"))
        speed_label = server.gui.add_text("active", "custom", disabled=True)

    # ---------------- weight presets ---------------- #
    weight_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    source_options = tuple(
        [f"preset:{p}" for p in PRESET_NAMES]
        + [f"file:{os.path.basename(p)}" for p in weight_files]
    )
    with server.gui.add_folder("Weight source"):
        source_dd = server.gui.add_dropdown(
            "Source", source_options,
            initial_value=(
                f"file:{os.path.basename(args.weights)}"
                if any(s == f"file:{os.path.basename(args.weights)}" for s in source_options)
                else f"preset:{args.preset}"
            ),
        )
        load_btn = server.gui.add_button("Load into sliders")
        lock_lr = server.gui.add_checkbox("Lock left = right", False)
        save_name = server.gui.add_text("save as", "tuned.json")
        save_btn = server.gui.add_button("Save current weights")
        source_note = server.gui.add_text("note", "-", disabled=True)

    # ---------------- weights ---------------- #
    with server.gui.add_folder("Weights"):
        w_pos = [server.gui.add_slider(f"pos {a}", 0.0, 400.0, 1.0, float(weights.position[i]))
                 for i, a in enumerate(ARMS)]
        w_ori = [server.gui.add_slider(f"ori {a}", 0.0, 100.0, 0.5, float(weights.orientation[i]))
                 for i, a in enumerate(ARMS)]
        w_manip = [server.gui.add_slider(f"manip {a}", 0.0, 1.0, 0.001, float(weights.manipulability[i]))
                   for i, a in enumerate(ARMS)]
        w_smooth = server.gui.add_slider("smoothness", 0.0, 8.0, 0.01, float(weights.smoothness))
        w_rest = server.gui.add_slider("rest posture", 0.0, 1.0, 0.001, float(weights.rest))
        w_self = server.gui.add_slider("self collision", 0.0, 300.0, 0.5, float(weights.self_collision))
        w_world = server.gui.add_slider("world collision", 0.0, 300.0, 0.5, float(weights.world_collision))
        w_barrier = server.gui.add_slider("limit barrier", 0.0, 50.0, 0.05, float(weights.limit_barrier))
        w_center = server.gui.add_slider(
            "joint centering", 0.0, 1.0, 0.01, float(getattr(weights, "centering", 0.0))
        )
        m_coll = server.gui.add_slider("collision margin", 0.005, 0.10, 0.005, float(weights.collision_margin))
        m_barrier = server.gui.add_slider("barrier margin", 0.02, 0.40, 0.01, float(weights.limit_barrier_margin))

    # ---------------- controller ---------------- #
    with server.gui.add_folder("Controller"):
        c_clamp = server.gui.add_checkbox("Clamp velocity", controller.clamp_velocity)
        c_vel = server.gui.add_slider("velocity limit [rad/s]", 0.5, 12.0, 0.1, float(controller.velocity_limit) if np.isfinite(controller.velocity_limit) else 6.0)
        c_acc = server.gui.add_slider("accel limit [rad/s^2] (0=off)", 0.0, 300.0, 5.0, float(controller.accel_limit))
        c_tlpf = server.gui.add_slider("target filter", 0.0, 0.95, 0.05, float(controller.target_lpf))
        c_olpf = server.gui.add_slider("output filter", 0.0, 0.95, 0.05, float(controller.output_lpf))

    with server.gui.add_folder("Out-of-reach handling"):
        g_clutch = server.gui.add_checkbox("Clutch: freeze at boundary", False)
        g_freeze_mm = server.gui.add_slider("freeze if error > [mm]", 5.0, 200.0, 5.0, 30.0)
        g_reengage_mm = server.gui.add_slider("re-engage within [mm]", 10.0, 200.0, 5.0, 50.0)
        g_reseed = server.gui.add_checkbox("Re-seed: escape stuck pose", False)
        r_state = server.gui.add_text("state", "tracking", disabled=True)

    # ---------------- readout ---------------- #
    with server.gui.add_folder("Readout"):
        r_pos = server.gui.add_text("pos err L/R/M [mm]", "-", disabled=True)
        r_ori = server.gui.add_text("ori err L/R/M [deg]", "-", disabled=True)
        r_jerk = server.gui.add_text("EE jerk RMS (100 ticks)", "-", disabled=True)
        r_vel = server.gui.add_text("joint vel max / acc max", "-", disabled=True)
        r_manip = server.gui.add_text("manipulability min", "-", disabled=True)
        r_clear = server.gui.add_text("self / world clearance [m]", "-", disabled=True)
        r_limit = server.gui.add_text("limit margin min [rad]", "-", disabled=True)
        r_jumps = server.gui.add_text("config jumps (100 ticks)", "-", disabled=True)
        r_time = server.gui.add_text("solve [ms]", "-", disabled=True)

    def current_weights() -> Weights:
        pos = [s.value for s in w_pos]
        ori = [s.value for s in w_ori]
        man = [s.value for s in w_manip]
        if lock_lr.value:  # mirror left onto right every tick, not just on toggle
            pos[1], ori[1], man[1] = pos[0], ori[0], man[0]
        return Weights(
            position=tuple(pos),
            orientation=tuple(ori),
            manipulability=tuple(man),
            smoothness=w_smooth.value,
            rest=w_rest.value,
            self_collision=w_self.value,
            world_collision=w_world.value,
            limit_barrier=w_barrier.value,
            centering=w_center.value,
            collision_margin=m_coll.value,
            limit_barrier_margin=m_barrier.value,
        )

    def current_controller() -> ControllerConfig:
        return ControllerConfig(
            dt=args.dt,
            velocity_limit=c_vel.value,
            clamp_velocity=c_clamp.value,
            accel_limit=c_acc.value,
            nominal_velocity=controller.nominal_velocity,
            target_lpf=c_tlpf.value,
            output_lpf=c_olpf.value,
            freeze_error_m=(g_freeze_mm.value / 1e3) if g_clutch.value else 0.0,
            reengage_m=g_reengage_mm.value / 1e3,
            reseed_error_m=(g_freeze_mm.value / 1e3) if g_reseed.value else 0.0,
        )

    def apply_weights(new: Weights) -> None:
        for i in range(len(ARMS)):
            w_pos[i].value = float(new.position[i])
            w_ori[i].value = float(new.orientation[i])
            w_manip[i].value = float(new.manipulability[i])
        w_smooth.value = float(new.smoothness)
        w_rest.value = float(new.rest)
        w_self.value = float(new.self_collision)
        w_world.value = float(new.world_collision)
        w_barrier.value = float(new.limit_barrier)
        w_center.value = float(getattr(new, "centering", 0.0))
        m_coll.value = float(new.collision_margin)
        m_barrier.value = float(new.limit_barrier_margin)

    @load_btn.on_click
    def _(_event) -> None:
        kind, _, name = source_dd.value.partition(":")
        if kind == "preset":
            _structure, new = preset(name)
            source_note.value = f"preset '{name}' (structure unchanged)"
        else:
            with open(os.path.join(RESULTS_DIR, name)) as fh:
                new = vector_to_weights(
                    np.asarray(json.load(fh)["vector"], dtype=np.float32)
                )
            source_note.value = f"loaded {name}"
        apply_weights(new)

    @lock_lr.on_update
    def _(_event) -> None:
        # Nothing in the robot is left/right asymmetric; the asymmetry in the
        # optimizer output is a fit artifact, not a design choice. Mirroring
        # left onto right makes that explicit while tuning.
        if lock_lr.value:
            w_pos[1].value = w_pos[0].value
            w_ori[1].value = w_ori[0].value
            w_manip[1].value = w_manip[0].value

    @save_btn.on_click
    def _(_event) -> None:
        cur = current_weights()
        ctrl = current_controller()
        path = os.path.join(RESULTS_DIR, save_name.value or "tuned.json")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(
                {
                    "weights": cur.flat(),
                    "vector": weights_to_vector(cur).tolist(),
                    "controller": {
                        "dt": ctrl.dt,
                        "velocity_limit": ctrl.velocity_limit,
                        "accel_limit": ctrl.accel_limit,
                        "nominal_velocity": ctrl.nominal_velocity,
                    },
                    "objective": None,
                    "workloads": [],
                },
                fh,
                indent=2,
            )
        source_note.value = f"saved {os.path.basename(path)}"

    @speed_btn.on_click
    def _(_event) -> None:
        p = SPEED_PROFILES[speed_btn.value]
        c_vel.value = p.velocity_limit
        c_acc.value = p.accel_limit
        w_smooth.value = p.smoothness
        m_coll.value = p.collision_margin
        w_self.value = p.self_collision
        w_world.value = p.world_collision
        speed_label.value = p.name

    @reset_btn.on_click
    def _(_event) -> None:
        ik.reset(q0)
        for i, h in enumerate(handles):
            h.position = tuple(base_pos[i])
            h.wxyz = tuple(base_wxyz[i])

    # rolling buffers matching the benchmark's metric definitions
    hist_q: deque = deque(maxlen=100)
    hist_ee: deque = deque(maxlen=100)
    hist_ms: deque = deque(maxlen=50)

    workload: Optional[W.Workload] = None
    workload_name = ""
    frame = 0

    ik.warmup(current_weights(), current_controller())
    ik.reset(q0)
    print("viser ready -- open the printed URL")

    while True:
        tick_start = time.perf_counter()
        weights_now = current_weights()
        controller_now = current_controller()

        if playing.value:
            if workload is None or workload_name != workload_dd.value:
                workload = W.BUILDERS[workload_dd.value](
                    base_pos, base_wxyz, steps=300, dt=args.dt
                )
                workload_name = workload_dd.value
                frame = 0
                progress.max = len(workload) - 1
                ik.reset(q0)
            if frame >= len(workload):
                if loop_cb.value:
                    frame = 0
                else:
                    playing.value = False
                    frame = len(workload) - 1
            tp = np.asarray(workload.positions[frame])
            tw = np.asarray(workload.wxyzs[frame])
            obstacle = workload.obstacle_at(frame)
            progress.value = frame
            for i, h in enumerate(handles):  # mirror targets onto the gizmos
                h.position = tuple(tp[i])
                h.wxyz = tuple(tw[i])
            frame += 1
        else:
            workload = None
            tp = np.stack([np.asarray(h.position) for h in handles])
            tw = np.stack([np.asarray(h.wxyz) for h in handles])
            obstacle = Sphere.from_center_and_radius(
                np.asarray(obstacle_handle.position, dtype=np.float32),
                np.asarray([obstacle_radius], dtype=np.float32),
            )

        t0 = time.perf_counter()
        q = ik.step(tp, tw, weights_now, controller_now, world_obstacle=obstacle)
        solve_ms = (time.perf_counter() - t0) * 1e3
        hist_ms.append(solve_ms)
        urdf_vis.update_cfg(q)

        # --- live metrics, same definitions as the benchmark ---------------- #
        ee_pos, _ = evaluator.ee_poses(q)
        pos_err, ori_err = evaluator.pose_error(q, tp, tw)
        manip_t, _, _ = evaluator.manipulability(q)
        margin = evaluator.limit_margin(q)
        hist_q.append(q.copy())
        hist_ee.append(ee_pos.copy())

        r_pos.value = " / ".join(f"{e * 1e3:.1f}" for e in pos_err)
        r_ori.value = " / ".join(f"{e:.1f}" for e in ori_err)
        r_manip.value = f"{float(np.min(manip_t[:2])):.4f}"
        r_clear.value = (
            f"{evaluator.self_clearance(q):+.3f} / {evaluator.world_clearance(q, obstacle):+.3f}"
        )
        r_limit.value = f"{float(np.min(margin)):.3f}"
        r_time.value = f"{solve_ms:.1f} (avg {np.mean(hist_ms):.1f})"
        r_state.value = (
            "FROZEN (bring your hand back)" if getattr(ik, "frozen", False)
            else "RE-SEEDING" if getattr(ik, "last_reseeded", False)
            else "tracking"
        )

        if len(hist_ee) >= 4:
            ee_arr = np.asarray(hist_ee)
            d3 = np.diff(ee_arr, n=3, axis=0) / args.dt**3
            r_jerk.value = f"{float(np.sqrt(np.mean(np.sum(d3[:, :2] ** 2, axis=-1)))):.1f}"
        if len(hist_q) >= 4:
            q_arr = np.asarray(hist_q)[:, evaluator.arm_joint_mask]
            dq = np.diff(q_arr, axis=0) / args.dt
            ddq = np.diff(dq, axis=0) / args.dt
            r_vel.value = f"{np.max(np.abs(dq)):.2f} / {np.max(np.abs(ddq)):.1f}"
            step_norm = np.linalg.norm(np.diff(q_arr, axis=0), axis=1)
            thresh = max(5.0 * float(np.median(step_norm)), 0.15)
            r_jumps.value = f"{int(np.sum(step_norm > thresh))}"

        elapsed = time.perf_counter() - tick_start
        if elapsed < args.dt:
            time.sleep(args.dt - elapsed)


if __name__ == "__main__":
    main()
