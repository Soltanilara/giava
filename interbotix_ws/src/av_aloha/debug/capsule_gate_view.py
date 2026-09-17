"""Viser inspector for the inter-arm capsule gate  (port 8095).

    JAX_PLATFORMS=cpu python capsule_gate_view.py

Answers two questions visually, which is what `capsule_gate.py`'s numbers
cannot do on their own:

  1. **Do the capsules actually cover the arms?**  The robot's collision
     meshes and the fitted capsules are drawn together.  Every capsule
     contains its link's mesh by construction (longest-OBB-axis fit with
     exact vertex containment), so nothing should ever poke out.  Turn the
     opacity up and look along the long edges -- that is the fit this whole
     model rests on, and the reason the old pyroki min-cylinder fit was
     replaced (it oriented plate-like links along their THIN dimension and
     turned a 10 cm finger into a 13 cm fat disc).

  2. **Does the gate actually stop the arms meeting?**  Drive the joints and
     watch the live distance.  The gate's verdict, the offending pair, and a
     line drawn between the two closest capsule surfaces update every frame.

COLOURS follow clearance, not link identity, so the eye goes straight to
whatever is closest to trouble:

    grey    not in any gated pair (intra-arm links, under `inter` scope)
    green   clear by more than 2x the margin
    amber   inside 2x the margin -- approaching
    red     INSIDE the margin: the gate would refuse this command

WHAT "NOT COLLIDING" MEANS HERE
===============================
The capsules are CIRCUMSCRIBED, so capsule distance is a LOWER BOUND on the
true mesh distance.  Green therefore proves the meshes are apart.  Red does
NOT prove they touch -- it means separation could not be proven, which is
exactly the standard a safety gate should hold.  The readout shows both the
capsule distance and (on demand) the true mesh distance for the closest
pair, so the gap between "conservative" and "actual" is visible rather than
something you have to take on faith.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
## ik/ holds the solver modules; ik/study/ holds the research code this
## viewer borrows mesh_ground_truth.mesh_distance from (lazily, in
## true_mesh_distance below).
for _p in (str(_HERE), str(_HERE.parent / "ik"),
           str(_HERE.parent / "ik" / "study"),
           str(_HERE.parent / "calibration")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PORT = 8095  # 8082 teleop_debug, 8090 traj, 8091 collision, 8092 ik
             # playground, 8093 cameras, 8094 world_view


def segment_closest_points(p0, p1, q0, q1):
    """Closest points between two 3D segments -- pyroki's own routine.

    A capsule IS a segment swept by a sphere, so once the segment-to-segment
    closest points are known the surface points are those offset by each
    radius along the connecting direction, which is what gets drawn.

    This delegates to `pyroki.collision._utils.closest_segment_to_segment_points`
    rather than reimplementing it, and that is not just tidiness.  An
    independent implementation agreed with pyroki to 1.7 um on ordinary
    configurations but diverged by 8 mm on one DEEPLY PENETRATING pair
    (-74 mm), where the closest points are genuinely ambiguous and the two
    tie-breaks picked different valid answers.  A line drawn from one
    solution beside a distance computed from the other is a viewer that
    contradicts itself exactly when the geometry is most alarming.  Sharing
    the routine makes that impossible by construction."""
    import jax.numpy as jnp

    from pyroki.collision import _utils

    a, b = _utils.closest_segment_to_segment_points(
        jnp.asarray(p0), jnp.asarray(p1), jnp.asarray(q0), jnp.asarray(q1))
    return np.asarray(a, dtype=float), np.asarray(b, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--margin", type=float, default=None,
                    help="metres; default = capsule_gate's GATE_MARGIN")
    ap.add_argument("--scope", default="inter", choices=["inter", "all"])
    ap.add_argument("--from-robot", action="store_true",
                    help="read MEASURED joint angles from the live arms "
                         "(needs ROS) instead of using the sliders")
    ap.add_argument("--rate", type=float, default=8.0)
    ap.add_argument("--coarse", action="store_true",
                    help="show the coarse single-capsule model only -- what "
                         "the 50 Hz control loop uses by default")
    args = ap.parse_args()

    import jaxlie
    import viser
    from viser.extras import ViserUrdf

    import robot_model as rm
    from capsule_gate import GATE_MARGIN, CapsuleGate, _arm_of

    print("loading robot and fitting capsules...")
    robot, urdf = rm.load(with_urdf=True)
    margin = GATE_MARGIN if args.margin is None else args.margin
    ## The viewer always uses the FINE tier: it is not on a 20 ms clock,
    ## and showing the coarse model's reach would understate what the
    ## robot can actually do by ~30 mm.
    gate = CapsuleGate(robot, urdf, margin=margin, scope=args.scope,
                       fine=not args.coarse)
    print(gate.describe())

    link_meshes = gate.coll.get_link_collision_meshes()
    link_names = list(gate.coll.link_names)
    n_act = robot.joints.num_actuated_joints
    act_names = list(robot.joints.actuated_names)
    lo = np.asarray(robot.joints.lower_limits, dtype=float)
    hi = np.asarray(robot.joints.upper_limits, dtype=float)

    ## Named poses, converted DRIVER -> URDF through the same bridge the
    ## control loop uses, so "rest" here is the rest the operator knows.
    poses: Dict[str, np.ndarray] = {"urdf zero (self-colliding!)": np.zeros(n_act)}
    try:
        from kinematics import JointFrameBridge, RobotFrames
        frames = RobotFrames()
        bridge = JointFrameBridge(frames.robot)
        for nm in ("rest", "forward", "high", "low"):
            try:
                poses[nm] = np.asarray(
                    bridge.to_urdf(frames.q_from_named_pose(nm)), dtype=float)
            except (ValueError, KeyError):
                pass
    except Exception as e:  # noqa: BLE001
        print(f"  named poses unavailable ({e}); sliders only")
        frames = bridge = None

    q = np.asarray(poses.get("rest", np.zeros(n_act)), dtype=np.float32).copy()

    ## Live joints, so the numbers on screen describe the arms in the room
    ## rather than a pose someone typed. Same path world_view uses.
    listener = None
    if args.from_robot:
        from common import ensure_ros_path
        ensure_ros_path()
        import rospy

        ## A subscriber created before init_node never receives anything.
        ## Without this the viewer connected happily, reported "no
        ## joint_states", and drew the startup pose forever -- which looks
        ## exactly like the arms being stuck in an old configuration.
        ## Every other tool in this package does this; this one did not.
        ## anonymous so it can run beside data_collection.py, which owns
        ## the "data_collection" node name.
        rospy.init_node("giava_capsule_gate_view", anonymous=True,
                        disable_signals=True)
        from kinematics import JointStateListener

        ## The middle waist servo carries a Homing_Offset, and the
        ## driver->URDF bridge is WRONG without it -- the whole middle arm
        ## draws rotated by that amount, which is exactly the "middle arm
        ## did not load properly" failure this viewer shipped with.  Read
        ## the register off the live driver the same way robot_control
        ## does; a viewer must never guess a frame constant it can read.
        waist_shift = 0.0
        try:
            import rospy
            from interbotix_xs_msgs.srv import RegisterValues
            rospy.wait_for_service("/puppet_middle/get_motor_registers",
                                   timeout=5.0)
            srv = rospy.ServiceProxy("/puppet_middle/get_motor_registers",
                                     RegisterValues)
            vals = srv("single", "waist", "Homing_Offset", 0).values
            if vals:
                ## Ticks, 4096 per 2*pi -- and the register is a SIGNED
                ## 32-bit value delivered unsigned, so a negative offset
                ## arrives as ~4.29e9 and must be wrapped. Same handling
                ## as robot_control.read_middle_waist_shift, verbatim.
                ticks = int(vals[0])
                if ticks >= (1 << 31):
                    ticks -= 1 << 32
                waist_shift = ticks * 2.0 * np.pi / 4096.0
            print(f"  middle waist Homing_Offset: {waist_shift:+.4f} rad")
        except Exception as e:  # noqa: BLE001
            print(f"  could not read middle Homing_Offset ({e}) -- the "
                  f"middle arm may draw rotated at the waist")
        ## Rebuild the bridge with the measured shift.
        from kinematics import JointFrameBridge as _JFB
        bridge = _JFB(frames.robot, waist_driver_shift=waist_shift)

        if frames is None or bridge is None:
            raise SystemExit(
                "--from-robot needs the kinematics bridge, which failed to "
                "load above. Live joint angles cannot be converted to URDF "
                "coordinates without it, and drawing raw driver values "
                "would silently show the wrong pose.")
        listener = JointStateListener()
        print("  waiting for joint_states from the arms...")
        t_wait = time.time()
        while not listener.ready() and time.time() - t_wait < 15.0:
            time.sleep(0.2)
        missing = listener.missing()
        if missing:
            print(f"\n  *** NO joint_states from {missing} ***")
            print(f"  Those arms are NOT live -- they will sit at the startup")
            print(f"  pose, which looks identical to the arms being frozen.")
            print(f"  Is the interbotix driver (roslaunch) running?\n")
        else:
            print("  all arms reporting -- display is LIVE")

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=2.5, height=2.5, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    server.gui.configure_theme(control_layout="fixed", control_width="large",
                               dark_mode=True)
    status = server.gui.add_markdown("")
    tabs = server.gui.add_tab_group()

    LIVE, MANUAL = "LIVE arms (measured)", "manual / sliders"
    source_dd = None
    _echo = [False]      # True while the live loop is writing slider values

    with tabs.add_tab("Pose"):
        if args.from_robot:
            source_dd = server.gui.add_dropdown(
                "display source", (LIVE, MANUAL), initial_value=LIVE)
            server.gui.add_markdown(
                "_**LIVE** mirrors the measured joint angles and the sliders "
                "follow along read-only. Moving a slider switches to "
                "**manual** so you can explore poses; switch back to LIVE to "
                "re-attach to the arms. Nothing here ever commands the "
                "robot -- this viewer only observes._")
        pose_dd = server.gui.add_dropdown(
            "named pose", tuple(poses), initial_value=(
                "rest" if "rest" in poses else tuple(poses)[0]))
        server.gui.add_markdown(
            "_`urdf zero` is genuinely self-colliding on this robot: the "
            "left base is mounted at yaw = pi, so at all-zero joints both "
            "grippers reach the workspace centre and overlap. It is in the "
            "list precisely so you can see the gate refuse it._")
        converge = server.gui.add_slider(
            "CONVERGE left+right waists", -1.5, 1.5, 0.01, 0.0)
        server.gui.add_markdown(
            "_Turns both gripper arms toward each other symmetrically. "
            "Sweep it and watch the clearance fall and the gate trip -- the "
            "bimanual mistake this exists to stop._")
        reset_btn = server.gui.add_button("reset to named pose")

    ## The two finger joints of a gripper are not independently
    ## commandable on the real robot: one motor ("gripper") drives both
    ## through a rack-and-pinion, so any state with them apart is one the
    ## arm can never actually be in. Give each gripper ONE slider instead
    ## of two, wired to both finger joints -- everywhere below keys off
    ## `sliders` by joint NAME, so mapping both names to the same widget is
    ## enough to make live updates, pose loads and manual drags all move
    ## the fingers together with no other code aware anything changed.
    from kinematics import FINGER_JOINTS

    sliders: Dict[str, object] = {}
    for arm in ("left", "right", "middle"):
        idxs = [i for i, n in enumerate(act_names) if n.startswith(arm + "_")]
        if not idxs:
            continue
        finger_names = set(FINGER_JOINTS.get(arm, ()))
        finger_idxs = [i for i in idxs if act_names[i] in finger_names]
        arm_idxs = [i for i in idxs if act_names[i] not in finger_names]
        def _clamped_init(i: int) -> float:
            ## `q`'s starting value (0 from the zeros() fill for any joint
            ## a named pose doesn't cover -- currently just the fingers)
            ## is not guaranteed to sit inside THIS joint's declared
            ## <limit>. It used to: the finger lower limit was 0 until
            ## 2026-09-02. viser's add_slider asserts min <= value <= max
            ## and crashes the whole viewer at startup if it doesn't, so
            ## clamp defensively for every slider, not just the ones
            ## known to be affected today -- and write the clamped value
            ## back into `q` so the first FK draw matches what the
            ## slider actually shows instead of silently starting the
            ## robot at an out-of-limit configuration.
            v = float(np.clip(q[i], lo[i], hi[i]))
            q[i] = v
            return v

        with tabs.add_tab(arm):
            for i in arm_idxs:
                sliders[act_names[i]] = server.gui.add_slider(
                    act_names[i][len(arm) + 1:],
                    float(lo[i]), float(hi[i]), 0.005, _clamped_init(i))
            if len(finger_idxs) == 2:
                i0, i1 = finger_idxs
                gsl = server.gui.add_slider(
                    f"gripper (0 closed .. {float(hi[i0]) * 1000:.0f} open, mm)",
                    float(lo[i0]), float(hi[i0]), 0.001, _clamped_init(i0))
                q[i1] = q[i0]     # both fingers track one slider value
                sliders[act_names[i0]] = gsl
                sliders[act_names[i1]] = gsl
            elif finger_idxs:      # unexpected count -- fall back rather
                for i in finger_idxs:   # than silently dropping a joint
                    sliders[act_names[i]] = server.gui.add_slider(
                        act_names[i][len(arm) + 1:],
                        float(lo[i]), float(hi[i]), 0.005, _clamped_init(i))

    ## Saved experiment trajectories, so a claimed result can be REPLAYED
    ## rather than believed. Written by the gate experiments into
    ## ik/study/results/gate_experiments/*.npz.
    EXP_DIR = _HERE.parent / "ik" / "study" / "results" / "gate_experiments"
    traj: Dict[str, dict] = {}
    if EXP_DIR.exists():
        for f in sorted(EXP_DIR.glob("*.npz")):
            try:
                z = np.load(f)
                traj[f.stem] = {"q": np.asarray(z["q"], dtype=np.float32),
                                "dist": np.asarray(z.get("model_dist", [])),
                                "alpha": np.asarray(z.get("alpha", [])),
                                "margin": float(z["margin"]) if "margin" in z
                                else margin}
            except Exception as e:  # noqa: BLE001
                print(f"  could not read {f.name}: {e}")

    play_state = {"on": False, "traj": None}
    if traj:
        with tabs.add_tab("Experiments"):
            server.gui.add_markdown(
                "_Replays of the gate experiments. Each frame is a "
                "configuration the gate actually accepted while being "
                "driven straight at a collision -- so the clearance curve "
                "below is measured, not asserted._")
            traj_dd = server.gui.add_dropdown(
                "run", tuple(traj), initial_value=tuple(traj)[0])
            frame_sl = server.gui.add_slider(
                "frame", 0, max(len(traj[tuple(traj)[0]]["q"]) - 1, 1), 1, 0)
            playing = server.gui.add_checkbox("play", False)
            exp_md = server.gui.add_markdown("")
            plot_holder: List[object] = []

            def redraw_plot() -> None:
                for h in plot_holder:
                    h.remove()
                plot_holder.clear()
                t = traj[traj_dd.value]
                d = t["dist"]
                if d.size == 0:
                    return
                x = np.arange(d.size, dtype=np.float64)
                import viser.uplot as uplot
                plot_holder.append(server.gui.add_uplot(
                    data=(x, d * 1e3,
                          np.full(d.size, t["margin"] * 1e3)),
                    series=(
                        uplot.Series(label="tick"),
                        uplot.Series(label="clearance (mm)",
                                     stroke="#46c878", width=2),
                        uplot.Series(label="margin", stroke="#e83c3c",
                                     width=1, dash=(4, 4)),
                    ),
                    title="what the gate allowed, tick by tick",
                    height=180, aspect=1.6))

            def show_frame(_=None) -> None:
                t = traj[traj_dd.value]
                i = int(np.clip(frame_sl.value, 0, len(t["q"]) - 1))
                if source_dd is not None and source_dd.value == LIVE:
                    source_dd.value = MANUAL
                q[:] = t["q"][i]
                _echo[0] = True
                try:
                    for nm, sl in sliders.items():
                        sl.value = float(q[act_names.index(nm)])
                finally:
                    _echo[0] = False
                bits = [f"**{traj_dd.value}** frame {i}/{len(t['q']) - 1}"]
                if t["dist"].size > i - 1 >= 0:
                    bits.append(f"model clearance "
                                f"**{t['dist'][min(i, t['dist'].size - 1)] * 1e3:.1f} mm**")
                if t["alpha"].size > i - 1 >= 0:
                    a = t["alpha"][min(i, t["alpha"].size - 1)]
                    bits.append(f"step allowed **{a * 100:.0f}%**"
                                + (" — held" if a <= 0 else ""))
                exp_md.content = "  ·  ".join(bits)
                update()

            @traj_dd.on_update
            def _(_) -> None:
                frame_sl.max = max(len(traj[traj_dd.value]["q"]) - 1, 1)
                frame_sl.value = 0
                redraw_plot()
                show_frame()

            frame_sl.on_update(show_frame)

            @playing.on_update
            def _(_) -> None:
                play_state["on"] = bool(playing.value)

            play_state["frame_sl"] = frame_sl
            redraw_plot()

    with tabs.add_tab("Display"):
        show_robot = server.gui.add_checkbox("robot mesh", True)
        show_caps = server.gui.add_checkbox("coarse capsules", True)
        show_hulls = server.gui.add_checkbox("GJK convex hulls", True)
        opacity = server.gui.add_slider("capsule opacity", 0.05, 1.0, 0.05, 0.4)
        margin_sl = server.gui.add_slider(
            "gate margin (mm)", 0.0, 80.0, 1.0, float(margin * 1e3))
        server.gui.add_markdown(
            "_The margin has to absorb what happens BETWEEN checks: per-tick "
            "link travel plus servo tracking error (~10 mm steady-state + "
            "~10 mm compliance, measured 2026-08-18). Below ~15 mm you are "
            "trusting the servos to track perfectly._")
        n_pairs_sl = server.gui.add_slider("pairs listed", 3, 15, 1, 6)
        mesh_check = server.gui.add_checkbox(
            "verify against true mesh distance (slow)", False)
        server.gui.add_markdown(
            "_Computes the exact mesh-to-mesh distance for the closest pair. "
            "It must always be >= the capsule distance; that inequality is "
            "the entire safety argument._")

    ## The GJK hulls -- what the gate's verdict is ACTUALLY computed from.
    ## Seeing these next to the capsules is the whole point: the capsule is
    ## the fat thing that says "collision", the hulls are the real shape.
    hull_meshes: Dict[str, list] = {}
    if gate.fine is not None:
        import trimesh as _tm
        for _n, _e in gate.fine.items():
            hs = _e.get("hulls", [])
            if hs:
                hull_meshes[_n] = [_tm.Trimesh(vertices=np.asarray(h),
                                               process=False).convex_hull
                                   for h in hs]
    hull_handles: Dict[str, object] = {}

    caps_handles: Dict[str, object] = {}
    line_handle: List[object] = []
    mesh_cache: Dict[str, object] = {}

    def true_mesh_distance(pair, cfg) -> Optional[float]:
        from mesh_ground_truth import mesh_distance
        from pyroki.collision._robot_collision import RobotCollision as _RC
        fk = robot.forward_kinematics(cfg)
        out = []
        for nm in pair:
            if nm not in mesh_cache:
                m = _RC._get_trimesh_collision_geometries(urdf, nm)
                mesh_cache[nm] = None if m.is_empty else m
            base = mesh_cache[nm]
            if base is None:
                return None
            mm = base.copy()
            mm.apply_transform(np.asarray(
                jaxlie.SE3(fk[link_names.index(nm)]).as_matrix()))
            out.append(mm)
        return float(mesh_distance(out[0], out[1]))

    def update(_=None) -> None:
        gate.margin = float(margin_sl.value) / 1e3
        cfg = q.astype(np.float32)
        urdf_vis.update_cfg(cfg)

        d = np.asarray(gate._dists(cfg))
        order = np.argsort(d)
        worst_per_link: Dict[str, float] = {}
        for k, (a, b) in enumerate(gate.pair_names):
            for nm in (a, b):
                worst_per_link[nm] = min(worst_per_link.get(nm, np.inf),
                                         float(d[k]))

        ## Capsules, coloured by how close each link is to a gated contact.
        fk = robot.forward_kinematics(cfg)
        for i, name in enumerate(link_names):
            mesh = link_meshes.get(name)
            if mesh is None or mesh.is_empty:
                continue
            c = worst_per_link.get(name)
            if c is None:
                col = (120, 120, 128)
            elif c < gate.margin:
                col = (232, 60, 60)
            elif c < 2.0 * gate.margin:
                col = (240, 176, 40)
            else:
                col = (70, 200, 120)
            key = f"/capsules/{name}"
            if key in caps_handles:
                caps_handles[key].remove()
            h = server.scene.add_mesh_simple(
                key,
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                faces=np.asarray(mesh.faces, dtype=np.uint32),
                color=col, opacity=float(opacity.value),
                visible=bool(show_caps.value))
            se3 = jaxlie.SE3(fk[i])
            h.position = tuple(np.asarray(se3.translation(), dtype=float))
            h.wxyz = tuple(np.asarray(se3.rotation().wxyz, dtype=float))
            caps_handles[key] = h

        ## GJK hulls, drawn at the same link transforms.
        for name, parts in hull_meshes.items():
            li = link_names.index(name) if name in link_names else None
            if li is None:
                continue
            se3 = jaxlie.SE3(fk[li])
            pos = tuple(np.asarray(se3.translation(), dtype=float))
            wxyz = tuple(np.asarray(se3.rotation().wxyz, dtype=float))
            for pi, hm in enumerate(parts):
                key = f"/hulls/{name}/{pi}"
                if key in hull_handles:
                    hull_handles[key].remove()
                h = server.scene.add_mesh_simple(
                    key,
                    vertices=np.asarray(hm.vertices, dtype=np.float32),
                    faces=np.asarray(hm.faces, dtype=np.uint32),
                    color=(120, 190, 255), opacity=0.85,
                    visible=bool(show_hulls.value))
                h.position, h.wxyz = pos, wxyz
                hull_handles[key] = h

        ## The connecting line for the closest pair: capsule = segment swept
        ## by a sphere, so shrink each segment's closest point outward by its
        ## own radius to land on the actual surfaces.
        for lh in line_handle:
            lh.remove()
        line_handle.clear()
        k0 = int(order[0])
        i0, j0 = gate.coll.active_idx_i[k0], gate.coll.active_idx_j[k0]
        world = gate.coll.coll.transform(jaxlie.SE3(fk))
        pos = np.asarray(world.pose.translation())
        ax = np.asarray(world.axis)
        rad = np.asarray(world.radius)
        hgt = np.asarray(world.height)
        pa = pos[i0] - 0.5 * hgt[i0] * ax[i0], pos[i0] + 0.5 * hgt[i0] * ax[i0]
        pb = pos[j0] - 0.5 * hgt[j0] * ax[j0], pos[j0] + 0.5 * hgt[j0] * ax[j0]
        ca, cb = segment_closest_points(pa[0], pa[1], pb[0], pb[1])
        v = cb - ca
        nv = float(np.linalg.norm(v))
        if nv > 1e-9:
            u = v / nv
            sa, sb = ca + u * float(rad[i0]), cb - u * float(rad[j0])
            col = (232, 60, 60) if d[k0] < gate.margin else (255, 220, 60)
            line_handle.append(server.scene.add_spline_catmull_rom(
                "/closest", positions=np.stack([sa, sb]),
                color=col, line_width=4.0))
            for tag, pnt in (("a", sa), ("b", sb)):
                line_handle.append(server.scene.add_icosphere(
                    f"/closest_{tag}", radius=0.006, color=col,
                    position=tuple(pnt)))

        ## The gate's real verdict comes from GJK, not from `d` -- showing
        ## the coarse number alone is what made the capsules look like they
        ## were reporting collisions that were not there.
        v_ok, v_dist, v_pair = gate.check(cfg)
        ok = v_ok
        a0, b0 = (v_pair if v_pair is not None else gate.pair_names[k0])

        pm = gate._pair_margin
        rows = ["| coarse | GJK verdict | margin | pair |",
                "|---|---|---|---|"]
        mats1 = gate.link_matrices(cfg[None, :]) if gate.fine is not None else None
        for k in order[:int(n_pairs_sl.value)]:
            k = int(k)
            a, b = gate.pair_names[k]
            m = float(pm[k])
            if gate.fine is not None and d[k] < m:
                fd = gate._gjk_pairs(mats1, [(0, k)])[0]
                verdict = ("**CLEAR**" if fd >= m else "**too close**")
            elif d[k] >= m:
                verdict = "clear"
            else:
                verdict = "—"
            mark = "" if d[k] >= m else " ⚠"
            rows.append(f"| {d[k] * 1e3:+.1f} mm{mark} | {verdict} | "
                        f"{m * 1e3:.0f} mm | {a} ↔ {b} |")
        rows.append("")
        rows.append("_**coarse** is the fat capsule model — it reads tens of "
                    "mm pessimistic and a negative value there does NOT mean "
                    "contact. **GJK verdict** is what the gate actually "
                    "enforces, computed on the true convex hulls._")

        extra = ""
        if mesh_check.value:
            dm = true_mesh_distance((a0, b0), cfg)
            if dm is not None:
                slack = dm - float(d[k0])
                extra = (f"\n\nTrue mesh distance for that pair: "
                         f"**{dm * 1e3:+.1f} mm** — the capsule reads "
                         f"{slack * 1e3:.1f} mm closer, which is the "
                         f"conservatism the guarantee is bought with"
                         + ("" if slack >= -1e-3 else
                            "\n\n**⚠ capsule read FARTHER than the mesh — "
                            "the fit is not circumscribed!**"))

        status.content = (
            f"# {'✅ GATE PASS' if ok else '⛔ GATE HOLD'}\n\n"
            f"## {v_dist * 1e3:+.1f} mm  &nbsp; `{a0}` ↔ `{b0}`\n\n"
            f"margin **{gate.margin * 1e3:.0f} mm** · scope `{gate.scope}` · "
            f"{len(gate.pair_names)} pairs\n\n"
            + (f"source: **{source_dd.value}**"
               + ("" if source_dd.value != LIVE else
                  f" — {'all arms reporting' if listener is not None and not listener.missing() else 'NOT RECEIVING joint_states'}")
               + "\n\n" if source_dd is not None else
               "source: **static / sliders** (no `--from-robot`)\n\n")
            + ("_Capsules are circumscribed, so this proves the meshes are "
               "apart._" if ok else
               "_The command would be REFUSED and every arm would hold. "
               "Capsules are conservative, so this means separation could "
               "not be PROVEN — not that the meshes certainly touch._")
            + extra + "\n\n" + "\n".join(rows))

    def apply_sliders(_=None) -> None:
        ## Touching a slider takes manual control. Without this the live
        ## loop overwrote q from the arms 8 times a second and the sliders
        ## appeared to do nothing at all.
        ##
        ## `_echo` guards re-entrancy: the live loop writes sl.value to keep
        ## the sliders honest, and in viser that write fires on_update just
        ## like a human drag would. Unguarded, the viewer would flip itself
        ## to MANUAL one frame after connecting to the arms.
        if _echo[0]:
            return
        if source_dd is not None and source_dd.value == LIVE:
            source_dd.value = MANUAL
        for nm, sl in sliders.items():
            q[act_names.index(nm)] = float(sl.value)
        update()

    _bound = set()
    for sl in sliders.values():
        ## Both finger names share one slider object now; bind once.
        if id(sl) in _bound:
            continue
        _bound.add(id(sl))
        sl.on_update(apply_sliders)

    def load_pose(_=None) -> None:
        if source_dd is not None:
            source_dd.value = MANUAL     # a named pose is not the live arms
        q[:] = np.asarray(poses[pose_dd.value], dtype=np.float32)
        converge.value = 0.0
        _echo[0] = True
        try:
            for nm, sl in sliders.items():
                sl.value = float(q[act_names.index(nm)])
        finally:
            _echo[0] = False
        update()

    pose_dd.on_update(load_pose)
    reset_btn.on_click(load_pose)

    def do_converge(_=None) -> None:
        base = np.asarray(poses[pose_dd.value], dtype=np.float32)
        v = float(converge.value)
        for arm, sign in (("left", -1.0), ("right", +1.0)):
            nm = f"{arm}_waist"
            if nm in act_names:
                i = act_names.index(nm)
                q[i] = np.clip(base[i] + sign * v, lo[i], hi[i])
                if nm in sliders:
                    sliders[nm].value = float(q[i])
        update()

    converge.on_update(do_converge)
    for w in (opacity, margin_sl, n_pairs_sl, mesh_check):
        w.on_update(update)
    show_caps.on_update(lambda _: [setattr(h, "visible", bool(show_caps.value))
                                   for h in caps_handles.values()])
    show_hulls.on_update(lambda _: [setattr(h, "visible", bool(show_hulls.value))
                                    for h in hull_handles.values()])

    @show_robot.on_update
    def _(_) -> None:
        ## ViserUrdf owns its own visibility. The previous version called
        ## server.scene.add_frame("/robot", ...) to toggle it, which does
        ## not hide the robot -- it REPLACES the ViserUrdf root node with a
        ## bare frame, orphaning the mesh handles. update_cfg() then went on
        ## writing to dead nodes, so the arms silently froze at whatever
        ## pose was last rendered while everything drawn separately (the
        ## hulls, the capsules) kept tracking the real robot. That is the
        ## "mesh stuck in an old pose with a light-blue shadow at the real
        ## pose" failure.
        urdf_vis.show_visual = bool(show_robot.value)

    update()
    print(f"\n  Capsule gate inspector:  http://localhost:{args.port}")
    print("  (VS Code will offer to forward the port; or ssh -L "
          f"{args.port}:localhost:{args.port})\n")
    while True:
        if listener is not None and (source_dd is None
                                     or source_dd.value == LIVE):
            try:
                q_new = listener.q_driver(frames)
                q[:] = np.asarray(bridge.to_urdf(q_new), dtype=np.float32)
                ## Keep the sliders showing the truth, without letting their
                ## on_update fire and bounce us back into manual.
                _echo[0] = True
                try:
                    for nm, sl in sliders.items():
                        v = float(q[act_names.index(nm)])
                        if abs(float(sl.value) - v) > 1e-4:
                            sl.value = v
                finally:
                    _echo[0] = False
                update()
            except Exception as e:  # noqa: BLE001
                print(f"  [live] {type(e).__name__}: {e}")
            time.sleep(1.0 / max(args.rate, 0.5))
            continue
        if play_state.get("on") and "frame_sl" in play_state:
            fs = play_state["frame_sl"]
            fs.value = 0 if fs.value >= fs.max else fs.value + 1
            time.sleep(0.04)
        else:
            time.sleep(0.1)


if __name__ == "__main__":
    main()
