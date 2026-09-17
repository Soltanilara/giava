"""Viser world view: robot, frames, cameras, and jog  (PHASE 8, 3D half).

Shows the world coordinate system with the robot in it, so the numbers this
package prints can be checked against the physical rig:

  * the world origin and its axes (giava.urdf's root link `base`)
  * the robot at its measured joint state, or an offline configuration
  * each arm's end-effector frame from forward kinematics, with a live
    numeric readout
  * each camera's pose, composed as T_world_camera = T_world_ee @ T_ee_camera,
    drawn as a frustum -- and labelled with whether that mount transform is
    CALIBRATED or merely nominal
  * jog buttons that move one arm a fixed step along a world axis, so a
    commanded 1 cm can be measured on the real arm with a ruler

Offline (no robot, no ROS):

    python calibration/world_view.py

Against the live robot, reading measured joints and jogging them:

    python calibration/world_view.py --from-robot --arms right

Then open http://localhost:8094 (forward the port over SSH / remote VS Code).
Port 8094 continues the repo block: 8082 teleop_debug_tool, 8090
view_trajectories, 8091 view_collision, 8092 ik playground, 8093 the
two-camera viewer.

NOTE ON JOG: it commands the deployed CoupledStudyIK, the same solver data
collection runs, and clamps every command exactly as data_collection does.
Pose tracking is a weighted soft cost, so a commanded 10 mm typically comes
out slightly short -- the readout shows commanded, IK-predicted and measured
displacement side by side for exactly that reason.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ is a SIBLING of this package now, not the
## parent it used to be.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
for _p in (str(HERE), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    ensure_ros_path,
    invert_T,
    format_T,
    matrix_to_quat_wxyz,
)
from kinematics import ARM_ORDER, WORLD_FRAME, RobotFrames  # noqa: E402
from rs_camera import PRODUCTION_COLOR  # noqa: E402
import camera_mount as CM  # noqa: E402
import tcp as TCP  # noqa: E402
import workspace as WS  # noqa: E402
from estop import ESTOP  # noqa: E402
from arm_config import ARM_CONFIG, POSES, URDF_PATH  # noqa: E402

AXES = {"+x": np.array([1.0, 0, 0]), "-x": np.array([-1.0, 0, 0]),
        "+y": np.array([0, 1.0, 0]), "-y": np.array([0, -1.0, 0]),
        "+z": np.array([0, 0, 1.0]), "-z": np.array([0, 0, -1.0])}

ARM_COLOR = {"left": (80, 170, 255), "right": (255, 140, 90),
             "middle": (150, 255, 140)}

## The repo's existing custom-pose store: teleop_debug_tool.py writes it and
## make_middle_offsets.py reads it.  Saving here keeps one format.
POSES_CUSTOM = SCRIPTS_DIR / "poses_custom.json"


class WorldView:
    def __init__(self, args):
        self.args = args
        self.frames = RobotFrames()
        self.arms: List[str] = list(args.arms)
        self.live = bool(args.from_robot)
        self.sessions: Dict[str, Any] = {}
        self.bridge = None
        self.q_driver = None
        self.jog_log: List[Dict[str, Any]] = []

        if self.live:
            self._connect()
        else:
            from kinematics import JointFrameBridge
            self.bridge = JointFrameBridge(self.frames.robot)
            self.q_driver = self.bridge.to_driver(self.frames.home_q())

    # -------------------------------------------------------------- #
    def _connect(self) -> None:
        """Commandable sessions for --arms, READ-ONLY listeners for the rest.

        An arm that is not being commanded still has to be DRAWN at its real
        pose.  Giving it a MotionSession would torque it on for no reason;
        leaving it out entirely would draw it at driver zero, which is a
        pose it is not in -- the display would silently lie about where the
        robot is.  So the others get a plain joint_states subscriber: real
        pose, no torque, no commands possible."""
        import rospy
        from sensor_msgs.msg import JointState
        from move_validation import MotionSession

        for arm in self.arms:
            self.sessions[arm] = MotionSession(arm, dry_run=False)
        any_s = next(iter(self.sessions.values()))
        self.bridge = any_s.bridge

        self._listen: Dict[str, Any] = {}
        for arm in ARM_ORDER:
            if arm in self.sessions:
                continue
            topic = f"/{ARM_CONFIG[arm]['robot_name']}/joint_states"
            self._listen[arm] = {"msg": None, "topic": topic}

            def _cb(msg, _arm=arm):
                self._listen[_arm]["msg"] = msg

            rospy.Subscriber(topic, JointState, _cb, queue_size=1)
            print(f"  listening (read-only, no torque) to {topic}")

        rospy.sleep(0.7)
        self.q_driver = np.zeros(self.frames.num_actuated, dtype=np.float64)
        self.read_q_driver()

    def arm_is_live(self, arm: str) -> bool:
        """True when this arm's drawn pose comes from real joint data."""
        if not self.live:
            return False
        if arm in self.sessions:
            return True
        return self._listen.get(arm, {}).get("msg") is not None

    def read_q_driver(self) -> np.ndarray:
        """Full DRIVER joint vector, live or offline."""
        if not self.live:
            return self.q_driver
        from kinematics import fill_fingers
        q = self.q_driver.copy()
        for arm, sess in self.sessions.items():
            n = ARM_CONFIG[arm]["num_joints"]
            js = sess.robots[arm].dxl.joint_states
            q[sess.joint_idx] = np.asarray(js.position[:n], dtype=np.float64)
            ## Fingers come off the driver's own gripper reading. Left at
            ## zero they draw as a clamped-shut hand, which is a pose the
            ## arm is not in -- the same class of silent lie as drawing a
            ## non-commanded arm at driver zero, which _connect() already
            ## refuses to do.
            fill_fingers(q, self.frames, arm, js)
        for arm, entry in getattr(self, "_listen", {}).items():
            msg = entry.get("msg")
            if msg is None:
                continue  # nothing heard yet -- flagged in the UI
            n = ARM_CONFIG[arm]["num_joints"]
            pos = np.asarray(msg.position[:n], dtype=np.float64)
            if pos.shape[0] == n:
                q[self.frames.joint_indices(arm)] = pos
            fill_fingers(q, self.frames, arm, msg)
        self.q_driver = q
        return q

    def q_urdf(self) -> np.ndarray:
        return self.bridge.to_urdf(self.read_q_driver())

    # -------------------------------------------------------------- #
    def pose_book(self) -> Dict[str, Dict[str, List[float]]]:
        """Named poses: arm_config.POSES plus anything saved by hand.

        poses_custom.json is the repo's existing store (teleop_debug_tool
        writes it, make_middle_offsets reads it), so custom poses saved here
        show up in those tools too rather than in a private format."""
        import json
        book = {a: {k: list(np.asarray(v, float))
                    for k, v in POSES.get(a, {}).items()}
                for a in ARM_ORDER}
        if POSES_CUSTOM.exists():
            try:
                for arm, entries in json.loads(POSES_CUSTOM.read_text()).items():
                    if arm in book and isinstance(entries, dict):
                        for name, q in entries.items():
                            book[arm][f"custom:{name}"] = list(q)
            except Exception as exc:
                print(f"  could not read {POSES_CUSTOM}: {exc}")
        return book

    def go_to_pose(self, arm: str, pose_name: str) -> Dict[str, Any]:
        """Move one arm to a named joint configuration, envelope-checked."""
        if not self.live:
            return {"error": "moving to a pose needs --from-robot"}
        if ESTOP.triggered:
            return {"error": "emergency stop was triggered -- restart first"}
        sess = self.sessions.get(arm)
        if sess is None:
            return {"error": f"{arm} was not started (see --arms)"}

        entry = self.pose_book().get(arm, {}).get(pose_name)
        if entry is None:
            return {"error": f"no pose '{pose_name}' for {arm}"}
        target_arm = np.asarray(entry, dtype=float)

        q_now = sess.read_q_driver()
        q_target = q_now.copy()
        if len(target_arm) != len(sess.joint_idx):
            return {"error": (f"pose '{pose_name}' has {len(target_arm)} "
                              f"joints, {arm} has {len(sess.joint_idx)}")}
        q_target[sess.joint_idx] = target_arm

        ## Check the whole interpolated path, not just the endpoint -- the
        ## arm sweeps through everything in between, and a named pose can
        ## easily route through a bar.
        steps = max(1, int(np.ceil(
            np.max(np.abs(q_target - q_now)) / self.args.max_step)))
        path = [q_now + (q_target - q_now) * (k / steps)
                for k in range(1, steps + 1)]
        _, env_bad, pts = WS.check_configs(self.frames, self.bridge,
                                           [q_now] + path, arm)
        if env_bad:
            return {"error": ("refused: the path leaves the measured free "
                              "space (" + ", ".join(b[0] for b in env_bad)
                              + ")")}

        from robot_control import interpolate_to_pose
        ## interpolate_to_pose carries the wrapped-encoder guard and the
        ## middle waist's nearest-2pi handling; do not reimplement them.
        interpolate_to_pose(sess.robots[arm], arm, target_arm.tolist(),
                            moving_time=self.args.moving_time,
                            accel_time=0.5 * self.args.moving_time)
        sess.settle(self.args.settle)
        T = sess.ee_pose(sess.read_q_driver())
        return {"arm": arm, "pose": pose_name,
                "ee_xyz_m": T[:3, 3].tolist(),
                "n_waypoints": steps,
                "envelope": WS.footprint(pts)}

    def save_pose(self, arm: str, name: str) -> Dict[str, Any]:
        """Record the arm's CURRENT measured joints under a name.

        Writes poses_custom.json (the repo's existing store) and returns a
        paste-ready arm_config.py constant, since that is where the poses
        that matter eventually live."""
        import json

        if not self.live:
            return {"error": "saving a pose needs --from-robot"}
        sess = self.sessions.get(arm)
        if sess is None:
            return {"error": f"{arm} was not started (see --arms)"}
        name = (name or "").strip()
        if not name:
            return {"error": "give the pose a name first"}

        q = sess.read_q_driver()[sess.joint_idx]
        book = {}
        if POSES_CUSTOM.exists():
            try:
                book = json.loads(POSES_CUSTOM.read_text())
            except Exception:
                book = {}
        book.setdefault(arm, {})[name] = [float(v) for v in q]
        POSES_CUSTOM.write_text(json.dumps(book, indent=2))

        const = f"{arm.upper()}_{name.upper().replace(' ', '_')}"
        snippet = (f"{const} = np.array({np.round(q, 6).tolist()}, "
                   f"dtype=float)")
        print(f"[pose] saved {arm}:{name}")
        print(f"[pose] {snippet}")
        return {"arm": arm, "pose": name,
                "q_driver": [float(v) for v in q],
                "file": str(POSES_CUSTOM),
                "arm_config_snippet": snippet,
                "arm_config_hint": (
                    f'add to POSES["{arm}"] as  "{name}": {const}')}

    # -------------------------------------------------------------- #
    def offset_for(self, arm: str, point: str):
        """T_flange_point for the point currently being commanded.

        Computed at call time rather than baked into the session, so the
        commanded point can be switched from the GUI without rebuilding the
        solver.  Returns (offset_or_None, label, note)."""
        if point == "flange":
            return None, "flange", "the gripper mounting plate"
        if point == "tcp":
            t = TCP.resolve(arm)
            if t["T_flange_tcp"] is None:
                return None, "flange", f"{arm} has no TCP; using the flange"
            return (t["T_flange_tcp"], "tcp",
                    f"grasp point [{t['provenance']}, "
                    f"measured={t['measured']}]")
        # a camera name
        m = CM.resolve_mount(point)
        if m["T_parent_optical"] is None:
            return None, "flange", (f"{point} has no mount transform "
                                    f"({m['provenance']}); using the flange")
        if m["rigid_to"] != ARM_CONFIG[arm]["ee_link"]:
            return None, "flange", (f"{point} is not mounted on {arm} "
                                    f"(it is rigid to {m['rigid_to']})")
        return (m["T_parent_optical"], point,
                f"camera optical centre [{m['provenance']}, "
                f"validated={m['validated']}]")

    def _pose_of(self, sess, q, offset):
        T = sess.ee_pose(q)
        return T if offset is None else T @ offset

    def _solve_for(self, sess, q_prev, T_point_target, offset):
        """IK toward a target expressed for the commanded point."""
        T = (T_point_target if offset is None
             else T_point_target @ invert_T(offset))
        return sess.solve(q_prev, T)

    # -------------------------------------------------------------- #
    def jog_rotate(self, arm: str, rot_axis: str, deg: float,
                   frame: str = "ee", point: str = "flange") -> Dict[str, Any]:
        """Rotate the commanded point in place by `deg` about an axis.

        frame='ee'    roll/pitch/yaw about the TOOL's own axes -- what
                      "pitch up 10 degrees" usually means.
        frame='world' about the world axes: yaw about +z (turn left/right),
                      pitch about +x (the left-right axis, so the tool tips
                      forward/back), roll about +y.

        The POSITION of the commanded point is held fixed, so this is a
        pivot rather than an arc.  With --tcp that pivot is the grasp
        point, which is usually what you want: the fingers stay put and the
        wrist swings around them.  Without it the flange stays put and the
        fingers sweep on the 72 mm lever."""
        if not self.live:
            return {"error": "rotation jog needs --from-robot"}
        if ESTOP.triggered:
            return {"error": "emergency stop was triggered -- restart first"}
        sess = self.sessions.get(arm)
        if sess is None:
            return {"error": f"{arm} was not started (see --arms)"}

        from handeye import so3_exp

        local = {"roll": np.array([1.0, 0, 0]),
                 "pitch": np.array([0, 1.0, 0]),
                 "yaw": np.array([0, 0, 1.0])}[rot_axis]
        ang = np.radians(deg)

        offset, label, note = self.offset_for(arm, point)
        q0 = sess.read_q_driver()
        T0 = self._pose_of(sess, q0, offset)

        ## Substeps so the solver never has to make a large orientation
        ## jump, and so the envelope check sees the whole sweep.
        n = max(1, int(np.ceil(abs(deg) / 3.0)))
        plan, q = [], q0.copy()
        for k in range(1, n + 1):
            dR = so3_exp(local * (ang * k / n))
            T = T0.copy()
            # pre-multiply = rotate about WORLD axes; post-multiply =
            # about the commanded point's own axes.
            T[:3, :3] = (dR @ T0[:3, :3]) if frame == "world" else (T0[:3, :3] @ dR)
            q = self._solve_for(sess, q, T, offset)
            plan.append(q.copy())

        _, env_bad, _ = WS.check_configs(sess.frames, sess.bridge,
                                         [q0] + plan, arm)
        if env_bad:
            return {"error": ("refused: leaves the measured free space ("
                              + ", ".join(b[0] for b in env_bad) + ")")}

        for qq in plan:
            sess.command(qq, self.args.moving_time, self.args.max_step,
                         self.args.limit_margin)
            time.sleep(self.args.moving_time)
        T_ik = self._pose_of(sess, plan[-1], offset)
        sess.settle(self.args.settle)
        T1 = self._pose_of(sess, sess.read_q_driver(), offset)

        from common import rotation_angle_deg
        ## Split the drift: what the SOLVER gave up, and what the HARDWARE
        ## then added.  Pose tracking is a soft cost (pos 50 / ori 10), so
        ## the solver may itself sacrifice position to hit an orientation --
        ## that is a tuning question.  Anything beyond it is servo error and
        ## gravity, amplified by the arm's reach, which is a hardware
        ## question.  One number cannot tell you which you are looking at.
        ik_drift = (T_ik[:3, 3] - T0[:3, 3]) * 1e3
        meas_drift = (T1[:3, 3] - T0[:3, 3]) * 1e3
        entry = {
            "arm": arm, "rotation_axis": rot_axis, "frame": frame,
            "commanded_deg": deg,
            "ik_predicted_deg": rotation_angle_deg(T0[:3, :3].T @ T_ik[:3, :3]),
            "measured_deg": rotation_angle_deg(T0[:3, :3].T @ T1[:3, :3]),
            "ik_predicted_drift_mm": ik_drift.tolist(),
            "point_drift_mm": meas_drift.tolist(),
            "hardware_drift_mm": (meas_drift - ik_drift).tolist(),
            "commanded_point": label,
            "point_note": note,
        }
        self.jog_log.append(entry)
        return entry

    # -------------------------------------------------------------- #
    def jog(self, arm: str, axis: str, step_m: float,
            point: str = "flange") -> Dict[str, Any]:
        """Move one arm `step_m` along a WORLD axis and report the result."""
        if not self.live:
            return {"error": "jog needs --from-robot"}
        if ESTOP.triggered:
            return {"error": ("emergency stop was triggered -- restart the "
                              "tool before commanding more motion")}
        sess = self.sessions[arm]
        offset, label, note = self.offset_for(arm, point)
        q0 = sess.read_q_driver()
        T0 = self._pose_of(sess, q0, offset)
        delta = AXES[axis] * step_m

        ## Plan the whole jog, then check where the arm would go BEFORE
        ## committing to the first command.  The IK collision model is
        ## self-collision only and knows nothing about the table or the
        ## frame bars, so the environment box is a separate check.
        q = q0.copy()
        steps = max(1, int(np.ceil(step_m / 0.005)))   # <=5 mm per command
        plan = []
        for k in range(1, steps + 1):
            Tk = T0.copy()
            Tk[:3, 3] = T0[:3, 3] + delta * (k / steps)
            q = self._solve_for(sess, q, Tk, offset)
            plan.append(q.copy())
        _, env_bad, _ = WS.check_configs(sess.frames, sess.bridge,
                                         [q0] + plan, arm)
        if env_bad:
            return {"error": ("refused: leaves the measured free space ("
                              + ", ".join(b[0] for b in env_bad) + ")")}

        for q in plan:
            sess.command(q, self.args.moving_time, self.args.max_step,
                         self.args.limit_margin)
            time.sleep(self.args.moving_time)
        T_ik = self._pose_of(sess, plan[-1], offset)
        sess.settle(self.args.settle)
        T1 = self._pose_of(sess, sess.read_q_driver(), offset)

        entry = {
            "arm": arm, "axis": axis, "step_m": step_m,
            "commanded_point": label, "point_note": note,
            "commanded_mm": (delta * 1e3).tolist(),
            "ik_predicted_mm": ((T_ik[:3, 3] - T0[:3, 3]) * 1e3).tolist(),
            "measured_mm": ((T1[:3, 3] - T0[:3, 3]) * 1e3).tolist(),
            "start_xyz_m": T0[:3, 3].tolist(),
            "end_xyz_m": T1[:3, 3].tolist(),
        }
        self.jog_log.append(entry)
        return entry


## ------------------------------------------------------------------ ##

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--arms", nargs="+", default=list(ARM_ORDER),
                    choices=list(ARM_ORDER))
    ap.add_argument("--from-robot", action="store_true",
                    help="read measured joints and enable jog (needs ROS)")
    ap.add_argument("--jog-step", type=float, default=0.01,
                    help="jog distance in METRES (default 0.01 = 1 cm)")
    ap.add_argument("--rate", type=float, default=10.0,
                    help="refresh rate Hz")
    ## The capture path reports achieved rate against the configured one,
    ## so these must exist as args rather than being hardcoded at the
    ## CameraWorker call site.
    ap.add_argument("--width", type=int, default=PRODUCTION_COLOR[0])
    ap.add_argument("--height", type=int, default=PRODUCTION_COLOR[1])
    ap.add_argument("--fps", type=int, default=PRODUCTION_COLOR[2])
    ap.add_argument("--cameras", nargs="*", default=None,
                    help="also show these cameras' LIVE images in the same "
                         "view, e.g. --cameras left_wrist top_scene "
                         "low_scene. Omit for frames-only (no camera "
                         "hardware needed). With --measure and no explicit "
                         "list, defaults to the two static scene cameras.")
    ap.add_argument("--measure", action="store_true",
                    help="add the Measure tab: clickable camera image "
                         "planes at their calibrated poses, epipolar "
                         "correspondence and click-to-3D. Needs "
                         "scene_extrinsics.py to have been run.")
    ap.add_argument("--measure-frame", default="auto",
                    choices=["auto", "base", "stereo"],
                    help="coordinates to measure in. 'base' insists on "
                         "robot world coordinates and refuses if the rig "
                         "is not anchored; 'stereo' uses the reference "
                         "camera's frame (lengths are still correct); "
                         "'auto' takes world if available.")
    ap.add_argument("--depth", action="store_true",
                    help="also stream aligned depth, as an INDEPENDENT "
                         "cross-check on triangulated depth. Off by "
                         "default and never used by the record loop -- the "
                         "D405 is short-range and unreliable at scene "
                         "distances.")
    ap.add_argument("--frustum-scale", type=float, default=0.35,
                    help="how far in front of each camera its clickable "
                         "image plane is drawn, in metres. Bigger is "
                         "easier to click; it does not affect any "
                         "geometry.")
    ap.add_argument("--moving-time", type=float, default=0.15)
    ap.add_argument("--max-step", type=float, default=0.10)
    ap.add_argument("--limit-margin", type=float, default=0.02)
    ap.add_argument("--settle", type=float, default=0.6)
    args = ap.parse_args()

    ## Measuring needs the two static scene cameras; asking for the tab
    ## without them is almost certainly not what was meant.
    if args.measure and not args.cameras:
        args.cameras = ["top_scene", "low_scene"]
        print("  --measure with no --cameras: defaulting to "
              "top_scene low_scene")

    if args.from_robot:
        ensure_ros_path()

    import viser
    import viser.transforms as vtf
    from viser.extras import ViserUrdf

    print()
    print("#" * 74)
    print("#  GIAVA world view")
    print("#" * 74)
    print(f"\n  mode: {'LIVE (measured joints, jog enabled)' if args.from_robot else 'offline (URDF default configuration)'}")
    if not args.from_robot:
        print("        the robot drawn is the URDF DEFAULT CONFIGURATION,")
        print("        not your arms. Pass --from-robot to see the real pose.")
    else:
        print(f"        commandable: {', '.join(args.arms)}")
        others = [a for a in ARM_ORDER if a not in args.arms]
        if others:
            print(f"        read-only  : {', '.join(others)} "
                  f"(drawn from joint_states, not torqued)")

    view = WorldView(args)
    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=2.4, height=2.4, cell_size=0.1)

    # World origin: the frame every number in this package refers to.
    server.scene.add_frame("/world", axes_length=0.25, axes_radius=0.006,
                           origin_radius=0.012)
    server.scene.add_label("/world/label", f"{WORLD_FRAME} (world origin)",
                           position=(0.0, 0.0, 0.06))

    urdf_vis = ViserUrdf(server, view.frames.urdf, root_node_name="/robot")

    ## A frame's origin dot is small and easy to lose against the mesh.
    ## These are deliberately large spheres marking the EXACT point each
    ## number refers to -- flange (what FK and the dataset report), TCP
    ## (where the fingers close), and each camera's optical centre.
    ee_frames, ee_labels, ee_dots = {}, {}, {}
    tcp_dots, tcp_labels = {}, {}
    cam_frames, cam_dots = {}, {}
    for arm in ARM_ORDER:
        link = ARM_CONFIG[arm]["ee_link"]
        col = ARM_COLOR[arm]
        ee_frames[arm] = server.scene.add_frame(
            f"/ee/{arm}", axes_length=0.09, axes_radius=0.004,
            origin_radius=0.001)
        ## position=(0,0,0) is LOAD-BEARING: this is a child of
        ## /ee/<arm>, whose pose is set every tick, so the dot must stay at
        ## its parent's origin.  Setting a world position here (or in the
        ## update loop) applies the translation twice and the sphere floats
        ## off at 2x the distance.
        ee_dots[arm] = server.scene.add_icosphere(
            f"/ee/{arm}/dot", radius=0.012, color=col, position=(0.0, 0.0, 0.0))
        ee_labels[arm] = server.scene.add_label(
            f"/ee/{arm}/label", f"{arm} FLANGE ({link})",
            position=(0, 0, 0.045))
        if TCP.resolve(arm)["T_flange_tcp"] is not None:
            ## /tcp/<arm> is TOP-LEVEL (no parent transform), so unlike
            ## the two above this one IS given an absolute world position
            ## every tick.
            tcp_dots[arm] = server.scene.add_icosphere(
                f"/tcp/{arm}", radius=0.012, color=(255, 60, 60))
            tcp_labels[arm] = server.scene.add_label(
                f"/tcp/{arm}/label", f"{arm} TCP (grasp point)",
                position=(0, 0, 0.04))

    for cam, mount in ((c, CM.resolve_mount(c)) for c in CM.CAMERA_MOUNTS):
        if mount["T_parent_optical"] is None:
            continue
        cam_frames[cam] = server.scene.add_camera_frustum(
            f"/cameras/{cam}", fov=np.radians(58.0), aspect=640 / 480,
            scale=0.07, color=(255, 210, 90))
        ## Child of the frustum, which is repositioned each tick -- same
        ## rule as the ee dots above.
        cam_dots[cam] = server.scene.add_icosphere(
            f"/cameras/{cam}/dot", radius=0.010, color=(255, 210, 90),
            position=(0.0, 0.0, 0.0))
        server.scene.add_label(f"/cameras/{cam}/label", f"{cam} optical centre",
                               position=(0.0, 0.0, 0.035))

    # ---------------- GUI ---------------- #
    ## Viser has ONE control panel -- there is no second/left panel in the
    ## API.  Tabs solve the actual problem (not scrolling past the arm
    ## controls to reach the cameras); 'large' widens the panel so the
    ## images and tables are readable.  For a literal side-by-side you can
    ## run two servers on two ports and tile the browser windows.
    server.gui.configure_theme(control_layout="fixed", control_width="large",
                               dark_mode=True)

    ## STOP lives ABOVE the tab group, never inside it: an emergency
    ## control must not be one click behind a tab.
    if args.from_robot:
        stop_btn = server.gui.add_button("STOP ALL ARMS")
        stop_note = server.gui.add_markdown(
            "_Halts every arm in place. Torque stays on, so nothing drops._")

        @stop_btn.on_click
        def _(_) -> None:
            report = ESTOP.halt(reason="STOP button")
            stop_note.content = ("**STOPPED**\n\n"
                                 + "\n".join(f"- {l}" for l in report))

    tabs = server.gui.add_tab_group()

    with tabs.add_tab("Frames"):
        server.gui.add_markdown(f"""
World frame = **`{WORLD_FRAME}`**, the root link of `giava.urdf`.

| axis | direction |
|---|---|
| **+x** | operator's **left** |
| **+y** | operator's **backward** (toward you) |
| **+z** | **up** |

Grid squares are 10 cm. All three arm bases sit 20 mm above z=0.

`T_a_b` maps points from **b** into **a**, and is the pose of **b** in **a**.
`T_world_camera = T_world_ee @ T_ee_camera`.
""")

    ## Live images go in their own tab, so they are one click away rather
    ## than a scroll past every arm control.
    live_images, cam_workers = {}, {}
    if args.cameras:
        from rs_camera import resolve as rs_resolve
        from sync_capture import CameraWorker

        with tabs.add_tab("Cameras"):
            for name, serial in rs_resolve(args.cameras):
                w = CameraWorker(name, args.width, args.height,
                                 args.fps, global_time=True,
                                 keep_images=True,
                                 with_depth=args.depth)
                w.start()
                cam_workers[name] = w
                live_images[name] = server.gui.add_image(
                    np.zeros((480, 640, 3), dtype=np.uint8),
                    label=f"{name}  ({serial})", format="jpeg",
                    jpeg_quality=70)
                print(f"  streaming {name} ({serial})")

    ## The Measure tab goes here, immediately after Cameras: it needs the
    ## workers that block just created, and it must sit before the arm
    ## controls so the images are one click away rather than a scroll past
    ## every jog button.
    measure_panel = None
    if args.measure:
        import measure_view
        measure_panel = measure_view.attach(
            server, tabs, list(args.cameras or []), cam_workers,
            world_view=view, depth_enabled=args.depth,
            frame=args.measure_frame, frustum_scale=args.frustum_scale)
        if measure_panel is not None:
            print(f"  measuring in frame '{measure_panel.rig.ref_frame}' "
                  f"with {', '.join(measure_panel.usable) or 'no cameras'}")
            for n, why in measure_panel.unusable.items():
                print(f"    {n}: unavailable -- {why}")
        else:
            ## The tab explains itself, but someone watching the terminal
            ## should not have to open a browser to find out why the
            ## feature they asked for is not there.
            print("  measure: NO RIG -- the scene cameras have no calibrated"
                  " pose yet.\n"
                  "           Live images still work; see the Measure tab, or"
                  " run\n"
                  "           python calibration/scene_extrinsics.py status")

    base_xy = {}
    for arm in ARM_ORDER:
        j = view.frames.tree.parent_joint.get(f"{arm}_base_link")
        base_xy[arm] = np.asarray(j.xyz, float)[:2] if j is not None else None

    readouts, joint_readouts = {}, {}
    tab_readouts = tabs.add_tab("Pose readout")
    for arm in ARM_ORDER:
        with tab_readouts, server.gui.add_folder(
                f"{arm}  —  {ARM_CONFIG[arm]['ee_link']}",
                expand_by_default=(arm == ARM_ORDER[0])):
            server.gui.add_markdown(
                "_The gripper `ee_link` is the **mounting plate**, not a "
                "TCP: the grasp point is ~72 mm further along local +z and "
                "is unverified._" if arm in ("left", "right") else
                "_A camera **mount** link, not an optical frame._")
            readouts[arm] = server.gui.add_markdown("")
            joint_readouts[arm] = server.gui.add_markdown("")

    cam_readouts = {}
    with tab_readouts, server.gui.add_folder("Camera pose in world"):
        server.gui.add_markdown(
            "_Composed as `T_world_ee @ T_ee_camera`. None of these mounts "
            "is calibrated — the provenance is shown for each. See "
            "`camera_mount.py`._")
        for cam in cam_frames:
            m = CM.resolve_mount(cam)
            with server.gui.add_folder(f"{cam}  [{m['provenance']}]",
                                       expand_by_default=False):
                cam_readouts[cam] = server.gui.add_markdown("")

    ## ---------------- Poses ---------------- ##
    with tabs.add_tab("Poses"):
        if not args.from_robot:
            server.gui.add_markdown(
                "_Read-only: relaunch with `--from-robot` to move the arms._")
        book = view.pose_book()
        pose_arm = server.gui.add_dropdown(
            "arm", tuple(view.arms), initial_value=view.arms[0])
        pose_pick = server.gui.add_dropdown(
            "pose", tuple(sorted(book.get(view.arms[0], {}))) or ("(none)",))
        go_btn = server.gui.add_button("GO TO POSE")
        pose_status = server.gui.add_markdown("_no move yet_")

        @pose_arm.on_update
        def _(_) -> None:
            names = tuple(sorted(view.pose_book().get(pose_arm.value, {})))
            pose_pick.options = names or ("(none)",)

        @go_btn.on_click
        def _(_) -> None:
            r = view.go_to_pose(pose_arm.value, pose_pick.value)
            if "error" in r:
                pose_status.content = f"**{r['error']}**"
            else:
                p_ = r["ee_xyz_m"]
                pose_status.content = (
                    f"**{r['arm']} -> {r['pose']}**\n\n"
                    f"{r['n_waypoints']} waypoints, envelope checked\n\n"
                    f"EE now `[{p_[0]:+.4f}, {p_[1]:+.4f}, {p_[2]:+.4f}]` m")
            print(f"  pose: {r}")

        server.gui.add_markdown("---\n**Save the current pose**")
        save_name = server.gui.add_text("name", "")
        save_btn = server.gui.add_button("SAVE CURRENT JOINTS")
        save_status = server.gui.add_markdown(
            "_Captures the arm's measured joints, writes "
            "`poses_custom.json`, and prints a line you can paste straight "
            "into `arm_config.py`._")

        @save_btn.on_click
        def _(_) -> None:
            r = view.save_pose(pose_arm.value, save_name.value)
            if "error" in r:
                save_status.content = f"**{r['error']}**"
                return
            names = tuple(sorted(view.pose_book().get(pose_arm.value, {})))
            pose_pick.options = names or ("(none)",)
            save_status.content = (
                f"**saved {r['arm']}:{r['pose']}**\n\n"
                f"written to `{Path(r['file']).name}`\n\n"
                f"paste into `arm_config.py`:\n\n"
                f"```python\n{r['arm_config_snippet']}\n```\n"
                f"then {r['arm_config_hint']}")

    ## ---------------- Capture ---------------- ##
    ## Position the arms with the Poses/Jog tabs, then capture from here.
    ## The pose block is recorded AT CAPTURE TIME, not at startup -- the
    ## whole point is that it describes where the arms actually were when
    ## the shutter fired.
    with tabs.add_tab("Capture"):
        if not cam_workers:
            server.gui.add_markdown(
                "_No cameras streaming. Relaunch with e.g._\n\n"
                "`--cameras left_wrist right_wrist`")
        else:
            server.gui.add_markdown(
                "_Position the arms first (Poses / Jog), then capture. Each "
                "shot stores both images, every clock, and the **pose at "
                "that moment** — so the extrinsics stay recoverable._")
            cap_mode = server.gui.add_dropdown(
                "mode", ("burst", "drift"), initial_value="burst")
            cap_shots = server.gui.add_slider("shots", min=1, max=60, step=1,
                                              initial_value=5)
            cap_interval = server.gui.add_slider(
                "interval (s), drift only", min=0.0, max=30.0, step=0.5,
                initial_value=5.0)
            cap_btn = server.gui.add_button("CAPTURE")
            cap_status = server.gui.add_markdown("_no capture yet_")

            capture_busy = {"running": False}
            ## ONE session accumulates across clicks. For reconstruction you
            ## want every viewpoint in a single folder to hand to a model,
            ## not a folder per button press -- each shot already carries
            ## its own pose, so poses from different arm configurations sit
            ## happily in the same session.
            from common import DIR_VISER as _DV, timestamp as _ts0
            session = {"dir": _DV.parent / "sessions" /
                                f"multipose_{_ts0()}",
                       "shots": [], "poses": 0}
            session_label = server.gui.add_markdown("")

            def _refresh_session():
                session_label.content = (
                    f"_session_ `{session['dir'].name}`  \n"
                    f"{len(session['shots'])} shot(s) from "
                    f"{session['poses']} pose(s)")

            _refresh_session()
            new_session_btn = server.gui.add_button("START NEW SESSION")

            @new_session_btn.on_click
            def _(_) -> None:
                from common import timestamp as _ts1
                session["dir"] = _DV.parent / "sessions" / f"multipose_{_ts1()}"
                session["shots"] = []
                session["poses"] = 0
                _refresh_session()
                cap_status.content = "_new session started_"

            def _run_capture(mode, shots, interval, session_dir):
                base = len(session["shots"])   # continue the shot numbering
                """Runs off the GUI thread so a long drift run does not
                freeze the viewer (or the STOP button)."""
                import cv2
                from capture_session import (analyse, intrinsics_block,
                                             pose_block, take_shot)

                workers = list(cam_workers.values())
                names = [w.cam.name for w in workers]
                (session_dir / "images").mkdir(parents=True, exist_ok=True)

                pose = None
                if view.live and view.bridge is not None:
                    try:
                        pose = pose_block(view.frames, view.bridge,
                                          view.read_q_driver(), names)
                    except Exception as exc:
                        print(f"  pose block failed: {exc}")

                out, last = [], {n: None for n in names}
                for k in range(shots):
                    i = base + k       # unique across poses in this session
                    shot = take_shot(workers, require_new=(mode == "burst"),
                                     last_frames=last)
                    if shot is None:
                        continue
                    for n in names:
                        last[n] = shot["records"][n].frame_number
                    entry = {"index": i, "reference_s": shot["reference_s"],
                             "spread_s": shot["spread_s"],
                             "timestamps_s": shot["timestamps_s"],
                             "cameras": {}}
                    ## Per-shot pose: with the arms jogged between shots the
                    ## session-start pose would be wrong for all but the
                    ## first image.
                    if view.live and view.bridge is not None:
                        try:
                            entry["robot_pose"] = pose_block(
                                view.frames, view.bridge,
                                view.read_q_driver(), names)
                        except Exception:
                            pass
                    for n in names:
                        rec = shot["records"][n]
                        fn = f"images/shot{i:04d}_{n}.png"
                        cv2.imwrite(str(session_dir / fn),
                                    cv2.cvtColor(rec.image, cv2.COLOR_RGB2BGR))
                        d = rec.timing_dict()
                        d["image"] = fn
                        entry["cameras"][n] = d
                    out.append(entry)
                    cap_status.content = (
                        f"**capturing {mode}** — {len(out)}/{shots}  \n"
                        f"last spread {shot['spread_s']*1e3:.2f} ms")
                    if interval > 0 and mode == "drift":
                        time.sleep(interval)

                ## Never let the ANALYSIS lose the CAPTURE. The images and
                ## poses are the irreplaceable part; statistics can be
                ## recomputed from session.json afterwards.
                session["shots"].extend(out)
                session["poses"] += 1
                _refresh_session()
                all_shots = session["shots"]
                try:
                    stats = (analyse(out, names, mode, args.fps)
                             if len(out) > 1 else {})
                except Exception as exc:
                    print(f"  statistics failed ({exc}); saving the shots "
                          f"anyway")
                    stats = {"error": str(exc)}
                from common import provenance as _prov, save_json as _save
                _save({
                    "metadata": _prov(f"world_view_capture_{mode}",
                                      cameras=names, mode=mode,
                                      interval_s=interval),
                    "hardware_sync": {
                        "external_sync_wired": False,
                        "statement": ("D405 has no inter_cam_sync_mode; "
                                      "these cameras free-run.")},
                    "intrinsics": intrinsics_block(workers),
                    "robot_pose": pose,
                    "statistics_last_burst": stats,
                    "n_poses": session["poses"],
                    "shots": all_shots,
                }, session_dir / "session.json", overwrite=True)

                try:
                    cap_status.content = _capture_md(all_shots, stats, session_dir,
                                                     pose, names)
                except Exception as exc:
                    cap_status.content = (f"**saved {len(out)} shot(s)** to "
                                          f"`{session_dir.name}`\n\n"
                                          f"_summary failed: {exc}_")
                print(f"  captured {len(out)} shot(s) -> {session_dir}")
                capture_busy["running"] = False

            @cap_btn.on_click
            def _(_) -> None:
                if capture_busy["running"]:
                    return
                capture_busy["running"] = True
                mode = cap_mode.value
                shots = int(cap_shots.value)
                interval = float(cap_interval.value) if mode == "drift" else 0.0
                session_dir = session["dir"]
                cap_status.content = f"**capturing {mode}** — 0/{shots}"
                threading.Thread(target=_run_capture,
                                 args=(mode, shots, interval, session_dir),
                                 daemon=True).start()

    ## ---------------- Jog ---------------- ##
    jog_result = None
    if args.from_robot:
        with tabs.add_tab("Jog"):
            jog_arm = server.gui.add_dropdown("arm", tuple(view.arms),
                                              initial_value=view.arms[0])
            ## WHICH POINT the jog acts on. Everything is a fixed offset
            ## from the flange, so switching this changes what stays put
            ## when you rotate: the flange, the fingers, or the camera.
            point_opts = ["flange", "tcp"] + sorted(
                c for c, m in ((c, CM.resolve_mount(c))
                               for c in CM.CAMERA_MOUNTS)
                if m["T_parent_optical"] is not None)
            jog_point = server.gui.add_dropdown(
                "move / rotate", tuple(point_opts), initial_value="flange")
            point_note = server.gui.add_markdown("")

            @jog_point.on_update
            def _(_) -> None:
                _, lbl, note = view.offset_for(jog_arm.value, jog_point.value)
                point_note.content = (
                    f"_commanding **{lbl}** — {note}_"
                    + ("" if lbl == jog_point.value else
                       "\n\n⚠ **fell back to the flange**"))

            @jog_arm.on_update
            def _(_) -> None:
                _, lbl, note = view.offset_for(jog_arm.value, jog_point.value)
                point_note.content = f"_commanding **{lbl}** — {note}_"
            jog_step = server.gui.add_slider("step (mm)", min=1.0, max=50.0,
                                             step=1.0,
                                             initial_value=args.jog_step * 1e3)
            jog_buttons = {ax: server.gui.add_button(ax) for ax in AXES}
            jog_result = server.gui.add_markdown("_no jog yet_")

            server.gui.add_markdown("---\n**Rotate** (pivots in place)")
            rot_frame = server.gui.add_dropdown(
                "about", ("ee", "world"), initial_value="ee")
            rot_deg = server.gui.add_slider("degrees", min=1.0, max=45.0,
                                            step=1.0, initial_value=10.0)
            rot_buttons = {}
            for _rax in ("roll", "pitch", "yaw"):
                for _sgn, _lbl in ((+1, "+"), (-1, "-")):
                    rot_buttons[(_rax, _sgn)] = server.gui.add_button(
                        f"{_lbl}{_rax}")
            server.gui.add_markdown(
                "_`ee` = the tool's own roll/pitch/yaw. `world` = about the "
                "world axes (yaw about +z, pitch about +x, roll about +y). "
                "The commanded point stays put and the arm swings around "
                "it — with `--tcp` that pivot is the grasp point._")

            def _make_rot(rax, sgn):
                @rot_buttons[(rax, sgn)].on_click
                def _(_) -> None:
                    r = view.jog_rotate(jog_arm.value, rax,
                                        sgn * rot_deg.value, rot_frame.value,
                                        jog_point.value)
                    jog_result.content = _rot_md(r)
                    print(f"  rotate {jog_arm.value} {rax} "
                          f"{sgn*rot_deg.value:+.0f} deg -> {r}")

            for _rax in ("roll", "pitch", "yaw"):
                for _sgn in (+1, -1):
                    _make_rot(_rax, _sgn)

            def _make(ax):
                @jog_buttons[ax].on_click
                def _(_) -> None:
                    r = view.jog(jog_arm.value, ax,
                                 jog_step.value * 1e-3, jog_point.value)
                    jog_result.content = _jog_md(r)
                    print(f"  jog {jog_arm.value} {ax} "
                          f"{jog_step.value:.0f} mm -> "
                          f"measured {np.linalg.norm(r.get('measured_mm', [0,0,0])):.2f} mm")
            for ax in AXES:
                _make(ax)
    else:
        with tabs.add_tab("Jog"):
            server.gui.add_markdown(
                "_Disabled: relaunch with `--from-robot` to move the real "
                "arms and measure a commanded step with a ruler._")

    if cam_workers:
        from sync_capture import _check_delivering
        time.sleep(1.2)
        _check_delivering(list(cam_workers.values()))

    print(f"\n  World view:  http://localhost:{args.port}\n", flush=True)

    period = 1.0 / max(args.rate, 1.0)
    try:
        while True:
            q_urdf = view.q_urdf()
            urdf_vis.update_cfg(np.asarray(q_urdf, dtype=np.float32))
            fk = view.frames.fk(q_urdf)

            q_drv = view.read_q_driver()
            for arm in ARM_ORDER:
                T = fk[ARM_CONFIG[arm]["ee_link"]]
                h = ee_frames[arm]
                h.position = tuple(T[:3, 3])
                h.wxyz = tuple(matrix_to_quat_wxyz(T[:3, :3]))
                src = ("live (commanded)" if arm in view.sessions
                       else "live (read-only)" if view.arm_is_live(arm)
                       else ("URDF default -- NOT the real arm"
                             if not view.live else
                             "NO JOINT DATA -- pose below is NOT real"))
                warn = "" if view.arm_is_live(arm) or not view.live else \
                    "\n\n**⚠ this arm's pose is not measured**"
                ## ee_dots live at /ee/<arm>/dot -- CHILDREN of the frame
                ## whose pose was just set above. Their position is
                ## relative to that parent, so it must stay at the origin;
                ## setting it to the world position again put them at
                ## twice the translation, floating off in space.
                if arm in tcp_dots:
                    ## /tcp/<arm> is top-level, so this one IS absolute.
                    tcp_dots[arm].position = tuple(TCP.tcp_pose(T, arm)[:3, 3])
                readouts[arm].content = (f"_source: {src}_{warn}\n\n"
                                         + _pose_md(T, base_xy[arm]))
                joint_readouts[arm].content = _joints_md(
                    view.frames, q_drv, q_urdf, arm)

            for cam, handle in cam_frames.items():
                mount = CM.resolve_mount(cam)
                T = fk[mount["rigid_to"]] @ mount["T_parent_optical"]
                handle.position = tuple(T[:3, 3])
                handle.wxyz = tuple(matrix_to_quat_wxyz(T[:3, :3]))
                ## Same parenting rule: /cameras/<cam>/dot is a child of
                ## the frustum, which was just placed. Leave it at origin.
                cam_readouts[cam].content = _pose_md(T)

            for name, h in live_images.items():
                rec = cam_workers[name].snapshot()
                if rec is not None and rec.image is not None:
                    h.image = rec.image

            if measure_panel is not None:
                ## Undistorting two 640x480 frames every tick is the most
                ## expensive thing in this loop; a failure here must not
                ## take the viewer (and the STOP button) down with it.
                try:
                    measure_panel.update()
                except Exception as e:  # noqa: BLE001
                    print(f"  [measure] {type(e).__name__}: {e}")

            time.sleep(period)
    except KeyboardInterrupt:
        print("\n  shutting down")
    finally:
        for w in cam_workers.values():
            w.stop()
        if view.jog_log:
            _save_jog(view)


def _pose_line(T: np.ndarray) -> str:
    return f"xyz [{T[0,3]:+.4f}, {T[1,3]:+.4f}, {T[2,3]:+.4f}] m"


def _pose_md(T: np.ndarray, base_xy=None) -> str:
    """Full pose plus the quantities you can actually put a ruler on."""
    from common import matrix_to_rpy

    p = T[:3, 3]
    rpy = np.degrees(matrix_to_rpy(T[:3, :3]))
    q = matrix_to_quat_wxyz(T[:3, :3])
    ## Spell out what each axis MEANS physically.  x/y/z alone invite the
    ## reader to substitute their own convention -- and the left-right vs
    ## front-back assignment is exactly the pair people swap.
    lines = [
        "| | x → operator **left** | y → operator **back** | z → **up** |",
        "|---|---|---|---|",
        f"| **pos (m)** | {p[0]:+.4f} | {p[1]:+.4f} | {p[2]:+.4f} |",
        f"| **pos (mm)** | {p[0]*1e3:+.1f} | {p[1]*1e3:+.1f} | {p[2]*1e3:+.1f} |",
        f"| **rpy (deg)** | {rpy[0]:+.2f} | {rpy[1]:+.2f} | {rpy[2]:+.2f} |",
        "",
        f"quat wxyz `[{q[0]:+.5f}, {q[1]:+.5f}, {q[2]:+.5f}, {q[3]:+.5f}]`",
    ]
    if base_xy is not None:
        # Two numbers a tape measure can check directly.
        radius = float(np.linalg.norm(p[:2] - np.asarray(base_xy, float)))
        lines += [
            "",
            f"**height above z=0**: {p[2]*1e3:.1f} mm",
            f"**horizontal distance from its own waist axis**: "
            f"{radius*1e3:.1f} mm",
        ]
    return "\n".join(lines)


def _joints_md(frames, q_driver, q_urdf, arm: str) -> str:
    """Driver and URDF joint values side by side.

    They differ for the middle arm (flipped axes, mounting offsets, and the
    waist's pi shift), which is exactly the kind of thing worth being able
    to see while standing at the robot."""
    names = ARM_CONFIG[arm]["joint_names"]
    idx = [frames.actuated_names.index(n) for n in names]
    out = ["| joint | driver (rad) | driver (deg) | URDF (rad) |",
           "|---|---|---|---|"]
    for n, i in zip(names, idx):
        d, u = float(q_driver[i]), float(q_urdf[i])
        flag = "" if abs(d - u) < 1e-6 else "  ⟵ differs"
        out.append(f"| `{n}` | {d:+.4f} | {np.degrees(d):+.2f} | "
                   f"{u:+.4f}{flag} |")
    return "\n".join(out)


def _capture_md(shots: List[Dict[str, Any]], stats: Dict[str, Any],
                session_dir, pose, names: List[str]) -> str:
    """What the Capture tab shows once a run finishes."""
    if not shots:
        return "**no shots captured** — did the cameras stop delivering?"
    lines = [f"**{len(shots)} shot(s)** → `{session_dir.name}`", ""]
    if stats:
        g = stats.get("inter_shot_gap_ms", {})
        o = stats.get("offset_ms", {})
        a = stats.get("alignment_spread_ms", {})
        lines += [
            "| | value |",
            "|---|---|",
            f"| inter-shot gap | {g.get('mean', float('nan')):.2f} ms |",
        ]
        if stats.get("achieved_rate_hz"):
            lines.append(f"| achieved rate | {stats['achieved_rate_hz']:.2f} Hz "
                         f"(configured {stats['configured_fps']}) |")
        lines += [
            f"| offset ({stats.get('delta_definition', '')}) | "
            f"{o.get('mean', float('nan')):+.3f} ms, std "
            f"{o.get('std', float('nan')):.3f} |",
            f"| alignment spread | max {a.get('max', float('nan')):.3f} ms |",
        ]
        d = stats.get("offset_drift_ms_per_s")
        if d is not None:
            lines.append(f"| clock drift | {d*60:+.4f} ms/min over "
                         f"{stats['offset_drift_window_s']:.0f} s |")
        else:
            lines.append("| clock drift | window too short to measure |")
        lines.append("")
    if pose is None:
        lines += ["⚠ **no pose recorded** — relaunch with `--from-robot` or "
                  "start the driver, or these images cannot be tied to an "
                  "arm configuration.", ""]
    else:
        ok = [c for c, v in pose["cameras"].items() if v["recoverable"]]
        rel = pose.get("relative_extrinsic")
        lines += [f"pose recorded; extrinsics recoverable for: "
                  f"**{', '.join(ok) if ok else 'none'}**"]
        if rel:
            lines.append(f"baseline `{rel['baseline_m']*1e3:.1f}` mm "
                         f"(`{rel['name']}`)")
        lines += ["", "_Mount transforms are still nominal, so the pose "
                  "block stores the joint angles too — the extrinsics can "
                  "be recomputed once a hand-eye calibration exists._"]
    return "\n".join(lines)


def _rot_md(r: Dict[str, Any]) -> str:
    if "error" in r:
        return f"**{r['error']}**"
    d = np.asarray(r["point_drift_mm"])
    ik = np.asarray(r.get("ik_predicted_drift_mm", [0, 0, 0]))
    hw = np.asarray(r.get("hardware_drift_mm", d))
    return f"""
**{r['arm']}  {r['rotation_axis']}  {r['commanded_deg']:+.0f} deg**
about the **{r['frame']}** axes, pivoting the **{r['commanded_point']}**

| | degrees |
|---|---|
| commanded | {abs(r['commanded_deg']):.2f} |
| IK predicted | {r['ik_predicted_deg']:.2f} |
| measured | {r['measured_deg']:.2f} |

**pivot drift** (should be ~0), mm:

| | x | y | z | ‖·‖ |
|---|---|---|---|---|
| solver gave up | {ik[0]:+.2f} | {ik[1]:+.2f} | {ik[2]:+.2f} | {np.linalg.norm(ik):.2f} |
| hardware added | {hw[0]:+.2f} | {hw[1]:+.2f} | {hw[2]:+.2f} | {np.linalg.norm(hw):.2f} |
| **total measured** | {d[0]:+.2f} | {d[1]:+.2f} | {d[2]:+.2f} | {np.linalg.norm(d):.2f} |

_Solver drift is a weight-tuning question (pos 50 / ori 10). Hardware
drift is servo error and gravity, amplified by the arm's reach._
"""


def _jog_md(r: Dict[str, Any]) -> str:
    if "error" in r:
        return f"**{r['error']}**"
    c, i, m = (np.asarray(r[k]) for k in
               ("commanded_mm", "ik_predicted_mm", "measured_mm"))
    return f"""
**{r['arm']}  {r['axis']}  {r['step_m']*1e3:.0f} mm**

| | x | y | z | ‖·‖ |
|---|---|---|---|---|
| commanded | {c[0]:+.2f} | {c[1]:+.2f} | {c[2]:+.2f} | {np.linalg.norm(c):.2f} |
| IK predicted | {i[0]:+.2f} | {i[1]:+.2f} | {i[2]:+.2f} | {np.linalg.norm(i):.2f} |
| measured | {m[0]:+.2f} | {m[1]:+.2f} | {m[2]:+.2f} | {np.linalg.norm(m):.2f} |

all mm. Now measure it on the real arm with a ruler.
"""


def _save_jog(view: WorldView) -> None:
    from common import DIR_ROBOT, ensure_dirs, provenance, save_json, timestamp

    ensure_dirs()
    path = DIR_ROBOT / f"world_view_jog_{timestamp()}.json"
    save_json({
        "metadata": provenance("world_view_jog"),
        "convention": ("Displacements are world-frame vectors in mm, "
                       "differences of the ee_link origin."),
        "how_to_complete": ("Add your ruler reading per jog as "
                            "physical_measured_mm."),
        "jogs": [dict(j, physical_measured_mm=None) for j in view.jog_log],
    }, path)
    print(f"  {len(view.jog_log)} jog(s) saved to {path}")


if __name__ == "__main__":
    main()
