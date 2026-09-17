"""Cartesian motion validation  (PHASE 2).

Commands a relative Cartesian displacement of one arm's end-effector frame
and reports three displacements side by side so they can be compared
against a ruler:

    commanded    what was asked for
    ik_predicted forward kinematics of the IK solution -- what the solver
                 believes it can reach.  Differs from commanded when the
                 pose is unreachable or the solver trades position against
                 orientation.
    measured     forward kinematics of the joint angles the servos actually
                 report afterwards -- "robot-reported displacement".

The physical ruler measurement is the fourth column, and it is the one
this whole script exists to be compared against.  Record it in the run's
JSON afterwards (there is a `physical_measured_m` field left null for it).

Everything reuses the deployed stack: the pyroki model from giava.urdf, the
CoupledStudyIK solver that data collection runs, interbotix for the joint
commands, and the same driver-feasibility clamp.  No parallel kinematics.

SAFETY
    The arm moves.  Nothing happens without an explicit confirmation
    (--yes to skip it), --dry-run does the whole computation and prints the
    predictions without commanding anything, and the arm returns to its
    starting joint configuration after each axis unless --no-return.

Examples

    # Solver-only preview -- no hardware, safe anywhere:
    python calibration/move_validation.py --arm right --dry-run

    # The full six-axis test at 50 mm, pausing for a ruler at each extreme:
    python calibration/move_validation.py --arm right --distance 0.05 --pause

    # One axis, 20 mm, in the end-effector's own frame:
    python calibration/move_validation.py --arm right --axes +z \\
        --distance 0.02 --frame ee
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ is a SIBLING of this package now, not the
## parent it used to be.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
for _p in (str(HERE), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    DIR_ROBOT,
    invert_T,
    ensure_ros_path,
    confirm,
    wait_for_enter,
    T_to_dict,
    ensure_dirs,
    format_T,
    matrix_to_quat_wxyz,
    provenance,
    rotation_angle_deg,
    save_json,
    timestamp,
)
from kinematics import ARM_ORDER, WORLD_FRAME, RobotFrames  # noqa: E402
from estop import ESTOP  # noqa: E402
import tcp as TCP  # noqa: E402
from arm_config import ARM_CONFIG, URDF_PATH  # noqa: E402

AXES = {
    "+x": np.array([1.0, 0.0, 0.0]), "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]), "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]), "-z": np.array([0.0, 0.0, -1.0]),
}
DEFAULT_AXES = ["+x", "-x", "+y", "-y", "+z", "-z"]


## ------------------------------------------------------------------ ##
## Robot session
## ------------------------------------------------------------------ ##

class MotionSession:
    """Owns the robot handles, the solver, and the FK model for one run."""

    def __init__(self, arm: str, dry_run: bool, control_dt: float = 0.02,
                 use_tcp: bool = False, point: str = None,
                 camera: str = None):
        self.arm = arm
        self.dry_run = dry_run

        ## WHICH POINT ON THE ARM IS BEING COMMANDED.
        ##
        ## The IK always tracks the flange (*_gripper_base) -- that is the
        ## link in ARM_CONFIG.  Everything else is a FIXED offset from it:
        ##
        ##     T_world_point = T_world_flange @ T_flange_point
        ##
        ## so a target expressed for that point converts back with
        ##
        ##     T_world_flange = T_world_point @ inv(T_flange_point)
        ##
        ## which is what solve() does.  One piece of arithmetic covers the
        ## grasp point and the camera alike; only the offset differs.
        self.point = point or ("tcp" if use_tcp else "flange")
        self.camera = camera
        self._offset = None            # None == the flange itself
        if self.point == "tcp":
            if arm not in TCP.TCP_ARMS:
                self.point = "flange"
            else:
                self._offset = TCP.resolve(arm)["T_flange_tcp"]
        elif self.point == "camera":
            import camera_mount as CM
            if camera is None:
                raise ValueError("point='camera' needs camera=<name>")
            mount = CM.resolve_mount(camera)
            if mount["T_parent_optical"] is None:
                raise ValueError(
                    f"camera '{camera}' has no mount transform "
                    f"(provenance='{mount['provenance']}') -- it cannot be "
                    f"commanded. Calibrate it with handeye.py first.")
            self._offset = mount["T_parent_optical"]
            self._mount_provenance = mount["provenance"]

        self.frames = RobotFrames()
        self.joint_idx = self.frames.joint_indices(arm)
        self.ee_link = self.frames.ee_link(arm)
        self.robots: Dict[str, Any] = {}
        self.waist_shift = 0.0

        if not dry_run:
            self._connect()

        from kinematics import JointFrameBridge
        self.bridge = JointFrameBridge(self.frames.robot,
                                       waist_driver_shift=self.waist_shift)

        ## Dry-run stand-in for the servos: starts at the URDF default,
        ## expressed in driver coordinates so every path below is identical
        ## to the live one.
        self._sim_q = self.bridge.to_driver(self.frames.home_q())

        from study_ik import CoupledStudyIK
        self.ik = CoupledStudyIK(
            self.frames.robot,
            str(URDF_PATH),
            ee_links={a: ARM_CONFIG[a]["ee_link"] for a in ARM_ORDER},
            control_dt=control_dt,
            waist_driver_shift=self.waist_shift,
        )

    def _connect(self) -> None:
        ensure_ros_path()
        import rospy
        from robot_control import (create_and_configure_robots,
                                   read_middle_waist_shift)

        ## disable_signals=True keeps rospy from installing its own SIGINT
        ## handler, so ours is the one that runs and the arms get halted
        ## before anything starts unwinding.
        rospy.init_node("giava_move_validation", anonymous=True,
                        disable_signals=True)
        ## Only the arm under test is created: creating the others would
        ## torque them on for no reason.
        self.robots = create_and_configure_robots((self.arm,))
        rospy.sleep(0.5)

        ## Arm the emergency stop the moment a real arm exists, before any
        ## motion is possible.
        for name, bot in self.robots.items():
            ESTOP.register(name, bot)
        ESTOP.install_signal_handlers()
        print(f"  {ESTOP.describe()}")
        if self.arm == "middle":
            self.waist_shift = read_middle_waist_shift(self.robots["middle"])

    # -------------------------------------------------------------- #
    def read_q_driver(self) -> np.ndarray:
        """Full-length DRIVER joint vector.

        Live: the servos' reported positions.  Dry run: the last commanded
        configuration, i.e. PERFECT TRACKING IS ASSUMED.  The dry run
        therefore exercises the solver and the frame arithmetic but says
        nothing about how the hardware actually follows -- its 'measured'
        column is simulated, and labelled as such in the output."""
        if self.dry_run:
            return self._sim_q.copy()
        q = np.zeros(self.frames.num_actuated, dtype=np.float64)
        n = ARM_CONFIG[self.arm]["num_joints"]
        q[self.joint_idx] = np.asarray(
            self.robots[self.arm].dxl.joint_states.position[:n], dtype=np.float64)
        return q

    @property
    def use_tcp(self) -> bool:
        """Back-compat for callers written before `point` existed."""
        return self.point == "tcp"

    def ee_pose(self, q_driver: np.ndarray) -> np.ndarray:
        """T_world_flange from a DRIVER joint vector (converted first).

        Always the FLANGE (*_gripper_base).  Use point_pose() for whichever
        point this session is actually commanding."""
        return self.frames.ee_pose(self.bridge.to_urdf(q_driver), self.arm)

    def point_pose(self, q_driver: np.ndarray) -> np.ndarray:
        """Pose of the point this session commands (flange / TCP / camera)."""
        T = self.ee_pose(q_driver)
        return T if self._offset is None else T @ self._offset

    def solve(self, q_prev_driver: np.ndarray, T_target: np.ndarray) -> np.ndarray:
        """One IK solve toward a 4x4 world-frame target.

        When a point other than the flange is being commanded the target
        describes THAT point, so it is converted to the flange pose the
        solver actually tracks.  That conversion is what makes the
        commanded point pivot in place instead of swinging on the offset
        as a lever."""
        if self._offset is not None:
            T_target = T_target @ invert_T(self._offset)
        pos = T_target[:3, 3]
        wxyz = matrix_to_quat_wxyz(T_target[:3, :3])
        return np.asarray(
            self.ik.solve(q_prev_driver.astype(np.float32),
                          {self.arm: (pos, wxyz)}), dtype=np.float64)

    def command(self, q_driver: np.ndarray, moving_time: float,
                max_step: float, limit_margin: float) -> np.ndarray:
        """Send one clamped joint command; returns what was actually sent.

        The clamp mirrors data_collection.py: interbotix rejects the ENTIRE
        group command if any joint is out of range or too fast, so an
        unclamped command can silently freeze the arm."""
        q_arm = q_driver[self.joint_idx]
        if self.dry_run:
            self._sim_q[self.joint_idx] = q_arm
            return q_arm

        bot = self.robots[self.arm]
        n = len(q_arm)
        ref = np.asarray(bot.dxl.joint_states.position[:n], dtype=float)
        getter = getattr(bot.arm, "get_joint_commands", None)
        if getter is not None:
            try:
                ref = np.asarray(getter(), dtype=float)[:n]
            except Exception:
                pass
        lo = np.asarray(bot.arm.group_info.joint_lower_limits, dtype=float)[:n]
        hi = np.asarray(bot.arm.group_info.joint_upper_limits, dtype=float)[:n]
        cmd = np.clip(q_arm, ref - max_step, ref + max_step)
        cmd = np.clip(cmd, lo + limit_margin, hi - limit_margin)
        if np.any(np.abs(cmd - q_arm) > 1e-6):
            j = int(np.argmax(np.abs(cmd - q_arm)))
            print(f"      [clamp] joint {j} "
                  f"{q_arm[j]:+.4f} -> {cmd[j]:+.4f}")
        bot.arm.set_joint_positions(cmd.tolist(), moving_time=moving_time,
                                    accel_time=0.5 * moving_time,
                                    blocking=False)
        return cmd

    def settle(self, seconds: float) -> None:
        if self.dry_run:
            return
        import rospy
        rospy.sleep(seconds)


## ------------------------------------------------------------------ ##
## One axis test
## ------------------------------------------------------------------ ##

def run_axis(sess: MotionSession, axis: str, distance: float, frame: str,
             steps: int, moving_time: float, max_step: float,
             limit_margin: float, settle: float, pause: bool,
             do_return: bool) -> Dict[str, Any]:
    """Displace, measure, report, and (optionally) go back."""
    print()
    print("=" * 74)
    print(f"  AXIS {axis}   {distance * 1e3:.1f} mm   in the "
          f"{'WORLD' if frame == 'world' else 'END-EFFECTOR'} frame")
    print("=" * 74)

    q_start = sess.read_q_driver()
    T_start = sess.point_pose(q_start)
    print("\n  START pose  T_%s_%s" % (WORLD_FRAME, sess.ee_link))
    print(format_T(T_start, indent="      "))

    ## Direction: world axes are fixed; ee axes rotate with the tool, so a
    ## '+z' test in ee frame follows the gripper's approach axis wherever it
    ## is pointing.
    unit = AXES[axis]
    if frame == "ee":
        direction = T_start[:3, :3] @ unit
    else:
        direction = unit
    commanded = distance * direction

    T_goal = T_start.copy()
    T_goal[:3, 3] = T_start[:3, 3] + commanded
    print(f"\n  commanded displacement [m]: "
          f"[{commanded[0]:+.5f}, {commanded[1]:+.5f}, {commanded[2]:+.5f}]"
          f"   |d| = {np.linalg.norm(commanded) * 1e3:.2f} mm")
    if frame == "ee":
        print(f"  ({axis} of the end-effector frame, expressed in world)")

    ## Interpolate: one big jump would be clamped to a crawl by the driver
    ## step limit and would also make the solver leave the warm-start basin.
    q_cmd = q_start.copy()
    for k in range(1, steps + 1):
        T_k = T_start.copy()
        T_k[:3, 3] = T_start[:3, 3] + commanded * (k / steps)
        q_sol = sess.solve(q_cmd, T_k)
        sess.command(q_sol, moving_time, max_step, limit_margin)
        q_cmd = q_sol
        if not sess.dry_run:
            time.sleep(moving_time)

    T_ik = sess.point_pose(q_cmd)
    ik_disp = T_ik[:3, 3] - T_start[:3, 3]

    sess.settle(settle)
    q_end = sess.read_q_driver()
    T_end = sess.point_pose(q_end)
    measured = T_end[:3, 3] - T_start[:3, 3]

    print("\n  END pose (measured joints -> FK)")
    print(format_T(T_end, indent="      "))

    _print_comparison(commanded, ik_disp, measured, T_start, T_end,
                      sess.dry_run)

    if pause and not sess.dry_run:
        print("\n  >>> Measure the physical displacement with a ruler now.")
        wait_for_enter("Press ENTER when done to continue... ")

    result = {
        "axis": axis,
        "frame": frame,
        "distance_m": distance,
        "commanded_m": commanded.tolist(),
        "ik_predicted_m": ik_disp.tolist(),
        "measured_m": measured.tolist(),
        ## Left for the operator to fill in from the ruler.  The whole
        ## point of the run is this comparison.
        "physical_measured_m": None,
        "commanded_norm_mm": float(np.linalg.norm(commanded) * 1e3),
        "ik_predicted_norm_mm": float(np.linalg.norm(ik_disp) * 1e3),
        "measured_norm_mm": float(np.linalg.norm(measured) * 1e3),
        "measured_minus_commanded_mm": ((measured - commanded) * 1e3).tolist(),
        "orientation_drift_deg": rotation_angle_deg(
            T_start[:3, :3].T @ T_end[:3, :3]),
        "pose_start": T_to_dict(T_start, WORLD_FRAME, sess.ee_link),
        "pose_end": T_to_dict(T_end, WORLD_FRAME, sess.ee_link),
        "q_driver_start": q_start[sess.joint_idx].tolist(),
        "q_driver_commanded": q_cmd[sess.joint_idx].tolist(),
        "q_driver_end": q_end[sess.joint_idx].tolist(),
    }

    if do_return:
        _return_to(sess, q_start, steps, moving_time, max_step, limit_margin,
                   settle)
        result["returned"] = True
        q_back = sess.read_q_driver()
        T_back = sess.point_pose(q_back)
        back_err = T_back[:3, 3] - T_start[:3, 3]
        result["return_residual_m"] = back_err.tolist()
        print(f"\n  returned to start; residual "
              f"{np.linalg.norm(back_err) * 1e3:.2f} mm")
    else:
        result["returned"] = False

    return result


def run_round_trip(sess: MotionSession, axis: str, distance: float,
                   frame: str, cycles: int, steps: int, moving_time: float,
                   max_step: float, limit_margin: float, settle: float,
                   pause_each: bool) -> Dict[str, Any]:
    """Go +axis then -axis, N times, and measure what does not come back.

    A perfect system returns to the start every cycle.  Real ones do not:
    gravity plus backlash makes the up leg fall short and the down leg
    overshoot, so each cycle leaves a net drift -- and it ACCUMULATES.
    This measures the size and the sign, separately at the encoder level
    and (via your ruler) at the physical level.

    The two differ, and the difference is the diagnosis:
      * encoder drift ~= physical drift  -> the servos are not reaching
        their setpoints; the encoders can see it. Servo/PID territory.
      * encoder returns but the ruler does not -> compliance DOWNSTREAM of
        the encoder (belt, gearbox, link flex). Invisible to the joint
        state, and not fixable in software.
    """
    unit = AXES[axis]
    print()
    print("=" * 74)
    print(f"  ROUND TRIP  {axis} then {_opposite(axis)}   "
          f"{distance * 1e3:.1f} mm   x{cycles} cycles")
    print("=" * 74)

    q_home = sess.read_q_driver()
    T_home = sess.point_pose(q_home)
    print("\n  start pose")
    print(format_T(T_home, indent="      "))

    rows = []
    q = q_home.copy()
    for c in range(1, cycles + 1):
        for leg, sign in ((axis, +1.0), (_opposite(axis), -1.0)):
            T_ref = sess.point_pose(sess.read_q_driver())
            direction = (T_ref[:3, :3] @ unit) if frame == "ee" else unit
            delta = sign * distance * direction
            T_goal = T_ref.copy()
            T_goal[:3, 3] = T_ref[:3, 3] + delta
            for k in range(1, steps + 1):
                Tk = T_ref.copy()
                Tk[:3, 3] = T_ref[:3, 3] + delta * (k / steps)
                q = sess.solve(q, Tk)
                sess.command(q, moving_time, max_step, limit_margin)
                if not sess.dry_run:
                    time.sleep(moving_time)
            sess.settle(settle)

        T_now = sess.point_pose(sess.read_q_driver())
        drift = T_now[:3, 3] - T_home[:3, 3]
        along = float(np.dot(drift, unit))
        rows.append({
            "cycle": c,
            "net_drift_mm": (drift * 1e3).tolist(),
            "drift_along_axis_mm": along * 1e3,
            "drift_norm_mm": float(np.linalg.norm(drift) * 1e3),
            "physical_drift_mm": None,
        })
        print(f"\n  cycle {c}: net drift "
              f"[{drift[0]*1e3:+.2f}, {drift[1]*1e3:+.2f}, "
              f"{drift[2]*1e3:+.2f}] mm   "
              f"along {axis}: {along*1e3:+.2f} mm")
        if pause_each and not sess.dry_run:
            pause(f"measure the net offset from the start (cycle {c})",
                  sess.dry_run)

    d = np.array([r["drift_along_axis_mm"] for r in rows])
    print()
    print("  " + "-" * 70)
    print(f"  encoder-level drift along {axis}: "
          f"mean {d.mean():+.2f} mm/cycle, total {d[-1]:+.2f} mm "
          f"after {cycles} cycle(s)")
    print("  " + "-" * 70)
    print(f"""
  Your ruler is the other half.  Record the physical offset per cycle in
  'physical_drift_mm'.  Then:
      encoder ~= physical   -> servos not reaching setpoint (PID/gains)
      encoder ~ 0, physical > 0 -> compliance below the encoder
                                   (belt/gearbox/flex), software cannot fix
""")
    return {"mode": "round_trip", "axis": axis, "cycles": cycles,
            "distance_m": distance, "frame": frame,
            "commanded_point": "tcp" if sess.use_tcp else "flange",
            "pose_start": T_to_dict(T_home, WORLD_FRAME, sess.ee_link),
            "cycles_data": rows,
            "encoder_mean_drift_mm_per_cycle": float(d.mean()),
            "how_to_complete": (
                "Fill physical_drift_mm per cycle from your ruler, signed "
                f"along {axis}.")}


def run_pose_sequence(sess: MotionSession, poses: List[str], settle: float,
                      moving_time: float, max_step: float,
                      limit_margin: float, pause_each: bool) -> Dict[str, Any]:
    """Visit named poses in order, reporting expected vs achieved EE change.

    'Expected' here is FK of the pose's own joint values -- where the arm
    WOULD put the end effector if the servos landed exactly on the
    commanded angles.  'Achieved' is FK of the angles they actually report.
    The difference is servo tracking; your ruler catches anything below the
    encoders on top of that.
    """
    import workspace as WS
    from robot_control import interpolate_to_pose

    print()
    print("=" * 74)
    print(f"  POSE SEQUENCE  ({sess.arm}):  {' -> '.join(poses)}")
    print("=" * 74)

    rows = []
    prev_T = sess.point_pose(sess.read_q_driver())
    prev_name = "start"
    print("\n  start pose")
    print(format_T(prev_T, indent="      "))

    for name in poses:
        q_now = sess.read_q_driver()
        q_target = q_now.copy()
        try:
            target_arm = sess.frames.q_from_named_pose(
                name, arms=(sess.arm,))[sess.joint_idx]
        except ValueError as exc:
            print(f"\n  {name}: {exc}")
            continue
        q_target[sess.joint_idx] = target_arm

        ## What FK says this pose should produce -- the prediction the
        ## measurement is compared against.
        T_expected = sess.point_pose(q_target)
        expected = T_expected[:3, 3] - prev_T[:3, 3]

        steps = max(1, int(np.ceil(
            np.max(np.abs(q_target - q_now)) / max_step)))
        path = [q_now + (q_target - q_now) * (k / steps)
                for k in range(1, steps + 1)]
        _, env_bad, pts = WS.check_configs(sess.frames, sess.bridge,
                                           [q_now] + path, sess.arm)
        print(f"\n  {prev_name} -> {name}"
              f"   ({steps} waypoints)")
        if env_bad and not sess.dry_run:
            print("      REFUSED: leaves the measured free space ("
                  + ", ".join(b[0] for b in env_bad) + ")")
            continue

        if sess.dry_run:
            sess._sim_q[sess.joint_idx] = target_arm
        else:
            interpolate_to_pose(sess.robots[sess.arm], sess.arm,
                                target_arm.tolist(),
                                moving_time=moving_time,
                                accel_time=0.5 * moving_time)
            sess.settle(settle)

        T_now = sess.point_pose(sess.read_q_driver())
        achieved = T_now[:3, 3] - prev_T[:3, 3]
        err = achieved - expected

        print(f"      {'':12s} {'x':>10s} {'y':>10s} {'z':>10s} {'norm':>10s}"
              "   [mm]")
        for lbl, v in (("expected", expected),
                       ("achieved" if not sess.dry_run else "achieved(SIM)",
                        achieved)):
            print(f"      {lbl:12s} {v[0]*1e3:>10.2f} {v[1]*1e3:>10.2f} "
                  f"{v[2]*1e3:>10.2f} {np.linalg.norm(v)*1e3:>10.2f}")
        print(f"      {'difference':12s} {err[0]*1e3:>10.2f} "
              f"{err[1]*1e3:>10.2f} {err[2]*1e3:>10.2f} "
              f"{np.linalg.norm(err)*1e3:>10.2f}")
        print(f"      {'physical':12s} {'?':>10s} {'?':>10s} {'?':>10s} "
              f"{'?':>10s}   <-- your ruler")
        print(f"      orientation change: "
              f"{rotation_angle_deg(prev_T[:3,:3].T @ T_now[:3,:3]):.2f} deg")

        rows.append({
            "from": prev_name, "to": name,
            "expected_mm": (expected * 1e3).tolist(),
            "achieved_mm": (achieved * 1e3).tolist(),
            "difference_mm": (err * 1e3).tolist(),
            "physical_measured_mm": None,
            "orientation_change_deg": rotation_angle_deg(
                prev_T[:3, :3].T @ T_now[:3, :3]),
            "pose_from": T_to_dict(prev_T, WORLD_FRAME, sess.ee_link),
            "pose_to": T_to_dict(T_now, WORLD_FRAME, sess.ee_link),
            "q_driver_target": target_arm.tolist(),
            "q_driver_reached": sess.read_q_driver()[sess.joint_idx].tolist(),
        })
        if pause_each and not sess.dry_run:
            pause(f"measure the {prev_name} -> {name} displacement",
                  sess.dry_run)
        prev_T, prev_name = T_now, name

    return {"mode": "pose_sequence", "arm": sess.arm, "poses": poses,
            "commanded_point": "tcp" if sess.use_tcp else "flange",
            "steps": rows,
            "how_to_complete": (
                "Fill physical_measured_mm per step from your ruler, as a "
                "world-frame 3-vector in mm.")}


def _opposite(axis: str) -> str:
    return ("-" if axis[0] == "+" else "+") + axis[1]


def _print_comparison(commanded, ik_disp, measured, T_start, T_end,
                      dry_run: bool = False) -> None:
    meas_label = "measured(SIM)" if dry_run else "measured"
    print()
    print("  " + "-" * 70)
    print(f"  {'':14s} {'x [mm]':>10s} {'y [mm]':>10s} {'z [mm]':>10s} "
          f"{'norm [mm]':>11s}")
    print("  " + "-" * 70)
    for name, v in (("commanded", commanded), ("ik_predicted", ik_disp),
                    (meas_label, measured)):
        print(f"  {name:14s} {v[0] * 1e3:>10.2f} {v[1] * 1e3:>10.2f} "
              f"{v[2] * 1e3:>10.2f} {np.linalg.norm(v) * 1e3:>11.2f}")
    err = measured - commanded
    print("  " + "-" * 70)
    print(f"  {'meas - cmd':14s} {err[0] * 1e3:>10.2f} {err[1] * 1e3:>10.2f} "
          f"{err[2] * 1e3:>10.2f} {np.linalg.norm(err) * 1e3:>11.2f}")
    print(f"  {'physical':14s} {'?':>10s} {'?':>10s} {'?':>10s} {'?':>11s}"
          "   <-- measure with a ruler")
    print("  " + "-" * 70)
    if dry_run:
        print("  measured(SIM) assumes the servos track the command exactly;")
        print("  it is the solver's own answer, NOT a hardware measurement.")
    drift = rotation_angle_deg(T_start[:3, :3].T @ T_end[:3, :3])
    print(f"  orientation drift over the move: {drift:.3f} deg "
          f"(should be ~0 -- the target held orientation fixed)")


def _return_to(sess: MotionSession, q_goal: np.ndarray, steps: int,
               moving_time: float, max_step: float, limit_margin: float,
               settle: float) -> None:
    """Go back in JOINT space -- the configuration we started from is known
    good, and re-solving IK for the start pose could land in a different
    branch (same pose, different elbow)."""
    print("\n  returning to the starting joint configuration...")
    q_now = sess.read_q_driver()
    for k in range(1, steps + 1):
        q_k = q_now + (q_goal - q_now) * (k / steps)
        sess.command(q_k, moving_time, max_step, limit_margin)
        if not sess.dry_run:
            time.sleep(moving_time)
    sess.settle(settle)


## ------------------------------------------------------------------ ##

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="right", choices=list(ARM_ORDER),
                    help="which arm to move (default: right)")
    ap.add_argument("--distance", type=float, default=0.05,
                    help="displacement magnitude in METRES (default: 0.05)")
    ## COMMA separated, not nargs="+": argparse's option prefix is '-', so
    ## `--axes +x -y` would parse `-y` as an unknown FLAG and abort. A single
    ## token starting with '+' is safe, and a lone negative axis still works
    ## with the '=' form: --axes=-y
    ap.add_argument("--axes", default=",".join(DEFAULT_AXES),
                    help="comma-separated axes to test, in order "
                         "(default: %(default)s). Note the comma: "
                         "`--axes +x,-x`, not `--axes +x -x`. For a single "
                         "negative axis use the '=' form: --axes=-y")
    ap.add_argument("--point", default=None,
                    choices=("flange", "tcp", "camera"),
                    help="which point to command: the gripper mounting "
                         "flange (default), the grasp point, or a CAMERA. "
                         "'camera' needs --camera and a calibrated mount.")
    ap.add_argument("--camera", default=None,
                    help="camera name when --point camera")
    ap.add_argument("--tcp", action="store_true",
                    help="command the GRASP POINT instead of the gripper "
                         "mounting flange. The flange sits ~72 mm short of "
                         "the fingers along its local +z, so without this a "
                         "commanded vertical move does not move the "
                         "fingertips by the same amount unless orientation "
                         "holds exactly. See tcp_offsets.json.")
    ap.add_argument("--frame", default="world", choices=("world", "ee"),
                    help="interpret the axes in the world frame (default) or "
                         "the end-effector's own frame")
    ap.add_argument("--steps", type=int, default=25,
                    help="interpolation steps per move (default: 25)")
    ap.add_argument("--moving-time", type=float, default=0.12,
                    help="interbotix moving_time per step [s]")
    ap.add_argument("--max-step", type=float, default=0.12,
                    help="max per-command joint delta [rad], driver clamp")
    ap.add_argument("--limit-margin", type=float, default=0.02,
                    help="cushion inside the driver's joint limits [rad]")
    ap.add_argument("--settle", type=float, default=1.0,
                    help="settle time before reading the final pose [s]")
    ap.add_argument("--pose-sequence", default=None,
                    help="comma-separated named poses to visit in order, "
                         "e.g. --pose-sequence forward,high,low,forward. "
                         "Reports FK-expected vs achieved EE change per "
                         "step, with a column for your ruler.")
    ap.add_argument("--round-trip", action="store_true",
                    help="go +axis then -axis repeatedly and measure what "
                         "does not come back. Quantifies the gravity/"
                         "backlash hysteresis that makes an up-then-down "
                         "pair land low.")
    ap.add_argument("--cycles", type=int, default=3,
                    help="round-trip cycles (default 3)")
    ap.add_argument("--pause", action="store_true",
                    help="wait for ENTER at each displaced pose so you can "
                         "measure it with a ruler")
    ap.add_argument("--no-return", action="store_true",
                    help="stay at the displaced pose instead of going back")
    ap.add_argument("--dry-run", action="store_true",
                    help="solve and report without touching the hardware")
    ap.add_argument("--yes", action="store_true",
                    help="skip the safety confirmation prompt")
    ap.add_argument("--out", default=None,
                    help="output JSON path (default: a timestamped file "
                         "under calibration/data/robot/)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow --out to replace an existing file")
    args = ap.parse_args()

    args.axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    unknown = [a for a in args.axes if a not in AXES]
    if unknown:
        ap.error(f"unknown axes {unknown}; choose from {sorted(AXES)}")

    ensure_dirs()

    print()
    print("#" * 74)
    print("#  GIAVA Cartesian motion validation  (PHASE 2)")
    print("#" * 74)
    print(f"""
  arm            {args.arm}   (ee_link '{ARM_CONFIG[args.arm]['ee_link']}')
  displacement   {args.distance * 1e3:.1f} mm along {' '.join(args.axes)}
  frame          {args.frame}
  mode           {'DRY RUN -- no hardware' if args.dry_run else 'LIVE -- THE ARM WILL MOVE'}
  STOP           Ctrl-C halts the arm in place (torque stays on)

  Reminder: '{ARM_CONFIG[args.arm]['ee_link']}' is the {'gripper mounting plate'
              if args.arm in ('left', 'right') else 'camera mount link'}, not a
  tool centre point.  A ruler held at the fingertips measures the same
  translation only while the orientation does not change -- which is why the
  orientation drift is reported for every move.
""")

    if not args.dry_run and not args.yes:
        print("  The arm will move. Clear the workspace.")
        if not confirm("Type 'yes' to continue: "):
            print("  aborted.")
            return

    sess = MotionSession(args.arm, args.dry_run, use_tcp=args.tcp,
                         point=args.point, camera=args.camera)

    ## Warm up the JIT before any motion so the first commanded step is not
    ## delayed by several seconds of compilation.
    q0 = sess.read_q_driver()
    print("  warming up the IK solver (JIT compile)...")
    t0 = time.perf_counter()
    sess.ik.warmup(q0.astype(np.float32))
    print(f"  ready ({time.perf_counter() - t0:.1f} s)")

    results: List[Dict[str, Any]] = []
    try:
        if args.pose_sequence:
            names = [p.strip() for p in args.pose_sequence.split(",")
                     if p.strip()]
            results.append(run_pose_sequence(
                sess, names, args.settle, args.moving_time, args.max_step,
                args.limit_margin, args.pause))
            _save(results, args, sess)
            return
        if args.round_trip:
            for axis in args.axes:
                results.append(run_round_trip(
                    sess, axis, args.distance, args.frame, args.cycles,
                    args.steps, args.moving_time, args.max_step,
                    args.limit_margin, args.settle, args.pause))
            _save(results, args, sess)
            return
        for axis in args.axes:
            results.append(run_axis(
                sess, axis, args.distance, args.frame, args.steps,
                args.moving_time, args.max_step, args.limit_margin,
                args.settle, args.pause, not args.no_return))
    except KeyboardInterrupt:
        print("\n  interrupted -- saving what was collected so far.")

    _summary(results)
    _save(results, args, sess)


def _summary(results: List[Dict[str, Any]]) -> None:
    if not results:
        return
    print()
    print("=" * 74)
    print("  SUMMARY   (all values in mm; 'physical' is yours to fill in)")
    print("=" * 74)
    print(f"  {'axis':6s} {'commanded':>10s} {'ik_pred':>10s} "
          f"{'measured':>10s} {'meas-cmd':>10s} {'drift deg':>10s}")
    print("  " + "-" * 66)
    for r in results:
        err = np.linalg.norm(np.asarray(r["measured_m"])
                             - np.asarray(r["commanded_m"])) * 1e3
        print(f"  {r['axis']:6s} {r['commanded_norm_mm']:>10.2f} "
              f"{r['ik_predicted_norm_mm']:>10.2f} "
              f"{r['measured_norm_mm']:>10.2f} {err:>10.2f} "
              f"{r['orientation_drift_deg']:>10.3f}")
    print("  " + "-" * 66)


def _save(results: List[Dict[str, Any]], args, sess: MotionSession) -> None:
    if args.out:
        path = Path(args.out)
    else:
        path = DIR_ROBOT / f"move_validation_{args.arm}_{timestamp()}.json"
    doc = {
        "metadata": provenance(
            "cartesian_motion_validation",
            arm=args.arm,
            ee_link=sess.ee_link,
            dry_run=bool(args.dry_run),
            distance_m=args.distance,
            axes=args.axes,
            axis_frame=args.frame,
            steps=args.steps,
            moving_time_s=args.moving_time,
        ),
        "convention": (
            "T_a_b maps points from frame b into frame a. Displacements are "
            "world-frame vectors in metres, differences of the ee_link "
            "origin."
        ),
        "ee_link_meaning": (
            "gripper mounting plate, NOT a tool centre point"
            if args.arm in ("left", "right")
            else "camera mount link (identical to 'middle_camera')"),
        "solver": "study_ik.CoupledStudyIK (the deployed data-collection IK)",
        "how_to_complete": (
            "Fill in physical_measured_m for each axis from your ruler "
            "measurement, in metres, as a 3-vector in the same world frame."
        ),
        "results": results,
    }
    save_json(doc, path, overwrite=args.overwrite)
    print(f"\n  saved to {path}")
    print("  Fill in each result's 'physical_measured_m' after measuring.\n")


if __name__ == "__main__":
    main()
