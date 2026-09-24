"""Scripted pick-and-place that moves the flower to the next placement cell
between rollouts, so a placement study runs without a human resetting the
scene.

HOW THE ARM KNOWS WHERE THINGS ARE -- and why this is not a camera calibration.

There is no solved top_scene extrinsic on this rig (calibration/data/extrinsics
is empty; `load_world_rig('top_scene')` raises).  What there IS: 129 training
episodes in which the flower's top_scene pixel at frame 0 and the right arm's
end-effector pose at the moment the gripper closed are both recorded.  A
least-squares affine fit of pixel -> EE-xyz on those gives the mapping this
task actually needs -- "where does the gripper go to grasp the thing at pixel
p" -- in the arm's own base frame, at grasp height, with the operator's grasp
offset baked in.  Residual: xy median 11.4 mm, p90 22.2 mm.  The gripper opens
to well over the flower's ~40 mm width, so that error is inside the grasp
tolerance for most attempts, and the routine verifies and retries rather than
trusting it.  Fit lives in assets/flower_px2ee.json; refit with
analysis/fit_px2ee.py if the top camera, the table or the gripper moves.

WHAT IT DOES, per reset:

    1. open gripper, locate the flower in top_scene (scene_features detector)
    2. hover above it, descend to grasp height, close, lift
    3. verify in top_scene that the flower LEFT its spot (else retry once)
    4. hover above the target cell, descend, open, lift, park
    5. verify the flower is within --reset-tol px of the target (else one
       corrective pick-place; then hand over to the operator)

SAFETY.  Every waypoint is IK-solved and then checked against the same
capsule + table gates the policy runs under; a waypoint the gate would scale
(alpha < 1) is REFUSED, not executed.  Moves are streamed by
move_arms_together at parking speed.  Nothing moves without --engage; a dry
run prints every waypoint, its joint solution and its gate verdict.  Failure
never loops silently: after the retry budget the routine prints what it saw
and waits for ENTER.

Used from rollout_policy.py via --auto-reset (see there), or standalone:

    python auto_reset.py --cell B2 [--engage]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from paths import ASSETS_DIR

FIT_PATH = ASSETS_DIR / "flower_px2ee.json"

HOVER_M = 0.06          # above grasp height for approach / carry
PLACE_LIFT_M = 0.004    # release this much above the fitted grasp height
GRASP_SETTLE_S = 0.5
MOVED_TOL_PX = 15       # object must move at least this far to count as picked
IK_TOL_MM = 10.0        # refuse a waypoint whose IK solution misses by more
IK_ITERS = 8            # re-seeded solves per waypoint (each is 20 iterations)
IK_CONVERGED_MM = 0.5


class AutoReset:
    def __init__(self, robots, gates, arm, latest_frames, frame_lock,
                 engage=False, tol_px=12, max_tries=2, cruise_speed=None,
                 log=print):
        self.robots, self.gates, self.arm = robots, gates, arm
        self.latest_frames, self.frame_lock = latest_frames, frame_lock
        self.engage, self.tol_px, self.max_tries = engage, tol_px, max_tries
        self.cruise_speed, self.log = cruise_speed, log
        try:
            import scene_features as sf
            from arm_config import ARM_CONFIG
            from gripper import GRIPPER_CLOSED, GRIPPER_OPEN, command_gripper
            from robot_control import move_arms_together
        except ImportError:
            from . import scene_features as sf
            from .arm_config import ARM_CONFIG
            from .gripper import GRIPPER_CLOSED, GRIPPER_OPEN, command_gripper
            from .robot_control import move_arms_together
        self.sf, self._move, self._grip = sf, move_arms_together, command_gripper
        self.OPEN, self.CLOSED = GRIPPER_OPEN, GRIPPER_CLOSED
        self.n_joints = ARM_CONFIG[arm]["num_joints"]
        fit = json.loads(FIT_PATH.read_text())
        self.W = np.asarray(fit["W_affine"], dtype=float)         # 3x3: [u v 1] -> xyz
        self.q_grasp = np.asarray(fit["q_grasp_wxyz"], dtype=float)
        self.q_grasp /= np.linalg.norm(self.q_grasp)
        self.log(f"[reset] px->EE fit from {fit['n_episodes']} episodes, "
                 f"xy residual median {fit['residual_xy_mm']['median']} mm")

    # ----------------------------------------------------------- perception
    def top_frame(self):
        with self.frame_lock:
            f = self.latest_frames.get("top_scene")
            return f.copy() if f is not None else None

    def locate(self, wait_s=2.0):
        """(u, v) of the flower in top_scene, or None.  Waits briefly for a
        frame that sees it -- the arm may still be clearing the view."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait_s:
            img = self.top_frame()
            if img is not None:
                cx, cy, _a, ok = self.sf.detect_object(img)
                if ok:
                    return np.array([cx, cy], dtype=float)
            time.sleep(0.1)
        return None

    # ----------------------------------------------------------- geometry
    def px_to_xyz(self, uv):
        return np.array([uv[0], uv[1], 1.0]) @ self.W

    def solve(self, xyz):
        """Joint target for the arm at EE pose (xyz, q_grasp), plus the gate
        verdict.  Seeds the IK from the current command reference so the
        solution is the nearby one, as the EE rollout branch does."""
        xyz = np.asarray(xyz, dtype=float)
        ## The coupled IK is a 20-iteration per-tick solver built for
        ## teleop, where the target moves a millimetre per tick.  For a
        ## 60 mm hop it lands 4-8 mm short in one call, so iterate: re-seed
        ## from the last answer until the residual is small.  Leaves the
        ## gates' own command reference untouched.
        seed = self.gates.q_cmd.copy()
        q_full = seed
        for _ in range(IK_ITERS):
            q_full = np.asarray(self.gates.ik.solve(
                q_full, {self.arm: (xyz, self.q_grasp)}), dtype=float)
            if np.linalg.norm(self._fk_xyz(q_full) - xyz) * 1e3 < IK_CONVERGED_MM:
                break
        q = q_full[self.gates.idx[self.arm]][: self.n_joints]
        ## The solver is best-effort: it returns the closest configuration it
        ## found, converged or not.  Check where that configuration actually
        ## puts the end-effector and refuse a waypoint it cannot reach.
        fk_xyz = self._fk_xyz(q_full)
        resid_mm = float(np.linalg.norm(fk_xyz - xyz) * 1e3)
        _q_sent, alpha, info = self._gate_check(q)
        if resid_mm > IK_TOL_MM:
            alpha = 0.0
            info = dict(info or {}, ik_residual_mm=round(resid_mm, 1),
                        reason="IK did not reach the target")
        return q, alpha, info

    def _fk_xyz(self, q_full_driver):
        """EE position the gates' own robot model puts the arm at, in the
        same base frame the px->EE fit was made in."""
        robot = self.gates.robot
        if not hasattr(self, "_ee_idx"):
            from arm_config import ARM_CONFIG
            self._ee_idx = list(robot.links.names).index(ARM_CONFIG[self.arm]["ee_link"])
        q_urdf = self.gates.ik.driver_to_urdf(np.asarray(q_full_driver, dtype=float))
        fk = np.asarray(robot.forward_kinematics(np.asarray(q_urdf, dtype=np.float32)))
        return np.asarray(fk[self._ee_idx, 4:7], dtype=float)   # wxyz_xyz -> xyz

    def _gate_check(self, q):
        """Ask the gates about the FULL move without committing it."""
        saved = self.gates.q_cmd.copy()
        out = self.gates.filter(self.arm, q)
        self.gates.q_cmd[:] = saved          # filter() advances q_cmd; undo
        return out

    # ----------------------------------------------------------- motion
    def _measured(self):
        js = self.robots[self.arm].dxl.joint_states
        return np.asarray(js.position[: self.n_joints], dtype=float)

    def goto(self, xyz, label):
        q, alpha, info = self.solve(xyz)
        self.log(f"[reset]   {label:14s} xyz={np.round(xyz, 3).tolist()} "
                 f"q={np.round(q, 3).tolist()} gate_alpha={alpha:.2f}"
                 + (f" {info}" if info else ""))
        if alpha < 1.0:
            raise RuntimeError(f"gate refuses waypoint {label}: {info}")
        if self.engage:
            self._move({self.arm: self.robots[self.arm]}, {self.arm: q},
                       cruise_speed=self.cruise_speed, verbose=False)
            self.gates.seed(self.arm, self._measured())
        return q

    def gripper(self, pos):
        if self.engage:
            self._grip(self.robots[self.arm], pos)
            time.sleep(GRASP_SETTLE_S)

    # ----------------------------------------------------------- routine
    def pick(self, uv):
        xyz = self.px_to_xyz(uv)
        self.gripper(self.OPEN)
        self.goto(xyz + [0, 0, HOVER_M], "hover(obj)")
        self.goto(xyz, "grasp")
        self.gripper(self.CLOSED)
        self.goto(xyz + [0, 0, HOVER_M], "lift")
        ## Did it come with us?  The spot it was at should now be empty.
        ## (A dry run cannot move anything, so the plan is the result.)
        if not self.engage:
            return True
        after = self.locate(wait_s=1.0)
        moved = after is None or np.linalg.norm(after - uv) > MOVED_TOL_PX
        return moved

    def place(self, uv_target):
        xyz = self.px_to_xyz(uv_target)
        self.goto(xyz + [0, 0, HOVER_M], "hover(target)")
        self.goto(xyz + [0, 0, PLACE_LIFT_M], "place")
        self.gripper(self.OPEN)
        self.goto(xyz + [0, 0, HOVER_M], "retreat")

    def run(self, uv_target, park):
        """Move the flower to uv_target.  `park` is the caller's parking
        routine (arm out of the top camera's view).  Returns final px error,
        or None if the operator had to intervene."""
        uv_target = np.asarray(uv_target, dtype=float)
        for attempt in range(1, self.max_tries + 1):
            if self.engage:
                self.gates.seed(self.arm, self._measured())
            uv = self.locate()
            if uv is None:
                self.log("[reset] flower not visible in top_scene")
                break
            err = np.linalg.norm(uv - uv_target)
            self.log(f"[reset] attempt {attempt}: flower at {np.round(uv, 1).tolist()}, "
                     f"target {uv_target.tolist()}, error {err:.1f} px")
            if err <= self.tol_px:
                self.log(f"[reset] already within {self.tol_px} px -- nothing to do")
                return err
            try:
                if not self.pick(uv):
                    self.log("[reset] pick did not move the object -- retrying")
                    self.gripper(self.OPEN)
                    continue
                self.place(uv_target)
            except RuntimeError as exc:
                self.log(f"[reset] {exc}")
                break
            park()
            if not self.engage:
                self.log("[reset] dry run: plan complete, nothing moved")
                return 0.0
            uv2 = self.locate()
            err2 = None if uv2 is None else float(np.linalg.norm(uv2 - uv_target))
            self.log(f"[reset] after place: {'not visible' if uv2 is None else np.round(uv2, 1).tolist()}"
                     f"  error {'-' if err2 is None else f'{err2:.1f}'} px")
            if err2 is not None and err2 <= self.tol_px:
                return err2
        self.log("[reset] *** could not place the flower automatically.  Fix the "
                 "scene by hand, then press ENTER to continue. ***")
        try:
            input()
        except EOFError:
            pass
        return None


def cell_px(label):
    """(u, v) for a place_grid label such as B2, R3 or M7."""
    try:
        import place_grid as pg
    except ImportError:
        from . import place_grid as pg
    ## cells_for returns the whole lattice (plus whichever R/M families the
    ## labels ask for); pick the one row whose label matches.
    for lbl, x, y, _flag in pg.cells_for([label]):
        if lbl == label:
            return float(x), float(y)
    raise SystemExit(f"unknown placement label {label!r} "
                     f"(lattice A1..D4, R<n>, M1..M12)")


def main():
    """Standalone: bring up cameras + the right arm + gates, move the flower
    to one cell, park.  Validate the routine here before trusting it inside
    a rollout run."""
    import argparse
    import threading

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", required=True, help="place_grid label, e.g. B2, R3, M7")
    ap.add_argument("--engage", action="store_true", help="actually move the arm")
    ap.add_argument("--tol", type=float, default=12.0, help="accept within this many px")
    ap.add_argument("--park", default="forward", help="right-arm pose to park at")
    ap.add_argument("--jax", default="cpu", choices=["cpu", "gpu"])
    args = ap.parse_args()
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)     # show each phase as it happens
    except Exception:
        pass
    say = lambda m: print(m, flush=True)
    say(f"[reset] starting  (cell {args.cell}, {'ENGAGED' if args.engage else 'DRY RUN -- nothing moves'})")
    say("[reset] importing robot/camera stack ...")

    try:
        import rospy
        from camera_manager import CameraConfig, get_active_cameras, setup_cameras, join_camera_workers
        from robot_control import apply_profile_limits, create_and_configure_robots, get_pose, move_arms_together
        from data_col_config import TeleopConfig
        import rollout_policy as rp
    except ImportError:
        import rospy
        from .camera_manager import CameraConfig, get_active_cameras, setup_cameras, join_camera_workers
        from .robot_control import apply_profile_limits, create_and_configure_robots, get_pose, move_arms_together
        from .data_col_config import TeleopConfig
        from . import rollout_policy as rp

    arm = "right"
    target = cell_px(args.cell)
    say(f"[reset] target cell {args.cell} = px {target}")

    ## create_robot() builds the Interbotix core with init_node=False and
    ## expects the caller to own the node -- rollout_policy.py does this too.
    ## Without it the SDK's subscribers/service waits never come up and the
    ## process sits silently: that was the first standalone run (2026-09-19).
    if not rospy.core.is_initialized():
        rospy.init_node("auto_reset", anonymous=True)
    say("[reset] connecting to the right arm over ROS (blocks until /puppet_right/xs_sdk answers) ...")
    robots = create_and_configure_robots([arm])
    apply_profile_limits(robots, TeleopConfig(control_dt=0.02), arm_names=[arm])
    say("[reset] building safety gates + IK (JAX compile, ~2-20 s) ...")
    gates = rp.SafetyGates(arm, [arm], jax_platform=args.jax)
    say(f"[gates] {gates.summary()}")

    camera_shutdown = threading.Event()
    frame_lock = threading.Lock()
    active = get_active_cameras(arm, CameraConfig(top_active=True, low_active=False))
    say(f"[reset] opening cameras {active} ...")
    latest_frames = {c: None for c in active}
    latest_ts = {c: None for c in active}
    pipelines = setup_cameras(active, camera_shutdown, frame_lock, latest_frames, latest_ts)
    say("[reset] cameras up")

    def park():
        if args.engage:
            move_arms_together({arm: robots[arm]}, {arm: get_pose(arm, args.park)}, verbose=False)

    try:
        park()
        time.sleep(1.0)                      # let the top camera see the table
        r = AutoReset(robots, gates, arm, latest_frames, frame_lock,
                      engage=args.engage, tol_px=args.tol)
        err = r.run(target, park=park)
        print(f"[reset] done: final error {err if err is None else round(err, 1)} px")
    finally:
        camera_shutdown.set()
        join_camera_workers()
        for _name, p in (pipelines or {}).items():      # dict, as rollout_policy treats it
            try:
                p.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
