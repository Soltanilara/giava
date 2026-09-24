"""Teleop frame/tracking debug instrument — record ground truth, replay, compare.

The tool that ends frame-mismatch guesswork:

  * LIVE POSES  — headset + both controllers in the app world (post-ingestion),
    fast row (rounded, ~20 Hz) and a slow 2 Hz row for comfortable reading.
  * RECORD      — one button records the FULL pose trajectory of all three
    devices (with timestamps and the session-yaw calibration stamped at start).
    Trajectories save to .npz and reload later — so the exact same recording
    can be replayed in simulation first and on the real arms afterwards.
  * REPLAY      — the matching arm (left ctrl -> left arm, right -> right,
    headset -> middle) is driven through the recorded trajectory by the REAL
    deployed solver (CoupledStudyIK) and the REAL interbotix driver, at a
    selectable control rate (5/10/20/50 Hz), auto or one step per click.
  * COMPARE     — per step, three rows: expected pose change from start,
    the end effector's measured change, and signed percent difference.
    Plus the commanded joint step (max joint and degrees) so "this motion
    demands a 90 deg forearm swing" is visible directly.
  * 3D VIEW     — viser shows giava.urdf mirroring the MEASURED joints (it is
    a dashboard here, not a solver), the expected EE path drawn at the arm
    (green) and the achieved path (orange).

Sim vs real is decided by the ROS launch, not by this script:
    real : roslaunch av_aloha 3arms_teleop.launch
    sim  : roslaunch av_aloha 3arms_sim.launch     (same driver, use_sim)

Run:  python teleop_debug_tool.py            # needs the driver running
      python teleop_debug_tool.py --no-robots   # headset/recording only
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse

from paths import URDF_PATH as _PATHS_URDF
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transform_utils import session_yaw_remap  # noqa: E402

TRAJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories")
ARMS = ("left", "right", "middle")
SRC_OF = {"left": "l", "right": "r", "middle": "h"}
BASE_REMAP = np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]])  # (fwd,left,up)->robot


# --------------------------------------------------------------------------- #
# trajectory container
# --------------------------------------------------------------------------- #
@dataclass
class Trajectory:
    """All three device streams plus the calibration stamped at record start."""

    t: List[float] = field(default_factory=list)
    pos: Dict[str, List[np.ndarray]] = field(
        default_factory=lambda: {"h": [], "l": [], "r": []})
    quat: Dict[str, List[np.ndarray]] = field(
        default_factory=lambda: {"h": [], "l": [], "r": []})
    remap: Optional[np.ndarray] = None  # session_yaw_remap at record start

    def append(self, t, d):
        self.t.append(t)
        for k, p, q in (("h", d.h_pos, d.h_quat), ("l", d.l_pos, d.l_quat),
                        ("r", d.r_pos, d.r_quat)):
            self.pos[k].append(np.asarray(p, float).copy())
            self.quat[k].append(np.asarray(q, float).copy())

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, t=np.asarray(self.t), remap=self.remap,
                 **{f"pos_{k}": np.asarray(v) for k, v in self.pos.items()},
                 **{f"quat_{k}": np.asarray(v) for k, v in self.quat.items()})

    @staticmethod
    def load(path) -> "Trajectory":
        d = np.load(path)
        tr = Trajectory()
        tr.t = list(d["t"])
        tr.remap = np.asarray(d["remap"])
        for k in ("h", "l", "r"):
            tr.pos[k] = list(d[f"pos_{k}"])
            tr.quat[k] = list(d[f"quat_{k}"])
        return tr


def resample(traj: Trajectory, src: str, hz: float):
    """Time-resample one device stream to the chosen control rate.

    Position: linear interpolation.  Rotation: slerp.  Returns (pos (N,3),
    quat_xyzw (N,4)) starting exactly at the recorded start pose."""
    t = np.asarray(traj.t) - traj.t[0]
    p, q = compose_world(traj, src)
    # drop tracking-dropout samples (exact zeros) so a mid-recording loss does
    # not interpolate the replay through a point a metre away
    ok = np.linalg.norm(np.asarray(traj.pos[src]), axis=1) > 1e-6
    if not ok.all():
        t, p, q = t[ok], p[ok], q[ok]
    if len(t) < 2:
        return p, q
    tn = np.arange(0.0, t[-1], 1.0 / hz)
    out_p = np.stack([np.interp(tn, t, p[:, i]) for i in range(3)], axis=1)
    # Slerp needs strictly increasing keyframes; dedupe identical timestamps.
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    out_q = Slerp(t[keep], R.from_quat(q[keep]))(np.clip(tn, t[0], t[keep][-1])).as_quat()
    return out_p, out_q


def expected_series(traj, src, hz, ee_pos0, ee_rot0, scale):
    """Expected EE poses in the ROBOT frame for the whole resampled trajectory."""
    p, q = resample(traj, src, hz)
    S = traj.remap if traj.remap is not None else BASE_REMAP
    R0 = R.from_quat(q[0]).as_matrix()
    pos_e, rot_e = [], []
    for k in range(len(p)):
        dR = R.from_quat(q[k]).as_matrix() @ R0.T
        pos_e.append(ee_pos0 + scale * (S @ (p[k] - p[0])))
        rot_e.append((S @ dR @ S.T) @ ee_rot0)
    return np.asarray(pos_e), rot_e


# Head-relative hand composition: OFF by default (2026-08 hardware test --
# with composition on, the gripper arms were HIGHLY affected by head motion;
# raw streams behave correctly).  Earlier evidence pointed the other way, so
# the switch stays: GIAVA_HANDS_HEAD_RELATIVE=1 re-enables composition for
# both live teleop and replay if the stand-up symptom ever returns.
HANDS_HEAD_RELATIVE = os.environ.get("GIAVA_HANDS_HEAD_RELATIVE", "0") == "1"
MIDDLE_BUTTON = os.environ.get("GIAVA_MIDDLE_BUTTON", "y")  # "y" or "either"


def compose_world(traj_or_dev, src):
    """World pose of one device stream.  With HANDS_HEAD_RELATIVE off (the
    default), hand streams are used raw; 'h' always passes through."""
    if not HANDS_HEAD_RELATIVE or src == "h":
        if isinstance(traj_or_dev, dict):
            return traj_or_dev[src]
        return (np.asarray(traj_or_dev.pos[src]),
                np.asarray(traj_or_dev.quat[src]))
    if isinstance(traj_or_dev, dict):  # live device dict {src: (pos, quat)}
        if src == "h":
            return traj_or_dev["h"]
        hp, hq = traj_or_dev["h"]
        rp, rq = traj_or_dev[src]
        Rh = R.from_quat(hq)
        return hp + Rh.apply(rp), (Rh * R.from_quat(rq)).as_quat()
    traj = traj_or_dev  # Trajectory: vectorized over all samples
    if src == "h":
        return np.asarray(traj.pos["h"]), np.asarray(traj.quat["h"])
    hp = np.asarray(traj.pos["h"])
    Rh = R.from_quat(np.asarray(traj.quat["h"]))
    p = hp + Rh.apply(np.asarray(traj.pos[src]))
    q = (Rh * R.from_quat(np.asarray(traj.quat[src]))).as_quat()
    return p, q


def rotvec_deg(rot, rot0):
    """Delta rotation rot·rot0^-1 as world rotation-vector, degrees.
    Components are about robot x (pitch), y (roll), z (yaw)."""
    return np.degrees(R.from_matrix(np.asarray(rot) @ np.asarray(rot0).T).as_rotvec())


def valid_pose(p):
    """The headset reports (0,0,0) when it loses a controller (seen with hands
    above the head / out of the tracking volume).  Exactly zero is never a real
    pose in this rig, so treat it as tracking-lost."""
    return float(np.linalg.norm(np.asarray(p, float))) > 1e-6


def fmt3(v, nd=3):
    return " ".join(f"{x:+.{nd}f}" for x in np.asarray(v, float))


def pct_row(e, a, tol):
    out = []
    for ev, av in zip(np.ravel(e), np.ravel(a)):
        out.append("  --  " if abs(ev) < tol else f"{(av - ev) / abs(ev) * 100:+5.0f}%")
    return " ".join(out)


# --------------------------------------------------------------------------- #
# main tool
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", default=str(_PATHS_URDF))
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--no-robots", action="store_true",
                    help="headset display + recording only (no ROS needed)")
    ap.add_argument("--no-headset", action="store_true",
                    help="replay saved trajectories only")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS,
                    help="which arms are launched (match the roslaunch "
                         "use_left/use_right/use_middle selection)")
    args = ap.parse_args()

    import viser
    from viser.extras import ViserUrdf
    from yourdfpy import URDF

    # ---- headset ---------------------------------------------------------- #
    headset = None
    if not args.no_headset:
        from webrtc_headset import WebRTCHeadset
        headset = WebRTCHeadset()
        headset.run_in_thread()
        print("waiting for headset channel", end="", flush=True)
        t0 = time.time()
        while not headset.data_channel_open:
            print(".", end="", flush=True)
            time.sleep(0.5)
            if time.time() - t0 > 45:
                print("\nno headset after 45 s -- continuing in replay-only mode")
                headset = None
                break
        if headset is not None:
            print(" ok")

    # ---- robots (optional) ------------------------------------------------ #
    robots = coupled_ik = robot = arm_data = None
    if not args.no_robots:
        try:
            import rospy
            from arm_config import ARM_CONFIG
            from robot_control import (build_robot_model,
                                       create_and_configure_robots,
                                       get_joint_positions,
                                       interpolate_to_pose,
                                       move_to_named_pose)
            from study_ik import CoupledStudyIK
            if not rospy.core.is_initialized():
                rospy.init_node("teleop_debug_tool", anonymous=True)
            robots = create_and_configure_robots(tuple(args.arms))
            robot, arm_data = build_robot_model("av")
            coupled_ik = CoupledStudyIK(
                robot, args.urdf,
                ee_links={a: ARM_CONFIG[a]["ee_link"] for a in ARMS})
            q_meas = np.asarray(
                robot.joint_var_cls(0).default_factory(), dtype=float).copy()
            for a in args.arms:
                q_meas[arm_data[a]["joint_indices"]] = get_joint_positions(robots[a])
            q_template = q_meas.copy()
            print("compiling coupled IK...")
            coupled_ik.warmup(q_meas.astype(np.float32))
            # All launched arms start from the forward pose (same behavior as
            # data_collection).  Per-arm try/except so ONE failing arm is
            # reported by name instead of failing silently -- in sim, joints
            # boot at driver 0.0, which for the multiturn middle waist displays
            # as a 180 deg rotation until this reset runs.
            # The real middle arm may still carry a Homing_Offset in the waist
            # servo (sim never does).  A nonzero register shifts every reported
            # waist angle, so FK/anchoring computes a wrong EE pose on the real
            # arm while sim behaves perfectly -- exactly a "sim great, real
            # wrong" split.  Refuse to let that pass silently.
            if "middle" in args.arms:
                try:
                    from robot_control import read_middle_waist_shift
                    _shift = read_middle_waist_shift(robots["middle"])
                    if abs(_shift) > 1e-6:
                        print("=" * 60)
                        print(f"!! middle waist Homing_Offset = {_shift:+.3f} rad")
                        print("!! this shifts all reported waist angles; the sim")
                        print("!! has no such offset. Revert before comparing:")
                        print("!!     python set_waist_homing_offset.py --degrees 0")
                        print("=" * 60)
                except Exception:
                    pass
            for a in args.arms:
                try:
                    print(f"[reset] {a} -> forward pose...")
                    move_to_named_pose(robots[a], a, "forward")
                    from robot_control import get_pose
                    tgt = np.asarray(get_pose(a, "forward"), float)
                    cur = np.asarray(get_joint_positions(robots[a]), float)[: len(tgt)]
                    diff = np.abs(cur - tgt)
                    if a == "middle":  # waist compares modulo 2pi
                        diff[0] = abs((cur[0] - tgt[0] + np.pi) % (2 * np.pi) - np.pi)
                    bad = np.flatnonzero(diff > 0.05)
                    if len(bad):
                        names = robots[a].arm.group_info.joint_names
                        for j in bad:
                            print(f"[reset] {a}.{names[j]} MISSED target: "
                                  f"cmd {tgt[j]:+.3f} measured {cur[j]:+.3f} "
                                  f"(driver limit or rejected command)")
                except Exception as exc:
                    print(f"[reset] {a} FAILED: {exc}")
            print("robots ready (forward pose)")
        except Exception as exc:
            print(f"ROBOTS UNAVAILABLE ({exc}) -- display/record mode only")
            robots = None

    def read_q():
        # Unlaunched arms keep their model-default joints; launched arms read live.
        q = q_template.copy()
        for a in args.arms:
            q[arm_data[a]["joint_indices"]] = get_joint_positions(robots[a])
        return q

    def ee_pose(q, arm):
        import jaxlie
        fk = robot.forward_kinematics(
            coupled_ik.driver_to_urdf(np.asarray(q, np.float32)))
        se3 = jaxlie.SE3(fk[arm_data[arm]["ee_index"]])
        return (np.asarray(se3.translation(), float),
                np.asarray(R.from_quat(np.roll(np.asarray(se3.rotation().wxyz), -1)).as_matrix()))

    # ---- viser dashboard --------------------------------------------------- #
    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1)
    urdf_vis = ViserUrdf(server, URDF.load(args.urdf), root_node_name="/robot")

    with server.gui.add_folder("Live poses (app world, m)"):
        fast = {k: server.gui.add_text(lbl, "-", disabled=True)
                for k, lbl in (("h", "Headset"), ("l", "Left controller"),
                               ("r", "Right controller"))}
        slow = {k: server.gui.add_text(f"{lbl} (2 Hz)", "-", disabled=True)
                for k, lbl in (("h", "Headset"), ("l", "Left ctrl"),
                               ("r", "Right ctrl"))}

    with server.gui.add_folder("Record"):
        traj_name = server.gui.add_text("trajectory name", "")
        rec_btn = server.gui.add_button("start recording")
        stop_btn = server.gui.add_button("stop + save")
        rec_status = server.gui.add_text("status", "idle", disabled=True)
        os.makedirs(TRAJ_DIR, exist_ok=True)
        files = sorted(os.listdir(TRAJ_DIR)) or ["(none)"]
        file_dd = server.gui.add_dropdown("saved trajectories", tuple(files),
                                          initial_value=files[-1])
        load_btn = server.gui.add_button("load selected")

    with server.gui.add_folder("Replay"):
        arm_dd = server.gui.add_dropdown("arm", tuple(args.arms),
                                         initial_value=args.arms[0])
        rate_dd = server.gui.add_dropdown("control rate [Hz]",
                                          ("5", "10", "20", "50"), initial_value="20")
        scale_sl = server.gui.add_slider("position scale", 0.2, 2.0, 0.05, 1.0)
        live_cb = server.gui.add_checkbox("LIVE teleop (arms follow devices)", False)
        btn_cb = server.gui.add_checkbox(
            "BUTTON teleop (X=left, A=right, Y=middle) + auto-record", False)
        prime_btn = server.gui.add_button("prime (anchor + draw path)")
        step_btn = server.gui.add_button("step >")
        auto_btn = server.gui.add_button("auto run")
        halt_btn = server.gui.add_button("halt")
        rp_status = server.gui.add_text("state", "idle", disabled=True)

    with server.gui.add_folder("Step compare (delta from start)"):
        hdr = server.gui.add_markdown(
            "`pos dx dy dz [m] | rot x(pitch) y(roll) z(yaw) [deg]`")
        row_e = server.gui.add_text("expected", "-", disabled=True)
        row_a = server.gui.add_text("achieved", "-", disabled=True)
        row_p = server.gui.add_text("% diff", "-", disabled=True)
        row_j = server.gui.add_text("joint step", "-", disabled=True)

    with server.gui.add_folder("Recovery"):
        fwd_btn = server.gui.add_button("all arms -> forward pose")
        reconf_btn = server.gui.add_button("reconfigure arm (re-seed from rest)")

    # ---- joint jog + pose capture ------------------------------------------ #
    # Sliders take their ranges from the RUNNING driver's group_info, so the
    # limits you see are what THIS launch (sim or real) actually enforces --
    # run the tool against both and compare the ranges to expose any
    # sim-vs-real mismatch directly.
    POSES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "poses_custom.json")
    jog_sliders: Dict[str, list] = {}
    jog_limits: Dict[str, tuple] = {}
    if robots is not None:
        with server.gui.add_folder("Joint jog + pose capture"):
            jog_en = server.gui.add_checkbox("jog enable (commands on slider move)", False)
            jog_status = server.gui.add_text("jog", "-", disabled=True)
            for a in args.arms:
                gi = robots[a].arm.group_info
                lo = np.asarray(gi.joint_lower_limits, float)
                hi = np.asarray(gi.joint_upper_limits, float)
                cur = get_joint_positions(robots[a])
                print(f"[limits] {a}: " + "  ".join(
                    f"{n}[{l:+.2f},{u:+.2f}]" for n, l, u in
                    zip(gi.joint_names, lo, hi)))
                with server.gui.add_folder(f"jog: {a}"):
                    sl = []
                    for j, name in enumerate(gi.joint_names):
                        sl.append(server.gui.add_slider(
                            f"{a}.{name}", float(lo[j]), float(hi[j]), 0.005,
                            float(np.clip(cur[j], lo[j], hi[j]))))
                    jog_sliders[a] = sl
                    jog_limits[a] = (lo, hi)

            def _jog_cb(arm):
                def cb(_):
                    if not jog_en.value or robots is None:
                        return
                    now = time.time()
                    if now - st.get("last_jog", 0.0) < 0.10:
                        return
                    st["last_jog"] = now
                    st["mode"] = "idle"
                    q_t = [x.value for x in jog_sliders[arm]]
                    try:
                        robots[arm].arm.set_joint_positions(
                            q_t, moving_time=0.4, accel_time=0.15, blocking=False)
                        jog_status.value = f"{arm} <- {fmt3(q_t, 2)}"
                    except Exception as exc:
                        jog_status.value = f"{arm} jog failed: {exc}"
                return cb
            for a in args.arms:
                for x in jog_sliders[a]:
                    x.on_update(_jog_cb(a))

            sync_btn = server.gui.add_button("sync sliders from measured")
            def _sync(_):
                for a in args.arms:
                    cur = get_joint_positions(robots[a])
                    lo, hi = jog_limits[a]
                    for j, x in enumerate(jog_sliders[a]):
                        x.value = float(np.clip(cur[j], lo[j], hi[j]))
                jog_status.value = "sliders = measured joints"
            sync_btn.on_click(_sync)
            # enabling jog first syncs, so the arm never jumps to stale sliders
            jog_en.on_update(lambda _: _sync(None) if jog_en.value else None)

            cap_arm_dd = server.gui.add_dropdown(
                "capture/goto arm", tuple(args.arms),
                initial_value=("middle" if "middle" in args.arms else args.arms[0]))
            pose_name = server.gui.add_text("pose name", "my_pose")
            cap_btn = server.gui.add_button("capture pose (measured joints)")
            def _capture(_):
                arm = cap_arm_dd.value
                q = list(map(float, get_joint_positions(robots[arm])))
                book = {}
                if os.path.exists(POSES_FILE):
                    book = json.load(open(POSES_FILE))
                book.setdefault(arm, {})[pose_name.value] = q
                json.dump(book, open(POSES_FILE, "w"), indent=2)
                saved_dd.options = tuple(_pose_keys())
                jog_status.value = f"saved {arm}:{pose_name.value}"
                print(f"[pose] {arm}:{pose_name.value} = np.array({np.round(q, 4).tolist()})")
            cap_btn.on_click(_capture)

            def _pose_keys():
                if not os.path.exists(POSES_FILE):
                    return ["(none)"]
                book = json.load(open(POSES_FILE))
                return [f"{a}:{n}" for a in book for n in book[a]] or ["(none)"]
            saved_dd = server.gui.add_dropdown("saved poses", tuple(_pose_keys()))
            goto_btn = server.gui.add_button("go to saved pose")
            def _goto(_):
                if ":" not in saved_dd.value:
                    return
                arm, name = saved_dd.value.split(":", 1)
                if arm not in args.arms:
                    jog_status.value = f"{arm} not launched"
                    return
                q = np.asarray(json.load(open(POSES_FILE))[arm][name], float)
                st["mode"] = "idle"
                jog_status.value = f"moving {arm} -> {name}..."
                try:
                    # includes the middle waist nearest-2pi frame handling
                    interpolate_to_pose(robots[arm], arm, q)
                    _sync(None)
                    jog_status.value = f"{arm} at {name}"
                except Exception as exc:
                    jog_status.value = f"goto failed: {exc}"
            goto_btn.on_click(_goto)

    # ---- shared state ------------------------------------------------------ #
    st = {"recording": None, "rec_at": None, "traj": None, "mode": "idle", "k": 0,
          "live": None, "last_dev": None,
          "wp": None, "exp_p": None, "exp_R": None, "ee0": None,
          "q_cmd": None, "next_due": 0.0, "achieved": [], "step_req": False,
          "last_head": None}

    # 3-2-1 countdown: click, walk to your start position, recording begins
    # when the counter hits zero (handled in the main loop).
    rec_btn.on_click(lambda _: st.update(rec_at=time.time() + 3.0, recording=None))
    def _stamp_remap():
        if st["recording"] is not None and st["last_head"] is not None:
            hp = np.eye(4)
            hp[:3, :3] = R.from_quat(st["last_head"][1]).as_matrix()
            rm = session_yaw_remap(hp, BASE_REMAP)
            st["recording"].remap = rm if rm is not None else BASE_REMAP
            rec_status.value = "RECORDING"

    def _finish_recording(prefix="traj"):
        tr = st["recording"]
        st["recording"] = None
        st["rec_at"] = None
        if tr is None or len(tr.t) < 5:
            rec_status.value = "nothing recorded"
            return
        nm = "".join(c if c.isalnum() or c in "-_" else "_"
                     for c in traj_name.value.strip())
        name = (f"{nm}_{time.strftime('%H%M%S')}.npz" if nm
                else f"{prefix}_{time.strftime('%H%M%S')}.npz")
        tr.save(os.path.join(TRAJ_DIR, name))
        st["traj"] = tr
        rec_status.value = f"saved {name} ({len(tr.t)} samples)"
        file_dd.options = tuple(sorted(os.listdir(TRAJ_DIR)))
        file_dd.value = name
    stop_btn.on_click(lambda _: _finish_recording())

    def _load(_):
        try:
            st["traj"] = Trajectory.load(os.path.join(TRAJ_DIR, file_dd.value))
            rec_status.value = f"loaded {file_dd.value} ({len(st['traj'].t)} samples)"
        except Exception as exc:
            rec_status.value = f"load failed: {exc}"
    load_btn.on_click(_load)

    def _prime(_):
        if st["traj"] is None:
            rp_status.value = "no trajectory loaded"
            return
        if robots is None:
            rp_status.value = "no robots connected"
            return
        arm = arm_dd.value
        hz = float(rate_dd.value)
        q = read_q()
        # re-anchor both our command state and the driver's reference
        st["q_cmd"] = q.copy()
        for a in args.arms:
            iface = robots[a].arm
            iface.joint_commands = list(get_joint_positions(robots[a]))
        p0, R0 = ee_pose(q, arm)
        st["ee0"] = (p0, R0)
        st["exp_p"], st["exp_R"] = expected_series(
            st["traj"], SRC_OF[arm], hz, p0, R0, scale_sl.value)
        st["wp"] = len(st["exp_p"])
        st["k"] = 0
        st["achieved"] = [p0.copy()]
        st["log_rows"] = []
        pts = st["exp_p"][:: max(1, len(st["exp_p"]) // 400)]
        try:
            server.scene.add_spline_catmull_rom(
                "/expected_path", points=pts, color=(40, 200, 80), line_width=3.0)
        except Exception:
            server.scene.add_point_cloud(
                "/expected_path", points=pts,
                colors=np.tile([[40, 200, 80]], (len(pts), 1)), point_size=0.004)
        st["mode"] = "primed"
        rp_status.value = f"primed: {st['wp']} steps @ {hz:g} Hz"
    prime_btn.on_click(_prime)

    def _btn_toggle(_):
        if btn_cb.value:
            if robots is None or headset is None:
                btn_cb.value = False
                rp_status.value = "need robots + headset for button teleop"
                return
            live_cb.value = False
            st["mode"] = "buttons"
            st["btn"] = {"S": None, "anchors": {}, "active": set()}
            rp_status.value = "BUTTON teleop armed: hold X / A"
        elif st["mode"] == "buttons":
            st["mode"] = "idle"
            rp_status.value = "button teleop off"
    btn_cb.on_update(_btn_toggle)

    def _live_toggle(_):
        if live_cb.value:
            btn_cb.value = False
        if not live_cb.value:
            if st["mode"] == "live":
                st["mode"] = "idle"
                rp_status.value = "live teleop off"
            return
        if robots is None or st["last_dev"] is None:
            live_cb.value = False
            rp_status.value = "need robots + headset for live teleop"
            return
        # Clutch: anchor every launched arm to its device RIGHT NOW, stamp the
        # session-yaw calibration from the current head pose, and re-anchor the
        # driver references.  Motion is then deltas-from-here at the chosen
        # scale -- the compare rows show expected vs achieved live, so the
        # scaling is visible as numbers while you move.
        q = read_q()
        st["q_cmd"] = q.copy()
        for a in args.arms:
            robots[a].arm.joint_commands = list(get_joint_positions(robots[a]))
        hp = np.eye(4)
        hp[:3, :3] = R.from_quat(st["last_dev"]["h"][1]).as_matrix()
        S = session_yaw_remap(hp, BASE_REMAP)
        if S is None:
            S = BASE_REMAP
        anchors = {}
        for a in args.arms:
            dev_p, dev_q = compose_world(st["last_dev"], SRC_OF[a])
            ee_p, ee_R = ee_pose(q, a)
            anchors[a] = (np.asarray(dev_p).copy(),
                          R.from_quat(dev_q).as_matrix(), ee_p, ee_R)
        st["live"] = {"S": S, "anchors": anchors}
        st["mode"] = "live"
        st["next_due"] = 0.0
        rp_status.value = "LIVE (deltas from anchor; toggle off to stop)"
    live_cb.on_update(_live_toggle)

    step_btn.on_click(lambda _: st.update(step_req=True))
    auto_btn.on_click(lambda _: st.update(mode="auto")
                      if st["mode"] in ("primed", "stepping") else None)
    def _halt(_):
        st["mode"] = "idle"
        live_cb.value = False
        rp_status.value = "halted"
    halt_btn.on_click(_halt)

    def _go_forward(_):
        if robots is None:
            return
        from robot_control import move_to_named_pose as _mtnp
        st["mode"] = "idle"
        for a in args.arms:
            try:
                rp_status.value = f"resetting {a}..."
                _mtnp(robots[a], a, "forward")
            except Exception as exc:
                rp_status.value = f"{a} reset FAILED: {exc}"
                return
        # re-anchor command state and driver references to the new pose
        st["q_cmd"] = read_q()
        for a in args.arms:
            robots[a].arm.joint_commands = list(get_joint_positions(robots[a]))
        rp_status.value = "all arms at forward pose"
    fwd_btn.on_click(_go_forward)

    def _reconfigure(_):
        """Re-seed escape from a folded configuration (validated: re-solving
        from rest recovered a stuck arm 86mm -> 6.5mm where no weight could).
        Solves the CURRENT ee target from the rest pose and glides there."""
        if robots is None:
            return
        arm = arm_dd.value
        q = read_q()
        p_now, R_now = ee_pose(q, arm)
        rest = np.asarray(robot.joint_var_cls(0).default_factory(), np.float32).copy()
        wxyz = np.roll(R.from_matrix(R_now).as_quat(), 1)
        q_alt = coupled_ik.solve(rest, {arm: (p_now, wxyz)})
        idx = arm_data[arm]["joint_indices"]
        cur = get_joint_positions(robots[arm])
        target = np.asarray(q_alt)[idx]
        n = max(2, int(np.ceil(np.max(np.abs(target - cur)) / 0.08)))
        for w in np.linspace(cur, target, n + 1)[1:]:
            robots[arm].arm.set_joint_positions(
                w.tolist(), moving_time=0.25, accel_time=0.1, blocking=True)
        st["q_cmd"] = read_q()
        rp_status.value = f"{arm} reconfigured ({n} waypoints)"
    reconf_btn.on_click(_reconfigure)

    def _execute_step():
        arm = arm_dd.value
        k = st["k"]
        pos_t = st["exp_p"][k]
        wxyz_t = np.roll(R.from_matrix(st["exp_R"][k]).as_quat(), 1)
        q_prev = st["q_cmd"]
        q_new = np.asarray(coupled_ik.solve(
            q_prev.astype(np.float32), {arm: (pos_t, wxyz_t)}), float)
        hz = float(rate_dd.value)
        mt = min(max(1.2 / hz, 0.1), 0.5)
        idx = arm_data[arm]["joint_indices"]
        ref = np.asarray(robots[arm].arm.get_joint_commands(), float)[: len(idx)]
        step_lim = 0.9 * 3.14159 * mt
        cmd = np.clip(q_new[idx], ref - step_lim, ref + step_lim)
        robots[arm].arm.set_joint_positions(
            cmd.tolist(), moving_time=mt, accel_time=0.5 * mt, blocking=False)
        dq = q_new[idx] - q_prev[idx]
        jmax = int(np.argmax(np.abs(dq)))
        q_prev[idx] = cmd
        st["q_cmd"] = q_prev
        st["k"] = k + 1

        # measure + compare (measured joints; FK through the same model)
        q_m = read_q()
        p_m, R_m = ee_pose(q_m, arm)
        p0, R0 = st["ee0"]
        e_p = st["exp_p"][k] - p0
        e_r = rotvec_deg(st["exp_R"][k], R0)
        a_p = p_m - p0
        a_r = rotvec_deg(R_m, R0)
        row_e.value = f"{fmt3(e_p)} | {fmt3(e_r, 1)}"
        row_a.value = f"{fmt3(a_p)} | {fmt3(a_r, 1)}"
        row_p.value = (pct_row(e_p, a_p, 0.002) + " | " + pct_row(e_r, a_r, 1.0))
        row_j.value = (f"step {k + 1}/{st['wp']}  max joint "
                       f"{np.degrees(np.abs(dq).max()):.1f} deg (idx {jmax})")
        st["achieved"].append(p_m.copy())
        st.setdefault("log_rows", []).append(
            np.concatenate([[time.time(), k], e_p, a_p, e_r, a_r]))
        if len(st["achieved"]) % 10 == 0:
            pts = np.asarray(st["achieved"])
            server.scene.add_point_cloud(
                "/achieved_path", points=pts,
                colors=np.tile([[240, 140, 40]], (len(pts), 1)), point_size=0.004)

    # ---- main loop --------------------------------------------------------- #
    print(f"dashboard: http://localhost:{args.port}")
    slow_t = 0.0
    tick = 0
    while True:
        loop_t0 = time.time()
        if headset is not None:
            d = headset.receive_data()
            if d is not None:
                st["last_head"] = (np.asarray(d.h_pos, float),
                                   np.asarray(d.h_quat, float))
                st["last_dev"] = {
                    "h": (np.asarray(d.h_pos, float), np.asarray(d.h_quat, float)),
                    "l": (np.asarray(d.l_pos, float), np.asarray(d.l_quat, float)),
                    "r": (np.asarray(d.r_pos, float), np.asarray(d.r_quat, float)),
                }
                st["btn_now"] = {"l": bool(d.l_button_one), "r": bool(d.r_button_one),
                                 "y": bool(d.l_button_two)}
                st["lost"] = {k: not valid_pose(st["last_dev"][k][0])
                              for k in ("h", "l", "r")}
                fz = st.setdefault("frozen", {k: [None, 0] for k in ("h", "l", "r")})
                for k in ("h", "l", "r"):
                    pv = st["last_dev"][k][0]
                    if fz[k][0] is not None and np.array_equal(pv, fz[k][0]):
                        fz[k][1] += 1
                    else:
                        fz[k][1] = 0
                    fz[k][0] = pv.copy()
                for k, p in (("h", d.h_pos), ("l", d.l_pos), ("r", d.r_pos)):
                    fast[k].value = fmt3(p) + (
                        "  << TRACKING LOST" if not valid_pose(p) else "")
                if time.time() - slow_t > 0.5:
                    slow_t = time.time()
                    for k, p in (("h", d.h_pos), ("l", d.l_pos), ("r", d.r_pos)):
                        slow[k].value = fmt3(p, 2)
                if st["rec_at"] is not None:
                    left_s = st["rec_at"] - time.time()
                    if left_s > 0:
                        rec_status.value = f"recording in {left_s:0.1f} s ..."
                    else:
                        st["rec_at"] = None
                        st["recording"] = Trajectory()
                        _stamp_remap()
                if st["recording"] is not None:
                    if st["recording"].remap is None:
                        _stamp_remap()
                    st["recording"].append(time.time(), d)
                    rec_status.value = f"RECORDING ({len(st['recording'].t)})"

        if robots is not None and tick % 5 == 0:
            try:
                urdf_vis.update_cfg(
                    coupled_ik.driver_to_urdf(read_q().astype(np.float32)))
            except Exception:
                pass

        # Changing the position scale mid-session must NOT jump the targets:
        # target = ee0 + scale*(p - p0), so a scale change multiplies the whole
        # accumulated delta (0.8 -> 0.25 = an instant 70% target shift).
        # Re-anchor every engaged arm at the moment the slider changes; motion
        # continues from the current pose at the new scale, no discontinuity.
        if st["mode"] in ("live", "buttons") and st.get("last_scale") != scale_sl.value:
            if st.get("last_scale") is not None:
                anchors = (st["live"]["anchors"] if st["mode"] == "live"
                           else st["btn"]["anchors"])
                engaged = (list(anchors) if st["mode"] == "live"
                           else list(st["btn"]["active"]))
                for a in engaged:
                    if st["last_dev"] is None:
                        break
                    dev_p, dev_q = compose_world(st["last_dev"], SRC_OF[a])
                    ee_p, ee_R = ee_pose(st["q_cmd"], a)
                    anchors[a] = (np.asarray(dev_p).copy(),
                                  R.from_quat(dev_q).as_matrix(), ee_p, ee_R)
                rp_status.value = (f"scale -> {scale_sl.value:.2f}: re-anchored, "
                                   "no jump")
            st["last_scale"] = scale_sl.value
        elif st.get("last_scale") is None:
            st["last_scale"] = scale_sl.value

        if (robots is not None and st["mode"] == "buttons"
                and st["last_dev"] is not None and st.get("btn_now") is not None):
            b = st["btn_now"]
            want = set()
            if b["l"] and "left" in args.arms:
                want.add("left")
            if b["r"] and "right" in args.arms:
                want.add("right")
            if "middle" in args.arms and (
                    b.get("y") if MIDDLE_BUTTON == "y" else (b["l"] or b["r"])):
                want.add("middle")
            bs = st["btn"]
            # session start: first button press stamps calibration + recording
            if want and not bs["active"]:
                hp = np.eye(4)
                hp[:3, :3] = R.from_quat(st["last_dev"]["h"][1]).as_matrix()
                S = session_yaw_remap(hp, BASE_REMAP)
                bs["S"] = S if S is not None else BASE_REMAP
                st["recording"] = Trajectory()
                st["recording"].remap = bs["S"]
                rp_status.value = "teleop ENABLED (recording)"
            # refuse to clutch an arm to a frozen (asleep/untracked) device --
            # the probe measured 0.5 m of drift on re-acquire, which is the
            # "arm drifts down while my hand is still" failure at startup.
            fz = st.get("frozen", {})
            lost = st.get("lost", {})
            for a in sorted(want - bs["active"]):
                if fz.get(SRC_OF[a], [None, 0])[1] > 25 or lost.get(SRC_OF[a]):
                    want.discard(a)
                    rp_status.value = (f"{a}: device "
                                       f"{'LOST (reads zeros)' if lost.get(SRC_OF[a]) else 'frozen (asleep?)'}"
                                       " -- bring it into view, then press again")
            # a frozen/lost device mid-session: HOLD that arm rather than track
            # garbage -- a zero pose would command a lunge to a point ~1 m away
            for a in list(want & bs["active"]):
                if fz.get(SRC_OF[a], [None, 0])[1] > 25 or lost.get(SRC_OF[a]):
                    want.discard(a)
                    if lost.get(SRC_OF[a]) and tick % 50 == 0:
                        rp_status.value = f"{a}: tracking lost -- arm held"
            # per-arm rising edge: clutch that arm to its device NOW
            for a in want - bs["active"]:
                dev_p, dev_q = compose_world(st["last_dev"], SRC_OF[a])
                q_now = read_q()
                st["q_cmd"] = q_now.copy() if st["q_cmd"] is None else st["q_cmd"]
                st["q_cmd"][arm_data[a]["joint_indices"]] =                     q_now[arm_data[a]["joint_indices"]]
                robots[a].arm.joint_commands = list(get_joint_positions(robots[a]))
                ee_p, ee_R = ee_pose(q_now, a)
                bs["anchors"][a] = (dev_p.copy(),
                                    R.from_quat(dev_q).as_matrix(), ee_p, ee_R)
            # session end: all buttons released -> save automatically
            if not want and bs["active"]:
                _finish_recording(prefix="teleop")
                rp_status.value = "teleop DISABLED (trajectory saved)"
            bs["active"] = want
            if want and time.time() >= st["next_due"]:
                try:
                    hz = float(rate_dd.value)
                    targets = {}
                    for a in want:
                        dev_p, dev_q = compose_world(st["last_dev"], SRC_OF[a])
                        p0, R0d, ee_p0, ee_R0 = bs["anchors"][a]
                        dR = R.from_quat(dev_q).as_matrix() @ R0d.T
                        pos_t = ee_p0 + scale_sl.value * (bs["S"] @ (dev_p - p0))
                        rot_t = (bs["S"] @ dR @ bs["S"].T) @ ee_R0
                        targets[a] = (pos_t,
                                      np.roll(R.from_matrix(rot_t).as_quat(), 1))
                    q_prev = st["q_cmd"]
                    q_new = np.asarray(coupled_ik.solve(
                        q_prev.astype(np.float32), targets), float)
                    mt = min(max(1.2 / hz, 0.1), 0.5)
                    step_lim = 0.9 * 3.14159 * mt
                    for a in want:
                        idx = arm_data[a]["joint_indices"]
                        ref = np.asarray(robots[a].arm.get_joint_commands(),
                                         float)[: len(idx)]
                        cmd = np.clip(q_new[idx], ref - step_lim, ref + step_lim)
                        robots[a].arm.set_joint_positions(
                            cmd.tolist(), moving_time=mt, accel_time=0.5 * mt,
                            blocking=False)
                        q_prev[idx] = cmd
                    st["q_cmd"] = q_prev
                    st["next_due"] = time.time() + 1.0 / hz
                except Exception as exc:
                    st["mode"] = "idle"
                    btn_cb.value = False
                    rp_status.value = f"button teleop error: {exc}"

        if (robots is not None and st["mode"] == "live"
                and time.time() >= st["next_due"] and st["last_dev"] is not None):
            try:
                hz = float(rate_dd.value)
                S = st["live"]["S"]
                targets = {}
                lost = st.get("lost", {})
                for a in args.arms:
                    if lost.get(SRC_OF[a]):
                        continue  # hold: zero pose = tracking dropout
                    dev_p, dev_q = compose_world(st["last_dev"], SRC_OF[a])
                    p0, R0d, ee_p0, ee_R0 = st["live"]["anchors"][a]
                    dR = R.from_quat(dev_q).as_matrix() @ R0d.T
                    pos_t = ee_p0 + scale_sl.value * (S @ (dev_p - p0))
                    rot_t = (S @ dR @ S.T) @ ee_R0
                    targets[a] = (pos_t, np.roll(R.from_matrix(rot_t).as_quat(), 1))
                q_prev = st["q_cmd"]
                q_new = np.asarray(coupled_ik.solve(
                    q_prev.astype(np.float32), targets), float)
                mt = min(max(1.2 / hz, 0.1), 0.5)
                step_lim = 0.9 * 3.14159 * mt
                for a in targets:
                    idx = arm_data[a]["joint_indices"]
                    ref = np.asarray(robots[a].arm.get_joint_commands(),
                                     float)[: len(idx)]
                    cmd = np.clip(q_new[idx], ref - step_lim, ref + step_lim)
                    robots[a].arm.set_joint_positions(
                        cmd.tolist(), moving_time=mt, accel_time=0.5 * mt,
                        blocking=False)
                    q_prev[idx] = cmd
                st["q_cmd"] = q_prev
                # live compare for the selected arm
                sel = arm_dd.value
                if sel in args.arms:
                    p0, R0d, ee_p0, ee_R0 = st["live"]["anchors"][sel]
                    q_m = read_q()
                    p_m, R_m = ee_pose(q_m, sel)
                    e_p = np.asarray(targets[sel][0]) - ee_p0
                    a_p = p_m - ee_p0
                    e_r = rotvec_deg(
                        R.from_quat(np.roll(targets[sel][1], -1)).as_matrix(), ee_R0)
                    a_r = rotvec_deg(R_m, ee_R0)
                    row_e.value = f"{fmt3(e_p)} | {fmt3(e_r, 1)}"
                    row_a.value = f"{fmt3(a_p)} | {fmt3(a_r, 1)}"
                    row_p.value = pct_row(e_p, a_p, 0.002) + " | " + pct_row(e_r, a_r, 1.0)
                st["next_due"] = time.time() + 1.0 / hz
            except Exception as exc:
                st["mode"] = "idle"
                live_cb.value = False
                rp_status.value = f"live error: {exc}"

        if robots is not None and st["mode"] in ("auto", "primed", "stepping"):
            due = st["mode"] == "auto" and time.time() >= st["next_due"]
            if st["step_req"] or due:
                st["step_req"] = False
                if st["mode"] == "primed":
                    st["mode"] = "stepping"
                if st["k"] < st["wp"]:
                    try:
                        _execute_step()
                        st["next_due"] = time.time() + 1.0 / float(rate_dd.value)
                        if st["mode"] == "auto":
                            rp_status.value = f"auto {st['k']}/{st['wp']}"
                    except Exception as exc:
                        st["mode"] = "idle"
                        rp_status.value = f"error: {exc}"
                else:
                    st["mode"] = "idle"
                    rows = np.asarray(st.pop("log_rows", []))
                    if len(rows):
                        os.makedirs(os.path.join(TRAJ_DIR, "logs"), exist_ok=True)
                        try:
                            from study_ik import describe_weights
                            wdesc = describe_weights()
                        except Exception:
                            wdesc = "unknown"
                        lp = os.path.join(
                            TRAJ_DIR, "logs",
                            f"log_{os.path.splitext(file_dd.value)[0]}_"
                            f"{arm_dd.value}_{rate_dd.value}hz_"
                            f"{time.strftime('%H%M%S')}.npz")
                        np.savez(lp, rows=rows, rate=float(rate_dd.value),
                                 scale=scale_sl.value, arm=arm_dd.value,
                                 traj=file_dd.value, weights=wdesc)
                        rp_status.value = f"complete -- log {os.path.basename(lp)}"
                    else:
                        rp_status.value = "trajectory complete"

        tick += 1
        time.sleep(max(0.0, 0.02 - (time.time() - loop_t0)))


if __name__ == "__main__":
    main()
