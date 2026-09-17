"""Touch calibration: fix the left-vs-right offset by making the arms touch.

THE PROBLEM
===========
When the two grippers are physically at the same point, FK through
giava.urdf draws them ~2 cm apart (left ~2 cm to the right and ~2 cm above
the right, 2026-09-03).  The right base was measured (1040 mm ruler,
2026-08-19); the left base is declared "symmetric" -- an assumption -- and
nothing has ever measured where the fingertip sits in its flange
(tcp_offsets.json is empty).  Any of base x/y/z/yaw, a joint zero, or
compliance can produce 2 cm, and a ruler cannot separate them.

WHAT YOU PHYSICALLY DO
======================
  1. Teleop as usual (data_collection.py or teleop.py running; the filter
     on or off does not matter -- this reads MEASURED joints).
  2. In a second terminal start the recorder:
         python calibration/touch_calibrate.py record
     It watches the grippers.  You never touch the keyboard once it is
     running.
  3. Bring the two gripper TIPS together so they just touch -- like
     touching two pencil tips -- then CLOSE BOTH GRIPPERS and hold still.
     After 5 s of both-closed-and-still it beeps and records.  OPEN a
     gripper to re-arm.  That is one touch.
  4. Move somewhere else -- higher, lower, nearer you, further away,
     closer to the left base, closer to the right base -- and ROTATE THE
     WRISTS differently each time (tips pointing down, sideways, toward
     you).  Touch again, hold through the beep.  Repeat.
  5. Touch LIGHTLY.  ~1 cm of this rig's error is servo droop plus flex
     below the encoders (move_validation --round-trip); pressing adds
     error that no calibration can remove.  Expect a 5-10 mm residual
     after the fit; a 1 mm fit means the tips were not really touching.

Why closed tips: the tip of a CLOSED gripper lies on the gripper's own
centreline, so it is the same point on both arms and the solver only has to
find how far out along the approach axis it sits (one number).  Open
fingers work too (--feature same: the same finger of each gripper;
--feature cross: left finger of one to right finger of the other, mirrored
about the opening axis), but then it is three numbers and easier to get
wrong.

WHAT THE SOLVER DOES
====================
At each touch the two chains must agree on one world point:

    T_corr . T_L,k . p  ==  T_R,k . p          for every touch k

T_L,k / T_R,k are the flange poses FK reports (current URDF), p the tip in
the flange frame, T_corr a rigid correction applied to the LEFT chain in
the world frame (the left base pose becomes T_corr . T_world_leftbase).
Least squares over T_corr (x, y, z, yaw; --full6 adds pitch/roll) and p.
The right arm is the reference: this fixes the arms RELATIVE to each other,
which is what bimanual work needs; it cannot say which arm is off in
absolute world terms (base_validation.py --protocol floor / waist-circle).
Base yaw and base x are only separable through touches spread nearer/
further from you (selftest: 0.3 m of that spread leaves ~4 mm / 0.7 deg
trading between them), and the tip offset is only observable when the two
wrists point in different directions -- hence step 4.

RUN
===
  # EASIEST -- button in the teleop loop (the recorder below cannot see the
  # controller; the headset link is exclusive to data_collection.py):
  GIAVA_TOUCH_RECORD=1 python data_collection.py --mode bimanual
  #   -> press B (GIAVA_TOUCH_BUTTON=b|a|x|y|lstick|rstick) with both grippers
  #      closed and the closed tips touching; each press appends to
  #      calibration/data/robot/touch_<stamp>.json.  Then 'solve' as below.

  python calibration/touch_calibrate.py record                     # or: close both grippers + hold 5 s = one touch
  python calibration/touch_calibrate.py record --hold 3 --n 12     # shorter hold, more touches
  python calibration/touch_calibrate.py record --every 10 --n 10   # timer + beep instead
  python calibration/touch_calibrate.py record --manual            # ENTER per touch, 'q' ends
  python calibration/touch_calibrate.py solve                      # newest run; prints residuals
  python calibration/touch_calibrate.py apply                      # dry run: shows the edits
  python calibration/touch_calibrate.py apply --write              # backs up + writes giava.urdf, tcp_offsets.json
  python calibration/touch_calibrate.py selftest                   # solver check, no hardware

One arm at a time against a FIXED point (a bolt head) also works in manual
record mode: type '3 l' when the left tip is on point 3, later '3 r' for the
right; the solver pairs them.  Timer mode is always tip-to-tip.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    URDF_PATH,
    T_from_xyz_rpy,
    ensure_dirs,
    invert_T,
    latest_matching,
    make_T,
    provenance,
    rpy_to_matrix,
    save_json,
    timestamp,
)

TCP_CONFIG = HERE / "tcp_offsets.json"
TCP_DERIVED = np.array([0.0, 0.00006, 0.0722])   # tcp.py's derived default
LEFT_BASE_JOINT = "base_left_base_link_fixed"
## How the touched feature maps between the two flanges.
##   closed : closed-gripper tip, on the centreline -> p = (0, 0, z), 1 param
##   same   : the same finger of each gripper        -> p, 3 params
##   cross  : left finger of one to right finger of the other: mirrored
##            about the opening axis (flange local x) -> p_R = diag(-1,1,1) p
FEATURES = ("closed", "same", "cross")
_MIRROR_X = np.diag([-1.0, 1.0, 1.0])


# --------------------------------------------------------------------------
# small SE(3) helpers
# --------------------------------------------------------------------------
def _rotvec_to_matrix(r: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    return R.from_rotvec(np.asarray(r, dtype=float)).as_matrix()


def _matrix_to_rpy(Rm: np.ndarray) -> np.ndarray:
    """URDF rpy = fixed-axis (extrinsic) x, y, z."""
    from scipy.spatial.transform import Rotation as R
    return R.from_matrix(Rm).as_euler("xyz")


def _params_to_T(x: np.ndarray, full6: bool) -> np.ndarray:
    """(tx, ty, tz, yaw[, pitch, roll]) -> 4x4 world-frame correction."""
    t = x[:3]
    if full6:
        rot = rpy_to_matrix([x[5], x[4], x[3]])       # roll, pitch, yaw
    else:
        rot = rpy_to_matrix([0.0, 0.0, x[3]])
    return make_T(t, rot)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
def pair_touches(touches: List[Dict[str, Any]]) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """-> [(T_L, T_R, point_id)] : one left and one right flange pose per point."""
    by_point: Dict[str, Dict[str, np.ndarray]] = {}
    for t in touches:
        pid = str(t["point"])
        arms = t["arms_touching"]
        for arm in ("left", "right"):
            if arm in arms or arms == "both":
                by_point.setdefault(pid, {})[arm] = np.asarray(t["T_world_flange"][arm], dtype=float)
    pairs = []
    for pid, d in by_point.items():
        if "left" in d and "right" in d:
            pairs.append((d["left"], d["right"], pid))
    return pairs


def _tip_vectors(x: np.ndarray, fit_tcp: bool, feature: str, p_fixed: np.ndarray):
    """(p_left, p_right) in their own flange frames, from the parameter vector."""
    if not fit_tcp:
        p = p_fixed
    elif feature == "closed":
        p = np.array([0.0, 0.0, x[-1]])
    else:
        p = x[-3:]
    p_R = _MIRROR_X @ p if feature == "cross" else p
    return p, p_R


def residuals(x: np.ndarray, pairs, full6: bool, fit_tcp: bool, p_fixed: np.ndarray,
              feature: str = "closed") -> np.ndarray:
    T_corr = _params_to_T(x, full6)
    p_L, p_R = _tip_vectors(x, fit_tcp, feature, p_fixed)
    hL, hR = np.append(p_L, 1.0), np.append(p_R, 1.0)
    out = []
    for T_L, T_R, _ in pairs:
        tip_L = (T_corr @ T_L @ hL)[:3]
        tip_R = (T_R @ hR)[:3]
        out.append(tip_L - tip_R)
    return np.concatenate(out)


def solve_touches(pairs, full6: bool = False, fit_tcp: Optional[bool] = None,
                  min_tcp_touches: int = 6, p_init: np.ndarray = TCP_DERIVED,
                  feature: str = "closed") -> Dict[str, Any]:
    from scipy.optimize import least_squares

    if feature not in FEATURES:
        raise ValueError(f"feature must be one of {FEATURES}")
    K = len(pairs)
    if K == 0:
        raise ValueError("no touch has BOTH a left and a right pose -- nothing to fit")
    n_tcp = 1 if feature == "closed" else 3
    if fit_tcp is None:
        # one extra unknown is cheap; three want more touches
        fit_tcp = K >= (3 if n_tcp == 1 else min_tcp_touches)
    n_base = 6 if full6 else 4
    n_par = n_base + (n_tcp if fit_tcp else 0)
    if 3 * K < n_par:
        raise ValueError(f"{K} touches give {3 * K} equations for {n_par} unknowns; "
                         f"need at least {int(np.ceil(n_par / 3))}")
    x0 = np.zeros(n_par)
    if fit_tcp:
        x0[-n_tcp:] = p_init[2] if n_tcp == 1 else p_init

    args_ = (pairs, full6, fit_tcp, p_init, feature)
    before = residuals(x0, *args_).reshape(K, 3)
    res = least_squares(residuals, x0, args=args_, method="lm")
    after = res.fun.reshape(K, 3)

    # observability: singular values of the Jacobian per parameter block
    J = res.jac
    sv = np.linalg.svd(J, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    # 1-sigma parameter uncertainty from the residual scatter
    dof = max(3 * K - n_par, 1)
    s2 = float((after ** 2).sum()) / dof
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
        sigma = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        sigma = np.full(n_par, np.nan)

    T_corr = _params_to_T(res.x, full6)
    tcp_names = [] if not fit_tcp else (["tcp_z"] if n_tcp == 1 else ["tcp_x", "tcp_y", "tcp_z"])
    names = ["tx", "ty", "tz", "yaw"] + (["pitch", "roll"] if full6 else []) + tcp_names
    p_L, _ = _tip_vectors(res.x, fit_tcp, feature, p_init)
    return {
        "n_touches": K,
        "point_ids": [pid for _, _, pid in pairs],
        "full6": full6,
        "fit_tcp": fit_tcp,
        "feature": feature,
        "params": dict(zip(names, res.x.tolist())),
        "sigma": dict(zip(names, sigma.tolist())),
        "T_corr_world": T_corr.tolist(),
        "tcp_xyz_m": p_L.tolist(),
        "residual_before_mm": {
            "per_touch": (np.linalg.norm(before, axis=1) * 1e3).tolist(),
            "rms": float(np.sqrt((before ** 2).sum(axis=1).mean()) * 1e3),
            "mean_vector": (before.mean(axis=0) * 1e3).tolist(),
        },
        "residual_after_mm": {
            "per_touch": (np.linalg.norm(after, axis=1) * 1e3).tolist(),
            "rms": float(np.sqrt((after ** 2).sum(axis=1).mean()) * 1e3),
            "max": float(np.linalg.norm(after, axis=1).max() * 1e3),
        },
        "jacobian_condition": cond,
        "success": bool(res.success),
    }


# --------------------------------------------------------------------------
# record (ROS: listen only)
# --------------------------------------------------------------------------
def cmd_record(args) -> Path:
    from kinematics import (GRIPPER_CLOSED_RAD, GRIPPER_DRIVER_JOINT, GRIPPER_OPEN_RAD,
                            JointFrameBridge, JointStateListener, RobotFrames)

    print(__doc__.split("RUN\n===")[0])
    frames = RobotFrames()
    # left/right are identity through the bridge; the middle waist shift only
    # matters for the middle arm, which this calibration does not use.
    bridge = JointFrameBridge(frames.robot, waist_driver_shift=0.0)

    import rospy
    if not rospy.core.is_initialized():
        rospy.init_node("giava_touch_calibrate", anonymous=True, disable_signals=True)
    listener = JointStateListener(arms=("left", "right"))
    t0 = time.time()
    print("  waiting for joint_states from left and right...")
    while not listener.ready() and time.time() - t0 < 15.0:
        time.sleep(0.1)
    if listener.missing():
        sys.exit(f"  no joint_states from {listener.missing()} -- is the driver up?")
    print("  both arms reporting.\n")
    mode = "manual" if args.manual else ("timer" if args.every is not None else "hold")
    idx_arms = frames.joint_indices("left") + frames.joint_indices("right")
    if mode == "hold":
        print(f"  HOLD MODE: a touch is recorded when BOTH grippers are closed and both arms")
        print(f"  have been still for {args.hold:.0f} s (< {args.still_rad * 1e3:.0f} mrad of joint motion).")
        print("  Bring the closed tips together, hold; after the beep OPEN a gripper to")
        print(f"  re-arm, then move to the next spot.  {args.n} touches, or Ctrl-C to stop early.\n")
    elif mode == "timer":
        print(f"  TIMER MODE: a touch is recorded every {args.every:.0f} s, {args.n} times.")
        print("  Close both grippers, bring the closed tips together, hold still through")
        print("  the beep; then move somewhere else with the wrists turned differently.")
        print("  Ctrl-C ends early and keeps what was recorded.\n")
    else:
        print("  Close both grippers, bring the closed tips together LIGHTLY, press ENTER.")
        print("  Vary the place and the wrist angles.  Type a point id and 'l'/'r' to")
        print("  record a one-arm touch of a fixed point (e.g. '3 l'); plain ENTER =")
        print("  tip-to-tip, both arms.  'q' finishes.\n")

    def _grippers_closed() -> bool:
        for a in ("left", "right"):
            msg = listener._msgs.get(a)
            if msg is None:
                return False
            try:
                k = list(msg.name).index(GRIPPER_DRIVER_JOINT)
                frac = (float(msg.position[k]) - GRIPPER_CLOSED_RAD) / (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)
            except (ValueError, IndexError, ZeroDivisionError):
                return False
            if frac > args.closed_frac:
                return False
        return True

    touches: List[Dict[str, Any]] = []
    auto_id = 0
    hist: List[Tuple[float, np.ndarray]] = []
    armed = True
    q_override: Optional[np.ndarray] = None
    while True:
        q_override = None
        if mode == "hold":
            if len(touches) >= args.n:
                break
            try:
                time.sleep(0.05)
            except KeyboardInterrupt:
                break
            now = time.time()
            if not _grippers_closed():
                if not armed:
                    print("\n  re-armed (a gripper opened) -- move to the next spot", flush=True)
                armed = True
                hist.clear()
                print(f"\r  [{len(touches)}/{args.n}] waiting: close BOTH grippers on the touch      ",
                      end="", flush=True)
                continue
            if not armed:
                print(f"\r  [{len(touches)}/{args.n}] recorded -- open a gripper to re-arm          ",
                      end="", flush=True)
                continue
            hist.append((now, listener.q_driver(frames)))
            hist = [h for h in hist if h[0] >= now - args.hold]
            held = now - hist[0][0]
            qs = np.array([h[1] for h in hist])
            motion = float(np.ptp(qs[:, idx_arms], axis=0).max()) if len(qs) > 1 else 0.0
            print(f"\r  [{len(touches)}/{args.n}] both closed, still for {held:4.1f}/{args.hold:.0f} s, "
                  f"motion {motion * 1e3:5.1f} mrad   ", end="", flush=True)
            if held < args.hold - 0.1:
                continue
            if motion > args.still_rad:
                # moving: the window slides; it will pass once the arms settle
                continue
            # settled: record the average of the last second (encoders are 1.5 mrad steps)
            tail = [h[1] for h in hist if h[0] >= now - 1.0] or [hist[-1][1]]
            q_override = np.mean(tail, axis=0)
            pid, arms = f"t{auto_id}", "both"
            auto_id += 1
            armed = False
            hist.clear()
            print("\a\n  RECORDING", flush=True)
        elif mode == "timer":
            if len(touches) >= args.n:
                break
            try:
                for remaining in range(int(args.every), 0, -1):
                    if remaining <= 3:
                        print(f"  {remaining}...", flush=True)
                    time.sleep(1.0)
                print("\a  RECORDING -- hold still", flush=True)
            except KeyboardInterrupt:
                break
            pid, arms = f"t{auto_id}", "both"
            auto_id += 1
        else:
            try:
                line = input(f"  [{len(touches)} recorded] touch> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if line == "q":
                break
            if line == "":
                pid, arms = f"t{auto_id}", "both"
                auto_id += 1
            else:
                parts = line.split()
                pid = parts[0]
                arms = {"l": "left", "r": "right", "b": "both"}.get(parts[1][0] if len(parts) > 1 else "b", "both")
        if q_override is not None:
            q_driver = q_override
        else:
            # a short average beats one sample: the encoders are 1.5 mrad steps
            qs = []
            for _ in range(10):
                qs.append(listener.q_driver(frames))
                time.sleep(0.02)
            q_driver = np.mean(qs, axis=0)
        q_urdf = bridge.to_urdf(q_driver)
        fk = frames.fk(q_urdf)
        entry = {
            "point": pid, "arms_touching": arms, "time": time.time(),
            "q_driver": {a: q_driver[frames.joint_indices(a)].tolist() for a in ("left", "right")},
            "T_world_flange": {a: fk[frames.ee_link(a)].tolist() for a in ("left", "right")},
        }
        touches.append(entry)
        gap = (np.asarray(entry["T_world_flange"]["left"]) @ np.append(TCP_DERIVED, 1) -
               np.asarray(entry["T_world_flange"]["right"]) @ np.append(TCP_DERIVED, 1))[:3]
        print(f"      recorded {pid} ({arms}); FK says the tips are "
              f"{np.linalg.norm(gap) * 1e3:.1f} mm apart "
              f"(dx {gap[0] * 1e3:+.1f}, dy {gap[1] * 1e3:+.1f}, dz {gap[2] * 1e3:+.1f})"
              + ("  -- open a gripper, move to a new spot" if mode == "hold" else
                 "  -- next in a moment, move to a new spot" if mode == "timer" else ""))

    if not touches:
        sys.exit("  nothing recorded.")
    ensure_dirs()
    out = args.out or (DIR_ROBOT / f"touch_{timestamp()}.json")
    save_json({"metadata": provenance("touch_calibrate_record", n_touches=len(touches)),
               "convention": "World frame is giava.urdf's root link 'base'. T_world_flange are "
                             "4x4 poses of *_gripper_base from FK through the URDF as it was "
                             "at record time (see metadata.urdf / git).",
               "touches": touches}, out, overwrite=bool(args.overwrite))
    print(f"\n  wrote {out}\n  next: python calibration/touch_calibrate.py solve {out}")
    return out


# --------------------------------------------------------------------------
# solve
# --------------------------------------------------------------------------
def _load_run(path_arg: Optional[str]) -> Tuple[Path, Dict[str, Any]]:
    if path_arg:
        p = Path(path_arg)
    else:
        p = latest_matching(DIR_ROBOT, "touch_*.json")
        if p is None:
            sys.exit(f"no touch_*.json under {DIR_ROBOT}; run 'record' first")
    return p, json.loads(Path(p).read_text())


def refk_touches(run: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute every T_world_flange from the stored q_driver through
    calibration/kinematics (pyroki FK + the driver->URDF bridge).  Use when
    the poses in a file are suspect -- the 2026-09-03 button-hook file had
    correct origins but wxyz/xyzw-swapped orientations."""
    from kinematics import JointFrameBridge, RobotFrames

    fr = RobotFrames()
    br = JointFrameBridge(fr.robot, waist_driver_shift=0.0)
    q = fr.home_q().copy()
    out = json.loads(json.dumps(run))
    for t in out["touches"]:
        for a, qa in t["q_driver"].items():
            q[fr.joint_indices(a)] = qa
        fk = fr.fk(br.to_urdf(q))
        for a in list(t["T_world_flange"]):
            t["T_world_flange"][a] = fk[fr.ee_link(a)].tolist()
    out.setdefault("metadata", {})["refk"] = "T_world_flange recomputed from q_driver by touch_calibrate.refk_touches"
    return out


def cmd_solve(args) -> Dict[str, Any]:
    path, run = _load_run(args.run)
    if args.refk:
        run = refk_touches(run)
        rp = path.with_name(path.stem + "_refk.json")
        save_json(run, rp, overwrite=True)
        print(f"re-FK'd poses from joints -> {rp}")
        path = rp
    pairs = pair_touches(run["touches"])
    fit_tcp = None if args.fit_tcp == "auto" else (args.fit_tcp == "yes")
    fit = solve_touches(pairs, full6=args.full6, fit_tcp=fit_tcp,
                        min_tcp_touches=args.min_tcp_touches, feature=args.feature)
    print(f"\n=== {path.name}: {fit['n_touches']} paired touches, feature '{fit['feature']}' "
          f"({'x y z yaw pitch roll' if fit['full6'] else 'x y z yaw'}"
          f"{' + tip' if fit['fit_tcp'] else ', tip held at derived 72.2 mm'}) ===")
    b, a = fit["residual_before_mm"], fit["residual_after_mm"]
    mv = b["mean_vector"]
    print(f"tip gap BEFORE: rms {b['rms']:.1f} mm, mean vector "
          f"(dx {mv[0]:+.1f}, dy {mv[1]:+.1f}, dz {mv[2]:+.1f}) mm  <- your '2 cm x 2 cm'")
    print(f"tip gap AFTER : rms {a['rms']:.1f} mm, max {a['max']:.1f} mm")
    print("per-touch after (mm):", " ".join(f"{v:.1f}" for v in a["per_touch"]))
    print("\nleft-chain correction (world frame), 1-sigma:")
    for k, v in fit["params"].items():
        s = fit["sigma"][k]
        unit = "mm" if k.startswith(("t", "tcp")) else "deg"
        scale = 1e3 if unit == "mm" else 180 / np.pi
        print(f"   {k:<6s} {v * scale:+8.2f} {unit}   +/- {s * scale:.2f}")
    print(f"jacobian condition {fit['jacobian_condition']:.1f}"
          + ("   <-- poorly conditioned: add touches with different wrist angles / places"
             if fit["jacobian_condition"] > 200 else ""))
    if a["rms"] < 1.0:
        print("NOTE: sub-millimetre residual is below the rig's compliance floor (~5-10 mm); "
              "check the touches were real contacts.")
    if a["rms"] > 15.0:
        print("NOTE: residual > 15 mm after the fit -- the model (rigid base offset) does not "
              "explain the data. Suspect a joint zero on one arm, or pressing during touches.")
    fit["run"] = str(path)
    if not args.no_save:
        out = path.with_name(path.stem + "_fit.json")
        save_json({"metadata": provenance("touch_calibrate_solve", run=str(path)), "fit": fit},
                  out, overwrite=True)
        print(f"\nwrote {out}\nnext: python calibration/touch_calibrate.py apply {out} [--write]")
    return fit


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------
_ORIGIN_RE = re.compile(r'<origin\s+xyz="([^"]+)"\s+rpy="([^"]+)"\s*/>')


def _left_base_block(urdf_text: str) -> Tuple[int, int]:
    i = urdf_text.index(f'<joint name="{LEFT_BASE_JOINT}"')
    j = urdf_text.index("</joint>", i)
    return i, j


def cmd_apply(args) -> None:
    if args.fit:
        fit_path = Path(args.fit)
    else:
        fit_path = latest_matching(DIR_ROBOT, "touch_*_fit.json")
        if fit_path is None:
            sys.exit("no touch_*_fit.json; run 'solve' first")
    fit = json.loads(fit_path.read_text())["fit"]
    T_corr = np.asarray(fit["T_corr_world"], dtype=float)
    p_tcp = np.asarray(fit["tcp_xyz_m"], dtype=float)

    urdf_path = Path(URDF_PATH)
    text = urdf_path.read_text()
    i, j = _left_base_block(text)
    block = text[i:j]
    m = _ORIGIN_RE.search(block)
    if m is None:
        sys.exit(f"no <origin xyz rpy/> inside {LEFT_BASE_JOINT}")
    xyz_old = np.array([float(v) for v in m.group(1).split()])
    rpy_old = np.array([float(v) for v in m.group(2).split()])
    T_old = T_from_xyz_rpy(xyz_old, rpy_old)
    T_new = T_corr @ T_old
    xyz_new = T_new[:3, 3]
    rpy_new = _matrix_to_rpy(T_new[:3, :3])
    # round-trip guard: the rpy we write must rebuild the matrix we solved
    if not np.allclose(T_from_xyz_rpy(xyz_new, rpy_new), T_new, atol=1e-9):
        sys.exit("rpy round-trip failed; refusing to write")

    stamp = time.strftime("%Y-%m-%d")
    new_origin = (f'<origin xyz="{xyz_new[0]:.6f} {xyz_new[1]:.6f} {xyz_new[2]:.6f}" '
                  f'rpy="{rpy_new[0]:.6f} {rpy_new[1]:.6f} {rpy_new[2]:.6f}" />')
    note = (f"\n    <!-- Touch-calibrated {stamp} against the right arm "
            f"(calibration/touch_calibrate.py, {fit_path.name}): was xyz=({xyz_old[0]:.4f} "
            f"{xyz_old[1]:.4f} {xyz_old[2]:.4f}) rpy=({rpy_old[0]:.4f} {rpy_old[1]:.4f} "
            f"{rpy_old[2]:.4f}); {fit['n_touches']} touches, residual rms "
            f"{fit['residual_after_mm']['rms']:.1f} mm. Datasets recorded before this used the old values. -->\n    ")
    new_block = block[:m.start()] + note.lstrip("\n").rstrip() + "\n    " + new_origin + block[m.end():]

    print(f"giava.urdf  {LEFT_BASE_JOINT}")
    print(f"   old  xyz=({xyz_old[0]:+.4f} {xyz_old[1]:+.4f} {xyz_old[2]:+.4f})  "
          f"rpy=({rpy_old[0]:+.4f} {rpy_old[1]:+.4f} {rpy_old[2]:+.4f})")
    print(f"   new  xyz=({xyz_new[0]:+.4f} {xyz_new[1]:+.4f} {xyz_new[2]:+.4f})  "
          f"rpy=({rpy_new[0]:+.4f} {rpy_new[1]:+.4f} {rpy_new[2]:+.4f})")
    d = (xyz_new - xyz_old) * 1e3
    print(f"   base moves by ({d[0]:+.1f}, {d[1]:+.1f}, {d[2]:+.1f}) mm, "
          f"yaw by {np.degrees(rpy_new[2] - rpy_old[2]):+.2f} deg")
    print(f"tcp_offsets.json  arms.left / arms.right  translation_xyz_m = "
          f"({p_tcp[0] * 1e3:.1f}, {p_tcp[1] * 1e3:.1f}, {p_tcp[2] * 1e3:.1f}) mm"
          + ("  (fitted)" if fit["fit_tcp"] else "  (derived default, not fitted)"))

    if not args.write:
        print("\ndry run -- nothing written. Re-run with --write.")
        return

    bak = urdf_path.with_name(urdf_path.name + f".bak-{timestamp()}")
    shutil.copy2(urdf_path, bak)
    urdf_path.write_text(text[:i] + new_block + text[j:])
    print(f"\nwrote {urdf_path}  (backup {bak.name})")

    if fit["fit_tcp"]:
        cfg = json.loads(TCP_CONFIG.read_text())
        tbak = TCP_CONFIG.with_name(TCP_CONFIG.name + f".bak-{timestamp()}")
        shutil.copy2(TCP_CONFIG, tbak)
        for arm in ("left", "right"):
            cfg.setdefault("arms", {})[arm] = {
                "translation_xyz_m": [round(float(v), 5) for v in p_tcp],
                "rpy_rad": [0.0, 0.0, 0.0],
                "provenance": f"measured:touch_calibrate_{stamp}",
                "measured": True,
                "note": f"{fit.get('feature', 'closed')}-gripper tip, fitted from {fit['n_touches']} "
                        f"tip-to-tip touches ({fit_path.name}); same part on both arms so one vector.",
            }
        TCP_CONFIG.write_text(json.dumps(cfg, indent=2))
        print(f"wrote {TCP_CONFIG}  (backup {tbak.name})")
    else:
        print("tcp_offsets.json left untouched (fingertip was not fitted).")
    print("\nRe-check: restart the viewer / teleop (they load giava.urdf at startup), bring the "
          "tips together, and the FK gap should now be within the fit residual.")


# --------------------------------------------------------------------------
# selftest (synthetic)
# --------------------------------------------------------------------------
def cmd_selftest(args) -> None:
    rng = np.random.default_rng(int(args.seed))
    from scipy.spatial.transform import Rotation as R

    # truth: the URDF's left base is off by this, and the fingertip is this
    T_true = make_T([0.021, -0.006, 0.019], rpy_to_matrix([0.0, 0.0, np.radians(1.2)]))
    p_true = np.array([0.0, 0.0, 0.075])       # closed tip: on the centreline
    noise = float(args.noise_mm) * 1e-3

    def random_flange(center):
        rot = R.random(random_state=rng).as_matrix()
        return make_T(center + rng.normal(0, 0.005, 3), rot)

    touches = []
    for k in range(int(args.n)):
        P = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.45, -0.15), rng.uniform(0.05, 0.45)])
        # right arm: flange placed so that T_R p_true == P (plus contact noise)
        T_R = random_flange(P)
        T_R[:3, 3] = P - T_R[:3, :3] @ p_true + rng.normal(0, noise, 3)
        # left arm TRUE pose puts its tip at P; what FK REPORTS is inv(T_true) applied
        T_L_true = random_flange(P)
        T_L_true[:3, 3] = P - T_L_true[:3, :3] @ p_true + rng.normal(0, noise, 3)
        T_L_reported = invert_T(T_true) @ T_L_true
        touches.append({"point": f"t{k}", "arms_touching": "both",
                        "T_world_flange": {"left": T_L_reported.tolist(), "right": T_R.tolist()}})
    pairs = pair_touches(touches)
    for full6, feature in ((False, "closed"), (False, "same"), (True, "same")):
        if feature == "closed":
            # regenerate with the tip on the centreline so the 1-param model is exact
            pass
        fit = solve_touches(pairs, full6=full6, fit_tcp=True, feature=feature)
        T_est = np.asarray(fit["T_corr_world"])
        dt = (T_est[:3, 3] - T_true[:3, 3]) * 1e3
        dyaw = np.degrees(_matrix_to_rpy(T_est[:3, :3])[2] - np.radians(1.2))
        dp = (np.asarray(fit["tcp_xyz_m"]) - p_true) * 1e3
        print(f"[{'full6' if full6 else 'xyz+yaw'} / {feature}] n={args.n} noise={args.noise_mm} mm: "
              f"base error ({dt[0]:+.2f}, {dt[1]:+.2f}, {dt[2]:+.2f}) mm, yaw {dyaw:+.3f} deg; "
              f"tcp error ({dp[0]:+.2f}, {dp[1]:+.2f}, {dp[2]:+.2f}) mm; "
              f"residual before {fit['residual_before_mm']['rms']:.1f} -> after "
              f"{fit['residual_after_mm']['rms']:.1f} mm; cond {fit['jacobian_condition']:.0f}")
    print("PASS" if np.all(np.abs(dt) < 3 * max(noise * 1e3, 0.5)) else "CHECK: base error exceeds 3x noise")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="listen to joint_states; ENTER records a touch")
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--hold", type=float, default=5.0,
                   help="hold mode (default): record once both grippers are closed and both "
                        "arms have been still this many seconds")
    r.add_argument("--still-rad", type=float, default=0.02,
                   help="max joint motion over the hold window that still counts as 'still'")
    r.add_argument("--closed-frac", type=float, default=0.3,
                   help="gripper counts as closed below this fraction of open travel")
    r.add_argument("--every", type=float, default=None,
                   help="timer mode instead: record every N seconds with a beep")
    r.add_argument("--manual", action="store_true", help="keyboard mode: ENTER per touch")
    r.add_argument("--n", type=int, default=10, help="touches to record (hold/timer modes)")

    s = sub.add_parser("solve", help="fit the left-base correction (+ fingertip)")
    s.add_argument("run", nargs="?", default=None, help="touch_<stamp>.json (default newest)")
    s.add_argument("--full6", action="store_true", help="also fit base pitch/roll")
    s.add_argument("--fit-tcp", choices=("auto", "yes", "no"), default="auto")
    s.add_argument("--min-tcp-touches", type=int, default=6)
    s.add_argument("--feature", choices=FEATURES, default="closed",
                   help="what touched what: closed tips (default), same finger, or crossed fingers")
    s.add_argument("--no-save", action="store_true")
    s.add_argument("--refk", action="store_true",
                   help="recompute the flange poses from the stored joint angles before fitting")

    a = sub.add_parser("apply", help="write the correction into giava.urdf / tcp_offsets.json")
    a.add_argument("fit", nargs="?", default=None, help="touch_<stamp>_fit.json (default newest)")
    a.add_argument("--write", action="store_true", help="actually write (default: dry run)")

    t = sub.add_parser("selftest", help="recover a known offset from synthetic touches")
    t.add_argument("--n", type=int, default=10)
    t.add_argument("--noise-mm", type=float, default=3.0)
    t.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    {"record": cmd_record, "solve": cmd_solve, "apply": cmd_apply, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
