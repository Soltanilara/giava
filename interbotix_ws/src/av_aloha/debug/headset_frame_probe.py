"""Step-by-step headset frame probe — measures instead of guessing.

Why this exists: the middle-arm axis mapping and the gripper-arm drift have
been debugged from symptom descriptions, and the latest symptoms (yaw-left
wrong while yaw-right correct) are ASYMMETRIC — something a pure axis swap
cannot produce, but a left-handed quaternion read as right-handed can (it
mirrors rotations instead of rotating them).  This script measures the actual
conventions from the live stream, one guided motion at a time, and prints the
verdict plus the exact correction.

No robots and no ROS needed — just the headset connected:

    python headset_frame_probe.py

Follow the prompts (each step is: return to neutral, press Enter, do ONE slow
motion, hold, press Enter).  ~3 minutes total.  Raw recordings are saved to
probe_data.npz for deeper offline analysis.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from webrtc_headset import WebRTCHeadset  # noqa: E402

AX = ("x", "y", "z")


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #
def sample_window(headset, seconds, label=""):
    """Collect (h_pos, h_quat, l_pos, l_quat, r_pos, r_quat) rows."""
    rows = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        d = headset.receive_data()
        if d is not None:
            rows.append(np.concatenate([
                np.asarray(d.h_pos, float), np.asarray(d.h_quat, float),
                np.asarray(d.l_pos, float), np.asarray(d.l_quat, float),
                np.asarray(d.r_pos, float), np.asarray(d.r_quat, float),
            ]))
        time.sleep(0.01)
    if not rows:
        raise RuntimeError(f"no headset data during '{label}' — is the app running?")
    return np.asarray(rows)


def median_pose(win):
    """Median h_pos/h_quat/l_pos over a window (quat: last sample, normalized)."""
    hp = np.median(win[:, 0:3], axis=0)
    hq = win[-1, 3:7] / np.linalg.norm(win[-1, 3:7])
    lp = np.median(win[:, 7:10], axis=0)
    lq = win[-1, 10:14] / np.linalg.norm(win[-1, 10:14])
    return hp, hq, lp, lq


def guided(headset, prompt, seconds=1.0):
    input(f"\n>> {prompt}\n   ...press Enter when IN POSITION (holding still): ")
    return sample_window(headset, seconds, prompt)


def dominant(v, thresh=0.7):
    """('+y', purity) for a vector; purity < thresh means the motion was mixed."""
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return "0", 0.0
    u = v / n
    i = int(np.argmax(np.abs(u)))
    return ("+" if u[i] > 0 else "-") + AX[i], float(abs(u[i]))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    print(__doc__)
    headset = WebRTCHeadset()
    headset.run_in_thread()  # starts the signaling/asyncio loop -- without
    #                          this the object never connects to anything

    print("waiting for headset data channel", end="", flush=True)
    t0 = time.time()
    while not headset.data_channel_open:
        print(".", end="", flush=True)
        time.sleep(0.5)
        if time.time() - t0 > 60:
            print("\nno connection after 60 s. Is the headset app running and "
                  "on the same network?  (data_collection must NOT be running "
                  "-- it would hold the connection.)")
            sys.exit(1)
    print(" channel open", end="", flush=True)
    while headset.receive_data() is None:
        print(".", end="", flush=True)
        time.sleep(0.2)
    print(" receiving.\n")

    log = {}

    # ---- step 0: noise floor ---------------------------------------------- #
    input(">> STEP 0  Sit/stand as STILL as you can, hands resting still.\n"
          "   Press Enter to record 5 s of stillness: ")
    win = sample_window(headset, 5.0, "stillness")
    log["still"] = win
    h_drift = np.ptp(win[:, 0:3], axis=0)
    l_drift = np.ptp(win[:, 7:10], axis=0)
    r_drift = np.ptp(win[:, 14:17], axis=0)
    print(f"   head pos peak-to-peak  [mm]: {np.round(h_drift * 1e3, 1)}")
    print(f"   left-hand raw p2p      [mm]: {np.round(l_drift * 1e3, 1)}")
    print(f"   right-hand raw p2p     [mm]: {np.round(r_drift * 1e3, 1)}"
          "   << if one hand is much noisier while both rest still,"
          " that controller's tracking is the problem")
    hq0 = win[0, 3:7]; hq1 = win[-1, 3:7]
    ang = 2 * np.degrees(np.arccos(np.clip(abs(np.dot(hq0, hq1)), 0, 1)))
    print(f"   head rotation wander  [deg]: {ang:.2f}")

    # ---- steps 1-3: head translations ------------------------------------- #
    trans_expect = {"lean LEFT ~20 cm": "left", "lean FORWARD ~20 cm": "fwd",
                    "stand TALLER ~15 cm (or stretch up)": "up"}
    base = guided(headset, "STEP 1a  Neutral pose, head centered")
    t_axes = {}
    for i, (motion, name) in enumerate(trans_expect.items()):
        hp0, _, _, _ = median_pose(base)
        win = guided(headset, f"STEP 1{'bcd'[i]}  {motion}, hold there")
        hp1, _, _, _ = median_pose(win)
        d = hp1 - hp0
        t_axes[name] = d / max(np.linalg.norm(d), 1e-9)
        axis, purity = dominant(d)
        print(f"   {name:5s}: raw axis {axis}  ({np.linalg.norm(d)*100:.1f} cm, "
              f"purity {purity:.2f}{'  << mixed, consider redoing' if purity < 0.85 else ''})")
        base = guided(headset, "  back to NEUTRAL")
        log[f"trans_{name}"] = win

    # ---- steps 2: head rotations ------------------------------------------ #
    rot_expect = {"YAW LEFT ~45 deg": "yawL", "YAW RIGHT ~45 deg": "yawR",
                  "PITCH UP (look at ceiling)": "pitchU",
                  "PITCH DOWN (look at floor)": "pitchD",
                  "ROLL LEFT (ear toward left shoulder)": "rollL"}
    r_axes = {}
    for i, (motion, name) in enumerate(rot_expect.items()):
        _, hq0, _, _ = median_pose(base)
        win = guided(headset, f"STEP 2{'abcde'[i]}  {motion}, hold there")
        _, hq1, _, _ = median_pose(win)
        rv = (R.from_quat(hq1) * R.from_quat(hq0).inv()).as_rotvec()
        ang = np.degrees(np.linalg.norm(rv))
        r_axes[name] = rv / max(np.linalg.norm(rv), 1e-9)
        axis, purity = dominant(rv)
        print(f"   {name:6s}: raw rot axis {axis}  ({ang:.0f} deg, purity {purity:.2f}"
              f"{'  << mixed, consider redoing' if purity < 0.85 else ''})")
        base = guided(headset, "  back to NEUTRAL, facing forward")
        log[f"rot_{name}"] = win

    # ---- step 3: composition leak ----------------------------------------- #
    input("\n>> STEP 3  Rest BOTH hands on a table so they CANNOT move.\n"
          "   After pressing Enter, rotate and bob your head freely for 6 s\n"
          "   WITHOUT touching the hands. Press Enter to start: ")
    win = sample_window(headset, 6.0, "leak")
    log["leak"] = win

    # ---- analysis --------------------------------------------------------- #
    print("\n" + "=" * 66)
    print("ANALYSIS")
    print("=" * 66)

    # translation basis (physical fwd/left/up in raw axes)
    T = np.column_stack([t_axes["fwd"], t_axes["left"], t_axes["up"]])
    detT = np.linalg.det(T)
    print(f"\ntranslation basis (columns = fwd,left,up in raw frame):\n{np.round(T, 2)}")
    print(f"det = {detT:+.2f}  ->  {'RIGHT-handed' if detT > 0 else 'LEFT-HANDED (mirror!)'}")

    # rotation-implied basis: yawL about +up, pitchU about -left(right +), rollL about -fwd
    up_r = r_axes["yawL"]
    up_r2 = -r_axes["yawR"]
    yaw_agree = float(np.dot(up_r, up_r2))
    left_r = -r_axes["pitchU"]
    left_r2 = r_axes["pitchD"]
    pitch_agree = float(np.dot(left_r, left_r2))
    fwd_r = -r_axes["rollL"]
    print(f"\nrotation-implied axes:  up={np.round(up_r,2)}  left={np.round(left_r,2)}  fwd={np.round(fwd_r,2)}")
    print(f"yaw L/R consistency  : {yaw_agree:+.2f}  (should be ~ +1; strongly "
          f"negative or ~0 means mirrored/mixed quaternions)")
    print(f"pitch U/D consistency: {pitch_agree:+.2f}")

    Rb = np.column_stack([fwd_r, left_r, up_r])
    detR = np.linalg.det(Rb)
    print(f"rotation basis det = {detR:+.2f}")

    # handedness verdict
    agree = float(np.trace(T.T @ Rb)) / 3.0
    print(f"\ntranslation-vs-rotation basis agreement: {agree:+.2f}")
    if detT > 0 and detR < 0:
        print("VERDICT: rotations are MIRRORED relative to translations —")
        print("         left-handed quaternion read as right-handed.")
        print("FIX    : negate the quaternion axis whose column disagrees below;")
        diff = np.round(T - Rb, 2)
        print(f"         T - R_basis =\n{diff}")
    elif abs(yaw_agree) < 0.7 or abs(pitch_agree) < 0.7:
        print("VERDICT: yaw/pitch probes are inconsistent with any fixed frame —")
        print("         redo steps 2a-2d slowly, one axis at a time.")
    else:
        print("VERDICT: rotations and translations share one frame.")
        print("         The correct HEAD_BASIS_FIX (raw -> fwd/left/up) is T^T:")
        print(np.round(T.T, 2))

    # leak quantification with current pipeline convention
    C = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # deployed HEAD_BASIS_FIX
    world_l, world_r = [], []
    for row in log["leak"]:
        hp, hq = row[0:3], row[3:7]
        Rh = C @ R.from_quat(hq).as_matrix() @ C.T
        world_l.append(C @ hp + Rh @ row[7:10])
        world_r.append(C @ hp + Rh @ row[14:17])
    leak_l = np.ptp(np.asarray(world_l), axis=0)
    leak_r = np.ptp(np.asarray(world_r), axis=0)
    print(f"\ncomposition leak (hands held still, head moving), CURRENT pipeline:")
    print(f"  composed LEFT  wander [mm]: {np.round(leak_l * 1e3, 1)}")
    print(f"  composed RIGHT wander [mm]: {np.round(leak_r * 1e3, 1)}")
    print("  (this directly becomes arm drift x position_scale; a much larger")
    print("   RIGHT value with similar raw noise = lever-arm effect of the")
    print("   right hand being farther from the head)")

    np.savez("probe_data.npz", **log)
    print("\nraw recordings saved to probe_data.npz — send the printed output back.")


if __name__ == "__main__":
    main()
