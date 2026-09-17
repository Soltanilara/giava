"""Offline analysis of probe_data.npz — answers three questions with numbers.

1. Which hand composition is right?  Replays the step-3 leak recording through
   three candidate pipelines and reports the wander of each (the correct one
   should be millimetres):
     A. current: head converted by HEAD_BASIS_FIX, then compose
     B. raw compose: T_world_hand = T_head_raw @ T_rel   (no basis conversion)
     C. no composition: hands used directly as world (pre-fix behavior)

2. What is the headset's LOCAL forward axis?  Needed to implement session-yaw
   calibration (the probe showed the app's world frame is z-up but yawed
   arbitrarily per session -- the old align_rotation_to_z_axis() insight).

3. What was this session's yaw offset?

    python analyze_probe.py [probe_data.npz]
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

C_CURRENT = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # deployed HEAD_BASIS_FIX
AX = ("x", "y", "z")


def wander(win, convert):
    """Peak-to-peak composed world-hand positions for one conversion choice."""
    out_l, out_r = [], []
    for row in win:
        hp, hq = row[0:3], row[3:7]
        Rh_raw = R.from_quat(hq).as_matrix()
        if convert == "current":
            Rh = C_CURRENT @ Rh_raw @ C_CURRENT.T
            hp_u = C_CURRENT @ hp
        else:
            Rh, hp_u = Rh_raw, hp
        if convert == "none":
            out_l.append(row[7:10]); out_r.append(row[14:17])
        else:
            out_l.append(hp_u + Rh @ row[7:10])
            out_r.append(hp_u + Rh @ row[14:17])
    return (np.ptp(np.asarray(out_l), axis=0) * 1e3,
            np.ptp(np.asarray(out_r), axis=0) * 1e3)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "probe_data.npz"
    d = np.load(path)
    still = d["still"]
    leak = d["leak"]

    print("=" * 66)
    print("1. HAND COMPOSITION SHOOTOUT (step-3 recording, hands were still)")
    print("=" * 66)
    verdict = []
    for name, key in (("A current (HEAD_BASIS_FIX)", "current"),
                      ("B raw compose (no conversion)", "raw"),
                      ("C no composition (hands as world)", "none")):
        lw, rw = wander(leak, key)
        score = float(max(lw.max(), rw.max()))
        verdict.append((score, name, key))
        print(f"  {name:34s} left p2p {np.round(lw,1)}  right p2p {np.round(rw,1)} mm")
    verdict.sort()
    print(f"\n  WINNER: {verdict[0][1]}  (max wander {verdict[0][0]:.1f} mm)")
    print("  -> data_collection should use this composition for the hands.")

    print()
    print("=" * 66)
    print("2. HEADSET LOCAL FORWARD AXIS (for session-yaw calibration)")
    print("=" * 66)
    # neutral head orientation from the stillness window
    hq0 = still[-1, 3:7] / np.linalg.norm(still[-1, 3:7])
    R0 = R.from_quat(hq0)

    # user-forward in the raw world, two independent estimates:
    hp0 = np.median(still[:, 0:3], axis=0)
    hp_fwd = np.median(d["trans_fwd"][:, 0:3], axis=0)
    f_trans = hp_fwd - hp0
    f_trans[2] = 0.0
    f_trans /= max(np.linalg.norm(f_trans), 1e-9)

    def rotaxis(key_end, key_start="still"):
        q0 = d[key_start][-1, 3:7]; q1 = d[key_end][-1, 3:7]
        rv = (R.from_quat(q1) * R.from_quat(q0).inv()).as_rotvec()
        return rv / max(np.linalg.norm(rv), 1e-9)

    left_w = -rotaxis("rot_pitchU")
    up_w = rotaxis("rot_yawL")
    f_rot = np.cross(left_w, up_w)
    f_rot[2] = 0.0
    f_rot /= max(np.linalg.norm(f_rot), 1e-9)

    print(f"  user-forward (translation probe): {np.round(f_trans, 2)}")
    print(f"  user-forward (rotation probes)  : {np.round(f_rot, 2)}")
    print(f"  agreement: {np.dot(f_trans, f_rot):+.2f}  (>0.9 = trustworthy)")
    f_world = f_trans + f_rot
    f_world[2] = 0.0
    f_world /= np.linalg.norm(f_world)

    # express user-forward in the head's LOCAL frame at neutral
    f_local = R0.inv().apply(f_world)
    i = int(np.argmax(np.abs(f_local)))
    sign = "+" if f_local[i] > 0 else "-"
    print(f"  forward in head-local frame: {np.round(f_local, 2)}")
    print(f"  LOCAL FORWARD AXIS = {sign}{AX[i]}  (purity {abs(f_local[i]):.2f})")
    print("  -> session-yaw calibration: at anchor, forward = horizontal")
    print(f"     projection of R_head @ ({sign}{AX[i]} unit vector).")

    print()
    print("=" * 66)
    print("3. THIS SESSION'S YAW OFFSET")
    print("=" * 66)
    yaw = np.degrees(np.arctan2(f_world[1], f_world[0]))
    print(f"  user-forward sits at {yaw:+.1f} deg from raw +x.")
    print("  A fixed remap assumes a fixed value here; the probe run proves it")
    print("  is session-dependent, so the mapping must be calibrated at every")
    print("  teleop enable rather than hard-coded.")


if __name__ == "__main__":
    main()
