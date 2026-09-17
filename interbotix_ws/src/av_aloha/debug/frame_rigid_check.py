"""Do the controllers and the headset report poses in the SAME world frame?

WHY
===
frame_calibrate measured 26 deg between the frame solved from head motion and
the one solved from hand motion.  That number cannot be trusted, because the
two came from different gestures: told "move 30 cm left", a hand travels an arc
set by shoulder and elbow, and "left" across the body is not the same world
direction as "left" out to the side.  Per-step residuals ran 5-22 deg, so 26
sits inside the method's own noise.

The confound is the operator, so remove the operator: move the head and both
hands as ONE rigid body.  Every device then undergoes an identical world
displacement by construction, and any residual difference between their
reported deltas is a real difference between the two pose paths.

It matters because session_yaw_remap derives the operator's heading from the
HEAD pose and applies it to HAND targets.  A fixed rotation between those paths
means the hands are mapped through the wrong frame however correct
HEAD_LOCAL_FWD is.

HOW TO HOLD IT -- no pressing controllers to your face
======================================================
Hold a controller in each hand at chest height, elbows tucked against your
ribs, and just KEEP YOUR ARMS LOCKED.  Then move your whole body: lean left,
lean forward, stand up out of the chair.  Locked arms make head and hands one
rigid object, which is all this needs -- pressing them against the headset was
never the point, and it puts the controllers somewhere awkward for no gain.

Nothing to press.  The script watches the stream and segments the motion
itself: hold still, move, hold still.  It tells you what it sees as it goes.

(Quest Pro controllers self-track from their own onboard cameras, so being out
of the headset's view is not in itself a problem -- but the stillness readout
below will tell you if a controller is actually dropping out.)

RUN
===
No arms, no ROS.  data_collection.py must NOT be running.

    python frame_rigid_check.py
    python frame_rigid_check.py --move-cm 20     # smaller motions
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frame_calibrate import (DegenerateMotion, check_solvable, read_new,  # noqa: E402
                             solve_remap)
from headset_link import make_headset  # noqa: E402

STEPS = [
    ("left",    np.array([1.0, 0.0, 0.0]),  "LEAN / STEP TO YOUR LEFT"),
    ("forward", np.array([0.0, -1.0, 0.0]), "LEAN FORWARD, away from where you started"),
    ("up",      np.array([0.0, 0.0, 1.0]),  "STAND UP TALLER (or stand up out of the chair)"),
]

BODIES = ("head", "left", "right")


class Stream:
    """Successive DISTINCT uplink packets.

    Not `receive_data()` in a tight loop: that returns the newest packet, so
    polling faster than the ~90 Hz uplink yields the same sample repeatedly and
    every derived statistic collapses.  Sixty back-to-back reads land inside a
    single packet, giving peak-to-peak exactly 0.0 mm -- which this script then
    reported as "not tracking" for all three devices, head included, while the
    app was streaming perfectly well.
    """

    def __init__(self, headset):
        self.headset = headset
        self.seq = None

    def read(self):
        d, self.seq = read_new(self.headset, self.seq)
        return {"head": np.asarray(d.h_pos, float),
                "left": np.asarray(d.l_pos, float),
                "right": np.asarray(d.r_pos, float)}


def wait_still(stream, window=0.7, tol=0.012, timeout=45.0, label=""):
    """Block until the head has held within `tol` for `window`, return medians.

    Segmenting on the stream instead of on a keypress: this test needs both
    hands occupied, so any protocol that requires a free hand to press Enter
    cannot be performed as specified -- which is how the first version of it
    ended up being done one-handed, with the controllers somewhere they were
    never meant to be.
    """
    buf = []
    t0 = time.time()
    while True:
        buf.append(stream.read())
        buf = buf[-int(window * 90):] if len(buf) > 4 else buf
        if len(buf) >= 12:
            h = np.array([b["head"] for b in buf])
            if float(np.max(np.ptp(h, axis=0))) < tol:
                return {b: np.median(np.array([x[b] for x in buf]), axis=0)
                        for b in BODIES}
        if time.time() - t0 > timeout:
            raise RuntimeError(f"never settled during '{label}' -- still moving?")


def wait_move(stream, origin, need_m, timeout=45.0, label=""):
    """Block until the head has travelled `need_m` from `origin`."""
    t0, shown = time.time(), 0.0
    while True:
        cur = stream.read()
        d = float(np.linalg.norm(cur["head"] - origin["head"]))
        if d >= need_m:
            return
        if d - shown > 0.05:
            shown = d
            print(f"\r   ...{d * 100:4.0f} cm", end="", flush=True)
        if time.time() - t0 > timeout:
            raise RuntimeError(f"only {d * 100:.0f} cm during '{label}' -- "
                               f"needed {need_m * 100:.0f}")


def angle_between(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--move-cm", type=float, default=25.0)
    args = ap.parse_args()
    need = args.move_cm / 100.0

    print(__doc__)
    headset = make_headset()
    headset.run_in_thread()
    print("waiting for the headset", end="", flush=True)
    t0 = time.time()
    while headset.receive_data() is None:
        print(".", end="", flush=True)
        time.sleep(0.3)
        if time.time() - t0 > 90:
            print("\nno input packets after 90 s -- check the console for a "
                  "protocol-mismatch warning.")
            sys.exit(1)
    print(" receiving.\n")

    print("Get set: a controller in each hand, elbows locked against your ribs.")
    print("Waiting for you to hold still...")
    stream = Stream(headset)
    base = wait_still(stream, label="setup")
    print("   still. Checking each device is really tracking:")
    probe = [stream.read() for _ in range(90)]   # ~1 s of REAL packets
    dead = []
    for b in BODIES:
        p2p = np.ptp(np.array([x[b] for x in probe]), axis=0) * 1e3
        flat = float(np.max(p2p)) == 0.0
        print(f"   {b:5s} p2p [mm]: {np.round(p2p, 1)}"
              + ("   <-- EXACTLY ZERO: not tracking" if flat else ""))
        if flat:
            dead.append(b)
    if dead:
        print(f"\n{', '.join(dead)} reporting bit-identical poses -- asleep or "
              f"dropped.\nWake them (a shake usually does it) and re-run.")
        headset.close()
        return

    measured = {b: [] for b in BODIES}
    expected, names = [], []

    for key, want, detail in STEPS:
        print(f"\n{'-' * 60}\n[{key}]  Hold still where you are.")
        a = wait_still(stream, label=f"{key} start")
        print(f"   neutral captured.  NOW: {detail}  (~{args.move_cm:.0f} cm), "
              f"then hold still.")
        wait_move(stream, a, need, label=key)
        print("\r   moving... waiting for you to settle.        ", end="", flush=True)
        b = wait_still(stream, label=f"{key} end")
        print("\r   captured.                                   ")
        line = "   "
        for body in BODIES:
            dv = b[body] - a[body]
            measured[body].append(dv)
            line += f"{body}={np.round(dv, 3)} "
        print(line)
        expected.append(want)
        names.append(key)

    # ---- rigid motion: the deltas should be the same VECTOR, not just the
    # same frame. Comparing lengths too catches a scale/units difference that a
    # rotation-only comparison would hide entirely.
    print("\n" + "=" * 68)
    print("Per-step agreement (rigid motion -> should be near-identical):")
    worst = 0.0
    for i, key in enumerate(names):
        h = measured["head"][i]
        for body in ("left", "right"):
            v = measured[body][i]
            ang = float(np.degrees(np.arccos(np.clip(
                (h @ v) / (np.linalg.norm(h) * np.linalg.norm(v)), -1, 1))))
            worst = max(worst, ang)
            print(f"   {key:8s} head vs {body:5s}: {ang:5.1f} deg, "
                  f"length ratio {np.linalg.norm(v) / np.linalg.norm(h):.2f}")

    try:
        for body in BODIES:
            check_solvable(measured[body], names, body)
    except DegenerateMotion as exc:
        print(f"\n*** cannot conclude: {exc}")
        headset.close()
        return

    Mh = solve_remap(measured["head"], expected)
    Mhand = solve_remap(measured["left"] + measured["right"], expected + expected)
    diff = angle_between(Mh, Mhand)
    print(f"\nhead frame vs hand frame: {diff:.1f} deg   "
          f"(worst per-step {worst:.1f} deg)")
    if diff < 8:
        print("  SAME FRAME. Deriving the heading from the head and applying it to\n"
              "  the hands is sound, and frame_calibrate's 26 deg was the operator.")
    else:
        print("  DIFFERENT FRAMES, with the operator removed -- this one is real.\n"
              "  session_yaw_remap builds W from the HEAD and applies it to HAND\n"
              "  targets, so the hands are mapped through a frame rotated by this\n"
              "  much. A bug, not a calibration constant.")
        print("\n  correction (hand_frame = C @ head_frame):")
        print(np.array2string(Mhand @ Mh.T, precision=4, suppress_small=True))
    if worst > 15:
        print("\n  NOTE: a large worst-per-step means your arms did not stay locked.\n"
              "  Re-run before believing the verdict either way.")

    headset.close()


if __name__ == "__main__":
    main()
