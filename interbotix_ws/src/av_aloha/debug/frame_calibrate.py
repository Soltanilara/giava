"""Guided frame calibration — measure the headset->robot remap instead of guessing it.

WHY
===
`R_arm_remap` / `R_cam_remap` (data_col_config.py) were validated against the
WebRTC app, and FRAMES.md says so explicitly: "validated by the working
gripper-arm mapping ... that feels correct on the real arms".  The gvlink app
is a different program reading poses from a different code path, and until
2026-08-26 its controller poses were identically zero -- so on this transport
those matrices have never been checked against anything.

`R_cam_remap` has a second problem.  Its derivation is documented as assuming
the head pose was converted by HEAD_BASIS_FIX at parse time; that conversion
was deleted from data_collection.py (the y-up hypothesis it rested on was
wrong) and the matrix was not re-derived.  A sign flip on one rotation axis --
pitch up giving pitch down -- is exactly what a stale basis precondition looks
like.

So: measure.  You move, this records, and it solves for the matrix.

WHAT IT ASSUMES
===============
The robot world frame, straight from FRAMES.md, read off giava.urdf:

    +x  operator's LEFT
    +y  operator's BACKWARD (toward you)
    +z  UP

so "reach forward, away from yourself" is robot -y.  Stand or sit in your
normal teleop position and keep that facing for the whole run -- the app's
world frame carries an arbitrary per-session yaw, so a calibration is only
valid for the pose you took it in.

RUN
===
No arms, no ROS.  data_collection.py must NOT be running -- it would hold the
headset ports.

    python frame_calibrate.py                 # translation + rotation
    python frame_calibrate.py --quick         # translation only (~90 s)
    python frame_calibrate.py --out my.npz

Each step is: get into the start position, press Enter, make ONE slow motion of
roughly the named size, hold still, press Enter.  Motions want to be big --
30 cm, not 5 -- so the signal clears the tracking noise measured in step 0.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))

from headset_link import make_headset  # noqa: E402

AX = ("x", "y", "z")

# What each guided motion SHOULD produce in the robot world frame (FRAMES.md).
# Unit vectors, because only direction is being solved for -- the scale factors
# live in cam_position_scale and friends, not in the remap.
TRANSLATION_STEPS = [
    ("left",    np.array([1.0, 0.0, 0.0]), "move ~30 cm to YOUR LEFT, keeping your facing"),
    ("forward", np.array([0.0, -1.0, 0.0]), "reach ~30 cm FORWARD, away from your body"),
    ("up",      np.array([0.0, 0.0, 1.0]), "move ~30 cm UP"),
]

# Rotations, as world-frame axis-angle.
#
# DERIVE these, do not read them off FRAMES.md's "pitch -> pitch about x" --
# that sentence names the axis and omits the SIGN, and guessing + cost a whole
# debugging round here: the measurements came back -x and -y, were reported as
# inversions, and were right all along.
#
# In the robot frame (+x operator LEFT, +y BACKWARD, +z UP), so operator
# forward = -y and operator right = -x.  The axis of a rotation taking unit
# vector a to unit vector b is a x b:
#
#   yaw left    forward -> left :  (-y) x (+x) = +z
#   pitch up    forward -> up   :  (-y) x (+z) = -x
#   roll right  up -> right     :  (+z) x (-x) = -y
ROTATION_STEPS = [
    ("yaw_left",   np.array([0.0, 0.0, 1.0]),  "YAW your head ~45 deg to the LEFT (turn to look left)"),
    ("pitch_up",   np.array([-1.0, 0.0, 0.0]), "PITCH your head ~30 deg UP (look at the ceiling)"),
    ("roll_right", np.array([0.0, -1.0, 0.0]), "ROLL your head ~30 deg RIGHT (right ear toward right shoulder)"),
]


# ------------------------------------------------------------------ capture --
def packet_seq(headset):
    """Sequence number of the newest uplink packet, or None if unavailable."""
    fresh = getattr(headset, "_fresh_input", None)
    if fresh is None:
        return None
    pkt = fresh()
    return None if pkt is None else getattr(pkt, "seq", None)


def read_new(headset, last_seq, timeout=2.0):
    """Block for a genuinely NEW packet.  Returns (data, seq).

    receive_data() hands back the newest packet, not the next one off a queue.
    Poll it faster than the ~90 Hz uplink and it returns the SAME sample over
    and over -- and a tight loop of sixty reads finishes in microseconds, so
    every one of them is identical.  Peak-to-peak then comes out at exactly
    0.0 mm, which is indistinguishable from a device that is not tracking at
    all.  That is precisely how it presented: head, left and right all
    'untracked' while the app was plainly streaming.

    Deduplicating on the sequence number makes the sample rate the uplink's
    rather than the loop's, which is the only rate that carries information.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = headset.receive_data()
        if d is not None:
            seq = packet_seq(headset)
            if seq is None or seq != last_seq:
                return d, seq
        time.sleep(0.002)
    raise RuntimeError("no new headset packets for 2 s -- did the link drop?")


def sample(headset, seconds, label=""):
    rows, t0, last = [], time.time(), None
    while time.time() - t0 < seconds:
        try:
            d, last = read_new(headset, last)
        except RuntimeError:
            break
        if d is not None:
            rows.append(np.concatenate([
                np.asarray(d.h_pos, float), np.asarray(d.h_quat, float),
                np.asarray(d.l_pos, float), np.asarray(d.l_quat, float),
                np.asarray(d.r_pos, float), np.asarray(d.r_quat, float),
            ]))
    if not rows:
        raise RuntimeError(f"no headset data during '{label}'. Is the app connected?")
    return np.asarray(rows)


SLICES = {"head": (0, 3, 3, 7), "left": (7, 10, 10, 14), "right": (14, 17, 17, 21)}


def median_pos(win, body):
    p0, p1, _, _ = SLICES[body]
    return np.median(win[:, p0:p1], axis=0)


def last_quat(win, body):
    _, _, q0, q1 = SLICES[body]
    q = win[-1, q0:q1]
    n = np.linalg.norm(q)
    # A zero-norm quaternion is an untracked device, not a rotation -- the same
    # (0,0,0,0) that used to crash receive_data(). Identity keeps the arithmetic
    # meaningful; the caller sees the flat delta and warns.
    return q / n if n > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0])


def prompt(msg, detail=""):
    input(f"\n>> {msg}\n   {detail}\n   press Enter when READY (holding still): ")


def dominant(v, thresh=0.75):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return "0", 0.0
    u = v / n
    i = int(np.argmax(np.abs(u)))
    return ("+" if u[i] > 0 else "-") + AX[i], float(abs(u[i]))


# ------------------------------------------------------------------- solving --
class DegenerateMotion(RuntimeError):
    """Not enough real motion to solve for a frame."""


def check_solvable(measured, names, body):
    """Refuse to solve rather than feed NaN to the SVD.

    An untracked controller reports a bit-identical pose forever, so every
    delta is exactly zero, `m / norm(m)` is NaN, and numpy's SVD fails with
    "SVD did not converge" -- an error about linear algebra for a problem that
    is entirely about a controller being asleep.  Say the real thing."""
    dead = [n for m, n in zip(measured, names) if np.linalg.norm(m) < 1e-6]
    if dead:
        raise DegenerateMotion(
            f"{body}: no motion at all on {', '.join(dead)} -- this device is "
            f"not being tracked.\n"
            f"     Exactly-zero poses mean the runtime dropped it (controller "
            f"asleep or set down,\n"
            f"     or hand tracking took over and blanked both controllers).\n"
            f"     Wake it, hold it in view, and re-run.")
    tiny = [n for m, n in zip(measured, names) if np.linalg.norm(m) < 0.03]
    if tiny:
        raise DegenerateMotion(
            f"{body}: under 3 cm of motion on {', '.join(tiny)} -- "
            f"indistinguishable from tracking noise.")


def solve_remap(measured, expected):
    """Nearest rotation matrix taking measured headset deltas to robot axes.

    Orthogonal Procrustes (Kabsch).  Solving for a ROTATION rather than reading
    off one dominant axis per motion matters: hand motions are never purely
    along one axis, and three sloppy-but-independent samples constrain a proper
    orthonormal frame far better than three separate argmax votes -- which can
    disagree and produce a matrix that is not a rotation at all.
    """
    A = np.column_stack([m / np.linalg.norm(m) for m in measured])   # headset
    B = np.column_stack([e / np.linalg.norm(e) for e in expected])   # robot
    U, _S, Vt = np.linalg.svd(B @ A.T)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    return U @ D @ Vt


def average_rotation(mats):
    """The rotation closest to several rotations.

    Not an elementwise mean -- that is not a rotation.  The nearest orthonormal
    matrix to the sum is, and for rotations this close together it is the
    obvious average.
    """
    U, _S, Vt = np.linalg.svd(sum(mats))
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    return U @ D @ Vt


def residuals(remap, measured, expected, names):
    out = []
    for m, e, name in zip(measured, expected, names):
        got = remap @ (m / np.linalg.norm(m))
        want = e / np.linalg.norm(e)
        deg = float(np.degrees(np.arccos(np.clip(got @ want, -1, 1))))
        out.append((name, deg, got))
    return out


def fmt_matrix(M, name):
    rows = ",\n             ".join(
        "[" + ", ".join(f"{v: .4f}" for v in row) + "]" for row in M)
    return (f"    {name}: np.ndarray = field(\n"
            f"        default_factory=lambda: np.array(\n"
            f"            [{rows}],\n"
            f"            dtype=float,\n"
            f"        )\n"
            f"    )")


# ---------------------------------------------------------------------- main --
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="translation only")
    ap.add_argument("--out", default="frame_calibration.npz")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds held per pose")
    args = ap.parse_args()

    print(__doc__)
    headset = make_headset()
    headset.run_in_thread()

    print("waiting for the headset", end="", flush=True)
    t0 = time.time()
    while headset.receive_data() is None:
        print(".", end="", flush=True)
        time.sleep(0.3)
        if time.time() - t0 > 90:
            print("\nNo input packets after 90 s.\n"
                  "  - is the app connected to THIS machine?\n"
                  "  - watch the console: an input-protocol mismatch is reported\n"
                  "    now ('REBUILD THE HEADSET APP') rather than dropped.\n")
            sys.exit(1)
    print(" receiving.\n")

    log = {}

    # ---- step 0: noise floor -------------------------------------------- #
    input(">> STEP 0  Hold everything STILL -- head and both controllers.\n"
          "   Press Enter to record 4 s of stillness: ")
    still = sample(headset, 4.0, "stillness")
    log["still"] = still
    noise = {}
    for body in ("head", "left", "right"):
        p0, p1, _, _ = SLICES[body]
        n = np.ptp(still[:, p0:p1], axis=0)
        noise[body] = float(np.linalg.norm(n))
        print(f"   {body:5s} peak-to-peak [mm]: {np.round(n * 1e3, 1)}")
    if max(noise.values()) > 0.05:
        print("\n   WARNING: >50 mm of motion while holding still. Something is\n"
              "   untracked or asleep; the calibration below will inherit it.")

    # ---- translation ------------------------------------------------------ #
    bodies = [("head", "your HEAD (camera arm)"),
              ("right", "your RIGHT controller"),
              ("left", "your LEFT controller")]
    results = {}

    for body, label in bodies:
        print(f"\n{'=' * 68}\nTRANSLATION -- {label}\n{'=' * 68}")
        measured, expected, names = [], [], []
        for key, want, detail in TRANSLATION_STEPS:
            prompt(f"[{body}/{key}] Return to a comfortable NEUTRAL position.",
                   "hold still there")
            a = median_pos(sample(headset, args.hold, key), body)
            prompt(f"[{body}/{key}] Now {detail}.", "hold at the END of the motion")
            b = median_pos(sample(headset, args.hold, key), body)

            d = b - a
            dist = float(np.linalg.norm(d))
            axis, purity = dominant(d)
            print(f"   moved {dist * 100:5.1f} cm, headset-frame delta "
                  f"{np.round(d, 3)}  -> mostly {axis} (purity {purity:.2f})")
            if dist < 0.08:
                print("   WARNING: under 8 cm. Too small to trust; redo this step "
                      "larger if the result looks wrong.")
            if purity < 0.75:
                print("   WARNING: mixed motion. Fine for the solve (it uses all "
                      "three), but check your facing stayed constant.")
            measured.append(d)
            expected.append(want)
            names.append(key)
            log[f"{body}_{key}"] = np.stack([a, b])

        try:
            check_solvable(measured, names, body)
        except DegenerateMotion as exc:
            print(f"\n   *** SKIPPING {body} ***\n     {exc}")
            continue
        M = solve_remap(measured, expected)
        results[body] = M
        print(f"\n   --- solved remap for {body} ---")
        for name, deg, _got in residuals(M, measured, expected, names):
            flag = "  <-- POOR" if deg > 20 else ""
            print(f"   {name:8s} residual {deg:5.1f} deg{flag}")
        print(f"   matrix:\n{np.array2string(M, precision=4, suppress_small=True)}")

    # ---- rotation --------------------------------------------------------- #
    if not args.quick:
        print(f"\n{'=' * 68}\nROTATION -- your HEAD (this is the pitch-inversion check)\n{'=' * 68}")
        M_head = results["head"]
        for key, want, detail in ROTATION_STEPS:
            prompt(f"[head/{key}] Face NEUTRAL, head level.", "hold still")
            qa = last_quat(sample(headset, args.hold, key), "head")
            prompt(f"[head/{key}] Now {detail}.", "hold at the END of the motion")
            qb = last_quat(sample(headset, args.hold, key), "head")

            # World-frame relative rotation, matching how data_collection
            # composes it: dR_world = R_end @ R_start^T.
            Ra, Rb = R.from_quat(qa).as_matrix(), R.from_quat(qb).as_matrix()
            w = R.from_matrix(Rb @ Ra.T).as_rotvec()
            ang = float(np.degrees(np.linalg.norm(w)))
            if ang < 8.0:
                print(f"   only {ang:.1f} deg of rotation -- too small, skipping")
                continue
            # Same conjugation the teleop uses: dR_robot = M @ dR @ M^T, whose
            # axis is M @ w.
            w_robot = M_head @ w
            axis, purity = dominant(w_robot)
            want_axis, _ = dominant(want)
            ok = axis == want_axis
            print(f"   rotated {ang:5.1f} deg -> robot axis {axis} "
                  f"(purity {purity:.2f}); expected {want_axis}   "
                  f"{'OK' if ok else '*** MISMATCH ***'}")
            if not ok and axis.lstrip("+-") == want_axis.lstrip("+-"):
                print("       same axis, OPPOSITE SIGN -- this is an inversion, "
                      "not a swap. Exactly the reported 'pitch up -> camera "
                      "pitches down'.")
            log[f"head_rot_{key}"] = np.stack([qa, qb])

    # ---- output ----------------------------------------------------------- #
    print(f"\n{'=' * 68}\nRESULT\n{'=' * 68}")
    print("\nThe gripper arms share one matrix; the camera arm has its own.\n"
          "Left and right should agree closely -- if they do not, one controller\n"
          "was tracking badly, so redo rather than averaging them.\n")
    ang_lr = float(np.degrees(np.arccos(np.clip(
        (np.trace(results["left"].T @ results["right"]) - 1) / 2, -1, 1))))
    print(f"left vs right disagreement: {ang_lr:.1f} deg")
    if ang_lr > 15:
        print("   Expect some: you cannot move both hands along the same world\n"
              "   direction: reach and shoulder geometry curve each hand its own\n"
              "   way, and 'left' across the body is not 'left' out to the side.\n"
              "   Only worry if a per-step residual is large AND that step's\n"
              "   motion was short -- that is tracking, not anatomy.")

    # ---- what the matrices above are, and are NOT ------------------------ #
    #
    # These solve for the WHOLE app-world -> robot mapping.  That is not what
    # R_arm_remap holds.  At anchor time data_col_config calls
    #
    #     session_yaw_remap(head_pose, base_remap) -> base_remap @ W
    #
    # and the result OVERRIDES the static matrix (compute_gripper_arm_target:
    # "if arm_state.session_remap is not None: remap_matrix = ...").  W already
    # takes app-world into (forward, left, up) from the operator's measured
    # heading, so R_arm_remap only maps (forward, left, up) -> robot -- and in
    # that convention the deployed [[0,1,0],[-1,0,0],[0,0,1]] is exactly right:
    # left->+x, forward->-y, up->+z, precisely FRAMES.md.
    #
    # So pasting the solved matrix as R_arm_remap would apply this session's
    # yaw twice.  It is printed as a diagnostic, never as a patch.
    #
    # What CAN be wrong is W, through HEAD_LOCAL_FWD -- the head-local axis
    # session_yaw_remap treats as gaze.  Get that wrong and every arm's mapping
    # is yawed by the angle between the assumed axis and the real one.
    hands = average_rotation([results[b] for b in ("right", "left") if b in results]) \
        if any(b in results for b in ("right", "left")) else None

    print("\nSolved app-world -> robot mappings (DIAGNOSTIC -- do not paste):")
    if hands is not None:
        print("  hands:\n" + np.array2string(hands, precision=3, suppress_small=True))
    print("  head:\n" + np.array2string(results["head"], precision=3, suppress_small=True))

    print(f"\n{'-' * 68}\nHEAD_LOCAL_FWD -- the value that actually needs checking\n{'-' * 68}")
    fwd_app = np.diff(log["head_forward"], axis=0)[0]
    fwd_app = np.array([fwd_app[0], fwd_app[1], 0.0])
    fwd_app /= np.linalg.norm(fwd_app)
    q = log.get("head_rot_yaw_left")
    if q is None:
        print("  (needs the rotation steps -- re-run without --quick)")
    else:
        Rh = R.from_quat(q[0] / np.linalg.norm(q[0])).as_matrix()
        best, best_ang = None, 1e9
        for nm, v in (("+x", [1, 0, 0]), ("-x", [-1, 0, 0]), ("+y", [0, 1, 0]),
                      ("-y", [0, -1, 0]), ("+z", [0, 0, 1]), ("-z", [0, 0, -1])):
            g = Rh @ np.array(v, float)
            gh = np.array([g[0], g[1], 0.0])
            n = np.linalg.norm(gh)
            if n < 1e-6:
                continue
            ang = float(np.degrees(np.arccos(np.clip((gh / n) @ fwd_app, -1, 1))))
            mark = ""
            if ang < best_ang:
                best, best_ang = nm, ang
            print(f"  head-local {nm}: {ang:6.1f} deg from your forward{mark}")
        print(f"\n  best match: head-local {best} ({best_ang:.1f} deg)")
        from transform_utils import HEAD_LOCAL_FWD as DEPLOYED
        dep = "".join(("+" if v > 0 else "-") + AX[i]
                      for i, v in enumerate(DEPLOYED) if abs(v) > 0.5)
        if dep != best:
            print(f"  DEPLOYED IS {dep}.  Every session_yaw_remap is yawed by the\n"
                  f"  angle between them -- which is the whole arm mapping.\n"
                  f"  Fix in transform_utils.py:  HEAD_LOCAL_FWD = "
                  f"np.array({[float(x) for x in (np.eye(3)[AX.index(best[1])] * (1 if best[0] == '+' else -1))]})")
        else:
            print(f"  matches the deployed HEAD_LOCAL_FWD ({dep}).")

    np.savez(args.out, **log)
    print(f"\nraw recordings -> {args.out}")
    print("\nNOTE: this measured YOUR facing in THIS session. The app's world\n"
          "frame carries an arbitrary per-session yaw, so re-run it if the\n"
          "mapping ever feels rotated after a restart.")
    headset.close()


if __name__ == "__main__":
    main()
