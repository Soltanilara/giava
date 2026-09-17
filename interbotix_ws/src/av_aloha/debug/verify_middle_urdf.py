"""Check giava.urdf's middle chain against the arm in front of you.

WHY THIS EXISTS
---------------
The 2026-08-21 fold (see middle_joint_offsets.json) moved a set of measured
driver<->URDF offsets out of that json and into giava.urdf's joint origins.
The fold is an exact re-parameterisation -- FK before and after agree to
3.2e-13 -- so it could not have INTRODUCED error.  What it did do is bake four
hand-measured constants into the geometry:

    middle_shoulder      origin pitch  -pi/2  ->  -pi/2 + 0.315   (18.05 deg)
    middle_upper_arm     origin pitch    0    ->    0    + 0.130   ( 7.45 deg)
    middle_upper_forearm origin roll   +pi/2  ->  +pi/2 + 0.070   ( 4.01 deg)
    middle_lower_forearm origin yaw    +pi/2  ->  +pi/2 + 0.090   ( 5.16 deg)

Before the fold every middle origin was an exact multiple of pi/2, and the
shoulder->elbow segment matched the vendor description (wx250s_7dof.urdf.xacro)
to 0.25 mm under a clean -pi/2 rotation about y.  So those four constants are
the only non-CAD numbers in the chain, and the shoulder/elbow pair is exactly
where the rendered figure is reported to be off.

They came from make_middle_offsets.py, which is a SINGLE eyeball match: jog the
sim until it looks like a photo of the real arm, subtract.  The capture that
produced them (poses_custom.json -> middle -> "urdf_forward") no longer exists
in the repo; the surviving "forward" entry reproduces the middle_wrist offset
(1.56) exactly but contradicts the shoulder (-0.065, not -0.315) and the elbow
(+0.04, not -0.13).  That is not enough to convict them, and it is not enough
to trust them either.  Hence: measure.

WHAT TO DO
----------
The waist turns about world z, so it cannot change any link's inclination from
horizontal.  That makes a digital level (a phone works) on two links a clean
two-measurement solve for the two suspect constants.

  1. Park the middle arm somewhere with the upper arm and the forearm both
     clearly off-horizontal -- the canonical 'forward' pose is fine.  Leave it
     torqued ON so it cannot sag while you measure.

  2. python verify_middle_urdf.py
     with the driver up, it reads joint_states and prints what giava.urdf
     PREDICTS for the inclination of each link.  Offline, pass the pose:
     python verify_middle_urdf.py --q 3.09 -1.31 0.88 -0.07 0.53 1.64 2.30

  3. Lay the level along the upper-arm link, then along the forearm link, and
     read the angle from horizontal (sign: nose-up positive).

  4. Feed them back:
     python verify_middle_urdf.py --measured-upper-arm 12.4 --measured-forearm -31.8
     It solves for the shoulder/elbow origin corrections that reconcile the
     URDF with what you measured, and tells you whether the answer is
     "the fold constants are right", "they should be zero", or neither.

A residual near 0.315 / 0.130 means the fold constants are wrong and the clean
vendor frame is right.  A residual near zero means the fold is correct and the
error is somewhere else -- look at the wrist/pan mount clocking next, both of
which middle_joint_offsets.json already flags as unverified.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = _giava_paths.REPO_ROOT / "giava.urdf"

MOVING = ["middle_base", "middle_shoulder", "middle_upper_arm", "middle_upper_forearm",
          "middle_lower_forearm", "middle_wrist", "middle_pan"]
CHAIN = ["base_middle_base_link_fixed"] + MOVING + ["middle_camera_cover_fixed"]

## The two constants under test, and the clean frames they replaced.
SUSPECT = {"middle_shoulder": (1, -np.pi / 2), "middle_upper_arm": (1, 0.0)}

## The two links whose inclination is measurable, and the joint whose origin
## translation defines their long axis.  That axis is the PIVOT-TO-PIVOT line
## -- shoulder-axis centre to elbow-axis centre, and elbow to forearm-roll --
## NOT any machined face on the link.  The wx250s upper arm is bent (0.04975 m
## of lateral offset over 0.25 m of reach), so its flat faces sit ~11.3 deg off
## the line that actually matters.  Measure the pivots, not the casting.
MEASURABLE = [("middle_upper_arm_link", "middle_upper_arm", "upper arm"),
              ("middle_upper_forearm_link", "middle_upper_forearm", "forearm")]


# --------------------------------------------------------------------- maths

def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axang(axis, q):
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def _T(R, p):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p
    return M


def load_joints(path):
    root = ET.parse(path).getroot()
    out = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        a = j.find("axis")
        g = lambda k, d: [float(v) for v in (o.get(k, d).split() if o is not None else d.split())]
        out[j.get("name")] = dict(
            type=j.get("type"), child=j.find("child").get("link"),
            xyz=np.array(g("xyz", "0 0 0")), rpy=np.array(g("rpy", "0 0 0")),
            axis=np.array([float(v) for v in a.get("xyz").split()]) if a is not None else np.zeros(3))
    return out


def fk(joints, q, overrides=None):
    """link name -> 4x4 in the robot base frame. `overrides` patches origin rpy."""
    M = np.eye(4)
    frames = {}
    qi = dict(zip(MOVING, q))
    for name in CHAIN:
        j = joints[name]
        rpy_vec = (overrides or {}).get(name, j["rpy"])
        M = M @ _T(_rpy(*rpy_vec), j["xyz"])
        if j["type"] != "fixed":
            M = M @ _T(_axang(j["axis"], qi[name]), np.zeros(3))
        frames[j["child"]] = M.copy()
    return frames


def inclinations(joints, q, overrides=None):
    """Angle above horizontal, in degrees, of each measurable link's long axis."""
    frames = fk(joints, q, overrides)
    out = {}
    for link, next_joint, label in MEASURABLE:
        d = joints[next_joint]["xyz"]
        d = d / np.linalg.norm(d)
        world = frames[link][:3, :3] @ d
        out[label] = float(np.degrees(np.arcsin(np.clip(world[2], -1.0, 1.0))))
    return out


def _with_deltas(joints, ds, de):
    """Origin rpy overrides for shoulder pitch += ds, elbow pitch += de."""
    ov = {}
    for name, delta in (("middle_shoulder", ds), ("middle_upper_arm", de)):
        r = joints[name]["rpy"].copy()
        r[1] += delta
        ov[name] = r
    return ov


def solve(joints, q, measured):
    """Least-squares (ds, de) added to the shoulder/elbow origin pitch."""
    x = np.zeros(2)
    for _ in range(60):
        def res(v):
            got = inclinations(joints, q, _with_deltas(joints, v[0], v[1]))
            return np.array([got["upper arm"] - measured[0], got["forearm"] - measured[1]])
        r = res(x)
        J = np.zeros((2, 2))
        for k in range(2):
            h = np.zeros(2)
            h[k] = 1e-6
            J[:, k] = (res(x + h) - r) / 1e-6
        try:
            step = np.linalg.solve(J + 1e-12 * np.eye(2), -r)
        except np.linalg.LinAlgError:
            break
        x = x + step
        if np.linalg.norm(step) < 1e-12:
            break
    return x, res(x)


# ----------------------------------------------------------------------- ros

def read_live():
    for p in (Path("/opt/ros/noetic/lib/python3/dist-packages"),
              Path("/home/devi/giava/interbotix_ws/devel/lib/python3/dist-packages")):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.append(str(p))
    import time
    import rospy
    from interbotix_xs_modules.core import InterbotixRobotXSCore
    rospy.init_node("verify_middle_urdf", anonymous=True)
    dxl = InterbotixRobotXSCore(robot_model="wx250s", robot_name="puppet_middle", init_node=False)
    time.sleep(1.0)
    return np.array([float(v) for v in dxl.robot_get_joint_states().position[:7]])


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--q", nargs=7, type=float,
                    help="middle joint values (driver frame); default: read the live arm")
    ap.add_argument("--measured-upper-arm", type=float,
                    help="level reading along the upper arm, deg above horizontal")
    ap.add_argument("--measured-forearm", type=float,
                    help="level reading along the forearm, deg above horizontal")
    ap.add_argument("--dz-upper-arm", type=float,
                    help="mm the ELBOW pivot sits above the SHOULDER pivot "
                         "(negative if below); converted to an angle for you")
    ap.add_argument("--dz-forearm", type=float,
                    help="mm the FOREARM-ROLL pivot sits above the ELBOW pivot")
    ap.add_argument("--write-correction", nargs=2, type=float, metavar=("DS", "DE"),
                    help="add DS/DE radians to the shoulder/elbow origin pitch "
                         "in the urdf and exit -- for iterating against a viewer")
    ap.add_argument("--urdf", default=str(URDF))
    args = ap.parse_args()

    joints = load_joints(args.urdf)

    if args.write_correction:
        ds, de = args.write_correction
        text = open(args.urdf, encoding="utf-8").read()
        for name, delta in (("middle_shoulder", ds), ("middle_upper_arm", de)):
            cur = joints[name]["rpy"]
            new_p = cur[1] + delta
            pat = re.compile(r'(<joint name="%s" type="revolute">\s*<origin[^>]*?rpy=")([^"]*)(")'
                             % name, re.S)
            m = pat.search(text)
            if not m:
                sys.exit(f"could not locate the {name} origin rpy in {args.urdf}")
            text = text[:m.start(2)] + f"{cur[0]} {new_p} {cur[2]}" + text[m.end(2):]
            print(f"{name:22s} pitch {cur[1]:+.6f} -> {new_p:+.6f}  ({delta:+.4f})")
        with open(args.urdf, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"\nwrote {args.urdf}.  Restart the viewer to see it.")
        print("NOTE: <limit> pairs are untouched -- the joint variable still means")
        print("      the driver value, so the limits did not move with the origin.")
        return

    if args.q is not None:
        q = np.array(args.q, float)
        src = "--q"
    else:
        try:
            q = read_live()
            src = "live joint_states"
        except Exception as exc:
            sys.exit(f"could not read the arm ({exc}).\nPass a pose instead: --q q0 .. q6")

    print(f"urdf : {args.urdf}")
    print(f"pose : {src}")
    for n, v in zip(MOVING, q):
        print(f"       {n:22s} {v:+8.4f} rad  ({np.degrees(v):+7.2f} deg)")

    ## The two hypotheses, side by side.  "clean" = the fold constants removed
    ## from the shoulder and elbow origins, everything else left exactly as is.
    clean = {n: np.array([joints[n]["rpy"][0], val, joints[n]["rpy"][2]])
             for n, (_, val) in SUSPECT.items()}
    asis = inclinations(joints, q)
    alt = inclinations(joints, q, clean)

    print("\npredicted inclination from horizontal (nose-up positive):")
    print(f"  {'link':12s} {'giava.urdf as-is':>18s} {'fold constants removed':>24s} {'spread':>9s}")
    for _, _, label in MEASURABLE:
        print(f"  {label:12s} {asis[label]:>15.2f} deg {alt[label]:>21.2f} deg "
              f"{abs(asis[label] - alt[label]):>6.2f} deg")

    ## Height difference between two pivots is far easier to measure than an
    ## angle, and needs no assumption about which face is parallel to what:
    ## the pivot separation is known exactly from the urdf, so
    ##     inclination = asin(dz / L).
    up, fa = args.measured_upper_arm, args.measured_forearm
    for dz, joint, lbl, setter in ((args.dz_upper_arm, "middle_upper_arm", "upper arm", "up"),
                                   (args.dz_forearm, "middle_upper_forearm", "forearm", "fa")):
        if dz is None:
            continue
        L = float(np.linalg.norm(joints[joint]["xyz"])) * 1000.0
        if abs(dz) > L:
            sys.exit(f"{lbl}: |dz| {abs(dz):.1f} mm exceeds the {L:.2f} mm pivot "
                     "separation -- check which two pivots you measured")
        ang = float(np.degrees(np.arcsin(dz / L)))
        print(f"\n{lbl}: dz {dz:+.1f} mm over a {L:.2f} mm pivot separation "
              f"-> {ang:+.2f} deg")
        if setter == "up":
            up = ang
        else:
            fa = ang

    if up is None or fa is None:
        print("\nMeasure, then re-run.  Either form works:")
        print("  --measured-upper-arm DEG --measured-forearm DEG      (level on the link)")
        print("  --dz-upper-arm MM --dz-forearm MM                    (ruler between pivots)")
        print("\nThe angle is between the PIVOT-TO-PIVOT line and horizontal:")
        for _, j, lbl in MEASURABLE:
            L = float(np.linalg.norm(joints[j]["xyz"])) * 1000.0
            print(f"  {lbl:10s} {L:7.2f} mm between pivot centres")
        return

    measured = np.array([up, fa])
    (ds, de), res = solve(joints, q, measured)
    print(f"\nmeasured: upper arm {measured[0]:+.2f} deg, forearm {measured[1]:+.2f} deg")
    print(f"residual after solve: {np.abs(res).max():.4f} deg")
    print("\ncorrection the measurement asks for, on top of the CURRENT urdf origins:")
    print(f"  middle_shoulder  pitch {ds:+.4f} rad ({np.degrees(ds):+6.2f} deg)")
    print(f"  middle_upper_arm pitch {de:+.4f} rad ({np.degrees(de):+6.2f} deg)")

    ## Which hypothesis did the arm just vote for?  "Clean" is an ABSOLUTE
    ## frame (shoulder pitch -pi/2, elbow 0), so the delta that reaches it
    ## depends on what the file currently holds -- compute it, don't assume
    ## the file is the unmodified fold.
    to_clean = np.array([-np.pi / 2 - joints["middle_shoulder"]["rpy"][1],
                         0.0 - joints["middle_upper_arm"]["rpy"][1]])
    if np.abs([ds, de]).max() < np.radians(1.5):
        print("\n=> the URDF as it stands already matches the arm. The fold constants are"
              "\n   right; look elsewhere for the discrepancy (wrist/pan mount clocking,"
              "\n   both flagged unverified in middle_joint_offsets.json).")
    elif np.abs(np.array([ds, de]) - to_clean).max() < np.radians(1.5):
        print("\n=> the arm agrees with the CLEAN vendor frame. The 0.315 / 0.130 fold"
              "\n   constants are wrong. Set middle_shoulder origin rpy pitch back to"
              "\n   -pi/2 and middle_upper_arm to 0, and re-derive both <limit> pairs.")
    else:
        print("\n=> neither hypothesis. The residual is a real, previously unmeasured"
              "\n   offset -- apply it to the origins and re-run to confirm it closes.")


if __name__ == "__main__":
    main()
