"""Physically measure the arm base placement in giava.urdf.

The base poses in the URDF are ASSERTIONS about how the rig is bolted
together.  Nothing in the software checks them, and every world-frame
quantity inherits their error: IK targets, inter-arm clearance, and any
camera extrinsic composed through T_world_ee.

The difficulty is that a base frame origin is a virtual point inside the
base plate and a base yaw is a direction, so neither can be measured with a
ruler directly.  Each protocol below turns one of them into a distance you
CAN measure, by using the robot itself as the instrument.

    waist-circle   base (x, y)      -- no IK, no FK, no camera
    straight       base yaw         -- one arm, one straightedge
    converge       relative pose    -- both arms, and yaw from two points

Run --list for the procedures, then e.g.

    python calibration/base_validation.py --protocol waist-circle --arm right
    python calibration/base_validation.py --protocol converge --dry-run

SAFETY: these move the arms through large sweeps.  Every waypoint is
checked against the deployed sphere self-collision model before anything
moves, and --dry-run does the whole thing with no hardware.
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
    append_json_run,
    make_T,
    confirm,
    wait_for_enter,
    ensure_dirs,
    provenance,
    save_json,
    timestamp,
)
from kinematics import ARM_ORDER, WORLD_FRAME, RobotFrames  # noqa: E402
from move_validation import MotionSession  # noqa: E402
from arm_config import ARM_CONFIG  # noqa: E402
import workspace as WS  # noqa: E402


## ------------------------------------------------------------------ ##
## Procedures (printed, so the lab steps and the code never drift apart)
## ------------------------------------------------------------------ ##

PROCEDURES = {
"waist-circle": """
WAIST-CIRCLE  --  measures a base's (x, y) in the world frame
=============================================================
Why it works
    base_link -> shoulder_link is a pure +z translation, and the waist
    rotates about that same z.  So the waist axis is a VERTICAL LINE
    THROUGH THE BASE ORIGIN's (x, y).  Rotate only the waist and every
    point on the arm sweeps a circle centred on that line.  Fit the circle
    through your marks and its centre IS the base (x, y) -- measured
    directly, with no IK, no FK, and no dependence on any other joint's
    calibration.

SPACE.  READ THIS FIRST.
    The arm sweeps an arc at whatever radius it is currently extended to.
    The sweep is RELATIVE to the arm's current waist angle and defaults to
    a 90 deg total arc -- it does NOT do a full revolution.  Before moving,
    the script prints the swept envelope of EVERY link (not just the
    gripper, which is not the widest part) including the configurations
    passed through in between, and asks you to confirm it fits.

    The IK collision model checks the robot against ITSELF only.  It knows
    nothing about the table, the frame bars or the workstation.  Record the
    free space in workspace_limits.json and the envelope gets checked
    automatically; until then, check it by eye.

    Tuck the arm in first if space is tight -- a smaller radius sweeps a
    smaller area.  Trade-off below.

How wide an arc, and how far extended?
    Two knobs, and they trade against each other.

    ARC.  A short arc pins the circle centre poorly along the radial
    direction.  For a symmetric arc of half-angle a and marking precision
    s, the centre error is roughly  s / (1 - cos a):

        full 360 deg       ->  ~s / sqrt(N)   (best, and radius-independent)
        total arc 180 deg  ->  1.0 x s
        total arc 120 deg  ->  2.0 x s
        total arc  90 deg  ->  3.4 x s        (default, untucked)
        total arc  60 deg  ->  7.5 x s
        total arc  40 deg  ->   16 x s        (too narrow to be useful)

    EXTENSION.  Radius sets how much floor the sweep needs.  It barely
    affects accuracy once you have a FULL circle -- there the centre error
    is about s/sqrt(N) whatever the radius, so long as the marks are far
    enough apart to tell apart.

    THE GOOD COMBINATION, if your rig has the headroom: --tuck --arc 360.
    Standing the arm upright and holding the gripper ~150 mm off its own
    waist axis lets it turn all the way round inside a ~450 mm square,
    while giving the best centre estimate available.  Compare that with an
    untucked 360 deg sweep, which needs the better part of a metre.

    Note the swept envelope is wider than the gripper's own radius -- the
    elbow and forearm swing further out than the fingertips.  The printed
    envelope accounts for every link; do not size your free space off the
    gripper radius alone.

What you do
    1. Tape paper on the table under the gripper.
    2. With --tuck, confirm the move into the upright pose first.
    3. Confirm the swept envelope the script prints.
    4. It rotates ONLY the waist to each angle in turn, pausing at each.
    4. At each pause, drop a plumb line from the SAME physical feature
       every time (a fingertip) and mark the point.
    5. Enter the marks into the saved JSON. The circle through them has its
       centre at the waist axis = the base origin's (x, y).

Reading the result
    Fit a circle to the marks (three non-collinear points determine one;
    more is better). Its centre, in your world reference, is the measured
    base (x, y). A consistent offset from the URDF value is a real base
    placement error. If the marks do not lie on a circle at all, something
    upstream is wrong -- a loose base, or the arm flexing under its own
    weight at that extension.
""",

"straight": """
STRAIGHT  --  measures a base's YAW
===================================
Why it works
    A yaw error rotates everything that arm does about its own base.  Ask
    the end-effector to travel along the world +x axis and it will instead
    travel along a line rotated by the yaw error.  Over a path of length
    L, a yaw error theta shows up as a sideways deviation

        d = L * sin(theta)        i.e.   theta = asin(d / L)

    which is a distance, and distances are measurable.  Long L makes small
    angles visible: over 300 mm, 1 mm of deviation is 0.19 deg.

What you do
    1. Clamp a straightedge on the table along your world +x reference,
       just under the gripper's path.
    2. The script steps the end-effector along world +x in a straight
       line, pausing at each step.
    3. At each pause measure the PERPENDICULAR distance from the
       straightedge to the same physical feature on the gripper.
    4. The deviation should be constant.  If it grows or shrinks
       monotonically along the path, the difference between the first and
       last reading is your yaw error over that length.

Reading the result
    theta = asin((d_end - d_start) / path_length).  Sign: if the far end
    drifts toward +y, the base yaw is rotated positively about +z.
    A CONSTANT non-zero offset is not yaw -- that is just where the
    straightedge sits, and it cancels.
""",

"floor": """
FLOOR  --  checks the vertical chain against the tabletop
=========================================================
Why it works
    Rest part of the gripper on the table.  That contact point is at the
    table's height, exactly, by definition.  Whatever the model says its
    height is, the difference is the error in the vertical chain: base z,
    link lengths, joint calibration, or the feature offset.

    The awkward part is knowing WHICH point is touching.  With any roll on
    the wrist it is not the TCP and not the flange -- it is whichever bit
    of geometry hangs lowest.  So this reads the arm's collision spheres
    (the same 180-sphere model the IK uses) and reports the lowest point of
    the actual geometry, not a frame origin.

What you do
    1. Rest the gripper on the tabletop, any roll, both arms if you like.
    2. Run it. Nothing moves -- it only reads joint states.
    3. Compare the reported lowest-point height against the table height.

Reading the result
    The table is world z = 0 by default (the arm bases sit on 20 mm risers,
    which is the 0.020 in the URDF, so the tabletop is the z = 0 plane).
    Override with --table-z if your rig differs.

    A consistent offset on BOTH arms points at the world z datum or the
    base height.  A difference BETWEEN the arms points at one arm's
    calibration.  Note the sphere model is inscribed and reads a few mm
    optimistic, so expect the lowest sphere to sit slightly ABOVE the true
    surface even when everything is right.
""",

"separation": """
SEPARATION  --  checks the base x-separation against one ruler reading
======================================================================
Why it works
    Each arm's forward kinematics is correct RELATIVE TO ITS OWN BASE.  If
    the two bases are modelled closer together than they really are, every
    world-frame quantity is fine per arm but the two arms appear too close
    to each other -- they can even look interpenetrating in the viewer
    while being nowhere near each other in the room.

    The gap between the two grippers is that error, directly.  Measure it
    once and the base separation follows:

        measured_gap^2 = (dx + delta)^2 + dy^2 + dz^2

    solved for delta, the correction to the modelled x-separation.

What you do
    1. Leave the arms wherever they are -- any pose works, but one with
       the grippers reasonably close and clearly visible is easiest to
       measure.  Nothing moves and nothing is torqued on.
    2. The script prints the gap the MODEL predicts, and what the gap
       would be for a few alternative base separations.
    3. Measure the actual gap between the same two features on the two
       gripper mounting plates.
    4. Pass it back with --measured-gap-mm and it back-solves the implied
       base separation.

Reading the result
    An implied separation close to the URDF's value (1040 mm since the
    2026-08-19 correction; 938 mm before it) means the model is
    right and the apparent closeness is just the render.  A consistently
    larger value across several different poses means the URDF's base
    x-separation is too small by that amount -- and every world-frame
    number involving both arms inherits it.

    Repeat at two or three quite different poses.  A real base error gives
    the SAME implied separation every time; measurement slop does not.
""",

"converge": """
CONVERGE  --  measures the RELATIVE pose of two bases (and their yaw)
=====================================================================
Why it works
    Command both arms to put their end-effectors at the SAME world point.
    If both base transforms are correct, the two frames physically
    coincide.  Whatever gap you can measure between them is the error in
    the relative base transform -- and this needs NO external reference
    frame at all, which is what makes it the most trustworthy of the
    three.

    Yaw falls out of doing it twice.  A relative yaw error produces a gap
    that CHANGES as you move the meeting point along y:

        relative_yaw ~= (gap_x at P2 - gap_x at P1) / (y2 - y1)

What you do
    1. The script brings both grippers to the same commanded world point
       and pauses.  They will not collide -- the meeting point is offset
       so the fingertips sit side by side rather than inside each other.
    2. Measure the gap between the two corresponding features (left
       fingertip to left fingertip) along each axis.
    3. It repeats at a second point displaced along y.  Measure again.
    4. Record both gaps.

Reading the result
    The gap at a single point is the relative translation error.  The
    CHANGE in the x-gap between the two points, divided by their y
    separation, is the relative yaw error in radians.

    This measures right-base-relative-to-left.  It cannot tell you which
    of the two is wrong in world terms -- only that they disagree.  Use
    waist-circle for the absolute placement of each.
""",
}


## ------------------------------------------------------------------ ##

def clearance_ok(sess: MotionSession, q_driver: np.ndarray,
                 margin: float) -> float:
    """Sphere-model clearance at a configuration, in metres."""
    return sess.ik.min_clearance(np.asarray(q_driver, dtype=np.float32))


def pause(msg: str, dry: bool) -> None:
    print(f"\n  >>> {msg}")
    if not dry:
        wait_for_enter("press ENTER when measured... ")


## ------------------------------------------------------------------ ##
## waist-circle
## ------------------------------------------------------------------ ##

def run_floor(frames, bridge, q_driver, args) -> Dict[str, Any]:
    """Lowest point of each arm's geometry vs the tabletop."""
    from arm_config import ARM_CONFIG
    from common import matrix_to_rpy
    import collision_models as CMod

    print(PROCEDURES["floor"])
    import json
    spheres = json.load(open(
        HERE.parent / "ik" / "study" / "results" / "sphere_decomposition.json"))

    fk = frames.fk(bridge.to_urdf(q_driver))
    rows = []
    for arm in args.arms if isinstance(args.arms, list) else ["left", "right"]:
        prefix = f"{arm}_"
        lowest_z, lowest_link = None, None
        for link, ent in spheres.items():
            if not link.startswith(prefix) or link not in fk:
                continue
            T = fk[link]
            c = np.asarray(ent["centers"], dtype=float)
            r = np.asarray(ent["radii"], dtype=float)
            # sphere centres into world, then the bottom of each sphere
            world = (T[:3, :3] @ c.T).T + T[:3, 3]
            z = world[:, 2] - r
            i = int(np.argmin(z))
            if lowest_z is None or z[i] < lowest_z:
                lowest_z, lowest_link = float(z[i]), link

        T_fl = fk[ARM_CONFIG[arm]["ee_link"]]
        rpy = np.degrees(matrix_to_rpy(T_fl[:3, :3]))
        err = lowest_z - args.table_z
        print(f"\n  {arm.upper()}")
        print(f"      flange z            {T_fl[2,3]*1e3:+8.1f} mm")
        print(f"      flange roll/pitch   {rpy[0]:+7.2f} / {rpy[1]:+7.2f} deg")
        print(f"      LOWEST geometry     {lowest_z*1e3:+8.1f} mm   "
              f"(on {lowest_link})")
        print(f"      table is at         {args.table_z*1e3:+8.1f} mm")
        print(f"      => discrepancy      {err*1e3:+8.1f} mm")
        rows.append({
            "arm": arm, "flange_z_m": float(T_fl[2, 3]),
            "flange_rpy_deg": rpy.tolist(),
            "lowest_point_z_m": lowest_z, "lowest_link": lowest_link,
            "table_z_m": args.table_z,
            "discrepancy_mm": err * 1e3,
            "resting_on_table": None,
        })

    print(f"""
  The sphere model is INSCRIBED -- the spheres sit INSIDE the real
  geometry -- so a correct chain reads the lowest point slightly ABOVE the
  table, i.e. a POSITIVE discrepancy. A negative number means the model
  puts geometry below a surface it is resting on, which cannot be true.
""")

    ## Ask NOW, not later.  Which arms were actually touching is the single
    ## piece of context that makes these numbers interpretable, and a field
    ## left in a JSON file to fill in afterwards never gets filled in.
    contact = args.contact
    if contact is None and not args.no_prompt:
        try:
            contact = input("  Which arms were touching the table? "
                            "[both/left/right/none]: ").strip().lower()
        except EOFError:
            contact = None
    if contact not in ("both", "left", "right", "none"):
        contact = "unrecorded"
    for r in rows:
        r["resting_on_table"] = (contact in ("both", r["arm"]))
    print(f"  recorded contact: {contact}")
    if contact != "both":
        print("  (only arms in contact carry information -- the others are "
              "just where they happen to be)")

    return {"protocol": "floor", "table_z_m": args.table_z,
            "contact": contact,
            "pressing_note": ("If an arm was PRESSING into the table rather "
                              "than resting, the links flex. Encoders do "
                              "not see flex, so FK places the tip BELOW the "
                              "surface -- a negative discrepancy that is "
                              "load, not calibration."),
            "arms": rows}


def _feature_offset(args):
    """T_flange_feature for the point you will actually put a ruler on.

    The flange ORIGIN is a virtual frame inside the gripper assembly -- you
    cannot measure to it.  So the gap is computed for a nameable physical
    feature instead, and the same offset is applied in each arm's own
    flange frame (which are differently oriented, so this is not a simple
    translation and must be done per arm)."""
    import tcp as TCP
    import camera_mount as CM

    if args.feature_offset is not None:
        o = np.asarray(args.feature_offset, dtype=float)
        return make_T(o, np.eye(3)), f"custom offset {o.tolist()} m"
    if args.feature == "flange":
        return np.eye(4), ("flange frame ORIGIN -- exact in the model but "
                           "NOT physically identifiable; use it only to "
                           "compare against another computation")
    if args.feature == "tcp":
        t = TCP.resolve("right")["T_flange_tcp"]
        return t, ("TCP / grasp point, between the fingertips "
                   "[derived, unmeasured]")
    m = CM.resolve_mount(f"{args.feature}")
    if m["T_parent_optical"] is not None:
        return m["T_parent_optical"], (f"{args.feature} optical centre "
                                       f"[{m['provenance']}]")
    raise SystemExit(f"  unknown feature '{args.feature}'")


def run_separation(frames, bridge, q_driver, args) -> Dict[str, Any]:
    """Compare the modelled gripper-to-gripper gap against a ruler."""
    from arm_config import ARM_CONFIG
    from common import make_T

    print(PROCEDURES["separation"])
    off, feat_desc = _feature_offset(args)
    fk = frames.fk(bridge.to_urdf(q_driver))
    ## Apply the offset in EACH arm's own flange frame -- they are oriented
    ## differently, so the feature is not the same world-frame offset on
    ## both.
    P = {a: (fk[ARM_CONFIG[a]["ee_link"]] @ off)[:3, 3]
         for a in ("left", "right")}
    d = P["left"] - P["right"]
    gap = float(np.linalg.norm(d))
    print(f"  MEASURING BETWEEN: {feat_desc}")
    print(f"  on the LEFT gripper and the RIGHT gripper -- the same "
          f"physical feature on each.\n")

    jL = frames.tree.parent_joint["left_base_link"]
    jR = frames.tree.parent_joint["right_base_link"]
    urdf_sep = float(abs(jL.xyz[0] - jR.xyz[0]))

    print(f"  left  flange  [{P['left'][0]:+.4f}, {P['left'][1]:+.4f}, "
          f"{P['left'][2]:+.4f}] m")
    print(f"  right flange  [{P['right'][0]:+.4f}, {P['right'][1]:+.4f}, "
          f"{P['right'][2]:+.4f}] m")
    print(f"\n  model says the gap between them is  "
          f"{gap * 1e3:.1f} mm")
    print(f"      dx {d[0]*1e3:+7.1f}   dy {d[1]*1e3:+6.1f}   "
          f"dz {d[2]*1e3:+7.1f}  mm")
    print(f"\n  URDF base x-separation: {urdf_sep * 1e3:.1f} mm\n")
    print(f"      {'base separation':>18s}  {'predicted gap':>14s}")
    print("      " + "-" * 36)
    preds = []
    for extra_mm in (0.0, 25.0, 50.0, 102.0, 150.0):
        dd = d.copy()
        dd[0] += extra_mm * 1e-3
        g = float(np.linalg.norm(dd))
        preds.append({"base_separation_mm": urdf_sep * 1e3 + extra_mm,
                      "predicted_gap_mm": g * 1e3})
        mark = "  <- URDF as-is" if extra_mm == 0 else ""
        print(f"      {urdf_sep*1e3 + extra_mm:15.1f} mm  {g*1e3:11.1f} mm"
              f"{mark}")
    print("      " + "-" * 36)

    result = {
        "protocol": "separation",
        "measured_between": feat_desc,
        "feature": args.feature,
        "urdf_base_x_separation_mm": urdf_sep * 1e3,
        "left_flange_xyz_m": P["left"].tolist(),
        "right_flange_xyz_m": P["right"].tolist(),
        "model_gap_mm": gap * 1e3,
        "delta_xyz_mm": (d * 1e3).tolist(),
        "predictions": preds,
        "measured_gap_mm": args.measured_gap_mm,
        "how_to_complete": (
            "Measure the gap between the same feature on each gripper "
            "mounting plate, then re-run with --measured-gap-mm <value>."),
    }

    if args.measured_gap_mm is not None:
        m = args.measured_gap_mm * 1e-3
        rhs = m * m - d[1] ** 2 - d[2] ** 2
        if rhs <= 0:
            ## The model's y/z offset alone already exceeds what you
            ## measured, so the discrepancy is NOT confined to x and the
            ## single-axis back-solve does not apply.  This is the normal
            ## case when the two features are physically TOUCHING: the
            ## whole 3-vector should be zero, and whatever the model says
            ## instead IS the error.
            print(f"""
  The model's y/z offset ({abs(d[1])*1e3:.1f}, {abs(d[2])*1e3:.1f} mm) already
  exceeds your {args.measured_gap_mm:.1f} mm reading, so the error is NOT
  confined to the x axis and a single-axis back-solve does not apply.

  Treat the model's full separation vector as the discrepancy:

      dx {d[0]*1e3:+8.2f} mm
      dy {d[1]*1e3:+8.2f} mm
      dz {d[2]*1e3:+8.2f} mm
      |d| {gap*1e3:7.2f} mm   vs {args.measured_gap_mm:.1f} mm measured

  If the two features really are in contact, every component should be
  ~0. A large dx alone points at the base x-separation; a mixed error
  points at the base placement in more than one axis, or at the feature
  offset you chose, or at joint calibration.
""")
            result["discrepancy_vector_mm"] = (d * 1e3).tolist()
            result["discrepancy_norm_mm"] = gap * 1e3
            result["single_axis_solve_valid"] = False
        else:
            dx_needed = float(np.sqrt(rhs))
            delta = dx_needed - abs(d[0])
            implied = urdf_sep + delta
            print(f"""
  MEASURED gap {args.measured_gap_mm:.1f} mm
      implies dx = {dx_needed*1e3:.1f} mm  (model has {abs(d[0])*1e3:.1f})
      correction to the base separation: {delta*1e3:+.1f} mm
      implied base x-separation: {implied*1e3:.1f} mm
        (URDF says {urdf_sep*1e3:.1f} mm)

  Repeat at two or three quite different poses. A real base error gives
  the SAME implied separation each time; measurement slop does not.
""")
            result.update({
                "implied_dx_mm": dx_needed * 1e3,
                "correction_mm": delta * 1e3,
                "implied_base_x_separation_mm": implied * 1e3,
            })
    return result


def run_waist_circle(sess: MotionSession, args) -> Dict[str, Any]:
    frames = sess.frames
    arm = sess.arm
    idx = sess.joint_idx
    base_link = f"{arm}_base_link"
    j = frames.tree.parent_joint[base_link]
    urdf_xy = np.asarray(j.xyz, dtype=float)[:2]

    print(PROCEDURES["waist-circle"])
    print(f"  URDF says {base_link} sits at "
          f"(x, y) = ({urdf_xy[0]:+.4f}, {urdf_xy[1]:+.4f}) m"
          f"   yaw = {np.degrees(j.rpy[2]):+.2f} deg")

    q0 = sess.read_q_driver()

    if args.tuck:
        q0 = _tuck_upright(sess, urdf_xy, args)
        if q0 is None:
            return {"protocol": "waist-circle", "arm": arm,
                    "aborted": True,
                    "reason": "could not reach the tucked pose"}

    waist0 = float(q0[idx[0]])

    ## RELATIVE and SYMMETRIC about wherever the waist is now.  Absolute
    ## angles would swing the arm to a fixed orientation regardless of
    ## where it started, which is exactly how you drive it into a bar.
    if args.waist_angles:
        offsets = [float(a) for a in args.waist_angles]
    else:
        half = args.arc / 2.0
        offsets = list(np.linspace(-half, +half, args.marks))
    targets = [waist0 + np.radians(o) for o in offsets]

    print(f"\n  sweeping the waist {args.arc:.0f} deg total "
          f"({offsets[0]:+.1f} .. {offsets[-1]:+.1f} deg) about its current "
          f"angle,\n  in {len(offsets)} marks, holding every other joint.")

    ## Swept envelope of EVERY link of this arm, through the intermediate
    ## configurations too -- the gripper is not the widest part.
    links = WS.arm_links(frames, arm)
    pts = WS.swept_points(frames, sess.bridge, q0, idx[0], targets, links)
    limits = WS.load_limits()
    fp = WS.footprint(pts, about_xy=urdf_xy)
    verified = WS.report_footprint(fp, limits, pts)

    if not sess.dry_run:
        if not verified:
            print("  This sweep has NOT been verified against the "
                  "environment.")
        if not confirm("Does that envelope fit your rig? Type 'yes': "):
            print("  aborted -- narrow the sweep with --arc, or tuck the "
                  "arm in to reduce the radius.")
            return {"protocol": "waist-circle", "arm": arm,
                    "aborted": True, "footprint": fp}

    marks = []
    for off, tgt in zip(offsets, targets):
        q = q0.copy()
        q[idx[0]] = tgt
        cl = clearance_ok(sess, q, args.margin)
        T = sess.ee_pose(q)
        r = float(np.linalg.norm(T[:3, 3][:2] - urdf_xy))
        print(f"\n  waist {off:+7.1f} deg (rel)   "
              f"EE (x,y) = ({T[0,3]:+.4f}, {T[1,3]:+.4f})  "
              f"radius {r*1e3:6.1f} mm   self-clearance {cl*1e3:+6.1f} mm")
        if cl < args.margin:
            print(f"      SKIPPED: self-collision clearance below "
                  f"{args.margin*1e3:.0f} mm")
            continue
        _goto_joint(sess, q, args)
        pause(f"mark the gripper feature on the table "
              f"(waist {off:+.1f} deg)", sess.dry_run)
        marks.append({"waist_offset_deg": off,
                      "waist_driver_rad": float(tgt),
                      "predicted_ee_xy": T[:2, 3].tolist(),
                      "radius_m": r,
                      "self_clearance_m": cl,
                      "measured_xy_m": None})

    _goto_joint(sess, q0, args)

    err_factor = 1.0 / max(1.0 - np.cos(np.radians(args.arc / 2.0)), 1e-6)
    print(f"""
  Returned to the starting configuration.

  NOW, on the table: fit a circle through the marks. Its centre is the
  waist axis = {base_link}'s (x, y). Three non-collinear marks determine a
  circle; more marks average down your marking error.

  At a {args.arc:.0f} deg arc the centre error is about {err_factor:.1f}x your
  marking precision, so ~{err_factor:.0f} mm if you mark to 1 mm.

  Compare the centre with the URDF value above and record it as
  'measured_base_xy_m' in the saved JSON.
""")
    return {"protocol": "waist-circle", "arm": arm, "base_link": base_link,
            "urdf_base_xy_m": urdf_xy.tolist(),
            "urdf_base_yaw_rad": float(j.rpy[2]),
            "arc_deg": args.arc,
            "waist_start_driver_rad": waist0,
            "waist_offsets_deg": offsets,
            "swept_footprint": fp,
            "environment_limits_known": WS.limits_known(limits),
            "environment_verified": bool(verified),
            "centre_error_factor_vs_marking_precision": err_factor,
            "marks": marks,
            "measured_base_xy_m": None,
            "how_to_complete": (
                "Fill measured_xy_m for each mark (metres, in your world "
                "reference), fit a circle through them, and set "
                "measured_base_xy_m to its centre.")}


def _tuck_upright(sess: MotionSession, urdf_xy, args):
    """Bring the arm upright, holding the gripper a short distance from its
    own waist axis.

    An arm standing straight up sweeps almost no floor area when the waist
    turns, which is what makes a FULL revolution practical in a confined
    rig.  It must not sit exactly ON the axis, though: at zero radius every
    mark lands on the same spot and the circle is degenerate.  A modest
    radius keeps the marks well separated while the swept envelope stays
    small."""
    target = np.array([urdf_xy[0] + args.tuck_radius, urdf_xy[1],
                       args.tuck_height], dtype=float)
    q = sess.read_q_driver()
    T = sess.ee_pose(q)
    Tt = T.copy()
    Tt[:3, 3] = target
    print(f"\n  TUCK: bringing the gripper upright to "
          f"({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}) m -- "
          f"{args.tuck_radius*1e3:.0f} mm from the waist axis, "
          f"{args.tuck_height*1e3:.0f} mm up.")

    q_new = sess.solve(q, Tt)
    Tp = sess.ee_pose(q_new)
    reached_r = float(np.linalg.norm(Tp[:3, 3][:2] - urdf_xy))
    cl = clearance_ok(sess, q_new, args.margin)
    print(f"        IK reaches ({Tp[0,3]:+.3f}, {Tp[1,3]:+.3f}, "
          f"{Tp[2,3]:+.3f})  radius {reached_r*1e3:.0f} mm  "
          f"self-clearance {cl*1e3:+.1f} mm")

    if cl < args.margin:
        print(f"        REFUSED: self-collision clearance below "
              f"{args.margin*1e3:.0f} mm")
        return None
    if reached_r < 0.05:
        print(f"        REFUSED: only {reached_r*1e3:.0f} mm from the axis. "
              f"Too close -- the marks would all land on top of each other "
              f"and the circle fit would be degenerate. Raise "
              f"--tuck-radius.")
        return None

    _, env_bad, _ = WS.check_configs(sess.frames, sess.bridge,
                                     [q, q_new], sess.arm)
    if env_bad:
        print(f"        REFUSED: the move into the tucked pose leaves the "
              f"measured free space ({', '.join(b[0] for b in env_bad)})")
        return None

    if not sess.dry_run and not confirm(
            "  Move to the tucked pose? Type 'yes': "):
        return None
    _goto_joint(sess, q_new, args)
    return sess.read_q_driver()


def _goto_joint(sess: MotionSession, q_target: np.ndarray, args) -> None:
    """Interpolate to a joint configuration in safe steps."""
    q_now = sess.read_q_driver()
    n = max(1, int(np.ceil(np.max(np.abs(q_target - q_now)) / args.max_step)))
    for k in range(1, n + 1):
        sess.command(q_now + (q_target - q_now) * (k / n),
                     args.moving_time, args.max_step, args.limit_margin)
        if not sess.dry_run:
            time.sleep(args.moving_time)
    sess.settle(args.settle)


## ------------------------------------------------------------------ ##
## straight
## ------------------------------------------------------------------ ##

def run_straight(sess: MotionSession, args) -> Dict[str, Any]:
    from common import matrix_to_quat_wxyz

    print(PROCEDURES["straight"])
    arm = sess.arm
    j = sess.frames.tree.parent_joint[f"{arm}_base_link"]
    print(f"  URDF says {arm} base yaw = {np.degrees(j.rpy[2]):+.3f} deg\n")

    q = sess.read_q_driver()
    T0 = sess.ee_pose(q)
    L = args.length
    n = args.stations
    print(f"  sweeping {L*1e3:.0f} mm along world +x in {n} stations\n")

    ## Solve the WHOLE path first and check its envelope before moving.
    ## Solving as we go would already have committed the arm to the first
    ## few stations by the time a later one turned out not to fit.
    plan, qp = [], q.copy()
    for k in range(n + 1):
        Tk = T0.copy()
        Tk[0, 3] = T0[0, 3] + L * (k / n - 0.5)
        qp = sess.solve(qp, Tk)
        plan.append((Tk, qp.copy()))
    ok, bad, pts = WS.check_configs(sess.frames, sess.bridge,
                                    [c for _, c in plan], arm)
    WS.report_footprint(WS.footprint(pts), WS.load_limits(), pts)
    if not sess.dry_run and not confirm(
            "Does that envelope fit your rig? Type 'yes': "):
        print("  aborted -- shorten the path with --length.")
        return {"protocol": "straight", "arm": arm, "aborted": True}

    stations = []
    for k in range(n + 1):
        T, q = plan[k]
        cl = clearance_ok(sess, q, args.margin)
        if cl < args.margin:
            print(f"  station {k}: clearance {cl*1e3:+.1f} mm -- SKIPPED")
            continue
        _goto_joint(sess, q, args)
        Tm = sess.ee_pose(sess.read_q_driver())
        print(f"  station {k}: commanded x = {T[0,3]:+.4f}   "
              f"reached ({Tm[0,3]:+.4f}, {Tm[1,3]:+.4f}, {Tm[2,3]:+.4f})  "
              f"clearance {cl*1e3:+.1f} mm")
        pause(f"measure the perpendicular distance to the straightedge "
              f"(station {k})", sess.dry_run)
        stations.append({"station": k, "commanded_x_m": float(T[0, 3]),
                         "reached_xyz_m": Tm[:3, 3].tolist(),
                         "clearance_m": cl,
                         "measured_offset_from_straightedge_m": None})

    print(f"""
  Yaw error, once the readings are in:
      theta = asin((d_last - d_first) / {L:.3f})
  A constant offset is just where the straightedge sits and cancels.
""")
    return {"protocol": "straight", "arm": arm,
            "urdf_base_yaw_rad": float(j.rpy[2]),
            "path_length_m": L, "axis": "world +x",
            "stations": stations, "measured_yaw_error_rad": None,
            "how_to_complete": (
                "Fill measured_offset_from_straightedge_m per station "
                "(metres, signed, +y positive), then "
                "measured_yaw_error_rad = asin((last-first)/path_length_m).")}


## ------------------------------------------------------------------ ##
## converge
## ------------------------------------------------------------------ ##

def run_converge(sess_by_arm: Dict[str, MotionSession], args) -> Dict[str, Any]:
    print(PROCEDURES["converge"])
    arms = list(sess_by_arm)
    any_sess = sess_by_arm[arms[0]]
    frames = any_sess.frames

    meet_points = [np.array([args.meet_x, y, args.meet_z], dtype=float)
                   for y in args.meet_y]
    print(f"  meeting at {len(meet_points)} point(s):")
    for p in meet_points:
        print(f"      ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m")
    print()

    results = []
    for pi, p in enumerate(meet_points):
        entry = {"point_index": pi, "commanded_world_xyz_m": p.tolist(),
                 "arms": {}, "measured_gap_xyz_m": None}
        for arm, sess in sess_by_arm.items():
            # Offset each arm slightly along the separation axis so the two
            # grippers sit side by side instead of trying to occupy the
            # same volume.
            sign = 1.0 if arm == "left" else -1.0
            target = p.copy()
            target[0] += sign * args.side_offset
            q = sess.read_q_driver()
            T = sess.ee_pose(q)
            Tt = T.copy()
            Tt[:3, 3] = target
            q_new = sess.solve(q, Tt)
            cl = clearance_ok(sess, q_new, args.margin)
            Tp = sess.ee_pose(q_new)
            env_ok, env_bad, _ = WS.check_configs(
                sess.frames, sess.bridge, [q, q_new], arm)
            print(f"  [{arm}] target ({target[0]:+.4f}, {target[1]:+.4f}, "
                  f"{target[2]:+.4f})  IK reaches "
                  f"({Tp[0,3]:+.4f}, {Tp[1,3]:+.4f}, {Tp[2,3]:+.4f})  "
                  f"self-clearance {cl*1e3:+.1f} mm")
            if cl < args.margin:
                print(f"      REFUSED: self-collision clearance below "
                      f"{args.margin*1e3:.0f} mm -- not moving this arm")
            elif env_bad:
                print(f"      REFUSED: leaves the measured free space "
                      f"({', '.join(b[0] for b in env_bad)})")
            else:
                _goto_joint(sess, q_new, args)
            entry["arms"][arm] = {
                "commanded_xyz_m": target.tolist(),
                "ik_reached_xyz_m": Tp[:3, 3].tolist(),
                "side_offset_m": sign * args.side_offset,
                "clearance_m": cl,
            }
        pause(f"measure the gap between the two grippers at point {pi}",
              any_sess.dry_run)
        results.append(entry)

    if len(meet_points) >= 2:
        dy = float(meet_points[-1][1] - meet_points[0][1])
        print(f"""
  Relative yaw, once both gaps are measured:
      relative_yaw_rad = (gap_x[last] - gap_x[first]) / {dy:+.3f}
  where gap_x is the x component of the measured gap at each point.
""")
    return {"protocol": "converge", "arms": arms,
            "side_offset_m": args.side_offset,
            "points": results, "measured_relative_yaw_rad": None,
            "how_to_complete": (
                "Fill measured_gap_xyz_m per point (metres, arm order "
                f"{arms[0]} minus {arms[1]}), then take the change in the x "
                "component divided by the y separation for relative yaw.")}


## ------------------------------------------------------------------ ##

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=sorted(PROCEDURES),
                    help="which measurement to run")
    ap.add_argument("--list", action="store_true",
                    help="print all procedures and exit")
    ap.add_argument("--arm", default="right", choices=list(ARM_ORDER))
    ap.add_argument("--arms", nargs=2, default=["left", "right"],
                    help="the two arms for 'converge'")
    ap.add_argument("--feature", default="tcp",
                    help="separation protocol: which physical point to "
                         "measure between on each gripper -- 'tcp' (grasp "
                         "point, default), 'flange' (frame origin, not "
                         "physically identifiable), or a camera name like "
                         "'right_wrist' (its optical centre).")
    ap.add_argument("--feature-offset", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "Z"),
                    help="separation protocol: custom feature offset in the "
                         "FLANGE frame, metres, overriding --feature")
    ap.add_argument("--contact", default=None,
                    choices=("both", "left", "right", "none"),
                    help="floor protocol: which arms were touching the "
                         "table (skips the prompt)")
    ap.add_argument("--no-prompt", action="store_true",
                    help="do not ask which arms were in contact")
    ap.add_argument("--table-z", type=float, default=0.0,
                    help="floor protocol: tabletop height in world z, "
                         "metres (default 0.0 -- the arm bases sit on 20 mm "
                         "risers, which is the 0.020 in the URDF, so the "
                         "tabletop is the z=0 plane)")
    ap.add_argument("--touching", action="store_true",
                    help="separation protocol: the two features are "
                         "physically in contact, i.e. the measured gap is "
                         "zero. Equivalent to --measured-gap-mm 0, and the "
                         "strongest version of this test -- it needs no "
                         "external reference at all.")
    ap.add_argument("--measured-gap-mm", type=float, default=None,
                    help="separation protocol: your ruler reading of the "
                         "gripper-to-gripper gap, in mm. Back-solves the "
                         "implied base separation.")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print without moving anything")
    ap.add_argument("--yes", action="store_true", help="skip the safety prompt")

    g = ap.add_argument_group("waist-circle")
    g.add_argument("--arc", type=float, default=90.0,
                   help="TOTAL waist sweep in degrees, centred on the arm's "
                        "current angle (default 90). Wider pins the circle "
                        "centre better but needs more room -- the procedure "
                        "text has the trade-off. It never does a full "
                        "revolution unless you ask for one.")
    g.add_argument("--marks", type=int, default=5,
                   help="how many marks across the arc (default 5)")
    g.add_argument("--waist-angles", nargs="+", type=float, default=None,
                   help="explicit waist OFFSETS in degrees from the current "
                        "angle, overriding --arc/--marks")
    g.add_argument("--tuck", action="store_true",
                   help="first bring the arm UPRIGHT and close to its own "
                        "waist axis, so a wide sweep needs almost no floor "
                        "space. With --arc 360 this is the most accurate "
                        "and the most compact way to run the protocol.")
    g.add_argument("--tuck-radius", type=float, default=0.15,
                   help="horizontal distance from the waist axis to hold the "
                        "gripper while tucked, metres (default 0.15). This "
                        "is the circle you will mark: big enough that the "
                        "marks are clearly separated, small enough to fit.")
    g.add_argument("--tuck-height", type=float, default=0.50,
                   help="height above the world z=0 plane to hold the "
                        "gripper while tucked, metres (default 0.50)")

    g = ap.add_argument_group("straight")
    g.add_argument("--length", type=float, default=0.30,
                   help="sweep length along world +x, metres (default 0.30)")
    g.add_argument("--stations", type=int, default=4,
                   help="measurement stations along the sweep")

    g = ap.add_argument_group("converge")
    g.add_argument("--meet-x", type=float, default=0.0)
    g.add_argument("--meet-y", nargs="+", type=float, default=[-0.20, 0.05],
                   help="y coordinates of the meeting points; two or more "
                        "are needed to get relative yaw")
    g.add_argument("--meet-z", type=float, default=0.35)
    g.add_argument("--side-offset", type=float, default=0.03,
                   help="lateral offset per arm so the grippers sit side by "
                        "side rather than inside each other, metres")

    g = ap.add_argument_group("motion")
    g.add_argument("--margin", type=float, default=0.005,
                   help="refuse any waypoint whose sphere-model clearance is "
                        "below this, metres (default 0.005). NOTE the sphere "
                        "model is INSCRIBED and reads up to ~18 mm optimistic "
                        "on true contacts, so ordinary safe poses report only "
                        "10-20 mm here -- a refusal threshold near 20 mm would "
                        "reject the whole workspace. The deployed IK uses "
                        "20 mm as a soft COST activation, not a hard limit.")
    g.add_argument("--max-step", type=float, default=0.10)
    g.add_argument("--moving-time", type=float, default=0.15)
    g.add_argument("--limit-margin", type=float, default=0.02)
    g.add_argument("--settle", type=float, default=0.8)
    ap.add_argument("--out", default=None,
                    help="accumulating file to append this run to (default: "
                         "one per protocol under calibration/data/robot/)")
    ap.add_argument("--separate-file", action="store_true",
                    help="write a standalone timestamped file instead of "
                         "appending to the accumulating one")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.list or not args.protocol:
        for name in sorted(PROCEDURES):
            print(PROCEDURES[name])
        if not args.protocol:
            print("  Pick one with --protocol <name>.")
        return

    if getattr(args, "touching", False) and args.measured_gap_mm is None:
        args.measured_gap_mm = 0.0

    ensure_dirs()
    print()
    print("#" * 74)
    print(f"#  Base placement validation  --  {args.protocol}")
    print("#" * 74)

    ## 'separation' reads joint_states and computes -- it never commands
    ## anything and never torques an arm on.  Warning about large sweeps
    ## there would be a lie, and lying in a safety prompt teaches people to
    ## click through the real ones.
    MOVES = {"waist-circle", "straight", "converge"}
    if args.protocol in MOVES and not args.dry_run and not args.yes:
        print("\n  The arms will move through large sweeps. Clear the "
              "workspace.")
        if not confirm("Type 'yes' to continue: "):
            print("  aborted.")
            return
    if args.protocol not in MOVES:
        print("\n  READ-ONLY: this protocol only reads joint states. "
              "Nothing moves,")
        print("  nothing is torqued on, no commands are sent.\n")

    if args.protocol in ("separation", "floor"):
        ## Read-only: nothing moves, nothing is torqued on.
        from kinematics import JointFrameBridge, JointStateListener
        from common import ensure_ros_path
        ensure_ros_path()
        import rospy

        frames = RobotFrames()
        if not rospy.core.is_initialized():
            rospy.init_node("giava_base_separation", anonymous=True,
                            disable_signals=True)
        bridge = JointFrameBridge(frames.robot)
        listener = JointStateListener(("left", "right"))
        rospy.sleep(1.0)
        if not listener.ready():
            raise SystemExit(f"  no joint_states from {listener.missing()} "
                             f"-- is the driver running?")
        q = listener.q_driver(frames)
        if args.protocol == "floor":
            result = run_floor(frames, bridge, q, args)
            tag = "arms"
        else:
            result = run_separation(frames, bridge, q, args)
            tag = "separation"
    elif args.protocol == "converge":
        sessions = {a: MotionSession(a, args.dry_run) for a in args.arms}
        for s in sessions.values():
            s.ik.warmup(s.read_q_driver().astype(np.float32))
        result = run_converge(sessions, args)
        tag = "_".join(args.arms)
    else:
        sess = MotionSession(args.arm, args.dry_run)
        sess.ik.warmup(sess.read_q_driver().astype(np.float32))
        result = (run_waist_circle(sess, args) if args.protocol == "waist-circle"
                  else run_straight(sess, args))
        tag = args.arm

    ## ACCUMULATE by default. These protocols only mean something across
    ## several poses -- "the same answer every time" is the whole test --
    ## so repeated runs go into one file rather than a scatter of
    ## timestamped ones you would have to collate by hand.
    path = (Path(args.out) if args.out else
            DIR_ROBOT / f"base_{args.protocol}_{tag}.json")
    if args.separate_file:
        path = path.with_name(f"{path.stem}_{timestamp()}.json")
        save_json({
            "metadata": provenance(f"base_validation_{args.protocol}",
                                   dry_run=bool(args.dry_run)),
            "convention": _CONVENTION,
            "procedure": PROCEDURES[args.protocol].strip(),
            "result": result,
        }, path, overwrite=args.overwrite)
        print(f"  saved to {path}")
    else:
        n = append_json_run(
            path,
            {"metadata": provenance(f"base_validation_{args.protocol}",
                                    dry_run=bool(args.dry_run)),
             "result": result},
            header={"protocol": args.protocol,
                    "convention": _CONVENTION,
                    "procedure": PROCEDURES[args.protocol].strip()})
        print(f"  appended as run {n - 1} -> {path}   ({n} run(s) total)")
        _cross_run_summary(path, args.protocol)
    print("  Fill in the measured_* fields after taking the readings.\n")


_CONVENTION = ("World frame is giava.urdf's root link 'base': +x operator's "
               "left, +y operator's backward, +z up. Metres and radians.")


def _cross_run_summary(path: Path, protocol: str) -> None:
    """Spread across runs -- the point of repeating at several poses.

    A real geometric error reproduces: the same implied number at every
    pose.  Measurement slop does not.  Printing mean and spread makes that
    distinction immediate instead of something to eyeball across files."""
    from common import load_json

    doc = load_json(path)
    runs = doc.get("runs", [])
    if len(runs) < 2:
        print("  (run this again at a different pose to see the spread)")
        return

    def collect(fn):
        out = []
        for r in runs:
            v = fn(r.get("result", {}))
            if v is not None and np.isfinite(v):
                out.append(float(v))
        return np.asarray(out)

    print()
    print("  " + "-" * 66)
    print(f"  ACROSS {len(runs)} RUNS")
    print("  " + "-" * 66)

    if protocol == "separation":
        v = collect(lambda r: r.get("implied_base_x_separation_mm"))
        if v.size >= 2:
            print(f"    implied base x-separation: "
                  f"mean {v.mean():.1f} mm   sd {v.std(ddof=1):.1f}   "
                  f"range {v.min():.1f}..{v.max():.1f}")
            ## Runs predating this field were all recorded while the URDF
            ## said 938 mm, so that is the correct historical fallback --
            ## do NOT bump it when the URDF changes.
            urdf = runs[-1]["result"].get("urdf_base_x_separation_mm", 938.0)
            print(f"    URDF says {urdf:.1f} mm  ->  "
                  f"correction {v.mean() - urdf:+.1f} mm")
            if v.std(ddof=1) < 10.0:
                print("    CONSISTENT across poses -- this looks like a real "
                      "base error.")
            else:
                print("    SCATTERED across poses -- more likely measurement "
                      "slop or a")
                print("    feature/joint problem than a base offset. Take "
                      "more readings.")
        else:
            print("    no implied separations yet -- pass --measured-gap-mm")
    elif protocol == "floor":
        for arm in ("left", "right"):
            ## Only runs where this arm was actually touching mean anything.
            v = collect(lambda r, a=arm: next(
                (x["discrepancy_mm"] for x in r.get("arms", [])
                 if x["arm"] == a and x.get("resting_on_table") is not False),
                None))
            n_all = sum(1 for r in runs if r.get("result", {}).get("arms"))
            if v.size >= 2:
                print(f"    {arm:5s} lowest-point discrepancy: "
                      f"mean {v.mean():+.1f} mm   sd {v.std(ddof=1):.1f}   "
                      f"range {v.min():+.1f}..{v.max():+.1f}"
                      f"   ({v.size}/{n_all} runs in contact)")
            elif v.size:
                print(f"    {arm:5s}: only {v.size} run(s) in contact")
            else:
                print(f"    {arm:5s}: never recorded as in contact")
    print("  " + "-" * 66)


if __name__ == "__main__":
    main()
