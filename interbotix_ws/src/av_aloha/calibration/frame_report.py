"""Coordinate-frame diagnostic for the GIAVA rig  (PHASE 1).

Prints the frame hierarchy of giava.urdf, states what each arm's
"end-effector pose" physically refers to, and reports the end-effector
poses at a chosen configuration in an explicitly stated convention.

Run (no hardware needed):

    conda run -n gym_av312 python calibration/frame_report.py
    conda run -n gym_av312 python calibration/frame_report.py --pose forward
    conda run -n gym_av312 python calibration/frame_report.py --tree --json out.json

Run against the live robot (needs ROS + the arms powered):

    conda run -n gym_av312 python calibration/frame_report.py --from-robot

Every transform printed is named T_a_b, meaning: maps points from frame b
into frame a, and equivalently IS the pose of b expressed in a.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ is a SIBLING of this package now, not the
## parent it used to be.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
for _p in (str(HERE), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    T_to_dict,
    ensure_ros_path,
    format_T,
    invert_T,
    provenance,
    save_json,
)
from kinematics import ARM_ORDER, WORLD_FRAME, RobotFrames, UrdfTree  # noqa: E402
from arm_config import ARM_CONFIG  # noqa: E402


## ------------------------------------------------------------------ ##
## Source constants for the TCP analysis.
##
## giava.urdf was flattened from the upstream interbotix vx300s
## description and lost the whole gripper tail, including the canonical
## grasp frame `ee_gripper_link`.  These are the offsets that tail
## contained, read from
##   interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/
##   interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro
## (joints ee_arm / gripper_bar / ee_bar / ee_gripper, all +x, all fixed).
## ------------------------------------------------------------------ ##

UPSTREAM_EE_ARM_X = 0.042825      # gripper_link  -> ee_arm_link
UPSTREAM_GRIPPER_BAR_X = 0.0      # ee_arm_link   -> gripper_bar_link
UPSTREAM_EE_BAR_X = 0.025875      # gripper_bar_link -> fingers_link
UPSTREAM_EE_GRIPPER_X = 0.0385    # fingers_link  -> ee_gripper_link
UPSTREAM_TCP_X = (UPSTREAM_EE_ARM_X + UPSTREAM_GRIPPER_BAR_X
                  + UPSTREAM_EE_BAR_X + UPSTREAM_EE_GRIPPER_X)  # 0.1072 m


def hr(char: str = "=", n: int = 78) -> str:
    return char * n


def section(title: str) -> None:
    print()
    print(hr())
    print(f"  {title}")
    print(hr())
    print()


## ------------------------------------------------------------------ ##
## Report sections
## ------------------------------------------------------------------ ##

def report_conventions() -> None:
    section("CONVENTIONS  (read this first)")
    print("""\
  Transform naming
      T_a_b  maps points from frame b into frame a:   p_a = T_a_b @ p_b
             and IS the pose of frame b expressed in frame a.
             Inverse:  T_b_a = inv(T_a_b).
             Compose with matching inner names:
                 T_world_camera = T_world_ee @ T_ee_camera

  Units       metres and radians, everywhere.
  Quaternions stored scalar-first (wxyz), matching pyroki / jaxlie forward
              kinematics and the IK solver's target format.  scipy's
              Rotation.as_quat() is xyzw -- convert at the boundary.
  rpy         URDF fixed-axis roll-pitch-yaw, i.e. Rz(yaw) @ Ry(pitch) @ Rx(roll).

  World frame The root link of giava.urdf, named 'base'.  Per FRAMES.md
              (validated against the deployed teleop remap):
                  +x  operator's LEFT
                  +y  operator's BACKWARD (toward the operator)
                  +z  UP
              Right-handed.  All three arm bases sit 20 mm above it.""")


def report_world_and_bases(tree: UrdfTree) -> None:
    section("WORLD FRAME AND ARM BASE PLACEMENT")
    print(f"  URDF          : {tree.path}")
    print(f"  robot name    : {tree.robot_name}")
    print(f"  root link     : '{tree.root}'   <-- this is the world frame")
    print(f"  links         : {len(tree.links)}")
    print(f"  joints        : {len(tree.joints)} "
          f"({sum(1 for j in tree.joints if j.type == 'fixed')} fixed, "
          f"{sum(1 for j in tree.joints if j.type == 'revolute')} revolute, "
          f"{sum(1 for j in tree.joints if j.type == 'prismatic')} prismatic)")
    print()
    print("  Arm bases (fixed joints from the world frame):")
    print()
    print("    (x = left/right axis, + toward the operator's LEFT;  "
          "y = front/back axis,")
    print("     + toward the operator;  z = up.  The left-right and "
          "front-back")
    print("     assignment is the pair most often swapped -- check it "
          "against the rig.)")
    print()
    print(f"    {'arm':8s} {'base link':22s} {'x':>9s} {'y':>9s} {'z':>9s} "
          f"{'yaw deg':>9s}")
    print("    " + "-" * 70)
    for arm in ARM_ORDER:
        prefix = arm
        base_link = f"{prefix}_base_link"
        j = tree.parent_joint.get(base_link)
        if j is None:
            continue
        print(f"    {arm:8s} {base_link:22s} "
              f"{j.xyz[0]:+9.4f} {j.xyz[1]:+9.4f} {j.xyz[2]:+9.4f} "
              f"{np.degrees(j.rpy[2]):+9.2f}")
    print()
    print("  Note: the LEFT arm's base is yawed 180 deg, so its local frame")
    print("  points opposite the right arm's.  World-frame targets (what the")
    print("  IK consumes) are unaffected; anything reasoning in an arm's own")
    print("  local frame must account for it.")
    _report_measurable_baselines(tree)


def _report_measurable_baselines(tree: UrdfTree) -> None:
    """Distances between arm bases that a tape measure can check directly.

    The URDF's base placement is an ASSERTION about how the rig is bolted
    together.  Nothing in the software verifies it, and every world-frame
    quantity -- IK targets, inter-arm collision, and eventually camera
    extrinsics -- is wrong by whatever this is wrong by."""
    p = {}
    for arm in ARM_ORDER:
        j = tree.parent_joint.get(f"{arm}_base_link")
        if j is not None:
            p[arm] = np.asarray(j.xyz, dtype=float)
    if len(p) < 3:
        return

    mid_line_y = 0.5 * (p["left"][1] + p["right"][1])
    checks = [
        ("left base  <-> right base", float(np.linalg.norm(p["left"] - p["right"])),
         "straight line between the two manipulator base centres"),
        ("  of which, along x", float(abs(p["left"][0] - p["right"][0])),
         "left/right separation across the workspace"),
        ("middle base <-> left base", float(np.linalg.norm(p["middle"] - p["left"])),
         "camera-arm base to left manipulator base"),
        ("middle base <-> right base", float(np.linalg.norm(p["middle"] - p["right"])),
         "camera-arm base to right manipulator base"),
        ("middle base behind the manipulator line",
         float(abs(p["middle"][1] - mid_line_y)),
         "perpendicular offset toward the operator"),
    ]

    print()
    print("  PHYSICALLY MEASURABLE -- check these with a tape measure:")
    print()
    print(f"      {'quantity':42s} {'URDF says':>12s}")
    print("      " + "-" * 56)
    for label, value, _ in checks:
        print(f"      {label:42s} {value * 1e3:>9.1f} mm")
    print("      " + "-" * 56)
    print("""
      Measure between the same physical features on each base (e.g. the
      centre of each base plate, or one mounting-hole to its twin).  The
      URDF puts all three base ORIGINS 20 mm above the world z=0 plane, so
      measure in the horizontal plane and compare the horizontal numbers.

      These are assertions, not measurements: nothing in the software
      checks them.  If the real rig disagrees, every world-frame quantity
      inherits the error -- IK targets, inter-arm clearance, and any camera
      extrinsic composed through T_world_ee.""")


def report_tree(tree: UrdfTree, full: bool) -> None:
    section("FRAME HIERARCHY")
    if not full:
        print("  (showing the middle/camera arm only -- pass --tree for all "
              "three arms)")
        print()
        print(f"  {tree.root}")
        _print_subtree(tree, "middle_base_link")
        return
    print(f"  {tree.root}")
    for line in tree.render_tree():
        print("  " + line)


def _print_subtree(tree: UrdfTree, link: str) -> None:
    j = tree.parent_joint[link]
    print(f"  `-- {link}  [{j.type}]"
          f"  xyz=({j.xyz[0]:+.4f},{j.xyz[1]:+.4f},{j.xyz[2]:+.4f})"
          f"  rpy=({j.rpy[0]:+.4f},{j.rpy[1]:+.4f},{j.rpy[2]:+.4f})")
    for line in tree.render_tree(_link=link, prefix="      "):
        print("  " + line)


def report_ee_definition(tree: UrdfTree) -> None:
    section("WHAT THE 'END-EFFECTOR POSE' ACTUALLY IS")
    print("""\
  The link each arm's pose is reported at comes from ARM_CONFIG in
  arm_config.py.  It is what forward kinematics returns, what the IK solver
  tracks, and what gets written into datasets.  None of the three is a tool
  centre point.  There is NO ee_link, ee_gripper_link, tcp, tool or tip
  frame anywhere in giava.urdf -- the flattening from the upstream
  interbotix description removed the entire gripper tail.
""")
    for arm in ARM_ORDER:
        link = ARM_CONFIG[arm]["ee_link"]
        chain = tree.chain_to_root(link)
        parent_j = tree.parent_joint[link]
        print(f"  {arm.upper():7s} ee_link = '{link}'")
        print(f"          reached by {parent_j.type} joint "
              f"'{parent_j.name}' from '{parent_j.parent}'")
        print(f"          chain: {' -> '.join(chain)}")
        print()

    print(hr("-"))
    print("  GRIPPER ARMS (left, right):  '*gripper_base'")
    print(hr("-"))
    print(f"""\
  This is the GRIPPER MOUNTING PLATE, not the grasp point.  It sits 35 mm
  along the wrist-rotate axis from '*gripper_link', through a fixed joint
  that also RE-ORIENTS the frame:

      xyz = (0.035, 0, 0)   rpy = (-1.570000, 0.000796, -1.570796)

  Frame convention at '*gripper_base' (per ik/robot_model.py, and
  confirmed by the finger joint geometry below):
      local +z  approach axis, wrist -> fingertips
      local +-x fingers open / close along this axis
      local +y  palm normal

  The fingers are prismatic children of this link, opening along local +-x:
      right_left_finger  origin (+0.0191, -0.014164, +0.021173) axis (0,0,-1)
      right_right_finger origin (-0.0191, -0.014164, +0.021173) axis (0,0,-1)
      travel 0 .. 0.041 m each  ->  38.2 mm to 120.2 mm tip separation

  WHERE THE GRASP POINT PROBABLY IS
  The upstream vx300s description (vendored at interbotix_xsarm_descriptions/
  urdf/vx300s.urdf.xacro) defines the canonical grasp frame ee_gripper_link
  as a chain of fixed +x offsets from gripper_link:
      ee_arm      +{UPSTREAM_EE_ARM_X:.6f}
      gripper_bar +{UPSTREAM_GRIPPER_BAR_X:.6f}
      ee_bar      +{UPSTREAM_EE_BAR_X:.6f}
      ee_gripper  +{UPSTREAM_EE_GRIPPER_X:.6f}
      total       +{UPSTREAM_TCP_X:.6f} m along gripper_link's +x""")
    T_gl_gb = tree.fixed_transform("right_gripper_link", "right_gripper_base")
    p = (invert_T(T_gl_gb) @ np.array([UPSTREAM_TCP_X, 0.0, 0.0, 1.0]))[:3]
    print(f"""
  Expressed in '*gripper_base' coordinates that lands at

      ({p[0]:+.5f}, {p[1]:+.5f}, {p[2]:+.5f})  m

  i.e. {p[2] * 1e3:.1f} mm out along the local +z approach axis, on the
  finger centreline.

  *** THIS IS A DERIVED CANDIDATE, NOT A MEASUREMENT. ***
  It assumes GIAVA's custom fingers (vx300s_8_custom_finger_*.stl) put the
  grasp point where the stock ones did, which is exactly the sort of thing
  that needs a ruler.  The finger STL origins in giava.urdf sit at the
  gripper_base origin, so they give no independent fingertip evidence.

  Until measured: the pose this pipeline reports for the gripper arms is
  the mounting plate, and any grasp-point claim is offset from it by an
  unvalidated ~{p[2] * 1e3:.0f} mm along local +z.
""")

    print(hr("-"))
    print("  CAMERA ARM (middle):  'middle_camera_cover'")
    print(hr("-"))
    print("""\
  A cosmetic link.  Both 'middle_camera_body' and 'middle_camera_cover' are
  attached to 'middle_camera' by fixed joints whose origin is EXACTLY
  identity (xyz = 0 0 0, rpy = 0 0 0), so all three links are the same
  frame.  'middle_camera_cover' is used as the IK target only because it is
  downstream of the 'middle_pan' joint -- with 'middle_pan_link' the solver
  could not see the camera-yaw motor at all (arm_config.py:11-15).

  Frame convention: local +x is the optical axis (ik/robot_model.py).
  At the URDF home configuration it points (0, -0.914, -0.407) -- toward
  the hand workspace, pitched down.

  This is a MOUNT frame, not an optical frame.  The OAK-D stereo pair
  actually bolted there has no pose defined relative to it anywhere in the
  repository.  See calibration/camera_mount.py.
""")


def report_cameras(tree: UrdfTree) -> None:
    section("CAMERA FRAMES IN THE URDF")
    print("""\
  There are NO camera optical frames in giava.urdf, and no camera links at
  all for the wrist cameras.  What exists:

  1. 'middle_camera' (+ its two identity children) -- a LINK on the camera
     arm, downstream of the 'middle_pan' joint.  A mount frame, not optical.

  2. The wrist RealSense D405s -- geometry ONLY.  Each gripper base carries
     an 'aloha_assets/d405_solid.stl' visual+collision mesh.  A mesh origin
     is not a frame: nothing in FK or TF reports a wrist camera pose.
""")
    for link in ("left_gripper_base", "right_gripper_base"):
        for v in tree.visuals.get(link, []):
            if "d405" not in str(v["mesh"]).lower():
                continue
            print(f"     {link}: {Path(str(v['mesh'])).name}")
            print(f"        xyz = ({v['xyz'][0]:+.6f}, {v['xyz'][1]:+.6f}, "
                  f"{v['xyz'][2]:+.6f})")
            print(f"        rpy = ({v['rpy'][0]:+.6f}, {v['rpy'][1]:+.6f}, "
                  f"{v['rpy'][2]:+.6f})"
                  f"   [{np.degrees(v['rpy'][0]):.1f} deg tilt about x]")
    print()
    print("  3. The static room cameras (top_scene, low_scene) are not in the")
    print("     URDF at all -- they are not attached to the robot.")
    print()
    print("  Run  python calibration/camera_mount.py  for the full mount")
    print("  report, including the nominal optical-frame transforms and what")
    print("  provenance each one has.")


def report_driver_bridge(frames: RobotFrames, enabled: bool) -> None:
    section("DRIVER <-> URDF JOINT FRAME BRIDGE")
    if not enabled:
        print("  (skipped -- pass --driver-frame to load it; it imports jax)")
        return
    from kinematics import JointFrameBridge
    print("""\
  The middle arm's servos do not agree with the URDF.  Forward kinematics
  on a MEASURED joint vector is wrong unless converted first.  This is
  owned by study_ik.py and middle_joint_offsets.json; reported here so the
  conversion in force right now is visible.
""")
    bridge = JointFrameBridge(frames.robot)
    for line in bridge.describe():
        print("    " + line)
    print()
    print("  The left and right arms are identity in both directions.")


def report_poses(frames: RobotFrames, q_urdf: np.ndarray, label: str,
                 links: Optional[List[str]]) -> Dict[str, Any]:
    section(f"END-EFFECTOR POSES  --  {label}")
    fk = frames.fk(q_urdf)
    out: Dict[str, Any] = {}
    for arm in ARM_ORDER:
        link = ARM_CONFIG[arm]["ee_link"]
        T = fk[link]
        print(f"  {arm.upper()}   T_{WORLD_FRAME}_{link}")
        print(f"        (pose of '{link}' expressed in the world frame "
              f"'{WORLD_FRAME}')")
        print(format_T(T, indent="        "))
        print()
        out[arm] = T_to_dict(T, WORLD_FRAME, link)
    if links:
        print(hr("-"))
        print("  Additional requested links:")
        print()
        for link in links:
            if link not in fk:
                print(f"    {link}: NOT A LINK IN THIS URDF")
                continue
            print(f"  T_{WORLD_FRAME}_{link}")
            print(format_T(fk[link], indent="        "))
            print()
            out[link] = T_to_dict(fk[link], WORLD_FRAME, link)
    return out


## ------------------------------------------------------------------ ##

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", default=None,
                    help="named pose from arm_config.POSES (driver coords), "
                         "e.g. forward/rest/high/low. Default: URDF home.")
    ap.add_argument("--from-robot", action="store_true",
                    help="read live joint states from the arms (needs ROS)")
    ap.add_argument("--arms", nargs="+", default=list(ARM_ORDER),
                    choices=list(ARM_ORDER),
                    help="arms to read when using --from-robot")
    ap.add_argument("--link", nargs="+", default=None,
                    help="extra link names to report poses for")
    ap.add_argument("--tree", action="store_true",
                    help="print the full three-arm frame hierarchy")
    ap.add_argument("--driver-frame", action="store_true",
                    help="report the driver<->URDF joint bridge (imports jax)")
    ap.add_argument("--json", default=None, help="write the report to JSON")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow --json to replace an existing file")
    args = ap.parse_args()

    print()
    print("#" * 78)
    print("#  GIAVA coordinate frame report  (PHASE 1)")
    print("#" * 78)

    report_conventions()

    frames = RobotFrames()
    tree = frames.tree

    report_world_and_bases(tree)
    report_tree(tree, args.tree)
    report_ee_definition(tree)
    report_cameras(tree)

    ## --from-robot needs the bridge regardless of the flag: measured joint
    ## values are in driver coordinates and FK would otherwise be wrong.
    need_bridge = args.driver_frame or args.from_robot or bool(args.pose)
    report_driver_bridge(frames, need_bridge)

    q_source: Dict[str, Any] = {}
    if args.from_robot:
        q_urdf, label, q_source = _q_from_robot(frames, args.arms)
    elif args.pose:
        from kinematics import JointFrameBridge
        q_driver = frames.q_from_named_pose(args.pose)
        q_urdf = JointFrameBridge(frames.robot).to_urdf(q_driver)
        label = f"named pose '{args.pose}' (driver coords -> URDF)"
        q_source = {"kind": "named_pose", "pose": args.pose,
                    "q_driver": q_driver.tolist()}
    else:
        q_urdf = frames.home_q()
        nz = frames.nonzero_home_joints()
        label = "pyroki default configuration for giava.urdf"
        q_source = {"kind": "urdf_default", "nonzero_joints": nz}
        print()
        print("  NOTE: the default configuration is NOT all zeros -- pyroki")
        print("  seeds joints from the URDF and clamps into their limits.")
        print(f"  {len(nz)} joints start away from zero:")
        for n, v in sorted(nz.items()):
            print(f"      {n:24s} {v:+.6f}")

    poses = report_poses(frames, q_urdf, label, args.link)

    print(hr())
    print("  Summary")
    print(hr())
    print("""\
  * World frame is giava.urdf's root link 'base': +x operator-left,
    +y operator-backward, +z up.
  * 'End-effector pose' means the GRIPPER MOUNTING PLATE for the hand arms
    and a CAMERA MOUNT LINK for the camera arm.  Neither is a TCP.
  * No TCP / tool offset is defined anywhere in the repository, and none is
    applied after FK by any code.
  * No camera optical frame exists in the URDF; the wrist cameras are mesh
    geometry only.
""")

    if args.json:
        _write_json(Path(args.json), frames, tree, q_urdf, label, q_source,
                    poses, args.overwrite)


def _q_from_robot(frames: RobotFrames, arms: List[str]):
    """Read measured joints and convert to URDF coordinates."""
    ensure_ros_path()
    import rospy
    from kinematics import JointFrameBridge, read_measured_q
    from robot_control import (create_and_configure_robots,
                               read_middle_waist_shift)

    rospy.init_node("giava_frame_report", anonymous=True, disable_signals=True)
    robots = create_and_configure_robots(tuple(arms))
    rospy.sleep(0.5)

    shift = (read_middle_waist_shift(robots["middle"])
             if "middle" in robots else 0.0)
    bridge = JointFrameBridge(frames.robot, waist_driver_shift=shift)
    q_driver = read_measured_q(robots, frames, arms)
    q_urdf = bridge.to_urdf(q_driver)
    return (q_urdf,
            f"MEASURED joint state, arms={','.join(arms)} "
            f"(driver coords -> URDF)",
            {"kind": "measured", "arms": list(arms),
             "waist_homing_offset_rad": float(shift),
             "q_driver": q_driver.tolist()})


def _write_json(path: Path, frames: RobotFrames, tree: UrdfTree,
                q_urdf: np.ndarray, label: str, q_source: Dict[str, Any],
                poses: Dict[str, Any], overwrite: bool) -> None:
    report = {
        "metadata": provenance("frame_report", configuration=label),
        "convention": (
            "T_a_b maps points from frame b into frame a; equivalently it is "
            "the pose of b expressed in a."
        ),
        "world_frame": {
            "link": tree.root,
            "axes": {"+x": "operator's left", "+y": "operator's backward",
                     "+z": "up"},
            "source": "giava.urdf root link; axes per FRAMES.md",
        },
        "arms": {
            arm: {
                "ee_link": ARM_CONFIG[arm]["ee_link"],
                "ee_link_is_tcp": False,
                "ee_link_meaning": (
                    "gripper mounting plate (35 mm past the wrist-rotate "
                    "axis, re-oriented)" if arm in ("left", "right")
                    else "camera mount link, identical to 'middle_camera'"),
                "joint_names": ARM_CONFIG[arm]["joint_names"],
                "robot_name": ARM_CONFIG[arm]["robot_name"],
                "robot_model": ARM_CONFIG[arm]["robot_model"],
            } for arm in ARM_ORDER
        },
        "tcp": {
            "defined_in_repository": False,
            "candidate_gripper_tcp_offset_in_gripper_base_m": [
                0.0, 0.0, round(float(
                    (invert_T(tree.fixed_transform(
                        "right_gripper_link", "right_gripper_base"))
                     @ np.array([UPSTREAM_TCP_X, 0, 0, 1.0]))[2]), 6)],
            "candidate_source": (
                "upstream interbotix vx300s.urdf.xacro ee_gripper_link, "
                "0.1072 m along gripper_link +x, re-expressed in "
                "gripper_base coordinates"),
            "validated": False,
        },
        "configuration": q_source,
        "q_urdf": q_urdf.tolist(),
        "actuated_joint_names": frames.actuated_names,
        "ee_poses": poses,
    }
    save_json(report, path, overwrite=overwrite)
    print(f"  JSON report written to {path}")


if __name__ == "__main__":
    main()
