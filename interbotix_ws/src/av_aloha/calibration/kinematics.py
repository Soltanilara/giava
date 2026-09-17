"""Frame graph + forward kinematics for the calibration tools.

This is a thin READ-ONLY layer over the machinery the robot already runs on.
Nothing here reimplements kinematics:

  * the robot model is ``pyroki.Robot.from_urdf(giava.urdf)`` -- the same
    object ``robot_control.build_robot_model`` hands to the IK solver;
  * forward kinematics is ``pyroki.Robot.forward_kinematics`` -- the same
    call behind ``robot_control.compute_fk_and_ee``;
  * the driver->URDF joint conversion is imported from ``study_ik``, so the
    middle arm's waist offset and per-joint assembly sign/offsets have
    exactly one definition in the repo (``middle_joint_offsets.json``).

The one thing it adds is an explicit 4x4 ``T_world_link`` view of the FK
output, because the rest of the pipeline composes transforms and the native
pyroki format (a wxyz_xyz 7-vector) is easy to mis-slice.

WORLD FRAME
-----------
``base`` -- the root link of giava.urdf.  Per FRAMES.md (validated against
the deployed teleop remap): +x is the operator's LEFT, +y is BACKWARD
(toward the operator), +z is UP.  All three arm bases sit 20 mm above it.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
## The robot stack is a flat directory of modules, imported by bare name
## (arm_config, robot_control, study_ik...).  Put it on the path so this
## package can be run either as `python -m calibration.x` or directly.
for _p in (str(SCRIPTS_DIR), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    URDF_PATH,
    T_from_xyz_rpy,
    make_T,
    quat_wxyz_to_matrix,
)

## Canonical robot definitions -- reused, never redefined here.
from arm_config import ARM_CONFIG, POSES  # noqa: E402

ARM_ORDER = ("left", "right", "middle")
WORLD_FRAME = "base"


## ------------------------------------------------------------------ ##
## URDF structure (parsed directly from XML -- no FK dependency, so the
## frame report works even without jax installed)
## ------------------------------------------------------------------ ##

class UrdfJoint:
    __slots__ = ("name", "type", "parent", "child", "xyz", "rpy", "axis",
                 "lower", "upper")

    def __init__(self, el: ET.Element):
        self.name = el.get("name")
        self.type = el.get("type")
        self.parent = el.find("parent").get("link")
        self.child = el.find("child").get("link")
        origin = el.find("origin")
        self.xyz = _floats(origin.get("xyz") if origin is not None else None)
        self.rpy = _floats(origin.get("rpy") if origin is not None else None)
        axis = el.find("axis")
        self.axis = _floats(axis.get("xyz")) if axis is not None else None
        lim = el.find("limit")
        self.lower = float(lim.get("lower")) if lim is not None and lim.get("lower") else None
        self.upper = float(lim.get("upper")) if lim is not None and lim.get("upper") else None

    @property
    def T_parent_child(self) -> np.ndarray:
        """Fixed part of the joint transform (at zero joint value)."""
        return T_from_xyz_rpy(self.xyz, self.rpy)

    def __repr__(self) -> str:
        return f"<UrdfJoint {self.name} {self.type} {self.parent}->{self.child}>"


def _floats(s: Optional[str], n: int = 3) -> np.ndarray:
    if not s:
        return np.zeros(n)
    return np.array([float(v) for v in s.split()], dtype=float)


class UrdfTree:
    """The link/joint graph of a URDF, for reporting and offset lookups."""

    def __init__(self, path: str = URDF_PATH):
        self.path = str(path)
        root = ET.parse(self.path).getroot()
        self.robot_name = root.get("name")
        self.links: List[str] = [l.get("name") for l in root.findall("link")]
        self.joints: List[UrdfJoint] = [UrdfJoint(j) for j in root.findall("joint")]
        self.joint_by_name = {j.name: j for j in self.joints}
        self.parent_joint: Dict[str, UrdfJoint] = {j.child: j for j in self.joints}
        self.children: Dict[str, List[str]] = {}
        for j in self.joints:
            self.children.setdefault(j.parent, []).append(j.child)
        parented = set(self.parent_joint)
        roots = [l for l in self.links if l not in parented]
        self.root = roots[0] if roots else self.links[0]
        ## Mesh geometry per link -- the only evidence in this URDF for where
        ## the wrist cameras physically sit (they have no link of their own).
        self.visuals: Dict[str, List[dict]] = {}
        for link in root.findall("link"):
            entries = []
            for vis in link.findall("visual"):
                mesh = vis.find("geometry/mesh")
                if mesh is None:
                    continue
                o = vis.find("origin")
                entries.append({
                    "mesh": mesh.get("filename"),
                    "xyz": _floats(o.get("xyz") if o is not None else None),
                    "rpy": _floats(o.get("rpy") if o is not None else None),
                })
            if entries:
                self.visuals[link.get("name")] = entries

    def chain_to_root(self, link: str) -> List[str]:
        """[root, ..., link] -- the frames a transform passes through."""
        out = [link]
        while link in self.parent_joint:
            link = self.parent_joint[link].parent
            out.append(link)
        return list(reversed(out))

    def joints_between(self, ancestor: str, link: str) -> List[UrdfJoint]:
        chain = self.chain_to_root(link)
        if ancestor not in chain:
            raise ValueError(f"'{ancestor}' is not an ancestor of '{link}'")
        chain = chain[chain.index(ancestor):]
        return [self.parent_joint[c] for c in chain[1:]]

    def fixed_transform(self, ancestor: str, link: str) -> np.ndarray:
        """T_ancestor_link accumulated through FIXED joints only.

        Raises if a movable joint lies between them -- in that case the
        transform is configuration dependent and must come from FK."""
        T = np.eye(4)
        for j in self.joints_between(ancestor, link):
            if j.type != "fixed":
                raise ValueError(
                    f"joint '{j.name}' between '{ancestor}' and '{link}' is "
                    f"{j.type}, not fixed -- use forward kinematics instead")
            T = T @ j.T_parent_child
        return T

    def render_tree(self, root: Optional[str] = None, prefix: str = "",
                    _link: Optional[str] = None) -> List[str]:
        """ASCII frame hierarchy, one line per link with its parent joint."""
        link = _link or root or self.root
        lines = []
        kids = self.children.get(link, [])
        for i, child in enumerate(kids):
            last = i == len(kids) - 1
            j = self.parent_joint[child]
            elbow = "`-- " if last else "|-- "
            detail = (f"[{j.type}]" if j.type == "fixed"
                      else f"[{j.type} '{j.name}']")
            lines.append(
                f"{prefix}{elbow}{child}  {detail}"
                f"  xyz=({j.xyz[0]:+.4f},{j.xyz[1]:+.4f},{j.xyz[2]:+.4f})"
                f"  rpy=({j.rpy[0]:+.4f},{j.rpy[1]:+.4f},{j.rpy[2]:+.4f})")
            lines += self.render_tree(prefix=prefix + ("    " if last else "|   "),
                                      _link=child)
        return lines


## ------------------------------------------------------------------ ##
## Driver <-> URDF joint frame bridge
## ------------------------------------------------------------------ ##

class JointFrameBridge:
    """Converts DRIVER joint vectors to URDF joint vectors, and back.

    The middle (camera) arm's servos do not agree with the URDF: its waist
    is a multiturn joint whose driver zero sits pi from the URDF zero, and
    several joints have flipped axes / mounting offsets measured on the
    real assembly.  ``study_ik`` owns that correction (and the file it
    reads, ``middle_joint_offsets.json``); this class just borrows it so
    there is one definition in the repo rather than two.

    The left/right arms are identity in both directions.

    IMPORTANT: forward kinematics on a MEASURED joint vector is wrong
    unless it goes through ``to_urdf`` first.  Every FK call in
    data_collection.py does this; so does everything here.
    """

    def __init__(self, robot, waist_driver_shift: float = 0.0):
        import study_ik  # lazy: pulls in jax; not needed for URDF-only reports

        self._study_ik = study_ik
        self.waist_urdf_offset = float(np.pi - waist_driver_shift)
        self.waist_driver_shift = float(waist_driver_shift)
        n = robot.joints.num_actuated_joints
        self.signs = np.ones(n, dtype=np.float64)
        self.offsets = np.zeros(n, dtype=np.float64)
        self.actuated_names = list(robot.joints.actuated_names)
        self.waist_idx = self.actuated_names.index("middle_base")
        self.loaded_offsets = study_ik._load_middle_offsets()
        for name, (sign, off) in self.loaded_offsets.items():
            if name == "middle_base":
                continue  # the waist has its own machinery above
            if name in self.actuated_names:
                j = self.actuated_names.index(name)
                self.signs[j] = sign
                self.offsets[j] = off

    def to_urdf(self, q_driver: Sequence[float]) -> np.ndarray:
        """driver -> URDF.  urdf = sign * (driver - offset); waist shifted."""
        q = np.asarray(q_driver, dtype=np.float64).copy()
        q = self.signs * (q - self.offsets)
        w = q[self.waist_idx] + self.waist_urdf_offset
        q[self.waist_idx] = (w + np.pi) % (2 * np.pi) - np.pi
        return q

    def to_driver(self, q_urdf: Sequence[float],
                  ref_driver: Optional[Sequence[float]] = None) -> np.ndarray:
        """URDF -> driver.  Picks the 2pi-equivalent waist value nearest
        ``ref_driver`` so a command never sweeps a full turn."""
        q = np.asarray(q_urdf, dtype=np.float64).copy()
        q = self.signs * q + self.offsets
        d = q[self.waist_idx] - self.waist_urdf_offset
        if ref_driver is not None:
            ref = float(np.asarray(ref_driver)[self.waist_idx])
            d += 2 * np.pi * np.round((ref - d) / (2 * np.pi))
        q[self.waist_idx] = d
        return q

    def describe(self) -> List[str]:
        out = [
            f"waist (middle_base): urdf = driver + {self.waist_urdf_offset:+.6f} rad"
            f"   (pi - driver shift {self.waist_driver_shift:+.6f};"
            f" physical re-clock + Homing_Offset)",
        ]
        if not self.loaded_offsets:
            out.append("per-joint assembly offsets: NONE loaded "
                       "(middle_joint_offsets.json missing, or sim detected)")
        for name, (sign, off) in sorted(self.loaded_offsets.items()):
            if name == "middle_base":
                continue
            out.append(f"{name:22s}: driver = {sign:+d} * urdf "
                       f"{off:+.4f}   ->   urdf = {sign:+d} * (driver "
                       f"{-off:+.4f})")
        return out


## ------------------------------------------------------------------ ##
## Forward kinematics
## ------------------------------------------------------------------ ##

class RobotFrames:
    """pyroki FK over giava.urdf, exposed as 4x4 T_world_link matrices."""

    def __init__(self, urdf_path: str = URDF_PATH):
        import pyroki as pk
        from yourdfpy import URDF

        self.urdf_path = str(urdf_path)
        self.urdf = URDF.load(self.urdf_path)
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.tree = UrdfTree(self.urdf_path)
        self.link_names: List[str] = list(self.robot.links.names)
        self.actuated_names: List[str] = list(self.robot.joints.actuated_names)
        self.num_actuated = self.robot.joints.num_actuated_joints

    # -------------------------------------------------------------- #
    def home_q(self) -> np.ndarray:
        """pyroki's default configuration for this URDF.

        NOT all zeros: pyroki seeds each joint from the URDF and clamps into
        the declared limits, so e.g. both shoulders start at -0.349 rad and
        the fingers half-open at 0.0205 m.  This is the same vector
        ``ik/robot_model.home_config`` returns, and the configuration
        every offline report here defaults to."""
        return np.asarray(
            self.robot.joint_var_cls(0).default_factory(), dtype=np.float64)

    def nonzero_home_joints(self) -> Dict[str, float]:
        """The joints whose default is not zero -- printed by the report so
        'home configuration' is never mistaken for 'all joints at zero'."""
        q = self.home_q()
        return {n: float(v) for n, v in zip(self.actuated_names, q)
                if abs(v) > 1e-9}

    def joint_indices(self, arm: str) -> List[int]:
        return [self.actuated_names.index(n)
                for n in ARM_CONFIG[arm]["joint_names"]]

    def ee_link(self, arm: str) -> str:
        return ARM_CONFIG[arm]["ee_link"]

    # -------------------------------------------------------------- #
    def fk(self, q_urdf: Sequence[float]) -> Dict[str, np.ndarray]:
        """{link_name: T_world_link} for every link, at a URDF configuration.

        `q_urdf` must already be in URDF coordinates -- run measured joint
        values through JointFrameBridge.to_urdf first."""
        q = np.asarray(q_urdf, dtype=np.float32)
        if q.shape[0] != self.num_actuated:
            raise ValueError(
                f"expected {self.num_actuated} actuated joints, got {q.shape[0]}")
        fk = np.asarray(self.robot.forward_kinematics(q))
        out = {}
        for i, name in enumerate(self.link_names):
            # pyroki returns wxyz_xyz: quaternion first, then translation.
            out[name] = make_T(fk[i, 4:7], quat_wxyz_to_matrix(fk[i, 0:4]))
        return out

    def link_pose(self, q_urdf: Sequence[float], link: str) -> np.ndarray:
        """T_world_link for one link."""
        return self.fk(q_urdf)[link]

    def ee_pose(self, q_urdf: Sequence[float], arm: str) -> np.ndarray:
        """T_world_ee for an arm, using the ee_link from arm_config.py."""
        return self.fk(q_urdf)[self.ee_link(arm)]

    # -------------------------------------------------------------- #
    def q_from_named_pose(self, pose_name: str,
                          arms: Sequence[str] = ARM_ORDER) -> np.ndarray:
        """Full DRIVER joint vector assembled from arm_config.POSES.

        Joints belonging to arms not listed (and the finger joints, which no
        pose table covers) stay at zero."""
        q = np.zeros(self.num_actuated, dtype=np.float64)
        for arm in arms:
            if pose_name not in POSES[arm]:
                raise ValueError(
                    f"pose '{pose_name}' is not defined for arm '{arm}' "
                    f"(have: {sorted(POSES[arm])})")
            q[self.joint_indices(arm)] = np.asarray(POSES[arm][pose_name],
                                                    dtype=np.float64)
        return q


## ------------------------------------------------------------------ ##
## Live joint state (optional -- requires ROS + hardware)
## ------------------------------------------------------------------ ##

## ------------------------------------------------------------------ ##
## Gripper opening: driver reading -> giava.urdf finger joints
## ------------------------------------------------------------------ ##
##
## The finger joints used to be left at zero everywhere, on the reasoning
## that "no pose table or IK cost touches them".  That is true of the SOLVER
## and false of everything that DRAWS or CHECKS the robot: a finger at zero
## is a closed gripper, so every viewer showed both hands clamped shut (the
## meshes even interpenetrate slightly at zero), and any clearance measured
## against those links was measured against a gripper that was never open.
## Fingers are among the links most likely to BE the closest inter-arm pair,
## so this was not a cosmetic gap.
##
## The two descriptions parameterise the same opening differently:
##
##   driver  publishes `left_finger` in METRES -- the half-gap from the
##           gripper's centre line (vx300s.urdf.xacro: [0.021, 0.057]),
##           with `right_finger` its exact negation.
##   giava   models each finger as a prismatic joint starting FINGER_ZERO_M
##           off the centre line and sliding OUTWARD, limits
##           [FINGER_CLOSED_M, FINGER_TRAVEL_M].
##
## so the same physical opening is  giava_q = |left_finger| - FINGER_ZERO_M.
FINGER_JOINTS: Dict[str, Tuple[str, str]] = {
    "left": ("left_left_finger", "left_right_finger"),
    "right": ("right_left_finger", "right_right_finger"),
}
FINGER_ZERO_M = 0.0191          # giava.urdf finger origin, |x| in gripper_base
FINGER_DRIVER_JOINT = "left_finger"   # kept for reference; no longer used for the aperture
GRIPPER_DRIVER_JOINT = "gripper"
## Gripper motor angle at the two ends of travel.  MUST match gripper.py
## (GRIPPER_OPEN / GRIPPER_CLOSED): that is what the loop commands.  The
## servo reports -0.376 at the commanded 0.0 "open" (mechanical stop),
## which maps to 88 % of travel rather than 100 % -- honest, not a bug.
GRIPPER_OPEN_RAD = 0.0
GRIPPER_CLOSED_RAD = -1.5
## giava.urdf finger <limit lower>/<upper>.  Lower moved 0 -> 0.010 on
## 2026-09-02 (hand edit, giava.urdf) so the URDF's own joint limit stops a
## live/manual value at the same "closed" position the real gripper's
## driver range already implies (half_gap >= 0.021 => giava_q >= 0.021 -
## FINGER_ZERO_M = 0.0019 in principle, but the true minimum measured on
## the assembly runs a little higher -- 0.010 is the hand-picked floor).
## Both constants MUST track the <limit> values in giava.urdf; nothing
## computes them from the URDF automatically.
FINGER_CLOSED_M = 0.010
FINGER_TRAVEL_M = 0.041         # giava.urdf finger <limit upper>


def finger_q_from_joint_state(msg) -> Optional[float]:
    """giava finger-joint value [m] from one arm's driver joint_states.

    Looked up BY NAME, never by index: the driver appends `gripper`,
    `left_finger` and `right_finger` after the arm joints, and an arm
    configured without a gripper has none of them -- positional slicing
    would silently read a wrist angle as a finger position.  Returns None
    when this arm publishes no gripper, which is the middle (camera) arm.

    Mapped from the GRIPPER MOTOR ANGLE, not from the driver's
    `left_finger`: interbotix derives that linear value with the stock
    vx300s gripper geometry, and on the ALOHA finger it spans only
    14.0 -> 21.5 mm (measured 2026-09-03: closed -> open), which the old
    `|left_finger| - 0.0191` formula clipped to "closed" at BOTH ends.
    The motor angle is what gripper.py commands and what the dataset
    records, so the viewer and the capsule gate now see the same aperture
    the rest of the pipeline does.
    """
    try:
        k = list(msg.name).index(GRIPPER_DRIVER_JOINT)
    except (ValueError, AttributeError):
        return None
    try:
        angle = float(msg.position[k])
    except (IndexError, TypeError):
        return None
    span = GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD
    frac = (angle - GRIPPER_CLOSED_RAD) / span if abs(span) > 1e-9 else 0.0
    ## Clipped into [FINGER_CLOSED_M, FINGER_TRAVEL_M]: a live reading past
    ## either calibrated endpoint would otherwise hand pyroki FK an
    ## out-of-limit joint value.
    return float(np.clip(FINGER_CLOSED_M + frac * (FINGER_TRAVEL_M - FINGER_CLOSED_M),
                         FINGER_CLOSED_M, FINGER_TRAVEL_M))


def fill_fingers(q: np.ndarray, frames: "RobotFrames", arm: str, msg) -> bool:
    """Write both of `arm`'s finger joints from its driver joint_states.

    Returns False when the arm publishes no finger reading, leaving `q`
    untouched -- so a caller can tell "the gripper is closed" from "this arm
    has no gripper", which a bare zero cannot."""
    names = FINGER_JOINTS.get(arm)
    if names is None:
        return False
    v = finger_q_from_joint_state(msg)
    if v is None:
        return False
    for n in names:
        try:
            q[frames.actuated_names.index(n)] = v
        except ValueError:
            return False
    return True


def read_measured_q(robots: Dict[str, object], frames: RobotFrames,
                    arms: Sequence[str] = ARM_ORDER) -> np.ndarray:
    """Assemble a full DRIVER joint vector from live interbotix state.

    `robots` is what ``robot_control.create_and_configure_robots`` returns.
    Finger joints are filled from the driver's own gripper reading (see
    `fill_fingers`); an arm without a gripper leaves them at zero."""
    q = np.zeros(frames.num_actuated, dtype=np.float64)
    for arm in arms:
        n = ARM_CONFIG[arm]["num_joints"]
        js = robots[arm].dxl.joint_states
        measured = np.asarray(js.position[:n], dtype=np.float64)
        q[frames.joint_indices(arm)] = measured
        fill_fingers(q, frames, arm, js)
    return q


class JointStateListener:
    """Read-only joint states for arms we must not torque on.

    Creating an InterbotixManipulatorXS energises the arm.  When a tool only
    needs to KNOW where an arm is -- to record a pose, or to draw it -- that
    is both unnecessary and unsafe.  Subscribing to the driver's
    joint_states topic gives the real pose with no torque and no way to
    command anything.

    Requires the interbotix driver to be running (roslaunch); without it the
    topics are silent and `ready()` stays False rather than returning a
    fabricated pose.
    """

    def __init__(self, arms: Sequence[str] = ARM_ORDER):
        from common import ensure_ros_path
        ensure_ros_path()
        import rospy
        from sensor_msgs.msg import JointState

        self.arms = list(arms)
        self._msgs: Dict[str, object] = {}
        self._subs = []
        for arm in self.arms:
            topic = f"/{ARM_CONFIG[arm]['robot_name']}/joint_states"

            def _cb(msg, _arm=arm):
                self._msgs[_arm] = msg

            self._subs.append(
                rospy.Subscriber(topic, JointState, _cb, queue_size=1))

    def ready(self, arm: Optional[str] = None) -> bool:
        arms = [arm] if arm else self.arms
        return all(a in self._msgs for a in arms)

    def missing(self) -> List[str]:
        return [a for a in self.arms if a not in self._msgs]

    def q_driver(self, frames: "RobotFrames") -> np.ndarray:
        """Full DRIVER joint vector; arms not heard from stay at zero.

        Check ready()/missing() first -- a zero here is 'not heard', not
        'the arm is at zero', and the two must never be confused."""
        q = np.zeros(frames.num_actuated, dtype=np.float64)
        for arm, msg in self._msgs.items():
            n = ARM_CONFIG[arm]["num_joints"]
            pos = np.asarray(msg.position[:n], dtype=np.float64)
            if pos.shape[0] == n:
                q[frames.joint_indices(arm)] = pos
            ## Fingers too, so a viewer drawing this vector shows the hands
            ## as open as they really are.  Left at zero they render (and
            ## collide) as permanently clamped shut.
            fill_fingers(q, frames, arm, msg)
        return q
