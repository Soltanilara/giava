"""Robot model loading for the IK study — the single source of truth.

Every part of the study (baseline solver, trajectory suite, metrics, ablation
variants) imports the robot description from here, so that every experiment
runs against the identical model, home configuration, and target links.

Facts about the model (giava.urdf, verified 2026-08-08):

- 23 actuated joints:
    right arm : right_waist, right_shoulder, right_elbow, right_forearm_roll,
                right_wrist_angle, right_wrist_rotate            (6 DoF)
    right hand: right_right_finger, right_left_finger            (2, prismatic)
    left arm  : left_waist ... left_wrist_rotate                 (6 DoF)
    left hand : left_right_finger, left_left_finger              (2, prismatic)
    middle arm: middle_base, middle_shoulder, middle_upper_arm,
                middle_upper_forearm, middle_lower_forearm,
                middle_wrist, middle_pan                         (7 DoF, camera)
- Arm bases: left at (+0.535, -0.021, 0.045), yaw pi-0.24 deg (touch-calibrated 2026-09-03; was the symmetric +0.520, -0.019, 0.02), right at (-0.520, -0.019, 0.02),
  middle (camera) at (0, +0.40, 0.02).  The hand arms face each other across
  the x axis; the camera arm looks at the workspace from +y.
- Home end-effector poses (URDF default configuration):
    left_gripper_base   : (+0.118, -0.019, 0.581)
    right_gripper_base  : (-0.118, -0.019, 0.581)
    middle_camera_cover : ( 0.000, +0.007, 0.319)
- Gripper frame convention (both hands): local +z is the approach axis
  (wrist -> fingertips), fingers open/close along local ±x, local +y is the
  palm normal.  At home both approach axes point at the midline, ~11° above
  horizontal.
- Camera frame convention: local +x is the optical axis; at home it points
  (0, -0.914, -0.407) — toward the hand workspace, pitched down.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pyroki as pk
from yourdfpy import URDF

## giava.urdf lives at the repo root (5 levels up from ik_study); override
## with GIAVA_URDF for non-standard checkouts.
import os as _os
from pathlib import Path as _Path

def _find_urdf() -> str:
    ## Walk up for the marker instead of counting parents[N]: this file has
    ## already moved once (data_collection_scripts/ik/study/ -> av_aloha/ik/).
    here = _Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "giava.urdf").exists():
            return str(cand / "giava.urdf")
    raise SystemExit(f"could not find giava.urdf above {here}")


URDF_PATH = _os.environ.get("GIAVA_URDF", _find_urdf())

ARM_NAMES: Tuple[str, str, str] = ("left", "right", "middle")
TARGET_LINKS: Tuple[str, str, str] = (
    "left_gripper_base",
    "right_gripper_base",
    "middle_camera_cover",
)

_CACHE: dict = {}


def load(with_urdf: bool = False):
    """Load the pyroki robot (cached).  Returns robot, or (robot, urdf)."""
    if "robot" not in _CACHE:
        urdf = URDF.load(URDF_PATH)
        _CACHE["urdf"] = urdf
        _CACHE["robot"] = pk.Robot.from_urdf(urdf)
    if with_urdf:
        return _CACHE["robot"], _CACHE["urdf"]
    return _CACHE["robot"]


def home_config(robot: pk.Robot) -> np.ndarray:
    """URDF default configuration — the initial state for every experiment."""
    return np.asarray(robot.joint_var_cls(0).default_factory(), dtype=np.float32)


def target_link_indices(robot: pk.Robot) -> np.ndarray:
    return np.asarray(
        [robot.links.names.index(n) for n in TARGET_LINKS], dtype=np.int32
    )


def home_poses(robot: pk.Robot, q: np.ndarray | None = None):
    """FK poses of the three target links.  Returns (pos (3,3), wxyz (3,4))."""
    import jaxlie

    if q is None:
        q = home_config(robot)
    fk = robot.forward_kinematics(np.asarray(q, dtype=np.float32))
    pos, wxyz = [], []
    for idx in target_link_indices(robot):
        se3 = jaxlie.SE3(fk[idx])
        pos.append(np.asarray(se3.translation(), dtype=np.float64))
        wxyz.append(np.asarray(se3.rotation().wxyz, dtype=np.float64))
    return np.stack(pos), np.stack(wxyz)


def finger_joint_mask(robot: pk.Robot) -> np.ndarray:
    """True for gripper finger joints.

    Fingers are actuated but kinematically downstream of every target link, so
    no pose cost constrains them.  Motion-quality and joint-limit metrics
    exclude them (their 0–0.041 m range would register as permanently
    'near-limit')."""
    return np.asarray(
        ["finger" in n for n in robot.joints.actuated_names], dtype=bool
    )


def arm_joint_indices(robot: pk.Robot) -> dict:
    """Actuated-joint indices per arm (fingers excluded)."""
    out: dict = {}
    for arm in ARM_NAMES:
        out[arm] = np.asarray(
            [
                i
                for i, n in enumerate(robot.joints.actuated_names)
                if n.startswith(arm) and "finger" not in n
            ],
            dtype=np.int32,
        )
    return out
