"""Pure kinematics: the URDF model and forward kinematics, with no hardware.

Split out of robot_control.py so that OFFLINE tools -- dataset builders, the
calibration selftest, anything that needs FK without a robot -- can import the
model without dragging in ROS and the Interbotix SDK.  That dependency is what
the `try: import rospy / except ImportError: rospy = None` guards in
robot_control.py used to paper over; with the split they are not needed.

Needs pyroki and yourdfpy.  Needs nothing else.
"""

from __future__ import annotations

import numpy as np
import pyroki as pk
from yourdfpy import URDF

from arm_config import ARM_CONFIG, URDF_PATH
from data_col_config import ARM_MODES


def build_robot_model(mode_idx):
    """Load giava.urdf and index it for the arms in `mode_idx`.

    Returns (robot, arm_data), where arm_data[arm] carries the joint indices,
    the end-effector link index, and the per-arm position limits used to keep
    commands inside what the driver will accept (it rejects the WHOLE group
    command if any single joint is out of range).
    """
    urdf = URDF.load(URDF_PATH)
    robot = pk.Robot.from_urdf(urdf)

    arm_data = {}
    for arm in ARM_MODES[mode_idx]:
        cfg = ARM_CONFIG[arm]
        joint_indices = [
            robot.joints.actuated_names.index(name)
            for name in cfg["joint_names"]
        ]
        arm_data[arm] = {
            "joint_indices": joint_indices,
            "ee_index": robot.links.names.index(cfg["ee_link"]),
            "lower_limits": np.asarray(robot.joints.lower_limits, dtype=float)[joint_indices],
            "upper_limits": np.asarray(robot.joints.upper_limits, dtype=float)[joint_indices],
        }

    return robot, arm_data


def compute_fk_and_ee(robot, q, arm_data):
    """Forward kinematics, plus the end-effector pose of each arm."""
    fk = robot.forward_kinematics(q)
    ee_positions = {arm: fk[arm_data[arm]["ee_index"]] for arm in arm_data}
    return fk, ee_positions
