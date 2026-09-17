import argparse
import copy

import numpy as np
import rospy

from robot_control import (
    create_and_configure_robots,
    interpolate_to_pose,
    get_joint_positions,
)
from data_col_config import ARM_MODES


JOINT_LIMITS = {
    "middle": [
        (-np.pi, np.pi),        # waist
        (-1.884956, 1.989675),  # shoulder
        (-2.146755, 1.605703),  # elbow
        (-np.pi, np.pi),        # forearm_roll
        (-1.745329, 2.146755),  # wrist_angle
        (-np.pi, np.pi),        # camera_roll
        (-np.pi, np.pi),        # camera_yaw
    ]
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="middle")
    return parser.parse_args()


def move_joint(bot, arm_name, pose):
    interpolate_to_pose(
        bot,
        arm_name,
        pose,
        moving_time=0.25,      # time per interpolation segment
        accel_time=0.12,
        blocking=True,
    )


def main():

    rospy.init_node("joint_range_calibration")

    args = parse_args()

    robot = create_and_configure_robots(
        ARM_MODES[args.mode]
    )[args.mode]

    rospy.sleep(1)

    joint_names = robot.arm.group_info.joint_names
    limits = JOINT_LIMITS[args.mode]

    start_pose = get_joint_positions(robot)

    print("\nStarting pose:")
    print(np.round(start_pose, 3))

    input("\nPress ENTER to begin...")

    for joint_idx, joint_name in enumerate(joint_names):

        lower, upper = limits[joint_idx]

        print("\n" + "=" * 60)
        print(f"{joint_name}")
        print(f"Range: {lower:.3f} -> {upper:.3f}")

        # ---------- Move to minimum ----------
        pose = copy.deepcopy(start_pose)
        pose[joint_idx] = lower

        print("\nMoving to MINIMUM...")
        move_joint(robot, args.mode, pose)

        input("Press ENTER to return to START...")

        move_joint(robot, args.mode, start_pose)

        # ---------- Move to maximum ----------
        input("Press ENTER to move to MAXIMUM...")

        pose = copy.deepcopy(start_pose)
        pose[joint_idx] = upper

        print("\nMoving to MAXIMUM...")
        move_joint(robot, args.mode, pose)

        input("Press ENTER to return to START...")

        move_joint(robot, args.mode, start_pose)

        input("Press ENTER for NEXT JOINT...")

    print("\nDone.")

    move_joint(robot, args.mode, start_pose)


if __name__ == "__main__":
    main()