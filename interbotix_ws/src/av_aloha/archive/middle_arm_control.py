import time
import rospy
import numpy as np

from interbotix_xs_modules.arm import InterbotixManipulatorXS


def print_joint_states(bot):

    names = bot.arm.core.joint_states.name
    pos = bot.arm.core.joint_states.position

    joint_map = dict(zip(names, pos))

    print("\njoint states:")

    for k, v in joint_map.items():
        print(f"{k}: {round(v, 3)}")


def main():

    rospy.init_node("real_wx250s_test")

    bot = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name=None,
        robot_name="puppet_middle",
        moving_time=2.0,
        accel_time=0.3,
        init_node=False,
    )

    rospy.sleep(1.0)

    print("\n========================")
    print("CONNECTED")
    print("========================")

    print("\njoint names:")
    print(bot.arm.group_info.joint_names)

    # --------------------------------------------------
    # FORCE OPERATING MODE
    # --------------------------------------------------

    print("\nsetting operating mode...")

    bot.dxl.robot_set_operating_modes(
        "group",
        "arm",
        "position",
    )

    rospy.sleep(0.5)

    # --------------------------------------------------
    # ENABLE TORQUE
    # --------------------------------------------------

    print("enabling torque...")

    bot.dxl.robot_torque_enable(
        "group",
        "arm",
        True,
    )

    rospy.sleep(0.5)

    # --------------------------------------------------
    # PRINT CURRENT STATES
    # --------------------------------------------------

    print_joint_states(bot)

    # --------------------------------------------------
    # TEST 1
    # --------------------------------------------------

    # print("\n========================")
    # print("TEST 1")
    # print("========================")

    # cmd1 = [0.3, -0.3, 0.5, 0.0, 0.0, 0.0]

    # print("sending:")
    # print(cmd1)

    # bot.arm.set_joint_positions(
    #     cmd1,
    #     moving_time=2.0,
    #     accel_time=0.5,
    #     blocking=True,
    # )

    # rospy.sleep(1.0)

    # print_joint_states(bot)

    # --------------------------------------------------
    # TEST 2
    # --------------------------------------------------

    print("\n========================")
    print("TEST 2")
    print("========================")

    cmd2 = [0.010737866163253784, -1.7, 0.03221359848976135, -0.08590292930603027, 2, 1.5999419689178467]


    print("sending:")
    print(cmd2)

    bot.arm.set_joint_positions(
        cmd2,
        moving_time=5.0,
        accel_time=0.5,
        blocking=True,
    )

    rospy.sleep(1.0)

    print_joint_states(bot)

    # --------------------------------------------------
    # HOME
    # # --------------------------------------------------

    # print("\nreturning home...")

    # bot.arm.go_to_home_pose()

    # rospy.sleep(2.0)

    # print("\nDONE")


if __name__ == "__main__":
    main()