import rospy
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from pynput import keyboard

from interbotix_xs_modules.arm import InterbotixManipulatorXS

import pyroki as pk
from yourdfpy import URDF
import sys
sys.path.append("/home/devi/giava/pyroki/examples")
import pyroki_snippets as pks

import jaxlie

# ============================================================
# CONFIG
# ============================================================

URDF_PATH = "/home/devi/giava/giava.urdf"

LEFT_EE_LINK = "left_gripper_base"

STEP = 0.01

MOVING_TIME = 0.2
ACCEL_TIME = 0.05
CONTROL_DT = 0.05

# ============================================================
# GLOBAL KEY STATE
# ============================================================

pressed_keys = set()

# ============================================================
# KEYBOARD CALLBACKS
# ============================================================

def on_press(key):
    pressed_keys.add(key)

def on_release(key):
    if key in pressed_keys:
        pressed_keys.remove(key)

# ============================================================
# MAIN
# ============================================================

def main():

    rospy.init_node("keyboard_xyz_teleop")

    # --------------------------------------------------------
    # ROBOT MODEL
    # --------------------------------------------------------

    urdf = URDF.load(URDF_PATH)

    robot = pk.Robot.from_urdf(urdf)

    LEFT_ARM_NAMES = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
    ]

    left_arm_indices = [
        robot.joints.actuated_names.index(name)
        for name in LEFT_ARM_NAMES
    ]

    # --------------------------------------------------------
    # INTERBOTIX
    # --------------------------------------------------------

    bot = InterbotixManipulatorXS(
        robot_model="vx300s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_left",
        moving_time=MOVING_TIME,
        accel_time=ACCEL_TIME,
        init_node=False,
    )

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    q = np.zeros(robot.joints.num_actuated_joints)

    current_q = np.array(
        bot.arm.get_joint_commands()
    )

    q[left_arm_indices] = current_q

    # --------------------------------------------------------
    # INITIAL FK
    # --------------------------------------------------------

    fk = robot.forward_kinematics(q)

    print(type(fk))
    print(len(fk))

    

    ee_index = robot.links.names.index(
        LEFT_EE_LINK
    )

    print(type(fk[ee_index]))
    print(fk[ee_index])

    print(np.shape(fk[ee_index]))


    T = jaxlie.SE3(
        fk[ee_index]
    ).as_matrix()

    target_position = T[:3,3].copy()

    full_joint_velocity_limits = np.zeros(
        robot.joints.num_actuated_joints
    )

    full_joint_velocity_limits[
        left_arm_indices
    ] = np.array([
        2.0,
        2.0,
        2.0,
        4.0,
        4.0,
        4.0,
    ])

    # --------------------------------------------------------
    # FIXED ORIENTATION
    # --------------------------------------------------------

    fixed_rot = R.from_euler(
        "xyz",
        [-90, 0, -90],
        degrees=True,
    )

    quat_xyzw = fixed_rot.as_quat()

    target_wxyz = np.array([
        quat_xyzw[3],
        quat_xyzw[0],
        quat_xyzw[1],
        quat_xyzw[2],
    ])

    print("\n====================================")
    print("Keyboard Cartesian Teleop")
    print("====================================")
    print("LEFT / RIGHT arrows -> X")
    print("UP / DOWN arrows    -> Z")
    print("A / D               -> Y")
    print("Q                   -> quit")
    print("====================================\n")

    # --------------------------------------------------------
    # KEYBOARD LISTENER
    # --------------------------------------------------------

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release,
    )

    listener.start()

    # ========================================================
    # LOOP
    # ========================================================

    while not rospy.is_shutdown():

        delta = np.zeros(3)

        # ----------------------------------------------------
        # X
        # ----------------------------------------------------

        if keyboard.Key.left in pressed_keys:
            delta[0] -= STEP

        if keyboard.Key.right in pressed_keys:
            delta[0] += STEP

        # ----------------------------------------------------
        # Z
        # ----------------------------------------------------

        if keyboard.Key.up in pressed_keys:
            delta[2] += STEP

        if keyboard.Key.down in pressed_keys:
            delta[2] -= STEP

        # ----------------------------------------------------
        # Y
        # ----------------------------------------------------

        if hasattr(keyboard.KeyCode, "from_char"):

            if keyboard.KeyCode.from_char('a') in pressed_keys:
                delta[1] += STEP

            if keyboard.KeyCode.from_char('d') in pressed_keys:
                delta[1] -= STEP

            if keyboard.KeyCode.from_char('q') in pressed_keys:
                break

        # ----------------------------------------------------
        # UPDATE TARGET
        # ----------------------------------------------------

        R_remap = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ])

        delta = R_remap @ delta

        target_position += delta

        # ----------------------------------------------------
        # IK
        # ----------------------------------------------------

        q_new = pks.solve_trajectory_ik(

            robot=robot,

            target_link_name=LEFT_EE_LINK,

            target_position=target_position,

            target_wxyz=target_wxyz,

            prev_q=q,

            dt=CONTROL_DT,

            joint_velocity_limits=full_joint_velocity_limits,

            left_arm_indices=left_arm_indices,
        )

        # ----------------------------------------------------
        # VALID IK
        # ----------------------------------------------------

        if q_new is not None:

            q = q_new

            left_q = q[left_arm_indices]

            bot.arm.set_joint_positions(
                left_q.tolist(),
                moving_time=MOVING_TIME,
                accel_time=ACCEL_TIME,
                blocking=False,
            )

            print(
                "target:",
                np.round(target_position, 3)
            )

        else:

            print("IK FAILED")

        time.sleep(CONTROL_DT)

    listener.stop()

if __name__ == "__main__":
    main()