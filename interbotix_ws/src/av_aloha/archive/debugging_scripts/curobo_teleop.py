import rospy
import time
import numpy as np

from scipy.spatial.transform import Rotation as R

from interbotix_xs_modules.arm import (
    InterbotixManipulatorXS
)

from interbotix_xs_msgs.msg import (
    JointSingleCommand
)

from yourdfpy import URDF
import jaxlie

from interbotix_ws.src.av_aloha.data_collection_scripts.webrtc_headset import WebRTCHeadset
from transform_utils import pose2mat


import depthai as dai
import time
import cv2
import numpy as np
from interbotix_ws.src.av_aloha.data_collection_scripts.webrtc_headset import WebRTCHeadset
from headset_control import HeadsetFullControl as HeadsetControl
from headset_utils import HeadsetFeedback
from transform_utils import (
    pose2mat,
)

import torch

import sys
sys.path.append(
    "/home/devi/giava/curobo"
)

from curobo.inverse_kinematics import (
    InverseKinematics,
    InverseKinematicsCfg,
)

from curobo.types import (
    GoalToolPose,
    Pose,
    JointState,
)

from curobo.kinematics import (
    Kinematics,
    KinematicsCfg,
)


# ============================================================
# CONFIG
# ============================================================

URDF_PATH = "/home/devi/giava/giava.urdf"

LEFT_EE_LINK = "left_gripper_base"
RIGHT_EE_LINK = "right_gripper_base"

CONTROL_DT = 0.05

MOVING_TIME = 0.07
ACCEL_TIME = 0.02

POSITION_SCALE = 1.0

ALPHA = 0.2

kin = Kinematics(
    KinematicsCfg.from_robot_yaml_file(
        "/home/devi/giava/curobo/curobo/content/configs/robot/giava_curobo.yml"
    )
)

print(kin.joint_names)

# ========================================================
# ROBOT MODEL
# ========================================================

ik_config = InverseKinematicsCfg.create(
    robot="/home/devi/giava/curobo/curobo/content/configs/robot/giava_curobo.yml",

    num_seeds=1,
    seed_solver_num_seeds=1,

    self_collision_check=True,

    acceleration_regularization_weight=100.0,
    velocity_regularization_weight=1.0,

    success_requires_convergence=False,
)

ik_solver = InverseKinematics(ik_config)

# ============================================================
# HELPERS
# ============================================================

def quat_xyzw_to_wxyz(q):

    return np.array([
        q[3],
        q[0],
        q[1],
        q[2],
    ])

def compute_fk(q_np):

    q_tensor = torch.tensor(
        q_np,
        device="cuda",
        dtype=torch.float32,
    ).unsqueeze(0)

    js = JointState.from_position(
        q_tensor,
        joint_names=kin.joint_names,
    )

    state = kin.compute_kinematics(js)

    left_pose = state.tool_poses["left_gripper_base"]
    right_pose = state.tool_poses["right_gripper_base"]

    return left_pose, right_pose


def solve_bimanual_ik(
    q_current,
    left_target_position,
    right_target_position,
    left_target_wxyz=[0.36085266, -0.60825634, -0.60777247, 0.36113995],

    right_target_wxyz=[0.3611399, -0.60777235, 0.6082564, -0.36085272],
    # left_target_wxyz,
    # right_target_wxyz,
):

    current_q = torch.tensor(
        q_current,
        device="cuda",
        dtype=torch.float32,
    ).unsqueeze(0)

    current_state = JointState.from_position(
        current_q,
        joint_names=kin.joint_names,
    )

    print("\n====================")
    print("CURRENT JOINT STATE")
    print("====================")

    print("q_current shape:", q_current.shape)

    print("q_current:")
    print(np.round(q_current, 3))

    print("Joint names:")
    print(kin.joint_names)

    # TEMPORARY POSITION-ONLY IK TEST

    goal_dict = {

        "left_gripper_base": Pose(
            position=torch.tensor(
                [left_target_position],
                device="cuda",
                dtype=torch.float32,
            ),
            quaternion=torch.tensor(
                [left_target_wxyz],
                device="cuda",
                dtype=torch.float32,
            ),
        ),

        "right_gripper_base": Pose(
            position=torch.tensor(
                [right_target_position],
                device="cuda",
                dtype=torch.float32,
            ),
            quaternion=torch.tensor(
                [right_target_wxyz],
                device="cuda",
                dtype=torch.float32,
            ),
        ),
    }

    print("\n====================")
    print("TARGET POSES")
    print("====================")

    print("LEFT TARGET POSITION:")
    print(np.round(left_target_position, 3))

    print("RIGHT TARGET POSITION:")
    print(np.round(right_target_position, 3))

    print("LEFT TARGET QUAT:")
    print(np.round(left_target_wxyz, 3))

    print("RIGHT TARGET QUAT:")
    print(np.round(right_target_wxyz, 3))

    fk_state = kin.compute_kinematics(current_state)

    left_fk = fk_state.tool_poses.get_link_pose(
        "left_gripper_base"
    )

    right_fk = fk_state.tool_poses.get_link_pose(
        "right_gripper_base"
    )

    

    print("\n====================")
    print("CURRENT FK")
    print("====================")

    left_current = (
        left_fk.position
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    right_current = (
        right_fk.position
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    print("\nLEFT CURRENT POSITION:")
    print(left_current)

    print("\nRIGHT CURRENT POSITION:")
    print(right_current)

    print("\nLEFT CURRENT QUAT:")
    print(
        left_fk.quaternion
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    print("\nRIGHT CURRENT QUAT:")
    print(
        right_fk.quaternion
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    print("\n====================")
    print("TARGET DELTAS")
    print("====================")

    print(
        "LEFT DELTA:",
        np.round(
            left_target_position - left_current,
            3
        )
    )

    print(
        "RIGHT DELTA:",
        np.round(
            right_target_position - right_current,
            3
        )
    )

    result = ik_solver.solve_pose(

        goal_tool_poses=GoalToolPose.from_poses(
            goal_dict,
            ordered_tool_frames=[
                "left_gripper_base",
                "right_gripper_base",
            ],
            num_goalset=1,
        ),

        current_state=current_state,

        return_seeds=1,
    )

    if result.success.any():

        q_new = (
            result.js_solution.position
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        print("\n====================")
        print("IK RESULT")
        print("====================")

        print("Success tensor:")
        print(result.success)

        print("Any success:")
        print(result.success.any())

        new_state = JointState.from_position(
            torch.tensor(
                q_new,
                device="cuda",
                dtype=torch.float32,
            ).unsqueeze(0),
            joint_names=kin.joint_names,
        )

        fk_new = kin.compute_kinematics(new_state)

        left_new = fk_new.tool_poses.get_link_pose(
            "left_gripper_base"
        )

        right_new = fk_new.tool_poses.get_link_pose(
            "right_gripper_base"
        )

        print("\n====================")
        print("POST-IK FK")
        print("====================")

        print("LEFT SOLVED POSITION:")
        print(
            left_new.position
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        print("RIGHT SOLVED POSITION:")
        print(
            right_new.position
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        return q_new

    return None

# ============================================================
# MAIN
# ============================================================

def main():

    rospy.init_node(
        "bimanual_vr_teleop"
    )

    # ========================================================
    # HEADSET
    # ========================================================

    headset = WebRTCHeadset()

    headset.run_in_thread()

    # CAMERA

    # ---- Setup headset ----
    headset_control = HeadsetControl()
    feedback = HeadsetFeedback()
    headset_control.reset()

    # ---- Setup pipeline ----
    pipeline = dai.Pipeline()

    cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    left_out = cam_left.requestOutput(
        (640, 480),
        type=dai.ImgFrame.Type.BGR888i,
        fps=25,
    )

    right_out = cam_right.requestOutput(
        (640, 480),
        type=dai.ImgFrame.Type.BGR888i,
        fps=25,
    )

    q_left = left_out.createOutputQueue()
    q_right = right_out.createOutputQueue()

    pipeline.start()

    time.sleep(0.5)

    # ========================================================
    # JOINT GROUPS
    # ========================================================

    LEFT_ARM_NAMES = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
    ]

    RIGHT_ARM_NAMES = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
    ]

    # MIDDLE_ARM_NAMES = [
    #     "waist",
    #     "shoulder",
    #     "elbow",
    #     "forearm_roll",
    #     "wrist_angle",
    #     "camera_roll",
    # ]

    MIDDLE_ARM_NAMES = [
        "middle_base_link",
        "middle_shoulder_link",
        "middle_upper_arm_link",
        "middle_upper_forearm_link",
        "middle_lower_forearm_link",
        "middle_wrist_link",
        "middle_pan_link",
    ]

    joint_names = kin.joint_names

    left_arm_indices = [
        joint_names.index(name)
        for name in LEFT_ARM_NAMES
    ]

    right_arm_indices = [
        joint_names.index(name)
        for name in RIGHT_ARM_NAMES
    ]

    middle_arm_indices = [
        joint_names.index(name)
        for name in MIDDLE_ARM_NAMES
    ]

    # GRIPPER
    left_gripper_command = JointSingleCommand(
        name="gripper"
    )

    right_gripper_command = JointSingleCommand(
        name="gripper"
    )

    LEFT_GRIPPER_OPEN = 1.5
    LEFT_GRIPPER_CLOSED = -1.5

    RIGHT_GRIPPER_OPEN = 1.5
    RIGHT_GRIPPER_CLOSED = -1.5

    # ========================================================
    # INTERBOTIX
    # ========================================================

    left_bot = InterbotixManipulatorXS(
        robot_model="vx300s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_left",
        moving_time=MOVING_TIME,
        accel_time=ACCEL_TIME,
        init_node=False,
    )

    left_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    left_bot.dxl.robot_torque_enable("single", "gripper", True)
    #left_bot.dxl.robot_set_operating_modes("group", "arm", "position")

    right_bot = InterbotixManipulatorXS(
        robot_model="vx300s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_right",
        moving_time=MOVING_TIME,
        accel_time=ACCEL_TIME,
        init_node=False,
    )

    right_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    right_bot.dxl.robot_torque_enable("single", "gripper", True)
    #right_bot.dxl.robot_set_operating_modes("group", "arm", "position")

    middle_bot = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name=None,
        robot_name="puppet_middle",
        moving_time=2.0,
        accel_time=0.5,
        init_node=False,
    )

    middle_bot.dxl.robot_set_operating_modes(
        "group",
        "arm",
        "position",
    )

    middle_bot.dxl.robot_torque_enable(
        "group",
        "arm",
        True,
    )

    middle_q = [0.05, -1.5, 0.05, -0.08, 2.0, 1.4, 0.0]

    # middle_bot.arm.set_joint_positions(
    #     middle_q.tolist(),
    #     moving_time=2.0,
    #     blocking=False,
    # )

    middle_bot.arm.set_joint_positions(
        middle_q[:6],
        moving_time=2.0,
        accel_time=0.5,
        blocking=True,
    )

    time.sleep(2)
    rospy.sleep(2.0)

    # print(
    #     middle_bot.arm.group_info.joint_names
    # )

    # print(
    #     len(
    #         middle_bot.arm.group_info.joint_names
    #     )
    # )

    rospy.sleep(1.0)

    # ========================================================
    # FULL ROBOT STATE
    # ========================================================

    q = np.zeros(
        len(kin.joint_names)
    )

    q[left_arm_indices] = np.array(
        left_bot.arm.get_joint_commands()
    )

    q[right_arm_indices] = np.array(
        right_bot.arm.get_joint_commands()
    )

    q[middle_arm_indices] = np.array(
        middle_q
    )

    # ========================================================
    # FK
    # ========================================================

    # fk = robot.forward_kinematics(q)

    # T_left = jaxlie.SE3(
    #     fk[left_ee_index]
    # ).as_matrix()

    # T_right = jaxlie.SE3(
    #     fk[right_ee_index]
    # ).as_matrix()

    left_pose, right_pose = compute_fk(q)

    T_left = np.eye(4)
    T_left[:3,3] = (
        left_pose.position
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    T_right = np.eye(4)
    T_right[:3,3] = (
        right_pose.position
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    # ========================================================
    # INITIAL TARGETS
    # ========================================================

    left_start_robot_position = (
        T_left[:3,3].copy()
    )

    right_start_robot_position = (
        T_right[:3,3].copy()
    )

    left_filtered_target_position = (
        left_start_robot_position.copy()
    )

    right_filtered_target_position = (
        right_start_robot_position.copy()
    )

    # ========================================================
    # FIXED ORIENTATION
    # ========================================================

    # left_fixed_rot = R.from_euler(
    #     "xyz",
    #     [-90, 0, -90],
    #     degrees=True,
    # )

    # right_fixed_rot = R.from_euler(
    #     "xyz",
    #     [-90, 0, 90],
    #     degrees=True,
    # )

    # left_target_wxyz = quat_xyzw_to_wxyz(
    #     left_fixed_rot.as_quat()
    # )

    # right_target_wxyz = quat_xyzw_to_wxyz(
    #     right_fixed_rot.as_quat()
    # )

    left_target_wxyz=[0.36085266, -0.60825634, -0.60777247, 0.36113995]

    right_target_wxyz=[0.3611399, -0.60777235, 0.6082564, -0.36085272]

    # ========================================================
    # REMAP
    # ========================================================

    R_remap_left = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ])

    R_remap_right = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ])

    # ========================================================
    # TELEOP STATE
    # ========================================================

    teleop_active = False

    left_start_controller_position = None
    right_start_controller_position = None

    left_start_controller_rot = None
    right_start_controller_rot = None

    left_start_robot_rot = None
    right_start_robot_rot = None

    # ========================================================
    # LOOP
    # ========================================================

    print("\nREADY")

    while not rospy.is_shutdown():

        headset_data = (
            headset.receive_data()
        )

        if headset_data is None:

            time.sleep(0.01)

            continue

        # ----------------------------------------------------
        # CONTROLLERS
        # ----------------------------------------------------

        left_controller = pose2mat(
            headset_data.l_pos,
            headset_data.l_quat,
        )

        right_controller = pose2mat(
            headset_data.r_pos,
            headset_data.r_quat,
        )

        button_pressed = (
            headset_data.r_button_one
        )

        left_trigger = headset_data.l_index_trigger
        right_trigger = headset_data.r_index_trigger

        """ 
        Debugging controller triggers (left/right, index/hand)
        
        left_trigger = headset_data.l_index_trigger
        if left_trigger > 0:
            print("LEFT TRIGGER:", left_trigger)
        right_trigger = headset_data.r_index_trigger
        if right_trigger > 0:
            print("RIGHT TRIGGER:", right_trigger)

        left_trigger2 = headset_data.l_hand_trigger
        if left_trigger2 > 0:
            print("LEFT HAND TRIGGER:", left_trigger2)
        right_trigger2 = headset_data.r_hand_trigger
        if right_trigger2 > 0:
            print("RIGHT HAND TRIGGER:", right_trigger2) """

        # ====================================================
        # ENABLE TELEOP
        # ====================================================

        if (
            button_pressed
            and not teleop_active
        ):

            teleop_active = True

            left_start_controller_position = (
                left_controller[:3,3].copy()
            )

            right_start_controller_position = (
                right_controller[:3,3].copy()
            )

            left_start_robot_position = (
                T_left[:3,3].copy()
            )

            right_start_robot_position = (
                T_right[:3,3].copy()
            )

            left_start_controller_rot = left_controller[:3,:3].copy()
            right_start_controller_rot = right_controller[:3,:3].copy()

            left_start_robot_rot = T_left[:3,:3].copy()
            right_start_robot_rot = T_right[:3,:3].copy()

            print("\\nTeleop ENABLED")

        # ====================================================
        # DISABLE TELEOP
        # ====================================================

        elif (
            not button_pressed
            and teleop_active
        ):

            teleop_active = False

            print("\\nTeleop DISABLED")

        # ====================================================
        # TELEOP
        # ====================================================

        if teleop_active:

            # SEND IMAGES TO HEADSET
            left_img = q_left.get().getCvFrame()
            right_img = q_right.get().getCvFrame()

            headset.send_images(left_img, right_img)

            # ------------------------------------------------
            # LEFT DELTA
            # ------------------------------------------------

            left_delta_controller = (

                left_controller[:3,3]

                - left_start_controller_position
            )

            left_delta_robot = (
                R_remap_left
                @ left_delta_controller
            )

            left_raw_target_position = (

                left_start_robot_position

                + POSITION_SCALE
                * left_delta_robot
            )

            left_filtered_target_position = (

                ALPHA
                * left_raw_target_position

                + (1 - ALPHA)
                * left_filtered_target_position
            )

            left_target_position = (
                left_filtered_target_position
            )

            # ------------------------------------------------
            # RIGHT DELTA
            # ------------------------------------------------

            right_delta_controller = (

                right_controller[:3,3]

                - right_start_controller_position
            )

            right_delta_robot = (
                R_remap_right
                @ right_delta_controller
            )

            right_raw_target_position = (

                right_start_robot_position

                + POSITION_SCALE
                * right_delta_robot
            )

            right_filtered_target_position = (

                ALPHA
                * right_raw_target_position

                + (1 - ALPHA)
                * right_filtered_target_position
            )

            right_target_position = (
                right_filtered_target_position
            )

            # =================================================
            # GRIPPERS
            # =================================================

            cmd = JointSingleCommand(
                name="gripper"
            )

            if right_trigger > 0:
                cmd.cmd = -1.7
                right_bot.gripper.core.pub_single.publish(cmd)
            else:
                cmd.cmd = 0.1
                right_bot.gripper.core.pub_single.publish(cmd)

            if left_trigger > 0:
                cmd.cmd = -1.7
                left_bot.gripper.core.pub_single.publish(cmd)
            else:
                cmd.cmd = 0.1
                left_bot.gripper.core.pub_single.publish(cmd)

            # =================================================
            # ROTATION
            # =================================================

            left_current_rot = left_controller[:3,:3]
            right_current_rot = right_controller[:3,:3]

            left_delta_rot = (
                left_current_rot
                @ left_start_controller_rot.T
            )

            right_delta_rot = (
                right_current_rot
                @ right_start_controller_rot.T
            )

            left_target_rot = (
                left_delta_rot
                @ left_start_robot_rot
            )

            right_target_rot = (
                right_delta_rot
                @ right_start_robot_rot
            )

            left_target_wxyz = quat_xyzw_to_wxyz(
                R.from_matrix(left_target_rot).as_quat()
            )

            right_target_wxyz = quat_xyzw_to_wxyz(
                R.from_matrix(right_target_rot).as_quat()
            )

            left_target_wxyz=[0.36085266, -0.60825634, -0.60777247, 0.36113995]

            right_target_wxyz=[0.3611399, -0.60777235, 0.6082564, -0.36085272]

            # print(vars(headset_data))

            # =================================================
            # WHOLE BODY IK
            # =================================================

            q_new = solve_bimanual_ik(
                q,
                left_target_position,
                right_target_position,
                left_target_wxyz,
                right_target_wxyz,
            )

            # =================================================
            # VALID IK
            # =================================================

            if q_new is not None:

                q = q_new

                # --------------------------------------------
                # LEFT
                # --------------------------------------------

                left_q = q[
                    left_arm_indices
                ]

                left_bot.arm.set_joint_positions(

                    left_q.tolist(),

                    moving_time=MOVING_TIME,

                    accel_time=ACCEL_TIME,

                    blocking=False,
                )

                # --------------------------------------------
                # RIGHT
                # --------------------------------------------

                right_q = q[
                    right_arm_indices
                ]

                right_bot.arm.set_joint_positions(

                    right_q.tolist(),

                    moving_time=MOVING_TIME,

                    accel_time=ACCEL_TIME,

                    blocking=False,
                )

                # --------------------------------------------
                # FK UPDATE
                # --------------------------------------------

                left_pose, right_pose = compute_fk(q)

                T_left[:3,3] = (
                    left_pose.position
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )

                T_right[:3,3] = (
                    right_pose.position
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )

            else:

                print("IK FAILED")

        time.sleep(CONTROL_DT)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()