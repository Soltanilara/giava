"""Bimanual IK

Same as 01_basic_ik.py, but with two end effectors!
"""

import time
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from yourdfpy import URDF
import numpy as np

import pyroki as pk
from viser.extras import ViserUrdf

import sys

sys.path.append("/home/devi/giava/pyroki/examples")

from pyroki_snippets._solve_ik_with_multiple_targets import (
    solve_ik_with_multiple_targets,
)
import pyroki_snippets as pks

v_max = 2
    

def main():
    urdf = URDF.load("/home/devi/giava/giava.urdf")
    target_link_names = [
        "left_gripper_base",
        "right_gripper_base",
        "middle_camera_cover"
    ]

    robot = pk.Robot.from_urdf(urdf)

    print(robot.joints.names)

    '''
    ('base_left_base_link_fixed', 'left_waist', 'left_shoulder', 'left_elbow', 'left_forearm_roll', 'left_wrist_angle', 
    'left_wrist_rotate', 'left_gripper_link_left_gripper_base_fixed', 'left_left_finger', 'left_right_finger', 
    'base_right_base_link_fixed', 'right_waist', 'right_shoulder', 'right_elbow', 'right_forearm_roll', 'right_wrist_angle', 
    'right_wrist_rotate', 'right_gripper_link_right_gripper_base_fixed', 'right_left_finger', 'right_right_finger', 
    'middle_base_link', 'middle_shoulder_link', 'middle_upper_arm_link', 'middle_upper_forearm_link', 
    'middle_lower_forearm_link', 'middle_wrist_link', 'middle_pan_link', 'base_middle_base_link_fixed', 
    'middle_camera_body_fixed', 'middle_camera_cover_fixed')
    '''

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # target_wxyzs = np.array([
    #     left_wxyz,
    #     right_wxyz,
    # ])

    # target_positions = np.array([
    #     left_pos,
    #     right_pos,
    # ])

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    q = np.zeros(robot.joints.num_actuated_joints)

    # # --- GUI button ---
    # save_button = server.gui.add_button("Save Current Pose")

    # @save_button.on_click
    # def _(_event):
    #     saved_q["value"] = q.copy()
    #     print("Saved q:", saved_q["value"])


    q_rest = np.zeros_like(q)  # or better: a natural pose

    q = np.array([
        0.2, 0.2, 0.2,  0.2, 0.2, 0.2, 0.2,
        0.2, 0.2, 0.2,  0.2, 0.2, 0.2, 0.2,
        0.2, 0.2, 0.2,  0.2, 1.2, 0.2, 0.2,  0.2,  0.2])

    while True:
        start_time = time.time()

        target_wxyzs = np.array([
            ik_target_0.wxyz,
            ik_target_1.wxyz,
            ik_target_2.wxyz
        ])

        target_positions = np.array([
            ik_target_0.position,
            ik_target_1.position,
            ik_target_2.position
        ])

        q_prev = q.copy()

        q_new = pks.solve_ik_with_multiple_targets(
            robot,
            target_link_names,
            target_wxyzs,
            target_positions,
            q_prev=q,
            smoothness_weight=0.1,
            rest_weight=0.05,
            q_rest=q_rest
        )

        dt = time.time() - start_time

        if q_new is not None:
            dq = q_new - q_prev
            # dq = np.clip(dq, -v_max * dt, v_max * dt)
            q = q_prev + dq

        if q is None:
            q = np.zeros(robot.joints.num_actuated_joints)

        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        urdf_vis.update_cfg(q)

if __name__ == "__main__":
    main()
