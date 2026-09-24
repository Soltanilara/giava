"""
Robot definitions, joint names, and predefined poses.
"""

import numpy as np

from paths import URDF_PATH as _URDF_PATH

URDF_PATH = str(_URDF_PATH)

RIGHT_EE_LINK = "right_gripper_base"
LEFT_EE_LINK = "left_gripper_base"
MIDDLE_EE_LINK = "middle_camera_cover"

LEFT_ARM_JOINT_NAMES = ["left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate"]
RIGHT_ARM_JOINT_NAMES = ["right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate"]
MIDDLE_ARM_JOINT_NAMES = ['middle_base', 'middle_shoulder', 'middle_upper_arm', 'middle_upper_forearm', 'middle_lower_forearm', 'middle_wrist', 'middle_pan']

ARM_CONFIG = {
    "left": {
        "robot_name": "puppet_left",
        "robot_model": "vx300s",
        "has_gripper": True,
        "num_joints": 6,
        "joint_names": LEFT_ARM_JOINT_NAMES,
        "ee_link": LEFT_EE_LINK,
    },

    "right": {
        "robot_name": "puppet_right",
        "robot_model": "vx300s",
        "has_gripper": True,
        "num_joints": 6,
        "joint_names": RIGHT_ARM_JOINT_NAMES,
        "ee_link": RIGHT_EE_LINK,
    },

    "middle": {
        "robot_name": "puppet_middle",
        "robot_model": "wx250s",
        "has_gripper": False,
        "num_joints": 7,
        "joint_names": MIDDLE_ARM_JOINT_NAMES,
        "ee_link": MIDDLE_EE_LINK,
    },
}

# Predefined joint configurations for left and right arms

HIGH = np.array([0.11, -0.48, 0.33, -0.03, 1.35, 0.05], dtype=float) # gripper position high camera pointing down at table
LOW = np.array([0.02, 0.037, 0.598, -0.143, 0.986, 0.038], dtype=float) # gripper position a few cm above the table
FORWARD = np.array([0.0, -1.27, 0.99, 0.0, 0.35, 0.0], dtype=float) # camera facing forward, arm lifted off from rest pose
REST = np.array([0.0, -1.9, 1.635, 0.0, 0.7, 0.0], dtype=float) # rest pose where the arm can be powered off safely
R_MED = np.array([0.064, -0.26, 0.495, 0.003, 1.25, -0.023], dtype=float) # New pose similar to but lower than high for easier teleoperation by sb with no experience (Caltrans demo)
DEFAULT_RESET_POSE = "forward"

# Predefined joint configurations for the middle arm
M_HIGH = np.array([3.1185829639434814, -0.13345633447170258, -0.7240389585494995, 0.0, 2.112116756439209, 1.6428934335708618, 2.3], dtype=float)
M_LOW = np.array([3.130854845046997, -1.3959225416183472, 1.087592363357544, -0.05675728991627693, 0.6856894493103027, 1.6444274187088013, 2.3], dtype=float)
# higher forward pose to be able to see tabletop setup and the left and right arm grippers for easier active vision control
M_FORWARD = np.array([2.8440005779266357, -1.3744468688964844, 0.31753402948379517, 0.12118448317050934, 1.4005244970321655, 1.5171070098876953, 2.4804470539093018], dtype=float)
M_REST = np.array([3.1323888301849365, -1.8944662809371948, 1.5938060283660889, -0.09357283264398575, 0.725572943687439, 1.5800002813339233, 2.3], dtype=float)

# Additional middle arm poses
M_FAR_SCENE = np.array([3.15079665184021, -1.4695535898208618, -0.49087387323379517, -0.01840776950120926, 2.112116756439209, 1.7241944074630737, 2.3], dtype=float)
M_LOOKING_LEFT = np.array([4.178563594818115, 0.6366020441055298, 0.04908738657832146, 1.3054176568984985, 1.8545827865600586, -0.6427379846572876, 2.3], dtype=float)
M_LOOKING_RIGHT = np.array([2.113825559616089, 0.771592378616333, -0.31139811873435974, 1.7272623777389526, -1.7717478275299072, 0.5629709362983704, 2.3], dtype=float)

# The previous forward pose which is not ideal for active vision because it is the same forward pose as the right and left, where visibility of the tabletop surface is not ideal
M_FORWARD_DEMO = np.array([-3.113981008529663, -1.4465439319610596, 0.9295923709869385, -0.003067961661145091, 1.2486604452133179, 1.58920419216156, 2.3193790912628174], dtype=float)

# Was considering having this as a pose for recording the arm and scene environment as a way to stand in for a person observing the actions of the arm and the block
# from a place where an observer would be standing in case I wanted to work on smth else while rolling out trained policies. Never actually deployed and will not be
# useful when the middle arm is being used in any data with active vision. 
M_WITNESS = np.array([-2.3331849575042725, -1.9773012399673462, -0.05368933081626892, -0.9433982372283936, 2.175184726715088, 1.228718638420105, 2.3163111209869385], dtype=float)

POSES = {
    "left": {"high": HIGH, "forward": FORWARD, "rest": REST, "low": LOW},
    "right": {"high": HIGH, "forward": FORWARD, "rest": REST, "low": LOW, "med": R_MED},
    "middle": {"high": M_HIGH, "forward": M_FORWARD, "rest": M_REST, "low": M_LOW, "far_scene": M_FAR_SCENE, "looking_left": M_LOOKING_LEFT, "looking_right": M_LOOKING_RIGHT, "witness": M_WITNESS, "forward_demo": M_FORWARD_DEMO},
}