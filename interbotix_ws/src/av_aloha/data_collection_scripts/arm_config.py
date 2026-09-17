"""
Robot definitions, joint names, and predefined poses.
"""

import numpy as np

try:
    from .paths import URDF_PATH as _URDF_PATH
except ImportError:  # run as a script, not a package member
    from paths import URDF_PATH as _URDF_PATH

URDF_PATH = str(_URDF_PATH)

RIGHT_EE_LINK = "right_gripper_base"
LEFT_EE_LINK = "left_gripper_base"
# Must be DOWNSTREAM of the 'middle_pan' joint (chain: middle_pan_link ->
# [pan] -> middle_camera -> middle_camera_cover): the IK pose cost can only
# control the camera-yaw motor if the tracked link is after it.  With
# middle_pan_link the pan joint was invisible to IK and the centering cost
# dragged it ~180 deg.  middle_camera_cover is what ik_study benchmarked.
MIDDLE_EE_LINK = "middle_camera_cover"

LEFT_ARM_JOINT_NAMES = ["left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate"]
RIGHT_ARM_JOINT_NAMES = ["right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate"]
MIDDLE_ARM_JOINT_NAMES = ['middle_base', 'middle_shoulder', 'middle_upper_arm', 'middle_upper_forearm', 'middle_lower_forearm', 'middle_wrist', 'middle_pan']
# MIDDLE_ARM_JOINT_NAMES = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'camera_roll']

# Configuration for each arm, including robot name, model, joint names, and end-effector link.

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

# Predefined joint configurations for useful poses.

HIGH = np.array([0.11, -0.48, 0.33, -0.03, 1.35, 0.05], dtype=float)
LOW = np.array([0.02, 0.037, 0.598, -0.143, 0.986, 0.038], dtype=float)
FORWARD = np.array([0.0, -1.27, 0.99, 0.0, 0.35, 0.0], dtype=float)
REST = np.array([0.0, -1.9, 1.635, 0.0, 0.7, 0.0], dtype=float)


DEFAULT_RESET_POSE = "forward"

M_HIGH = np.array([3.1185829639434814, -0.13345633447170258, -0.7240389585494995, 0.0, 2.112116756439209, 1.6428934335708618, 2.3], dtype=float)
M_LOW = np.array([3.130854845046997, -1.3959225416183472, 1.087592363357544, -0.05675728991627693, 0.6856894493103027, 1.6444274187088013, 2.3], dtype=float)
## Captured off the real arm with oak_pose_finder.py (2026-09-09).  Measured by
## FK against giava.urdf: the camera sits at (0.013, 0.377, 0.467) looking at
## (0.058, 0.512) on the table, with the waist 17 deg off centre (URDF -0.298
## rad) rather than straight down the middle.  All seven joints are inside
## their giava.urdf limits.
##
## That is 16 cm higher than the forward pose it replaces, which sat too low at
## z=0.309.  A first attempt at z=0.489 overshot -- high enough that the view
## had pulled back off the work area -- so this one gives back 2 cm of height
## and 11 cm of reach, which brings the look-at point in from 0.596 to 0.512.
M_FORWARD = np.array([2.8440005779266357, -1.3744468688964844, 0.31753402948379517, 0.12118448317050934, 1.4005244970321655, 1.5171070098876953, 2.4804470539093018], dtype=float)
# retired 2026-09-09, too high (cam z 0.489, look-at pulled out to y=0.596):
# [2.899223804473877, -1.4680196046829224, -0.029145635664463043, 0.09664079546928406, 1.7656118869781494, 1.6183497905731201, 2.31170916557312]
# retired 2026-09-09, too low (cam z 0.309):
# [-3.113981008529663, -1.4465439319610596, 0.9295923709869385, -0.003067961661145091, 1.2486604452133179, 1.58920419216156, 2.3193790912628174]
# 2.8578062057495117, -0.8237476944923401, -0.18867963552474976, -0.1457281857728958, 1.934349775314331, 1.2870099544525146, 2.600097417831421], dtype=float)  # canonical forward (2026-08)
M_REST = np.array([3.1323888301849365, -1.8944662809371948, 1.5938060283660889, -0.09357283264398575, 0.725572943687439, 1.5800002813339233, 2.3], dtype=float)
#M_REST = np.array([3.097107410430908, -1.8392430543899536, 1.5968739986419678, -0.07209710031747818, 0.7025632262229919, 1.6168158054351807, 2.3331849575042725], dtype=float)
# M start pose: position: []


M_FAR_SCENE = np.array([3.15079665184021, -1.4695535898208618, -0.49087387323379517, -0.01840776950120926, 2.112116756439209, 1.7241944074630737, 2.3], dtype=float)
M_LOOKING_LEFT = np.array([4.178563594818115, 0.6366020441055298, 0.04908738657832146, 1.3054176568984985, 1.8545827865600586, -0.6427379846572876, 2.3], dtype=float)
M_LOOKING_RIGHT = np.array([2.113825559616089, 0.771592378616333, -0.31139811873435974, 1.7272623777389526, -1.7717478275299072, 0.5629709362983704, 2.3], dtype=float)

## The middle-arm 'forward' the shape_sorter dataset was COLLECTED from
## (2026-09-07/08).  M_FORWARD was changed on 2026-09-09 -- the new pose
## differs by 38 deg at the elbow and ~20 deg at the waist, which is a
## genuinely different viewpoint, not a 2*pi relabel.  Verified against the
## data: the median middle-arm pose at the first frame of all 138 episodes
## matches THIS to 0.046 rad, and the new M_FORWARD to 5.94 rad.
##
## Any policy trained on that dataset must start here or it is out of
## distribution before its first tick:
##     --start-pose right=forward,middle=forward_demo
## Kept as a separate name rather than reverting M_FORWARD, so whatever the
## new 'forward' was chosen for still works.
M_FORWARD_DEMO = np.array([-3.113981008529663, -1.4465439319610596, 0.9295923709869385, -0.003067961661145091, 1.2486604452133179, 1.58920419216156, 2.3193790912628174], dtype=float)

## Third-person WITNESS view, found by hand with oak_pose_finder.py
## (2026-09-08): the camera arm parked where an observer would stand, so a
## rollout films itself for later review.  For runs where the middle arm is
## NOT commanded -- rollout_policy.py --mode right with
## --witness-cameras oak_left.  Do NOT use it as the start pose for a
## right_av policy: every shape_sorter demo began with the middle arm at
## 'forward', and starting elsewhere is out of distribution from tick one.
## Incidentally 46 deg clear of the +/-pi middle_base seam, unlike 'forward'
## (-3.114) and 'rest' (+3.132), so it does not sit in the stall region.
## Note middle_lower_forearm is 2.175 against an upper limit of 2.237 -- only
## 3.5 deg of headroom, fine for a static pose, tight if anything drives it.
M_WITNESS = np.array([-2.3331849575042725, -1.9773012399673462, -0.05368933081626892, -0.9433982372283936, 2.175184726715088, 1.228718638420105, 2.3163111209869385], dtype=float)

## Mid-workspace start pose for the RIGHT arm, captured off the hardware
## 2026-09-09.  Reachable from anywhere without swinging through the table,
## and every joint sits well inside its limits -- the tightest is wrist_angle
## with 0.98 rad of headroom, against 1.5-3.2 rad on the rest.  That margin is
## the point: it is the pose to hand a first-time operator, because no small
## controller motion from here takes a joint anywhere near a limit.
##
## Given as 6 arm joints.  The capture was 9 values (the full joint_states:
## 6 arm + gripper + the two symmetric finger joints); POSES carries arm
## joints only, and the gripper is commanded on its own path.
R_MED = np.array([0.0644271969795227, -0.26384469866752625, 0.49547579884529114, 0.003067961661145091, 1.2532622814178467, -0.023009711876511574], dtype=float)

POSES = {
    "left": {"high": HIGH, "forward": FORWARD, "rest": REST, "low": LOW},
    "right": {"high": HIGH, "forward": FORWARD, "rest": REST, "low": LOW, "med": R_MED},
    "middle": {"high": M_HIGH, "forward": M_FORWARD, "rest": M_REST, "low": M_LOW, "far_scene": M_FAR_SCENE, "looking_left": M_LOOKING_LEFT, "looking_right": M_LOOKING_RIGHT, "witness": M_WITNESS, "forward_demo": M_FORWARD_DEMO},
}

"""
------------------
MIDDLE ARM POSES
------------------

REST: position: [-3.1323888301849365, -1.8944662809371948, 1.5938060283660889, -0.09357283264398575, 0.725572943687439, 1.5800002813339233, -3.9131851196289062]

FORWARD: position: [2.8440005779266357, -1.3744468688964844, 0.31753402948379517, 0.12118448317050934, 1.4005244970321655, 1.5171070098876953, 2.4804470539093018]   (2026-09-09, camera at z=0.467, looks at y=0.512)

LOW: position: [-3.130854845046997, -1.3959225416183472, 1.087592363357544, -0.05675728991627693, 0.6856894493103027, 1.6444274187088013, 2.3]

HIGH: position: [-3.1185829639434814, -0.13345633447170258, -0.7240389585494995, 0.0, 2.172116756439209, 1.6428934335708618, 2.3]

FAR SCENE: position: [-3.15079665184021, -1.4695535898208618, -0.49087387323379517, -0.01840776950120926, 2.172116756439209, 1.7241944074630737, 2.3]

LOOKING LEFT: position: [-4.178563594818115, 0.6366020441055298, 0.04908738657832146, 1.3054176568984985, 1.8545827865600586, -0.6427379846572876, 2.3]

LOOKING RIGHT: position: [-2.113825559616089, 0.771592378616333, -0.31139811873435974, 1.7272623777389526, -1.7717478275299072, 0.5629709362983704, 2.3]

-------------------------
MIDDLE ARM JOINT LIMITS
-------------------------

  - waist:        -6.35, 0.00
  - shoulder:     -1.89, 1.70
  - elbow:        -2.13, 1.58
  - forearm_roll: -1.58, 1.51
  - wrist_angle:  -1.54, 2.12
  - camera_roll:  -2.78, 2.99
  - camera_yaw:   -3.10, 3.07        (1.00 = looking right)


"""