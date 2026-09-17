import sys
sys.path.append(
    "/home/devi/giava/curobo"
)

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.scene import Scene, Cuboid, Sphere, Mesh
from curobo.types import JointState

# Joint state representation
q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Forward kinematics
kin = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
js = JointState.from_position(q, joint_names=kin.joint_names)
state = kin.compute_kinematics(js)

# Scene representation
scene = Scene(
    cuboid=[Cuboid(name="table", dims=[1, 1, 0.1], pose=[0, 0, 0.5, 1, 0, 0, 0])],
)

# Inverse kinematics
ik_config = InverseKinematicsCfg.create(robot="franka.yml")
ik = InverseKinematics(ik_config)
result = ik.solve_pose(goal_tool_poses=target_poses)

# Trajectory optimization
trajopt = TrajectoryOptimizer(trajopt_config)
trajectory = trajopt.solve_pose(goal_tool_poses=target_poses)

# Model predictive control
mpc = ModelPredictiveControl(mpc_config)
result = mpc.optimize_action_sequence(current_state)

"""
Public API Modules:

- :mod:`curobo.kinematics` - Forward kinematics
- :mod:`curobo.inverse_kinematics` - Inverse kinematics solver
- :mod:`curobo.trajectory_optimizer` - Trajectory optimization
- :mod:`curobo.motion_planner` - Motion planning
- :mod:`curobo.model_predictive_control` - Model predictive control
- :mod:`curobo.motion_retargeter` - Motion retargeting using inverse kinematics and model predictive control
- :mod:`curobo.scene` - Scene representation with obstacles
- :mod:`curobo.collision_checking` - Robot collision checking (for custom pipelines)
- :mod:`curobo.perception` - Perception utilities (robot segmentation)
- :mod:`curobo.robot_builder` - Build robot configs from URDF
- :mod:`curobo.viewer` - Visualization (Rerun, Viser)
- :mod:`curobo.types` - Common data types (JointState, Pose, etc.)

"""