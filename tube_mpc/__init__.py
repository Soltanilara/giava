"""Tube MPC reference filter for constraint-aware teleoperation.

Sits between a task-space/IK target stream and the robot's joint commands.
Kinematic-level (double-integrator) model, so the plant is exactly LTI and
the robust machinery (constraint tightening, invariant terminal set) is
exact rather than approximated.

Modules:
    model       -- LTI double-integrator joint-space model + limits
    sets        -- LQR gain, support functions, mRPI / tightening margins
    controller  -- the TubeMPC receding-horizon QP (OSQP backend)
    reference   -- reference extrapolators (hold / decayed velocity)
    urdf_limits -- pull joint limits out of a URDF

See MATH.md for the full derivation and README.md for usage.
"""

from tube_mpc.model import JointLimits, KinematicModel
from tube_mpc.controller import TubeMPC
from tube_mpc.config import FilterConfig
from tube_mpc.adapter import FilterRunner, SimAdapter, TeleopAdapter
from tube_mpc.reference import hold_reference, decayed_velocity_reference
from tube_mpc.urdf_limits import parse_urdf_limits

__all__ = [
    "JointLimits",
    "KinematicModel",
    "TubeMPC",
    "FilterConfig",
    "FilterRunner",
    "SimAdapter",
    "TeleopAdapter",
    "hold_reference",
    "decayed_velocity_reference",
    "parse_urdf_limits",
]
