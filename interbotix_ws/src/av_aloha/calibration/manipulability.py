"""How well can this configuration move in each direction?

Some arm configurations are simply bad at moving certain ways.  Near a
singularity, or near a reach limit, a small end-effector motion needs a
large joint motion -- and the solver, which is minimising a weighted cost,
would rather tilt the wrist than contort the arm.  That is why a commanded
"straight up" can come out short and angled: it is not a bug, it is the
configuration.

This quantifies it, so the intuition has numbers behind it.

THE MEASURE
    Take the translational Jacobian J (3 x n) of the tracked point.  For a
    unit end-effector velocity along a direction d, the least-norm joint
    velocity is J^+ d, so

        cost(d)  = || J^+ d ||      rad of joint motion per metre of EE motion
        gain(d)  = 1 / cost(d)      m of EE motion per rad of joint motion

    A LOW gain means that direction is expensive: the arm has to move a lot
    to achieve a little, tracking will lag, and the solver will happily
    trade the position for something cheaper.

    The singular values of J give the same information as an ellipsoid --
    the directions the arm moves easily are the long axes.  The ratio
    sigma_max / sigma_min is the condition number; large means the
    configuration is close to degenerate, and the smallest singular vector
    is the direction that is nearly impossible.

WHAT IT DOES NOT MODEL
    Gravity, servo stiffness, backlash and link flex.  This is pure
    kinematics: it says what the geometry allows, not what the hardware
    delivers.  A direction can have excellent manipulability and still
    track badly because the arm is holding its own weight.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ is a SIBLING of this package now, not the
## parent it used to be.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
for _p in (str(HERE), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from arm_config import ARM_CONFIG  # noqa: E402

WORLD_AXES = {
    "+x": np.array([1.0, 0.0, 0.0]), "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]), "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]), "-z": np.array([0.0, 0.0, -1.0]),
}


def translational_jacobian(frames, bridge, q_driver: np.ndarray, arm: str,
                           point_fn=None, eps: float = 1e-5) -> np.ndarray:
    """3 x n_arm_joints Jacobian of the tracked point, in world.

    Finite differences on the same FK the rest of the pipeline uses, so it
    cannot drift from it.  `point_fn(q_driver) -> 4x4` lets the caller
    measure the TCP instead of the flange."""
    idx = frames.joint_indices(arm)
    if point_fn is None:
        def point_fn(q):
            return frames.ee_pose(bridge.to_urdf(q), arm)

    q0 = np.asarray(q_driver, dtype=np.float64)
    p0 = point_fn(q0)[:3, 3]
    J = np.zeros((3, len(idx)))
    for k, j in enumerate(idx):
        qp = q0.copy()
        qp[j] += eps
        qm = q0.copy()
        qm[j] -= eps
        J[:, k] = (point_fn(qp)[:3, 3] - point_fn(qm)[:3, 3]) / (2 * eps)
    return J


def analyse(J: np.ndarray) -> Dict[str, Any]:
    """Per-axis gains plus the manipulability ellipsoid."""
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    Jp = np.linalg.pinv(J)

    axes = {}
    for name, d in WORLD_AXES.items():
        cost = float(np.linalg.norm(Jp @ d))       # rad per metre
        axes[name] = {
            "joint_rad_per_metre": cost,
            "metre_per_joint_rad": (1.0 / cost) if cost > 1e-12 else float("inf"),
            # Joint motion needed for a 50 mm move -- the size this pipeline
            # actually commands.
            "joint_rad_for_50mm": cost * 0.05,
        }

    return {
        "singular_values": S.tolist(),
        "condition_number": float(S[0] / S[-1]) if S[-1] > 1e-12 else float("inf"),
        # Yoshikawa's measure: volume of the ellipsoid.
        "manipulability": float(np.prod(S)),
        "easiest_direction_world": U[:, 0].tolist(),
        "hardest_direction_world": U[:, -1].tolist(),
        "ellipsoid_axes_world": U.tolist(),
        "ellipsoid_radii": S.tolist(),
        "axes": axes,
    }


def report(a: Dict[str, Any], indent: str = "  ") -> str:
    lines = [
        f"{indent}singular values : "
        + ", ".join(f"{v:.4f}" for v in a["singular_values"]),
        f"{indent}condition number: {a['condition_number']:.2f}"
        "   (large = close to degenerate)",
        f"{indent}manipulability  : {a['manipulability']:.6f}"
        "   (ellipsoid volume; larger is better)",
        "",
        f"{indent}{'direction':10s} {'joint rad per 50 mm':>21s} "
        f"{'relative cost':>14s}",
        indent + "-" * 48,
    ]
    costs = {k: v["joint_rad_for_50mm"] for k, v in a["axes"].items()}
    best = min(costs.values())
    for name in ("+x", "-x", "+y", "-y", "+z", "-z"):
        c = costs[name]
        rel = c / best if best > 0 else float("inf")
        bar = "#" * min(40, int(round(rel * 4)))
        lines.append(f"{indent}{name:10s} {c:>21.4f} {rel:>13.2f}x  {bar}")
    lines += [
        indent + "-" * 48,
        f"{indent}easiest direction: "
        + np.array2string(np.asarray(a["easiest_direction_world"]),
                          precision=3, suppress_small=True),
        f"{indent}hardest direction: "
        + np.array2string(np.asarray(a["hardest_direction_world"]),
                          precision=3, suppress_small=True),
        "",
        f"{indent}Kinematics only -- gravity, servo stiffness, backlash and",
        f"{indent}link flex are NOT modelled here.",
    ]
    return "\n".join(lines)


def main() -> None:
    import argparse

    from kinematics import ARM_ORDER, JointFrameBridge, RobotFrames

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="right", choices=list(ARM_ORDER))
    ap.add_argument("--pose", default=None,
                    help="named pose from arm_config.POSES (default: the "
                         "URDF default configuration)")
    ap.add_argument("--tcp", action="store_true",
                    help="analyse the grasp point instead of the flange")
    ap.add_argument("--from-robot", action="store_true",
                    help="use the arm's measured joint state (needs ROS)")
    args = ap.parse_args()

    frames = RobotFrames()
    bridge = JointFrameBridge(frames.robot)

    if args.from_robot:
        from common import ensure_ros_path
        ensure_ros_path()
        import rospy
        from robot_control import create_and_configure_robots
        rospy.init_node("giava_manipulability", anonymous=True,
                        disable_signals=True)
        robots = create_and_configure_robots((args.arm,))
        rospy.sleep(0.5)
        n = ARM_CONFIG[args.arm]["num_joints"]
        q = np.zeros(frames.num_actuated)
        q[frames.joint_indices(args.arm)] = np.asarray(
            robots[args.arm].dxl.joint_states.position[:n], dtype=float)
        label = "measured joint state"
    elif args.pose:
        q = frames.q_from_named_pose(args.pose, arms=(args.arm,))
        label = f"named pose '{args.pose}'"
    else:
        q = bridge.to_driver(frames.home_q())
        label = "URDF default configuration"

    point_fn = None
    if args.tcp:
        import tcp as TCP

        def point_fn(qd):
            return TCP.tcp_pose(frames.ee_pose(bridge.to_urdf(qd), args.arm),
                                args.arm)

    J = translational_jacobian(frames, bridge, q, args.arm, point_fn)
    a = analyse(J)

    print()
    print("#" * 70)
    print(f"#  Directional manipulability  --  {args.arm}"
          f"{' (TCP)' if args.tcp else ' (flange)'}")
    print("#" * 70)
    print(f"\n  configuration: {label}\n")
    print(report(a))
    print()


if __name__ == "__main__":
    main()
