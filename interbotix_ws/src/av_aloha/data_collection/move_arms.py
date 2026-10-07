import numpy as np
import rospy

import argparse
import sys

from arm_config import ARM_CONFIG
from robot_control import (
    create_and_configure_robots,
    move_arms_together,
    get_pose,
)

from data_col_config import ARM_MODES


def parse_args():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()

    parser.add_argument(
        "--mode",
        choices=["left", "right", "middle", "bimanual", "all", "av"],
        default="right",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="skip the live-session check and move anyway",
    )

    group.add_argument(
        "--high",
        action="store_true",
        help="Move to high reset pose",
    )

    group.add_argument(
        "--forward",
        action="store_true",
        help="Move to forward-facing reset pose",
    )

    group.add_argument(
        "--rest",
        action="store_true",
        help="Move to rest pose",
    )

    group.add_argument(
        "--low",
        action="store_true",
        help="Move to lowered pose",
    )

    return parser.parse_args()

def _abort_if_actively_driven(arm_names, window_s=0.5):
    """Refuse to move an arm a live control loop is streaming commands to.

    teleop / data_collection publish JointGroupCommand at ~50 Hz while a
    drive session is ACTIVE, and publish NOTHING while idle (the arms just
    hold on servo torque) -- so half a second of listening cleanly separates
    the two.  Moving an arm out from under an active stream makes two nodes
    fight at 50 Hz: the arm judders between targets and snaps to whichever
    publisher wins the last tick.  Resetting an arm mid-session is never
    that; make the operator release the drive button first.

    An IDLE session is fine to move under: its next enable runs the
    stale-command check and re-anchors from measured joints (see
    command_state_is_stale / sync_robot_state).
    """
    from interbotix_xs_msgs.msg import JointGroupCommand

    live = set()
    subs = [
        rospy.Subscriber(
            f"/{ARM_CONFIG[a]['robot_name']}/commands/joint_group",
            JointGroupCommand,
            lambda _msg, a=a: live.add(a),
        )
        for a in arm_names
    ]
    rospy.sleep(window_s)
    for s in subs:
        s.unregister()
    if live:
        sys.exit(
            f"[move_arms] REFUSING to move {', '.join(sorted(live))}: a control "
            f"loop is actively streaming commands to it (seen on its "
            f"joint_group topic within {window_s:.1f}s).\n"
            "Release the drive button so the arms are idle -- or stop the "
            "session -- then re-run.  --force overrides."
        )


def main():
    rospy.init_node("move_arm", anonymous=True)

    # print("rospy node initialized")

    args = parse_args()

    # print(f"Read args: {args}")

    pose_name = "forward"

    if args.high:
        pose_name = "high"
    elif args.rest:
        pose_name = "rest"
    elif args.low:
        pose_name = "low"

    # print(f"Read pose name: {pose_name}")

    arm_names = ARM_MODES[args.mode]

    if not args.force:
        _abort_if_actively_driven(arm_names)

    print(f"Creating and configuring the arms.")

    robots = create_and_configure_robots(arm_names)

    # for arm_name, bot in robots.items():
    #     print(f"Joint names for {arm_name}: {bot.arm.group_info.joint_names}")

    # print(f"Created and configured robots for arms: {arm_names}")

    rospy.sleep(1)

    # print(f"Moving {args.mode} arms to pose '{pose_name}'")

    ## One shared trapezoid for every arm, and it BLOCKS until the arms have
    ## measurably settled.  The old per-arm loop ran every arm but the last
    ## on a daemon thread (interpolate_to_pose(blocking=False)) and blocked
    ## only on the last one -- so when the last arm (often the SHORTEST
    ## move) settled, main() returned, the process exited, and the daemon
    ## streams died mid-flight.  Each surviving goal register held a
    ## mid-trajectory setpoint, which is exactly the observed "move_arms
    ## stops before completing the motion" on multi-arm modes.
    move_arms_together(
        robots, {a: get_pose(a, pose_name) for a in arm_names})

if __name__ == "__main__":
    main()