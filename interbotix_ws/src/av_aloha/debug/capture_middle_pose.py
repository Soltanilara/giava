"""Hand-guide the middle arm and capture joint poses for arm_config.py.

Workflow (driver limits are temporarily widened to ±10 on the waist so
nothing blocks you):

    python capture_middle_pose.py

    f + Enter   torque OFF the whole middle arm  ** HOLD THE ARM FIRST —
                shoulder/elbow will drop under gravity! **
    n + Enter   torque ON (arm holds where you put it)
    p + Enter   print the current 7-joint pose as a ready-to-paste
                arm_config.py line
    <name> + Enter  same as p, but labeled (e.g. 'forward' prints
                M_FORWARD = np.array([...]))
    x + Enter   exit (torques back ON first)

Move the arm by hand to each pose you want (forward / rest / high / low /
far_scene / looking_left / looking_right), torque on, then print it.
Paste the printed lines into arm_config.py, replacing the old M_* arrays —
they will be in the CURRENT driver frame, which is all that matters now:
the IK bridge and the startup interpolation are frame-aware as of today.

Afterwards, tell Claude the waist values you measured so the driver limits
can be re-tightened around the real operating range (the ±10 is temporary).
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import sys
import time
from pathlib import Path

import numpy as np

from paths import ROS_DEVEL_SITE

for ros_path in (
    Path("/opt/ros/noetic/lib/python3/dist-packages"),
    ROS_DEVEL_SITE,
):
    if ros_path.is_dir() and str(ros_path) not in sys.path:
        sys.path.append(str(ros_path))

import rospy  # noqa: E402
from interbotix_xs_modules.core import InterbotixRobotXSCore  # noqa: E402


def main() -> None:
    rospy.init_node("capture_middle_pose")
    dxl = InterbotixRobotXSCore(
        robot_model="wx250s",
        robot_name="puppet_middle",
        init_node=False,
    )
    time.sleep(1.0)

    names = list(dxl.robot_get_joint_states().name)
    print(f"joints: {names}")
    print(__doc__.split("Workflow")[1])

    torqued = True
    while not rospy.is_shutdown():
        try:
            cmd = input("[f=off n=on p=print <name>=print-labeled x=exit] > ").strip()
        except EOFError:
            break
        if cmd == "f":
            print("*** HOLD THE ARM — torquing off in 2 s ***")
            time.sleep(2.0)
            dxl.robot_torque_enable("group", "arm", False)
            torqued = False
            print("torque OFF — move the arm by hand")
        elif cmd == "n":
            dxl.robot_torque_enable("group", "arm", True)
            torqued = True
            print("torque ON — arm holds position")
        elif cmd == "x":
            if not torqued:
                dxl.robot_torque_enable("group", "arm", True)
                print("(torqued back on)")
            break
        elif cmd:
            label = "POSE" if cmd == "p" else cmd.upper()
            js = dxl.robot_get_joint_states()
            q = [float(v) for v in js.position[:7]]
            arr = ", ".join(f"{v!r}" for v in q)
            print(f"\nM_{label} = np.array([{arr}], dtype=float)\n")
            for n, v in zip(names, q):
                print(f"    {n:14s} {v:+.4f}")
            print()


if __name__ == "__main__":
    main()
