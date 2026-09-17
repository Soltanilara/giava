"""Live OAK view + middle-arm joint readout, for finding a camera pose by hand.

    python oak_pose_finder.py                 # stream + readout, torque stays ON
    python oak_pose_finder.py --no-ros        # stream only, no arm connection

Point the camera arm where you want it, watch what the camera actually sees on
this screen, and press `s` to capture the pose.  The captured joint vector is
printed ready to paste and appended to poses_custom.json under "middle".

WHY THIS EXISTS.  A pose is only as good as the view it gives, and the view
lives inside the headset -- so choosing one by echoing joint states means
guessing.  This puts the camera's own image on the monitor beside the numbers.

THE OAK IS AN EXCLUSIVE DEVICE.  data_collection.py opens it for the headset,
so that must NOT be running while this is.  Same for any other OAK script.

HAND-GUIDING.  `t` releases torque so the arm can be moved by hand.  The
middle arm carries the cameras and WILL fall under gravity when released:
    *** TAKE ITS WEIGHT BEFORE PRESSING t, AND KEEP HOLDING IT. ***
`t` again re-enables torque, which holds wherever it currently is.  Torque is
re-enabled automatically on exit.
"""
from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

CUSTOM_POSES = HERE / "poses_custom.json"
ARM = "middle"


def open_oak(width=640, height=480, fps=30):
    """(device, queue) for the left colour stream.

    Deliberately minimal and independent of camera_manager: this tool runs
    alone, and mirroring the recorder's rectification/zoom would only make the
    preview differ from what it is meant to show."""
    import depthai as dai

    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        raise SystemExit(
            "No OAK device found. Check the USB connection, and make sure "
            "nothing else has it open (data_collection.py holds it for the "
            "headset -- quit that first).")
    device = dai.Device()
    pipeline = dai.Pipeline(device)
    ## CAM_C is what camera_manager binds to 'oak_left' (see the note by
    ## oak_camera_pipeline) -- match it so this preview is the same eye the
    ## dataset records.
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    out = cam.requestOutput((width, height), type=dai.ImgFrame.Type.BGR888i,
                            fps=fps)
    q = out.createOutputQueue(maxSize=2, blocking=False)
    pipeline.start()
    return device, pipeline, q


def load_custom():
    if CUSTOM_POSES.exists():
        try:
            return json.loads(CUSTOM_POSES.read_text())
        except Exception as exc:
            print(f"[poses] {CUSTOM_POSES} unreadable ({exc}); starting fresh")
    return {}


def save_custom(name, q):
    data = load_custom()
    data.setdefault(ARM, {})[name] = [float(x) for x in q]
    CUSTOM_POSES.write_text(json.dumps(data, indent=2))
    print(f"[poses] saved '{name}' to {CUSTOM_POSES}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-ros", action="store_true",
                    help="stream only; do not connect to the arm")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--scale", type=float, default=1.5,
                    help="preview window scale")
    args = ap.parse_args()

    bot = joint_names = None
    if not args.no_ros:
        import rospy
        from arm_config import ARM_CONFIG
        from robot_control import create_and_configure_robots
        rospy.init_node("oak_pose_finder", anonymous=True)
        bot = create_and_configure_robots([ARM])[ARM]
        joint_names = ARM_CONFIG[ARM]["joint_names"]
        print(f"[arm] {ARM} connected ({len(joint_names)} joints)")

    device, pipeline, q = open_oak(args.width, args.height)
    print("[oak] streaming CAM_C (the eye recorded as oak_left)")
    print("\nkeys:  s save pose   t toggle torque (SUPPORT THE ARM FIRST)   "
          "p print pose   q quit\n")

    torque_on = True
    win = "oak_pose_finder"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(args.width * args.scale),
                     int(args.height * args.scale))

    def read_q():
        if bot is None:
            return None
        return np.asarray(bot.dxl.joint_states.position[:len(joint_names)],
                          dtype=float)

    def print_pose(cur):
        print("\n  paste into arm_config.py:")
        print("  M_<NAME> = np.array([" +
              ", ".join(f"{v:.6f}" for v in cur) + "], dtype=float)")
        print("  per joint: " + "  ".join(
            f"{n.replace('middle_', '')}={v:+.3f}"
            for n, v in zip(joint_names, cur)))

    try:
        while True:
            pkt = q.tryGet()
            if pkt is not None:
                img = pkt.getCvFrame()
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cur = read_q()
                lines = []
                if cur is not None:
                    lines.append("  ".join(
                        f"{n.replace('middle_','')[:6]}:{v:+.2f}"
                        for n, v in zip(joint_names, cur)))
                if bot is None:
                    lines.append("stream only (--no-ros): no arm connection")
                else:
                    lines.append(
                        "TORQUE ON  (t = release, hold the arm first)"
                        if torque_on else
                        "TORQUE OFF -- ARM IS FREE, SUPPORT IT")
                for k, text in enumerate(lines):
                    y = 22 + 24 * k
                    color = ((0, 0, 255) if (not torque_on and k == len(lines) - 1)
                             else (255, 255, 255))
                    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 1, cv2.LINE_AA)
                cv2.imshow(win, img)

            k = cv2.waitKey(20) & 0xFF
            if k == ord("q"):
                break
            if k == ord("t") and bot is not None:
                torque_on = not torque_on
                bot.dxl.robot_torque_enable("group", "arm", torque_on)
                print(f"[arm] torque {'ON' if torque_on else 'OFF'}"
                      + ("" if torque_on else "  <-- HOLD THE ARM"))
            if k in (ord("s"), ord("p")) and bot is not None:
                cur = read_q()
                print_pose(cur)
                if k == ord("s"):
                    name = input("  name for this pose (ENTER to skip): ").strip()
                    if name:
                        save_custom(name, cur)
    finally:
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            device.close()
        except Exception:
            pass
        if bot is not None and not torque_on:
            ## Never leave the arm limp: it is holding cameras over a table.
            bot.dxl.robot_torque_enable("group", "arm", True)
            print("[arm] torque re-enabled on exit")


if __name__ == "__main__":
    main()
