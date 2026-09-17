import time
import numpy as np

from real_env import make_real_env
from interbotix_ws.src.av_aloha.data_collection_scripts.webrtc_headset import WebRTCHeadset
from headset_control import HeadsetFullControl
from headset_utils import HeadsetFeedback
from constants import REAL_DT


def main():
    print("=== Initializing Teleoperation System ===")

    # --- ENV (robots + cameras already handled inside) ---
    env = make_real_env(init_node=True)
    ts = env.reset()

    # --- HEADSET ---
    headset = WebRTCHeadset()
    headset.run_in_thread()

    # --- CONTROL ---
    controller = HeadsetFullControl()
    controller.reset()

    action = np.zeros(23)

    print("Waiting for user alignment...")

    # =========================
    # WAIT FOR START
    # =========================
    while True:
        start_time = time.time()

        ts = env.get_obs()
        headset_data = headset.receive_data()

        if headset_data is not None:
            action, feedback = controller.run(
                headset_data,
                ts["poses"]["left"],
                ts["poses"]["right"],
                ts["poses"]["middle"],
            )

            if (
                headset_data.r_button_one
                and not feedback.head_out_of_sync
                and not feedback.left_out_of_sync
                and not feedback.right_out_of_sync
            ):
                controller.start(headset_data, ts["poses"]["middle"])
                print("Teleoperation started.")
                break

        feedback = HeadsetFeedback()
        feedback.info = "Align headset + hold A to start"
        headset.send_feedback(feedback)

        send_images(headset, ts)

        time.sleep(max(0, REAL_DT - (time.time() - start_time)))

    # =========================
    # MAIN LOOP
    # =========================
    while True:
        step_start = time.time()

        ts = env.get_obs()
        headset_data = headset.receive_data()

        if headset_data is not None:
            action, feedback = controller.run(
                headset_data,
                ts["poses"]["left"],
                ts["poses"]["right"],
                ts["poses"]["middle"],
            )

            # stop condition
            if not headset_data.r_button_one:
                print("Stopping teleoperation.")
                break

        # --- APPLY ACTION ---
        ts, _, _, _, info = env.step(action)

        # --- SEND FEEDBACK ---
        feedback.info = info
        headset.send_feedback(feedback)

        # --- STREAM CAMERAS ---
        send_images(headset, ts)

        time.sleep(max(0, REAL_DT - (time.time() - step_start)))


def send_images(headset, ts):
    try:
        images = ts["images"]

        # Primary stereo view
        left = images["cam_left_wrist"]
        right = images["cam_right_wrist"]

        # OPTIONAL: overlay OAK (middle camera)
        if "oak" in images:
            import cv2

            oak = images["oak"]
            h, w, _ = left.shape
            oak_small = cv2.resize(oak, (w // 3, h // 3))

            left[-oak_small.shape[0]:, -oak_small.shape[1]:] = oak_small
            right[-oak_small.shape[0]:, -oak_small.shape[1]:] = oak_small

        headset.send_images(left, right)

    except Exception as e:
        print(f"[WARN] Image streaming failed: {e}")


if __name__ == "__main__":
    main()