'''
Simple code to send green frames to headset
data types are uint8
'''

import numpy as np
import time
from interbotix_ws.src.av_aloha.data_collection_scripts.webrtc_headset import WebRTCHeadset

def main():
    # Start WebRTC
    headset = WebRTCHeadset()
    headset.run_in_thread()

    print("Waiting for WebRTC connection...")

    # Give enough time for offer/answer + track setup
    time.sleep(8)

    print("Starting GREEN frame stream...")

    # Create a constant GREEN frame
    green_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    green_frame[:, :, 1] = 255

    # data checks
    # print("Frame dtype:", green_frame.dtype)
    # print("Frame shape:", green_frame.shape)

    while True:
        start = time.time()

        # Send frames to headset
        headset.send_images(green_frame, green_frame)

        print("Sent GREEN frame")

        # ~30 FPS
        time.sleep(max(0, 1/30 - (time.time() - start)))


if __name__ == "__main__":
    main()
