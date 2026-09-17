# Stream video from your OAK camera to the headset and overlay controller pose (relative to headset frame) as text.

import depthai as dai
import time
import cv2
import numpy as np
from interbotix_ws.src.av_aloha.data_collection_scripts.webrtc_headset import WebRTCHeadset
from headset_control import HeadsetFullControl as HeadsetControl
from headset_utils import HeadsetFeedback
from transform_utils import (pose2mat)

# ---- Setup headset ----
headset = WebRTCHeadset()
headset.run_in_thread()
headset_control = HeadsetControl()
feedback = HeadsetFeedback()
headset_control.reset()

# ---- Setup pipeline ----
pipeline = dai.Pipeline()

cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

# Request data from the camera with a certain file type and fps
left_out = cam_left.requestOutput(
    (640, 480),
    type=dai.ImgFrame.Type.BGR888i,
    fps=25,
)

right_out = cam_right.requestOutput(
    (640, 480),
    type=dai.ImgFrame.Type.BGR888i,
    fps=25,
)

# Create a queue for left and right to store camera data
q_left = left_out.createOutputQueue()
q_right = right_out.createOutputQueue()

pipeline.start()

time.sleep(0.5)

# ---- Main loop ----
while pipeline.isRunning():
    # get an image from left and right queues and send to headset
    left_img = q_left.get().getCvFrame()
    right_img = q_right.get().getCvFrame()

    headset.send_images(left_img, right_img)

    # receive data from headset (important later when we overlay controller pose information)
    headset_data = headset.receive_data()

    if headset_data is None:
        continue 

    """ # --- Extract positions ---
    h_pos = headset_data.h_pos
    l_pos = headset_data.l_pos
    r_pos = headset_data.r_pos

    # --- Compute naive relative ---
    l_rel = l_pos - h_pos
    r_rel = r_pos - h_pos

    # --- Build string ---
    feedback.info = (
        f"HEAD\n"
        f"x: {h_pos[0]:.3f} y: {h_pos[1]:.3f} z: {h_pos[2]:.3f}\n\n"
        f"LEFT\n"
        f"x: {l_pos[0]:.3f} y: {l_pos[1]:.3f} z: {l_pos[2]:.3f}\n\n"
        f"RIGHT\n"
        f"x: {r_pos[0]:.3f} y: {r_pos[1]:.3f} z: {r_pos[2]:.3f}\n\n"
        f"REL\n"
        f"L-H: ({l_rel[0]:.3f}, {l_rel[1]:.3f}, {l_rel[2]:.3f})\n"
        f"R-H: ({r_rel[0]:.3f}, {r_rel[1]:.3f}, {r_rel[2]:.3f})"
    )

    headset.send_feedback(feedback) """
