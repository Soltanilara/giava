import depthai as dai
import time
import cv2
import os

save_dir = "oak_debug_frames"
os.makedirs(save_dir, exist_ok=True)

pipeline = dai.Pipeline()

cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

left_out = cam_left.requestOutput(
    (640, 480),
    type=dai.ImgFrame.Type.GRAY8,
    fps=90,
)

right_out = cam_right.requestOutput(
    (640, 480),
    type=dai.ImgFrame.Type.GRAY8,
    fps=90,
)

q_left = left_out.createOutputQueue()
q_right = right_out.createOutputQueue()

pipeline.start()
time.sleep(0.5)

left_img = q_left.get().getCvFrame()
right_img = q_right.get().getCvFrame()

left_path = os.path.join(save_dir, "left_raw.png")
right_path = os.path.join(save_dir, "right_raw.png")
side_by_side_path = os.path.join(save_dir, "stereo_pair.png")

cv2.imwrite(left_path, left_img)
cv2.imwrite(right_path, right_img)

stereo_pair = cv2.hconcat([left_img, right_img])
cv2.imwrite(side_by_side_path, stereo_pair)

print(f"Saved: {left_path}")
print(f"Saved: {right_path}")
print(f"Saved: {side_by_side_path}")

pipeline.stop()