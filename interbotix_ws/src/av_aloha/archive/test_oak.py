import depthai as dai
import cv2
import os
import time

save_dir = "oak_v3_debug"
os.makedirs(save_dir, exist_ok=True)

pipeline = dai.Pipeline()

cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

left_gray = cam_left.requestOutput(
    (640, 480),
    type=dai.ImgFrame.Type.GRAY8,
    fps=30,
)

right_gray = cam_right.requestOutput(
    (640, 480),
    type=dai.ImgFrame.Type.GRAY8,
    fps=30,
)

left_bgr = cam_left.requestOutput(
    (640, 480),
    type=dai.ImgFrame.Type.BGR888p,
    fps=30,
)

right_bgr = cam_right.requestOutput(
    (640, 480),
    type=dai.ImgFrame.Type.BGR888p,
    fps=30,
)

q_left_gray = left_gray.createOutputQueue()
q_right_gray = right_gray.createOutputQueue()
q_left_bgr = left_bgr.createOutputQueue()
q_right_bgr = right_bgr.createOutputQueue()

pipeline.start()
time.sleep(0.5)

left_gray_img = q_left_gray.get().getCvFrame()
right_gray_img = q_right_gray.get().getCvFrame()
left_bgr_img = q_left_bgr.get().getCvFrame()
right_bgr_img = q_right_bgr.get().getCvFrame()

cv2.imwrite(f"{save_dir}/left_gray.png", left_gray_img)
cv2.imwrite(f"{save_dir}/right_gray.png", right_gray_img)
cv2.imwrite(f"{save_dir}/left_bgr.png", left_bgr_img)
cv2.imwrite(f"{save_dir}/right_bgr.png", right_bgr_img)

cv2.imwrite(f"{save_dir}/gray_pair.png", cv2.hconcat([left_gray_img, right_gray_img]))
cv2.imwrite(f"{save_dir}/bgr_pair.png", cv2.hconcat([left_bgr_img, right_bgr_img]))

print("Saved debug images:")
print(" left_gray.png ", left_gray_img.shape, left_gray_img.dtype)
print(" right_gray.png", right_gray_img.shape, right_gray_img.dtype)
print(" left_bgr.png  ", left_bgr_img.shape, left_bgr_img.dtype)
print(" right_bgr.png ", right_bgr_img.shape, right_bgr_img.dtype)