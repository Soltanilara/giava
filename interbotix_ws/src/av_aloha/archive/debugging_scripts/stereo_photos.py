'''
Stream both OAK cameras simultaneously, saving synchronized left/right pairs when 'c' is pressed.
Files are timestamp-matched as pair_XXX_TIMESTAMP.jpg with automatic save verification.
'''

#!/usr/bin/env python3

import os
import time
from pathlib import Path
import stat

import cv2
import depthai as dai
import numpy as np

LEFT_DIR = Path("stereo/calibration_photos_B")   # CAM_B
RIGHT_DIR = Path("stereo/calibration_photos_C")  # CAM_C


def verify_save(filepath, min_size_bytes=10000):
    # Check if file was actually written and has reasonable size
    if not filepath.exists():
        return False, "file does not exist"
    
    size = filepath.stat().st_size
    if size < min_size_bytes:
        return False, f"file too small ({size} bytes)"
    
    return True, f"OK ({size//1000} KB)"


def main():
    LEFT_DIR.mkdir(parents=True, exist_ok=True)
    RIGHT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== SAVING TO ===")
    print(f"Left:  {LEFT_DIR.resolve()}")
    print(f"Right: {RIGHT_DIR.resolve()}")
    print()

    pipeline = dai.Pipeline()

    cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    left_out = cam_left.requestOutput(
        size=(1280, 800),
        type=dai.ImgFrame.Type.BGR888p,
        fps=10,   # Slower = more stable saves
    )

    cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    right_out = cam_right.requestOutput(
        size=(1280, 800),
        type=dai.ImgFrame.Type.BGR888p,
        fps=10,
    )

    q_left = left_out.createOutputQueue()
    q_right = right_out.createOutputQueue()

    pipeline.start()

    capture_idx = 0
    latest_left = None
    latest_right = None

    print("✓ Dual stream ready. CLICK WINDOW → press 'c' to capture.")
    cv2.namedWindow("B(left) | C(right)", cv2.WINDOW_NORMAL)

    while pipeline.isRunning():
        # Get latest frames
        if q_left.has():
            latest_left = q_left.get().getCvFrame()
        if q_right.has():
            latest_right = q_right.get().getCvFrame()

        if latest_left is not None and latest_right is not None:
            combined = np.hstack([latest_left, latest_right])
            cv2.imshow("B(left) | C(right)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("c") and latest_left is not None and latest_right is not None:
            timestamp = int(time.time() * 1000)
            left_path = LEFT_DIR / f"pair_{capture_idx:03d}_{timestamp}.jpg"
            right_path = RIGHT_DIR / f"pair_{capture_idx:03d}_{timestamp}.jpg"

            # Save as JPG (more reliable)
            left_ok = cv2.imwrite(str(left_path), latest_left, [cv2.IMWRITE_JPEG_QUALITY, 95])
            right_ok = cv2.imwrite(str(right_path), latest_right, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Verify actual save
            left_status = verify_save(left_path)
            right_status = verify_save(right_path)

            if left_status[0] and right_status[0]:
                print(f"✓ pair_{capture_idx:03d}_{timestamp}  [{left_status[1]} | {right_status[1]}]")
                capture_idx += 1
            else:
                print(f"✗ pair_{capture_idx:03d}_{timestamp}")
                print(f"  left:  {left_status[1]}")
                print(f"  right: {right_status[1]}")
                # Delete failed files
                if left_path.exists():
                    left_path.unlink()
                if right_path.exists():
                    right_path.unlink()

    cv2.destroyAllWindows()
    print(f"\nFinal count: {capture_idx} verified pairs")


if __name__ == "__main__":
    main()