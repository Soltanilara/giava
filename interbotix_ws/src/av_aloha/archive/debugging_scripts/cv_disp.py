# Simple script to test if OpenCV is working properly by creating a blank image and displaying it in a window.

import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()