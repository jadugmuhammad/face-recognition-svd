import cv2
import numpy as np

# Create an image
img = np.zeros((20, 20), dtype=np.uint8)

# Left eye at (5, 5)
img[5, 5] = 255
# Right eye at (15, 15)
img[15, 15] = 128

left_eye = np.array([5, 5], dtype=np.float64)
right_eye = np.array([15, 15], dtype=np.float64)

delta = right_eye - left_eye
angle = np.degrees(np.arctan2(delta[1], delta[0]))
print(f"delta: {delta}, angle: {angle}")

M = cv2.getRotationMatrix2D((10, 10), angle, 1.0)
out = cv2.warpAffine(img, M, (20, 20))

ly, lx = np.where(out == 255)
ry, rx = np.where(out == 128)
print(f"Left eye rotated: x={lx[0]}, y={ly[0]}")
print(f"Right eye rotated: x={rx[0]}, y={ry[0]}")
