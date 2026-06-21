import cv2
import numpy as np

# Create an image with a white dot at the right side (10, 5)
img = np.zeros((20, 20), dtype=np.uint8)
img[5, 15] = 255  # y=5, x=15

# Rotate it by 90 degrees
M = cv2.getRotationMatrix2D((10, 10), 90, 1.0)
out = cv2.warpAffine(img, M, (20, 20))

# Find the white dot
y, x = np.where(out == 255)
print(f"Original: x=15, y=5")
print(f"Rotated: x={x[0]}, y={y[0]}")
