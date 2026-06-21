import cv2
import numpy as np

img = np.zeros((100, 100), dtype=np.uint8)
# Left eye
ly, lx = 30, 20
# Right eye is lower
ry, rx = 50, 80

img[ly, lx] = 255
img[ry, rx] = 255

from core.preprocessing.aligner import align_face
out = align_face(img, [(lx, ly), (rx, ry)], output_size=(100, 100))

# Find the eyes in the output
y, x = np.where(out == 255)
# Sort by x
pts = sorted(zip(x, y))
print("Input Left:", (lx, ly))
print("Input Right:", (rx, ry))
print("Output points:", pts)
