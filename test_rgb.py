import cv2
import numpy as np
img = np.zeros((100, 100, 3), dtype=np.uint8)
try:
    cv2.equalizeHist(img)
except Exception as e:
    print("Error:", e)
