from data.loaders import att_loader
from core.preprocessing import detector
import cv2

images, ids = att_loader.load(split="train")
total = 0
eye_fails = 0
for img in images:
    face_box = detector.detect_face(img)
    if face_box is not None:
        total += 1
        x, y, w, h = face_box
        crop = img[y:y+h, x:x+w]
        eyes = detector.detect_eyes(crop)
        if eyes is None:
            eye_fails += 1

print(f"Total faces detected: {total}")
print(f"Eye detection failed: {eye_fails}")
