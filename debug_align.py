import cv2
import numpy as np
from data.loaders import att_loader, fgnet_loader
from scripts.build_eigenspace import preprocess_image_haar, preprocess_image_landmarks

att_images, _ = att_loader.load(split="train")
att_vec = preprocess_image_haar(att_images[0])
cv2.imwrite("debug_att.jpg", att_vec.reshape(100, 100) * 255.0)

fgnet_images, fgnet_ids, _ = fgnet_loader.load()
fgnet_landmarks = fgnet_loader.load_landmarks()

# Find one that has landmarks
for i, img in enumerate(fgnet_images):
    # Just need the filename base
    import os
    from data.loaders.fgnet_loader import FILENAME_PATTERN
    images_dir = "data/raw/fgnet/images"
    fn = sorted(os.listdir(images_dir))[i]
    stem = os.path.splitext(fn)[0].lower()
    if stem in fgnet_landmarks:
        fg_vec = preprocess_image_landmarks(img, fgnet_landmarks[stem])
        cv2.imwrite("debug_fgnet.jpg", fg_vec.reshape(100, 100) * 255.0)
        break
