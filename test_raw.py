from core.matching import distances
from data.loaders import att_loader
from core.preprocessing import normalizer
from core.decomposition.eigenfaces import Eigenfaces
import numpy as np

images, ids = att_loader.load(split="train")

# Train Eigenfaces directly on raw images (scaled to 100x100)
import cv2
def raw_prep(img):
    img = cv2.resize(img, (100, 100))
    img = normalizer.normalize(img)
    return normalizer.flatten(img)

vecs = [raw_prep(img) for img in images]
ef = Eigenfaces(n_components=50).fit(vecs)

# Same person
s01_idx = [i for i, sid in enumerate(ids) if sid == "s01"]
a = ef.transform(vecs[s01_idx[0]])
b = ef.transform(vecs[s01_idx[1]])
dist_same = distances.cosine(a, b)

# Diff person
s02_idx = [i for i, sid in enumerate(ids) if sid == "s02"]
c = ef.transform(vecs[s02_idx[0]])
dist_diff = distances.cosine(a, c)

print(f"Raw Same distance: {dist_same:.4f}")
print(f"Raw Diff distance: {dist_diff:.4f}")

