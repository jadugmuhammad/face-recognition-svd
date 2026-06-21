from data.loaders import att_loader
from core.preprocessing import normalizer
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import numpy as np
import cv2

images, ids = att_loader.load(split="train")
def raw_prep(img):
    img = cv2.resize(img, (100, 100))
    img = normalizer.normalize(img)
    return normalizer.flatten(img)

vecs = np.array([raw_prep(img) for img in images])

# Unwhitened
pca = PCA(n_components=50, whiten=False).fit(vecs)
coeffs = pca.transform(vecs)

s01 = [i for i, sid in enumerate(ids) if sid == "s01"]
s02 = [i for i, sid in enumerate(ids) if sid == "s02"]

print("Unwhitened:")
print(f" Same: {cosine(coeffs[s01[0]], coeffs[s01[1]]):.4f}")
print(f" Diff: {cosine(coeffs[s01[0]], coeffs[s02[0]]):.4f}")

# Whitened
pca_w = PCA(n_components=50, whiten=True).fit(vecs)
coeffs_w = pca_w.transform(vecs)

print("\nWhitened:")
print(f" Same: {cosine(coeffs_w[s01[0]], coeffs_w[s01[1]]):.4f}")
print(f" Diff: {cosine(coeffs_w[s01[0]], coeffs_w[s02[0]]):.4f}")

