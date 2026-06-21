import numpy as np
from core.matching import distances, threshold
from data.loaders import att_loader, yale_loader
from scripts.build_eigenspace import preprocess_image_haar
from core.decomposition.eigenfaces import Eigenfaces

att_tr, _ = att_loader.load(split="train")
yale_tr, _, _ = yale_loader.load(split="train")

train_vecs = []
for img in att_tr + yale_tr:
    v = preprocess_image_haar(img)
    if v is not None:
        train_vecs.append(v)

ef = Eigenfaces(n_components=50)
ef.pca.whiten = True
ef.fit(train_vecs)

att_te, att_id = att_loader.load(split="test")
yale_te, yale_id, _ = yale_loader.load(split="test")

test_vecs = []
test_ids = []
for img, sid in zip(att_te + yale_te, att_id + yale_id):
    v = preprocess_image_haar(img)
    if v is not None:
        test_vecs.append(v)
        test_ids.append(sid)

test_coeffs = ef.transform(test_vecs)

gen = []
imp = []
for i in range(len(test_coeffs)):
    for j in range(i+1, len(test_coeffs)):
        d = distances.cosine(test_coeffs[i], test_coeffs[j])
        if test_ids[i] == test_ids[j]:
            gen.append(d)
        else:
            imp.append(d)

eer, thresh = threshold.compute_eer(np.array(gen), np.array(imp))
print(f"Test EER: {eer:.4f}")
print(f"Genuine Mean: {np.mean(gen):.4f}")
print(f"Impostor Mean: {np.mean(imp):.4f}")
print(f"Threshold: {thresh:.4f}")
