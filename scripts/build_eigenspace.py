"""CLI script: load datasets -> preprocess -> fit Eigenfaces -> calibrate
threshold -> save artifacts to artifacts/.

Usage:
    python scripts/build_eigenspace.py
"""

import json
import os
import sys

# Make the project root importable when run as `python scripts/...py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.decomposition.eigenfaces import Eigenfaces
from core.matching import distances, threshold
from core.pipeline import _preprocess_single
from core.preprocessing import aligner, detector, normalizer
from data.loaders import att_loader, lfw_loader, yale_loader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_SIZE = (100, 100)  # (width, height)
N_COMPONENTS = 150
ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)




# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """Run the full eigenspace build + calibration pipeline.

    Steps:
        1. Load AT&T and Yale training data.
        2. Preprocess all training images (detect -> align -> normalize).
        3. Fit Eigenfaces PCA on the training vectors.
        4. Load FG-NET with landmarks for cross-age calibration.
        5. Build genuine/impostor pairs from FG-NET.
        6. Compute distance scores per pair, per metric.
        7. Normalize scores and compute ROC/EER.
        8. Save artifacts.
    """
    print("=" * 60)
    print("Building eigenspace + calibrating thresholds")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load training data
    # ------------------------------------------------------------------
    print("\n[1/8] Loading AT&T data (for Training + Calibration)...")
    att_images, att_ids = att_loader.load(split="both")
    print(f"  AT&T: {len(att_images)} images, {len(set(att_ids))} subjects")

    print("[1/8] Loading LFW data (for Training only)...")
    lfw_images, lfw_ids = lfw_loader.load(min_faces=2)
    print(f"  LFW: {len(lfw_images)} images, {len(set(lfw_ids))} subjects")

    print("[1/8] Loading Yale data (for Training only)...")
    try:
        yale_images, yale_ids, _ = yale_loader.load(split="both")
        print(f"  Yale: {len(yale_images)} images, {len(set(yale_ids))} subjects")
    except FileNotFoundError:
        print("  Yale: Dataset not found in data/raw/yale_faces/. Skipping Yale.")
        yale_images, yale_ids = [], []

    # ------------------------------------------------------------------
    # Step 2: Preprocess training images
    # ------------------------------------------------------------------
    print("\n[2/8] Preprocessing training images...")
    train_vectors = []
    calib_vectors = []
    calib_ids = []
    skipped = 0

    for i, (img, sid) in enumerate(zip(att_images, att_ids)):
        vec = _preprocess_single(img, IMAGE_SIZE)
        if vec is not None:
            train_vectors.append(vec)
            calib_vectors.append(vec)
            calib_ids.append(f"att_{sid}")
        else:
            skipped += 1
        if (i + 1) % 100 == 0:
            print(f"  AT&T: {i + 1}/{len(att_images)} processed")

    for i, (img, sid) in enumerate(zip(lfw_images, lfw_ids)):
        vec = _preprocess_single(img, IMAGE_SIZE)
        if vec is not None:
            train_vectors.append(vec)
        else:
            skipped += 1
        if (i + 1) % 500 == 0:
            print(f"  LFW: {i + 1}/{len(lfw_images)} processed")

    for i, (img, sid) in enumerate(zip(yale_images, yale_ids)):
        vec = _preprocess_single(img, IMAGE_SIZE)
        if vec is not None:
            train_vectors.append(vec)
        else:
            skipped += 1
        if (i + 1) % 500 == 0:
            print(f"  Yale: {i + 1}/{len(yale_images)} processed")

    train_matrix = np.array(train_vectors)
    print(
        f"  Total: {len(train_vectors)} vectors "
        f"({skipped} skipped due to detection failure)"
    )
    print(f"  Matrix shape: {train_matrix.shape}")



    print(f"  Matrix shape: {train_matrix.shape}")

    # ------------------------------------------------------------------
    # Step 3: Fit Eigenfaces
    # ------------------------------------------------------------------
    print(f"\n[3/8] Fitting Eigenfaces (n_components={N_COMPONENTS})...")
    eigenfaces = Eigenfaces(n_components=N_COMPONENTS)
    eigenfaces.fit(train_matrix)
    total_variance = eigenfaces.explained_variance_ratio.sum()
    print(f"  Explained variance: {total_variance:.4f} ({total_variance*100:.1f}%)")

    # Transform calibration vectors (ONLY AT&T) into eigenspace
    calib_matrix = np.array(calib_vectors)
    calib_coeffs = eigenfaces.transform(calib_matrix)

    # Group by subject
    subject_indices: dict[str, list[int]] = {}
    for idx, sid in enumerate(calib_ids):
        subject_indices.setdefault(sid, []).append(idx)

    genuine_distances = {"cosine": [], "euclidean": []}
    impostor_distances = {"cosine": [], "euclidean": []}

    subjects = sorted(list(subject_indices.keys()))

    # Genuine pairs: same subject, different images
    for sid in subjects:
        indices = subject_indices[sid]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = calib_coeffs[indices[i]], calib_coeffs[indices[j]]
                genuine_distances["cosine"].append(distances.cosine(a, b))
                genuine_distances["euclidean"].append(distances.euclidean(a, b))

    # Impostor pairs: different subjects (sample to keep manageable)
    rng = np.random.RandomState(42)
    n_impostor_target = len(genuine_distances["cosine"])
    impostor_count = 0
    max_attempts = n_impostor_target * 10

    for _ in range(max_attempts):
        if impostor_count >= n_impostor_target:
            break
        s1, s2 = rng.choice(subjects, size=2, replace=False)
        i1 = rng.choice(subject_indices[s1])
        i2 = rng.choice(subject_indices[s2])
        a, b = calib_coeffs[i1], calib_coeffs[i2]
        impostor_distances["cosine"].append(distances.cosine(a, b))
        impostor_distances["euclidean"].append(distances.euclidean(a, b))
        impostor_count += 1

    print(
        f"  Genuine pairs: {len(genuine_distances['cosine'])}, "
        f"Impostor pairs: {len(impostor_distances['cosine'])}"
    )

    # ------------------------------------------------------------------
    # Step 5: Compute calibration statistics
    # ------------------------------------------------------------------
    print("\n[5/8] Computing calibration statistics...")

    calibration = {"image_size": list(IMAGE_SIZE), "n_components": N_COMPONENTS}

    for metric in ["cosine", "euclidean"]:
        gen = np.array(genuine_distances[metric])
        imp = np.array(impostor_distances[metric])

        mean_all = float(np.concatenate([gen, imp]).mean())
        std_all = float(np.concatenate([gen, imp]).std())

        # Normalize all scores to confidence values
        gen_conf = np.array([
            distances.normalize_score(d, mean_all, std_all) for d in gen
        ])
        imp_conf = np.array([
            distances.normalize_score(d, mean_all, std_all) for d in imp
        ])

        # ROC and EER
        fpr, tpr, thresholds = threshold.compute_roc(gen_conf, imp_conf)
        eer_value, eer_threshold = threshold.compute_eer(gen_conf, imp_conf)

        calibration[metric] = {
            "mean": mean_all,
            "std": std_all,
            "genuine_mean": float(gen.mean()),
            "genuine_std": float(gen.std()),
            "impostor_mean": float(imp.mean()),
            "impostor_std": float(imp.std()),
            "eer": float(eer_value),
            "eer_threshold": float(eer_threshold),
            "roc_fpr": fpr.tolist(),
            "roc_tpr": tpr.tolist(),
            "roc_thresholds": thresholds.tolist(),
        }

        calibration[f"{metric}_threshold"] = float(eer_threshold)

        print(
            f"  {metric}: EER={eer_value:.4f}, "
            f"threshold={eer_threshold:.4f}, "
            f"genuine_mean={gen.mean():.4f}, "
            f"impostor_mean={imp.mean():.4f}"
        )

    # ------------------------------------------------------------------
    # Step 6: Save artifacts
    # ------------------------------------------------------------------
    print(f"\n[6/8] Saving artifacts to {ARTIFACTS_DIR}/...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Eigenspace data
    np.savez_compressed(
        os.path.join(ARTIFACTS_DIR, "eigenspace.npz"),
        mean_face=eigenfaces.mean_face,
        components=eigenfaces.components,
        explained_variance=eigenfaces.explained_variance,
        explained_variance_ratio=eigenfaces.explained_variance_ratio,
        image_size=np.array(IMAGE_SIZE),
    )
    print("  Saved: eigenspace.npz")

    # Calibration data
    with open(os.path.join(ARTIFACTS_DIR, "calibration.json"), "w") as f:
        json.dump(calibration, f, indent=2)
    print("  Saved: calibration.json")

    print("\n" + "=" * 60)
    print("Done! Eigenspace built and thresholds calibrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
