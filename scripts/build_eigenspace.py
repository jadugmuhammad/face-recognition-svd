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
from core.matching import distances, ensemble, threshold
from core.preprocessing import aligner, detector, normalizer
from data.loaders import att_loader, fgnet_loader, yale_loader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_SIZE = (100, 100)  # (width, height)
N_COMPONENTS = 50
ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def preprocess_image_haar(image):
    """Preprocess a single image using Haar Cascade detection.

    Returns the flattened, normalized face vector, or None if
    face/eye detection fails.
    """
    face_box = detector.detect_face(image)
    if face_box is None:
        return None

    x, y, w, h = face_box
    face_crop = image[y : y + h, x : x + w]

    eyes = detector.detect_eyes(face_crop)
    if eyes is None:
        # Fallback: estimate eye positions at 30%/70% x, 35% y
        ew = int(w * 0.30)
        ew2 = int(w * 0.70)
        eh = int(h * 0.35)
        eyes = [(ew, eh), (ew2, eh)]

    aligned = aligner.align_face(face_crop, eyes, output_size=IMAGE_SIZE)
    normalized = normalizer.normalize(aligned)
    return normalizer.flatten(normalized)


def preprocess_image_landmarks(image, landmarks_array):
    """Preprocess a single image using pre-computed landmarks.

    The landmarks provide exact eye positions, bypassing Haar Cascade.
    """
    eye_positions = fgnet_loader.get_eye_positions_from_landmarks(
        landmarks_array
    )
    aligned = aligner.align_face(image, eye_positions, output_size=IMAGE_SIZE)
    normalized = normalizer.normalize(aligned)
    return normalizer.flatten(normalized)


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
    print("\n[1/8] Loading AT&T training data...")
    att_images, att_ids = att_loader.load(split="train")
    print(f"  AT&T: {len(att_images)} images, {len(set(att_ids))} subjects")

    print("[1/8] Loading Yale training data...")
    yale_images, yale_ids, _ = yale_loader.load(split="train")
    print(f"  Yale: {len(yale_images)} images, {len(set(yale_ids))} subjects")

    # ------------------------------------------------------------------
    # Step 2: Preprocess training images
    # ------------------------------------------------------------------
    print("\n[2/8] Preprocessing training images...")
    train_vectors = []
    skipped = 0

    for i, img in enumerate(att_images):
        vec = preprocess_image_haar(img)
        if vec is not None:
            train_vectors.append(vec)
        else:
            skipped += 1
        if (i + 1) % 100 == 0:
            print(f"  AT&T: {i + 1}/{len(att_images)} processed")

    for i, img in enumerate(yale_images):
        vec = preprocess_image_haar(img)
        if vec is not None:
            train_vectors.append(vec)
        else:
            skipped += 1

    train_matrix = np.array(train_vectors)
    print(
        f"  Total: {len(train_vectors)} vectors "
        f"({skipped} skipped due to detection failure)"
    )
    print(f"  Matrix shape: {train_matrix.shape}")

    # ------------------------------------------------------------------
    # Step 3: Fit Eigenfaces
    # ------------------------------------------------------------------
    print(f"\n[3/8] Fitting Eigenfaces (n_components={N_COMPONENTS})...")
    eigenfaces = Eigenfaces(n_components=N_COMPONENTS)
    eigenfaces.fit(train_matrix)
    total_variance = eigenfaces.explained_variance_ratio.sum()
    print(f"  Explained variance: {total_variance:.4f} ({total_variance*100:.1f}%)")

    # ------------------------------------------------------------------
    # Step 4: Load FG-NET for calibration
    # ------------------------------------------------------------------
    print("\n[4/8] Loading FG-NET for calibration...")
    fgnet_images, fgnet_ids, fgnet_ages = fgnet_loader.load()
    fgnet_landmarks = fgnet_loader.load_landmarks()
    print(
        f"  FG-NET: {len(fgnet_images)} images, "
        f"{len(set(fgnet_ids))} subjects, "
        f"{len(fgnet_landmarks)} landmarks"
    )

    # Preprocess FG-NET images
    print("\n[5/8] Preprocessing FG-NET images...")
    fgnet_vectors = []
    fgnet_valid_ids = []
    fgnet_valid_ages = []
    fgnet_skipped = 0

    # Build a mapping from image index to landmark key
    fgnet_filenames = sorted(os.listdir(
        os.path.join(os.path.normpath("data/raw/fgnet"), "images")
    ))
    fgnet_stems = []
    for fn in fgnet_filenames:
        match = fgnet_loader.FILENAME_PATTERN.match(os.path.splitext(fn)[0])
        if match:
            fgnet_stems.append(os.path.splitext(fn)[0].lower())
        else:
            fgnet_stems.append(None)

    for i, (img, sid, age) in enumerate(
        zip(fgnet_images, fgnet_ids, fgnet_ages)
    ):
        # Try landmark-based preprocessing first
        stem = fgnet_stems[i] if i < len(fgnet_stems) else None
        vec = None

        if stem and stem in fgnet_landmarks:
            try:
                vec = preprocess_image_landmarks(img, fgnet_landmarks[stem])
            except Exception:
                vec = None

        # Fallback to Haar Cascade
        if vec is None:
            vec = preprocess_image_haar(img)

        if vec is not None:
            fgnet_vectors.append(vec)
            fgnet_valid_ids.append(sid)
            fgnet_valid_ages.append(age)
        else:
            fgnet_skipped += 1

        if (i + 1) % 200 == 0:
            print(f"  FG-NET: {i + 1}/{len(fgnet_images)} processed")

    print(
        f"  Processed: {len(fgnet_vectors)} "
        f"({fgnet_skipped} skipped)"
    )

    # ------------------------------------------------------------------
    # Step 5: Build genuine/impostor pairs
    # ------------------------------------------------------------------
    print("\n[6/8] Building genuine/impostor pairs...")

    # Transform FG-NET vectors into eigenspace
    fgnet_matrix = np.array(fgnet_vectors)
    fgnet_coeffs = eigenfaces.transform(fgnet_matrix)

    # Group by subject
    subject_indices: dict[str, list[int]] = {}
    for idx, sid in enumerate(fgnet_valid_ids):
        subject_indices.setdefault(sid, []).append(idx)

    genuine_distances = {"euclidean": [], "cosine": [], "mahalanobis": []}
    impostor_distances = {"euclidean": [], "cosine": [], "mahalanobis": []}

    subjects = sorted(subject_indices.keys())

    # Genuine pairs: same subject, different images
    for sid in subjects:
        indices = subject_indices[sid]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = fgnet_coeffs[indices[i]], fgnet_coeffs[indices[j]]
                genuine_distances["euclidean"].append(
                    distances.euclidean(a, b)
                )
                genuine_distances["cosine"].append(distances.cosine(a, b))
                genuine_distances["mahalanobis"].append(
                    distances.mahalanobis(a, b, eigenfaces.explained_variance)
                )

    # Impostor pairs: different subjects (sample to keep manageable)
    rng = np.random.RandomState(42)
    n_impostor_target = len(genuine_distances["euclidean"])
    impostor_count = 0
    max_attempts = n_impostor_target * 10

    for _ in range(max_attempts):
        if impostor_count >= n_impostor_target:
            break
        s1, s2 = rng.choice(subjects, size=2, replace=False)
        i1 = rng.choice(subject_indices[s1])
        i2 = rng.choice(subject_indices[s2])
        a, b = fgnet_coeffs[i1], fgnet_coeffs[i2]
        impostor_distances["euclidean"].append(distances.euclidean(a, b))
        impostor_distances["cosine"].append(distances.cosine(a, b))
        impostor_distances["mahalanobis"].append(
            distances.mahalanobis(a, b, eigenfaces.explained_variance)
        )
        impostor_count += 1

    print(
        f"  Genuine pairs: {len(genuine_distances['euclidean'])}, "
        f"Impostor pairs: {len(impostor_distances['euclidean'])}"
    )

    # ------------------------------------------------------------------
    # Step 6: Compute calibration statistics
    # ------------------------------------------------------------------
    print("\n[7/8] Computing calibration statistics...")

    calibration = {"image_size": list(IMAGE_SIZE), "n_components": N_COMPONENTS}

    for metric in ["euclidean", "cosine", "mahalanobis"]:
        gen = np.array(genuine_distances[metric])
        imp = np.array(impostor_distances[metric])

        mean_all = float(np.concatenate([gen, imp]).mean())
        std_all = float(np.concatenate([gen, imp]).std())

        # Normalize all scores to confidence values
        gen_conf = np.array([
            ensemble.normalize_score(d, mean_all, std_all) for d in gen
        ])
        imp_conf = np.array([
            ensemble.normalize_score(d, mean_all, std_all) for d in imp
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

        print(
            f"  {metric}: EER={eer_value:.4f}, "
            f"threshold={eer_threshold:.4f}, "
            f"genuine_mean={gen.mean():.4f}, "
            f"impostor_mean={imp.mean():.4f}"
        )

    # Compute ensemble threshold (average of per-metric EER thresholds)
    eer_thresholds = [
        calibration[m]["eer_threshold"]
        for m in ["euclidean", "cosine", "mahalanobis"]
    ]
    calibration["ensemble_threshold"] = float(np.mean(eer_thresholds))
    print(
        f"\n  Ensemble threshold: {calibration['ensemble_threshold']:.4f}"
    )

    # ------------------------------------------------------------------
    # Step 7: Save artifacts
    # ------------------------------------------------------------------
    print(f"\n[8/8] Saving artifacts to {ARTIFACTS_DIR}/...")
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
