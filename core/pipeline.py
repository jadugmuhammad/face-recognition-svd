"""Orchestrates the full face-comparison pipeline: preprocessing ->
decomposition -> matching -> decision.

This is the only module ``app/`` is allowed to import core logic from -- it
deliberately knows nothing about Streamlit.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np

from core.decomposition.eigenfaces import Eigenfaces
from core.matching import distances, ensemble, threshold
from core.preprocessing import aligner, detector, normalizer

# Default paths (relative to project root).
_DEFAULT_EIGENSPACE = os.path.join("artifacts", "eigenspace.npz")
_DEFAULT_CALIBRATION = os.path.join("artifacts", "calibration.json")

# Module-level cache so artifacts are loaded only once per process.
_cache: dict = {}


@dataclass
class FaceComparisonResult:
    """Result of comparing two face images.

    Attributes:
        is_same: Final SAME/DIFFERENT decision.
        confidence: Combined confidence score in [0, 1].
        metric_scores: Per-metric raw distance scores,
            e.g. ``{"euclidean": 3.14, ...}``.
        metric_confidences: Per-metric normalized confidence scores
            in [0, 1].
        threshold: The threshold used for the decision.
        reconstruction_a: Reconstructed (PCA-denoised) version of
            image A as a 2-D array, or ``None``.
        reconstruction_b: Reconstructed (PCA-denoised) version of
            image B as a 2-D array, or ``None``.
    """

    is_same: bool
    confidence: float
    metric_scores: dict
    metric_confidences: dict = field(default_factory=dict)
    threshold: float = 0.5
    reconstruction_a: np.ndarray | None = None
    reconstruction_b: np.ndarray | None = None


def _load_artifacts(eigenspace_path=None, calibration_path=None):
    """Load and cache eigenspace + calibration artifacts.

    Returns:
        tuple[Eigenfaces, dict]: The fitted Eigenfaces model and
        the calibration dictionary.
    """
    eigenspace_path = eigenspace_path or _DEFAULT_EIGENSPACE
    calibration_path = calibration_path or _DEFAULT_CALIBRATION

    cache_key = (eigenspace_path, calibration_path)
    if cache_key in _cache:
        return _cache[cache_key]

    # Load eigenspace
    if not os.path.isfile(eigenspace_path):
        raise FileNotFoundError(
            f"Eigenspace artifacts not found: {eigenspace_path}. "
            "Run `python scripts/build_eigenspace.py` first."
        )

    data = np.load(eigenspace_path)
    image_size = tuple(data["image_size"])
    n_components = data["components"].shape[0]

    ef = Eigenfaces(n_components=n_components)
    # Manually set the fitted PCA state.
    ef.pca.mean_ = data["mean_face"]
    ef.pca.components_ = data["components"]
    ef.pca.explained_variance_ = data["explained_variance"]
    ef.pca.explained_variance_ratio_ = data["explained_variance_ratio"]
    ef.pca.n_components_ = n_components
    ef.pca.n_features_in_ = data["mean_face"].shape[0]

    ef.mean_face = data["mean_face"].copy()
    ef.components = data["components"].copy()
    ef.explained_variance = data["explained_variance"].copy()

    # Load calibration
    if not os.path.isfile(calibration_path):
        raise FileNotFoundError(
            f"Calibration data not found: {calibration_path}. "
            "Run `python scripts/build_eigenspace.py` first."
        )

    with open(calibration_path, "r") as f:
        calibration = json.load(f)

    _cache[cache_key] = (ef, calibration, image_size)
    return ef, calibration, image_size


def _preprocess_single(image, image_size):
    """Preprocess a single input image for comparison.

    Uses Haar Cascade detection (for user-uploaded photos).

    Args:
        image: Grayscale image as a numpy array.
        image_size: Target ``(width, height)`` for alignment.

    Returns:
        numpy.ndarray | None: Flattened, normalized face vector, or
        ``None`` if preprocessing fails.
    """
    face_box = detector.detect_face(image)
    if face_box is None:
        # Fallback: treat the entire image as a face.
        face_crop = image
        h, w = image.shape[:2]
        eyes = [(int(w * 0.30), int(h * 0.35)), (int(w * 0.70), int(h * 0.35))]
    else:
        x, y, w, h = face_box
        face_crop = image[y : y + h, x : x + w]
        eyes = detector.detect_eyes(face_crop)
        if eyes is None:
            eyes = [
                (int(w * 0.30), int(h * 0.35)),
                (int(w * 0.70), int(h * 0.35)),
            ]

    aligned = aligner.align_face(face_crop, eyes, output_size=image_size)
    normalized = normalizer.normalize(aligned)
    return normalizer.flatten(normalized)


def compare(image_a, image_b, config: dict | None = None) -> FaceComparisonResult:
    """Compare two face images end-to-end.

    Args:
        image_a: First input image (grayscale numpy array).
        image_b: Second input image (grayscale numpy array).
        config: Optional overrides:
            - ``"n_components"`` (int): ignored for now (uses trained value).
            - ``"metric_mode"`` (str): ``"ensemble"`` | ``"euclidean"``
              | ``"cosine"`` | ``"mahalanobis"``.
            - ``"threshold"`` (float): decision threshold override.
            - ``"eigenspace_path"`` (str): path to ``eigenspace.npz``.
            - ``"calibration_path"`` (str): path to ``calibration.json``.

    Returns:
        FaceComparisonResult with the decision and supporting details.

    Raises:
        FileNotFoundError: If artifacts have not been built yet.
        ValueError: If face preprocessing fails for either image.
    """
    config = config or {}

    # Load artifacts
    ef, calibration, image_size = _load_artifacts(
        eigenspace_path=config.get("eigenspace_path"),
        calibration_path=config.get("calibration_path"),
    )

    # Preprocess both images
    vec_a = _preprocess_single(image_a, image_size)
    vec_b = _preprocess_single(image_b, image_size)

    if vec_a is None:
        raise ValueError(
            "Preprocessing failed for image A — no face could be detected "
            "or aligned."
        )
    if vec_b is None:
        raise ValueError(
            "Preprocessing failed for image B — no face could be detected "
            "or aligned."
        )

    # Transform into eigenspace
    coeffs_a = ef.transform(vec_a)
    coeffs_b = ef.transform(vec_b)

    # Compute raw distances
    raw_scores = {
        "euclidean": distances.euclidean(coeffs_a, coeffs_b),
        "cosine": distances.cosine(coeffs_a, coeffs_b),
        "mahalanobis": distances.mahalanobis(
            coeffs_a, coeffs_b, ef.explained_variance
        ),
    }

    # Normalize to confidence scores using calibration stats
    metric_confidences = {}
    for metric in raw_scores:
        cal = calibration.get(metric, {})
        mean = cal.get("mean", 0.0)
        std = cal.get("std", 1.0)
        metric_confidences[metric] = ensemble.normalize_score(
            raw_scores[metric], mean, std
        )

    # Determine which metrics to use
    metric_mode = config.get("metric_mode", "ensemble")

    if metric_mode == "ensemble":
        confidence = ensemble.combine(metric_confidences)
    elif metric_mode in metric_confidences:
        confidence = metric_confidences[metric_mode]
    else:
        confidence = ensemble.combine(metric_confidences)

    # Decision
    thresh = config.get(
        "threshold", calibration.get("ensemble_threshold", 0.5)
    )
    is_same = threshold.decide(confidence, thresh)

    # Reconstructions for visualization
    recon_a = ef.inverse_transform(coeffs_a).reshape(
        image_size[1], image_size[0]
    )
    recon_b = ef.inverse_transform(coeffs_b).reshape(
        image_size[1], image_size[0]
    )

    return FaceComparisonResult(
        is_same=is_same,
        confidence=confidence,
        metric_scores=raw_scores,
        metric_confidences=metric_confidences,
        threshold=thresh,
        reconstruction_a=recon_a,
        reconstruction_b=recon_b,
    )
