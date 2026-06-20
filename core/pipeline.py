"""Orchestrates the full face-comparison pipeline: preprocessing ->
decomposition -> matching -> decision.

This is the only module `app/` is allowed to import core logic from -- it
deliberately knows nothing about Streamlit.
"""

from dataclasses import dataclass


@dataclass
class FaceComparisonResult:
    """Result of comparing two face images.

    Attributes:
        is_same: Final SAME/DIFFERENT decision.
        confidence: Combined confidence score in [0, 1].
        metric_scores: Per-metric raw scores, e.g. {"euclidean": ...}.
        reconstruction_a: Reconstructed (PCA-denoised) version of image A.
        reconstruction_b: Reconstructed (PCA-denoised) version of image B.
    """

    is_same: bool
    confidence: float
    metric_scores: dict
    reconstruction_a = None
    reconstruction_b = None


def compare(image_a, image_b, config: dict | None = None) -> FaceComparisonResult:
    """Compare two face images end-to-end.

    Args:
        image_a: First input image (raw, unprocessed).
        image_b: Second input image (raw, unprocessed).
        config: Optional overrides (n_components, metric_mode, threshold).

    Returns:
        FaceComparisonResult with the decision and supporting details.
    """
    raise NotImplementedError(
        "Akan diimplementasikan setelah core.preprocessing, "
        "core.decomposition, dan core.matching siap."
    )
