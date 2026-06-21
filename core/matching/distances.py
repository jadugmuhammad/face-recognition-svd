"""Distance/similarity metrics between two eigenspace coefficient vectors."""

import numpy as np
from scipy.spatial.distance import cosine as _scipy_cosine


def cosine(vector_a, vector_b) -> float:
    """Cosine distance (1 − cosine similarity) between two vectors.

    Args:
        vector_a: First eigenspace coefficient vector.
        vector_b: Second eigenspace coefficient vector.

    Returns:
        float: Value in ``[0, 2]``. Identical directions → 0,
        orthogonal → 1, opposite → 2.
    """
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)

    # Guard against zero vectors.
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return 1.0  # undefined → treat as maximally dissimilar

    return float(_scipy_cosine(a, b))


def normalize_score(raw_score: float, mean: float, std: float) -> float:
    """Z-score normalize a raw distance, then map to [0, 1] via sigmoid.

    A raw distance **smaller** than the mean (closer faces) maps to a
    confidence **above** 0.5, and vice versa.  The sigmoid ensures the
    output is smoothly bounded in ``(0, 1)``.

    Args:
        raw_score: The raw distance value.
        mean: Mean distance from calibration data.
        std: Standard deviation from calibration data.

    Returns:
        float: Normalized confidence in ``(0, 1)``.
    """
    if std < 1e-12:
        # Degenerate case — all calibration distances were identical.
        return 0.5

    # Z-score (negate so that smaller distance → higher confidence).
    z = -(raw_score - mean) / std

    # Sigmoid to map to (0, 1).
    confidence = 1.0 / (1.0 + np.exp(-z))
    return float(confidence)
