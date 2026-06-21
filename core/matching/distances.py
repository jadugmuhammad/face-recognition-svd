"""Distance/similarity metrics between two eigenspace coefficient vectors."""

import numpy as np
from scipy.spatial.distance import cosine as _scipy_cosine


def euclidean(vector_a, vector_b) -> float:
    """Euclidean distance between two coefficient vectors.

    Args:
        vector_a: First eigenspace coefficient vector.
        vector_b: Second eigenspace coefficient vector.

    Returns:
        float: L2 distance (≥ 0). Identical vectors → 0.
    """
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


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


def mahalanobis(vector_a, vector_b, explained_variance) -> float:
    """Mahalanobis-style distance, weighted by PCA explained variance.

    This is effectively a variance-weighted Euclidean distance:
    components with large variance (common patterns) get less weight,
    while components with small variance (unique details) get more
    weight.  This matches the original Eigenfaces paper
    (Turk & Pentland, 1991).

    Args:
        vector_a: First eigenspace coefficient vector.
        vector_b: Second eigenspace coefficient vector.
        explained_variance: Per-component variance from the fitted
            Eigenfaces model, used as the (diagonal) covariance
            estimate.

    Returns:
        float: Weighted distance (≥ 0).
    """
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)
    var = np.asarray(explained_variance, dtype=np.float64)

    # Avoid division by zero for near-zero variance components.
    var_safe = np.maximum(var, 1e-12)

    diff = a - b
    return float(np.sqrt(np.sum(diff ** 2 / var_safe)))
