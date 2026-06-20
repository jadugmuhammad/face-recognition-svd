"""Distance/similarity metrics between two eigenspace coefficient vectors."""


def euclidean(vector_a, vector_b) -> float:
    """Euclidean distance between two coefficient vectors."""
    raise NotImplementedError


def cosine(vector_a, vector_b) -> float:
    """Cosine distance (1 - cosine similarity) between two vectors."""
    raise NotImplementedError


def mahalanobis(vector_a, vector_b, explained_variance) -> float:
    """Mahalanobis-style distance, weighted by PCA explained variance.

    Args:
        vector_a: First eigenspace coefficient vector.
        vector_b: Second eigenspace coefficient vector.
        explained_variance: Per-component variance from the fitted
            Eigenfaces model, used as the (diagonal) covariance estimate.
    """
    raise NotImplementedError
