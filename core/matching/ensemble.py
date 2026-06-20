"""Combines multiple normalized distance metrics into one confidence score."""


def normalize_score(raw_score: float, mean: float, std: float) -> float:
    """Z-score normalize a raw distance using calibration statistics."""
    raise NotImplementedError


def combine(scores: dict, weights: dict | None = None) -> float:
    """Combine normalized per-metric scores into one confidence value.

    Args:
        scores: e.g. {"euclidean": 0.8, "cosine": 0.9, "mahalanobis": 0.75}.
        weights: Optional per-metric weights, defaults to equal weighting.

    Returns:
        float: Combined confidence score in [0, 1].
    """
    raise NotImplementedError
