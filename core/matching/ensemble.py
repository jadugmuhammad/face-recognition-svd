"""Combines multiple normalized distance metrics into one confidence score."""

import numpy as np


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


def combine(scores: dict, weights: dict | None = None) -> float:
    """Combine normalized per-metric scores into one confidence value.

    Args:
        scores: e.g. ``{"euclidean": 0.8, "cosine": 0.9,
            "mahalanobis": 0.75}``.
        weights: Optional per-metric weights.  Defaults to equal
            weighting.  Weights are automatically normalized to sum
            to 1.

    Returns:
        float: Combined confidence score in ``[0, 1]``.

    Raises:
        ValueError: If *scores* is empty.
    """
    if not scores:
        raise ValueError("scores dict must not be empty")

    metric_names = list(scores.keys())

    if weights is None:
        w = {name: 1.0 / len(metric_names) for name in metric_names}
    else:
        # Normalize weights to sum to 1.
        total = sum(weights.get(name, 0.0) for name in metric_names)
        if total < 1e-12:
            # All weights are zero — fall back to equal weighting.
            w = {name: 1.0 / len(metric_names) for name in metric_names}
        else:
            w = {
                name: weights.get(name, 0.0) / total
                for name in metric_names
            }

    combined = sum(scores[name] * w[name] for name in metric_names)
    return float(np.clip(combined, 0.0, 1.0))
