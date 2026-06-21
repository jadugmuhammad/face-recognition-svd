"""Threshold calibration: ROC curve, Equal Error Rate (EER), and the
SAME/DIFFERENT decision rule.
"""

import numpy as np
from sklearn.metrics import roc_curve as _sklearn_roc_curve


def decide(confidence: float, threshold: float) -> bool:
    """Apply the threshold rule.

    Args:
        confidence: Combined confidence score in ``[0, 1]``.
        threshold: Decision boundary.

    Returns:
        bool: ``True`` if *confidence* indicates the same person
        (i.e. ``confidence >= threshold``).
    """
    return confidence >= threshold


def compute_roc(genuine_scores, impostor_scores):
    """Compute ROC curve points from genuine/impostor score distributions.

    Args:
        genuine_scores: Confidence scores for genuine (same-person) pairs.
        impostor_scores: Confidence scores for impostor (different-person)
            pairs.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        ``(fpr, tpr, thresholds)`` arrays suitable for plotting.
    """
    genuine_scores = np.asarray(genuine_scores, dtype=np.float64)
    impostor_scores = np.asarray(impostor_scores, dtype=np.float64)

    # Labels: 1 = genuine, 0 = impostor.
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores)),
    ])
    scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = _sklearn_roc_curve(labels, scores)
    return fpr, tpr, thresholds


def compute_eer(genuine_scores, impostor_scores):
    """Compute the Equal Error Rate and its corresponding threshold.

    The EER is the point on the ROC curve where the False Positive
    Rate equals the False Negative Rate (1 − TPR).  It summarizes
    overall system accuracy in a single number — lower is better.

    Args:
        genuine_scores: Confidence scores for genuine pairs.
        impostor_scores: Confidence scores for impostor pairs.

    Returns:
        tuple[float, float]: ``(eer_value, eer_threshold)`` where
        ``eer_value`` is the error rate at the EER point and
        ``eer_threshold`` is the corresponding decision threshold.
    """
    fpr, tpr, thresholds = compute_roc(genuine_scores, impostor_scores)
    fnr = 1.0 - tpr

    # Find the point where FPR ≈ FNR (they cross).
    # We look for the index where the sign of (FPR - FNR) changes.
    diff = fpr - fnr

    # Handle edge cases.
    if len(diff) < 2:
        return 0.5, 0.5

    # Find the crossing point via interpolation.
    idx = np.argmin(np.abs(diff))
    eer_value = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])

    return eer_value, eer_threshold
