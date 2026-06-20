"""Threshold calibration: ROC curve, Equal Error Rate (EER), and the
SAME/DIFFERENT decision rule.
"""


def decide(confidence: float, threshold: float) -> bool:
    """Apply the threshold rule.

    Returns:
        bool: True if `confidence` indicates the same person.
    """
    raise NotImplementedError


def compute_roc(genuine_scores, impostor_scores):
    """Compute ROC curve points from genuine/impostor score distributions."""
    raise NotImplementedError


def compute_eer(genuine_scores, impostor_scores) -> float:
    """Compute the Equal Error Rate and its corresponding threshold."""
    raise NotImplementedError
