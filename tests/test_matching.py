"""Tests for core.matching — distances, ensemble, and threshold."""

import numpy as np
import pytest

from core.matching import distances, threshold


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

class TestDistances:
    """Tests for core.matching.distances."""

    def test_cosine_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(distances.cosine(v, v)) < 1e-10

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(distances.cosine(a, b) - 1.0) < 1e-10

    def test_cosine_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(distances.cosine(a, b) - 2.0) < 1e-10

    def test_cosine_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        result = distances.cosine(a, b)
        assert result == 1.0  # treated as maximally dissimilar

    def test_cosine_symmetry(self):
        rng = np.random.RandomState(42)
        a, b = rng.randn(10), rng.randn(10)
        assert abs(distances.cosine(a, b) - distances.cosine(b, a)) < 1e-10

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    """Tests for core.matching.distances.normalize_score."""

    def test_normalize_score_mean_returns_half(self):
        """A raw score equal to the mean should give ~0.5."""
        result = distances.normalize_score(raw_score=5.0, mean=5.0, std=1.0)
        assert abs(result - 0.5) < 1e-10

    def test_normalize_score_below_mean_higher_confidence(self):
        """Smaller distance (below mean) → higher confidence."""
        result = distances.normalize_score(raw_score=3.0, mean=5.0, std=1.0)
        assert result > 0.5

    def test_normalize_score_above_mean_lower_confidence(self):
        """Larger distance (above mean) → lower confidence."""
        result = distances.normalize_score(raw_score=7.0, mean=5.0, std=1.0)
        assert result < 0.5

    def test_normalize_score_bounded(self):
        """Output should always be in [0, 1]."""
        for raw in [-100, -1, 0, 1, 100]:
            result = distances.normalize_score(raw, mean=0.0, std=1.0)
            assert 0.0 <= result <= 1.0

    def test_normalize_score_zero_std(self):
        result = distances.normalize_score(5.0, mean=5.0, std=0.0)
        assert result == 0.5


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

class TestThreshold:
    """Tests for core.matching.threshold."""

    def test_decide_above_threshold(self):
        assert threshold.decide(0.8, 0.5) is True

    def test_decide_below_threshold(self):
        assert threshold.decide(0.3, 0.5) is False

    def test_decide_at_threshold(self):
        assert threshold.decide(0.5, 0.5) is True

    def test_compute_roc_shape(self):
        genuine = np.array([0.8, 0.9, 0.7, 0.85])
        impostor = np.array([0.2, 0.3, 0.4, 0.15])
        fpr, tpr, thresholds = threshold.compute_roc(genuine, impostor)
        assert len(fpr) == len(tpr) == len(thresholds)
        assert len(fpr) > 0

    def test_compute_roc_boundary_values(self):
        genuine = np.array([0.9, 0.8, 0.85])
        impostor = np.array([0.1, 0.2, 0.15])
        fpr, tpr, _ = threshold.compute_roc(genuine, impostor)
        # ROC should start at (0,0) and end at (1,1)
        assert fpr[0] == 0.0
        assert fpr[-1] == 1.0

    def test_compute_eer_well_separated(self):
        """Well-separated distributions should have low EER."""
        genuine = np.random.RandomState(42).normal(0.9, 0.05, 100)
        impostor = np.random.RandomState(43).normal(0.1, 0.05, 100)
        eer_value, eer_threshold = threshold.compute_eer(genuine, impostor)
        assert eer_value < 0.1, f"EER too high for well-separated data: {eer_value}"
        assert 0.0 < eer_threshold < 1.0

    def test_compute_eer_overlapping(self):
        """Heavily overlapping distributions should have higher EER."""
        genuine = np.random.RandomState(42).normal(0.5, 0.2, 100)
        impostor = np.random.RandomState(43).normal(0.5, 0.2, 100)
        eer_value, _ = threshold.compute_eer(genuine, impostor)
        # With completely overlapping, EER should be around 0.5
        assert eer_value > 0.2
