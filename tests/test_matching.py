"""Tests for core.matching — distances, ensemble, and threshold."""

import numpy as np
import pytest

from core.matching import distances, ensemble, threshold


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

class TestDistances:
    """Tests for core.matching.distances."""

    def test_euclidean_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert distances.euclidean(v, v) == 0.0

    def test_euclidean_known_value(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert abs(distances.euclidean(a, b) - 5.0) < 1e-10

    def test_euclidean_symmetry(self):
        rng = np.random.RandomState(42)
        a, b = rng.randn(10), rng.randn(10)
        assert abs(distances.euclidean(a, b) - distances.euclidean(b, a)) < 1e-10

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

    def test_mahalanobis_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        var = np.array([1.0, 1.0, 1.0])
        assert distances.mahalanobis(v, v, var) == 0.0

    def test_mahalanobis_uniform_variance_equals_euclidean(self):
        """With uniform variance=1, Mahalanobis should equal Euclidean."""
        rng = np.random.RandomState(42)
        a, b = rng.randn(10), rng.randn(10)
        var = np.ones(10)
        mah = distances.mahalanobis(a, b, var)
        euc = distances.euclidean(a, b)
        assert abs(mah - euc) < 1e-10

    def test_mahalanobis_high_variance_reduces_weight(self):
        """Components with high variance should contribute less."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        var_uniform = np.array([1.0, 1.0])
        var_weighted = np.array([100.0, 1.0])
        d_uniform = distances.mahalanobis(a, b, var_uniform)
        d_weighted = distances.mahalanobis(a, b, var_weighted)
        # The weighted version should be smaller (first component downweighted)
        assert d_weighted < d_uniform

    def test_mahalanobis_symmetry(self):
        rng = np.random.RandomState(42)
        a, b = rng.randn(10), rng.randn(10)
        var = np.abs(rng.randn(10)) + 0.1
        assert abs(
            distances.mahalanobis(a, b, var) - distances.mahalanobis(b, a, var)
        ) < 1e-10


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class TestEnsemble:
    """Tests for core.matching.ensemble."""

    def test_normalize_score_mean_returns_half(self):
        """A raw score equal to the mean should give ~0.5."""
        result = ensemble.normalize_score(raw_score=5.0, mean=5.0, std=1.0)
        assert abs(result - 0.5) < 1e-10

    def test_normalize_score_below_mean_higher_confidence(self):
        """Smaller distance (below mean) → higher confidence."""
        result = ensemble.normalize_score(raw_score=3.0, mean=5.0, std=1.0)
        assert result > 0.5

    def test_normalize_score_above_mean_lower_confidence(self):
        """Larger distance (above mean) → lower confidence."""
        result = ensemble.normalize_score(raw_score=7.0, mean=5.0, std=1.0)
        assert result < 0.5

    def test_normalize_score_bounded(self):
        """Output should always be in [0, 1]."""
        for raw in [-100, -1, 0, 1, 100]:
            result = ensemble.normalize_score(raw, mean=0.0, std=1.0)
            assert 0.0 <= result <= 1.0

    def test_normalize_score_zero_std(self):
        result = ensemble.normalize_score(5.0, mean=5.0, std=0.0)
        assert result == 0.5

    def test_combine_equal_weights(self):
        scores = {"euclidean": 0.8, "cosine": 0.6, "mahalanobis": 0.7}
        result = ensemble.combine(scores)
        expected = (0.8 + 0.6 + 0.7) / 3.0
        assert abs(result - expected) < 1e-10

    def test_combine_custom_weights(self):
        scores = {"euclidean": 1.0, "cosine": 0.0}
        weights = {"euclidean": 1.0, "cosine": 1.0}
        result = ensemble.combine(scores, weights)
        assert abs(result - 0.5) < 1e-10

    def test_combine_single_metric(self):
        scores = {"cosine": 0.75}
        result = ensemble.combine(scores)
        assert abs(result - 0.75) < 1e-10

    def test_combine_clipped_to_01(self):
        scores = {"a": 0.0, "b": 0.0}
        result = ensemble.combine(scores)
        assert 0.0 <= result <= 1.0

    def test_combine_empty_raises(self):
        with pytest.raises(ValueError):
            ensemble.combine({})


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
