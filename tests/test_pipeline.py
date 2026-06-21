"""Integration tests for core.pipeline — end-to-end comparison."""

import numpy as np
import pytest

from core.pipeline import compare, FaceComparisonResult


def _load_att_pair_same():
    """Load two images of the SAME person from AT&T."""
    from data.loaders import att_loader
    images, ids = att_loader.load(split="train")
    # Find two images of s01
    s01_images = [img for img, sid in zip(images, ids) if sid == "s01"]
    return s01_images[0], s01_images[1]


def _load_att_pair_different():
    """Load two images of DIFFERENT people from AT&T."""
    from data.loaders import att_loader
    images, ids = att_loader.load(split="train")
    s01 = next(img for img, sid in zip(images, ids) if sid == "s01")
    s02 = next(img for img, sid in zip(images, ids) if sid == "s02")
    return s01, s02


class TestPipeline:
    """Integration tests for the full comparison pipeline."""

    def test_compare_returns_result(self):
        img_a, img_b = _load_att_pair_same()
        result = compare(img_a, img_b)
        assert isinstance(result, FaceComparisonResult)

    def test_result_has_all_fields(self):
        img_a, img_b = _load_att_pair_same()
        result = compare(img_a, img_b)
        assert isinstance(result.is_same, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert "euclidean" in result.metric_scores
        assert "cosine" in result.metric_scores
        assert "mahalanobis" in result.metric_scores
        assert "euclidean" in result.metric_confidences
        assert result.reconstruction_a is not None
        assert result.reconstruction_b is not None

    def test_reconstruction_shape(self):
        img_a, img_b = _load_att_pair_same()
        result = compare(img_a, img_b)
        # Image size is 100×100 (height, width)
        assert result.reconstruction_a.shape == (100, 100)
        assert result.reconstruction_b.shape == (100, 100)

    def test_pipeline_produces_different_scores(self):
        """Same-person and different-person comparisons should produce
        distinct metric scores (the pipeline is actually computing
        something meaningful, not returning constants).
        """
        img_same_a, img_same_b = _load_att_pair_same()
        img_diff_a, img_diff_b = _load_att_pair_different()

        result_same = compare(img_same_a, img_same_b)
        result_diff = compare(img_diff_a, img_diff_b)

        # The scores should be numerically different
        assert result_same.metric_scores != result_diff.metric_scores
        # Both should produce valid confidences
        assert 0.0 <= result_same.confidence <= 1.0
        assert 0.0 <= result_diff.confidence <= 1.0

    def test_metric_mode_single(self):
        """Single-metric mode should use only that metric's confidence."""
        img_a, img_b = _load_att_pair_same()
        result = compare(img_a, img_b, config={"metric_mode": "cosine"})
        # Confidence should equal the cosine metric confidence
        assert abs(result.confidence - result.metric_confidences["cosine"]) < 1e-10

    def test_custom_threshold(self):
        img_a, img_b = _load_att_pair_same()
        # Very high threshold → should reject
        result_strict = compare(img_a, img_b, config={"threshold": 0.999})
        assert result_strict.is_same is False

        # Very low threshold → should accept
        result_lenient = compare(img_a, img_b, config={"threshold": 0.001})
        assert result_lenient.is_same is True

    def test_identical_images(self):
        """Comparing an image to itself should give very high confidence."""
        img_a, _ = _load_att_pair_same()
        result = compare(img_a, img_a)
        assert result.confidence > 0.5
        assert result.metric_scores["euclidean"] < 1e-6
        assert result.metric_scores["cosine"] < 1e-6
