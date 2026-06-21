"""Tests for data.loaders — AT&T and LFW dataset loaders."""

import numpy as np
import pytest

from data.loaders import att_loader, yale_loader, lfw_loader


# ---------------------------------------------------------------------------
# AT&T loader
# ---------------------------------------------------------------------------

class TestATTLoader:
    """Tests for data.loaders.att_loader."""

    def test_load_train(self):
        images, subject_ids = att_loader.load(split="train")
        # 40 subjects × 9 training images = 360
        assert len(images) == 360
        assert len(subject_ids) == 360

    def test_load_test(self):
        images, subject_ids = att_loader.load(split="test")
        # 40 subjects × 1 test image = 40
        assert len(images) == 40
        assert len(subject_ids) == 40

    def test_load_both(self):
        images, subject_ids = att_loader.load(split="both")
        assert len(images) == 400
        assert len(subject_ids) == 400

    def test_image_format(self):
        images, _ = att_loader.load(split="train")
        img = images[0]
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.ndim == 2  # grayscale
        assert img.shape == (112, 92)  # AT&T native size

    def test_subject_ids_format(self):
        _, subject_ids = att_loader.load(split="train")
        # All IDs should be zero-padded like "s01", "s02", ...
        for sid in subject_ids:
            assert sid.startswith("s")
            assert len(sid) == 3  # "sNN"

    def test_unique_subjects(self):
        _, subject_ids = att_loader.load(split="train")
        unique = set(subject_ids)
        assert len(unique) == 40

    def test_invalid_split(self):
        with pytest.raises(ValueError, match="split must be"):
            att_loader.load(split="invalid")


# ---------------------------------------------------------------------------
# Yale loader
# ---------------------------------------------------------------------------

class TestYaleLoader:
    """Tests for data.loaders.yale_loader."""

    def test_load_train(self):
        images, subject_ids, conditions = yale_loader.load(split="train")
        # 15 subjects × ~10 unique images each (after dedup)
        assert len(images) > 0
        assert len(images) == len(subject_ids) == len(conditions)

    def test_load_test(self):
        images, subject_ids, conditions = yale_loader.load(split="test")
        # 15 subjects × 1 test image
        assert len(images) == 15
        assert len(subject_ids) == 15

    def test_load_both(self):
        images, subject_ids, conditions = yale_loader.load(split="both")
        assert len(images) > 15  # more than just test

    def test_image_format(self):
        images, _, _ = yale_loader.load(split="train")
        img = images[0]
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.ndim == 2
        assert img.shape == (243, 320)  # Yale native size

    def test_no_duplicates(self):
        """Ensure .glasses and .glasses.gif are not both loaded."""
        images, subject_ids, conditions = yale_loader.load(split="train")
        # For a single subject, "glasses" should appear at most once
        s01_conditions = [
            c for s, c in zip(subject_ids, conditions) if s == "s01"
        ]
        glasses_count = s01_conditions.count("glasses")
        assert glasses_count <= 1, (
            f"Expected at most 1 'glasses' for s01, got {glasses_count}"
        )

    def test_conditions_extracted(self):
        _, _, conditions = yale_loader.load(split="train")
        # Should contain known conditions
        condition_set = set(conditions)
        assert "normal" in condition_set
        assert "happy" in condition_set or "sad" in condition_set

    def test_unique_subjects(self):
        _, subject_ids, _ = yale_loader.load(split="train")
        unique = set(subject_ids)
        assert len(unique) == 15

    def test_invalid_split(self):
        with pytest.raises(ValueError, match="split must be"):
            yale_loader.load(split="invalid")


# ---------------------------------------------------------------------------
# LFW loader
# ---------------------------------------------------------------------------

class TestLFWLoader:
    """Tests for data.loaders.lfw_loader."""

    def test_load_images(self):
        images, subject_ids = lfw_loader.load(min_faces=50) # High min_faces to speed up tests
        assert len(images) > 0
        assert len(images) == len(subject_ids)

    def test_image_format(self):
        images, _ = lfw_loader.load(min_faces=50)
        img = images[0]
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.ndim == 2
