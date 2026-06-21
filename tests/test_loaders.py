"""Tests for data.loaders — AT&T and LFW dataset loaders."""

import numpy as np
import pytest

from data.loaders import att_loader, lfw_loader


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
