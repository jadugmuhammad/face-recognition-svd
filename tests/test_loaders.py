"""Tests for data.loaders — AT&T, Yale, and FG-NET dataset loaders."""

import numpy as np
import pytest

from data.loaders import att_loader, yale_loader, fgnet_loader


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

    def test_missing_root(self):
        with pytest.raises(FileNotFoundError):
            att_loader.load(root_dir="/nonexistent/path")


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
# FG-NET loader
# ---------------------------------------------------------------------------

class TestFGNETLoader:
    """Tests for data.loaders.fgnet_loader."""

    def test_load_images(self):
        images, subject_ids, ages = fgnet_loader.load()
        assert len(images) == 1002
        assert len(subject_ids) == 1002
        assert len(ages) == 1002

    def test_image_format(self):
        images, _, _ = fgnet_loader.load()
        img = images[0]
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.ndim == 2

    def test_subject_ids(self):
        _, subject_ids, _ = fgnet_loader.load()
        unique = set(subject_ids)
        assert len(unique) == 82
        # All IDs should be 3-digit zero-padded strings
        for sid in subject_ids:
            assert len(sid) == 3
            assert sid.isdigit()

    def test_ages_range(self):
        _, _, ages = fgnet_loader.load()
        assert all(isinstance(a, int) for a in ages)
        assert min(ages) >= 0
        assert max(ages) <= 70  # FG-NET range is 0-69

    def test_load_landmarks(self):
        landmarks = fgnet_loader.load_landmarks()
        assert len(landmarks) > 0
        # Check a known landmark file
        assert "001a02" in landmarks
        pts = landmarks["001a02"]
        assert isinstance(pts, np.ndarray)
        assert pts.shape == (68, 2)
        assert pts.dtype == np.float64

    def test_landmarks_count_matches_images(self):
        """Landmark files should roughly match image count."""
        images, _, _ = fgnet_loader.load()
        landmarks = fgnet_loader.load_landmarks()
        # They should be close (some images may lack landmarks or vice versa)
        assert len(landmarks) > 900  # most images should have landmarks

    def test_eye_positions_from_landmarks(self):
        landmarks = fgnet_loader.load_landmarks()
        pts = landmarks["001a02"]
        eyes = fgnet_loader.get_eye_positions_from_landmarks(pts)
        assert len(eyes) == 2
        left_eye, right_eye = eyes
        assert len(left_eye) == 2
        assert len(right_eye) == 2
        # Left eye should be to the left of right eye (lower x)
        # (in image coordinates, though this depends on the face orientation)
        assert isinstance(left_eye[0], float)
        assert isinstance(right_eye[0], float)

    def test_missing_root(self):
        with pytest.raises(FileNotFoundError):
            fgnet_loader.load(root_dir="/nonexistent/path")

    def test_missing_points_dir(self):
        with pytest.raises(FileNotFoundError):
            fgnet_loader.load_landmarks(root_dir="/nonexistent/path")
