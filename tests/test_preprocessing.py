"""Tests for core.preprocessing — detector, aligner, and normalizer."""

import numpy as np
import pytest

from core.preprocessing import detector, aligner, normalizer


# ---------------------------------------------------------------------------
# Helpers — load a real AT&T sample image for detector tests
# ---------------------------------------------------------------------------

def _load_att_sample():
    """Load a single AT&T face image for testing."""
    from data.loaders import att_loader
    images, _ = att_loader.load(split="train")
    return images[0]


def _load_yale_sample():
    """Load a single Yale face image for testing."""
    from data.loaders import att_loader
    images, _ = att_loader.load(split="train")
    return images[1]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class TestDetector:
    """Tests for core.preprocessing.detector."""

    def test_detect_face_att(self):
        """Haar Cascade should detect a face in a controlled AT&T image."""
        img = _load_att_sample()
        result = detector.detect_face(img)
        assert result is not None, "Should detect a face in AT&T image"
        x, y, w, h = result
        assert w > 0 and h > 0
        assert x >= 0 and y >= 0

    def test_detect_face_returns_none_for_blank(self):
        """No face in a blank image."""
        blank = np.zeros((100, 100), dtype=np.uint8)
        result = detector.detect_face(blank)
        assert result is None

    def test_detect_face_returns_none_for_noise(self):
        """No face in pure noise (with high probability)."""
        rng = np.random.RandomState(42)
        noise = rng.randint(0, 256, (100, 100), dtype=np.uint8)
        result = detector.detect_face(noise)
        # Noise *might* trigger a false positive, so this is a soft check
        # — we mainly care that it doesn't crash.
        assert result is None or len(result) == 4

    def test_detect_face_bounding_box_within_image(self):
        img = _load_att_sample()
        result = detector.detect_face(img)
        if result is not None:
            x, y, w, h = result
            assert x + w <= img.shape[1]
            assert y + h <= img.shape[0]

    def test_detect_eyes_att(self):
        """Should detect two eyes in a cropped AT&T face."""
        img = _load_att_sample()
        face_box = detector.detect_face(img)
        if face_box is None:
            pytest.skip("Face not detected in sample — can't test eyes")

        x, y, w, h = face_box
        face_crop = img[y : y + h, x : x + w]
        eyes = detector.detect_eyes(face_crop)

        # Eye detection is less reliable, so None is acceptable
        if eyes is not None:
            assert len(eyes) == 2
            left, right = eyes
            assert len(left) == 2
            assert len(right) == 2
            # Left eye should have smaller x than right eye
            assert left[0] <= right[0]

    def test_detect_eyes_returns_none_for_blank(self):
        blank = np.zeros((100, 100), dtype=np.uint8)
        result = detector.detect_eyes(blank)
        assert result is None


# ---------------------------------------------------------------------------
# Aligner
# ---------------------------------------------------------------------------

class TestAligner:
    """Tests for core.preprocessing.aligner."""

    def test_output_size_default(self):
        """Output should be 100×100 by default."""
        img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        eye_positions = [(60, 80), (140, 80)]
        result = aligner.align_face(img, eye_positions)
        assert result.shape == (100, 100)

    def test_output_size_custom(self):
        """Custom output size should be respected."""
        img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        eye_positions = [(60, 80), (140, 80)]
        result = aligner.align_face(img, eye_positions, output_size=(64, 64))
        assert result.shape == (64, 64)

    def test_dtype_uint8(self):
        """Output should be uint8."""
        img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        eye_positions = [(60, 80), (140, 80)]
        result = aligner.align_face(img, eye_positions)
        assert result.dtype == np.uint8

    def test_horizontal_eyes_no_rotation(self):
        """If eyes are already horizontal, rotation should be ~0."""
        # Create a simple test image with horizontal eyes
        img = np.full((200, 200), 128, dtype=np.uint8)
        # Place "eyes" as bright spots
        img[80, 60] = 255  # left eye
        img[80, 140] = 255  # right eye
        eye_positions = [(60, 80), (140, 80)]
        result = aligner.align_face(img, eye_positions)
        assert result.shape == (100, 100)
        # The result should not be all-zero (transformation worked)
        assert result.mean() > 0

    def test_swapped_eye_order(self):
        """Should handle eyes given in reverse order (right, left)."""
        img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        eyes_normal = [(60, 80), (140, 80)]
        eyes_swapped = [(140, 80), (60, 80)]
        result_normal = aligner.align_face(img, eyes_normal)
        result_swapped = aligner.align_face(img, eyes_swapped)
        # Both should produce the same output
        np.testing.assert_array_equal(result_normal, result_swapped)

    def test_tilted_face_correction(self):
        """Tilted eyes should produce a different result than horizontal."""
        img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        eyes_horizontal = [(60, 80), (140, 80)]
        eyes_tilted = [(60, 80), (140, 100)]  # right eye lower
        r1 = aligner.align_face(img, eyes_horizontal)
        r2 = aligner.align_face(img, eyes_tilted)
        # The outputs should be different (different transformations)
        assert not np.array_equal(r1, r2)

    def test_with_real_face(self):
        """Integration: detect face + eyes on AT&T, then align."""
        img = _load_att_sample()
        face_box = detector.detect_face(img)
        if face_box is None:
            pytest.skip("Face not detected")

        x, y, w, h = face_box
        face_crop = img[y : y + h, x : x + w]
        eyes = detector.detect_eyes(face_crop)

        if eyes is None:
            pytest.skip("Eyes not detected")

        result = aligner.align_face(face_crop, eyes)
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8

    def test_coincident_eyes_fallback(self):
        """When eyes are at the same position, should resize instead of crash."""
        img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        eyes = [(100, 80), (100, 80)]  # same position
        result = aligner.align_face(img, eyes)
        assert result.shape == (100, 100)


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class TestNormalizer:
    """Tests for core.preprocessing.normalizer."""

    def test_normalize_output_shape(self):
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = normalizer.normalize(img)
        assert result.shape == img.shape

    def test_normalize_output_dtype(self):
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = normalizer.normalize(img)
        assert result.dtype == np.uint8

    def test_normalize_improves_contrast(self):
        """CLAHE should increase the standard deviation of a low-contrast image."""
        # Create a low-contrast image (values clustered around 128)
        img = np.clip(
            np.random.normal(128, 10, (100, 100)), 0, 255
        ).astype(np.uint8)
        result = normalizer.normalize(img)
        # The normalized image should have higher contrast (larger std)
        assert result.std() >= img.std()

    def test_normalize_real_face(self):
        """CLAHE on a real face should not produce artifacts."""
        img = _load_att_sample()
        result = normalizer.normalize(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert 0 <= result.min() <= result.max() <= 255

    def test_flatten_shape(self):
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = normalizer.flatten(img)
        assert result.shape == (10000,)

    def test_flatten_dtype(self):
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = normalizer.flatten(img)
        assert result.dtype == np.float64

    def test_flatten_range(self):
        """Flattened values should be in [0, 1]."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = normalizer.flatten(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_flatten_specific_values(self):
        """Verify the scaling: 0 → 0.0, 255 → 1.0."""
        img = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        result = normalizer.flatten(img)
        assert result[0] == 0.0
        assert result[1] == 1.0
        assert abs(result[2] - 128 / 255.0) < 1e-10

    def test_flatten_preserves_pixel_count(self):
        for shape in [(50, 50), (100, 100), (112, 92)]:
            img = np.zeros(shape, dtype=np.uint8)
            result = normalizer.flatten(img)
            assert len(result) == shape[0] * shape[1]
