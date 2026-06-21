"""Tests for core.decomposition — Eigenfaces PCA wrapper."""

import numpy as np
import pytest

from core.decomposition.eigenfaces import Eigenfaces


def _make_synthetic_faces(n_samples=100, n_pixels=400, n_true_components=10):
    """Generate synthetic 'face' data with known dimensionality."""
    rng = np.random.RandomState(42)
    # Low-rank data: n_true_components real dimensions embedded in n_pixels
    basis = rng.randn(n_true_components, n_pixels)
    coefficients = rng.randn(n_samples, n_true_components)
    data = coefficients @ basis
    # Shift to positive range (like pixel values)
    data -= data.min()
    data /= data.max()
    return data


class TestEigenfaces:
    """Tests for core.decomposition.eigenfaces.Eigenfaces."""

    def test_fit_sets_attributes(self):
        data = _make_synthetic_faces()
        ef = Eigenfaces(n_components=5)
        ef.fit(data)
        assert ef.mean_face is not None
        assert ef.components is not None
        assert ef.explained_variance is not None

    def test_mean_face_shape(self):
        data = _make_synthetic_faces(n_pixels=400)
        ef = Eigenfaces(n_components=5)
        ef.fit(data)
        assert ef.mean_face.shape == (400,)

    def test_components_shape(self):
        data = _make_synthetic_faces(n_pixels=400)
        ef = Eigenfaces(n_components=5)
        ef.fit(data)
        assert ef.components.shape == (5, 400)

    def test_explained_variance_shape(self):
        data = _make_synthetic_faces()
        ef = Eigenfaces(n_components=5)
        ef.fit(data)
        assert ef.explained_variance.shape == (5,)

    def test_explained_variance_ratio_sums_to_at_most_one(self):
        data = _make_synthetic_faces()
        ef = Eigenfaces(n_components=5)
        ef.fit(data)
        ratio_sum = ef.explained_variance_ratio.sum()
        assert ratio_sum <= 1.0 + 1e-6

    def test_explained_variance_ratio_is_sorted_descending(self):
        data = _make_synthetic_faces()
        ef = Eigenfaces(n_components=10)
        ef.fit(data)
        ratios = ef.explained_variance_ratio
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1] - 1e-10

    def test_transform_output_shape_batch(self):
        data = _make_synthetic_faces(n_samples=50, n_pixels=400)
        ef = Eigenfaces(n_components=8)
        ef.fit(data)
        coeffs = ef.transform(data)
        assert coeffs.shape == (50, 8)

    def test_transform_output_shape_single(self):
        data = _make_synthetic_faces(n_samples=50, n_pixels=400)
        ef = Eigenfaces(n_components=8)
        ef.fit(data)
        coeffs = ef.transform(data[0])  # single vector
        assert coeffs.shape == (8,)

    def test_inverse_transform_shape(self):
        data = _make_synthetic_faces(n_pixels=400)
        ef = Eigenfaces(n_components=10)
        ef.fit(data)
        coeffs = ef.transform(data[:5])
        reconstructed = ef.inverse_transform(coeffs)
        assert reconstructed.shape == (5, 400)

    def test_inverse_transform_single(self):
        data = _make_synthetic_faces(n_pixels=400)
        ef = Eigenfaces(n_components=10)
        ef.fit(data)
        coeffs = ef.transform(data[0])
        reconstructed = ef.inverse_transform(coeffs)
        assert reconstructed.shape == (400,)

    def test_reconstruction_quality(self):
        """With enough components, reconstruction should be close to original."""
        data = _make_synthetic_faces(
            n_samples=100, n_pixels=400, n_true_components=5
        )
        ef = Eigenfaces(n_components=10)  # more than true rank
        ef.fit(data)
        coeffs = ef.transform(data)
        reconstructed = ef.inverse_transform(coeffs)
        # Mean squared error should be very small
        mse = np.mean((data - reconstructed) ** 2)
        assert mse < 1e-6, f"MSE too high: {mse}"

    def test_get_eigenface_1d(self):
        data = _make_synthetic_faces(n_pixels=400)
        ef = Eigenfaces(n_components=5)
        ef.fit(data)
        face = ef.get_eigenface(0)
        assert face.shape == (400,)

    def test_get_eigenface_2d(self):
        data = _make_synthetic_faces(n_pixels=400)
        ef = Eigenfaces(n_components=5)
        ef.fit(data)
        face = ef.get_eigenface(0, image_shape=(20, 20))
        assert face.shape == (20, 20)

    def test_fit_returns_self(self):
        data = _make_synthetic_faces()
        ef = Eigenfaces(n_components=5)
        result = ef.fit(data)
        assert result is ef
