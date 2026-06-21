"""Eigenfaces: PCA/SVD-based face decomposition, built on scikit-learn."""

import numpy as np
from sklearn.decomposition import PCA


class Eigenfaces:
    """Wraps sklearn.decomposition.PCA for face decomposition tasks.

    Provides:
        - fit(): build the eigenspace from a training set of face vectors
        - transform(): project a face vector into eigenspace coefficients
        - inverse_transform(): reconstruct a face from its coefficients

    After ``fit()``, the following attributes are available:

    Attributes:
        mean_face: The mean face vector (average of all training faces).
        components: The principal components (eigenfaces), shape
            ``(n_components, n_pixels)``.
        explained_variance: Per-component variance captured.
    """

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, whiten=True)
        self.mean_face = None
        self.components = None
        self.explained_variance = None

    def fit(self, face_vectors):
        """Fit the eigenspace on a matrix of training face vectors.

        Args:
            face_vectors: Array of shape ``(n_samples, n_pixels)``.
                Each row is a flattened, normalized face image.

        Returns:
            self: For method chaining.
        """
        face_vectors = np.asarray(face_vectors, dtype=np.float64)
        self.pca.fit(face_vectors)

        # Cache key attributes for easy access.
        self.mean_face = self.pca.mean_.copy()
        self.components = self.pca.components_.copy()
        self.explained_variance = self.pca.explained_variance_.copy()

        return self

    def transform(self, face_vectors, k: int | None = None):
        """Project face vectors into eigenspace coefficients.

        Args:
            face_vectors: Array of shape ``(n_samples, n_pixels)`` or
                ``(n_pixels,)`` for a single face.
            k: Optional. The number of principal components to use for
                the projection. If provided, returns only the first ``k``
                coefficients.

        Returns:
            numpy.ndarray: Coefficients of shape
            ``(n_samples, n_components)`` or ``(n_components,)`` for
            a single face.
        """
        face_vectors = np.asarray(face_vectors, dtype=np.float64)
        single = face_vectors.ndim == 1
        if single:
            face_vectors = face_vectors.reshape(1, -1)

        coefficients = self.pca.transform(face_vectors)
        if k is not None:
            k = min(k, self.n_components)
            coefficients = coefficients[:, :k]

        return coefficients[0] if single else coefficients

    def inverse_transform(self, coefficients, k: int | None = None):
        """Reconstruct face vectors from eigenspace coefficients.

        Args:
            coefficients: Array of shape ``(n_samples, n_components)``
                or ``(n_components,)`` for a single face.
            k: Optional. If the provided coefficients represent fewer
                components than the fitted model (e.g., if they were
                truncated using ``transform(..., k=k)``), they will be
                zero-padded to reconstruct the face using only the first
                ``k`` eigenfaces.

        Returns:
            numpy.ndarray: Reconstructed face vectors.
        """
        coefficients = np.asarray(coefficients, dtype=np.float64)
        single = coefficients.ndim == 1
        if single:
            coefficients = coefficients.reshape(1, -1)

        # Pad with zeros if coefficients are truncated (e.g., length k < n_components)
        current_len = coefficients.shape[1]
        target_len = self.pca.n_components_
        if current_len < target_len:
            pad_width = target_len - current_len
            coefficients = np.pad(coefficients, ((0, 0), (0, pad_width)), mode='constant')

        reconstructed = self.pca.inverse_transform(coefficients)

        return reconstructed[0] if single else reconstructed

    @property
    def explained_variance_ratio(self):
        """Per-component explained variance ratio (sums to ≤ 1.0).

        Useful for scree plots showing how much variance each
        eigenface captures.
        """
        return self.pca.explained_variance_ratio_

    def get_eigenface(self, index, image_shape=None):
        """Get the i-th eigenface (principal component).

        Args:
            index: Component index (0 = most important).
            image_shape: If provided, reshape the component back to
                2D for visualization.

        Returns:
            numpy.ndarray: The eigenface vector (1D) or image (2D).
        """
        eigenface = self.components[index]
        if image_shape is not None:
            eigenface = eigenface.reshape(image_shape)
        return eigenface
