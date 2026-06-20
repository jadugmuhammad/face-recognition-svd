"""Eigenfaces: PCA/SVD-based face decomposition, built on scikit-learn."""

from sklearn.decomposition import PCA


class Eigenfaces:
    """Wraps sklearn.decomposition.PCA for face decomposition tasks.

    Provides:
        - fit(): build the eigenspace from a training set of face vectors
        - transform(): project a face vector into eigenspace coefficients
        - inverse_transform(): reconstruct a face from its coefficients
    """

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, face_vectors):
        """Fit the eigenspace on a matrix of training face vectors.

        Args:
            face_vectors: Array of shape (n_samples, n_pixels).
        """
        raise NotImplementedError

    def transform(self, face_vectors):
        """Project face vectors into eigenspace coefficients."""
        raise NotImplementedError

    def inverse_transform(self, coefficients):
        """Reconstruct face vectors from eigenspace coefficients."""
        raise NotImplementedError
