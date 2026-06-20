"""Illumination normalization and vectorization of aligned face images."""


def normalize(face_image):
    """Apply histogram equalization for illumination invariance.

    Args:
        face_image: Aligned grayscale face image.

    Returns:
        numpy.ndarray: Equalized image, same shape as input.
    """
    raise NotImplementedError


def flatten(face_image):
    """Flatten a 2D face image into a 1D feature vector.

    Args:
        face_image: 2D grayscale image.

    Returns:
        numpy.ndarray: 1D vector of length height * width.
    """
    raise NotImplementedError
