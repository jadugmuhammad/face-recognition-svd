"""Illumination normalization and vectorization of aligned face images."""

import cv2
import numpy as np


def normalize(face_image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    for illumination invariance.

    CLAHE is preferred over plain ``equalizeHist`` because it avoids
    over-amplifying noise in homogeneous regions while still improving
    local contrast — important for faces under uneven lighting (e.g.
    Yale's extreme left/right light conditions).

    Args:
        face_image: Aligned grayscale face image (``uint8``).

    Returns:
        numpy.ndarray: Equalized image, same shape and dtype as input.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(face_image)


def flatten(face_image):
    """Flatten a 2D face image into a 1D feature vector.

    The pixel values are converted to ``float64`` and scaled to
    ``[0, 1]``.  This normalization ensures that PCA operates on
    values in a consistent range regardless of the original bit depth.

    Args:
        face_image: 2D grayscale image (``uint8`` or ``float``).

    Returns:
        numpy.ndarray: 1D vector of length ``height * width``,
        dtype ``float64``, with values in ``[0, 1]``.
    """
    return face_image.astype(np.float64).ravel() / 255.0
