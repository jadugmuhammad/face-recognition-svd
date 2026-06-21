"""Face and eye detection using OpenCV Haar Cascades.

Used only to locate the face region for cropping/alignment -- the
identification decision itself is handled entirely by PCA/SVD in
core.decomposition and core.matching.
"""

import cv2
import numpy as np

# Lazy-loaded cascade classifiers (loaded once, cached as module globals).
_face_cascade = None
_eye_cascade = None


def _get_face_cascade():
    """Load (or return cached) frontal-face Haar Cascade."""
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            raise RuntimeError(
                f"Failed to load face cascade from {cascade_path}"
            )
    return _face_cascade


def _get_eye_cascade():
    """Load (or return cached) eye Haar Cascade."""
    global _eye_cascade
    if _eye_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        _eye_cascade = cv2.CascadeClassifier(cascade_path)
        if _eye_cascade.empty():
            raise RuntimeError(
                f"Failed to load eye cascade from {cascade_path}"
            )
    return _eye_cascade


def detect_face(image):
    """Detect the largest face in an image.

    Args:
        image: Grayscale image as a numpy array (``uint8``).

    Returns:
        tuple[int, int, int, int] | None: ``(x, y, width, height)``
        bounding box of the detected face, or ``None`` if no face
        was found.
    """
    cascade = _get_face_cascade()

    # Try with default parameters first, then relax if nothing found.
    for scale_factor, min_neighbors in [(1.1, 5), (1.05, 3), (1.1, 2)]:
        faces = cascade.detectMultiScale(
            image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) > 0:
            break

    if len(faces) == 0:
        return None

    # Return the largest detected face (by area).
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    return (int(x), int(y), int(w), int(h))


def detect_eyes(face_image):
    """Detect eye positions within an already-cropped face image.

    Args:
        face_image: Grayscale, cropped face image as a numpy array.

    Returns:
        list[tuple[int, int]] | None: Two ``(x, y)`` eye center
        coordinates sorted left-to-right, or ``None`` if fewer than
        two eyes were detected (after fallback attempts).
    """
    cascade = _get_eye_cascade()
    h, w = face_image.shape[:2]

    # Search only in the upper half of the face (eyes are typically there).
    search_region = face_image[0 : int(h * 0.65), :]

    # Try with progressively relaxed parameters.
    eyes = None
    for scale_factor, min_neighbors, min_size_ratio in [
        (1.1, 5, 0.08),
        (1.05, 3, 0.06),
        (1.1, 2, 0.05),
    ]:
        min_eye_size = (int(w * min_size_ratio), int(w * min_size_ratio))
        detected = cascade.detectMultiScale(
            search_region,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_eye_size,
        )
        if len(detected) >= 2:
            eyes = detected
            break

    if eyes is None or len(eyes) < 2:
        return None

    # Compute center of each detected eye region.
    eye_centers = []
    for (ex, ey, ew, eh) in eyes:
        center_x = ex + ew // 2
        center_y = ey + eh // 2
        eye_centers.append((int(center_x), int(center_y)))

    # Sort by x-coordinate (left to right) and take the two most
    # separated ones (to avoid picking the same eye twice).
    eye_centers.sort(key=lambda e: e[0])

    if len(eye_centers) == 2:
        return eye_centers

    # If more than 2 eyes detected, pick the leftmost and rightmost.
    return [eye_centers[0], eye_centers[-1]]
