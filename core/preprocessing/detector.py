"""Face and eye detection using OpenCV Haar Cascades.

Used only to locate the face region for cropping/alignment -- the
identification decision itself is handled entirely by PCA/SVD in
core.decomposition and core.matching.
"""


def detect_face(image):
    """Detect the largest face in an image.

    Args:
        image: Grayscale image as a numpy array.

    Returns:
        tuple[int, int, int, int] | None: (x, y, width, height) bounding box
        of the detected face, or None if no face was found.
    """
    raise NotImplementedError


def detect_eyes(face_image):
    """Detect eye positions within an already-cropped face image.

    Args:
        face_image: Grayscale, cropped face image as a numpy array.

    Returns:
        list[tuple[int, int]]: Detected eye center coordinates.
    """
    raise NotImplementedError
