"""Rotates and scales a face crop so both eyes sit on a horizontal line at a
consistent position -- this is what makes Eigenfaces tolerant to head tilt
and varying photo framing.
"""


def align_face(face_image, eye_positions, output_size=(100, 100)):
    """Align a face crop based on detected eye positions.

    Args:
        face_image: Cropped face image as a numpy array.
        eye_positions: List of two (x, y) eye center coordinates.
        output_size: Target (width, height) of the aligned output.

    Returns:
        numpy.ndarray: Rotated, scaled, and cropped face image.
    """
    raise NotImplementedError
