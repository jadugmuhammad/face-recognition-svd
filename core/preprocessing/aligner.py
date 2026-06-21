"""Rotates and scales a face crop so both eyes sit on a horizontal line at a
consistent position -- this is what makes Eigenfaces tolerant to head tilt
and varying photo framing.
"""

import cv2
import numpy as np


# Default desired eye positions as fractions of the output image size.
# Left eye at ~30% from left, right eye at ~70% from left, both at ~35%
# from top.  This leaves room for forehead and chin.
_DEFAULT_LEFT_EYE_POS = (0.30, 0.35)
_DEFAULT_RIGHT_EYE_POS = (0.70, 0.35)


def align_face(face_image, eye_positions, output_size=(100, 100)):
    """Align a face crop based on detected eye positions.

    The transformation rotates the image so both eyes are horizontal,
    scales so that the inter-eye distance is consistent, and translates
    so that the eyes land at pre-defined positions within the output
    frame.

    This function works with eye positions from **any** source — Haar
    Cascade ``detect_eyes()``, FG-NET landmark-based eye centers, or
    manually specified coordinates.

    Args:
        face_image: Grayscale face image as a numpy array (the full
            image or a face crop — as long as *eye_positions* are
            relative to this image's coordinate system).
        eye_positions: List of two ``(x, y)`` eye center coordinates.
            The first should be the left eye (from the subject's
            perspective, i.e. the one on the **right** side of the
            image), and the second the right eye.  If the order is
            uncertain, the function will sort them left-to-right
            based on x-coordinate.
        output_size: Target ``(width, height)`` of the aligned output.

    Returns:
        numpy.ndarray: Rotated, scaled, and cropped face image of
        shape ``(height, width)`` with ``uint8`` dtype.
    """
    out_w, out_h = output_size

    # Ensure eye_positions are floats.
    left_eye = np.array(eye_positions[0], dtype=np.float64)
    right_eye = np.array(eye_positions[1], dtype=np.float64)

    # Sort by x so left_eye is truly the one with smaller x.
    if left_eye[0] > right_eye[0]:
        left_eye, right_eye = right_eye, left_eye

    # --- Compute the affine transformation ---

    # Desired eye positions in the output image.
    desired_left = np.array(
        [_DEFAULT_LEFT_EYE_POS[0] * out_w, _DEFAULT_LEFT_EYE_POS[1] * out_h]
    )
    desired_right = np.array(
        [_DEFAULT_RIGHT_EYE_POS[0] * out_w, _DEFAULT_RIGHT_EYE_POS[1] * out_h]
    )

    # Angle between the two eyes.
    delta = right_eye - left_eye
    angle = np.degrees(np.arctan2(delta[1], delta[0]))

    # Distance between eyes (current vs desired).
    dist_current = np.linalg.norm(delta)
    dist_desired = np.linalg.norm(desired_right - desired_left)

    if dist_current < 1e-6:
        # Eyes are at the same position — can't align.  Just resize.
        return cv2.resize(face_image, (out_w, out_h))

    scale = dist_desired / dist_current

    # Center of rotation = midpoint between the two eyes.
    eyes_center = ((left_eye + right_eye) / 2).astype(np.float64)

    # Rotation matrix (rotation + scale around eyes_center).
    M = cv2.getRotationMatrix2D(
        (float(eyes_center[0]), float(eyes_center[1])),
        float(angle),
        float(scale),
    )

    # Adjust the translation so that the eyes_center in the output
    # lands at the desired midpoint.
    desired_center = (desired_left + desired_right) / 2
    M[0, 2] += desired_center[0] - eyes_center[0]
    M[1, 2] += desired_center[1] - eyes_center[1]

    # Apply the affine transformation.
    aligned = cv2.warpAffine(
        face_image,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return aligned.astype(np.uint8)
