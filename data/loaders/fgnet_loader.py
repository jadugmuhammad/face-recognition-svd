"""Loads the FG-NET aging dataset.

Filenames encode subject ID and age, e.g. ``001A02.JPG`` -> subject "001",
age 2. This metadata is what makes genuine cross-age validation pairs
possible (see scripts/build_eigenspace.py).

Expected layout after manual download (requires a license request -- see
README):
    data/raw/fgnet/images/001A02.JPG ...
    data/raw/fgnet/points/001a02.pts ...  (68-point landmarks)
"""

import os
import re

import numpy as np
from PIL import Image

# Matches filenames like "001A02.JPG", "001A43a.JPG", "009a16A.dat".
# Groups: subject_id (digits), age (digits), optional suffix (letters).
FILENAME_PATTERN = re.compile(
    r"(?P<subject_id>\d+)[Aa](?P<age>\d+)(?P<suffix>[a-zA-Z])?",
    re.IGNORECASE,
)


def load(root_dir="data/raw/fgnet"):
    """Load all FG-NET images with parsed subject ID and age metadata.

    Args:
        root_dir: Path to the extracted FG-NET dataset root.  The
            function expects an ``images/`` subdirectory containing
            ``.JPG`` files.

    Returns:
        tuple[list[numpy.ndarray], list[str], list[int]]: A triple of
        ``(images, subject_ids, ages)`` where every image is a 2-D
        grayscale ``uint8`` array, subject IDs are zero-padded strings
        like ``"001"``, and ages are integers.

    Raises:
        FileNotFoundError: If the images subdirectory does not exist.
    """
    images_dir = os.path.join(os.path.normpath(root_dir), "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"FG-NET images directory not found: {images_dir}"
        )

    images: list[np.ndarray] = []
    subject_ids: list[str] = []
    ages: list[int] = []

    for filename in sorted(os.listdir(images_dir)):
        filepath = os.path.join(images_dir, filename)
        if not os.path.isfile(filepath):
            continue

        match = FILENAME_PATTERN.match(os.path.splitext(filename)[0])
        if match is None:
            continue  # skip files that don't match the pattern

        try:
            img = Image.open(filepath).convert("L")
        except Exception:
            continue

        images.append(np.array(img, dtype=np.uint8))
        subject_ids.append(match.group("subject_id").zfill(3))
        ages.append(int(match.group("age")))

    return images, subject_ids, ages


def load_landmarks(root_dir="data/raw/fgnet"):
    """Load FG-NET 68-point facial landmarks from ``.pts`` files.

    Each ``.pts`` file has the format::

        version: 1
        n_points: 68
        {
        x1 y1
        x2 y2
        ...
        }

    The landmark indices follow a standard 68-point scheme where
    points 36–41 are the left eye and 42–47 are the right eye.

    Args:
        root_dir: Path to the FG-NET dataset root.  Expects a
            ``points/`` subdirectory.

    Returns:
        dict[str, numpy.ndarray]: Mapping from image stem (e.g.
        ``"001a02"``) to a ``(68, 2)`` float array of ``(x, y)``
        landmark coordinates.

    Raises:
        FileNotFoundError: If the points subdirectory does not exist.
    """
    points_dir = os.path.join(os.path.normpath(root_dir), "points")
    if not os.path.isdir(points_dir):
        raise FileNotFoundError(
            f"FG-NET points directory not found: {points_dir}"
        )

    landmarks: dict[str, np.ndarray] = {}

    for filename in sorted(os.listdir(points_dir)):
        if not filename.lower().endswith(".pts"):
            continue
        filepath = os.path.join(points_dir, filename)

        stem = os.path.splitext(filename)[0].lower()
        points = _parse_pts_file(filepath)
        if points is not None:
            landmarks[stem] = points

    return landmarks


def _parse_pts_file(filepath):
    """Parse a single ``.pts`` landmark file.

    Args:
        filepath: Absolute path to the ``.pts`` file.

    Returns:
        numpy.ndarray | None: A ``(n_points, 2)`` float array, or
        ``None`` if the file could not be parsed.
    """
    try:
        with open(filepath, "r") as f:
            lines = f.read().strip().splitlines()
    except OSError:
        return None

    # Clean up carriage returns.
    lines = [line.strip() for line in lines]

    points = []
    inside_block = False
    for line in lines:
        if line == "{":
            inside_block = True
            continue
        if line == "}":
            break
        if inside_block:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    points.append((x, y))
                except ValueError:
                    continue

    if not points:
        return None
    return np.array(points, dtype=np.float64)


def get_eye_positions_from_landmarks(landmarks_array):
    """Extract left and right eye center positions from 68-point landmarks.

    Uses the standard 68-point landmark convention:
        - Left eye:  points 36–41
        - Right eye: points 42–47

    Args:
        landmarks_array: A ``(68, 2)`` array of landmark coordinates.

    Returns:
        list[tuple[float, float]]: Two ``(x, y)`` tuples representing
        the center of the left eye and the center of the right eye.
    """
    left_eye = landmarks_array[36:42].mean(axis=0)
    right_eye = landmarks_array[42:48].mean(axis=0)
    return [
        (float(left_eye[0]), float(left_eye[1])),
        (float(right_eye[0]), float(right_eye[1])),
    ]
