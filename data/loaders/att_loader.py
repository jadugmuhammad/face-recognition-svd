"""Loads the AT&T (ORL) face dataset.

Expected layout after manual download (see README):
    data/raw/att_faces/Training/s1/ ... s40/  (9 images each)
    data/raw/att_faces/Testing/s1/  ... s40/  (1 image each)
"""

import os
import re

import numpy as np
from PIL import Image

# Matches subject folder names like "s1", "s23", etc.
_SUBJECT_DIR_RE = re.compile(r"^s(\d+)$")


def load(root_dir="data/raw/att_faces", split="both"):
    """Load AT&T face images.

    The dataset is expected to be pre-split into ``Training/`` and
    ``Testing/`` subdirectories, each containing per-subject folders
    (``s1/``, ``s2/``, …) with ``.pgm`` images.

    Args:
        root_dir: Path to the extracted AT&T dataset root.
        split: Which partition to load — ``"train"``, ``"test"``, or
            ``"both"`` (default).

    Returns:
        tuple[list[numpy.ndarray], list[str]]: A pair of
        ``(images, subject_ids)`` where every image is a 2-D
        grayscale ``uint8`` array and the corresponding subject ID
        is a string like ``"s01"``.

    Raises:
        FileNotFoundError: If *root_dir* (or the requested split
            subdirectory) does not exist.
        ValueError: If *split* is not one of the accepted values.
    """
    valid_splits = ("train", "test", "both")
    if split not in valid_splits:
        raise ValueError(
            f"split must be one of {valid_splits}, got {split!r}"
        )

    root_dir = os.path.normpath(root_dir)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"AT&T root directory not found: {root_dir}")

    # Determine which subdirectories to scan.
    dirs_to_scan: list[str] = []
    if split in ("train", "both"):
        train_dir = os.path.join(root_dir, "Training")
        if os.path.isdir(train_dir):
            dirs_to_scan.append(train_dir)
        else:
            raise FileNotFoundError(
                f"Training directory not found: {train_dir}"
            )
    if split in ("test", "both"):
        test_dir = os.path.join(root_dir, "Testing")
        if os.path.isdir(test_dir):
            dirs_to_scan.append(test_dir)
        else:
            raise FileNotFoundError(
                f"Testing directory not found: {test_dir}"
            )

    images: list[np.ndarray] = []
    subject_ids: list[str] = []

    for parent_dir in dirs_to_scan:
        # Iterate over subject folders in sorted order for determinism.
        for entry in sorted(os.listdir(parent_dir)):
            if not _SUBJECT_DIR_RE.match(entry):
                continue  # skip non-subject entries (e.g. TestData.txt)
            subject_path = os.path.join(parent_dir, entry)
            if not os.path.isdir(subject_path):
                continue

            # Normalize subject ID to zero-padded form (e.g. "s01").
            subject_num = int(_SUBJECT_DIR_RE.match(entry).group(1))
            subject_id = f"s{subject_num:02d}"

            for img_file in sorted(os.listdir(subject_path)):
                if not img_file.lower().endswith(".pgm"):
                    continue
                img_path = os.path.join(subject_path, img_file)
                img = Image.open(img_path).convert("L")
                images.append(np.array(img, dtype=np.uint8))
                subject_ids.append(subject_id)

    return images, subject_ids
