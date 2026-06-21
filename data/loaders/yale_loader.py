"""Loads the (Extended) Yale Face Database B.

Expected layout after manual download (see README):
    data/raw/yale_faces/Training/s01/ ... s15/  (11 images each)
    data/raw/yale_faces/Testing/s01/  ... s15/  (1 image each)

All files are GIF images internally (even those without a ``.gif``
extension, like ``subject01.normal``).  PIL detects format from the
file header, so extension does not matter.

Deduplication: files like ``subject01.glasses`` and
``subject01.glasses.gif`` are byte-identical — only one is kept.
"""

import os
import re

import numpy as np
from PIL import Image

_SUBJECT_DIR_RE = re.compile(r"^s(\d+)$")


def _extract_condition(filename):
    """Extract the lighting/expression condition from a Yale filename.

    Examples:
        ``"subject01.normal"``     → ``"normal"``
        ``"subject01.glasses.gif"`` → ``"glasses"``
        ``"subject01.gif"``        → ``"centerlight"``  (the default shot)

    Returns:
        str: condition label.
    """
    # Strip any .gif suffix first.
    base = filename
    if base.lower().endswith(".gif"):
        base = base[:-4]

    # Pattern: "subjectNN.condition" or just "subjectNN"
    parts = base.split(".", 1)
    if len(parts) == 2 and parts[1]:
        return parts[1].lower()
    # Bare "subjectNN.gif" → conventionally the center-light image.
    return "centerlight"


def _deduplicate_files(filenames):
    """Remove duplicates where 'X' and 'X.gif' both exist.

    When both ``subject01.glasses`` and ``subject01.glasses.gif`` are
    present, keep only the one **without** the ``.gif`` extension
    (arbitrary but deterministic choice).

    Returns:
        list[str]: deduplicated, sorted list of filenames.
    """
    # Build a set of base names (strip .gif) to detect collisions.
    base_to_files: dict[str, list[str]] = {}
    for fn in filenames:
        base = fn[:-4] if fn.lower().endswith(".gif") else fn
        base_to_files.setdefault(base, []).append(fn)

    result = []
    for base in sorted(base_to_files):
        candidates = base_to_files[base]
        if len(candidates) == 1:
            result.append(candidates[0])
        else:
            # Prefer the version without .gif extension.
            non_gif = [c for c in candidates if not c.lower().endswith(".gif")]
            result.append(non_gif[0] if non_gif else candidates[0])
    return sorted(result)


def load(root_dir="data/raw/yale_faces", split="both"):
    """Load Yale face images.

    Args:
        root_dir: Path to the extracted Yale dataset root.
        split: Which partition to load — ``"train"``, ``"test"``, or
            ``"both"`` (default).

    Returns:
        tuple[list[numpy.ndarray], list[str], list[str]]: A triple of
        ``(images, subject_ids, conditions)`` where every image is a
        2-D grayscale ``uint8`` array, subject IDs are strings like
        ``"s01"``, and conditions describe lighting/expression
        (e.g. ``"normal"``, ``"happy"``, ``"leftlight"``).

    Raises:
        FileNotFoundError: If *root_dir* or the requested split
            subdirectory does not exist.
        ValueError: If *split* is not one of the accepted values.
    """
    valid_splits = ("train", "test", "both")
    if split not in valid_splits:
        raise ValueError(
            f"split must be one of {valid_splits}, got {split!r}"
        )

    root_dir = os.path.normpath(root_dir)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Yale root directory not found: {root_dir}")

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
    conditions: list[str] = []

    for parent_dir in dirs_to_scan:
        for entry in sorted(os.listdir(parent_dir)):
            if not _SUBJECT_DIR_RE.match(entry):
                continue
            subject_path = os.path.join(parent_dir, entry)
            if not os.path.isdir(subject_path):
                continue

            subject_num = int(_SUBJECT_DIR_RE.match(entry).group(1))
            subject_id = f"s{subject_num:02d}"

            # List all files and deduplicate.
            all_files = [
                f for f in os.listdir(subject_path)
                if os.path.isfile(os.path.join(subject_path, f))
            ]
            unique_files = _deduplicate_files(all_files)

            for img_file in unique_files:
                img_path = os.path.join(subject_path, img_file)
                try:
                    img = Image.open(img_path).convert("L")
                except Exception:
                    continue  # skip files that PIL cannot read
                images.append(np.array(img, dtype=np.uint8))
                subject_ids.append(subject_id)
                conditions.append(_extract_condition(img_file))

    return images, subject_ids, conditions
