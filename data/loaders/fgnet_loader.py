"""Loads the FG-NET aging dataset.

Filenames encode subject ID and age, e.g. `001A02.JPG` -> subject "001",
age 2. This metadata is what makes genuine cross-age validation pairs
possible (see scripts/build_eigenspace.py).

Expected layout after manual download (requires a license request -- see
README): data/raw/fgnet/001A02.JPG ...
"""

import re

FILENAME_PATTERN = re.compile(r"(?P<subject_id>\d+)A(?P<age>\d+)", re.IGNORECASE)


def load(root_dir="data/raw/fgnet"):
    """Load all FG-NET images with parsed subject ID and age metadata.

    Args:
        root_dir: Path to the extracted FG-NET dataset.

    Returns:
        tuple[list[numpy.ndarray], list[str], list[int]]: images, subject
        IDs, and ages.
    """
    raise NotImplementedError
