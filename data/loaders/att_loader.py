"""Loads the AT&T (ORL) face dataset.

Expected layout after manual download (see README):
    data/raw/att_faces/s1/1.pgm ... s40/10.pgm
"""


def load(root_dir="data/raw/att_faces"):
    """Load all AT&T face images.

    Args:
        root_dir: Path to the extracted AT&T dataset.

    Returns:
        tuple[list[numpy.ndarray], list[str]]: images and their subject IDs.
    """
    raise NotImplementedError
