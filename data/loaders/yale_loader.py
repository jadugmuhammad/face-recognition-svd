"""Loads the (Extended) Yale Face Database B.

Expected layout after manual download (see README):
    data/raw/yale_faces/yaleB01/yaleB01_P00A+000E+00.pgm ...
"""


def load(root_dir="data/raw/yale_faces"):
    """Load all Yale face images.

    Args:
        root_dir: Path to the extracted Yale dataset.

    Returns:
        tuple[list[numpy.ndarray], list[str]]: images and their subject IDs.
    """
    raise NotImplementedError
