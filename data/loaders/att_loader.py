"""Loads the AT&T (ORL) face dataset.

Automatically downloads and extracts the dataset if not found.
"""

import os
import re
import sys
import zipfile
import urllib.request

import numpy as np
from PIL import Image

_ATT_URL = "https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip"
_ATT_FILENAME = "att_faces.zip"
_ATT_EXTRACT_DIR = "att_faces"
_SUBJECT_DIR_RE = re.compile(r"^s(\d+)$")

def _download_progress(count, block_size, total_size):
    """Callback for urlretrieve to print download progress."""
    global _last_percent
    percent = int(count * block_size * 100 / total_size)
    
    if percent % 10 == 0 and percent != getattr(_download_progress, "last_percent", -1):
        sys.stdout.write(f"\rDownloading AT&T Dataset... {percent}%")
        sys.stdout.flush()
        _download_progress.last_percent = percent
        if percent >= 100:
            sys.stdout.write("\rDownloading AT&T Dataset... 100% (Done)\n")
            sys.stdout.flush()


def download_and_extract(raw_dir="data/raw"):
    """Downloads and extracts AT&T if it doesn't exist."""
    os.makedirs(raw_dir, exist_ok=True)
    extract_path = os.path.join(raw_dir, _ATT_EXTRACT_DIR)
    
    if os.path.isdir(extract_path) and len(os.listdir(extract_path)) > 0:
        return extract_path
        
    archive_path = os.path.join(raw_dir, _ATT_FILENAME)
    
    if not os.path.isfile(archive_path):
        print("AT&T Dataset not found. Starting download (~4MB)...")
        _download_progress.last_percent = -1
        try:
            urllib.request.urlretrieve(_ATT_URL, archive_path, reporthook=_download_progress)
        except Exception as e:
            if os.path.exists(archive_path):
                os.remove(archive_path)
            raise RuntimeError(f"Failed to download AT&T dataset: {e}")
            
    print("Extracting AT&T Dataset...")
    try:
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
    except Exception as e:
        raise RuntimeError(f"Failed to extract AT&T dataset: {e}")
        
    print("Extraction complete.")
    return extract_path


def load(root_dir="data/raw", split="both"):
    """Load AT&T face images.

    Args:
        root_dir: The directory where `data/raw` is located.
        split: Which partition to load — ``"train"`` (images 1-9), ``"test"`` (image 10), or
            ``"both"`` (default).

    Returns:
        tuple[list[numpy.ndarray], list[str]]: A pair of
        ``(images, subject_ids)`` where every image is a 2-D
        grayscale ``uint8`` array and the corresponding subject ID
        is a string like ``"s01"``.
    """
    valid_splits = ("train", "test", "both")
    if split not in valid_splits:
        raise ValueError(
            f"split must be one of {valid_splits}, got {split!r}"
        )

    extract_path = download_and_extract(root_dir)

    images: list[np.ndarray] = []
    subject_ids: list[str] = []

    for entry in sorted(os.listdir(extract_path)):
        if not _SUBJECT_DIR_RE.match(entry):
            continue
            
        subject_path = os.path.join(extract_path, entry)
        if not os.path.isdir(subject_path):
            continue

        subject_num = int(_SUBJECT_DIR_RE.match(entry).group(1))
        subject_id = f"s{subject_num:02d}"

        for img_file in sorted(os.listdir(subject_path)):
            if not img_file.lower().endswith(".pgm"):
                continue
                
            # Each subject has 1.pgm to 10.pgm
            # We assign 1-9 to train, 10 to test
            img_num = int(os.path.splitext(img_file)[0])
            is_test = (img_num == 10)
            
            if split == "train" and is_test:
                continue
            if split == "test" and not is_test:
                continue
                
            img_path = os.path.join(subject_path, img_file)
            img = Image.open(img_path).convert("L")
            images.append(np.array(img, dtype=np.uint8))
            subject_ids.append(subject_id)

    return images, subject_ids
