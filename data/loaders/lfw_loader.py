"""Loads the Labeled Faces in the Wild (LFW) dataset.

Automatically downloads and extracts the dataset if not found.
"""

import os
import sys
import tarfile
import urllib.request

import numpy as np
from PIL import Image

_LFW_URL = "https://ndownloader.figshare.com/files/5976015"
_LFW_FILENAME = "lfw-funneled.tgz"
_LFW_EXTRACT_DIR = "lfw_funneled"

def _download_progress(count, block_size, total_size):
    """Callback for urlretrieve to print download progress."""
    global _last_percent
    percent = int(count * block_size * 100 / total_size)
    
    # Only print every 5% to avoid spamming the console
    if percent % 5 == 0 and percent != getattr(_download_progress, "last_percent", -1):
        sys.stdout.write(f"\rDownloading LFW Dataset... {percent}%")
        sys.stdout.flush()
        _download_progress.last_percent = percent
        if percent >= 100:
            sys.stdout.write("\rDownloading LFW Dataset... 100% (Done)\n")
            sys.stdout.flush()


def download_and_extract(raw_dir="data/raw"):
    """Downloads and extracts LFW if it doesn't exist."""
    os.makedirs(raw_dir, exist_ok=True)
    extract_path = os.path.join(raw_dir, _LFW_EXTRACT_DIR)
    
    if os.path.isdir(extract_path):
        return extract_path
        
    archive_path = os.path.join(raw_dir, _LFW_FILENAME)
    
    if not os.path.isfile(archive_path):
        print("LFW Dataset not found. Starting download (~233MB)...")
        _download_progress.last_percent = -1
        try:
            urllib.request.urlretrieve(_LFW_URL, archive_path, reporthook=_download_progress)
        except Exception as e:
            if os.path.exists(archive_path):
                os.remove(archive_path)
            raise RuntimeError(f"Failed to download LFW dataset: {e}")
            
    print("Extracting LFW Dataset... this may take a moment.")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=raw_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to extract LFW dataset: {e}")
        
    print("Extraction complete.")
    return extract_path

def load(root_dir="data/raw", min_faces=2):
    """Load LFW face images.
    
    Args:
        root_dir: The directory where `data/raw` is located.
        min_faces: Minimum number of images a subject must have to be included.
                   Using min_faces >= 2 ensures we have genuine pairs for calibration.

    Returns:
        tuple[list[numpy.ndarray], list[str]]: A pair of
        ``(images, subject_ids)`` where every image is a 2-D
        grayscale ``uint8`` array and the corresponding subject ID
        is a string like ``"George_W_Bush"``.
    """
    extract_path = download_and_extract(root_dir)
    
    images: list[np.ndarray] = []
    subject_ids: list[str] = []
    
    # Iterate over subject folders
    for entry in sorted(os.listdir(extract_path)):
        subject_path = os.path.join(extract_path, entry)
        if not os.path.isdir(subject_path):
            continue
            
        img_files = sorted([f for f in os.listdir(subject_path) if f.lower().endswith(".jpg")])
        if len(img_files) < min_faces:
            continue
            
        for img_file in img_files:
            img_path = os.path.join(subject_path, img_file)
            img = Image.open(img_path).convert("L")
            images.append(np.array(img, dtype=np.uint8))
            subject_ids.append(entry)

    return images, subject_ids
