"""CLI script: load datasets -> preprocess -> fit Eigenfaces -> calibrate
threshold -> save artifacts to artifacts/.

Usage:
    python scripts/build_eigenspace.py
"""

import os
import sys

# Make the project root importable when run as `python scripts/...py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Run the full eigenspace build + calibration pipeline.

    Planned steps:
        1. Load AT&T, Yale, and FG-NET via data.loaders.*
        2. Hold out ~10-15% of subjects per dataset for calibration only
           (not used to fit the eigenspace, to avoid biased thresholds).
        3. Preprocess every image via core.preprocessing.*
        4. Fit core.decomposition.Eigenfaces on the remaining training set.
        5. Build genuine/impostor pairs from the holdout set (FG-NET pairs
           give true cross-age genuine pairs).
        6. Compute ROC/EER per metric via core.matching.threshold.
        7. Save mean_face, components, explained_variance, and calibration
           stats into artifacts/eigenspace.npz and artifacts/calibration.json.
    """
    raise NotImplementedError(
        "Implementasi setelah data/loaders, core/preprocessing, dan "
        "core/decomposition siap."
    )


if __name__ == "__main__":
    main()
