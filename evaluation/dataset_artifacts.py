"""Copy truncation / normalization sidecars from train split to holdout (same encoder pipeline)."""

from __future__ import annotations

import shutil
from pathlib import Path


_ARTIFACT_GLOBS = (
    "indices_top_*.json",
    "embds_max*.json",
    "embds_min*.json",
    "pca_mean_top_*.pt",
    "pca_components_top_*.pt",
)


def copy_truncation_artifacts(train_folder: Path | str, holdout_folder: Path | str) -> list[Path]:
    train_folder = Path(train_folder)
    holdout_folder = Path(holdout_folder)
    copied: list[Path] = []
    for pattern in _ARTIFACT_GLOBS:
        for src in train_folder.glob(pattern):
            dst = holdout_folder / src.name
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied
