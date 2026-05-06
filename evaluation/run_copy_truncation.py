"""Copy truncation / normalization JSON sidecars from train split to holdout."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.dataset_artifacts import copy_truncation_artifacts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_folder", type=Path, required=True)
    p.add_argument("--holdout_folder", type=Path, required=True)
    args = p.parse_args()
    copied = copy_truncation_artifacts(args.train_folder, args.holdout_folder)
    for c in copied:
        print(c)


if __name__ == "__main__":
    main()
