"""Load training YAML and H5 datasets for arbitrary folders (train / holdout)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from trainings.config.config import initialise_instance, read_training_params_from_yaml
from trainings.dataloader.dataloader import H5Dataset


def ensure_folder_path(folder: str | Path) -> str:
    s = Path(folder).expanduser().resolve().as_posix()
    if not s.endswith("/"):
        s += "/"
    return s


def load_training_params(checkpoint_dir: Path | str) -> Dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    tp, _ = read_training_params_from_yaml(checkpoint_dir / "params.yaml")
    return tp


def h5_dataset_for_folder(
    checkpoint_dir: Path | str,
    data_folder: Path | str,
    *,
    device: str | None = None,
) -> H5Dataset:
    """
    Build an ``H5Dataset`` using hyperparameters from a trained run's ``params.yaml``,
    but reading embeddings from ``data_folder`` (train or holdout split).

    Truncation indices and min/max JSONs are resolved relative to ``data_folder``
    (they should be copied from the train run or recomputed there).
    """
    tp = load_training_params(checkpoint_dir)
    tp["folder_path"] = ensure_folder_path(data_folder)
    tp["logger"] = None
    if device is not None:
        tp["device"] = device
    return initialise_instance(H5Dataset, tp)


def load_decoder_checkpoint(checkpoint_dir: Path | str, device: str) -> tuple:
    """Return ``(decoder, tp, train_dataset)`` from a run folder (``params.yaml`` + ``model.pt``)."""
    from inference.abstract import SFDInference

    inf = SFDInference(folder_path=checkpoint_dir, device=device)
    return inf.decoder, inf.tp, inf.dataset
