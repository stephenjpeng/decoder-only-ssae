"""Compositional embedding prediction vs holdout ground truth (normalized space)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.composition import (
    property_block_means_trainable_inputs,
    predict_embedding_compositional,
)
from evaluation.io import ensure_folder_path, h5_dataset_for_folder, load_decoder_checkpoint


@torch.no_grad()
def compositional_holdout_metrics(
    checkpoint_dir: Path | str,
    holdout_folder: Path | str,
    *,
    device: str | None = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    holdout_folder = ensure_folder_path(holdout_folder)

    decoder, tp, train_dataset = load_decoder_checkpoint(checkpoint_dir, device=device)
    holdout_ds = h5_dataset_for_folder(checkpoint_dir, holdout_folder, device=device)

    decoder.eval()
    decoder = decoder.to(dev)

    model_name = tp["model_name"]
    n_repeat = int(tp["n_repeat"])

    block_means = None
    train_mask = train_dataset.mask_reduced.to(dev)
    if model_name == "model_trainable_inputs":
        block_means = property_block_means_trainable_inputs(
            decoder, train_mask, n_repeat, dev
        )

    mses = []
    cosines = []
    for idx in range(len(holdout_ds)):
        target, mask_row = holdout_ds[idx]
        target = target.unsqueeze(0).to(dev).float()
        mask_row = mask_row.to(dev)
        pred = predict_embedding_compositional(
            decoder,
            model_name,
            mask_row,
            mask_reduced_train=train_mask,
            n_repeat=n_repeat,
            device=dev,
            block_means=block_means,
        )
        mses.append(torch.nn.functional.mse_loss(pred, target, reduction="mean").item())
        cosines.append(
            torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()
        )

    return {
        "checkpoint": str(Path(checkpoint_dir).resolve()),
        "holdout_folder": str(Path(holdout_folder).resolve()),
        "model_name": model_name,
        "n_holdout": len(holdout_ds),
        "mse_mean": float(sum(mses) / len(mses)),
        "cosine_mean": float(sum(cosines) / len(cosines)),
        "per_index_mse": mses,
        "per_index_cosine": cosines,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare compositional decoder prediction to holdout embeddings."
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--holdout_folder", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_json", type=Path, default=None)
    args = p.parse_args()

    out = compositional_holdout_metrics(args.checkpoint, args.holdout_folder, device=args.device)
    text = json.dumps(out, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
