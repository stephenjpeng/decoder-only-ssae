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
    ssae_device: str | None = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    dev_dec = torch.device(ssae_device) if ssae_device else dev

    holdout_folder = ensure_folder_path(holdout_folder)

    decoder, tp, train_dataset = load_decoder_checkpoint(checkpoint_dir, device=str(dev_dec))
    holdout_ds = h5_dataset_for_folder(checkpoint_dir, holdout_folder, device=device)

    decoder.eval()
    decoder = decoder.to(dev_dec)

    model_name = tp["model_name"]
    n_repeat = int(tp["n_repeat"])

    block_means = None
    train_mask = train_dataset.mask_reduced.to(dev_dec)
    if model_name == "model_trainable_inputs":
        block_means = property_block_means_trainable_inputs(
            decoder, train_mask, n_repeat, dev_dec
        )

    mses = []
    squared_error_sums = []
    cosines = []
    target_sum = torch.zeros(holdout_ds.dim_x, dtype=torch.float64, device=dev_dec)
    target_square_sum = torch.zeros_like(target_sum)
    for idx in range(len(holdout_ds)):
        target, mask_row = holdout_ds[idx]
        target = target.unsqueeze(0).to(dev_dec).float()
        mask_row = mask_row.to(dev_dec)
        pred = predict_embedding_compositional(
            decoder,
            model_name,
            mask_row,
            mask_reduced_train=train_mask,
            n_repeat=n_repeat,
            device=dev_dec,
            block_means=block_means,
        )
        squared_error = (pred - target).square()
        mses.append(squared_error.mean().item())
        squared_error_sums.append(squared_error.sum().item())
        cosines.append(
            torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()
        )
        target_double = target.squeeze(0).double()
        target_sum += target_double
        target_square_sum += target_double.square()

    n_holdout = len(holdout_ds)
    target_mean = target_sum / n_holdout
    target_variance_sum = (
        target_square_sum / n_holdout - target_mean.square()
    ).clamp_min(0).sum().item()
    mean_squared_error_sum = float(sum(squared_error_sums) / n_holdout)
    fvu = mean_squared_error_sum / target_variance_sum

    return {
        "checkpoint": str(Path(checkpoint_dir).resolve()),
        "holdout_folder": str(Path(holdout_folder).resolve()),
        "model_name": model_name,
        "n_holdout": n_holdout,
        "mse_mean": float(sum(mses) / n_holdout),
        "cosine_mean": float(sum(cosines) / n_holdout),
        "target_variance_sum": target_variance_sum,
        "fvu": fvu,
        "r2": 1.0 - fvu,
        "per_index_mse": mses,
        "per_index_squared_error_sum": squared_error_sums,
        "per_index_cosine": cosines,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare compositional decoder prediction to holdout embeddings."
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--holdout_folder", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--ssae_device",
        type=str,
        default=None,
        help="Place the SSAE decoder on a different device than --device. Defaults to --device.",
    )
    p.add_argument("--output_json", type=Path, default=None)
    args = p.parse_args()

    out = compositional_holdout_metrics(
        args.checkpoint,
        args.holdout_folder,
        device=args.device,
        ssae_device=args.ssae_device,
    )
    text = json.dumps(out, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
