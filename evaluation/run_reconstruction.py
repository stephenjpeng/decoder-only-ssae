"""SSAE reconstruction metrics on the training dataset (index-aligned with ``decoder.Y``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.io import load_decoder_checkpoint


@torch.no_grad()
def reconstruction_metrics(
    checkpoint_dir: Path | str,
    *,
    device: str | None = None,
) -> dict:
    """
    Per-prompt reconstruction using ``decoder(batch_size=1, batch_idx=idx)``.

    Uses the dataset path in ``params.yaml`` (must match the run used to train ``model.pt``).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    decoder, tp, dataset = load_decoder_checkpoint(checkpoint_dir, device=device)

    decoder.eval()
    decoder = decoder.to(dev)

    mses = []
    cosines = []
    for idx in range(len(dataset)):
        target, _ = dataset[idx]
        target = target.unsqueeze(0).to(dev).float()
        pred = decoder(batch_size=1, batch_idx=idx)
        mses.append(torch.nn.functional.mse_loss(pred, target, reduction="mean").item())
        cosines.append(
            torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()
        )

    return {
        "n_prompts": len(dataset),
        "mse_mean": float(sum(mses) / len(mses)),
        "cosine_mean": float(sum(cosines) / len(cosines)),
        "per_index_mse": mses,
        "per_index_cosine": cosines,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="MSE/cosine reconstruction for each training prompt index."
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_json", type=Path, default=None)
    args = p.parse_args()

    out = reconstruction_metrics(args.checkpoint, device=args.device)
    text = json.dumps(out, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
