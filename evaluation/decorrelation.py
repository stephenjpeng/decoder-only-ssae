"""Mean off-diagonal cosine similarity between concept sub-vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.composition import property_block_means_trainable_inputs
from evaluation.io import load_decoder_checkpoint


@torch.no_grad()
def _cosine_matrix(rows: torch.Tensor) -> torch.Tensor:
    """rows: (n, d) -> (n, n) pairwise cosine."""
    rows = torch.nn.functional.normalize(rows, dim=1)
    return rows @ rows.T


@torch.no_grad()
def concept_subvector_cosine_stats(
    checkpoint_dir: Path | str,
    *,
    device: str | None = None,
    include_full_matrix: bool = False,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    decoder, tp, train_dataset = load_decoder_checkpoint(checkpoint_dir, device=device)
    decoder = decoder.to(dev)
    model_name = tp["model_name"]
    n_repeat = int(tp["n_repeat"])
    n_props = train_dataset.mask_reduced.shape[1]

    if model_name == "model_trainable_inputs":
        block_means = property_block_means_trainable_inputs(
            decoder, train_dataset.mask_reduced.to(dev), n_repeat, dev
        )
        rows = block_means
    elif model_name == "model_avg_feature":
        w = decoder.Y.weight[1:]  # drop padding row; shape (n_props, n_repeat)
        rows = w
    else:
        raise ValueError(f"Unsupported model for decorrelation: {model_name}")

    cm = _cosine_matrix(rows.float())
    mask_off = ~torch.eye(n_props, dtype=torch.bool, device=cm.device)
    off = cm[mask_off]
    out = {
        "model_name": model_name,
        "n_properties": int(n_props),
        "mean_off_diagonal_cosine": float(off.mean().item()),
        "std_off_diagonal_cosine": float(off.std().item()),
        "min_off_diagonal_cosine": float(off.min().item()),
        "max_off_diagonal_cosine": float(off.max().item()),
    }
    if include_full_matrix:
        out["cosine_matrix"] = cm.cpu().tolist()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Concept sub-vector cosine correlation matrix.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--include_matrix",
        action="store_true",
        help="Include full cosine matrix in JSON (can be large).",
    )
    p.add_argument("--output_json", type=Path, default=None)
    args = p.parse_args()
    out = concept_subvector_cosine_stats(args.checkpoint, device=args.device, include_full_matrix=args.include_matrix)
    text = json.dumps(out, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
