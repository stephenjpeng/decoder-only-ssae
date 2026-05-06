"""Train shallow AE on train embeddings; report holdout reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from baselines.unsup_ae import train_shallow_ae
from evaluation.io import ensure_folder_path, h5_dataset_for_folder


@torch.no_grad()
def _stack_X(dataset, device):
    xs = []
    for i in range(len(dataset)):
        x, _ = dataset[i]
        xs.append(x)
    return torch.stack(xs, dim=0).to(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train_folder", type=Path, required=True)
    p.add_argument("--holdout_folder", type=Path, required=True)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_json", type=Path, default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)

    train_ds = h5_dataset_for_folder(args.checkpoint, ensure_folder_path(args.train_folder))
    hold_ds = h5_dataset_for_folder(args.checkpoint, ensure_folder_path(args.holdout_folder))

    X_tr = _stack_X(train_ds, dev)
    X_ho = _stack_X(hold_ds, dev)

    k = min(args.latent_dim, X_tr.shape[1] // 2, max(8, X_tr.shape[0] // 4))
    fit = train_shallow_ae(X_tr, latent_dim=k, steps=args.steps, device=device)
    model = fit.model
    with torch.no_grad():
        pred_h, _ = model(X_ho)
        mse = torch.nn.functional.mse_loss(pred_h, X_ho).item()
        cos = torch.nn.functional.cosine_similarity(pred_h, X_ho, dim=-1).mean().item()

    out = {
        "latent_dim": k,
        "steps": args.steps,
        "train_mse_after_fit": fit.train_mse,
        "holdout_mse": float(mse),
        "holdout_cosine": float(cos),
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
