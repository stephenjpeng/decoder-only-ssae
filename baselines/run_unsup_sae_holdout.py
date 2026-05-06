"""Train sparse bottleneck AE on train embeddings; score holdout reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from baselines.unsup_sparse_ae import train_sparse_ae
from evaluation.io import ensure_folder_path, h5_dataset_for_folder


@torch.no_grad()
def _stack_X(dataset, device):
    xs = []
    for i in range(len(dataset)):
        x, _ = dataset[i]
        xs.append(x)
    return torch.stack(xs, dim=0).to(device)


def main() -> None:
    p = argparse.ArgumentParser(description="Unsupervised sparse AE baseline (MSE + L1 latent).")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train_folder", type=Path, required=True)
    p.add_argument("--holdout_folder", type=Path, required=True)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--l1_weight", type=float, default=1e-3)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_json", type=Path, default=None)
    p.add_argument("--save_model", type=Path, default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)

    train_ds = h5_dataset_for_folder(args.checkpoint, ensure_folder_path(args.train_folder))
    hold_ds = h5_dataset_for_folder(args.checkpoint, ensure_folder_path(args.holdout_folder))

    X_tr = _stack_X(train_ds, dev)
    X_ho = _stack_X(hold_ds, dev)

    k = min(args.latent_dim, max(16, X_tr.shape[1] // 4))
    fit = train_sparse_ae(
        X_tr,
        latent_dim=k,
        steps=args.steps,
        l1_weight=args.l1_weight,
        device=device,
    )
    model = fit.model
    with torch.no_grad():
        pred_h, _ = model(X_ho)
        mse = torch.nn.functional.mse_loss(pred_h, X_ho).item()
        cos = torch.nn.functional.cosine_similarity(pred_h, X_ho, dim=-1).mean().item()

    out = {
        "latent_dim": k,
        "steps": args.steps,
        "l1_weight": args.l1_weight,
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
    if args.save_model is not None:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "meta": out}, args.save_model)


if __name__ == "__main__":
    main()
