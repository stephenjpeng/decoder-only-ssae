"""Fit simple baselines on training embeddings and score holdout MSE/cosine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.io import ensure_folder_path, h5_dataset_for_folder


@torch.no_grad()
def _stack_dataset_tensors(dataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ms = []
    for i in range(len(dataset)):
        x, m = dataset[i]
        xs.append(x)
        ms.append(m.float())
    return torch.stack(xs, dim=0).to(device), torch.stack(ms, dim=0).to(device)


def fit_mean_arithmetic(
    X: torch.Tensor, M: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(mu, deltas)`` with ``mu`` (1, d), ``deltas`` (n_props, d)."""
    mu = X.mean(dim=0, keepdim=True)
    n_props = M.shape[1]
    deltas = []
    for p in range(n_props):
        pos = M[:, p] > 0.5
        neg = ~pos
        if pos.any() and neg.any():
            deltas.append(X[pos].mean(0) - X[neg].mean(0))
        elif pos.any():
            deltas.append(X[pos].mean(0) - mu.squeeze(0))
        else:
            deltas.append(torch.zeros(X.shape[1], device=X.device, dtype=X.dtype))
    return mu, torch.stack(deltas, dim=0)


def predict_mean_arithmetic(mu: torch.Tensor, deltas: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return mu + (M.unsqueeze(-1) * deltas.unsqueeze(0)).sum(dim=1)


def fit_ridge(M: torch.Tensor, X: torch.Tensor, lam: float) -> torch.Tensor:
    """Ridge weights W (m, d) with intercept: append column of ones to M.

    Uses float64 internally for the linear solve to avoid numerical singularity
    on large or ill-conditioned design matrices, then casts back to input dtype.
    """
    orig_dtype = M.dtype
    M = M.double()
    X = X.double()
    ones = torch.ones(M.shape[0], 1, device=M.device, dtype=torch.float64)
    M_aug = torch.cat([M, ones], dim=1)
    m = M_aug.shape[1]
    g = M_aug.T @ M_aug + lam * torch.eye(m, device=M.device, dtype=torch.float64)
    rhs = M_aug.T @ X
    return torch.linalg.solve(g, rhs).to(orig_dtype)


def predict_linear(M: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(M.shape[0], 1, device=M.device, dtype=M.dtype)
    M_aug = torch.cat([M, ones], dim=1)
    return M_aug @ W


def fit_pca(X: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean (1, d) and orthonormal basis P (d, k) for PCA reconstruction."""
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    _, _, vt = torch.linalg.svd(Xc, full_matrices=False)
    P = vt.T[:, :k]
    return mu, P


def predict_pca(X: torch.Tensor, mu: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    Xc = X - mu
    coeff = Xc @ P
    return mu + coeff @ P.T


def metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    mse = torch.nn.functional.mse_loss(pred, target, reduction="mean").item()
    cos = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()
    return {"mse_mean": float(mse), "cosine_mean": float(cos)}


def main() -> None:
    p = argparse.ArgumentParser(description="Mean-arithmetic, ridge, and PCA baselines.")
    p.add_argument("--checkpoint", type=Path, required=True, help="For params.yaml + device settings.")
    p.add_argument("--train_folder", type=Path, required=True)
    p.add_argument("--holdout_folder", type=Path, required=True)
    p.add_argument("--ridge_lambda", type=float, default=1e-2)
    p.add_argument("--pca_dim", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_json", type=Path, default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)

    train_ds = h5_dataset_for_folder(args.checkpoint, ensure_folder_path(args.train_folder))
    hold_ds = h5_dataset_for_folder(args.checkpoint, ensure_folder_path(args.holdout_folder))

    X_tr, M_tr = _stack_dataset_tensors(train_ds, dev)
    X_ho, M_ho = _stack_dataset_tensors(hold_ds, dev)

    mu, deltas = fit_mean_arithmetic(X_tr, M_tr)
    pred_ma = predict_mean_arithmetic(mu, deltas, M_ho)
    out_ma = metrics(pred_ma, X_ho)

    W = fit_ridge(M_tr, X_tr, args.ridge_lambda)
    pred_r = predict_linear(M_ho, W)
    out_r = metrics(pred_r, X_ho)

    k = min(args.pca_dim, X_tr.shape[1], X_tr.shape[0])
    mu_p, P = fit_pca(X_tr, k)
    pred_p = predict_pca(X_ho, mu_p, P)
    out_p = metrics(pred_p, X_ho)

    result = {
        "train_folder": str(Path(args.train_folder).resolve()),
        "holdout_folder": str(Path(args.holdout_folder).resolve()),
        "mean_arithmetic": out_ma,
        "ridge": {**out_r, "lambda": args.ridge_lambda},
        "pca": {**out_p, "k": k},
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
