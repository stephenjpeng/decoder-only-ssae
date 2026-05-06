"""Unsupervised sparse bottleneck autoencoder (MSE + L1 on latent)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SparseAEFitResult:
    model: nn.Module
    train_mse: float


class SparseAE(nn.Module):
    def __init__(self, dim_in: int, latent_dim: int):
        super().__init__()
        self.enc = nn.Linear(dim_in, latent_dim)
        self.dec = nn.Linear(latent_dim, dim_in)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.enc(x))
        return self.dec(h), h


def train_sparse_ae(
    X: torch.Tensor,
    *,
    latent_dim: int,
    steps: int = 3000,
    lr: float = 1e-3,
    l1_weight: float = 1e-3,
    device: str | None = None,
) -> SparseAEFitResult:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    X = X.to(dev)
    dim_in = X.shape[1]
    model = SparseAE(dim_in, latent_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        recon, h = model(X)
        loss = torch.nn.functional.mse_loss(recon, X) + l1_weight * h.abs().mean()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        recon, _ = model(X)
        mse = torch.nn.functional.mse_loss(recon, X).item()
    return SparseAEFitResult(model=model, train_mse=float(mse))
