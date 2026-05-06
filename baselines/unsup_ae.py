"""Train a shallow bottleneck autoencoder (MSE + optional L1 on latent)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class UnsupAEFitResult:
    model: nn.Module
    train_mse: float


class ShallowAE(nn.Module):
    def __init__(self, dim_in: int, latent_dim: int):
        super().__init__()
        self.enc = nn.Linear(dim_in, latent_dim)
        self.dec = nn.Linear(latent_dim, dim_in)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.dec(torch.relu(h)), h


def train_shallow_ae(
    X: torch.Tensor,
    *,
    latent_dim: int,
    steps: int = 2000,
    lr: float = 1e-3,
    l1_weight: float = 1e-4,
    device: str | None = None,
) -> UnsupAEFitResult:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    X = X.to(dev)
    dim_in = X.shape[1]
    model = ShallowAE(dim_in, latent_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    last_loss = 0.0
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
    return UnsupAEFitResult(model=model, train_mse=float(mse))
