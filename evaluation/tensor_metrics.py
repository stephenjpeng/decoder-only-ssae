from __future__ import annotations

import torch


def batch_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target, reduction="mean")


def batch_cosine(pred: torch.Tensor, target: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(pred, target, dim=dim).mean()
