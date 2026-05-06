"""Linear multi-label probe on frozen CLIP image features (attribute presence)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class MultiLabelLinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_labels: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass
class ProbeFitResult:
    model: MultiLabelLinearProbe
    train_loss_final: float


def train_multilabel_probe(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    steps: int = 500,
    lr: float = 1e-2,
    device: str | None = None,
) -> ProbeFitResult:
    """
    ``X`` (N, d), ``Y`` (N, L) binary multi-label targets in {0,1}.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    X = X.to(dev).float()
    Y = Y.to(dev).float()
    n, d = X.shape
    _, l = Y.shape
    model = MultiLabelLinearProbe(d, l).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    last = 0.0
    for _ in range(steps):
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, Y)
        loss.backward()
        opt.step()
        last = loss.item()
    model.eval()
    return ProbeFitResult(model=model, train_loss_final=last)


@torch.no_grad()
def probe_micro_metrics(
    model: MultiLabelLinearProbe,
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    threshold: float = 0.5,
    device: str | None = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model = model.to(dev)
    X = X.to(dev).float()
    Y = Y.to(dev).float()
    logits = model(X)
    prob = torch.sigmoid(logits)
    pred = (prob >= threshold).float()
    tp = (pred * Y).sum().item()
    fp = (pred * (1 - Y)).sum().item()
    fn = ((1 - pred) * Y).sum().item()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    acc_label = (pred.eq(Y)).float().mean().item()
    return {
        "micro_precision": prec,
        "micro_recall": rec,
        "micro_f1": f1,
        "per_label_accuracy_mean": acc_label,
    }
