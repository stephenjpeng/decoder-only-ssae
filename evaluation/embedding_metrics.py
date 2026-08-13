"""
Embedding reconstruction metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EmbeddingMetricResult:
    mse_mean: float
    cosine_mean: float
    target_variance_sum: float
    fvu: float
    r2: float
    per_sample_mse: list[float]
    per_sample_cosine: list[float]


def embedding_metrics(pred: torch.Tensor, target: torch.Tensor) -> EmbeddingMetricResult:
    """
    Compute holdout metrics matching the E1 FVU definition.

    Uses EXACTLY this FVU formula:
    - target_variance_sum = target.var(dim=0, unbiased=False).sum()
    - mse_per_sample = ((target - pred) ** 2).sum(dim=1)  # shape (n,)
    - mean_squared_error_per_sample_sum = mse_per_sample.mean()
    - fvu = mean_squared_error_per_sample_sum / target_variance_sum
    - r2 = 1.0 - fvu
    """
    # FVU formula from E1
    target_variance_sum = target.var(dim=0, unbiased=False).sum()
    mse_per_sample = ((target - pred) ** 2).sum(dim=1)  # shape (n,)
    mean_squared_error_per_sample_sum = mse_per_sample.mean()
    fvu = mean_squared_error_per_sample_sum / target_variance_sum
    r2 = 1.0 - fvu
    mse_mean = mse_per_sample.mean().item()

    # per-sample cosine similarity
    cosine_per_sample = F.cosine_similarity(pred, target, dim=1)  # shape (n,)
    cosine_mean = cosine_per_sample.mean().item()

    return EmbeddingMetricResult(
        mse_mean=mse_mean,
        cosine_mean=cosine_mean,
        target_variance_sum=target_variance_sum.item(),
        fvu=fvu.item(),
        r2=r2.item(),
        per_sample_mse=mse_per_sample.tolist(),
        per_sample_cosine=cosine_per_sample.tolist(),
    )


def metrics_to_dict(result: EmbeddingMetricResult) -> dict[str, object]:
    """
    Convert to a JSON-serializable dict, omitting per-sample arrays.
    """
    d = asdict(result)
    # remove per-sample lists for clean summary output
    d.pop("per_sample_mse", None)
    d.pop("per_sample_cosine", None)
    return d
