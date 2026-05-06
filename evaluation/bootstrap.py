"""Bootstrap confidence intervals for benchmark means."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Return ``(point_mean, lo, hi)`` for the sample mean using bootstrap percentiles.
    NaNs are dropped.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        m = float(arr[0])
        return m, m, m
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = arr.size
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = arr[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1.0 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)
