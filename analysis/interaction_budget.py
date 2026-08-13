"""Interaction budget analysis utilities."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.bootstrap import bootstrap_mean_ci


def compute_gap_closure(
    fvu_plain: float, fvu_pairwise: float, fvu_ssae: float
) -> dict[str, float | None]:
    """Compute pairwise improvement, h19 improvement, and gap closed.

    pairwise_improvement = (plain - pw) / plain
    h_improvement = (plain - ssae) / plain
    gap_closed = (plain - pw) / (plain - ssae)

    If abs(plain - ssae) < 1e-12, gap_closed is None.
    """
    pairwise_improvement = (fvu_plain - fvu_pairwise) / fvu_plain if fvu_plain != 0 else 0.0
    h_improvement = (fvu_plain - fvu_ssae) / fvu_plain if fvu_plain != 0 else 0.0

    denom = fvu_plain - fvu_ssae
    if abs(denom) < 1e-12:
        gap_closed = None
    else:
        gap_closed = (fvu_plain - fvu_pairwise) / denom

    return {
        "pairwise_improvement": pairwise_improvement,
        "h_improvement": h_improvement,
        "gap_closed": gap_closed,
    }


def load_baseline_metrics(path: Path) -> dict:
    """Load a baseline_metrics.json file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def bootstrap_paired_diff(
    a: list[float], b: list[float], *, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Bootstrap CI for paired difference (a[i] - b[i]).

    Returns (mean_diff, lo, hi).
    """
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    diff = [a[i] - b[i] for i in range(len(a))]
    return bootstrap_mean_ci(diff, n_boot=n_boot, seed=seed, alpha=alpha)


def win_rate(a: list[float], b: list[float]) -> float:
    """Fraction where a[i] < b[i]."""
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    wins = sum(1 for i in range(len(a)) if a[i] < b[i])
    return wins / len(a)


def summarize_interaction_budget(
    baseline_metrics: dict,
    ssae_fvu_values: dict[str, float] | None = None,
) -> dict:
    """Summarize interaction budget from baseline_metrics.json dict.

    Returns summary stats including n_train, n_holdout, embedding_dim,
    plain_ridge_fvu, pairwise_ridge_fvu, pairwise_improvement, and
    gap_closure fields if ssae_fvu_values provided.
    """
    summary = {
        "n_train": baseline_metrics["n_train"],
        "n_holdout": baseline_metrics["n_holdout"],
        "embedding_dim": baseline_metrics["embedding_dim"],
        "plain_ridge_fvu": baseline_metrics["plain_ridge"]["metrics"]["fvu"],
        "pairwise_ridge_fvu": baseline_metrics["pairwise_ridge"]["metrics"]["fvu"],
    }

    plain_fvu = summary["plain_ridge_fvu"]
    pairwise_fvu = summary["pairwise_ridge_fvu"]
    summary["pairwise_improvement"] = (
        (plain_fvu - pairwise_fvu) / plain_fvu if plain_fvu != 0 else 0.0
    )

    # pairwise win rate if per-sample available (would need to load per_sample.csv)
    # for now omit unless explicitly computed

    if ssae_fvu_values:
        ssae_fvu = ssae_fvu_values.get("fvu", 0.0)
        gap = compute_gap_closure(plain_fvu, pairwise_fvu, ssae_fvu)
        summary.update(gap)

    return summary
