"""Leave-pair-out analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trainings.utils.run_manifest import write_run_manifest


def load_split_metrics(
    baseline_metrics_path: Path,
    ssae_metric_paths: list[Path],
) -> dict:
    """Load baseline_metrics.json and SSAE metric JSONs.

    Returns dict with:
        fvu_plain, fvu_pairwise, fvu_ssae_seeds, fvu_ssae_mean, fvu_ssae_sd
    """
    with open(baseline_metrics_path, encoding="utf-8") as f:
        baseline = json.load(f)

    fvu_plain = baseline["plain_ridge"]["metrics"]["fvu"]
    fvu_pairwise = baseline["pairwise_ridge"]["metrics"]["fvu"]

    fvu_ssae_seeds = []
    for path in ssae_metric_paths:
        with open(path, encoding="utf-8") as f:
            met = json.load(f)
        fvu_ssae_seeds.append(met["fvu"])

    fvu_ssae_mean = float(np.mean(fvu_ssae_seeds)) if fvu_ssae_seeds else 0.0
    fvu_ssae_sd = float(np.std(fvu_ssae_seeds, ddof=1)) if len(fvu_ssae_seeds) > 1 else 0.0

    return {
        "fvu_plain": fvu_plain,
        "fvu_pairwise": fvu_pairwise,
        "fvu_ssae_seeds": fvu_ssae_seeds,
        "fvu_ssae_mean": fvu_ssae_mean,
        "fvu_ssae_sd": fvu_ssae_sd,
    }


def split_contrast(pair_metrics: dict, random_metrics: dict) -> dict:
    """Compute h19_gain_contraction and pairwise_gain_contraction.

    Positive improvement = lower FVU.
    h19_improvement = (fvu_plain - fvu_h19) / fvu_plain
    pairwise_improvement = (fvu_plain - fvu_pairwise) / fvu_plain
    gap_closed = (fvu_plain - fvu_pairwise) / (fvu_plain - fvu_h19)
    """
    pair_plain = pair_metrics["fvu_plain"]
    pair_pw = pair_metrics["fvu_pairwise"]
    pair_h19 = pair_metrics["fvu_ssae_mean"]

    random_plain = random_metrics["fvu_plain"]
    random_pw = random_metrics["fvu_pairwise"]
    random_h19 = random_metrics["fvu_ssae_mean"]

    # h19 gain contraction
    pair_h19_improvement = (pair_plain - pair_h19) / pair_plain if pair_plain != 0 else 0.0
    random_h19_improvement = (random_plain - random_h19) / random_plain if random_plain != 0 else 0.0
    h19_gain_contraction = pair_h19_improvement / random_h19_improvement if random_h19_improvement != 0 else None

    # pairwise gain contraction
    pair_pw_improvement = (pair_plain - pair_pw) / pair_plain if pair_plain != 0 else 0.0
    random_pw_improvement = (random_plain - random_pw) / random_plain if random_plain != 0 else 0.0
    pairwise_gain_contraction = pair_pw_improvement / random_pw_improvement if random_pw_improvement != 0 else None

    # gap closed (pair only)
    denom = pair_plain - pair_h19
    if abs(denom) < 1e-12:
        pair_gap_closed = None
    else:
        pair_gap_closed = (pair_plain - pair_pw) / denom

    # gap closed (random only)
    denom = random_plain - random_h19
    if abs(denom) < 1e-12:
        random_gap_closed = None
    else:
        random_gap_closed = (random_plain - random_pw) / denom

    return {
        "pair_h19_improvement": pair_h19_improvement,
        "random_h19_improvement": random_h19_improvement,
        "h19_gain_contraction": h19_gain_contraction,
        "pair_pairwise_improvement": pair_pw_improvement,
        "random_pairwise_improvement": random_pw_improvement,
        "pairwise_gain_contraction": pairwise_gain_contraction,
        "pair_gap_closed": pair_gap_closed,
        "random_gap_closed": random_gap_closed,
    }


def run_analysis(
    slug: str,
    pair_baseline: Path,
    random_baseline: Path,
    pair_ssae_metrics: list[Path],
    random_ssae_metrics: list[Path],
    output_dir: Path,
) -> dict:
    """Run full leave-pair-out analysis.

    Writes summary.json and run_manifest.json. Returns summary dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_met = load_split_metrics(pair_baseline, pair_ssae_metrics)
    random_met = load_split_metrics(random_baseline, random_ssae_metrics)

    contrast = split_contrast(pair_met, random_met)

    summary = {
        "slug": slug,
        "pair": pair_met,
        "random": random_met,
        "contrast": contrast,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_run_manifest(
        output_dir,
        run_kind="leave_pair_out",
        config={"slug": slug},
    )

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Leave-pair-out analysis.")
    p.add_argument("--slug", required=True)
    p.add_argument("--pair_baseline", type=Path, required=True)
    p.add_argument("--random_baseline", type=Path, required=True)
    p.add_argument("--pair_ssae", type=Path, action="append", default=[])
    p.add_argument("--random_ssae", type=Path, action="append", default=[])
    p.add_argument("--output_dir", type=Path, required=True)
    args = p.parse_args()

    run_analysis(
        slug=args.slug,
        pair_baseline=args.pair_baseline,
        random_baseline=args.random_baseline,
        pair_ssae_metrics=args.pair_ssae,
        random_ssae_metrics=args.random_ssae,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
