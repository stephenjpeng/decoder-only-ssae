"""Summarise OpenAI VLM judgements for the E3 targeted-editing batch"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd


def _target_from_benchmark_dir(path: str) -> str:
    bench = Path(path)
    if bench.parent.parent.name == "bench_e3_cache":
        return bench.parent.name
    return bench.name.split("_", 1)[0]


def _run_from_benchmark_dir(path: str) -> str:
    bench = Path(path)
    if bench.parent.parent.name == "bench_e3_cache":
        return "baselines"
    return bench.name.split("_", 1)[1]


def _score_for_phrase(out: dict[str, Any], phrase: str) -> float | None:
    if not phrase:
        return None
    for item in out.get("attribute_scores", []):
        if item.get("phrase") == phrase:
            value = item.get("score")
            return None if value is None else float(value)
    return None


def _safe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _mean(values: list[float | None]) -> float | None:
    vals = [value for value in values if value is not None]
    return None if not vals else mean(vals)


def _load_flat_records(jsonl_path: Path) -> pd.DataFrame:
    """Load one VLM JSONL row per image into scalar columns"""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out = rec.get("vlm", {})
            rows.append(
                {
                    "target": _target_from_benchmark_dir(rec["benchmark_dir"]),
                    "run": _run_from_benchmark_dir(rec["benchmark_dir"]),
                    "method": rec["method"],
                    "variant": rec["variant"],
                    "sample_idx": rec["sample_idx"],
                    "benchmark_dir": rec["benchmark_dir"],
                    "image": rec["image"],
                    "prompt_judged": rec["prompt_judged"],
                    "edit_attribute": rec.get("edit_attribute", ""),
                    "swap_target_attribute": rec.get("swap_target_attribute", ""),
                    "match_full_prompt": _safe_float(out.get("match_full_prompt")),
                    "non_target_preserved": _safe_float(
                        out.get("non_target_preserved")
                    ),
                    "edit_attribute_score": _score_for_phrase(
                        out, rec.get("edit_attribute", "")
                    ),
                    "swap_target_attribute_score": _score_for_phrase(
                        out, rec.get("swap_target_attribute", "")
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarise(jsonl_path: Path) -> pd.DataFrame:
    """Return group-level VLM metrics by target, run, method, and variant"""
    df = _load_flat_records(jsonl_path)
    metrics = [
        "match_full_prompt",
        "non_target_preserved",
        "edit_attribute_score",
        "swap_target_attribute_score",
    ]
    out = (
        df.groupby(["target", "run", "method", "variant"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    counts = (
        df.groupby(["target", "run", "method", "variant"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    return counts.merge(out, on=["target", "run", "method", "variant"])


def _bootstrap_ci(values: np.ndarray, *, n_boot: int = 5000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def paired_diffs(jsonl_path: Path) -> pd.DataFrame:
    """Compare VLM scores on matched target/sample/variant rows"""
    df = _load_flat_records(jsonl_path)
    df["method_label"] = np.where(
        df["method"].eq("ssae_compose"), df["run"].map(lambda run: f"ssae_{run}"), df["method"]
    )
    metrics = [
        "match_full_prompt",
        "non_target_preserved",
        "edit_attribute_score",
        "swap_target_attribute_score",
    ]
    comparisons = [
        ("ssae_L1", "ridge_embed"),
        ("ssae_L2_h2048", "ridge_embed"),
        ("ssae_L1", "prompt_only"),
        ("ssae_L2_h2048", "prompt_only"),
        ("ssae_L2_h2048", "ssae_L1"),
    ]
    rows = []
    for target in sorted(df["target"].unique()):
        for variant in ["deleted", "swapped", "post"]:
            scoped = df[(df["target"].eq(target)) & (df["variant"].eq(variant))]
            wide = scoped.pivot_table(
                index="sample_idx",
                columns="method_label",
                values=metrics,
                aggfunc="first",
            )
            for left, right in comparisons:
                if left not in wide.columns.get_level_values(1):
                    continue
                if right not in wide.columns.get_level_values(1):
                    continue
                for metric in metrics:
                    if (metric, left) not in wide.columns or (metric, right) not in wide.columns:
                        continue
                    diffs = (wide[(metric, left)] - wide[(metric, right)]).dropna().to_numpy()
                    if len(diffs) == 0:
                        continue
                    ci_low, ci_high = _bootstrap_ci(diffs)
                    rows.append(
                        {
                            "target": target,
                            "variant": variant,
                            "left_method": left,
                            "right_method": right,
                            "metric": metric,
                            "mean_diff_left_minus_right": float(diffs.mean()),
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "n": int(len(diffs)),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("results/vlm_e3_full/e3_vlm.jsonl"),
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("results/analysis/e3_vlm_openai/e3_vlm_by_group.csv"),
    )
    parser.add_argument(
        "--paired_out_csv",
        type=Path,
        default=Path("results/analysis/e3_vlm_openai/e3_vlm_paired_diffs.csv"),
    )
    args = parser.parse_args()

    df = summarise(args.jsonl)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    diffs = paired_diffs(args.jsonl)
    diffs.to_csv(args.paired_out_csv, index=False)

    print(df.to_string(index=False))
    print("\npaired differences")
    print(diffs.to_string(index=False))


if __name__ == "__main__":
    main()
