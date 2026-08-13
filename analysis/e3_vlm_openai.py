"""Summarise OpenAI VLM judgements for the E3 targeted-editing batch"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

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


def summarise(jsonl_path: Path) -> pd.DataFrame:
    """Return group-level VLM metrics by target, run, method, and variant"""
    groups: dict[tuple[str, str, str, str], dict[str, list[float | None]]] = defaultdict(
        lambda: defaultdict(list)
    )

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out = rec.get("vlm", {})
            key = (
                _target_from_benchmark_dir(rec["benchmark_dir"]),
                _run_from_benchmark_dir(rec["benchmark_dir"]),
                rec["method"],
                rec["variant"],
            )
            groups[key]["match_full_prompt"].append(_safe_float(out.get("match_full_prompt")))
            groups[key]["non_target_preserved"].append(
                _safe_float(out.get("non_target_preserved"))
            )
            groups[key]["edit_attribute_score"].append(
                _score_for_phrase(out, rec.get("edit_attribute", ""))
            )
            groups[key]["swap_target_attribute_score"].append(
                _score_for_phrase(out, rec.get("swap_target_attribute", ""))
            )

    rows = []
    for (target, run, method, variant), metrics in sorted(groups.items()):
        rows.append(
            {
                "target": target,
                "run": run,
                "method": method,
                "variant": variant,
                "n": len(metrics["match_full_prompt"]),
                "match_full_prompt": _mean(metrics["match_full_prompt"]),
                "non_target_preserved": _mean(metrics["non_target_preserved"]),
                "edit_attribute_score": _mean(metrics["edit_attribute_score"]),
                "swap_target_attribute_score": _mean(
                    metrics["swap_target_attribute_score"]
                ),
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
    args = parser.parse_args()

    df = summarise(args.jsonl)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
