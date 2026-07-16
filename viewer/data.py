"""Read-only loaders over existing evaluation.run_image_benchmark output artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


def load_holdout_prompts(holdout_folder: Path) -> list[dict]:
    with open(Path(holdout_folder) / "prompts.json", "r", encoding="utf-8") as f:
        return json.load(f)


def discover_runs(results_root: Path) -> list[Path]:
    root = Path(results_root)
    if not root.is_dir():
        return []
    return sorted({p.parent for p in root.glob("**/per_sample.csv")})


@dataclass
class RunData:
    label: str
    output_dir: Path
    per_sample: pd.DataFrame
    summary: dict
    methods: list[str] = field(default_factory=list)


def load_run(label: str, output_dir: Path) -> RunData:
    output_dir = Path(output_dir)
    per_sample = pd.read_csv(output_dir / "per_sample.csv")

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    methods = summary.get("methods") or list(dict.fromkeys(per_sample["method"]))
    return RunData(label=label, output_dir=output_dir, per_sample=per_sample, summary=summary, methods=methods)


def image_path(run: RunData, method: str, sample_idx: int) -> Path:
    return run.output_dir / "images" / method / f"{sample_idx:05d}.png"


def metrics_for(run: RunData, sample_idx: int, method: str) -> dict:
    rows = run.per_sample[
        (run.per_sample["sample_idx"] == sample_idx) & (run.per_sample["method"] == method)
    ]
    if rows.empty:
        return {}
    row = rows.iloc[0].to_dict()
    return {k: v for k, v in row.items() if pd.notna(v)}


def numeric_metric_columns(run: RunData) -> list[str]:
    non_metric = {"task", "sample_idx", "method", "prompt"}
    return [c for c in run.per_sample.columns if c not in non_metric]
