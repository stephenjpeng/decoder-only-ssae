"""Read-only loaders over existing evaluation.run_image_benchmark output artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


_NON_METRIC_COLUMNS = ("task", "sample_idx", "method", "prompt")


def load_holdout_prompts(holdout_folder: Path) -> list[dict]:
    with open(Path(holdout_folder) / "prompts.json", "r", encoding="utf-8") as f:
        return json.load(f)


def discover_runs(results_root: Path) -> list[Path]:
    """Any subdirectory that contains a per_sample.csv OR an images/ directory."""
    root = Path(results_root)
    if not root.is_dir():
        return []
    found: set[Path] = set()
    for p in root.glob("**/per_sample.csv"):
        found.add(p.parent)
    for p in root.glob("**/images"):
        if p.is_dir():
            found.add(p.parent)
    return sorted(found)


@dataclass
class RunData:
    label: str
    output_dir: Path
    per_sample: pd.DataFrame
    summary: dict
    methods: list[str] = field(default_factory=list)


def _methods_from_disk(output_dir: Path) -> list[str]:
    images_dir = output_dir / "images"
    if not images_dir.is_dir():
        return []
    return sorted(d.name for d in images_dir.iterdir() if d.is_dir())


def load_run(label: str, output_dir: Path) -> RunData:
    output_dir = Path(output_dir)

    csv_path = output_dir / "per_sample.csv"
    if csv_path.exists():
        per_sample = pd.read_csv(csv_path)
    else:
        per_sample = pd.DataFrame(columns=list(_NON_METRIC_COLUMNS))

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    if summary.get("methods"):
        methods = list(summary["methods"])
    elif not per_sample.empty:
        methods = list(dict.fromkeys(per_sample["method"]))
    else:
        methods = _methods_from_disk(output_dir)

    return RunData(
        label=label, output_dir=output_dir, per_sample=per_sample, summary=summary, methods=methods
    )


def image_path(run: RunData, method: str, sample_idx: int, variant: str = "post") -> Path:
    subdir = "images" if variant == "post" else f"images_{variant}"
    return run.output_dir / subdir / method / f"{sample_idx:05d}.png"


def available_variants(run: RunData) -> list[str]:
    if not run.output_dir.is_dir():
        return []
    variants: list[str] = []
    for d in sorted(run.output_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name == "images":
            variants.append("post")
        elif d.name.startswith("images_"):
            variants.append(d.name[len("images_"):])
    return variants


def metrics_for(run: RunData, sample_idx: int, method: str) -> dict:
    if run.per_sample.empty:
        return {}
    rows = run.per_sample[
        (run.per_sample["sample_idx"] == sample_idx) & (run.per_sample["method"] == method)
    ]
    if rows.empty:
        return {}
    row = rows.iloc[0].to_dict()
    return {k: v for k, v in row.items() if pd.notna(v)}


def numeric_metric_columns(run: RunData) -> list[str]:
    if run.per_sample.empty:
        return []
    non_metric = set(_NON_METRIC_COLUMNS)
    numeric = set(run.per_sample.select_dtypes(include="number").columns)
    return [c for c in run.per_sample.columns if c in numeric and c not in non_metric]


def sample_ids_from_images(run: RunData) -> set[int]:
    """sample_idx values inferred from `<output>/images/*/*.png` filenames."""
    images_dir = run.output_dir / "images"
    if not images_dir.is_dir():
        return set()
    ids: set[int] = set()
    for method_dir in images_dir.iterdir():
        if not method_dir.is_dir():
            continue
        for png in method_dir.glob("*.png"):
            try:
                ids.add(int(png.stem))
            except ValueError:
                continue
    return ids


def available_sample_ids(run: RunData) -> set[int]:
    """Union of sample_idx from per_sample.csv and from images/*/*.png filenames."""
    ids: set[int] = set()
    if not run.per_sample.empty and "sample_idx" in run.per_sample.columns:
        ids.update(int(i) for i in run.per_sample["sample_idx"].unique())
    ids.update(sample_ids_from_images(run))
    return ids


def run_progress(run: RunData) -> dict:
    """Coarse progress signal for badging in-progress runs."""
    n_rows = int(len(run.per_sample))
    ids = available_sample_ids(run)
    n_samples = len(ids)
    n_methods = len(run.methods) or 1
    expected_rows = n_samples * n_methods
    complete = bool(expected_rows) and n_rows >= expected_rows
    return {
        "n_rows": n_rows,
        "n_samples": n_samples,
        "n_methods": n_methods,
        "expected_rows": expected_rows,
        "complete": complete,
        "has_csv": (run.output_dir / "per_sample.csv").exists(),
        "has_summary": bool(run.summary),
    }


def edit_info_for(run: RunData, sample_idx: int) -> dict:
    """Return the (single) edit choice recorded for a sample across methods.

    Fields: edit_pid, edit_attribute, swap_target_pid, swap_target_attribute, swapped_prompt.
    Empty dict if the run has no edit columns or no row for this sample.
    """
    if run.per_sample.empty:
        return {}
    keys = ("edit_pid", "edit_attribute", "swap_target_pid", "swap_target_attribute", "swapped_prompt")
    present = [k for k in keys if k in run.per_sample.columns]
    if not present:
        return {}
    rows = run.per_sample[run.per_sample["sample_idx"] == sample_idx]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {k: row[k] for k in present if pd.notna(row[k]) and row[k] != ""}
