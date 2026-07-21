"""Read-only loaders over existing evaluation.run_image_benchmark output artifacts.

Supports two layouts:

* **Legacy monolithic**: every method's images and metrics live under a single run folder.
* **Cache-backed**: the run holds SSAE artifacts and a ``manifest.json`` pointing at a shared
  baseline cache directory; baseline methods' images and rows are pulled from there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


_NON_METRIC_COLUMNS = ("task", "sample_idx", "method", "prompt")

_METHOD_ORDER = (
    "gt_embed",
    "ssae_compose",
    "mean_arithmetic",
    "ridge_embed",
    "prompt_only",
)


def load_holdout_prompts(holdout_folder: Path) -> list[dict]:
    with open(Path(holdout_folder) / "prompts.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _is_baseline_cache_dir(d: Path) -> bool:
    mf = d / "manifest.json"
    if not mf.exists():
        return False
    try:
        m = json.loads(mf.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return "dataset_id" in m and "baseline_cache_dir" not in m


def discover_runs(results_root: Path) -> list[Path]:
    """Subdirectories with per_sample.csv OR images/, excluding shared baseline caches."""
    root = Path(results_root)
    if not root.is_dir():
        return []
    found: set[Path] = set()
    for p in root.glob("**/per_sample.csv"):
        found.add(p.parent)
    for p in root.glob("**/images"):
        if p.is_dir():
            found.add(p.parent)
    return sorted(d for d in found if not _is_baseline_cache_dir(d))


@dataclass
class RunData:
    label: str
    output_dir: Path
    per_sample: pd.DataFrame
    summary: dict
    methods: list[str] = field(default_factory=list)
    baseline_cache_dir: Path | None = None
    baseline_methods: list[str] = field(default_factory=list)


def _methods_from_disk(output_dir: Path) -> list[str]:
    images_dir = output_dir / "images"
    if not images_dir.is_dir():
        return []
    return sorted(d.name for d in images_dir.iterdir() if d.is_dir())


def _read_manifest(output_dir: Path) -> dict:
    mf = output_dir / "manifest.json"
    if not mf.exists():
        return {}
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(_NON_METRIC_COLUMNS))
    return pd.read_csv(path)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _order_methods(methods: list[str]) -> list[str]:
    seen = list(dict.fromkeys(methods))
    known = [m for m in _METHOD_ORDER if m in seen]
    extra = [m for m in seen if m not in _METHOD_ORDER]
    return known + extra


def _merge_summary(run_summary: dict, cache_summary: dict) -> dict:
    merged = dict(cache_summary)
    for k, v in run_summary.items():
        if k == "per_method" and isinstance(v, dict):
            per = dict(merged.get("per_method") or {})
            per.update(v)
            merged["per_method"] = per
        elif k == "methods" and isinstance(v, list):
            cache_methods = list(cache_summary.get("methods") or [])
            merged["methods"] = _order_methods(list(dict.fromkeys(cache_methods + list(v))))
        else:
            merged[k] = v
    return merged


def load_run(label: str, output_dir: Path) -> RunData:
    output_dir = Path(output_dir)

    manifest = _read_manifest(output_dir)
    cache_dir_raw = manifest.get("baseline_cache_dir")
    baseline_methods: list[str] = list(manifest.get("baseline_methods") or [])
    cache_dir = Path(cache_dir_raw) if cache_dir_raw else None
    if cache_dir is not None and not cache_dir.is_dir():
        cache_dir = None
        baseline_methods = []

    run_csv = _load_csv(output_dir / "per_sample.csv")
    run_summary = _load_json(output_dir / "summary.json")

    if cache_dir is not None:
        cache_csv = _load_csv(cache_dir / "per_sample.csv")
        cache_summary = _load_json(cache_dir / "summary.json")
        if not cache_csv.empty:
            per_sample = pd.concat([run_csv, cache_csv], ignore_index=True, sort=False)
        else:
            per_sample = run_csv
        summary = _merge_summary(run_summary, cache_summary)
    else:
        per_sample = run_csv
        summary = run_summary

    if summary.get("methods"):
        methods = _order_methods(list(summary["methods"]))
    elif not per_sample.empty:
        methods = _order_methods(list(dict.fromkeys(per_sample["method"])))
    else:
        methods = _order_methods(_methods_from_disk(output_dir))

    if cache_dir is not None and baseline_methods:
        methods = _order_methods(list(dict.fromkeys(methods + list(baseline_methods))))

    return RunData(
        label=label,
        output_dir=output_dir,
        per_sample=per_sample,
        summary=summary,
        methods=methods,
        baseline_cache_dir=cache_dir,
        baseline_methods=baseline_methods,
    )


def image_path(run: RunData, method: str, sample_idx: int, variant: str = "post") -> Path:
    subdir = "images" if variant == "post" else f"images_{variant}"
    if run.baseline_cache_dir is not None and method in run.baseline_methods:
        return run.baseline_cache_dir / subdir / method / f"{sample_idx:05d}.png"
    return run.output_dir / subdir / method / f"{sample_idx:05d}.png"


def _variants_in_dir(d: Path) -> list[str]:
    if not d.is_dir():
        return []
    out: list[str] = []
    for entry in sorted(d.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == "images":
            out.append("post")
        elif entry.name.startswith("images_"):
            out.append(entry.name[len("images_"):])
    return out


def available_variants(run: RunData) -> list[str]:
    variants = _variants_in_dir(run.output_dir)
    if run.baseline_cache_dir is not None:
        for v in _variants_in_dir(run.baseline_cache_dir):
            if v not in variants:
                variants.append(v)
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


def _sample_ids_in_dir(root: Path) -> set[int]:
    images_dir = root / "images"
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


def sample_ids_from_images(run: RunData) -> set[int]:
    """sample_idx values inferred from images/*/*.png in run and (if any) cache dir."""
    ids = _sample_ids_in_dir(run.output_dir)
    if run.baseline_cache_dir is not None:
        ids.update(_sample_ids_in_dir(run.baseline_cache_dir))
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
