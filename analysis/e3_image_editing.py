"""Analyse E3 targeted editing and magnitude-sensitivity outputs

Reads only small CSV/JSON artifacts produced by the rendering jobs. Images stay on S3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.bootstrap import bootstrap_mean_ci

CONCEPT_LABELS = {
    "gun": "holding a gun",
    "hat": "and a hat",
    "beach": "at the beach",
}

METHOD_ORDER = [
    "prompt_only",
    "ridge_embed",
    "mean_arithmetic",
    "linear_probe_direction",
    "ssae_L1",
    "ssae_L2_h2048",
]

E3_METRICS = [
    "clip_normal_vs_target_phrase",
    "clip_deleted_vs_target_phrase",
    "delete_delta_target",
    "clip_image_vs_residual_prompt",
    "ssim_pre_post_edit",
    "mse_pixel_pre_post_edit",
    "clip_swapped_vs_replacement_phrase",
    "clip_swapped_vs_target_phrase",
    "swap_gain_replacement_over_target",
    "clip_swap_image_vs_swapped_prompt",
    "ssim_swap_vs_normal",
    "mse_pixel_swap_vs_normal",
]

MAG_METRICS = [
    "clip_image_vs_concept",
    "clip_mean_vs_other_attrs",
    "clip_min_vs_other_attrs",
]


def _stable_seed(*parts: object) -> int:
    """Deterministic 32-bit seed from analysis labels"""
    data = "\u241f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big")


def _ci(values: pd.Series, *, seed: int = 0) -> dict[str, float | int]:
    mean, lo, hi = bootstrap_mean_ci(values.to_numpy(), seed=seed)
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": int(values.notna().sum())}


def load_e3_rows(root: Path) -> pd.DataFrame:
    """Load full-strength E3 rows from run folders and shared baseline caches"""
    rows: list[pd.DataFrame] = []

    for path in sorted((root / "bench_e3_cache").glob("*/*/per_sample.csv")):
        concept = path.parts[-3]
        df = pd.read_csv(path)
        df["concept_key"] = concept
        df["checkpoint"] = "baseline"
        rows.append(df)

    for path in sorted((root / "bench_e3").glob("*/per_sample.csv")):
        run_name = path.parent.name
        if run_name.endswith("_L2_h2048"):
            concept = run_name.removesuffix("_L2_h2048")
            checkpoint = "L2_h2048"
        elif run_name.endswith("_L1"):
            concept = run_name.removesuffix("_L1")
            checkpoint = "L1"
        else:
            raise ValueError(f"cannot parse E3 run name: {run_name}")

        df = pd.read_csv(path)
        df["concept_key"] = concept
        df["checkpoint"] = checkpoint
        df["method"] = f"ssae_{checkpoint}"
        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"no E3 per_sample.csv files found under {root}")

    all_rows = pd.concat(rows, ignore_index=True)
    all_rows["target_concept"] = all_rows["concept_key"].map(CONCEPT_LABELS)
    all_rows["delete_delta_target"] = (
        all_rows["clip_normal_vs_target_phrase"]
        - all_rows["clip_deleted_vs_target_phrase"]
    )
    all_rows["swap_gain_replacement_over_target"] = (
        all_rows["clip_swapped_vs_replacement_phrase"]
        - all_rows["clip_swapped_vs_target_phrase"]
    )
    return all_rows


def summarise_e3(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarise full-strength E3 rows by concept and method"""
    records: list[dict[str, object]] = []
    for (concept, method), group in rows.groupby(["concept_key", "method"]):
        record: dict[str, object] = {
            "concept": concept,
            "target_concept": CONCEPT_LABELS.get(concept, concept),
            "method": method,
            "n": int(group["sample_idx"].nunique()),
        }
        for metric in E3_METRICS:
            stats = _ci(group[metric], seed=_stable_seed(concept, method, metric))
            for key, value in stats.items():
                record[f"{metric}_{key}"] = value
        records.append(record)
    out = pd.DataFrame(records)
    out["method_rank"] = out["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
    return out.sort_values(["concept", "method_rank", "method"]).drop(columns="method_rank")


def load_e3_probe_rows(root: Path) -> pd.DataFrame:
    """Load probe-intervention E3 rows from bench_e3_probe run folders"""
    rows: list[pd.DataFrame] = []
    probe_root = root / "bench_e3_probe"
    if not probe_root.is_dir():
        return pd.DataFrame()
    for path in sorted(probe_root.glob("*_probe/per_sample.csv")):
        run_name = path.parent.name
        concept = run_name.removesuffix("_probe")
        df = pd.read_csv(path)
        df["concept_key"] = concept
        df["method"] = "linear_probe_direction"
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["target_concept"] = out["concept_key"].map(CONCEPT_LABELS)
    if "delete_delta_target" not in out.columns and "clip_normal_vs_target_phrase" in out.columns:
        out["delete_delta_target"] = (
            out["clip_normal_vs_target_phrase"] - out["clip_deleted_vs_target_phrase"]
        )
    if "swap_gain_replacement_over_target" not in out.columns and "clip_swapped_vs_replacement_phrase" in out.columns:
        out["swap_gain_replacement_over_target"] = (
            out["clip_swapped_vs_replacement_phrase"] - out["clip_swapped_vs_target_phrase"]
        )
    return out


def paired_method_diffs(rows: pd.DataFrame) -> pd.DataFrame:
    """Paired method differences for the main E3 comparisons"""
    comparisons = [
        ("ssae_L1", "ridge_embed"),
        ("ssae_L2_h2048", "ridge_embed"),
        ("ssae_L2_h2048", "prompt_only"),
        ("ssae_L2_h2048", "ssae_L1"),
        ("ssae_L2_h2048", "linear_probe_direction"),
        ("linear_probe_direction", "prompt_only"),
        ("linear_probe_direction", "ridge_embed"),
    ]
    metrics = [
        "delete_delta_target",
        "clip_image_vs_residual_prompt",
        "ssim_pre_post_edit",
        "mse_pixel_pre_post_edit",
        "swap_gain_replacement_over_target",
        "clip_swap_image_vs_swapped_prompt",
        "ssim_swap_vs_normal",
        "mse_pixel_swap_vs_normal",
    ]

    records: list[dict[str, object]] = []
    for concept, concept_rows in rows.groupby("concept_key"):
        for left, right in comparisons:
            left_rows = concept_rows[concept_rows["method"] == left]
            right_rows = concept_rows[concept_rows["method"] == right]
            if left_rows.empty or right_rows.empty:
                continue
            merged = left_rows.merge(
                right_rows,
                on="sample_idx",
                suffixes=("_left", "_right"),
            )
            for metric in metrics:
                diff = merged[f"{metric}_left"] - merged[f"{metric}_right"]
                stats = _ci(diff, seed=_stable_seed(concept, left, right, metric))
                records.append(
                    {
                        "concept": concept,
                        "left_method": left,
                        "right_method": right,
                        "metric": metric,
                        "mean_diff_left_minus_right": stats["mean"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "n": stats["n"],
                    }
                )
    return pd.DataFrame(records)


def load_magnitude_rows(root: Path) -> pd.DataFrame:
    """Load L1/L2 magnitude curve rows"""
    rows: list[pd.DataFrame] = []
    for path in sorted((root / "magnitude").glob("300k_*/per_sample.csv")):
        run_name = path.parent.name
        checkpoint = run_name.removeprefix("300k_")
        df = pd.read_csv(path)
        df["checkpoint"] = checkpoint
        df["method"] = f"ssae_{checkpoint}"
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"no magnitude per_sample.csv files found under {root}")
    return pd.concat(rows, ignore_index=True)


def summarise_magnitude(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarise magnitude curves by checkpoint, concept and alpha"""
    records: list[dict[str, object]] = []
    for (checkpoint, concept, magnitude), group in rows.groupby(
        ["checkpoint", "concept", "magnitude"]
    ):
        record: dict[str, object] = {
            "checkpoint": checkpoint,
            "method": f"ssae_{checkpoint}",
            "concept": concept,
            "magnitude": float(magnitude),
            "n": int(group["sample_idx"].nunique()),
        }
        for metric in MAG_METRICS:
            stats = _ci(group[metric], seed=_stable_seed(checkpoint, concept, magnitude, metric))
            for key, value in stats.items():
                record[f"{metric}_{key}"] = value
        records.append(record)
    return pd.DataFrame(records).sort_values(["checkpoint", "concept", "magnitude"])


def magnitude_monotonicity(rows: pd.DataFrame) -> pd.DataFrame:
    """Report Spearman-like monotonicity without requiring scipy"""
    records: list[dict[str, object]] = []
    for (checkpoint, concept), group in rows.groupby(["checkpoint", "concept"]):
        by_mag = group.groupby("magnitude")["clip_image_vs_concept"].mean().sort_index()
        mags = by_mag.index.to_numpy(dtype=float)
        vals = by_mag.to_numpy(dtype=float)
        corr = float(np.corrcoef(pd.Series(mags).rank(), pd.Series(vals).rank())[0, 1])
        records.append(
            {
                "checkpoint": checkpoint,
                "method": f"ssae_{checkpoint}",
                "concept": concept,
                "spearman_like_r": corr,
                "delta_m1_minus_m0": float(vals[-1] - vals[0]),
                "m0": float(vals[0]),
                "m1": float(vals[-1]),
            }
        )
    return pd.DataFrame(records).sort_values(["checkpoint", "concept"])


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as markdown without optional pandas dependencies"""
    rounded = df.round(4).copy()
    headers = list(rounded.columns)
    rows = rounded.astype(object).where(pd.notna(rounded), "").values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown_report(
    out_path: Path,
    e3_summary: pd.DataFrame,
    e3_diffs: pd.DataFrame,
    mag_summary: pd.DataFrame,
    mag_mono: pd.DataFrame,
) -> None:
    """Write a compact, text-first report for the sprint notes"""
    lines = [
        "# E3 targeted editing and magnitude curves",
        "",
        "Status: analysed from S3 CSV/JSON artifacts. VLM judge not run.",
        "",
        "## E3 full-strength edit summary",
        "",
        _markdown_table(e3_summary),
        "",
        "## Main paired differences",
        "",
        _markdown_table(e3_diffs),
        "",
        "## Magnitude curve summary",
        "",
        _markdown_table(mag_summary),
        "",
        "## Magnitude monotonicity",
        "",
        _markdown_table(mag_mono),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/analysis/e3_image_editing"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    e3_rows = load_e3_rows(args.results_root)
    e3_summary = summarise_e3(e3_rows)
    e3_diffs = paired_method_diffs(e3_rows)
    mag_rows = load_magnitude_rows(args.results_root)
    mag_summary = summarise_magnitude(mag_rows)
    mag_mono = magnitude_monotonicity(mag_rows)

    e3_rows.to_csv(args.output_dir / "e3_combined_rows.csv", index=False)
    e3_summary.to_csv(args.output_dir / "e3_summary.csv", index=False)
    e3_diffs.to_csv(args.output_dir / "e3_paired_diffs.csv", index=False)
    mag_summary.to_csv(args.output_dir / "magnitude_summary.csv", index=False)
    mag_mono.to_csv(args.output_dir / "magnitude_monotonicity.csv", index=False)

    write_markdown_report(
        args.output_dir / "e3_image_editing_report.md",
        e3_summary,
        e3_diffs,
        mag_summary,
        mag_mono,
    )

    manifest = {
        "input_roots": [
            "results/bench_e3/",
            "results/bench_e3_cache/",
            "results/magnitude/",
        ],
        "outputs": sorted(path.name for path in args.output_dir.iterdir()),
        "vlm_judge_run": False,
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
