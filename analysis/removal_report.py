"""Build an HTML report for E3 removal metrics and image examples"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

BUCKET = "s3://ssae-runs-sp"
TARGET_ORDER = ["gun", "hat", "beach"]
METHOD_ORDER = [
    "prompt_only",
    "ridge_embed",
    "mean_arithmetic",
    "ssae_L1",
    "ssae_L2_h2048",
]
IMAGE_METHOD_ORDER = ["gt_embed", *METHOD_ORDER]
METHOD_LABELS = {
    "gt_embed": "Original gt_embed",
    "prompt_only": "Prompt modification",
    "ridge_embed": "Ridge",
    "mean_arithmetic": "Mean arithmetic",
    "ssae_L1": "SSAE-L1",
    "ssae_L2_h2048": "SSAE-L2 h2048",
}
TARGET_LABELS = {
    "gun": "holding a gun",
    "hat": "and a hat",
    "beach": "at the beach",
}


def _target_from_benchmark_dir(path: str) -> str:
    bench = Path(path)
    if bench.parent.parent.name == "bench_e3_cache":
        return bench.parent.name
    return bench.name.split("_", 1)[0]


def _method_label(rec: dict[str, Any]) -> str:
    method = rec["method"]
    if method != "ssae_compose":
        return method
    run = Path(rec["benchmark_dir"]).name.split("_", 1)[1]
    return f"ssae_{run}"


def _load_vlm_records(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out = rec.get("vlm", {})
            target = _target_from_benchmark_dir(rec["benchmark_dir"])
            method = _method_label(rec)
            target_score = None
            for item in out.get("attribute_scores", []):
                if item.get("phrase") == rec.get("edit_attribute", ""):
                    target_score = item.get("score")
                    break
            rows.append(
                {
                    "target": target,
                    "sample_idx": int(rec["sample_idx"]),
                    "method": method,
                    "variant": rec["variant"],
                    "benchmark_dir": rec["benchmark_dir"],
                    "image": rec["image"],
                    "prompt_judged": rec.get("prompt_judged", ""),
                    "match_full_prompt": out.get("match_full_prompt"),
                    "non_target_preserved": out.get("non_target_preserved"),
                    "target_attribute_score": target_score,
                    "notes": out.get("notes", ""),
                }
            )
    return pd.DataFrame(rows)


def _fmt(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _ci(mean: Any, low: Any, high: Any, digits: int = 3) -> str:
    if pd.isna(mean):
        return ""
    return f"{float(mean):.{digits}f} [{float(low):.{digits}f}, {float(high):.{digits}f}]"


def _copy_image(s3_uri: str, local_path: Path) -> None:
    if local_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["aws", "s3", "cp", s3_uri, str(local_path), "--only-show-errors"],
        check=True,
    )


def _build_summary_table(e3_summary: pd.DataFrame, vlm_group: pd.DataFrame) -> str:
    vlm_deleted = vlm_group[vlm_group["variant"].eq("deleted")].copy()
    vlm_deleted["method_label"] = vlm_deleted.apply(
        lambda row: f"ssae_{row['run']}" if row["method"] == "ssae_compose" else row["method"],
        axis=1,
    )

    rows = []
    for target in TARGET_ORDER:
        for method in METHOD_ORDER:
            e3 = e3_summary[
                e3_summary["concept"].eq(target) & e3_summary["method"].eq(method)
            ]
            vlm = vlm_deleted[
                vlm_deleted["target"].eq(target) & vlm_deleted["method_label"].eq(method)
            ]
            if e3.empty:
                continue
            e = e3.iloc[0]
            v = vlm.iloc[0] if not vlm.empty else None
            rows.append(
                "<tr>"
                f"<td>{html.escape(TARGET_LABELS[target])}</td>"
                f"<td>{html.escape(METHOD_LABELS[method])}</td>"
                f"<td>{int(e['n'])}</td>"
                f"<td>{_ci(e['delete_delta_target_mean'], e['delete_delta_target_ci_low'], e['delete_delta_target_ci_high'])}</td>"
                f"<td>{_ci(e['clip_deleted_vs_target_phrase_mean'], e['clip_deleted_vs_target_phrase_ci_low'], e['clip_deleted_vs_target_phrase_ci_high'])}</td>"
                f"<td>{_ci(e['clip_image_vs_residual_prompt_mean'], e['clip_image_vs_residual_prompt_ci_low'], e['clip_image_vs_residual_prompt_ci_high'])}</td>"
                f"<td>{_ci(e['ssim_pre_post_edit_mean'], e['ssim_pre_post_edit_ci_low'], e['ssim_pre_post_edit_ci_high'])}</td>"
                f"<td>{_ci(e['mse_pixel_pre_post_edit_mean'], e['mse_pixel_pre_post_edit_ci_low'], e['mse_pixel_pre_post_edit_ci_high'])}</td>"
                f"<td>{_fmt(v['edit_attribute_score'] if v is not None else None)}</td>"
                f"<td>{_fmt(v['match_full_prompt'] if v is not None else None)}</td>"
                f"<td>{_fmt(v['non_target_preserved'] if v is not None else None)}</td>"
                "</tr>"
            )
    header = """
<table>
<thead>
<tr>
<th>Removed concept</th>
<th>Method</th>
<th>n</th>
<th>CLIP target drop ↑</th>
<th>CLIP target after deletion ↓</th>
<th>CLIP residual prompt ↑</th>
<th>SSIM pre/post ↑</th>
<th>Pixel MSE pre/post ↓</th>
<th>VLM target present ↓</th>
<th>VLM prompt match ↑</th>
<th>VLM non-target preserved ↑</th>
</tr>
</thead>
<tbody>
"""
    return header + "\n".join(rows) + "\n</tbody>\n</table>"


def _build_paired_table(clip_diffs: pd.DataFrame, vlm_diffs: pd.DataFrame) -> str:
    rows = []
    comparisons = [("ssae_L1", "prompt_only"), ("ssae_L2_h2048", "prompt_only")]
    for target in TARGET_ORDER:
        for left, right in comparisons:
            clip = clip_diffs[
                clip_diffs["concept"].eq(target)
                & clip_diffs["left_method"].eq(left)
                & clip_diffs["right_method"].eq(right)
            ]
            vlm = vlm_diffs[
                vlm_diffs["target"].eq(target)
                & vlm_diffs["variant"].eq("deleted")
                & vlm_diffs["left_method"].eq(left)
                & vlm_diffs["right_method"].eq(right)
            ]

            def clip_metric(metric: str) -> str:
                hit = clip[clip["metric"].eq(metric)]
                if hit.empty:
                    return ""
                row = hit.iloc[0]
                return _ci(
                    row["mean_diff_left_minus_right"],
                    row["ci_low"],
                    row["ci_high"],
                )

            def vlm_metric(metric: str) -> str:
                hit = vlm[vlm["metric"].eq(metric)]
                if hit.empty:
                    return ""
                row = hit.iloc[0]
                return _ci(
                    row["mean_diff_left_minus_right"],
                    row["ci_low"],
                    row["ci_high"],
                )

            rows.append(
                "<tr>"
                f"<td>{html.escape(TARGET_LABELS[target])}</td>"
                f"<td>{html.escape(METHOD_LABELS[left])} − Prompt modification</td>"
                f"<td>{clip_metric('delete_delta_target')}</td>"
                f"<td>{clip_metric('ssim_pre_post_edit')}</td>"
                f"<td>{clip_metric('mse_pixel_pre_post_edit')}</td>"
                f"<td>{vlm_metric('edit_attribute_score')}</td>"
                f"<td>{vlm_metric('match_full_prompt')}</td>"
                f"<td>{vlm_metric('non_target_preserved')}</td>"
                "</tr>"
            )
    return """
<table>
<thead>
<tr>
<th>Removed concept</th>
<th>Paired comparison</th>
<th>CLIP target-drop diff ↑</th>
<th>SSIM diff ↑</th>
<th>Pixel MSE diff ↓</th>
<th>VLM target-present diff ↓</th>
<th>VLM prompt-match diff ↑</th>
<th>VLM non-target diff ↑</th>
</tr>
</thead>
<tbody>
""" + "\n".join(rows) + "\n</tbody>\n</table>"


def _complete_sample_indices(vlm: pd.DataFrame, target: str, n: int) -> list[int]:
    required = [
        ("gt_embed", "post"),
        ("prompt_only", "deleted"),
        ("ridge_embed", "deleted"),
        ("mean_arithmetic", "deleted"),
        ("ssae_L1", "deleted"),
        ("ssae_L2_h2048", "deleted"),
    ]
    sample_indices = []
    scoped = vlm[vlm["target"].eq(target)]
    for sample_idx in sorted(scoped["sample_idx"].unique()):
        rows = scoped[scoped["sample_idx"].eq(sample_idx)]
        keys = set(zip(rows["method"], rows["variant"]))
        if all(key in keys for key in required):
            sample_indices.append(int(sample_idx))
        if len(sample_indices) >= n:
            break
    return sample_indices


def _build_image_grid(vlm: pd.DataFrame, out_dir: Path, examples_per_target: int) -> str:
    sections = []
    image_dir = out_dir / "images"
    for target in TARGET_ORDER:
        target_sections = [f"<h3>{html.escape(TARGET_LABELS[target])}</h3>"]
        for sample_idx in _complete_sample_indices(vlm, target, examples_per_target):
            cells = []
            for method in IMAGE_METHOD_ORDER:
                variant = "post" if method == "gt_embed" else "deleted"
                hit = vlm[
                    vlm["target"].eq(target)
                    & vlm["sample_idx"].eq(sample_idx)
                    & vlm["method"].eq(method)
                    & vlm["variant"].eq(variant)
                ]
                if hit.empty:
                    cells.append("<td class='missing'>missing</td>")
                    continue
                row = hit.iloc[0]
                local = image_dir / target / f"{sample_idx:05d}" / f"{method}.png"
                s3_uri = f"{BUCKET}/{row['benchmark_dir']}/{row['image']}"
                _copy_image(s3_uri, local)
                rel = local.relative_to(out_dir)
                caption = (
                    f"target={_fmt(row['target_attribute_score'])}<br>"
                    f"match={_fmt(row['match_full_prompt'])}<br>"
                    f"preserve={_fmt(row['non_target_preserved'])}"
                    if method != "gt_embed"
                    else "source render"
                )
                cells.append(
                    "<td>"
                    f"<div class='method'>{html.escape(METHOD_LABELS[method])}</div>"
                    f"<img src='{html.escape(str(rel))}' alt='{html.escape(method)} sample {sample_idx}'>"
                    f"<div class='caption'>{caption}</div>"
                    "</td>"
                )
            prompt = html.escape(str(hit.iloc[0]["prompt_judged"])) if not hit.empty else ""
            target_sections.append(
                f"<h4>Sample {sample_idx}</h4>"
                f"<p class='prompt'>{prompt}</p>"
                "<table class='grid'><tbody><tr>"
                + "\n".join(cells)
                + "</tr></tbody></table>"
            )
        sections.append("\n".join(target_sections))
    return "\n".join(sections)


def _html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #222; }}
h1, h2, h3 {{ line-height: 1.2; }}
p {{ max-width: 980px; }}
table {{ border-collapse: collapse; margin: 16px 0 28px; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
th {{ background: #f5f5f5; text-align: left; }}
.grid {{ table-layout: fixed; }}
.grid td {{ width: 16.6%; text-align: center; }}
.grid img {{ max-width: 100%; height: auto; border: 1px solid #ccc; }}
.method {{ font-weight: 600; margin-bottom: 6px; }}
.caption {{ color: #555; font-size: 12px; line-height: 1.35; margin-top: 6px; }}
.prompt {{ color: #555; font-size: 13px; max-width: 1200px; }}
.note {{ background: #fff8dc; border-left: 4px solid #c8a600; padding: 10px 12px; max-width: 980px; }}
.missing {{ color: #888; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/analysis/removal_report"),
    )
    parser.add_argument(
        "--examples_per_target",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    e3_summary = pd.read_csv("results/analysis/e3_image_editing/e3_summary.csv")
    clip_diffs = pd.read_csv("results/analysis/e3_image_editing/e3_paired_diffs.csv")
    vlm_group = pd.read_csv("results/analysis/e3_vlm_openai/e3_vlm_by_group.csv")
    vlm_diffs = pd.read_csv("results/analysis/e3_vlm_openai/e3_vlm_paired_diffs.csv")
    vlm = _load_vlm_records(Path("results/vlm_e3_full/e3_vlm.jsonl"))
    summary_table = _build_summary_table(e3_summary, vlm_group)
    paired_table = _build_paired_table(clip_diffs, vlm_diffs)
    image_grid = _build_image_grid(vlm, args.out_dir, args.examples_per_target)

    body = f"""
<h1>E3 removal metrics and image examples</h1>
<p>This report covers the 300k targeted deletion benchmark. It compares SSAE feature deletion against prompt modification, ridge, and mean-arithmetic baselines on the same tuples and diffusion seeds.</p>
<p class="note">Interpretation: lower target-present scores after deletion are better. Higher residual-prompt, SSIM, and non-target-preserved scores are better. The OpenAI VLM scores are diagnostic until the prepared 50-image human validation set is labelled.</p>
<h2>Branch-relevant conclusion</h2>
<p>Feature deletion does not beat prompt modification. The decisive failure is hat deletion: OpenAI VLM target-present gaps versus prompt deletion are +0.985 for SSAE-L1 and +0.965 for SSAE-L2 h2048, far outside the 5-point near-parity band. CLIP also shows hat deletion is flat for feature methods while prompt deletion moves the target phrase.</p>
<h2>Deletion metrics by method</h2>
{summary_table}
<h2>Paired differences versus prompt modification</h2>
<p>Rows are feature method minus prompt modification on matched samples. For VLM target-present and pixel MSE, negative is better. For target-drop, SSIM, prompt-match and non-target preservation, positive is better.</p>
{paired_table}
<h2>Representative generated images</h2>
<p>Each row shows the original source render and deletion outputs for the same sample. Captions show VLM target-present, full-prompt match, and non-target preservation scores for deleted images.</p>
{image_grid}
"""
    out_path = args.out_dir / "removal_metrics_report.html"
    out_path.write_text(_html_page("E3 removal metrics", body), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
