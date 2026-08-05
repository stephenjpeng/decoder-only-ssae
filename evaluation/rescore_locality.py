"""Retroactively compute locality-aware CLIP metrics on existing benchmark images.

The original benchmark (`run_image_benchmark.py`) stores `clip_image_vs_residual_prompt`
scored on the **full** (post-edit) image. That is not a locality test: the full image
still contains the dropped attribute, so its CLIP-vs-residual is nearly the same as its
CLIP-vs-full-prompt. This module re-scores the already-rendered images with the metrics
called out in `results/bench_out/ablation_report.html` &sect;3:

  * `clip_full_vs_edit_attr`     &mdash; how strongly the full render contains the target attribute.
  * `clip_dropped_vs_edit_attr`  &mdash; residual attribute signal after drop-one edit (lower is better).
  * `clip_dropped_vs_residual`   &mdash; drop-one image still matches the residual prompt (higher is better).
  * `clip_delta_edit_attr`       &mdash; full &minus; dropped on the edit-attribute phrase; large positive means the edit removed the attribute.
  * `clip_delta_residual`        &mdash; full &minus; dropped on the residual prompt; small means the rest was preserved.
  * `clip_locality_score`        &mdash; delta_edit_attr &minus; delta_residual; the surgical-ness we care about.
  * `clip_swap_vs_edit_attr`     &mdash; swap image against the *original* attribute (should be low).
  * `clip_swap_vs_target_attr`   &mdash; swap image against the *target* attribute (should be high).
  * `clip_swap_selectivity`      &mdash; target &minus; original on the swap image.

Images are read from either the SSAE benchmark dir (`<bench_dir>/<run>/images{,_pre_edit,_swapped}/ssae_compose/`)
or the shared baseline cache (`<baseline_dir>/<dataset_id>/images{,_pre_edit,_swapped}/<method>/`).
Output is a companion CSV `per_sample_locality_extra.csv` next to each `per_sample.csv`.

Usage:
    python -m evaluation.rescore_locality \
        --bench_dir results/bench_out \
        --baseline_dir results/bench_baseline_cache \
        [--configs topk_100000_L1,topk_100000_L2_h2048,topk_300000_L1,topk_300000_L2_h2048] \
        [--datasets 1c19e56f7560606d,59dbe58c95d0cb13] \
        [--device cuda|cpu]

Idempotent: existing rows in the extra CSV are skipped unless `--overwrite` is passed.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from evaluation.clip_scorer import CLIPScorer

EXTRA_FIELDS = [
    "sample_idx",
    "method",
    "clip_full_vs_edit_attr",
    "clip_dropped_vs_edit_attr",
    "clip_dropped_vs_residual",
    "clip_delta_edit_attr",
    "clip_delta_residual",
    "clip_locality_score",
    "clip_swap_vs_edit_attr",
    "clip_swap_vs_target_attr",
    "clip_swap_selectivity",
]

# Methods that render images and admit locality edits. gt_embed is skipped (no edits).
EDITABLE_METHODS = ("ssae_compose", "ridge_embed", "mean_arithmetic", "prompt_only")


@dataclass
class ImageRoots:
    """Where to find full / dropped / swapped images for a given method."""
    full: Path
    dropped: Path
    swapped: Path


def _resolve_roots(method: str, bench_run_dir: Path, baseline_dir: Path | None, dataset_id: str | None) -> ImageRoots:
    if method == "ssae_compose":
        return ImageRoots(
            full=bench_run_dir / "images" / method,
            dropped=bench_run_dir / "images_pre_edit" / method,
            swapped=bench_run_dir / "images_swapped" / method,
        )
    if baseline_dir is None or dataset_id is None:
        raise ValueError(f"baseline dir + dataset_id needed for method {method}")
    root = baseline_dir / dataset_id
    return ImageRoots(
        full=root / "images" / method,
        dropped=root / "images_pre_edit" / method,
        swapped=root / "images_swapped" / method,
    )


def _load_manifest_dataset_id(bench_run_dir: Path) -> str:
    with open(bench_run_dir / "manifest.json", encoding="utf-8") as f:
        return json.load(f)["dataset_id"]


def _load_extra_index(path: Path) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(int(r["sample_idx"]), r["method"]) for r in csv.DictReader(f)}


def _score_row(
    scorer: CLIPScorer,
    row: dict,
    roots: ImageRoots,
) -> dict:
    """Compute locality metrics for a single (sample_idx, method) row."""
    idx = int(row["sample_idx"])
    edit_attribute = row.get("edit_attribute", "")
    residual = _residual_prompt_from_row(row)
    swap_target = row.get("swap_target_attribute", "")

    out = {"sample_idx": idx, "method": row["method"]}

    full_path = roots.full / f"{idx:05d}.png"
    dropped_path = roots.dropped / f"{idx:05d}.png"
    swap_path = roots.swapped / f"{idx:05d}.png"

    if edit_attribute and full_path.exists():
        out["clip_full_vs_edit_attr"] = scorer.image_text_cosine(full_path, edit_attribute)
    if edit_attribute and dropped_path.exists():
        out["clip_dropped_vs_edit_attr"] = scorer.image_text_cosine(dropped_path, edit_attribute)
    if residual and dropped_path.exists():
        out["clip_dropped_vs_residual"] = scorer.image_text_cosine(dropped_path, residual)

    # Deltas: only if both components populated.
    if "clip_full_vs_edit_attr" in out and "clip_dropped_vs_edit_attr" in out:
        out["clip_delta_edit_attr"] = out["clip_full_vs_edit_attr"] - out["clip_dropped_vs_edit_attr"]
    full_vs_residual = row.get("clip_image_vs_residual_prompt", "")
    try:
        full_vs_residual_f = float(full_vs_residual)
    except (TypeError, ValueError):
        full_vs_residual_f = None
    if full_vs_residual_f is not None and "clip_dropped_vs_residual" in out:
        out["clip_delta_residual"] = full_vs_residual_f - out["clip_dropped_vs_residual"]
    if "clip_delta_edit_attr" in out and "clip_delta_residual" in out:
        out["clip_locality_score"] = out["clip_delta_edit_attr"] - out["clip_delta_residual"]

    if edit_attribute and swap_path.exists():
        out["clip_swap_vs_edit_attr"] = scorer.image_text_cosine(swap_path, edit_attribute)
    if swap_target and swap_path.exists():
        out["clip_swap_vs_target_attr"] = scorer.image_text_cosine(swap_path, swap_target)
    if "clip_swap_vs_edit_attr" in out and "clip_swap_vs_target_attr" in out:
        out["clip_swap_selectivity"] = out["clip_swap_vs_target_attr"] - out["clip_swap_vs_edit_attr"]

    return out


def _residual_prompt_from_row(row: dict) -> str:
    """Reconstruct the residual (attribute-dropped) prompt from the CSV row.

    The benchmark stored `swapped_prompt` but not the residual; rebuild by removing the
    edit attribute clause from the original prompt.
    """
    prompt = row.get("prompt", "")
    attr = row.get("edit_attribute", "")
    if not prompt or not attr:
        return ""
    # Attributes are comma-separated in the prompt (e.g. "A blond girl, with blue eyes, ...").
    parts = [p.strip() for p in prompt.split(",")]
    parts = [p for p in parts if p and p != attr and not p.endswith(attr)]
    return ", ".join(parts)


def _rows_for_run(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["task"] == "holdout_unseen_tuple"]


def rescore_run(
    scorer: CLIPScorer,
    csv_path: Path,
    roots_by_method: dict[str, ImageRoots],
    out_path: Path,
    *,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Rescore one per_sample.csv; returns (scored, skipped)."""
    rows = _rows_for_run(csv_path)
    already = _load_extra_index(out_path) if not overwrite else set()
    new_rows: list[dict] = []
    scored = skipped = 0
    for r in rows:
        method = r["method"]
        if method not in roots_by_method:
            continue
        idx = int(r["sample_idx"])
        if (idx, method) in already:
            skipped += 1
            continue
        # Need at least an edit to score anything.
        if not r.get("edit_attribute"):
            continue
        row_out = _score_row(scorer, r, roots_by_method[method])
        new_rows.append(row_out)
        scored += 1

    if new_rows or not out_path.exists():
        _write_extra_csv(out_path, new_rows, append=out_path.exists())
    return scored, skipped


def _write_extra_csv(path: Path, rows: list[dict], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXTRA_FIELDS)
        if mode == "w":
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in EXTRA_FIELDS})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_dir", type=Path, default=Path("results/bench_out"))
    ap.add_argument("--baseline_dir", type=Path, default=Path("results/bench_baseline_cache"))
    ap.add_argument("--configs", type=str, default=None,
                    help="Comma-separated bench run names; default = all subdirs of bench_dir.")
    ap.add_argument("--datasets", type=str, default=None,
                    help="Comma-separated dataset_ids under baseline_dir; default = all subdirs.")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--include_prompt_only", action="store_true",
                    help="Also rescore prompt_only baseline images (off by default; they use the "
                         "residual prompt directly and don't test embedding locality).")
    args = ap.parse_args()

    scorer = CLIPScorer(device=args.device)

    # --- SSAE benchmark runs ---
    if args.configs:
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    else:
        configs = sorted(p.name for p in args.bench_dir.iterdir()
                         if p.is_dir() and (p / "per_sample.csv").exists())

    total_scored = total_skipped = 0
    for cfg in configs:
        run_dir = args.bench_dir / cfg
        csv_path = run_dir / "per_sample.csv"
        if not csv_path.exists():
            print(f"[skip] {cfg}: no per_sample.csv")
            continue
        dsid = _load_manifest_dataset_id(run_dir)
        roots = {"ssae_compose": _resolve_roots("ssae_compose", run_dir, None, None)}
        out_path = run_dir / "per_sample_locality_extra.csv"
        s, sk = rescore_run(scorer, csv_path, roots, out_path, overwrite=args.overwrite)
        print(f"[bench ] {cfg} (ds={dsid[:8]}...): scored={s} skipped={sk} -> {out_path.name}")
        total_scored += s
        total_skipped += sk

    # --- Shared baseline caches ---
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = sorted(p.name for p in args.baseline_dir.iterdir()
                          if p.is_dir() and (p / "per_sample.csv").exists())

    methods = ["ridge_embed", "mean_arithmetic"]
    if args.include_prompt_only:
        methods.append("prompt_only")

    for dsid in datasets:
        run_dir = args.baseline_dir / dsid
        csv_path = run_dir / "per_sample.csv"
        roots = {m: _resolve_roots(m, run_dir, args.baseline_dir, dsid) for m in methods}
        out_path = run_dir / "per_sample_locality_extra.csv"
        s, sk = rescore_run(scorer, csv_path, roots, out_path, overwrite=args.overwrite)
        print(f"[base  ] ds={dsid[:8]}...: scored={s} skipped={sk} -> {out_path.name}")
        total_scored += s
        total_skipped += sk

    print(f"done. scored={total_scored} skipped={total_skipped}")


if __name__ == "__main__":
    main()
