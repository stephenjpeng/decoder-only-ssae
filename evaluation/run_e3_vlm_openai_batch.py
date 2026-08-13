"""Run OpenAI VLM judging over E3 post/deleted/swapped image variants

The generic ``run_vlm_openai_batch`` script only reads ``images/<method>/<idx>.png``.
E3 needs three variants per editable row:

- post: the unedited reconstruction/render
- deleted: ``images_pre_edit`` from target deletion or category-marginal removal
- swapped: ``images_swapped`` from on-manifold replacement

This script reads the same ``per_sample.csv`` rows and emits one JSONL record per image variant.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from evaluation.vlm_openai import openai_judge_image

VARIANTS = ("post", "deleted", "swapped")
VARIANT_DIRS = {
    "post": "images",
    "deleted": "images_pre_edit",
    "swapped": "images_swapped",
}


def _attrs(prompt: str) -> list[str]:
    return [part.strip() for part in prompt.split(",") if part.strip()]


def _variant_prompt(row: dict[str, str], variant: str) -> str:
    if variant == "post":
        return row.get("prompt", "")
    if variant == "deleted":
        prompt = row.get("prompt", "")
        target = row.get("edit_attribute", "")
        attrs = [attr for attr in _attrs(prompt) if attr != target]
        return ", ".join(attrs)
    if variant == "swapped":
        return row.get("swapped_prompt", "") or row.get("prompt", "")
    raise ValueError(f"unknown variant {variant}")


def _attribute_phrases(row: dict[str, str], variant_prompt: str) -> list[str]:
    phrases = _attrs(variant_prompt)
    for key in ("edit_attribute", "swap_target_attribute"):
        value = row.get(key, "")
        if value and value not in phrases:
            phrases.append(value)
    return phrases


def _image_path(bench: Path, row: dict[str, str], variant: str) -> Path:
    method = row["method"]
    idx = int(row["sample_idx"])
    return bench / VARIANT_DIRS[variant] / method / f"{idx:05d}.png"


def _record_for_row(
    bench: Path,
    row: dict[str, str],
    variant: str,
    *,
    model: str,
) -> dict[str, Any] | None:
    image_path = _image_path(bench, row, variant)
    if not image_path.is_file():
        return None

    prompt = _variant_prompt(row, variant)
    attributes = _attribute_phrases(row, prompt)
    try:
        out = openai_judge_image(
            image_path,
            full_prompt=prompt,
            attribute_phrases=attributes,
            model=model,
        )
    except (OSError, ValueError, RuntimeError, ImportError, EnvironmentError) as err:
        out = {"error": str(err), "match_full_prompt": None}
    except Exception as err:
        if not err.__class__.__module__.startswith("openai"):
            raise
        # external API failures should not erase prior JSONL progress
        out = {"error": str(err), "match_full_prompt": None}

    return {
        "benchmark_dir": str(bench),
        "sample_idx": int(row["sample_idx"]),
        "method": row["method"],
        "variant": variant,
        "image": str(image_path.relative_to(bench)),
        "prompt_judged": prompt,
        "edit_attribute": row.get("edit_attribute", ""),
        "swap_target_attribute": row.get("swap_target_attribute", ""),
        "vlm": out,
    }


def _load_existing_keys(path: Path) -> set[tuple[str, int, str, str]]:
    keys: set[tuple[str, int, str, str]] = set()
    if not path.exists():
        return keys
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            keys.add(
                (
                    rec.get("benchmark_dir", ""),
                    int(rec.get("sample_idx", -1)),
                    rec.get("method", ""),
                    rec.get("variant", ""),
                )
            )
    return keys


def _summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    errors = 0
    for rec in records:
        out = rec.get("vlm", {})
        if "error" in out:
            errors += 1
            continue
        value = out.get("match_full_prompt")
        if value is None:
            continue
        key = f"{rec['variant']}::{rec['method']}"
        groups.setdefault(key, []).append(float(value))
    return {
        "n_records": len(records),
        "n_errors": errors,
        "per_variant_method_mean_match_full_prompt": {
            key: sum(vals) / len(vals) for key, vals in groups.items() if vals
        },
        "per_variant_method_n": {key: len(vals) for key, vals in groups.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_dir", type=Path, action="append", required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_out", type=Path, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--sleep_s", type=float, default=0.2)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(VARIANTS),
        help="Comma-separated subset of post,deleted,swapped",
    )
    args = parser.parse_args()

    variants = tuple(part.strip() for part in args.variants.split(",") if part.strip())
    for variant in variants:
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant {variant!r}; valid: {', '.join(VARIANTS)}")

    existing = _load_existing_keys(args.output_jsonl) if args.resume else set()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[Path, dict[str, str], str]] = []
    for bench in args.benchmark_dir:
        csv_path = bench / "per_sample.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for variant in variants:
                    key = (str(bench), int(row["sample_idx"]), row["method"], variant)
                    if key in existing:
                        continue
                    if _image_path(bench, row, variant).is_file():
                        tasks.append((bench, row, variant))

    if args.max_records is not None:
        tasks = tasks[: args.max_records]

    mode = "a" if args.resume else "w"
    new_records: list[dict[str, Any]] = []
    with open(args.output_jsonl, mode, encoding="utf-8") as out_f:
        for bench, row, variant in tqdm(tasks):
            rec = _record_for_row(bench, row, variant, model=args.model)
            if rec is None:
                continue
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            new_records.append(rec)
            time.sleep(args.sleep_s)

    all_records: list[dict[str, Any]] = []
    with open(args.output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))

    summary = _summarise(all_records)
    summary.update(
        {
            "model": args.model,
            "benchmark_dirs": [str(path) for path in args.benchmark_dir],
            "variants": list(variants),
            "new_records_this_run": len(new_records),
        }
    )
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
