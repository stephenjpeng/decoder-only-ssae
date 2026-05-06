"""Batch VLM judging for ``run_image_benchmark`` outputs (OpenAI vision models)."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from tqdm import tqdm

from evaluation.vlm_openai import openai_judge_image


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run OpenAI vision judge on images from an image_benchmark output folder."
    )
    p.add_argument(
        "--benchmark_dir",
        type=Path,
        required=True,
        help="Directory containing per_sample.csv and images/ subfolder.",
    )
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--sleep_s", type=float, default=0.2, help="Pause between API calls.")
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--per_sample_out", type=Path, default=None)
    p.add_argument("--summary_out", type=Path, default=None)
    args = p.parse_args()

    bench = args.benchmark_dir
    csv_path = bench / "per_sample.csv"
    per_out = args.per_sample_out or (bench / "vlm_per_sample.jsonl")
    sum_out = args.summary_out or (bench / "vlm_summary.json")

    rows_in = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_in.append(row)

    if args.max_rows is not None:
        rows_in = rows_in[: args.max_rows]

    agg: dict[str, list[float]] = {}
    per_out.parent.mkdir(parents=True, exist_ok=True)

    with open(per_out, "w", encoding="utf-8") as jf:
        for row in tqdm(rows_in):
            idx = int(row["sample_idx"])
            method = row["method"]
            rel = Path("images") / method / f"{idx:05d}.png"
            img_path = bench / rel
            if not img_path.is_file():
                continue
            prompt = row.get("prompt", "")
            attrs: list[str] = []
            if prompt:
                attrs = [x.strip() for x in prompt.split(",") if x.strip()]

            try:
                out = openai_judge_image(
                    img_path,
                    full_prompt=prompt,
                    attribute_phrases=attrs,
                    model=args.model,
                )
            except Exception as e:
                out = {"error": str(e), "match_full_prompt": None}

            rec = {
                "sample_idx": idx,
                "method": method,
                "image": str(rel),
                "vlm": out,
            }
            jf.write(json.dumps(rec) + "\n")

            if "error" not in out and out.get("match_full_prompt") is not None:
                agg.setdefault(method, []).append(float(out["match_full_prompt"]))

            time.sleep(args.sleep_s)

    summary = {
        "model": args.model,
        "per_method_mean_match_full_prompt": {
            k: sum(v) / len(v) if v else None for k, v in agg.items()
        },
        "per_method_n": {k: len(v) for k, v in agg.items()},
    }
    with open(sum_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
