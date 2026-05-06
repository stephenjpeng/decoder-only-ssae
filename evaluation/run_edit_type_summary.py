"""Aggregate benchmark metrics by ``edit_type`` using a small JSON manifest.

Manifest format (list of objects)::

    [
      {"sample_idx": 0, "method": "ssae_compose", "edit_type": "swap_property"},
      {"sample_idx": 1, "method": "ssae_compose", "edit_type": "remove_property"}
    ]

Joins on ``(sample_idx, method)`` with ``per_sample.csv`` and reports mean CLIP / failure
rates per ``edit_type``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per_sample_csv", type=Path, required=True)
    p.add_argument("--manifest_json", type=Path, required=True)
    p.add_argument("--output_json", type=Path, default=None)
    args = p.parse_args()

    with open(args.manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    key_to_edit = {}
    for item in manifest:
        k = (int(item["sample_idx"]), str(item["method"]))
        key_to_edit[k] = str(item.get("edit_type", "unspecified"))

    rows_by_key = {}
    with open(args.per_sample_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = (int(row["sample_idx"]), row["method"])
            rows_by_key[k] = row

    by_edit: dict[str, list[dict]] = defaultdict(list)
    missing = 0
    for k, edit_t in key_to_edit.items():
        r = rows_by_key.get(k)
        if r is None:
            missing += 1
            continue
        by_edit[edit_t].append(r)

    summary = {"missing_keys": missing, "by_edit_type": {}}
    for edit_t, rs in by_edit.items():
        clip_vals = []
        fail_vals = []
        for r in rs:
            c = r.get("clip_image_vs_full_prompt", "")
            if c != "" and c is not None:
                clip_vals.append(float(c))
            f = r.get("clip_fail", "")
            if f != "" and f is not None:
                fail_vals.append(float(f))
        summary["by_edit_type"][edit_t] = {
            "n": len(rs),
            "mean_clip_image_vs_full_prompt": sum(clip_vals) / len(clip_vals)
            if clip_vals
            else None,
            "mean_clip_fail": sum(fail_vals) / len(fail_vals) if fail_vals else None,
        }

    text = json.dumps(summary, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
