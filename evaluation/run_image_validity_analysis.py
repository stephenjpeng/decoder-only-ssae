"""
CLI for image validity analysis.

Usage:
    python -m evaluation.run_image_validity_analysis \
        --manifest path/to/manifest.jsonl \
        --scores path/to/scores.csv \
        --output-json path/to/results.json \
        --output-csv path/to/summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluation.image_validity import (
    analyze_image_validity,
    write_results_json,
    write_summary_csv,
)


def main() -> int:
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Analyze image validity scores with Wilson confidence intervals"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="path to render manifest JSONL",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        required=True,
        help="path to human scoring CSV",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="path to write machine-readable JSON results",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="path to write human-readable summary CSV",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=0.90,
        help="minimum observed validity rate to pass gate (default 0.90)",
    )

    args = parser.parse_args()

    try:
        results = analyze_image_validity(
            manifest_path=args.manifest,
            scores_path=args.scores,
            gate_threshold=args.gate_threshold,
        )

        write_results_json(results, args.output_json)
        write_summary_csv(results, args.output_csv)

        print(f"wrote JSON results to {args.output_json}")
        print(f"wrote CSV summary to {args.output_csv}")

        # print key summary to stdout
        overall = results["overall"]
        print(
            f"\noverall validity: {overall['n_valid']}/{overall['n_total']} = {overall['rate']:.2%}"
        )
        print(f"95% CI: [{overall['ci_lower']:.2%}, {overall['ci_upper']:.2%}]")
        print(f"passes {args.gate_threshold:.0%} gate: {overall['passes_gate']}")

        return 0

    except Exception as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
