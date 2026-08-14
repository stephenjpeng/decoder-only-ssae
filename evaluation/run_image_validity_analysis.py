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

from evaluation.image_qualification import PromptSpec
from evaluation.image_validity import (
    analyze_image_validity,
    write_results_json,
    write_summary_csv,
)


def required_bakeoff_arms(prompt_spec_path: Path) -> set[tuple[str, str]]:
    """Load the frozen model-property arm universe used by the qualification gate"""
    spec = PromptSpec.load(prompt_spec_path)
    return {
        (model, property_id)
        for model in spec.bakeoff_design.model_backbones
        for property_id in spec.bakeoff_design.property_ids
    }


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
        "--blinding-mapping",
        type=Path,
        help="private mode-0600 mapping for an opaque scoring sheet",
    )
    parser.add_argument(
        "--unblinded-scores-output",
        type=Path,
        help="optional private canonical score CSV written during unblinding",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=0.90,
        help="minimum observed validity rate to pass gate (default 0.90)",
    )
    parser.add_argument(
        "--prompt-spec",
        type=Path,
        default=Path("evaluation/config/image_validity_prompts.yaml"),
        help="frozen prompt spec that defines every required bakeoff arm",
    )

    args = parser.parse_args()

    try:
        results = analyze_image_validity(
            manifest_path=args.manifest,
            scores_path=args.scores,
            gate_threshold=args.gate_threshold,
            blinding_mapping_path=args.blinding_mapping,
            unblinded_scores_output=args.unblinded_scores_output,
            required_model_property_arms=required_bakeoff_arms(args.prompt_spec),
        )

        write_results_json(results, args.output_json)
        write_summary_csv(results, args.output_csv)

        print(f"wrote JSON results to {args.output_json}")
        print(f"wrote CSV summary to {args.output_csv}")

        # pooled validity is diagnostic; every required arm must pass the gate
        overall = results["overall"]
        print(
            f"\npooled bakeoff validity (diagnostic, not gate): "
            f"{overall['n_valid']}/{overall['n_total']} = {overall['rate']:.2%}"
        )
        print(f"95% CI: [{overall['ci_lower']:.2%}, {overall['ci_upper']:.2%}]")
        decision = results["qualification_decision"]
        outcome = "PASS" if decision["passes_gate"] else "FAIL"
        print(
            f"qualification decision: {outcome}; all "
            f"{decision['required_model_property_arms']} model-property arms must meet "
            f"{args.gate_threshold:.0%}"
        )
        for arm in decision["failed_arms"]:
            print(
                f"failed arm: {arm['model_backbone']} / {arm['property_id']} "
                f"({arm['n_valid']}/{arm['n_total']} = {arm['rate']:.2%})"
            )

        return 0

    except Exception as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
