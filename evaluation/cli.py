"""Unified entry point: ``python -m evaluation.cli <subcommand> ...``."""

from __future__ import annotations

import argparse
import subprocess
import sys


def _run(module: str, argv: list[str]) -> int:
    cmd = [sys.executable, "-m", module] + argv
    return subprocess.call(cmd)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluation / baselines helpers (delegates to other modules)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("reconstruction", help="SSAE train-set reconstruction MSE/cosine")
    sub.add_parser("compositional", help="Holdout compositional embedding metrics")
    sub.add_parser("decorrelation", help="Concept sub-vector cosine stats (JSON)")
    sub.add_parser(
        "decorrelation_plot", help="Save concept cosine heatmap PNG from a checkpoint"
    )
    sub.add_parser("copy_truncation", help="Copy indices / min-max JSONs train -> holdout")
    sub.add_parser("image_benchmark", help="SD3 images + CLIP/LPIPS/DINO vs baselines")
    sub.add_parser("vlm_openai_batch", help="OpenAI vision judge on image_benchmark output")
    sub.add_parser("clip_probe", help="Train CLIP linear probe; eval on benchmark images")
    sub.add_parser("edit_summary", help="Aggregate per_sample.csv by edit_type manifest")
    sub.add_parser("clip", help="CLIP image_text or text_text (pass through args)")
    sub.add_parser("baselines", help="Mean arithmetic, ridge, PCA on holdout")
    sub.add_parser("unsup_ae", help="Shallow AE baseline on holdout")
    sub.add_parser("unsup_sae", help="Sparse L1 AE baseline on holdout")
    sub.add_parser("sweep", help="Generate ablation YAMLs (see experiments.sweep_generate)")

    args, rest = p.parse_known_args()

    if args.cmd == "reconstruction":
        sys.exit(_run("evaluation.run_reconstruction", rest))
    if args.cmd == "compositional":
        sys.exit(_run("evaluation.run_compositional_embeddings", rest))
    if args.cmd == "decorrelation":
        sys.exit(_run("evaluation.decorrelation", rest))
    if args.cmd == "decorrelation_plot":
        sys.exit(_run("evaluation.run_decorrelation_plot", rest))
    if args.cmd == "copy_truncation":
        sys.exit(_run("evaluation.run_copy_truncation", rest))
    if args.cmd == "image_benchmark":
        sys.exit(_run("evaluation.run_image_benchmark", rest))
    if args.cmd == "vlm_openai_batch":
        sys.exit(_run("evaluation.run_vlm_openai_batch", rest))
    if args.cmd == "clip_probe":
        sys.exit(_run("evaluation.run_clip_probe", rest))
    if args.cmd == "edit_summary":
        sys.exit(_run("evaluation.run_edit_type_summary", rest))
    if args.cmd == "clip":
        sys.exit(_run("evaluation.run_clip_scores", rest))
    if args.cmd == "baselines":
        sys.exit(_run("baselines.run_baselines", rest))
    if args.cmd == "unsup_ae":
        sys.exit(_run("baselines.run_unsup_ae_holdout", rest))
    if args.cmd == "unsup_sae":
        sys.exit(_run("baselines.run_unsup_sae_holdout", rest))
    if args.cmd == "sweep":
        sys.exit(_run("experiments.sweep_generate", rest))


if __name__ == "__main__":
    main()
