"""Fit a CLIP linear multi-label probe on training images; evaluate on benchmark images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.clip_linear_probe import probe_micro_metrics, train_multilabel_probe
from evaluation.clip_scorer import CLIPScorer
from evaluation.io import ensure_folder_path


def _mask_matrix_from_prompts_json(folder: Path) -> tuple[np.ndarray, Properties]:
    fp = ensure_folder_path(folder)
    props = Properties(folder_path=fp, logger=None)
    n = len(props.prompts)
    L = props.n_properties
    mat = np.zeros((n, L), dtype=np.float32)
    for tid in range(n):
        mat[tid] = props.tid_to_rm[tid]
    return mat, props


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_folder", type=Path, required=True)
    p.add_argument(
        "--train_images_subdir",
        type=Path,
        required=True,
        help="PNG files 00000.png ... in train index order.",
    )
    p.add_argument("--holdout_folder", type=Path, required=True)
    p.add_argument("--benchmark_dir", type=Path, required=True)
    p.add_argument("--eval_method", type=str, default="ssae_compose")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--clip_device", type=str, default=None)
    p.add_argument("--probe_device", type=str, default=None)
    p.add_argument("--save_probe", type=Path, default=None)
    p.add_argument("--output_json", type=Path, default=None)
    args = p.parse_args()

    clip_dev = args.clip_device or ("cuda" if torch.cuda.is_available() else "cpu")
    probe_dev = args.probe_device or clip_dev

    Y_tr, props = _mask_matrix_from_prompts_json(args.train_folder)
    n = Y_tr.shape[0]
    train_paths = [args.train_images_subdir / f"{i:05d}.png" for i in range(n)]
    if not train_paths[0].is_file():
        raise FileNotFoundError(f"Missing {train_paths[0]}")

    scorer = CLIPScorer(device=clip_dev)
    X_tr = scorer.image_encode_paths_batch([str(p) for p in train_paths])
    fit = train_multilabel_probe(
        X_tr, torch.from_numpy(Y_tr), steps=args.steps, device=probe_dev
    )
    probe = fit.model

    if args.save_probe is not None:
        args.save_probe.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": probe.state_dict(),
                "n_properties": props.n_properties,
                "clip_model": scorer.model_name,
            },
            args.save_probe,
        )

    props_h = Properties(folder_path=ensure_folder_path(args.holdout_folder), logger=None)
    n_ho = len(props_h.prompts)
    Y_ho = np.zeros((n_ho, props_h.n_properties), dtype=np.float32)
    for tid in range(n_ho):
        Y_ho[tid] = props_h.tid_to_rm[tid]

    bench_csv = args.benchmark_dir / "per_sample.csv"
    with open(bench_csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["method"] == args.eval_method]
    if not rows:
        with open(bench_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    xs: list[str] = []
    ys: list[np.ndarray] = []
    for r in rows:
        idx = int(r["sample_idx"])
        img = args.benchmark_dir / "images" / r["method"] / f"{idx:05d}.png"
        if not img.is_file():
            continue
        xs.append(str(img))
        ys.append(Y_ho[idx])

    if not xs:
        raise RuntimeError("No images found for evaluation.")

    X_ev = scorer.image_encode_paths_batch(xs)
    Y_ev = torch.from_numpy(np.stack(ys, axis=0))
    metrics = probe_micro_metrics(probe, X_ev, Y_ev, device=probe_dev)

    out = {
        "train_probe_loss_final": fit.train_loss_final,
        "eval_method_filter": args.eval_method,
        "n_eval": len(xs),
        **metrics,
    }
    print(json.dumps(out, indent=2))
    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
