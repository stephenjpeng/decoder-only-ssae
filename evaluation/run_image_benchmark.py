"""
Image-level compositional benchmark for held-out concept tuples.

Compares **ground-truth embeddings**, **SSAE compositional** embeddings, **mean-direction**
and **ridge** embedding baselines, and **prompt-only** generation. Uses matched seeds per
sample (deterministic in ``base_seed``).

Writes ``per_sample.csv``, ``summary.json``, and PNGs under ``<output>/images/<method>/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from baselines.run_baselines import (
    fit_mean_arithmetic,
    fit_ridge,
    predict_linear,
    predict_mean_arithmetic,
)
from evaluation.bootstrap import bootstrap_mean_ci
from evaluation.clip_scorer import CLIPScorer
from evaluation.composition import (
    property_block_means_trainable_inputs,
    predict_embedding_compositional,
)
from evaluation.io import ensure_folder_path, h5_dataset_for_folder, load_decoder_checkpoint
from evaluation.lpips_metric import lpips_alex
from evaluation.sd3_pack import pack_sd3_from_truncated_normalized
from inference.image_generation.image_generator import ImageGenerator

_METHOD_ORDER = (
    "gt_embed",
    "ssae_compose",
    "mean_arithmetic",
    "ridge_embed",
    "prompt_only",
)


def _sort_methods(methods: tuple[str, ...]) -> tuple[str, ...]:
    seen = set(methods)
    ordered = tuple(m for m in _METHOD_ORDER if m in seen)
    extra = tuple(m for m in methods if m not in ordered)
    return ordered + extra


def _stack_cpu(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ms = [], []
    for i in range(len(dataset)):
        x, m = dataset[i]
        xs.append(x)
        ms.append(m.float())
    return torch.stack(xs, dim=0), torch.stack(ms, dim=0)


def _active_phrases_from_mask(dataset, mask_row: torch.Tensor) -> list[str]:
    props = dataset.properties
    m = mask_row.flatten().long()
    out = []
    for pid in range(m.numel()):
        if m[pid].item() > 0:
            out.append(props.pid_to_property[pid])
    return out


def _write_placeholder_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224), color=(120, 120, 120)).save(path)


def _sample_seed(base_seed: int, idx: int, method: str) -> int:
    h = int.from_bytes(hashlib.md5(method.encode()).digest()[:4], "big")
    return base_seed + idx * 1_000_003 + (h % 1_000_000)


def _ci_dict(vals: list[float], n_boot: int, seed: int) -> dict:
    if not vals:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    mean, lo, hi = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)}


def run_image_benchmark(
    checkpoint_dir: Path,
    holdout_folder: Path,
    output_dir: Path,
    *,
    max_samples: int | None = None,
    base_seed: int = 0,
    simulated: bool = False,
    sd_device: str = "cuda",
    clip_device: str | None = None,
    clip_failure_threshold: float = 0.2,
    n_bootstrap: int = 2000,
    methods: tuple[str, ...] = (
        "gt_embed",
        "ssae_compose",
        "mean_arithmetic",
        "ridge_embed",
        "prompt_only",
    ),
    skip_lpips: bool = False,
    ridge_lambda: float = 1e-2,
    skip_dino: bool = True,
    dino_device: str | None = None,
    locality_drop_one_attr: bool = False,
) -> dict:
    clip_device = clip_device or ("cuda" if torch.cuda.is_available() else "cpu")
    holdout_folder = Path(ensure_folder_path(holdout_folder))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_root = output_dir / "images"
    img_root.mkdir(exist_ok=True)

    methods = _sort_methods(methods)

    decoder, tp, train_ds = load_decoder_checkpoint(checkpoint_dir, device=sd_device)
    holdout_ds = h5_dataset_for_folder(checkpoint_dir, holdout_folder)
    decoder.eval()

    model_name = tp["model_name"]
    n_repeat = int(tp["n_repeat"])
    dev_dec = torch.device(sd_device)

    block_means = None
    train_mask = train_ds.mask_reduced.to(dev_dec)
    if model_name == "model_trainable_inputs":
        block_means = property_block_means_trainable_inputs(
            decoder, train_mask, n_repeat, dev_dec
        )

    X_tr_cpu, M_tr_cpu = _stack_cpu(train_ds)
    mu_ma, deltas_ma = fit_mean_arithmetic(X_tr_cpu, M_tr_cpu)
    W_ridge = fit_ridge(M_tr_cpu, X_tr_cpu, ridge_lambda)

    prompts_path = holdout_folder / "prompts.json"
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_meta = json.load(f)

    n = len(holdout_ds)
    if max_samples is not None:
        n = min(n, max_samples)

    gen = ImageGenerator(simulated=simulated, device=sd_device)
    clip_ok = not simulated
    clip_scorer = CLIPScorer(device=clip_device) if clip_ok else None

    dino_bundle = None
    if not skip_dino and not simulated:
        from evaluation.dino_embed import load_dinov2_vits14

        dino_bundle = load_dinov2_vits14(dino_device or clip_device)

    rows: list[dict] = []
    ref_paths: dict[int, Path] = {}

    for idx in range(n):
        prompt_text = prompts_meta[idx]["prompt"]
        mask_row = holdout_ds.properties.tid_to_rm[idx]
        mask_t = torch.tensor(mask_row, dtype=torch.int16, device=dev_dec)
        attrs = _active_phrases_from_mask(holdout_ds, mask_t.float())

        x_tgt, mask_row_ds = holdout_ds[idx]
        x_tgt = x_tgt.unsqueeze(0).to(dev_dec).float()
        mask_row_ds = mask_row_ds.to(dev_dec)

        pred_ssae = predict_embedding_compositional(
            decoder,
            model_name,
            mask_row_ds,
            mask_reduced_train=train_mask,
            n_repeat=n_repeat,
            device=dev_dec,
            block_means=block_means,
        )
        mse_ssae = torch.nn.functional.mse_loss(pred_ssae, x_tgt).item()
        cos_ssae = torch.nn.functional.cosine_similarity(pred_ssae, x_tgt, dim=-1).mean().item()

        M_row_cpu = mask_row_ds.float().cpu().unsqueeze(0)
        pred_ma = predict_mean_arithmetic(mu_ma, deltas_ma, M_row_cpu).to(dev_dec)
        mse_ma = torch.nn.functional.mse_loss(pred_ma, x_tgt).item()

        pred_ridge = predict_linear(M_row_cpu, W_ridge).to(dev_dec)
        mse_ridge = torch.nn.functional.mse_loss(pred_ridge, x_tgt).item()

        for method in methods:
            out_path = img_root / method / f"{idx:05d}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            seed_i = _sample_seed(base_seed, idx, method)

            if method == "gt_embed":
                pe, pp = pack_sd3_from_truncated_normalized(holdout_ds, idx, x_tgt.cpu())
            elif method == "ssae_compose":
                pe, pp = pack_sd3_from_truncated_normalized(
                    holdout_ds, idx, pred_ssae.detach().cpu()
                )
            elif method == "mean_arithmetic":
                pe, pp = pack_sd3_from_truncated_normalized(holdout_ds, idx, pred_ma.cpu())
            elif method == "ridge_embed":
                pe, pp = pack_sd3_from_truncated_normalized(holdout_ds, idx, pred_ridge.cpu())
            elif method == "prompt_only":
                pe, pp = None, None
            else:
                raise ValueError(f"Unknown method {method}")

            if not simulated:
                if method == "prompt_only":
                    gen.generate_image_from_prompt(
                        prompt_text, out_path, use_negative_prompts=False, seed=seed_i
                    )
                else:
                    pe = pe.to(sd_device)
                    pp = pp.to(sd_device)
                    gen.generate_image_from_embd(pe, pp, out_path, seed=seed_i)
            else:
                _write_placeholder_png(out_path)

            if method == "gt_embed":
                ref_paths[idx] = out_path

            row = {
                "task": "holdout_unseen_tuple",
                "sample_idx": idx,
                "method": method,
                "prompt": prompt_text,
                "mse_embedding_vs_gt": "",
                "cosine_embedding_vs_gt": "",
                "dino_cosine_vs_gt_embed": "",
                "clip_image_vs_residual_prompt": "",
            }
            if method == "ssae_compose":
                row["mse_embedding_vs_gt"] = mse_ssae
                row["cosine_embedding_vs_gt"] = cos_ssae
            elif method == "mean_arithmetic":
                row["mse_embedding_vs_gt"] = mse_ma
            elif method == "ridge_embed":
                row["mse_embedding_vs_gt"] = mse_ridge

            if clip_scorer is not None:
                row["clip_image_vs_full_prompt"] = clip_scorer.image_text_cosine(
                    out_path, prompt_text
                )
                al = clip_scorer.image_attribute_alignment(out_path, attrs)
                row["clip_mean_vs_attrs"] = al["mean_cosine_attr"]
                row["clip_min_vs_attrs"] = al["min_cosine_attr"]
                row["clip_fail"] = float(row["clip_image_vs_full_prompt"] < clip_failure_threshold)
                if locality_drop_one_attr and len(attrs) > 1:
                    residual_prompt = ", ".join(attrs[1:])
                    row["clip_image_vs_residual_prompt"] = clip_scorer.image_text_cosine(
                        out_path, residual_prompt
                    )
            else:
                row["clip_image_vs_full_prompt"] = ""
                row["clip_mean_vs_attrs"] = ""
                row["clip_min_vs_attrs"] = ""
                row["clip_fail"] = ""

            if (
                dino_bundle is not None
                and method != "gt_embed"
                and idx in ref_paths
                and not simulated
            ):
                from evaluation.dino_embed import dino_cosine_similarity

                d_m, d_tf, d_dev = dino_bundle
                row["dino_cosine_vs_gt_embed"] = dino_cosine_similarity(
                    ref_paths[idx], out_path, d_m, d_tf, d_dev
                )

            if not skip_lpips and method != "gt_embed" and idx in ref_paths and not simulated:
                lp = lpips_alex(ref_paths[idx], out_path, device=clip_device)
                row["lpips_vs_gt_embed"] = lp if lp is not None else ""
            else:
                row["lpips_vs_gt_embed"] = ""

            rows.append(row)

    csv_path = output_dir / "per_sample.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    summary = _aggregate_summary(
        rows,
        methods=list(methods),
        clip_failure_threshold=clip_failure_threshold,
        n_bootstrap=n_bootstrap,
        base_seed=base_seed,
    )
    summary["ridge_lambda"] = ridge_lambda
    summary["methods"] = list(methods)

    mse_ssae = [float(r["mse_embedding_vs_gt"]) for r in rows if r["method"] == "ssae_compose" and r["mse_embedding_vs_gt"] != ""]
    cos_ssae = [float(r["cosine_embedding_vs_gt"]) for r in rows if r["method"] == "ssae_compose" and r["cosine_embedding_vs_gt"] != ""]
    if mse_ssae:
        summary["ssae_holdout_embedding_space"] = {
            "mse_mean": float(np.mean(mse_ssae)),
            "cosine_mean": float(np.mean(cos_ssae)) if cos_ssae else float("nan"),
            "mse_ci_95": _ci_dict(mse_ssae, n_bootstrap, base_seed + 1),
            "cosine_ci_95": _ci_dict(cos_ssae, n_bootstrap, base_seed + 2),
        }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _aggregate_summary(
    rows: list[dict],
    *,
    methods: list[str],
    clip_failure_threshold: float,
    n_bootstrap: int,
    base_seed: int,
) -> dict:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    out: dict = {
        "clip_failure_threshold": clip_failure_threshold,
        "per_method": {},
    }

    for mi, m in enumerate(methods):
        xs = by_method.get(m, [])
        clip_vals = [
            float(r["clip_image_vs_full_prompt"])
            for r in xs
            if r.get("clip_image_vs_full_prompt") != ""
            and r["clip_image_vs_full_prompt"] is not None
        ]
        attr_mean_vals = [
            float(r["clip_mean_vs_attrs"])
            for r in xs
            if r.get("clip_mean_vs_attrs") != "" and r["clip_mean_vs_attrs"] is not None
        ]
        lpips_vals = [
            float(r["lpips_vs_gt_embed"])
            for r in xs
            if r.get("lpips_vs_gt_embed") != "" and r["lpips_vs_gt_embed"] is not None
        ]
        fail_vals = [
            float(r["clip_fail"])
            for r in xs
            if r.get("clip_fail") != "" and r["clip_fail"] is not None
        ]
        dino_vals = [
            float(r["dino_cosine_vs_gt_embed"])
            for r in xs
            if r.get("dino_cosine_vs_gt_embed") != "" and r["dino_cosine_vs_gt_embed"] is not None
        ]
        resid_vals = [
            float(r["clip_image_vs_residual_prompt"])
            for r in xs
            if r.get("clip_image_vs_residual_prompt") != ""
            and r["clip_image_vs_residual_prompt"] is not None
        ]

        seed_m = base_seed + 17 * mi
        out["per_method"][m] = {
            "clip_image_vs_full_prompt": _ci_dict(clip_vals, n_bootstrap, seed_m),
            "clip_mean_vs_active_attrs": _ci_dict(attr_mean_vals, n_bootstrap, seed_m + 3),
            "failure_rate_clip_below_threshold": float(np.mean(fail_vals))
            if fail_vals
            else float("nan"),
            "lpips_vs_gt_embed": _ci_dict(lpips_vals, n_bootstrap, seed_m + 5),
            "dino_cosine_vs_gt_embed": _ci_dict(dino_vals, n_bootstrap, seed_m + 7),
            "clip_image_vs_residual_prompt": _ci_dict(resid_vals, n_bootstrap, seed_m + 9),
        }

    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--holdout_folder", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--simulated", action="store_true")
    p.add_argument("--sd_device", type=str, default="cuda")
    p.add_argument("--clip_device", type=str, default=None)
    p.add_argument("--clip_failure_threshold", type=float, default=0.2)
    p.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument(
        "--methods",
        type=str,
        default="gt_embed,ssae_compose,mean_arithmetic,ridge_embed,prompt_only",
    )
    p.add_argument("--skip_lpips", action="store_true")
    p.add_argument("--ridge_lambda", type=float, default=1e-2)
    p.add_argument(
        "--dino",
        action="store_true",
        help="Compute DINOv2 cosine similarity vs gt_embed reference (extra compute/GPU).",
    )
    p.add_argument("--dino_device", type=str, default=None)
    p.add_argument(
        "--locality_drop_one_attr",
        action="store_true",
        help="CLIP image vs prompt with the first attribute phrase removed (locality proxy).",
    )
    args = p.parse_args()

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    summary = run_image_benchmark(
        args.checkpoint,
        args.holdout_folder,
        args.output_dir,
        max_samples=args.max_samples,
        base_seed=args.base_seed,
        simulated=args.simulated,
        sd_device=args.sd_device,
        clip_device=args.clip_device,
        clip_failure_threshold=args.clip_failure_threshold,
        n_bootstrap=args.n_bootstrap,
        methods=methods,
        skip_lpips=args.skip_lpips,
        ridge_lambda=args.ridge_lambda,
        skip_dino=not args.dino,
        dino_device=args.dino_device,
        locality_drop_one_attr=args.locality_drop_one_attr,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
