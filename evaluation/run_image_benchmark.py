"""
Image-level compositional benchmark for held-out concept tuples.

Compares **ground-truth embeddings**, **SSAE compositional** embeddings, **mean-direction**
and **ridge** embedding baselines, and **prompt-only** generation. Uses matched seeds per
sample (deterministic in ``base_seed``). Per-method image similarity vs. the ``gt_embed``
rendering is reported via CLIP, LPIPS, DINO cosine, and pixel-space MSE/SSIM.

Locality tests (both optional; independently enable-able in the same run). A single
attribute is randomly sampled per holdout row (seeded by ``base_seed + idx`` so the pick
is reproducible), shared between the two tests when both are on:

* ``--locality_drop_one_attr``: renders a same-seed **pre-edit** image per method with
  the chosen attribute's mask bit zeroed (or its phrase dropped from the prompt for
  ``prompt_only``). Reports pixel MSE/SSIM between the pre- and post-edit renders as an
  "edit surgical-ness" proxy under attribute removal.
* ``--locality_swap_one_attr``: renders a same-seed **swap** image per method with the
  chosen attribute's mask bit flipped to a different property in the same category (e.g.
  blond -> brunette), or the corresponding phrase substituted in the prompt for
  ``prompt_only``. Reports pixel MSE/SSIM between the swap and normal renders (surgical-
  ness under a value swap) and CLIP alignment of the swap image against the swapped prompt.

Writes ``per_sample.csv``, ``summary.json``, and PNGs under ``<output>/images/<method>/``
(post-edit), ``<output>/images_pre_edit/<method>/`` (only with ``--locality_drop_one_attr``),
and ``<output>/images_swapped/<method>/`` (only with ``--locality_swap_one_attr``).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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
from evaluation.pixel_metrics import pixel_mse, ssim
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


def _choose_edit_pid(mask_row: torch.Tensor, rng: random.Random) -> int | None:
    """Randomly sample one active-property id from ``mask_row``. None if no attribute is active."""
    m = mask_row.flatten().long()
    active = [pid for pid in range(m.numel()) if m[pid].item() > 0]
    if not active:
        return None
    return rng.choice(active)


def _choose_swap_target_pid(
    dataset, active_pid: int, rng: random.Random
) -> int | None:
    """Return a random pid in the same category as ``active_pid`` but different from it, or None."""
    props = dataset.properties
    cid = props.pid_to_cid[active_pid]
    alternatives = [p for p in props.cid_to_pids[cid] if p != active_pid]
    if not alternatives:
        return None
    return rng.choice(alternatives)


def _write_placeholder_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224), color=(120, 120, 120)).save(path)


def _sample_seed(base_seed: int, idx: int) -> int:
    return base_seed + idx * 1_000_003


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
    skip_pixel_metrics: bool = False,
    ridge_lambda: float = 1e-2,
    skip_dino: bool = True,
    dino_device: str | None = None,
    locality_drop_one_attr: bool = False,
    locality_swap_one_attr: bool = False,
) -> dict:
    clip_device = clip_device or ("cuda" if torch.cuda.is_available() else "cpu")
    holdout_folder = Path(ensure_folder_path(holdout_folder))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_root = output_dir / "images"
    img_root.mkdir(exist_ok=True)
    img_root_pre = output_dir / "images_pre_edit"
    if locality_drop_one_attr:
        img_root_pre.mkdir(exist_ok=True)
    img_root_swap = output_dir / "images_swapped"
    if locality_swap_one_attr:
        img_root_swap.mkdir(exist_ok=True)

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
        active_pids_sorted = [
            pid for pid in range(mask_t.numel()) if mask_t[pid].item() > 0
        ]

        x_tgt, mask_row_ds = holdout_ds[idx]
        x_tgt = x_tgt.unsqueeze(0).to(dev_dec).float()
        mask_row_ds = mask_row_ds.to(dev_dec)

        sample_rng = random.Random(base_seed + idx)
        do_drop = locality_drop_one_attr and len(attrs) > 1
        do_swap = locality_swap_one_attr and len(attrs) > 1

        edit_pid: int | None = None
        edit_attribute = ""
        swap_target_pid: int | None = None
        swap_target_attribute = ""
        residual_prompt = ""
        swapped_prompt = ""

        if do_drop or do_swap:
            edit_pid = _choose_edit_pid(mask_t, sample_rng)
            if edit_pid is None:
                do_drop = False
                do_swap = False
            else:
                edit_attribute = holdout_ds.properties.pid_to_property[edit_pid]
                edit_position = active_pids_sorted.index(edit_pid)
                if do_drop:
                    residual_prompt = ", ".join(
                        a for i, a in enumerate(attrs) if i != edit_position
                    )
                if do_swap:
                    swap_target_pid = _choose_swap_target_pid(
                        holdout_ds, edit_pid, sample_rng
                    )
                    if swap_target_pid is None:
                        # category has only one property, nothing to swap to
                        do_swap = False
                    else:
                        swap_target_attribute = holdout_ds.properties.pid_to_property[
                            swap_target_pid
                        ]
                        swapped_prompt = ", ".join(
                            swap_target_attribute if i == edit_position else a
                            for i, a in enumerate(attrs)
                        )

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

        pred_ssae_pre = pred_ma_pre = pred_ridge_pre = None
        if do_drop:
            mask_pre_ds = mask_row_ds.clone()
            mask_pre_ds[edit_pid] = 0
            pred_ssae_pre = predict_embedding_compositional(
                decoder,
                model_name,
                mask_pre_ds,
                mask_reduced_train=train_mask,
                n_repeat=n_repeat,
                device=dev_dec,
                block_means=block_means,
            )
            M_pre_cpu = mask_pre_ds.float().cpu().unsqueeze(0)
            pred_ma_pre = predict_mean_arithmetic(mu_ma, deltas_ma, M_pre_cpu).to(dev_dec)
            pred_ridge_pre = predict_linear(M_pre_cpu, W_ridge).to(dev_dec)

        pred_ssae_swap = pred_ma_swap = pred_ridge_swap = None
        if do_swap:
            mask_swap_ds = mask_row_ds.clone()
            mask_swap_ds[edit_pid] = 0
            mask_swap_ds[swap_target_pid] = 1
            pred_ssae_swap = predict_embedding_compositional(
                decoder,
                model_name,
                mask_swap_ds,
                mask_reduced_train=train_mask,
                n_repeat=n_repeat,
                device=dev_dec,
                block_means=block_means,
            )
            M_swap_cpu = mask_swap_ds.float().cpu().unsqueeze(0)
            pred_ma_swap = predict_mean_arithmetic(mu_ma, deltas_ma, M_swap_cpu).to(dev_dec)
            pred_ridge_swap = predict_linear(M_swap_cpu, W_ridge).to(dev_dec)

        for method in methods:
            out_path = img_root / method / f"{idx:05d}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            seed_i = _sample_seed(base_seed, idx)

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
                "edit_pid": edit_pid if edit_pid is not None else "",
                "edit_attribute": edit_attribute,
                "swap_target_pid": swap_target_pid if swap_target_pid is not None else "",
                "swap_target_attribute": swap_target_attribute,
                "swapped_prompt": swapped_prompt,
                "mse_embedding_vs_gt": "",
                "cosine_embedding_vs_gt": "",
                "dino_cosine_vs_gt_embed": "",
                "clip_image_vs_residual_prompt": "",
                "mse_pixel_vs_gt_embed": "",
                "ssim_vs_gt_embed": "",
                "mse_pixel_pre_post_edit": "",
                "ssim_pre_post_edit": "",
                "mse_pixel_swap_vs_normal": "",
                "ssim_swap_vs_normal": "",
                "clip_swap_image_vs_swapped_prompt": "",
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
                if do_drop:
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

            if (
                not skip_pixel_metrics
                and method != "gt_embed"
                and idx in ref_paths
                and not simulated
            ):
                row["mse_pixel_vs_gt_embed"] = pixel_mse(ref_paths[idx], out_path)
                row["ssim_vs_gt_embed"] = ssim(ref_paths[idx], out_path)

            if do_drop and method != "gt_embed":
                pre_path = img_root_pre / method / f"{idx:05d}.png"
                pre_path.parent.mkdir(parents=True, exist_ok=True)

                if method == "prompt_only":
                    pe_pre, pp_pre = None, None
                elif method == "ssae_compose":
                    pe_pre, pp_pre = pack_sd3_from_truncated_normalized(
                        holdout_ds, idx, pred_ssae_pre.detach().cpu()
                    )
                elif method == "mean_arithmetic":
                    pe_pre, pp_pre = pack_sd3_from_truncated_normalized(
                        holdout_ds, idx, pred_ma_pre.cpu()
                    )
                elif method == "ridge_embed":
                    pe_pre, pp_pre = pack_sd3_from_truncated_normalized(
                        holdout_ds, idx, pred_ridge_pre.cpu()
                    )
                else:
                    raise ValueError(f"Unknown method {method}")

                if not simulated:
                    if method == "prompt_only":
                        gen.generate_image_from_prompt(
                            residual_prompt, pre_path, use_negative_prompts=False, seed=seed_i
                        )
                    else:
                        pe_pre = pe_pre.to(sd_device)
                        pp_pre = pp_pre.to(sd_device)
                        gen.generate_image_from_embd(pe_pre, pp_pre, pre_path, seed=seed_i)
                else:
                    _write_placeholder_png(pre_path)

                if not skip_pixel_metrics and not simulated:
                    row["mse_pixel_pre_post_edit"] = pixel_mse(pre_path, out_path)
                    row["ssim_pre_post_edit"] = ssim(pre_path, out_path)

            if do_swap and method != "gt_embed":
                swap_path = img_root_swap / method / f"{idx:05d}.png"
                swap_path.parent.mkdir(parents=True, exist_ok=True)

                if method == "prompt_only":
                    pe_sw, pp_sw = None, None
                elif method == "ssae_compose":
                    pe_sw, pp_sw = pack_sd3_from_truncated_normalized(
                        holdout_ds, idx, pred_ssae_swap.detach().cpu()
                    )
                elif method == "mean_arithmetic":
                    pe_sw, pp_sw = pack_sd3_from_truncated_normalized(
                        holdout_ds, idx, pred_ma_swap.cpu()
                    )
                elif method == "ridge_embed":
                    pe_sw, pp_sw = pack_sd3_from_truncated_normalized(
                        holdout_ds, idx, pred_ridge_swap.cpu()
                    )
                else:
                    raise ValueError(f"Unknown method {method}")

                if not simulated:
                    if method == "prompt_only":
                        gen.generate_image_from_prompt(
                            swapped_prompt, swap_path, use_negative_prompts=False, seed=seed_i
                        )
                    else:
                        pe_sw = pe_sw.to(sd_device)
                        pp_sw = pp_sw.to(sd_device)
                        gen.generate_image_from_embd(pe_sw, pp_sw, swap_path, seed=seed_i)
                else:
                    _write_placeholder_png(swap_path)

                if not skip_pixel_metrics and not simulated:
                    row["mse_pixel_swap_vs_normal"] = pixel_mse(swap_path, out_path)
                    row["ssim_swap_vs_normal"] = ssim(swap_path, out_path)

                if clip_scorer is not None:
                    row["clip_swap_image_vs_swapped_prompt"] = (
                        clip_scorer.image_text_cosine(swap_path, swapped_prompt)
                    )

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
        mse_pixel_vals = [
            float(r["mse_pixel_vs_gt_embed"])
            for r in xs
            if r.get("mse_pixel_vs_gt_embed") != "" and r["mse_pixel_vs_gt_embed"] is not None
        ]
        ssim_vals = [
            float(r["ssim_vs_gt_embed"])
            for r in xs
            if r.get("ssim_vs_gt_embed") != "" and r["ssim_vs_gt_embed"] is not None
        ]
        mse_pre_post_vals = [
            float(r["mse_pixel_pre_post_edit"])
            for r in xs
            if r.get("mse_pixel_pre_post_edit") != ""
            and r["mse_pixel_pre_post_edit"] is not None
        ]
        ssim_pre_post_vals = [
            float(r["ssim_pre_post_edit"])
            for r in xs
            if r.get("ssim_pre_post_edit") != "" and r["ssim_pre_post_edit"] is not None
        ]
        mse_swap_vals = [
            float(r["mse_pixel_swap_vs_normal"])
            for r in xs
            if r.get("mse_pixel_swap_vs_normal") != ""
            and r["mse_pixel_swap_vs_normal"] is not None
        ]
        ssim_swap_vals = [
            float(r["ssim_swap_vs_normal"])
            for r in xs
            if r.get("ssim_swap_vs_normal") != "" and r["ssim_swap_vs_normal"] is not None
        ]
        clip_swap_vals = [
            float(r["clip_swap_image_vs_swapped_prompt"])
            for r in xs
            if r.get("clip_swap_image_vs_swapped_prompt") != ""
            and r["clip_swap_image_vs_swapped_prompt"] is not None
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
            "mse_pixel_vs_gt_embed": _ci_dict(mse_pixel_vals, n_bootstrap, seed_m + 11),
            "ssim_vs_gt_embed": _ci_dict(ssim_vals, n_bootstrap, seed_m + 13),
            "mse_pixel_pre_post_edit": _ci_dict(mse_pre_post_vals, n_bootstrap, seed_m + 15),
            "ssim_pre_post_edit": _ci_dict(ssim_pre_post_vals, n_bootstrap, seed_m + 17),
            "mse_pixel_swap_vs_normal": _ci_dict(mse_swap_vals, n_bootstrap, seed_m + 19),
            "ssim_swap_vs_normal": _ci_dict(ssim_swap_vals, n_bootstrap, seed_m + 21),
            "clip_swap_image_vs_swapped_prompt": _ci_dict(
                clip_swap_vals, n_bootstrap, seed_m + 23
            ),
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
    p.add_argument(
        "--skip_pixel_metrics",
        action="store_true",
        help="Skip pixel-space MSE/SSIM vs gt_embed reference image.",
    )
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
        help=(
            "Locality proxy under attribute removal. A single active attribute is randomly "
            "sampled per sample (seeded by base_seed + idx). Renders a same-seed pre-edit "
            "image per method with that attribute's mask bit zeroed / phrase dropped, and "
            "reports CLIP-vs-residual-prompt on the normal image plus pixel MSE/SSIM between "
            "the pre- and post-edit renders (edit surgical-ness)."
        ),
    )
    p.add_argument(
        "--locality_swap_one_attr",
        action="store_true",
        help=(
            "Locality proxy under attribute value swap. Same random attribute pick as "
            "--locality_drop_one_attr; instead of dropping, swap the value to a different "
            "property in the same category (e.g. blond -> brunette). Renders a same-seed "
            "swap image per method and reports pixel MSE/SSIM vs the normal render plus "
            "CLIP alignment of the swap image against the swapped prompt."
        ),
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
        skip_pixel_metrics=args.skip_pixel_metrics,
        ridge_lambda=args.ridge_lambda,
        skip_dino=not args.dino,
        dino_device=args.dino_device,
        locality_drop_one_attr=args.locality_drop_one_attr,
        locality_swap_one_attr=args.locality_swap_one_attr,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
