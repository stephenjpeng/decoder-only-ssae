"""
Image-level compositional benchmark for held-out concept tuples.

Compares **ground-truth embeddings**, **SSAE compositional** embeddings, **mean-direction**
and **ridge** embedding baselines, and three **prompt-side** conditioning paths. Uses
matched seeds per sample (deterministic in ``base_seed``). Per-method image similarity vs.
the ``gt_embed`` rendering is reported via CLIP, LPIPS, DINO cosine, and pixel-space
MSE/SSIM.

Rendering ladder (see ``evaluation/method_labels.py``):

* ``native_prompt`` — **Native text generation**. Prompt text is passed directly to the
  diffusion pipeline without calling encode_prompt. This is the deployment-realistic
  baseline.
* ``prompt_only`` — **Exact full-embedding round-trip**. The prompt is encoded and the
  full text-encoder output (333x4096 + 2048 pooled) is immediately decoded. Despite the
  key name, this is NOT true native generation. The key is kept for cache compatibility;
  the label clarifies the computation.
* ``prompt_modified_packed`` — **Prompt modification (packed top-k)**. The prompt is
  encoded but only the SSAE's top-k coordinates survive; the rest are filled with the
  training mean, matching how every embedding-space method is packed. Use for controlled-
  subspace analysis. Never average or conflate these three paths.

Locality tests (both optional; independently enable-able in the same run). A single
attribute is randomly sampled per holdout row (seeded by ``base_seed + idx`` so the pick
is reproducible), shared between the two tests when both are on:

* ``--locality_drop_one_attr``: renders a same-seed **pre-edit** image per method with
  the chosen attribute's mask bit zeroed (or its phrase dropped from the prompt for
  ``native_prompt`` and ``prompt_only``). Reports pixel MSE/SSIM between the pre- and
  post-edit renders as an "edit surgical-ness" proxy under attribute removal.
* ``--locality_swap_one_attr``: renders a same-seed **swap** image per method with the
  chosen attribute's mask bit flipped to a different property in the same category (e.g.
  blond -> brunette), or the corresponding phrase substituted in the prompt for
  ``native_prompt`` and ``prompt_only``. Reports pixel MSE/SSIM between the swap and
  normal renders (surgical-ness under a value swap) and CLIP alignment of the swap image
  against the swapped prompt.

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
from evaluation.baseline_cache import BASELINE_METHODS, BaselineCache, load_or_create
from evaluation.bootstrap import bootstrap_mean_ci
from evaluation.clip_scorer import CLIPScorer
from evaluation.method_labels import (
    DEFAULT_METHODS,
    METHOD_ORDER,
    conditioning_map,
)
from evaluation.method_labels import sort_methods as _sort_methods
from evaluation.run_probe_intervention_fit import load_and_validate_probe_artifact
from evaluation.composition import (
    property_block_means_trainable_inputs,
    predict_embedding_compositional,
)
from evaluation.io import ensure_folder_path, h5_dataset_for_folder, load_decoder_checkpoint
from evaluation.lpips_metric import lpips_alex
from evaluation.pixel_metrics import pixel_mse, ssim
from evaluation.sd3_pack import (
    compute_or_load_full_mean,
    flatten_sd3_conditioning,
    get_full_concat_embedding_untruncated,
    pack_sd3_from_full_flat_topk,
    pack_sd3_from_truncated_normalized,
    packer_fingerprint,
)
from inference.image_generation.image_generator import ImageGenerator
from trainings.utils.run_manifest import checkpoint_fingerprint, write_run_manifest

DEFAULT_BASELINE_CACHE_ROOT = Path("results/bench_baseline_cache")


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


def _resolve_property(dataset, name: str) -> int:
    """Property id for a phrase, or a loud error listing the valid ones."""
    props = dataset.properties
    for pid, phrase in props.pid_to_property.items():
        if phrase == name:
            return int(pid)
    raise SystemExit(
        f"unknown property {name!r}. Valid: "
        + ", ".join(repr(props.pid_to_property[p]) for p in sorted(props.pid_to_property))
    )


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


def _category_marginal_mask(dataset, mask_row: torch.Tensor, target_pid: int) -> torch.Tensor:
    """Replace ``target_pid`` with the uniform marginal over its category"""
    props = dataset.properties
    cid = props.pid_to_cid[target_pid]
    members = list(props.cid_to_pids[cid])
    out = mask_row.float().clone()
    out[target_pid] = 0.0
    for pid in members:
        out[pid] = 1.0 / len(members)
    return out


def _packer_metadata(fill_policy: str, drop_operator: str) -> dict:
    """Cache/run identity for policy choices that change rendered pixels"""
    meta = dict(packer_fingerprint())
    meta["fill_policy"] = fill_policy
    meta["drop_operator"] = drop_operator
    return meta


def _write_placeholder_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224), color=(120, 120, 120)).save(path)


def _sample_seed(base_seed: int, idx: int) -> int:
    return base_seed + idx * 1_000_003


def _encode_and_pack_prompt(
    gen: ImageGenerator,
    dataset,
    prompt_text: str,
    seed: int,
    template: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``prompt_modified_packed`` conditioning: re-encode, keep top-k, train-mean the rest.

    Distinct from ``prompt_only``, which hands the text encoder's full output straight to
    the pipeline. Here the prompt is pushed through the same information bottleneck as the
    SSAE and ridge predictions, so the comparison isolates the *edit operator* rather than
    the number of conditioning coordinates. See ``evaluation/method_labels.py``.

    The same ``seed`` is used for encoding and later for diffusion, matching the
    ``prompt_only`` path exactly so the two prompt rows differ only in conditioning.
    """
    prompt_embeds, _, pooled_prompt_embeds, _ = gen.get_embds_text_encoder(
        prompt=prompt_text, use_negative_prompts=False, seed=seed
    )
    full_flat = flatten_sd3_conditioning(prompt_embeds, pooled_prompt_embeds)
    return pack_sd3_from_full_flat_topk(dataset, full_flat, template=template)


def _find_ref_training_tid(
    train_mask: torch.Tensor,
    mask_swap: torch.Tensor,
    swap_target_pid: int,
) -> int:
    """Return the training sample with swap_target_pid active and highest mask cosine similarity to mask_swap."""
    target_active = train_mask[:, swap_target_pid].float() > 0.5
    if target_active.any():
        candidates = train_mask[target_active].float()
        candidate_indices = torch.where(target_active)[0]
    else:
        candidates = train_mask.float()
        candidate_indices = torch.arange(len(train_mask), device=train_mask.device)
    mask_swap_f = mask_swap.float()
    mask_swap_norm = mask_swap_f / (mask_swap_f.norm() + 1e-8)
    cand_norms = candidates / (candidates.norm(dim=1, keepdim=True) + 1e-8)
    sims = cand_norms @ mask_swap_norm
    return int(candidate_indices[sims.argmax()].item())


def _merge_row(cached: dict, fresh: dict) -> dict:
    """Fresh values win iff non-empty; else keep cached."""
    merged = dict(cached)
    for k, v in fresh.items():
        if v == "" or v is None:
            continue
        merged[k] = v
    return merged


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
    ssae_device: str | None = None,
    baseline_device: str | None = None,
    clip_device: str | None = None,
    clip_failure_threshold: float = 0.2,
    n_bootstrap: int = 2000,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    skip_lpips: bool = False,
    skip_pixel_metrics: bool = False,
    ridge_lambda: float = 1e-2,
    skip_dino: bool = True,
    dino_device: str | None = None,
    locality_drop_one_attr: bool = False,
    locality_swap_one_attr: bool = False,
    baseline_cache_root: Path | None = None,
    use_baseline_cache: bool = True,
    baselines_only: bool = False,
    target_property: str | None = None,
    replacement_property: str | None = None,
    max_matched: int | None = None,
    fill_policy: str = "train_mean",
    drop_operator: str = "zero",
    probe_artifact_path: Path | None = None,
) -> dict:
    ssae_device = ssae_device or sd_device
    baseline_device = baseline_device or ssae_device
    clip_device = clip_device or ("cuda" if torch.cuda.is_available() else "cpu")
    holdout_folder = Path(ensure_folder_path(holdout_folder))
    if fill_policy not in {"train_mean", "source_prompt"}:
        raise ValueError("fill_policy must be 'train_mean' or 'source_prompt'")
    if drop_operator not in {"zero", "category_marginal"}:
        raise ValueError("drop_operator must be 'zero' or 'category_marginal'")

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

    if baselines_only:
        methods = tuple(m for m in methods if m != "ssae_compose")
    methods = _sort_methods(methods)

    # load and validate probe artifact before touching the GPU
    probe_artifact: dict | None = None
    if "linear_probe_direction" in methods:
        if probe_artifact_path is None:
            raise SystemExit("--probe_artifact is required when 'linear_probe_direction' is in --methods")
        if "gt_embed" not in methods:
            raise SystemExit("'gt_embed' must be in --methods when using 'linear_probe_direction' (its post image is reused)")

    decoder, tp, train_ds = load_decoder_checkpoint(checkpoint_dir, device=ssae_device)
    holdout_ds = h5_dataset_for_folder(checkpoint_dir, holdout_folder)
    decoder.eval()

    model_name = tp["model_name"]
    n_repeat = int(tp["n_repeat"])
    dev_dec = torch.device(ssae_device)
    dev_base = torch.device(baseline_device)

    want_ssae = ("ssae_compose" in methods) and not baselines_only
    block_means = None
    train_mask = train_ds.mask_reduced.to(dev_dec)
    if want_ssae and model_name == "model_trainable_inputs":
        block_means = property_block_means_trainable_inputs(
            decoder, train_mask, n_repeat, dev_dec
        )

    X_tr_cpu, M_tr_cpu = _stack_cpu(train_ds)

    # Default fill template. Source-prompt fill is selected per sample below because
    # the deployment-style template is the unedited prompt's full embedding.
    train_mean_template = compute_or_load_full_mean(train_ds).cpu()
    pack_template = train_mean_template

    cache: BaselineCache | None = None
    cache_methods: set[str] = set()
    if use_baseline_cache:
        cache_root = Path(baseline_cache_root) if baseline_cache_root else DEFAULT_BASELINE_CACHE_ROOT
        cache = load_or_create(
            cache_root,
            holdout_folder=holdout_folder,
            train_x=X_tr_cpu,
            train_mask=M_tr_cpu,
            base_seed=base_seed,
            ridge_lambda=ridge_lambda,
            sd3_fingerprint=ImageGenerator.fingerprint(),
            packer_fingerprint=_packer_metadata(fill_policy, drop_operator),
        )
        cache_methods = {m for m in methods if m in BASELINE_METHODS}
        loaded_fits = cache.load_fits()
        if loaded_fits is not None:
            mu_ma, deltas_ma, W_ridge = loaded_fits
        else:
            mu_ma, deltas_ma = fit_mean_arithmetic(X_tr_cpu, M_tr_cpu)
            W_ridge = fit_ridge(M_tr_cpu, X_tr_cpu, ridge_lambda)
            cache.save_fits(mu_ma=mu_ma, deltas_ma=deltas_ma, W_ridge=W_ridge)
    else:
        mu_ma, deltas_ma = fit_mean_arithmetic(X_tr_cpu, M_tr_cpu)
        W_ridge = fit_ridge(M_tr_cpu, X_tr_cpu, ridge_lambda)

    prompts_path = holdout_folder / "prompts.json"
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_meta = json.load(f)

    n = len(holdout_ds)
    if max_samples is not None:
        n = min(n, max_samples)

    # E3 targeted-concept mode. `target_property` names the concept to delete/replace;
    # `replacement_property` fixes the on-manifold counterfactual instead of sampling one.
    target_pid = _resolve_property(holdout_ds, target_property) if target_property else None
    replacement_pid = (
        _resolve_property(holdout_ds, replacement_property) if replacement_property else None
    )
    if replacement_pid is not None:
        if target_pid is None:
            raise SystemExit("--replacement_property requires --target_property")
        props = holdout_ds.properties
        if props.pid_to_cid[replacement_pid] != props.pid_to_cid[target_pid]:
            raise SystemExit(
                f"replacement {replacement_property!r} is not in the same category as "
                f"target {target_property!r}; a cross-category swap is not a valid "
                f"one-hot prompt and would leave the design's row space"
            )
    n_matched = 0

    # now that train_ds is loaded, validate the artifact against its config
    if "linear_probe_direction" in methods and probe_artifact_path is not None:
        probe_artifact = load_and_validate_probe_artifact(probe_artifact_path, train_ds)

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
        if fill_policy == "source_prompt":
            pack_template = get_full_concat_embedding_untruncated(holdout_ds, idx).cpu()
        else:
            pack_template = train_mean_template
        x_tgt = x_tgt.unsqueeze(0).to(dev_dec).float()
        x_tgt_base = x_tgt if dev_base == dev_dec else x_tgt.to(dev_base)
        mask_row_ds = mask_row_ds.to(dev_dec)

        # Targeted-concept mode (E3): restrict to tuples containing the target concept and
        # always edit that concept, instead of sampling an attribute at random. Skipping here
        # rather than filtering upfront keeps `idx` aligned with the holdout row id, so cache
        # entries and per-sample rows stay comparable across runs.
        if target_pid is not None:
            if mask_t[target_pid].item() <= 0:
                continue
            if max_matched is not None and n_matched >= max_matched:
                break
            n_matched += 1

        sample_rng = random.Random(base_seed + idx)
        do_drop = locality_drop_one_attr and len(attrs) > 1
        do_swap = locality_swap_one_attr and len(attrs) > 1

        edit_pid: int | None = None
        edit_attribute = ""
        swap_target_pid: int | None = None
        swap_target_attribute = ""
        ref_tid_swap: int | None = None
        residual_prompt = ""
        swapped_prompt = ""

        if do_drop or do_swap:
            edit_pid = (
                target_pid if target_pid is not None else _choose_edit_pid(mask_t, sample_rng)
            )
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
                    swap_target_pid = (
                        replacement_pid
                        if replacement_pid is not None
                        else _choose_swap_target_pid(holdout_ds, edit_pid, sample_rng)
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

        if want_ssae:
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
        else:
            pred_ssae = None
            mse_ssae = float("nan")
            cos_ssae = float("nan")

        M_row_cpu = mask_row_ds.float().cpu().unsqueeze(0)
        pred_ma = predict_mean_arithmetic(mu_ma, deltas_ma, M_row_cpu).to(dev_base)
        mse_ma = torch.nn.functional.mse_loss(pred_ma, x_tgt_base).item()

        pred_ridge = predict_linear(M_row_cpu, W_ridge).to(dev_base)
        mse_ridge = torch.nn.functional.mse_loss(pred_ridge, x_tgt_base).item()

        pred_ssae_pre = pred_ma_pre = pred_ridge_pre = None
        if do_drop:
            if drop_operator == "category_marginal":
                mask_pre_ds = _category_marginal_mask(holdout_ds, mask_row_ds, edit_pid).to(dev_dec)
            else:
                mask_pre_ds = mask_row_ds.clone().float()
                mask_pre_ds[edit_pid] = 0
            if want_ssae:
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
            pred_ma_pre = predict_mean_arithmetic(mu_ma, deltas_ma, M_pre_cpu).to(dev_base)
            pred_ridge_pre = predict_linear(M_pre_cpu, W_ridge).to(dev_base)

        pred_ssae_swap = pred_ma_swap = pred_ridge_swap = None
        if do_swap:
            mask_swap_ds = mask_row_ds.clone()
            mask_swap_ds[edit_pid] = 0
            mask_swap_ds[swap_target_pid] = 1
            ref_tid_swap = _find_ref_training_tid(train_mask, mask_swap_ds, swap_target_pid)
            if want_ssae:
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
            pred_ma_swap = predict_mean_arithmetic(mu_ma, deltas_ma, M_swap_cpu).to(dev_base)
            pred_ridge_swap = predict_linear(M_swap_cpu, W_ridge).to(dev_base)

        for method in methods:
            is_cached_method = cache is not None and method in cache_methods
            seed_i = _sample_seed(base_seed, idx)

            if is_cached_method:
                out_path = cache.image_path(method, idx, "post")
                cache.ensure_variant_dir(method, "post")
            else:
                out_path = img_root / method / f"{idx:05d}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)

            post_cached = is_cached_method and out_path.exists() and cache.has_row(method, idx)
            if method == "gt_embed":
                ref_paths[idx] = out_path

            if post_cached:
                # image and row already in cache; nothing to generate for this variant
                pass
            else:
                if method == "gt_embed":
                    pe, pp = pack_sd3_from_truncated_normalized(holdout_ds, idx, x_tgt.cpu(), template=pack_template)
                elif method == "ssae_compose":
                    pe, pp = pack_sd3_from_truncated_normalized(
                        holdout_ds, idx, pred_ssae.detach().cpu(), template=pack_template
                    )
                elif method == "mean_arithmetic":
                    pe, pp = pack_sd3_from_truncated_normalized(holdout_ds, idx, pred_ma.cpu(), template=pack_template)
                elif method == "ridge_embed":
                    pe, pp = pack_sd3_from_truncated_normalized(holdout_ds, idx, pred_ridge.cpu(), template=pack_template)
                elif method == "native_prompt":
                    pe, pp = None, None
                elif method == "prompt_only":
                    pe, pp = None, None
                elif method == "prompt_modified_packed":
                    pe, pp = _encode_and_pack_prompt(
                        gen, holdout_ds, prompt_text, seed_i, pack_template
                    )
                elif method == "linear_probe_direction":
                    # post is the true source embedding — same image as gt_embed
                    # hard-link from the gt_embed post image instead of rendering again
                    gt_post_path = ref_paths.get(idx)
                    if gt_post_path is not None and gt_post_path.exists():
                        import os
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        if not out_path.exists():
                            try:
                                os.link(gt_post_path, out_path)
                            except OSError:
                                import shutil
                                shutil.copy2(gt_post_path, out_path)
                    pe, pp = None, None  # skip generation below
                else:
                    raise ValueError(f"Unknown method {method}")

                if not simulated:
                    if method == "native_prompt":
                        gen.generate_image_from_prompt_native(
                            prompt_text, out_path, use_negative_prompts=False, seed=seed_i
                        )
                    elif method == "prompt_only":
                        gen.generate_image_from_prompt(
                            prompt_text, out_path, use_negative_prompts=False, seed=seed_i
                        )
                    elif method == "linear_probe_direction":
                        pass  # hard-linked above; no diffusion call needed
                    else:
                        pe = pe.to(sd_device)
                        pp = pp.to(sd_device)
                        gen.generate_image_from_embd(pe, pp, out_path, seed=seed_i)
                else:
                    if method != "linear_probe_direction":
                        _write_placeholder_png(out_path)
                    elif not out_path.exists():
                        _write_placeholder_png(out_path)

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
                # E3 efficacy: CLIP against the *target phrase alone*, on the unedited render
                # and on each edited render. The drop from one to the other is the efficacy
                # signal; comparing whole-prompt CLIP cannot isolate the target concept.
                "clip_normal_vs_target_phrase": "",
                "clip_deleted_vs_target_phrase": "",
                "clip_swapped_vs_target_phrase": "",
                "clip_swapped_vs_replacement_phrase": "",
            }
            if not post_cached:
                if method == "ssae_compose":
                    row["mse_embedding_vs_gt"] = mse_ssae
                    row["cosine_embedding_vs_gt"] = cos_ssae
                elif method == "mean_arithmetic":
                    row["mse_embedding_vs_gt"] = mse_ma
                elif method == "ridge_embed":
                    row["mse_embedding_vs_gt"] = mse_ridge
                elif method == "linear_probe_direction":
                    # post is identical to gt_embed — mse is 0 by construction
                    row["mse_embedding_vs_gt"] = 0.0
                    row["cosine_embedding_vs_gt"] = 1.0
                    row["intervention_source"] = "true_holdout_topk_embedding"
                    if probe_artifact is not None:
                        row["probe_artifact"] = str(probe_artifact_path)
                        row["probe_delete_alpha"] = probe_artifact.get("delete_alpha", "")
                        row["probe_replace_alpha"] = probe_artifact.get("replace_alpha", "")

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
                    if edit_attribute:
                        row["clip_normal_vs_target_phrase"] = clip_scorer.image_text_cosine(
                            out_path, edit_attribute
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
                if is_cached_method:
                    pre_path = cache.image_path(method, idx, "pre_edit")
                    cache.ensure_variant_dir(method, "pre_edit")
                else:
                    pre_path = img_root_pre / method / f"{idx:05d}.png"
                    pre_path.parent.mkdir(parents=True, exist_ok=True)
                pre_cached = is_cached_method and pre_path.exists()

                if not pre_cached:
                    if method == "native_prompt":
                        pe_pre, pp_pre = None, None
                    elif method == "prompt_only":
                        pe_pre, pp_pre = None, None
                    elif method == "prompt_modified_packed":
                        pe_pre, pp_pre = _encode_and_pack_prompt(
                            gen, holdout_ds, residual_prompt, seed_i, pack_template
                        )
                    elif method == "ssae_compose":
                        pe_pre, pp_pre = pack_sd3_from_truncated_normalized(
                            holdout_ds, idx, pred_ssae_pre.detach().cpu(), template=pack_template
                        )
                    elif method == "mean_arithmetic":
                        pe_pre, pp_pre = pack_sd3_from_truncated_normalized(
                            holdout_ds, idx, pred_ma_pre.cpu(), template=pack_template
                        )
                    elif method == "ridge_embed":
                        pe_pre, pp_pre = pack_sd3_from_truncated_normalized(
                            holdout_ds, idx, pred_ridge_pre.cpu(), template=pack_template
                        )
                    elif method == "linear_probe_direction" and probe_artifact is not None:
                        # deletion: x_tgt - delete_alpha * delete_direction
                        d_del = probe_artifact["delete_direction"].to(x_tgt.device)
                        alpha_del = probe_artifact.get("delete_alpha") or 0.0
                        x_deleted = x_tgt - alpha_del * d_del
                        pe_pre, pp_pre = pack_sd3_from_truncated_normalized(
                            holdout_ds, idx, x_deleted.cpu(), template=pack_template
                        )
                    else:
                        raise ValueError(f"Unknown method {method}")

                    if not simulated:
                        if method == "native_prompt":
                            gen.generate_image_from_prompt_native(
                                residual_prompt, pre_path, use_negative_prompts=False, seed=seed_i
                            )
                        elif method == "prompt_only":
                            gen.generate_image_from_prompt(
                                residual_prompt, pre_path, use_negative_prompts=False, seed=seed_i
                            )
                        else:
                            pe_pre = pe_pre.to(sd_device)
                            pp_pre = pp_pre.to(sd_device)
                            gen.generate_image_from_embd(pe_pre, pp_pre, pre_path, seed=seed_i)
                    else:
                        _write_placeholder_png(pre_path)

                if not skip_pixel_metrics and not simulated and not pre_cached:
                    row["mse_pixel_pre_post_edit"] = pixel_mse(pre_path, out_path)
                    row["ssim_pre_post_edit"] = ssim(pre_path, out_path)

                if clip_scorer is not None and not pre_cached and edit_attribute:
                    row["clip_deleted_vs_target_phrase"] = clip_scorer.image_text_cosine(
                        pre_path, edit_attribute
                    )

            if do_swap and method != "gt_embed":
                if is_cached_method:
                    swap_path = cache.image_path(method, idx, "swapped")
                    cache.ensure_variant_dir(method, "swapped")
                else:
                    swap_path = img_root_swap / method / f"{idx:05d}.png"
                    swap_path.parent.mkdir(parents=True, exist_ok=True)
                swap_cached = is_cached_method and swap_path.exists()

                if not swap_cached:
                    if method == "native_prompt":
                        pe_sw, pp_sw = None, None
                    elif method == "prompt_only":
                        pe_sw, pp_sw = None, None
                    elif method == "prompt_modified_packed":
                        pe_sw, pp_sw = _encode_and_pack_prompt(
                            gen, holdout_ds, swapped_prompt, seed_i, pack_template
                        )
                    elif method == "ssae_compose":
                        pe_sw, pp_sw = pack_sd3_from_truncated_normalized(
                            train_ds, ref_tid_swap, pred_ssae_swap.detach().cpu(), template=pack_template
                        )
                    elif method == "mean_arithmetic":
                        pe_sw, pp_sw = pack_sd3_from_truncated_normalized(
                            train_ds, ref_tid_swap, pred_ma_swap.cpu(), template=pack_template
                        )
                    elif method == "ridge_embed":
                        pe_sw, pp_sw = pack_sd3_from_truncated_normalized(
                            train_ds, ref_tid_swap, pred_ridge_swap.cpu(), template=pack_template
                        )
                    elif method == "linear_probe_direction" and probe_artifact is not None:
                        # replacement: x_tgt + replace_alpha * replace_direction
                        d_rep = probe_artifact["replace_direction"].to(x_tgt.device)
                        alpha_rep = probe_artifact.get("replace_alpha") or 0.0
                        x_replaced = x_tgt + alpha_rep * d_rep
                        pe_sw, pp_sw = pack_sd3_from_truncated_normalized(
                            holdout_ds, idx, x_replaced.cpu(), template=pack_template
                        )
                    else:
                        raise ValueError(f"Unknown method {method}")

                    if not simulated:
                        if method == "native_prompt":
                            gen.generate_image_from_prompt_native(
                                swapped_prompt, swap_path, use_negative_prompts=False, seed=seed_i
                            )
                        elif method == "prompt_only":
                            gen.generate_image_from_prompt(
                                swapped_prompt, swap_path, use_negative_prompts=False, seed=seed_i
                            )
                        else:
                            pe_sw = pe_sw.to(sd_device)
                            pp_sw = pp_sw.to(sd_device)
                            gen.generate_image_from_embd(pe_sw, pp_sw, swap_path, seed=seed_i)
                    else:
                        _write_placeholder_png(swap_path)

                if not skip_pixel_metrics and not simulated and not swap_cached:
                    row["mse_pixel_swap_vs_normal"] = pixel_mse(swap_path, out_path)
                    row["ssim_swap_vs_normal"] = ssim(swap_path, out_path)

                if clip_scorer is not None and not swap_cached:
                    row["clip_swap_image_vs_swapped_prompt"] = (
                        clip_scorer.image_text_cosine(swap_path, swapped_prompt)
                    )
                    if edit_attribute:
                        row["clip_swapped_vs_target_phrase"] = clip_scorer.image_text_cosine(
                            swap_path, edit_attribute
                        )
                    if swap_target_attribute:
                        row["clip_swapped_vs_replacement_phrase"] = (
                            clip_scorer.image_text_cosine(swap_path, swap_target_attribute)
                        )

            if is_cached_method:
                cached_row = cache.get_row(method, idx)
                merged = _merge_row(cached_row, row) if cached_row else row
                cache.upsert_row(merged)
            else:
                rows.append(row)

    local_methods = [m for m in methods if not (cache is not None and m in cache_methods)]

    csv_path = output_dir / "per_sample.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    summary = _aggregate_summary(
        rows,
        methods=local_methods,
        clip_failure_threshold=clip_failure_threshold,
        n_bootstrap=n_bootstrap,
        base_seed=base_seed,
    )
    packer_meta = _packer_metadata(fill_policy, drop_operator)
    summary["ridge_lambda"] = ridge_lambda
    summary["methods"] = list(local_methods)
    summary["fill_policy"] = fill_policy
    summary["drop_operator"] = drop_operator
    summary["packer_fingerprint"] = packer_meta

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

    if cache is not None:
        cache_method_list = [m for m in methods if m in cache_methods]
        cache_rows = list(cache.rows.values())
        cache_summary = _aggregate_summary(
            cache_rows,
            methods=cache_method_list,
            clip_failure_threshold=clip_failure_threshold,
            n_bootstrap=n_bootstrap,
            base_seed=base_seed,
        )
        cache_summary["ridge_lambda"] = ridge_lambda
        cache_summary["methods"] = list(cache_method_list)
        cache_summary["fill_policy"] = fill_policy
        cache_summary["drop_operator"] = drop_operator
        cache_summary["packer_fingerprint"] = packer_meta
        cache.write(
            methods=cache_method_list,
            summary=cache_summary,
            extra_manifest={
                "locality_drop_populated": locality_drop_one_attr,
                "locality_swap_populated": locality_swap_one_attr,
                "n_samples": n,
                "fill_policy": fill_policy,
                "drop_operator": drop_operator,
            },
        )
    else:
        cache_method_list = []

    # Legacy manifest kept for the existing report builders, now with the conditioning
    # record AUG-01 requires. Paths are stored POSIX-relative-safe (resolve()) so a
    # manifest written on one machine is at least diagnosable on another.
    manifest = {
        "dataset_id": cache.dataset_id if cache is not None else None,
        "baseline_cache_dir": str(cache.dir.resolve()) if cache is not None else None,
        "baseline_methods": cache_method_list,
        "local_methods": local_methods,
        "methods": list(methods),
        "base_seed": base_seed,
        "ridge_lambda": ridge_lambda,
        "locality_drop_one_attr": locality_drop_one_attr,
        "locality_swap_one_attr": locality_swap_one_attr,
        "n_samples": n,
        # AUG-01 acceptance criterion: native-vs-packed conditioning is explicit, per
        # method, in every manifest — so no downstream table can silently place a native
        # and a packed row in the same statistical comparison.
        "method_conditioning": conditioning_map(methods),
        "packer_fingerprint": packer_meta,
        "truncate_embds_topk": tp.get("truncate_embds_topk"),
        "fill_policy": fill_policy,
        "drop_operator": drop_operator,
        "target_property": target_property,
        "replacement_property": replacement_property,
        "n_matched": n_matched if target_pid is not None else None,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # AUG-02 / E0: full provenance — git SHA, resolved config, seeds, dataset and packer
    # fingerprints, checkpoint hash, exact command line.
    write_run_manifest(
        output_dir,
        run_kind="image_benchmark",
        config=tp,
        seed=base_seed,
        dataset=holdout_ds,
        extra_datasets={"train": train_ds},
        packer_fingerprint=packer_meta,
        model_fingerprint=checkpoint_fingerprint(checkpoint_dir),
        extra={
            "benchmark": manifest,
            "sd3_fingerprint": ImageGenerator.fingerprint(),
            "simulated": simulated,
            "max_samples": max_samples,
            "n_samples_scored": n,
            "clip_failure_threshold": clip_failure_threshold,
            "n_bootstrap": n_bootstrap,
            "skip_lpips": skip_lpips,
            "skip_pixel_metrics": skip_pixel_metrics,
            "skip_dino": skip_dino,
            "use_baseline_cache": use_baseline_cache,
            "baselines_only": baselines_only,
            "fill_policy": fill_policy,
            "drop_operator": drop_operator,
        },
    )

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

        def _col(name):
            return [float(r[name]) for r in xs
                    if r.get(name) not in ("", None)]

        tgt_normal = _col("clip_normal_vs_target_phrase")
        tgt_deleted = _col("clip_deleted_vs_target_phrase")
        tgt_swapped = _col("clip_swapped_vs_target_phrase")
        repl_swapped = _col("clip_swapped_vs_replacement_phrase")

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
            # E3 efficacy on the target concept. The deletion/replacement drop is the
            # signal; the absolute values are not comparable across concepts.
            "clip_normal_vs_target_phrase": _ci_dict(tgt_normal, n_bootstrap, seed_m + 25),
            "clip_deleted_vs_target_phrase": _ci_dict(tgt_deleted, n_bootstrap, seed_m + 27),
            "clip_swapped_vs_target_phrase": _ci_dict(tgt_swapped, n_bootstrap, seed_m + 29),
            "clip_swapped_vs_replacement_phrase": _ci_dict(repl_swapped, n_bootstrap, seed_m + 31),
            "efficacy_delta_clip_deletion": (
                float(np.mean(tgt_normal) - np.mean(tgt_deleted))
                if tgt_normal and tgt_deleted else float("nan")
            ),
            "efficacy_delta_clip_replacement": (
                float(np.mean(tgt_normal) - np.mean(tgt_swapped))
                if tgt_normal and tgt_swapped else float("nan")
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
    p.add_argument(
        "--ssae_device",
        type=str,
        default=None,
        help=(
            "Where to load the SSAE decoder. Defaults to --sd_device. Set to 'cpu' or a "
            "different CUDA index to free GPU memory shared with the SD3.5 pipeline "
            "(useful when the decoder is trained without top-k truncation)."
        ),
    )
    p.add_argument(
        "--baseline_device",
        type=str,
        default=None,
        help=(
            "Where to run the ridge / mean-arithmetic baseline predictions + MSE against "
            "the holdout target. Fits always run on CPU regardless. Defaults to "
            "--ssae_device; set to 'cpu' to keep the baselines fully off GPU even when "
            "the SSAE runs on CUDA."
        ),
    )
    p.add_argument("--clip_device", type=str, default=None)
    p.add_argument("--clip_failure_threshold", type=float, default=0.2)
    p.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help=(
            "Comma-separated method keys. Known keys: "
            + ", ".join(METHOD_ORDER)
            + ". Rendering ladder: 'native_prompt' passes text directly to the pipeline "
            "without encode_prompt (true native). 'prompt_only' encodes and immediately "
            "decodes through the full embedding (exact round-trip, kept for cache "
            "compatibility). 'prompt_modified_packed' re-encodes and packs to top-k "
            "(controlled subspace). Do not conflate or average them."
        ),
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
    p.add_argument(
        "--baseline_cache_root",
        type=Path,
        default=DEFAULT_BASELINE_CACHE_ROOT,
        help=(
            "Root directory holding the shared per-dataset baseline cache. Non-SSAE methods "
            "(gt_embed, mean_arithmetic, ridge_embed, native_prompt, prompt_only, "
            "prompt_modified_packed) are populated here once per (holdout, training data, "
            "base_seed, ridge_lambda, SD3.5 fingerprint) tuple and reused by subsequent runs. "
            "Default: results/bench_baseline_cache."
        ),
    )
    p.add_argument(
        "--no_baseline_cache",
        action="store_true",
        help=(
            "Disable the shared baseline cache; write baselines into the run folder as in "
            "the legacy monolithic layout."
        ),
    )
    p.add_argument(
        "--target_property",
        type=str,
        default=None,
        help=(
            "E3 targeted-concept mode. Restrict evaluation to holdout tuples containing this "
            "property and always edit *it*, instead of sampling a random active attribute. "
            "Combine with --locality_drop_one_attr (off-manifold deletion) and/or "
            "--locality_swap_one_attr (on-manifold replacement). Example: 'holding a gun'. "
            "IMPORTANT: pre-edit/swapped cache entries are keyed only by (method, sample_idx), "
            "so different targets MUST use different --baseline_cache_root values or they will "
            "overwrite each other's edited renders."
        ),
    )
    p.add_argument(
        "--replacement_property",
        type=str,
        default=None,
        help=(
            "Fixed on-manifold counterfactual for --locality_swap_one_attr, e.g. "
            "'holding a coffee' for target 'holding a gun'. Must be in the same category as "
            "the target; a cross-category swap is not a valid one-hot prompt. Defaults to a "
            "per-sample random same-category value."
        ),
    )
    p.add_argument(
        "--max_matched",
        type=int,
        default=None,
        help=(
            "Cap the number of tuples actually evaluated in targeted-concept mode. Unlike "
            "--max_samples (which caps how many holdout rows are scanned), this caps how many "
            "rows containing the target are scored, so the n per concept is predictable."
        ),
    )
    p.add_argument(
        "--fill_policy",
        type=str,
        default="train_mean",
        choices=["train_mean", "source_prompt"],
        help=(
            "Template for non-top-k coordinates when packing embedding-space predictions. "
            "train_mean matches the primary benchmark; source_prompt uses each row's original "
            "full prompt embedding and is the Q8 deployment-style robustness check."
        ),
    )
    p.add_argument(
        "--drop_operator",
        type=str,
        default="zero",
        choices=["zero", "category_marginal"],
        help=(
            "Operator for --locality_drop_one_attr. zero removes the target block; "
            "category_marginal replaces it with the uniform category marginal, the E4 "
            "identified-erasure operator. Replacement/swap renders are unchanged."
        ),
    )
    p.add_argument(
        "--baselines_only",
        action="store_true",
        help=(
            "Skip SSAE rendering; populate the baseline cache and exit. Useful for pre-"
            "warming a dataset's baselines before comparing several SSAE checkpoints."
        ),
    )
    p.add_argument(
        "--probe_artifact",
        type=Path,
        default=None,
        help=(
            "Path to a probe_artifact.pt file produced by run_probe_intervention_fit. "
            "Required when 'linear_probe_direction' is in --methods."
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
        ssae_device=args.ssae_device,
        baseline_device=args.baseline_device,
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
        baseline_cache_root=args.baseline_cache_root,
        use_baseline_cache=not args.no_baseline_cache,
        baselines_only=args.baselines_only,
        target_property=args.target_property,
        replacement_property=args.replacement_property,
        max_matched=args.max_matched,
        fill_policy=args.fill_policy,
        drop_operator=args.drop_operator,
        probe_artifact_path=args.probe_artifact,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
