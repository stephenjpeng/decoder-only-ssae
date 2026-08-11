"""Concept-strength / magnitude-sensitivity sweep.

For each concept, samples prompts where the concept is active (or inactive, with
``--sample_source absent``) and re-decodes them with that concept's mask entry replaced
by a scalar from ``--magnitudes`` (e.g. -10, -2, -1, 0, 1, 2, 5, 10) instead of the usual
0/1. Reports CLIP concept-presence score as a function of magnitude, and whether negative
values behave differently from the explicit mask=0 removal baseline.

Output layout mirrors ``run_image_benchmark.py`` (``per_sample.csv`` with a ``method``
column, images under ``images/<method>/<idx>.png``) so ``evaluation.run_vlm_openai_batch``
can be pointed at this output directory unmodified for an optional VLM concept-presence
judge.

Fill policy for non-top-k dimensions
------------------------------------
Uses **train-mean fill**, matching ``run_image_benchmark``. This was previously oracle fill:
the packer was called without ``template=``, so the ~1.36M coordinates the SSAE does not
predict came from the *true* holdout embedding of the row being edited. That leaks ground
truth into every render and, worse for this script's purpose, makes the resulting
efficacy/collateral numbers incomparable with the E3 benchmark, which uses train-mean fill.
Since the whole point of the magnitude sweep is to place methods on a shared
efficacy-vs-collateral Pareto plot against E3's rows, the two had to agree.

Pass ``--oracle_fill`` to restore the old behaviour if you need to reproduce a pre-fix
artifact. Do not mix the two in one comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from scipy.stats import spearmanr

from evaluation.bootstrap import bootstrap_mean_ci
from evaluation.clip_scorer import CLIPScorer
from evaluation.composition import property_block_means_trainable_inputs
from evaluation.io import ensure_folder_path, h5_dataset_for_folder, load_decoder_checkpoint
from evaluation.magnitude import predict_embedding_magnitude, scaled_mask_row
from evaluation.sd3_pack import (
    compute_or_load_full_mean,
    pack_sd3_from_truncated_normalized,
    packer_fingerprint,
)
from inference.image_generation.image_generator import ImageGenerator
from trainings.utils.run_manifest import checkpoint_fingerprint, write_run_manifest


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _magnitude_key(m: float) -> str:
    return f"{m:g}"


def _magnitude_slug(m: float) -> str:
    return _magnitude_key(m).replace("-", "neg").replace(".", "p")


def _sample_seed(base_seed: int, idx: int, method: str) -> int:
    h = int.from_bytes(hashlib.md5(method.encode()).digest()[:4], "big")
    return base_seed + idx * 1_000_003 + (h % 1_000_000)


def _write_placeholder_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224), color=(120, 120, 120)).save(path)


def _ci_dict(vals: list[float], n_boot: int, seed: int) -> dict:
    if not vals:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    mean, lo, hi = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)}


def _select_sample_indices(
    mask_reduced: torch.Tensor,
    pid: int,
    *,
    source: str,
    n_samples: int,
    seed: int,
) -> list[int]:
    col = mask_reduced[:, pid]
    if source == "present":
        candidates = (col > 0).nonzero(as_tuple=True)[0].tolist()
    elif source == "absent":
        candidates = (col == 0).nonzero(as_tuple=True)[0].tolist()
    else:
        raise ValueError(f"Unknown sample_source: {source}")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return sorted(candidates[:n_samples])


def _active_phrases_excluding(dataset, mask_row: torch.Tensor, exclude_pid: int) -> list[str]:
    props = dataset.properties
    m = mask_row.flatten().long()
    out = []
    for pid in range(m.numel()):
        if pid != exclude_pid and m[pid].item() > 0:
            out.append(props.pid_to_property[pid])
    return out


def run_magnitude_sensitivity(
    checkpoint_dir: Path,
    data_folder: Path,
    output_dir: Path,
    *,
    concepts: list[str],
    magnitudes: list[float],
    n_samples_per_concept: int = 5,
    sample_source: str = "present",
    base_seed: int = 0,
    simulated: bool = False,
    sd_device: str = "cuda",
    clip_device: str | None = None,
    n_bootstrap: int = 2000,
    skip_plot: bool = False,
    oracle_fill: bool = False,
) -> dict:
    clip_device = clip_device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_folder = Path(ensure_folder_path(data_folder))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_root = output_dir / "images"
    img_root.mkdir(exist_ok=True)

    decoder, tp, train_ds = load_decoder_checkpoint(checkpoint_dir, device=sd_device)
    data_ds = h5_dataset_for_folder(checkpoint_dir, data_folder)
    decoder.eval()

    model_name = tp["model_name"]
    n_repeat = int(tp["n_repeat"])
    dev_dec = torch.device(sd_device)

    train_mask = train_ds.mask_reduced.to(dev_dec)
    block_means = None
    if model_name == "model_trainable_inputs":
        block_means = property_block_means_trainable_inputs(
            decoder, train_mask, n_repeat, dev_dec
        )

    with open(data_folder / "prompts.json", "r", encoding="utf-8") as f:
        prompts_meta = json.load(f)

    # Train-mean fill template, shared across every magnitude and sample so the only thing
    # varying along a curve is the edited concept. `None` restores the legacy oracle fill.
    pack_template = None if oracle_fill else compute_or_load_full_mean(train_ds).cpu()

    gen = ImageGenerator(simulated=simulated, device=sd_device)
    clip_scorer = None if simulated else CLIPScorer(device=clip_device)

    rows: list[dict] = []

    for ci, concept in enumerate(concepts):
        if concept not in data_ds.properties.property_to_pid:
            raise ValueError(f"concept '{concept}' not found in properties.json")
        pid = data_ds.properties.property_to_pid[concept]

        sample_idxs = _select_sample_indices(
            data_ds.mask_reduced,
            pid,
            source=sample_source,
            n_samples=n_samples_per_concept,
            seed=base_seed + ci,
        )

        for idx in sample_idxs:
            prompt_text = prompts_meta[idx]["prompt"]
            mask_row = data_ds.mask_reduced[idx]
            other_attrs = _active_phrases_excluding(data_ds, mask_row, pid)

            for magnitude in magnitudes:
                mask_row_scaled = scaled_mask_row(mask_row, pid, magnitude)
                pred = predict_embedding_magnitude(
                    decoder,
                    model_name,
                    mask_row_scaled,
                    n_repeat=n_repeat,
                    device=dev_dec,
                    mask_reduced_train=train_mask,
                    block_means=block_means,
                )

                method = f"{_slug(concept)}__m{_magnitude_slug(magnitude)}"
                out_path = img_root / method / f"{idx:05d}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                seed_i = _sample_seed(base_seed, idx, method)

                if not simulated:
                    pe, pp = pack_sd3_from_truncated_normalized(
                        data_ds, idx, pred.detach().cpu(), template=pack_template
                    )
                    gen.generate_image_from_embd(
                        pe.to(sd_device), pp.to(sd_device), out_path, seed=seed_i
                    )
                else:
                    _write_placeholder_png(out_path)

                row = {
                    "sample_idx": idx,
                    "concept": concept,
                    "pid": pid,
                    "magnitude": magnitude,
                    "sample_source": sample_source,
                    "method": method,
                    "prompt": prompt_text,
                    "clip_image_vs_concept": "",
                    "clip_image_vs_full_prompt": "",
                    "clip_mean_vs_other_attrs": "",
                    "clip_min_vs_other_attrs": "",
                }

                if clip_scorer is not None:
                    row["clip_image_vs_concept"] = clip_scorer.image_text_cosine(
                        out_path, concept
                    )
                    row["clip_image_vs_full_prompt"] = clip_scorer.image_text_cosine(
                        out_path, prompt_text
                    )
                    if other_attrs:
                        al = clip_scorer.image_attribute_alignment(out_path, other_attrs)
                        row["clip_mean_vs_other_attrs"] = al["mean_cosine_attr"]
                        row["clip_min_vs_other_attrs"] = al["min_cosine_attr"]

                rows.append(row)

    csv_path = output_dir / "per_sample.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    summary = _aggregate_summary(
        rows, concepts=concepts, magnitudes=magnitudes, n_bootstrap=n_bootstrap, base_seed=base_seed
    )
    summary["model_name"] = model_name
    summary["sample_source"] = sample_source
    summary["magnitudes"] = magnitudes
    # Recorded because magnitude curves are only comparable with E3/AUG-05 rows under the
    # same fill policy; a mismatch here silently invalidates any joint Pareto plot.
    summary["fill_policy"] = "oracle_fill" if oracle_fill else "train_mean_fill"
    summary["packer_fingerprint"] = packer_fingerprint()
    summary["truncate_embds_topk"] = tp.get("truncate_embds_topk")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_run_manifest(
        output_dir,
        run_kind="magnitude_sensitivity",
        config=tp,
        seed=base_seed,
        dataset=data_ds,
        extra_datasets={"train": train_ds},
        packer_fingerprint=packer_fingerprint(),
        model_fingerprint=checkpoint_fingerprint(checkpoint_dir),
        extra={
            "concepts": concepts,
            "magnitudes": magnitudes,
            "n_samples_per_concept": n_samples_per_concept,
            "sample_source": sample_source,
            "fill_policy": "oracle_fill" if oracle_fill else "train_mean_fill",
            "simulated": simulated,
        },
    )

    if not skip_plot and not simulated:
        _plot_magnitude_curves(summary, output_dir / "magnitude_sensitivity.png")

    return summary


def _aggregate_summary(
    rows: list[dict],
    *,
    concepts: list[str],
    magnitudes: list[float],
    n_bootstrap: int,
    base_seed: int,
) -> dict:
    by_concept_mag: dict[tuple[str, float], list[float]] = defaultdict(list)
    by_concept_mag_other: dict[tuple[str, float], list[float]] = defaultdict(list)
    by_concept_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if r["clip_image_vs_concept"] == "":
            continue
        score = float(r["clip_image_vs_concept"])
        by_concept_mag[(r["concept"], r["magnitude"])].append(score)
        by_concept_pairs[r["concept"]].append((r["magnitude"], score))
        if r["clip_mean_vs_other_attrs"] != "":
            by_concept_mag_other[(r["concept"], r["magnitude"])].append(
                float(r["clip_mean_vs_other_attrs"])
            )

    per_concept_magnitude: dict[str, dict] = {}
    monotonicity: dict[str, dict] = {}
    negative_vs_removal: dict[str, dict] = {}

    for ci, concept in enumerate(concepts):
        per_mag = {}
        for mi, m in enumerate(magnitudes):
            vals = by_concept_mag.get((concept, m), [])
            other_vals = by_concept_mag_other.get((concept, m), [])
            seed_m = base_seed + 101 * ci + 7 * mi
            per_mag[_magnitude_key(m)] = {
                "concept_presence": _ci_dict(vals, n_bootstrap, seed_m),
                "non_target_preservation": _ci_dict(other_vals, n_bootstrap, seed_m + 3),
            }
        per_concept_magnitude[concept] = per_mag

        pairs = by_concept_pairs.get(concept, [])
        if len(pairs) >= 2:
            mags = [p[0] for p in pairs]
            scores = [p[1] for p in pairs]
            rho, pval = spearmanr(mags, scores)
            monotonicity[concept] = {
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "n": len(pairs),
            }

        zero_ci = per_mag.get(_magnitude_key(0.0), {}).get("concept_presence")
        if zero_ci is not None and zero_ci["n"] > 0:
            neg_comparisons = {}
            for m in magnitudes:
                if m >= 0:
                    continue
                neg_ci = per_mag[_magnitude_key(m)]["concept_presence"]
                if neg_ci["n"] == 0:
                    continue
                overlap = not (
                    neg_ci["ci_high"] < zero_ci["ci_low"]
                    or neg_ci["ci_low"] > zero_ci["ci_high"]
                )
                neg_comparisons[_magnitude_key(m)] = {
                    "ci_overlaps_removal_baseline": overlap,
                    "differs_from_removal": not overlap,
                }
            negative_vs_removal[concept] = neg_comparisons

    return {
        "per_concept_magnitude": per_concept_magnitude,
        "monotonicity": monotonicity,
        "negative_vs_removal": negative_vs_removal,
    }


def _plot_magnitude_curves(summary: dict, output_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for concept, per_mag in summary["per_concept_magnitude"].items():
        pairs = sorted(
            (
                (float(m), d["concept_presence"])
                for m, d in per_mag.items()
                if d["concept_presence"]["n"] > 0
            ),
            key=lambda t: t[0],
        )
        if not pairs:
            continue
        xs = [p[0] for p in pairs]
        means = [p[1]["mean"] for p in pairs]
        los = [p[1]["mean"] - p[1]["ci_low"] for p in pairs]
        his = [p[1]["ci_high"] - p[1]["mean"] for p in pairs]
        ax.errorbar(xs, means, yerr=[los, his], marker="o", capsize=3, label=concept)

    ax.set_xlabel("mask magnitude")
    ax.set_ylabel("CLIP image-vs-concept cosine")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data_folder", type=Path, required=True, help="Train or holdout folder.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--concepts",
        type=str,
        required=True,
        help="Comma-separated property phrases, e.g. 'holding a gun,A blond girl'.",
    )
    p.add_argument(
        "--magnitudes",
        type=str,
        default="-10,-2,-1,0,1,2,5,10",
        help="Comma-separated mask values to test (0 = explicit removal, 1 = normal presence).",
    )
    p.add_argument("--n_samples_per_concept", type=int, default=5)
    p.add_argument("--sample_source", type=str, default="present", choices=["present", "absent"])
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--simulated", action="store_true")
    p.add_argument("--sd_device", type=str, default="cuda")
    p.add_argument("--clip_device", type=str, default=None)
    p.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument("--skip_plot", action="store_true")
    p.add_argument(
        "--oracle_fill",
        action="store_true",
        help=(
            "Restore the legacy fill policy: take the ~1.36M non-top-k coordinates from the "
            "edited row's TRUE embedding instead of the training mean. This leaks ground "
            "truth and makes results incomparable with run_image_benchmark (which uses "
            "train-mean fill), so it is off by default. Only for reproducing pre-fix artifacts."
        ),
    )
    args = p.parse_args()

    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    magnitudes = [float(m.strip()) for m in args.magnitudes.split(",") if m.strip()]

    summary = run_magnitude_sensitivity(
        args.checkpoint,
        args.data_folder,
        args.output_dir,
        concepts=concepts,
        magnitudes=magnitudes,
        n_samples_per_concept=args.n_samples_per_concept,
        sample_source=args.sample_source,
        base_seed=args.base_seed,
        simulated=args.simulated,
        sd_device=args.sd_device,
        clip_device=args.clip_device,
        n_bootstrap=args.n_bootstrap,
        skip_plot=args.skip_plot,
        oracle_fill=args.oracle_fill,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
