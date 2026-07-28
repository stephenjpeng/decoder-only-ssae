"""Top-k ceiling experiment: generate images from k-truncated GT embeddings.

For each k the ground-truth embedding keeps its top-k dims (ranked by max-min
range across the dataset, matching `H5Dataset.get_indices_truncate_embds_topk`)
and the remaining dims are replaced with the per-dim mean. The resulting image
is an upper bound on what any SSAE trained at that k could achieve, since the
SSAE at best reconstructs the top-k slots and the non-top-k dims come from a
prompt-independent baseline (here, the mean; at inference in `abstract.py` they
come from a specific training-set embedding, but the mean is the closest
prompt-agnostic proxy for a ceiling curve).

PCA is intentionally disabled so we can sweep k past the sample-count limit of
`torch.pca_lowrank` (which requires k <= min(n_samples, n_features)).

Usage:
    python -m evaluation.run_topk_ceiling \
        --dataset_folder data/sd35_prompts_v1/ \
        --prompt_indices 0,42,100 \
        --ks 100,500,1000,2000,3000,3500 \
        --out results/topk_ceiling/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import torch

from backbones import get_backbone
from trainings.config.config import initialise_instance
from trainings.dataloader.dataloader import H5Dataset


def _load_dataset(folder_path: Path) -> H5Dataset:
    tp = {
        "folder_path": str(folder_path).rstrip("/") + "/",
        "truncate_embds_topk": None,
        "pca_rotation": False,
        "normalize": None,
        "logger": None,
        "simulated": False,
    }
    return initialise_instance(H5Dataset, tp)


def _dim_stats(
    dataset: H5Dataset,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Streaming per-dim mean, variance, max, min over the full dataset.

    We keep accumulators in float64 because dim_x can be ~1.36M for SD3.5 and
    float32 sums of ~4k samples drift enough to distort the variance ordering.
    """
    n = len(dataset)
    x0, _ = dataset[0]

    running_sum = x0.to(torch.float64).clone()
    running_sq = (x0.to(torch.float64) ** 2).clone()
    dim_max = x0.clone()
    dim_min = x0.clone()

    for i in range(1, n):
        x, _ = dataset[i]
        x64 = x.to(torch.float64)
        running_sum += x64
        running_sq += x64 * x64
        dim_max = torch.maximum(dim_max, x)
        dim_min = torch.minimum(dim_min, x)

    mean = (running_sum / n).to(torch.float32)
    var = ((running_sq / n) - (running_sum / n) ** 2).clamp_min_(0.0).to(torch.float32)
    return mean, var, dim_max, dim_min


def _split_streams(flat: torch.Tensor, stream_specs) -> dict[str, torch.Tensor]:
    streams: dict[str, torch.Tensor] = {}
    offset = 0
    for spec in stream_specs:
        n = spec.flat_dim
        streams[spec.name] = flat[offset : offset + n].reshape(*spec.shape)
        offset += n
    return streams


def _variance_summary(
    variances: torch.Tensor,
    ranges: torch.Tensor,
    order: torch.Tensor,
    ks: Sequence[int],
    dim_x: int,
) -> list[dict]:
    total_var = float(variances.sum())
    total_range = float(ranges.sum())
    var_sorted_cum = torch.cumsum(variances[order], dim=0)
    range_sorted_cum = torch.cumsum(ranges[order], dim=0)

    rows = []
    for k in ks:
        rows.append(
            {
                "k": int(k),
                "pct_variance": float(var_sorted_cum[k - 1] / total_var * 100.0),
                "pct_range": float(range_sorted_cum[k - 1] / total_range * 100.0),
                "dim_x": int(dim_x),
            }
        )
    return rows


def _save_summary(rows: list[dict], out_dir: Path) -> None:
    with open(out_dir / "variance_summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "variance_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "pct_variance", "pct_range", "dim_x"])
        w.writeheader()
        for row in rows:
            w.writerow(row)


def run(
    dataset_folder: Path,
    prompt_indices: Sequence[int],
    ks: Sequence[int],
    out_dir: Path,
    *,
    device: str = "cuda",
    seed: int = 0,
    generate_images: bool = True,
    include_gt: bool = True,
    include_mean_baseline: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(dataset_folder)
    with open(Path(dataset_folder, "prompts.json"), "r") as f:
        prompts = json.load(f)

    print(f"Computing per-dim stats over {len(dataset)} samples, dim={dataset.dim_x}...")
    mean, variances, dim_max, dim_min = _dim_stats(dataset)
    ranges = dim_max - dim_min
    order = ranges.argsort(descending=True)

    ks_clean = sorted({int(k) for k in ks if 0 < int(k) <= dataset.dim_x})
    if not ks_clean:
        raise ValueError(f"No valid k in {list(ks)} for dim_x={dataset.dim_x}")

    rows = _variance_summary(variances, ranges, order, ks_clean, dataset.dim_x)
    _save_summary(rows, out_dir)
    for row in rows:
        print(
            f"k={row['k']:>6d}  pct_variance={row['pct_variance']:6.3f}%  "
            f"pct_range={row['pct_range']:6.3f}%"
        )

    if not generate_images:
        return {"variance_summary": rows}

    backbone = get_backbone(dataset.backbone_name, device=device, **dataset.backbone_kwargs)
    backbone.load()

    per_prompt = []
    for idx in prompt_indices:
        if not (0 <= idx < len(dataset)):
            print(f"[skip] idx={idx} out of range")
            continue

        entry = prompts[idx]
        prompt_text = entry["prompt"] if isinstance(entry, dict) else str(entry)
        prompt_dir = out_dir / f"prompt_{idx:06d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        with open(prompt_dir / "prompt.txt", "w") as f:
            f.write(prompt_text + "\n")
        print(f"[idx={idx}] {prompt_text}")

        gt_flat, _ = dataset[idx]

        if include_gt:
            backbone.decode(
                streams=_split_streams(gt_flat, dataset.stream_specs),
                output_path=str(prompt_dir / "gt.png"),
                seed=seed,
            )
        if include_mean_baseline:
            backbone.decode(
                streams=_split_streams(mean, dataset.stream_specs),
                output_path=str(prompt_dir / "mean_baseline.png"),
                seed=seed,
            )

        for k in ks_clean:
            topk = order[:k]
            recon = mean.clone()
            recon[topk] = gt_flat[topk]
            backbone.decode(
                streams=_split_streams(recon, dataset.stream_specs),
                output_path=str(prompt_dir / f"topk_{k:06d}.png"),
                seed=seed,
            )

        per_prompt.append(
            {"idx": int(idx), "prompt": prompt_text, "dir": str(prompt_dir)}
        )

    with open(out_dir / "prompts_generated.json", "w") as f:
        json.dump(per_prompt, f, indent=2)

    return {"variance_summary": rows, "prompts": per_prompt}


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_folder", type=Path, required=True,
                   help="Folder containing embds/ and prompts.json (H5Dataset root).")
    p.add_argument("--prompt_indices", type=_parse_int_list, required=True,
                   help="Comma-separated prompt indices to reconstruct.")
    p.add_argument("--ks", type=_parse_int_list, required=True,
                   help="Comma-separated top-k values to sweep.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_images", action="store_true",
                   help="Only compute variance summary; skip image generation.")
    p.add_argument("--skip_gt", action="store_true")
    p.add_argument("--skip_mean_baseline", action="store_true")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run(
        dataset_folder=args.dataset_folder,
        prompt_indices=args.prompt_indices,
        ks=args.ks,
        out_dir=args.out,
        device=device,
        seed=args.seed,
        generate_images=not args.skip_images,
        include_gt=not args.skip_gt,
        include_mean_baseline=not args.skip_mean_baseline,
    )


if __name__ == "__main__":
    main()
