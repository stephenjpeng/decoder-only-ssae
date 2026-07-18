"""Elbow analysis on the PCA spectrum of the embedding space.

Complements `elbow_topk.py`, which ranks dimensions by per-dim range. This
script instead computes the singular values of the centred embedding matrix
and locates the elbow of the explained-variance curve. The output is
diagnostic: unlike top-K dim selection, PCA truncation would require storing a
projection matrix and applying it in `H5Dataset.__getitem__`, which is not
wired into the training pipeline. Use this to decide whether a PCA-based
truncation would be substantially more efficient than the current range-based
top-K.

Since typical embedding matrices are wide (n_prompts << d), the singular
spectrum is computed via the Gram matrix `X_c @ X_c.T` (n x n), which is
tractable in memory even when the raw X is not.

Run from the repo root:
    python -m analysis.elbow_pca --folder_path results/mps_run/ --threshold 0.95
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.elbow_topk import detect_cumulative, detect_kneedle, detect_second_deriv
from trainings.dataloader.dataloader import H5Dataset


@dataclass
class PCAElbowResult:
    k_kneedle: int
    k_cumulative: int
    k_second_deriv: int
    n_components: int
    n_prompts: int
    n_dims: int
    threshold: float
    total_variance: float
    explained_variance_at_k_kneedle: float
    explained_variance_at_k_cumulative: float
    explained_variance_at_k_second_deriv: float


def compute_singular_values(X: torch.Tensor) -> np.ndarray:
    """Singular values (descending) of the centred embedding matrix.

    Uses the Gram trick: sigma_i^2 == eigenvalue_i of X_c @ X_c.T. For (n, d)
    with n << d this avoids ever materialising V of shape (d, d).
    """
    X_centered = X - X.mean(dim=0, keepdim=True)
    gram = X_centered @ X_centered.T
    eigvals = torch.linalg.eigvalsh(gram)
    eigvals = eigvals.flip(0)
    # eigh can produce tiny negative eigenvalues from floating-point noise.
    eigvals = torch.clamp(eigvals, min=0.0)
    return torch.sqrt(eigvals).cpu().numpy()


def analyse(singular_values: np.ndarray, threshold: float, n_prompts: int, n_dims: int) -> PCAElbowResult:
    explained_variance = singular_values**2
    total = float(explained_variance.sum())
    explained_variance_ratio = explained_variance / max(total, 1e-12)
    cum = np.cumsum(explained_variance_ratio)

    # Kneedle / second-derivative operate on the sorted magnitude curve; the shape
    # is identical for singular values and explained variance, but we key results
    # to explained variance because that's the reported metric.
    k_kneedle = detect_kneedle(explained_variance_ratio)
    k_cum = detect_cumulative(explained_variance_ratio, threshold)
    k_d2 = detect_second_deriv(explained_variance_ratio)

    return PCAElbowResult(
        k_kneedle=k_kneedle,
        k_cumulative=k_cum,
        k_second_deriv=k_d2,
        n_components=len(singular_values),
        n_prompts=n_prompts,
        n_dims=n_dims,
        threshold=threshold,
        total_variance=total,
        explained_variance_at_k_kneedle=float(cum[k_kneedle - 1]),
        explained_variance_at_k_cumulative=float(cum[k_cum - 1]),
        explained_variance_at_k_second_deriv=float(cum[k_d2 - 1]),
    )


def plot(singular_values: np.ndarray, result: PCAElbowResult, out_path: str) -> None:
    n = len(singular_values)
    ks = np.arange(1, n + 1)
    ev = singular_values**2
    ev_ratio = ev / max(ev.sum(), 1e-12)
    cum = np.cumsum(ev_ratio)

    fig, (ax_scree, ax_cum) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_scree.plot(ks, ev_ratio, color="black", linewidth=1.0)
    ax_scree.set_xscale("log")
    ax_scree.set_yscale("log")
    ax_scree.set_ylabel("explained variance ratio")
    ax_scree.set_title("PCA scree plot")
    _mark_k(ax_scree, result.k_kneedle, "kneedle", "tab:blue")
    _mark_k(ax_scree, result.k_cumulative, f"cum >= {result.threshold:.2f}", "tab:orange")
    _mark_k(ax_scree, result.k_second_deriv, "2nd deriv", "tab:green")
    ax_scree.legend(loc="upper right")

    ax_cum.plot(ks, cum, color="black", linewidth=1.0)
    ax_cum.set_xscale("log")
    ax_cum.axhline(result.threshold, color="tab:orange", linestyle=":", linewidth=1.0)
    ax_cum.set_ylabel("cumulative explained variance")
    ax_cum.set_xlabel("k (number of principal components, log scale)")
    _mark_k(ax_cum, result.k_kneedle, "kneedle", "tab:blue")
    _mark_k(ax_cum, result.k_cumulative, f"cum >= {result.threshold:.2f}", "tab:orange")
    _mark_k(ax_cum, result.k_second_deriv, "2nd deriv", "tab:green")

    fig.suptitle(
        f"PCA elbow (n_prompts={result.n_prompts}, n_dims={result.n_dims}, "
        f"max_components={result.n_components})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _mark_k(ax, k: int, label: str, color: str) -> None:
    ax.axvline(k, color=color, linestyle="--", linewidth=1.0, label=f"{label} (k={k})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--folder_path",
        required=True,
        help="Dataset folder containing embds/ (matches dataloader.folder_path).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Cumulative explained-variance threshold for the `cumulative` method.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Where to save the elbow figure. Defaults to <folder_path>/elbow_pca.png.",
    )
    parser.add_argument(
        "--json_out",
        default=None,
        help="Where to save the recommended-k JSON. Defaults to <folder_path>/elbow_pca.json.",
    )
    parser.add_argument(
        "--n_prompts_subsample",
        type=int,
        default=None,
        help=(
            "Optionally subsample this many prompts (post-load) before computing the "
            "spectrum. Useful when the full training set has too many prompts for "
            "eigh(Gram) to fit in memory."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for subsampling.",
    )
    args = parser.parse_args()

    folder_path = args.folder_path
    if not folder_path.endswith("/"):
        folder_path = folder_path + "/"

    out_png = args.out or os.path.join(folder_path, "elbow_pca.png")
    out_json = args.json_out or os.path.join(folder_path, "elbow_pca.json")

    dataset = H5Dataset(
        folder_path=folder_path,
        truncate_n_prompts=None,
        truncate_embds_topk=None,
        add_property_is_the_same=False,
        normalize=None,
    )

    X = dataset.get_X()
    n_prompts, n_dims = X.shape

    if args.n_prompts_subsample is not None and args.n_prompts_subsample < n_prompts:
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(n_prompts, generator=g)[: args.n_prompts_subsample]
        X = X[idx]
        n_prompts = X.shape[0]
        print(f"Subsampled to {n_prompts} prompts (seed={args.seed}).")

    singular_values = compute_singular_values(X)
    result = analyse(singular_values, threshold=args.threshold, n_prompts=n_prompts, n_dims=n_dims)
    plot(singular_values, result, out_png)

    with open(out_json, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"n_prompts / n_dims = {result.n_prompts} / {result.n_dims}")
    print(f"n_components       = {result.n_components}")
    print(f"k_kneedle          = {result.k_kneedle}  (cum EV = {result.explained_variance_at_k_kneedle:.4f})")
    print(f"k_cumulative@{result.threshold:.2f} = {result.k_cumulative}")
    print(f"k_second_deriv     = {result.k_second_deriv}  (cum EV = {result.explained_variance_at_k_second_deriv:.4f})")
    print(f"figure -> {out_png}")
    print(f"json   -> {out_json}")


if __name__ == "__main__":
    main()
