"""Elbow analysis for `truncate_embds_topk` selection.

The training pipeline ranks embedding dimensions by `max(X) - min(X)` across
prompts and keeps the top k (default k=1000, hard-coded in
`params_default.yaml`). This script computes the same ranking, then applies
three complementary methods to locate the elbow of the sorted-range curve so
the truncation threshold can be chosen empirically rather than by fiat.

Methods:
  - kneedle:     perpendicular distance from the chord [(0, diff[0]), (N-1, diff[N-1])]
                 in a unit-normalised curve; the elbow is argmax distance.
  - cumulative:  smallest k such that cumsum(diff[:k]) / sum(diff) >= threshold.
  - second_deriv: argmax of a smoothed discrete second derivative of diff.

Run from the repo root:
    python -m analysis.elbow_topk --folder_path results/mps_run/ --threshold 0.95
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from trainings.dataloader.dataloader import H5Dataset


@dataclass
class ElbowResult:
    k_kneedle: int
    k_cumulative: int
    k_second_deriv: int
    n_dims: int
    threshold: float
    total_range: float
    range_at_k_kneedle: float
    range_at_k_cumulative: float
    range_at_k_second_deriv: float
    cumulative_fraction_at_k_kneedle: float
    cumulative_fraction_at_k_second_deriv: float


def compute_sorted_diff(dataset: H5Dataset) -> np.ndarray:
    """Reproduce the range-per-dim ranking used by `get_indices_truncate_embds_topk`."""
    X = dataset.get_X()
    diff = torch.max(X, dim=0)[0] - torch.min(X, dim=0)[0]
    return diff.sort(descending=True).values.cpu().numpy()


def detect_kneedle(y: np.ndarray) -> int:
    """Return k in [1, N] with maximum perpendicular distance to the endpoint chord.

    The input curve is unit-normalised on both axes; k is 1-indexed so it can be
    used directly as a "keep top-k" cutoff.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n == 0:
        return 0
    if n == 1:
        return 1
    x = np.arange(n, dtype=np.float64) / (n - 1)
    y_range = y.max() - y.min()
    if y_range == 0:
        return n
    y_norm = (y - y.min()) / y_range
    x0, y0 = 0.0, float(y_norm[0])
    x1, y1 = 1.0, float(y_norm[-1])
    dx, dy = x1 - x0, y1 - y0
    norm = np.hypot(dx, dy)
    # abs so orientation of the chord (ascending vs descending) doesn't matter.
    distances = np.abs(dy * x - dx * y_norm + (dx * y0 - dy * x0)) / norm
    return int(np.argmax(distances)) + 1


def detect_cumulative(y: np.ndarray, threshold: float) -> int:
    """Smallest k s.t. cumsum(y[:k]) / sum(y) >= threshold. Assumes y >= 0."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n == 0:
        return 0
    total = float(y.sum())
    if total == 0:
        return n
    cum_frac = np.cumsum(y) / total
    hits = cum_frac >= threshold
    if not hits.any():
        return n
    return int(np.argmax(hits)) + 1


def detect_second_deriv(y: np.ndarray, smoothing_sigma: float = 2.0) -> int:
    """Argmax of a smoothed discrete second derivative of y."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 3:
        return n
    smoothed = _gaussian_smooth(y, sigma=smoothing_sigma)
    d2 = smoothed[:-2] - 2 * smoothed[1:-1] + smoothed[2:]
    # +2 because d2[i] corresponds to smoothed[i+1] and we want a 1-indexed k.
    return int(np.argmax(d2)) + 2


# Back-compat aliases so downstream imports (elbow_pca) keep working.
kneedle_k = detect_kneedle
cumulative_k = detect_cumulative
second_deriv_k = detect_second_deriv


def _gaussian_smooth(y: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return y.astype(np.float64, copy=False)
    radius = int(np.ceil(3 * sigma))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(xs**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    # Reflect padding avoids edge artefacts at k=0 and k=N.
    padded = np.pad(y.astype(np.float64), radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def analyse(diff_sorted: np.ndarray, threshold: float) -> ElbowResult:
    n = len(diff_sorted)
    total = float(diff_sorted.sum())
    cum = np.cumsum(diff_sorted) / max(total, 1e-12)

    k_kneedle = detect_kneedle(diff_sorted)
    k_cum = detect_cumulative(diff_sorted, threshold)
    k_d2 = detect_second_deriv(diff_sorted)

    return ElbowResult(
        k_kneedle=k_kneedle,
        k_cumulative=k_cum,
        k_second_deriv=k_d2,
        n_dims=n,
        threshold=threshold,
        total_range=total,
        range_at_k_kneedle=float(diff_sorted[k_kneedle - 1]),
        range_at_k_cumulative=float(diff_sorted[k_cum - 1]),
        range_at_k_second_deriv=float(diff_sorted[k_d2 - 1]),
        cumulative_fraction_at_k_kneedle=float(cum[k_kneedle - 1]),
        cumulative_fraction_at_k_second_deriv=float(cum[k_d2 - 1]),
    )


def plot(
    diff_sorted: np.ndarray,
    result: ElbowResult,
    out_path: str,
) -> None:
    n = len(diff_sorted)
    ks = np.arange(1, n + 1)
    cum = np.cumsum(diff_sorted) / max(diff_sorted.sum(), 1e-12)

    fig, (ax_scree, ax_cum) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_scree.plot(ks, diff_sorted, color="black", linewidth=1.0)
    ax_scree.set_xscale("log")
    ax_scree.set_yscale("log")
    ax_scree.set_ylabel("range (max - min) per dim")
    ax_scree.set_title("Sorted per-dim range")
    _mark_k(ax_scree, result.k_kneedle, "kneedle", "tab:blue")
    _mark_k(ax_scree, result.k_cumulative, f"cum >= {result.threshold:.2f}", "tab:orange")
    _mark_k(ax_scree, result.k_second_deriv, "2nd deriv", "tab:green")
    ax_scree.legend(loc="upper right")

    ax_cum.plot(ks, cum, color="black", linewidth=1.0)
    ax_cum.set_xscale("log")
    ax_cum.axhline(result.threshold, color="tab:orange", linestyle=":", linewidth=1.0)
    ax_cum.set_ylabel("cumulative fraction of total range")
    ax_cum.set_xlabel("k (number of dims kept, log scale)")
    _mark_k(ax_cum, result.k_kneedle, "kneedle", "tab:blue")
    _mark_k(ax_cum, result.k_cumulative, f"cum >= {result.threshold:.2f}", "tab:orange")
    _mark_k(ax_cum, result.k_second_deriv, "2nd deriv", "tab:green")

    fig.suptitle(f"Elbow analysis (n_dims={result.n_dims})")
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
        help="Cumulative-range threshold for the `cumulative` method.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Where to save the elbow figure. Defaults to <folder_path>/elbow_analysis.png.",
    )
    parser.add_argument(
        "--json_out",
        default=None,
        help="Where to save the recommended-k JSON. Defaults to <folder_path>/elbow_analysis.json.",
    )
    parser.add_argument(
        "--truncate_n_prompts",
        type=int,
        default=None,
        help="Optionally cap the number of prompts scanned (debug only).",
    )
    args = parser.parse_args()

    folder_path = args.folder_path
    if not folder_path.endswith("/"):
        folder_path = folder_path + "/"

    out_png = args.out or os.path.join(folder_path, "elbow_analysis.png")
    out_json = args.json_out or os.path.join(folder_path, "elbow_analysis.json")

    dataset = H5Dataset(
        folder_path=folder_path,
        truncate_n_prompts=args.truncate_n_prompts,
        truncate_embds_topk=None,
        add_property_is_the_same=False,
        normalize=None,
    )

    diff_sorted = compute_sorted_diff(dataset)
    result = analyse(diff_sorted, threshold=args.threshold)
    plot(diff_sorted, result, out_png)

    with open(out_json, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"n_dims             = {result.n_dims}")
    print(f"k_kneedle          = {result.k_kneedle}  (cum frac = {result.cumulative_fraction_at_k_kneedle:.4f})")
    print(f"k_cumulative@{result.threshold:.2f} = {result.k_cumulative}")
    print(f"k_second_deriv     = {result.k_second_deriv}  (cum frac = {result.cumulative_fraction_at_k_second_deriv:.4f})")
    print(f"figure -> {out_png}")
    print(f"json   -> {out_json}")


if __name__ == "__main__":
    main()
