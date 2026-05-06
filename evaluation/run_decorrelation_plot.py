"""Save a heatmap PNG of concept sub-vector cosine similarity."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.decorrelation import concept_subvector_cosine_stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output_png", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    out = concept_subvector_cosine_stats(
        args.checkpoint, device=args.device, include_full_matrix=True
    )
    mat = np.array(out["cosine_matrix"], dtype=np.float64)
    labels = [f"p{i}" for i in range(mat.shape[0])]

    fig, ax = plt.subplots(figsize=(max(6, mat.shape[0] * 0.25), max(5, mat.shape[0] * 0.25)))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Concept sub-vector cosine similarity")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=200)
    plt.close(fig)
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
