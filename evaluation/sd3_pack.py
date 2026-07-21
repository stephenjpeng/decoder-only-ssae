"""Pack truncated normalized SSAE outputs into full SD3.5 text conditioning tensors.

Fill policy for non-top-k dimensions
------------------------------------
The SSAE only predicts the ``truncate_embds_topk`` highest-range coordinates of
the flattened SD3.5 text embedding (T5 flat concat pooled). The remaining
~1.36M coordinates have to come from somewhere when we hand a tensor to the
diffusion pipeline. Two policies are supported:

* ``oracle_fill`` (legacy): fill non-top-k dims with the *true* holdout
  embedding's values. Compact, but leaks ground truth into image-space
  evaluation — every method's rendered image gets most of its signal from the
  true target. Retained only for back-compat with older callers.
* ``train_mean_fill`` (recommended, used by ``run_image_benchmark``): fill non-
  top-k dims with the per-coordinate mean of the training set. No leakage from
  the specific holdout row; identical template across all methods and samples.

The fill policy is exposed via :func:`packer_fingerprint` so that the shared
baseline cache in ``evaluation/baseline_cache.py`` invalidates when the policy
changes (baseline images are rendered end-to-end and are policy-dependent).
"""

from __future__ import annotations

from pathlib import Path

import torch

from trainings.dataloader.dataloader import H5Dataset


PACKER_FILL_POLICY = "train_mean_fill"
PACKER_VERSION = 2


def packer_fingerprint() -> dict:
    return {"fill_policy": PACKER_FILL_POLICY, "packer_version": PACKER_VERSION}


def get_full_concat_embedding_untruncated(dataset: H5Dataset, tid: int) -> torch.Tensor:
    """Full concatenated [T5 flat | pooled] vector before top-k truncation."""
    indices_truncate = dataset.indices_truncate_embds_topk
    normalize = dataset.normalize
    dataset.indices_truncate_embds_topk = None
    dataset.normalize = None
    try:
        full_embd, _ = dataset.__getitem__(tid)
    finally:
        dataset.indices_truncate_embds_topk = indices_truncate
        dataset.normalize = normalize
    return full_embd


def compute_or_load_full_mean(
    dataset: H5Dataset, sidecar_name: str = "full_embd_mean.pt"
) -> torch.Tensor:
    """Per-coordinate mean of the untruncated flat embedding over all rows.

    Cached to ``<dataset.folder_path>/<sidecar_name>``. Meant to be called on the
    *training* dataset once per checkpoint; the resulting template is reused for
    every packing call in a run and shared across methods.
    """
    path = Path(dataset.folder_path) / sidecar_name
    if path.exists():
        return torch.load(path, map_location="cpu")

    saved_idx = dataset.indices_truncate_embds_topk
    saved_norm = dataset.normalize
    dataset.indices_truncate_embds_topk = None
    dataset.normalize = None
    try:
        n = len(dataset)
        if n == 0:
            raise ValueError(f"empty dataset at {dataset.folder_path}")
        x0, _ = dataset.__getitem__(0)
        acc = x0.to(torch.float64)
        for i in range(1, n):
            x, _ = dataset.__getitem__(i)
            acc.add_(x.to(torch.float64))
        mean = (acc / n).to(torch.float32)
    finally:
        dataset.indices_truncate_embds_topk = saved_idx
        dataset.normalize = saved_norm

    torch.save(mean, path)
    return mean


@torch.no_grad()
def pack_sd3_from_truncated_normalized(
    dataset: H5Dataset,
    tid: int,
    embd_topk_normalized: torch.Tensor,
    *,
    template: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Patch normalized top-k SSAE output into a full-dim embedding, then return
    ``(prompt_embeds, pooled_prompt_embeds)`` for the diffusion pipeline.

    ``embd_topk_normalized`` shape ``(1, dim_x)`` or ``(dim_x,)`` in the same
    normalized space as ``H5Dataset.__getitem__`` after truncation.

    ``template`` — full-dim (1.36M + 2048) fill for non-top-k coordinates. When
    provided (recommended: pass ``compute_or_load_full_mean(train_ds)``), the
    result is leakage-free with respect to the holdout row ``tid``. When
    ``None`` (legacy path), the true untruncated embedding for row ``tid`` is
    used — image-space metrics computed against this render leak ground truth
    through the ~99.93% of coordinates the SSAE didn't predict.
    """
    if template is None:
        full_embd = get_full_concat_embedding_untruncated(dataset, tid).clone()
    else:
        full_embd = template.clone()

    x = embd_topk_normalized.view(-1).to(full_embd.device)
    x_denorm = dataset.denormalize(x.unsqueeze(0)).view(-1)
    idx = dataset.indices_truncate_embds_topk
    if idx is None:
        full_embd = x_denorm.to(full_embd.dtype)
    else:
        full_embd[idx] = x_denorm.to(full_embd.dtype)

    embd = full_embd[:-2048].reshape(1, 333, 4096).to(torch.bfloat16)
    pooled = full_embd[-2048:].unsqueeze(0).to(torch.bfloat16)
    return embd, pooled
