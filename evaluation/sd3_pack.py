"""Pack truncated normalized SSAE outputs into full SD3.5 text conditioning tensors."""

from __future__ import annotations

import torch

from trainings.dataloader.dataloader import H5Dataset


def get_full_concat_embedding_untruncated(dataset: H5Dataset, tid: int) -> torch.Tensor:
    """Full concatenated [T5 flat | pooled] vector before top-k truncation (same as inference)."""
    indices_truncate = dataset.indices_truncate_embds_topk
    normalize = dataset.normalize
    dataset.indices_truncate_embds_topk = None
    dataset.normalize = None
    full_embd, _ = dataset.__getitem__(tid)
    dataset.indices_truncate_embds_topk = indices_truncate
    dataset.normalize = normalize
    return full_embd


@torch.no_grad()
def pack_sd3_from_truncated_normalized(
    dataset: H5Dataset,
    tid: int,
    embd_topk_normalized: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Patch normalized top-k SSAE output into the full embedding for row ``tid``,
    then return ``(prompt_embeds, pooled_prompt_embeds)`` for the diffusion pipeline.

    ``embd_topk_normalized`` shape ``(1, dim_x)`` or ``(dim_x,)`` in the same normalized
    space as ``H5Dataset.__getitem__`` after truncation.
    """
    full_embd = get_full_concat_embedding_untruncated(dataset, tid).clone()
    x = embd_topk_normalized.view(-1).to(full_embd.device)
    x_denorm = dataset.denormalize(x.unsqueeze(0)).view(-1)
    idx = dataset.indices_truncate_embds_topk
    if idx is None:
        raise ValueError("dataset.indices_truncate_embds_topk must be set for SD3 packing")
    full_embd[idx] = x_denorm.to(full_embd.dtype)

    embd = full_embd[:-2048].reshape(1, 333, 4096).to(torch.bfloat16)
    pooled = full_embd[-2048:].unsqueeze(0).to(torch.bfloat16)
    return embd, pooled
