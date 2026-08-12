"""
Build composed latent rows from trained decoders (no change to model classes).

Trainable-inputs path: per-property block means of ``Y * mask`` on the training set.
Avg-feature path: one embedding index per property; inactive slots use padding index 0.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def _mask_full_from_reduced(mask_reduced: torch.Tensor, n_repeat: int) -> torch.Tensor:
    return mask_reduced.float().repeat_interleave(n_repeat, dim=1)


@torch.no_grad()
def property_block_means_trainable_inputs(
    decoder: torch.nn.Module,
    mask_reduced_train: torch.Tensor,
    n_repeat: int,
    device: torch.device,
) -> torch.Tensor:
    """
    ``mask_reduced_train``: (n_train, n_properties) int/bool on device or CPU.
    Returns tensor of shape ``(n_properties, n_repeat)`` with mean activated block per property.
    """
    mask_reduced_train = mask_reduced_train.to(device).float()
    Y = decoder.Y
    mf = _mask_full_from_reduced(mask_reduced_train, n_repeat)
    Ym = Y * mf
    n_props = mask_reduced_train.shape[1]
    n_repeat_i = n_repeat
    means = []
    for p in range(n_props):
        sl = slice(p * n_repeat_i, (p + 1) * n_repeat_i)
        sel = mask_reduced_train[:, p] > 0.5
        if sel.any():
            means.append(torch.relu(Ym[sel][:, sl]).mean(dim=0))
        else:
            means.append(torch.zeros(n_repeat_i, device=device, dtype=Y.dtype))
    return torch.stack(means, dim=0)


@torch.no_grad()
def latent_row_from_property_means_trainable(
    mask_reduced_row: torch.Tensor,
    block_means: torch.Tensor,
    n_repeat: int,
    device: torch.device,
) -> torch.Tensor:
    """``mask_reduced_row`` (n_properties,), ``block_means`` (n_properties, n_repeat)."""
    n_props = mask_reduced_row.shape[0]
    z = torch.zeros(n_props * n_repeat, device=device, dtype=block_means.dtype)
    m = mask_reduced_row.to(device).float()
    for p in range(n_props):
        if m[p] != 0:
            sl = slice(p * n_repeat, (p + 1) * n_repeat)
            z[sl] = block_means[p] * m[p]
    return z.unsqueeze(0)


@torch.no_grad()
def decode_latent_row_trainable(decoder: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    """``z`` shape (1, n_features); block means are already post-ReLU, so skip activation."""
    return decoder.linear(z)


@torch.no_grad()
def decode_mask_row_avg_feature(decoder: torch.nn.Module, mask_reduced_row: torch.Tensor) -> torch.Tensor:
    """Single prompt row; ``mask_reduced_row`` can contain binary or fractional weights"""
    device = mask_reduced_row.device
    n_props = mask_reduced_row.shape[0]
    idx = torch.arange(1, n_props + 1, device=device, dtype=torch.long)
    emb = decoder.activation(decoder.Y(idx))
    weights = mask_reduced_row.to(device).float().view(n_props, 1)
    emb = (emb * weights).reshape(1, -1)
    return decoder.linear(emb)


def predict_embedding_compositional(
    decoder: torch.nn.Module,
    model_name: str,
    mask_reduced_row: torch.Tensor,
    *,
    mask_reduced_train: torch.Tensor | None = None,
    n_repeat: int,
    device: torch.device,
    block_means: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Return normalized-space prediction (1, dim_output).

    For ``model_trainable_inputs``, pass ``mask_reduced_train`` and optional precomputed
    ``block_means`` (else computed from decoder + train mask).
    """
    if model_name == "model_avg_feature":
        return decode_mask_row_avg_feature(decoder, mask_reduced_row.to(device))

    if model_name != "model_trainable_inputs":
        raise ValueError(f"Unsupported model_name: {model_name}")

    if block_means is None:
        if mask_reduced_train is None:
            raise ValueError("trainable_inputs needs mask_reduced_train or block_means")
        block_means = property_block_means_trainable_inputs(
            decoder, mask_reduced_train, n_repeat, device
        )
    z = latent_row_from_property_means_trainable(
        mask_reduced_row, block_means, n_repeat, device
    )
    return decode_latent_row_trainable(decoder, z)
