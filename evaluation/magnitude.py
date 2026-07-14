"""Concept-strength / magnitude-sensitivity edits.

Instead of the usual binary presence mask, scale a single concept's mask entry by an
arbitrary scalar (e.g. 2, 10, -10) and decode with the frozen checkpoint -- no retraining.

Reuses the per-property "block mean" feature approximation that ``evaluation.composition``
already uses for held-out compositional combinations (mean activated ``Y`` block per property
for ``model_trainable_inputs``; the property's own embedding row for ``model_avg_feature``).
That approximation is what makes this work uniformly on train- or holdout-folder samples and
on either decoder variant: ``model_trainable_inputs`` only has trained ``Y`` rows for prompts
seen during training, so a held-out sample's own row doesn't exist and the block mean is the
sample-independent stand-in for "this concept's sub-vector".
"""

from __future__ import annotations

import torch

from evaluation.composition import (
    decode_latent_row_trainable,
    property_block_means_trainable_inputs,
)


def scaled_mask_row(
    mask_reduced_row: torch.Tensor, target_pid: int, magnitude: float
) -> torch.Tensor:
    """Clone a binary ``mask_reduced_row`` and overwrite ``target_pid`` with ``magnitude``."""
    row = mask_reduced_row.float().clone()
    row[target_pid] = float(magnitude)
    return row


@torch.no_grad()
def _latent_row_scaled_trainable(
    mask_row_scaled: torch.Tensor,
    block_means: torch.Tensor,
    n_repeat: int,
    device: torch.device,
) -> torch.Tensor:
    """Like ``composition.latent_row_from_property_means_trainable``, but scales each
    property's block by its (possibly non-binary, possibly negative) mask value instead
    of gating at 0.5.
    """
    n_props = mask_row_scaled.shape[0]
    z = torch.zeros(n_props * n_repeat, device=device, dtype=block_means.dtype)
    m = mask_row_scaled.to(device).float()
    for p in range(n_props):
        if m[p] != 0:
            sl = slice(p * n_repeat, (p + 1) * n_repeat)
            z[sl] = m[p] * block_means[p]
    return z.unsqueeze(0)


@torch.no_grad()
def _decode_scaled_avg_feature(
    decoder: torch.nn.Module, mask_row_scaled: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Like ``composition.decode_mask_row_avg_feature``, but scales the target property's
    embedding row by its mask value instead of a binary index lookup.

    The model's own index-based masking (``idx = arange(1, n+1) * mask``) only works for
    0/1 values -- a magnitude like 10 would look up an unrelated property at ``pid * 10``.
    So active properties are looked up at their normal integer index first, then the
    embedding is scaled explicitly before the activation.
    """
    n_props = mask_row_scaled.shape[0]
    m = mask_row_scaled.to(device).float()
    active = (m != 0).nonzero(as_tuple=True)[0]
    idx = torch.zeros(1, n_props, device=device, dtype=torch.long)
    idx[0, active] = active + 1
    emb = decoder.Y(idx)
    emb = emb * m.view(1, n_props, 1)
    emb = decoder.activation(emb)
    emb = emb.reshape(1, -1)
    return decoder.linear(emb)


def predict_embedding_magnitude(
    decoder: torch.nn.Module,
    model_name: str,
    mask_row_scaled: torch.Tensor,
    *,
    n_repeat: int,
    device: torch.device,
    mask_reduced_train: torch.Tensor | None = None,
    block_means: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a normalized-space prediction ``(1, dim_output)`` for a mask row where one
    concept's entry has been replaced by an arbitrary scalar (see ``scaled_mask_row``).
    """
    if model_name == "model_avg_feature":
        return _decode_scaled_avg_feature(decoder, mask_row_scaled, device)

    if model_name != "model_trainable_inputs":
        raise ValueError(f"Unsupported model_name: {model_name}")

    if block_means is None:
        if mask_reduced_train is None:
            raise ValueError("trainable_inputs needs mask_reduced_train or block_means")
        block_means = property_block_means_trainable_inputs(
            decoder, mask_reduced_train, n_repeat, device
        )
    z = _latent_row_scaled_trainable(mask_row_scaled, block_means, n_repeat, device)
    return decode_latent_row_trainable(decoder, z)
