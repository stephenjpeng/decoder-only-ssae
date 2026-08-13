"""Single source of truth for benchmark method keys, labels and conditioning semantics.

Rendering ladder: native, exact round-trip, and packed top-k
-------------------------------------------------------------
The SD3.5 image benchmark distinguishes three conditioning paths:

``native_prompt`` — **true native text generation**
    Prompt text is passed directly to ``StableDiffusion3Pipeline.__call__`` without calling
    ``encode_prompt()`` first. The pipeline's internal text encoders process the text as
    part of the diffusion forward pass. This is the deployment-realistic baseline.

``prompt_only`` — **exact full-embedding round-trip**
    The prompt is encoded via ``encode_prompt()`` and the full 333x4096 (+2048 pooled)
    embedding is immediately passed to ``generate_image_from_embd()``. Despite the key name,
    this is NOT true native generation — it round-trips through the full embedding space.
    The key is kept for existing cache compatibility; the label clarifies the computation.
    All ~1.36M coordinates carry text signal.

``prompt_modified_packed`` — **packed top-k**
    The prompt is encoded, but only the ``truncate_embds_topk`` coordinates the SSAE
    predicts are kept; every other coordinate is overwritten with the training mean,
    exactly as an SSAE or ridge prediction is packed. This puts prompt modification in
    the same information-restricted subspace as the feature-editing methods.

Source code, manifest labels, and tests must not conflate these three paths.
Do not average them. They answer different questions.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- keys

#: Methods whose renders do not depend on the SSAE checkpoint, and are therefore shared
#: through ``evaluation.baseline_cache``.
BASELINE_METHODS: tuple[str, ...] = (
    "gt_embed",
    "mean_arithmetic",
    "ridge_embed",
    "native_prompt",
    "prompt_only",
    "prompt_modified_packed",
)

#: Canonical display order for tables and figures.
METHOD_ORDER: tuple[str, ...] = (
    "gt_embed",
    "ssae_compose",
    "mean_arithmetic",
    "ridge_embed",
    "linear_probe_direction",
    "native_prompt",
    "prompt_only",
    "prompt_modified_packed",
)

#: Default method set for ``run_image_benchmark``. ``prompt_modified_packed`` is opt-in:
#: it doubles prompt-side rendering cost and the plan lists it as an open decision
#: ("whether it is required in the main table or only as a robustness check").
DEFAULT_METHODS: tuple[str, ...] = (
    "gt_embed",
    "ssae_compose",
    "mean_arithmetic",
    "ridge_embed",
    "native_prompt",
    "prompt_only",
)

# ------------------------------------------------------------------- conditioning

#: How each method's conditioning tensor reaches the diffusion pipeline. Recorded in every
#: run manifest so a result row can never be silently compared across conditioning paths.
#:
#: ``direct_text``      - prompt text sent directly to pipeline, no precomputed embeddings
#: ``full_embedding``   - full text-encoder output (all coordinates), no packing
#: ``packed``           - top-k coordinates only, remainder filled from training mean
CONDITIONING: dict[str, str] = {
    "gt_embed": "packed",
    "ssae_compose": "packed",
    "mean_arithmetic": "packed",
    "ridge_embed": "packed",
    "linear_probe_direction": "packed",
    "native_prompt": "direct_text",
    "prompt_only": "full_embedding",
    "prompt_modified_packed": "packed",
}

#: Longer machine-readable description of the conditioning path, for manifests.
CONDITIONING_DETAIL: dict[str, str] = {
    "gt_embed": "topk_true_embedding_train_mean_fill",
    "ssae_compose": "topk_ssae_prediction_train_mean_fill",
    "mean_arithmetic": "topk_mean_arithmetic_prediction_train_mean_fill",
    "ridge_embed": "topk_ridge_prediction_train_mean_fill",
    "linear_probe_direction": "topk_true_source_embedding_plus_calibrated_probe_direction_train_mean_fill",
    "native_prompt": "direct_text_pipeline_conditioning",
    "prompt_only": "exact_full_embedding_round_trip",
    "prompt_modified_packed": "topk_reencoded_prompt_train_mean_fill",
}

# -------------------------------------------------------------------- presentation

#: Full labels — use in report headings, captions and the paper.
METHOD_LABEL: dict[str, str] = {
    "gt_embed": "GT embed (packed top-k oracle)",
    "ssae_compose": "SSAE compose",
    "mean_arithmetic": "Mean-arithmetic",
    "ridge_embed": "Ridge",
    "linear_probe_direction": "Linear probe direction",
    "native_prompt": "Native text generation",
    "prompt_only": "Exact full-embedding round-trip",
    "prompt_modified_packed": "Prompt modification (packed top-k)",
}

#: Short labels for dense tables, axis ticks and legends where the full label will not fit.
METHOD_LABEL_SHORT: dict[str, str] = {
    "gt_embed": "GT embed",
    "ssae_compose": "SSAE compose",
    "mean_arithmetic": "Mean-arith",
    "ridge_embed": "Ridge",
    "linear_probe_direction": "Probe dir",
    "native_prompt": "Native text",
    "prompt_only": "Full embed round-trip",
    "prompt_modified_packed": "Prompt mod (packed)",
}

METHOD_COLOR: dict[str, str] = {
    "gt_embed": "#374151",
    "ssae_compose": "#2563eb",
    "mean_arithmetic": "#0891b2",
    "ridge_embed": "#7c3aed",
    "linear_probe_direction": "#dc2626",
    "native_prompt": "#16a34a",
    "prompt_only": "#b45309",
    "prompt_modified_packed": "#059669",
}


def label(method: str, *, short: bool = False) -> str:
    """Display label for ``method``; unknown keys fall back to the raw key."""
    table = METHOD_LABEL_SHORT if short else METHOD_LABEL
    return table.get(method, method)


def color(method: str) -> str:
    return METHOD_COLOR.get(method, "#6b7280")


def sort_methods(methods) -> tuple[str, ...]:
    """Canonical order first, then any unrecognised methods in their original order."""
    seen = list(dict.fromkeys(methods))
    ordered = tuple(m for m in METHOD_ORDER if m in seen)
    extra = tuple(m for m in seen if m not in ordered)
    return ordered + extra


def conditioning_map(methods) -> dict[str, dict[str, str]]:
    """Per-method conditioning record for a run manifest."""
    return {
        m: {
            "conditioning": CONDITIONING.get(m, "unknown"),
            "detail": CONDITIONING_DETAIL.get(m, "unknown"),
            "label": label(m),
        }
        for m in methods
    }
