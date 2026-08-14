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
    The key is kept for method-name compatibility; the label clarifies the computation.
    All ~1.36M coordinates carry text signal.

``prompt_modified_packed`` — **packed top-k**
    The prompt is encoded, but only the ``truncate_embds_topk`` coordinates the SSAE
    predicts are kept. Other coordinates use the active runtime fill policy. An
    untruncated dataset keeps the complete prompt embedding and applies no fill. This puts
    prompt modification in the same subspace as the feature-editing methods.

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

#: Default method set for ``run_image_benchmark``. ``prompt_modified_packed`` is opt-in
#: because it adds another prompt-side render for each sample.
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
#: ``packed``           - active coordinates plus the runtime fill, or full untruncated
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

#: Stable conditioning operation before runtime fill and truncation are appended.
CONDITIONING_DETAIL: dict[str, str] = {
    "gt_embed": "true_embedding",
    "ssae_compose": "ssae_prediction",
    "mean_arithmetic": "mean_arithmetic_prediction",
    "ridge_embed": "ridge_prediction",
    "linear_probe_direction": "true_source_embedding_plus_calibrated_probe_direction",
    "native_prompt": "direct_text_pipeline_conditioning",
    "prompt_only": "exact_full_embedding_round_trip",
    "prompt_modified_packed": "reencoded_prompt",
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


def conditioning_map(
    methods,
    *,
    fill_policy: str,
    truncate_embds_topk: int | None,
    t5_max_sequence_length: int,
) -> dict[str, dict[str, str | int | None]]:
    """Build truthful runtime conditioning records for a benchmark manifest."""
    records: dict[str, dict[str, str | int | None]] = {}
    for method in methods:
        conditioning = CONDITIONING.get(method, "unknown")
        base_detail = CONDITIONING_DETAIL.get(method, "unknown")
        record: dict[str, str | int | None] = {
            "conditioning": conditioning,
            "detail": base_detail,
            "label": label(method),
            "fill_policy": None,
            "truncate_embds_topk": None,
            "t5_max_sequence_length": None,
        }

        if conditioning == "packed":
            if truncate_embds_topk is None:
                record["detail"] = f"full_untruncated_{base_detail}_no_fill"
            else:
                record.update(
                    {
                        "detail": (
                            f"topk_{truncate_embds_topk}_{base_detail}_"
                            f"{fill_policy}_fill"
                        ),
                        "fill_policy": fill_policy,
                        "truncate_embds_topk": truncate_embds_topk,
                    }
                )
        elif method in {"native_prompt", "prompt_only"}:
            record["detail"] = (
                f"{base_detail}_t5_max_{t5_max_sequence_length}"
            )
            record["t5_max_sequence_length"] = t5_max_sequence_length

        # this path is re-encoded before top-k packing, so both contracts apply
        if method == "prompt_modified_packed":
            record["t5_max_sequence_length"] = t5_max_sequence_length

        records[method] = record
    return records
