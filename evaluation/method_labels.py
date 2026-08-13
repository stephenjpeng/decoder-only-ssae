"""Single source of truth for benchmark method keys, labels and conditioning semantics.

AUG-01 background
-----------------
The proposal previously listed **Prompt modification** and **Prompt-only** as two separate
methods. They were the same code path: ``prompt_only`` calls
``ImageGenerator.generate_image_from_prompt()``, which encodes the text and immediately
calls ``generate_image_from_embd()``. One computation, two names.

The fix keeps the cache/method key ``prompt_only`` — existing caches under
``results/bench_baseline_cache/`` stay readable — but renames it everywhere a human reads
it to **Prompt modification (native/full embedding)**, and adds a genuinely distinct
method, ``prompt_modified_packed``, for the controlled-subspace comparison.

The two prompt methods differ in *conditioning*, which is the axis that matters:

``prompt_only`` — **native**
    The modified prompt is encoded and the resulting full 333x4096 (+2048 pooled) tensor is
    handed to the diffusion pipeline untouched. All ~1.36M coordinates carry text signal.
    This is what a deployed prompt-rewriting defense actually does, so it is the right row
    for the *practical* comparison against feature editing.

``prompt_modified_packed`` — **packed**
    The modified prompt is encoded, but only the ``truncate_embds_topk`` coordinates the
    SSAE predicts are kept; every other coordinate is overwritten with the training mean,
    exactly as an SSAE or ridge prediction is packed. This puts prompt modification in the
    same information-restricted subspace as the feature-editing methods, which is the right
    row for an *apples-to-apples* analysis.

Do not average the two. They answer different questions.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- keys

#: Methods whose renders do not depend on the SSAE checkpoint, and are therefore shared
#: through ``evaluation.baseline_cache``.
BASELINE_METHODS: tuple[str, ...] = (
    "gt_embed",
    "mean_arithmetic",
    "ridge_embed",
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
    "prompt_only",
)

# ------------------------------------------------------------------- conditioning

#: How each method's conditioning tensor reaches the diffusion pipeline. Recorded in every
#: run manifest so a result row can never be silently compared across conditioning paths.
#:
#: ``native``  - full text-encoder output, no coordinates replaced
#: ``packed``  - top-k coordinates only, remainder filled from the training mean
CONDITIONING: dict[str, str] = {
    "gt_embed": "packed",
    "ssae_compose": "packed",
    "mean_arithmetic": "packed",
    "ridge_embed": "packed",
    "linear_probe_direction": "packed",
    "prompt_only": "native",
    "prompt_modified_packed": "packed",
}

#: Longer machine-readable description of the conditioning path, for manifests.
CONDITIONING_DETAIL: dict[str, str] = {
    "gt_embed": "topk_true_embedding_train_mean_fill",
    "ssae_compose": "topk_ssae_prediction_train_mean_fill",
    "mean_arithmetic": "topk_mean_arithmetic_prediction_train_mean_fill",
    "ridge_embed": "topk_ridge_prediction_train_mean_fill",
    "linear_probe_direction": "topk_true_source_embedding_plus_calibrated_probe_direction_train_mean_fill",
    "prompt_only": "native_full_text_encoder_output",
    "prompt_modified_packed": "topk_reencoded_prompt_train_mean_fill",
}

# -------------------------------------------------------------------- presentation

#: Full labels — use in report headings, captions and the paper.
METHOD_LABEL: dict[str, str] = {
    "gt_embed": "GT embed (oracle)",
    "ssae_compose": "SSAE compose",
    "mean_arithmetic": "Mean-arithmetic",
    "ridge_embed": "Ridge",
    "linear_probe_direction": "Linear probe direction",
    "prompt_only": "Prompt modification (native/full embedding)",
    "prompt_modified_packed": "Prompt modification (packed top-k)",
}

#: Short labels for dense tables, axis ticks and legends where the full label will not fit.
METHOD_LABEL_SHORT: dict[str, str] = {
    "gt_embed": "GT embed",
    "ssae_compose": "SSAE compose",
    "mean_arithmetic": "Mean-arith",
    "ridge_embed": "Ridge",
    "linear_probe_direction": "Probe dir",
    "prompt_only": "Prompt mod (native)",
    "prompt_modified_packed": "Prompt mod (packed)",
}

METHOD_COLOR: dict[str, str] = {
    "gt_embed": "#374151",
    "ssae_compose": "#2563eb",
    "mean_arithmetic": "#0891b2",
    "ridge_embed": "#7c3aed",
    "linear_probe_direction": "#dc2626",
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
