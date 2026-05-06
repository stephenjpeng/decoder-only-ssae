"""Locality / non-target preservation proxies using CLIP text space."""

from __future__ import annotations

from evaluation.clip_metrics import clip_text_text_similarity


def text_preserved_similarity(original_prompt: str, edited_prompt: str, **kwargs) -> float:
    """
    High cosine similarity means edited caption stays close to the original in CLIP text space
    (useful as a cheap non-target language prior; not a substitute for image probes).
    """
    return clip_text_text_similarity(original_prompt, edited_prompt, **kwargs)
