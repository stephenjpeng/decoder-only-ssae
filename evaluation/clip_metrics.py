"""Optional CLIP image–text and text–text similarity (``transformers`` CLIP)."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def load_clip_model(
    model_name: str = "openai/clip-vit-base-patch32",
    device: str | None = None,
):
    from transformers import CLIPModel, CLIPProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    return model, proc, device


@torch.no_grad()
def clip_image_text_similarity(
    image_path: Path | str,
    text: str,
    *,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str | None = None,
) -> float:
    """Cosine similarity between CLIP image and text embeddings."""
    model, proc, dev = load_clip_model(model_name, device=device)
    image = Image.open(image_path).convert("RGB")
    inputs = proc(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    out = model(**inputs)
    im = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    tx = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((im * tx).sum(dim=-1).squeeze().item())


@torch.no_grad()
def clip_text_text_similarity(
    text_a: str,
    text_b: str,
    *,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str | None = None,
) -> float:
    """Cosine similarity between CLIP text embeddings of two strings."""
    model, proc, dev = load_clip_model(model_name, device=device)
    inputs = proc(text=[text_a, text_b], return_tensors="pt", padding=True)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    e = model.get_text_features(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    )
    e = e / e.norm(dim=-1, keepdim=True)
    return float((e[0] * e[1]).sum().item())
