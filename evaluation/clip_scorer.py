"""Reusable CLIP scoring (single model load) for benchmarks."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


class CLIPScorer:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ):
        from transformers import CLIPModel, CLIPProcessor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model_name = model_name

    @torch.no_grad()
    def image_text_cosine(self, image_path: Path | str, text: str) -> float:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        im = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        tx = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((im * tx).sum(dim=-1).squeeze().item())

    @torch.no_grad()
    def image_encode(self, image_path: Path | str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        pv = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        feats = self.model.get_image_features(pixel_values=pv)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def text_encode(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def image_encode_paths_batch(
        self,
        paths: list[Path | str],
        *,
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Stacked L2-normalized CLIP image features ``(N, d)`` on CPU."""
        out_list = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            pv = self.processor(images=images, return_tensors="pt", padding=True).pixel_values.to(
                self.device
            )
            feats = self.model.get_image_features(pixel_values=pv)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out_list.append(feats.cpu())
        return torch.cat(out_list, dim=0)

    def image_attribute_alignment(
        self, image_path: Path | str, attribute_phrases: list[str]
    ) -> dict:
        """
        Cheap multi-attribute proxy: CLIP cosine between the image and each attribute phrase.
        Returns mean / min over active attributes (higher = better alignment with listed concepts).
        """
        if not attribute_phrases:
            return {"mean_cosine_attr": float("nan"), "min_cosine_attr": float("nan"), "n_attrs": 0}
        im = self.image_encode(image_path)
        tx = self.text_encode(attribute_phrases)
        sims = (im @ tx.T).squeeze(0)
        vals = sims.tolist()
        return {
            "mean_cosine_attr": float(sum(vals) / len(vals)),
            "min_cosine_attr": float(min(vals)),
            "n_attrs": len(vals),
            "per_attr_cosine": vals,
        }
