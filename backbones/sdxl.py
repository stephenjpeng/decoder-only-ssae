from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from backbones.base import Backbone, StreamSpec, StreamTensors
from backbones.registry import register_backbone


@register_backbone("sdxl")
class SdxlBackbone(Backbone):
    """Stable Diffusion XL base 1.0 with its dual CLIP-L + CLIP-G text stack.

    Emits two streams matching the SDXL pipeline's `encode_prompt` outputs:
    the concatenated hidden states `seq` of shape [77, 2048] and the pooled
    text embedding `pooled` of shape [1280]. `decode()` reconstructs an image
    via the SDXL base pipeline.
    """

    stream_specs = [
        StreamSpec(name="seq", shape=(77, 2048), dtype="float16", h5_file="embds.h5"),
        StreamSpec(name="pooled", shape=(1280,), dtype="float16", h5_file="embds_pooled.h5"),
    ]

    def __init__(
        self,
        device: str | torch.device = "cuda",
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    ) -> None:
        super().__init__(device=device)
        self.model_id = model_id
        self.pipeline = None

    def load(self) -> None:
        if self._loaded:
            return

        from diffusers import StableDiffusionXLPipeline

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.pipeline = self.pipeline.to(self.device)
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
        except (ModuleNotFoundError, AttributeError):
            pass

        self._loaded = True

    def unload(self) -> None:
        self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().unload()

    @torch.no_grad()
    def encode(self, prompt: str) -> StreamTensors:
        if not self._loaded:
            self.load()
        (
            prompt_embeds,
            _negative_prompt_embeds,
            pooled_prompt_embeds,
            _negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(prompt=prompt, device=self.pipeline.device)
        return {
            "seq": prompt_embeds.squeeze(0),
            "pooled": pooled_prompt_embeds.squeeze(0),
        }

    @torch.no_grad()
    def decode(
        self,
        streams: StreamTensors,
        output_path: str | Path,
        seed: int | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        **_: Any,
    ) -> Path:
        if not self._loaded:
            self.load()

        seq = streams["seq"]
        pooled = streams["pooled"]
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
        seq = seq.to(torch.float16)
        pooled = pooled.to(torch.float16)

        # SDXL requires negative embeddings matching prompt_embeds' shape.
        # A zero tensor gives an unconditioned baseline consistent with
        # empty-string negative prompting.
        negative_seq = torch.zeros_like(seq)
        negative_pooled = torch.zeros_like(pooled)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.pipeline.device).manual_seed(seed)

        image = self.pipeline(
            prompt=None,
            prompt_embeds=seq,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=negative_seq,
            negative_pooled_prompt_embeds=negative_pooled,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        output_path = Path(output_path)
        image.save(output_path)
        return output_path
