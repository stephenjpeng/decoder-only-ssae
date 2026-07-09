from __future__ import annotations

import torch

from backbones.base import Backbone, StreamSpec, StreamTensors
from backbones.registry import register_backbone


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@register_backbone("sd35_turbo_text_only")
class Sd35TurboTextOnlyBackbone(Backbone):
    """SD3.5 Large Turbo's text encoders, without the transformer/VAE.

    `encode_prompt` never touches the transformer or VAE, so this loads only
    the three tokenizers and three text encoders (CLIP-L, CLIP-G, T5-XXL) in
    plain (non-NF4) precision. Unlike `sd35_large_turbo`, this has no
    bitsandbytes dependency and no CUDA requirement, so it runs on MPS/CPU as
    well as CUDA. Emits the same two streams (`seq`, `pooled`) as
    `sd35_large_turbo` and has no `decode()` path.
    """

    stream_specs = [
        StreamSpec(name="seq", shape=(333, 4096), dtype="float16", h5_file="embds.h5"),
        StreamSpec(name="pooled", shape=(2048,), dtype="float16", h5_file="embds_pooled.h5"),
    ]

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-3.5-large-turbo",
        dtype: str = "bfloat16",
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__(device=device)
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"dtype must be one of {list(_DTYPE_MAP)}, got {dtype!r}")
        self.model_id = model_id
        self.dtype = dtype
        self.pipeline = None

    def load(self) -> None:
        if self._loaded:
            return

        from diffusers import StableDiffusion3Pipeline
        from transformers import T5EncoderModel

        torch_dtype = _DTYPE_MAP[self.dtype]
        if self.device.type == "cpu" and torch_dtype != torch.float32:
            torch_dtype = torch.float32

        text_encoder_3 = T5EncoderModel.from_pretrained(
            self.model_id, subfolder="text_encoder_3", torch_dtype=torch_dtype
        )
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.model_id,
            transformer=None,
            vae=None,
            text_encoder_3=text_encoder_3,
            torch_dtype=torch_dtype,
        )
        self.pipeline.to(self.device)

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
        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt, prompt, prompt
        )
        return {
            "seq": prompt_embeds.squeeze(0),
            "pooled": pooled_prompt_embeds.squeeze(0),
        }
