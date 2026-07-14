from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from backbones.base import Backbone, StreamSpec, StreamTensors
from backbones.registry import register_backbone


@register_backbone("sd35_large_turbo")
class Sd35LargeTurboBackbone(Backbone):
    """Stable Diffusion 3.5 Large Turbo with NF4-quantized T5 text encoder.

    Emits two streams: the T5 sequence embedding `seq` of shape [333, 4096]
    and the pooled CLIP embedding `pooled` of shape [2048], both float16 on
    disk. `decode()` reconstructs an image via the SD3.5 pipeline.
    """

    stream_specs = [
        StreamSpec(name="seq", shape=(333, 4096), dtype="float16", h5_file="embds.h5"),
        StreamSpec(name="pooled", shape=(2048,), dtype="float16", h5_file="embds_pooled.h5"),
    ]

    def __init__(self, device: str | torch.device = "cuda") -> None:
        super().__init__(device=device)
        self.pipeline = None

    def load(self) -> None:
        if self._loaded:
            return

        from diffusers import (
            BitsAndBytesConfig,
            SD3Transformer2DModel,
            StableDiffusion3Pipeline,
        )
        from transformers import T5EncoderModel

        model_id = "stabilityai/stable-diffusion-3.5-large-turbo"

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        transformer = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
        )
        t5 = T5EncoderModel.from_pretrained(
            "diffusers/t5-nf4", torch_dtype=torch.bfloat16
        )

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder_3=t5,
            torch_dtype=torch.bfloat16,
        )
        # NF4-quantized submodules (transformer, text_encoder_3) are pinned
        # to GPU by bitsandbytes at load time and refuse .to(). Move only the
        # non-quantized components; encode_prompt then finds everything on
        # the same device.
        for attr in ("text_encoder", "text_encoder_2", "vae"):
            module = getattr(self.pipeline, attr, None)
            if module is not None:
                module.to(self.device)
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
        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt, prompt, prompt
        )
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
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        max_sequence_length: int = 512,
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
        seq = seq.to(torch.bfloat16)
        pooled = pooled.to(torch.bfloat16)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.pipeline.device).manual_seed(seed)

        image = self.pipeline(
            prompt=None,
            prompt_embeds=seq,
            pooled_prompt_embeds=pooled,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            generator=generator,
        ).images[0]

        output_path = Path(output_path)
        image.save(output_path)
        return output_path
