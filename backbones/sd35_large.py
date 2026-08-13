from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from backbones.base import Backbone, StreamSpec, StreamTensors
from backbones.registry import register_backbone


@register_backbone("sd35_large")
class Sd35LargeBackbone(Backbone):
    """Stable Diffusion 3.5 Large with memory-conscious non-Turbo defaults.

    The default generation uses 28 denoising steps, guidance scale 3.5, and a
    1024 by 1024 output. The 256-token T5 context preserves the SD3.5 stream
    contract: `seq` has shape [333, 4096] and `pooled` has shape [2048]. CUDA
    loading uses model CPU offload so the full pipeline does not occupy one GPU.
    """

    MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
    DEFAULT_NUM_INFERENCE_STEPS = 28
    DEFAULT_GUIDANCE_SCALE = 3.5
    DEFAULT_MAX_SEQUENCE_LENGTH = 256
    DEFAULT_HEIGHT = 1024
    DEFAULT_WIDTH = 1024

    stream_specs = [
        StreamSpec(name="seq", shape=(333, 4096), dtype="float16", h5_file="embds.h5"),
        StreamSpec(
            name="pooled",
            shape=(2048,),
            dtype="float16",
            h5_file="embds_pooled.h5",
        ),
    ]

    def __init__(
        self,
        device: str | torch.device = "cuda",
        model_id: str = MODEL_ID,
    ) -> None:
        super().__init__(device=device)
        self.model_id = model_id
        self.pipeline = None
        self._inference_dtype = (
            torch.bfloat16 if self.device.type == "cuda" else torch.float32
        )

    def load(self) -> None:
        """Load the pipeline with CPU offload when CUDA is the target device."""
        if self._loaded:
            return

        from diffusers import StableDiffusion3Pipeline

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.model_id,
            torch_dtype=self._inference_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        # VAE tiling bounds its peak activation memory for 1024 px generation
        try:
            self.pipeline.enable_vae_tiling()
        except AttributeError:
            pass

        if self.device.type == "cuda":
            # Diffusers moves one component at a time and returns it to CPU
            # after use, rather than placing the full SD3.5 stack on the GPU
            gpu_id = self.device.index if self.device.index is not None else 0
            self.pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            self.pipeline = self.pipeline.to(self.device)

        self._loaded = True

    def unload(self) -> None:
        """Release the pipeline and cached CUDA allocations."""
        self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().unload()

    @torch.no_grad()
    def encode(self, prompt: str) -> StreamTensors:
        """Encode one prompt into the fixed SD3.5 sequence and pooled streams."""
        if not self._loaded:
            self.load()

        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            max_sequence_length=self.DEFAULT_MAX_SEQUENCE_LENGTH,
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
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        **_: Any,
    ) -> Path:
        """Decode SD3.5 streams into an image, using a fixed seed when supplied."""
        if not self._loaded:
            self.load()

        if max_sequence_length != self.DEFAULT_MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "max_sequence_length must be 256 for the fixed [333, 4096] "
                "sequence stream"
            )

        seq = streams["seq"]
        pooled = streams["pooled"]
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)

        # Positive and classifier-free guidance embeddings must share a device
        # before Diffusers concatenates them
        execution_device = self.pipeline._execution_device
        seq = seq.to(device=execution_device, dtype=self._inference_dtype)
        pooled = pooled.to(device=execution_device, dtype=self._inference_dtype)

        # A CPU generator works with Diffusers CPU offload and keeps seed
        # behavior stable across CUDA device assignments
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        image = self.pipeline(
            prompt=None,
            prompt_embeds=seq,
            pooled_prompt_embeds=pooled,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=self.DEFAULT_MAX_SEQUENCE_LENGTH,
            height=height,
            width=width,
            generator=generator,
        ).images[0]

        output_path = Path(output_path)
        image.save(output_path)
        return output_path
