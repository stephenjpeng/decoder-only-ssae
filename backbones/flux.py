"""FLUX (Black Forest Labs) text-encoder backbone.

FLUX uses a dual text stack: CLIP-L for a 768-d pooled embedding and T5-XXL
for a sequence embedding of shape `[max_sequence_length, 4096]`. Unlike
SD3.5 the sequence length is user-configurable (default 256 for schnell,
512 for dev), so this backbone is parametric on `max_length` — the value is
recorded in the manifest and re-hydrated at inference time.

`decode()` synthesizes the RoPE `text_ids` tensor FLUX expects at the
pipeline call site; it is deterministic given the sequence length, so we
don't store it per prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from backbones.base import Backbone, StreamSpec, StreamTensors
from backbones.registry import register_backbone


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# FLUX defaults: schnell is 4-step turbo (Apache-2.0, guidance_scale=0);
# dev is 50-step distilled (non-commercial license, guidance ~3.5). Keep
# schnell as the default so callers get a working, redistributable pipeline
# out of the box.
_SCHNELL = "black-forest-labs/FLUX.1-schnell"
_DEV = "black-forest-labs/FLUX.1-dev"


@register_backbone("flux")
class FluxBackbone(Backbone):
    """Generic FLUX backbone; defaults target FLUX.1-schnell.

    `model_id` and `max_length` are overridable via the extraction CLI's
    `--model-id` / `--max-length` flags (recorded in `manifest.json` so
    inference reconstructs the same shapes).
    """

    def __init__(
        self,
        model_id: str = _SCHNELL,
        max_length: int = 256,
        dtype: str = "bfloat16",
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__(device=device)
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"dtype must be one of {list(_DTYPE_MAP)}, got {dtype!r}")
        self.model_id = model_id
        self.max_length = int(max_length)
        self.dtype = dtype

        # T5-XXL hidden size is 4096; CLIP-L pooled is 768. These are fixed
        # for the FLUX family, so we don't expose them as constructor args.
        self.stream_specs = [
            StreamSpec(
                name="seq",
                shape=(self.max_length, 4096),
                dtype="float16",
                h5_file="embds.h5",
            ),
            StreamSpec(
                name="pooled",
                shape=(768,),
                dtype="float16",
                h5_file="embds_pooled.h5",
            ),
        ]

        self.pipeline = None

    def load(self) -> None:
        if self._loaded:
            return

        from diffusers import FluxPipeline

        torch_dtype = _DTYPE_MAP[self.dtype]
        if self.device.type != "cuda" and torch_dtype != torch.float32:
            torch_dtype = torch.float32

        self.pipeline = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
        )
        # Match SD3.5 backbone's memory strategy: don't move to a single
        # device (FLUX transformer is large); let diffusers handle placement.
        try:
            self.pipeline.enable_model_cpu_offload()
        except AttributeError:
            self.pipeline = self.pipeline.to(self.device)

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

        # FluxPipeline.encode_prompt returns (prompt_embeds, pooled_prompt_embeds, text_ids).
        # `prompt` -> CLIP-L (pooled), `prompt_2` -> T5-XXL (sequence). Passing
        # the same string to both matches the SD3.5 backbone's convention.
        prompt_embeds, pooled_prompt_embeds, _text_ids = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            max_sequence_length=self.max_length,
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
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        height: int = 1024,
        width: int = 1024,
        **_: Any,
    ) -> Path:
        if not self._loaded:
            self.load()

        # Reasonable defaults per FLUX variant; overridable per-call.
        is_schnell = "schnell" in self.model_id.lower()
        if num_inference_steps is None:
            num_inference_steps = 4 if is_schnell else 50
        if guidance_scale is None:
            guidance_scale = 0.0 if is_schnell else 3.5

        torch_dtype = _DTYPE_MAP[self.dtype]
        if self.device.type != "cuda" and torch_dtype != torch.float32:
            torch_dtype = torch.float32

        seq = streams["seq"]
        pooled = streams["pooled"]
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
        seq = seq.to(torch_dtype)
        pooled = pooled.to(torch_dtype)

        # FLUX expects a `text_ids` positional tensor alongside the T5
        # sequence embeddings. It is deterministic (all zeros) for text
        # tokens; the image side has its own `latent_image_ids` that the
        # pipeline computes internally.
        text_ids = torch.zeros(
            seq.shape[1], 3, device=seq.device, dtype=seq.dtype
        )

        generator = None
        if seed is not None:
            gen_device = seq.device if seq.device.type != "meta" else self.device
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        image = self.pipeline(
            prompt_embeds=seq,
            pooled_prompt_embeds=pooled,
            text_ids=text_ids,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=self.max_length,
            height=height,
            width=width,
            generator=generator,
        ).images[0]

        output_path = Path(output_path)
        image.save(output_path)
        return output_path


@register_backbone("flux_dev")
class FluxDevBackbone(FluxBackbone):
    """Preset targeting `black-forest-labs/FLUX.1-dev` (non-commercial license).

    Uses the FLUX.1-dev default context length of 512 and 50-step / cfg~3.5
    generation. `model_id` is still overridable in case you point at a
    downstream fine-tune with the same architecture.
    """

    def __init__(
        self,
        model_id: str = _DEV,
        max_length: int = 512,
        dtype: str = "bfloat16",
        device: str | torch.device = "cuda",
        **_: Any,
    ) -> None:
        super().__init__(
            model_id=model_id,
            max_length=max_length,
            dtype=dtype,
            device=device,
        )
