import torchvision

torchvision.disable_beta_transforms_warning()

import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def check_tensor_shape(tensor, tensor_name, correct_shape):
    if not list(tensor.shape) == correct_shape:
        raise Exception(f"Wrong {tensor_name} dimension {tensor.shape}")


class ImageGenerator:
    MODEL_ID = "stabilityai/stable-diffusion-3.5-large-turbo"
    NUM_INFERENCE_STEPS = 4
    GUIDANCE_SCALE = 0.0
    MAX_SEQUENCE_LENGTH = 512

    @classmethod
    def fingerprint(cls) -> dict:
        return {
            "model_id": cls.MODEL_ID,
            "num_inference_steps": cls.NUM_INFERENCE_STEPS,
            "guidance_scale": cls.GUIDANCE_SCALE,
            "max_sequence_length": cls.MAX_SEQUENCE_LENGTH,
        }

    def __init__(self, simulated: bool = True, device: str = "cpu") -> None:
        self.simulated = simulated
        self.device = device

        self.pipeline = None

        print("Loading pipeline...")

        if not self.simulated:
            self.load_pipeline()

        print("Pipeline loaded.")

    def load_pipeline(self):
        from diffusers import (
            BitsAndBytesConfig,
            SD3Transformer2DModel,
            StableDiffusion3Pipeline,
        )
        from transformers import T5EncoderModel

        model_id = self.MODEL_ID

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
        )

        t5_nf4 = T5EncoderModel.from_pretrained(
            "diffusers/t5-nf4", torch_dtype=torch.bfloat16
        )

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=model_nf4,
            text_encoder_3=t5_nf4,
            torch_dtype=torch.bfloat16,
        )

        # NF4-quantized modules (transformer, text_encoder_3) are pinned to
        # GPU by bitsandbytes at load time and refuse .to(). Both
        # enable_model_cpu_offload and pipeline.to() try to move them and
        # fail on current bnb + diffusers. Move only the non-quantized
        # components so every call site finds tensors on the same device.
        for attr in ("text_encoder", "text_encoder_2", "vae"):
            module = getattr(self.pipeline, attr, None)
            if module is not None:
                module.to(self.device)

    @torch.no_grad()
    def generate_image_from_prompt(
        self,
        prompt: str,
        image_name: Path | str,
        use_negative_prompts: bool = False,
        seed: int | None = None,
    ):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.get_embds_text_encoder(
            prompt=prompt, use_negative_prompts=use_negative_prompts, seed=seed
        )

        self.generate_image_from_embd(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            image_name=image_name,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            seed=seed,
        )

    @torch.no_grad()
    def generate_image_from_embd(
        self,
        prompt_embeds: torch.tensor,
        pooled_prompt_embeds: torch.tensor,
        image_name: Path | str,
        negative_prompt_embeds: torch.tensor = None,
        negative_pooled_prompt_embeds: torch.tensor = None,
        seed: int | None = None,
    ) -> None:
        setup_seed(0 if seed is None else seed)

        if self.simulated:
            return self._generate_image_from_embd_simulated(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                image_name=image_name,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            )
        else:
            return self._generate_image_from_embd(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                image_name=image_name,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            )

    def _generate_image_from_embd(
        self,
        prompt_embeds: torch.tensor,
        pooled_prompt_embeds: torch.tensor,
        image_name: Path | str,
        negative_prompt_embeds: torch.tensor = None,
        negative_pooled_prompt_embeds: torch.tensor = None,
    ) -> None:
        image = self.pipeline(
            prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=self.NUM_INFERENCE_STEPS,
            guidance_scale=self.GUIDANCE_SCALE,
            max_sequence_length=self.MAX_SEQUENCE_LENGTH,
        ).images[0]
        image.save(image_name)
        print(f"Image saved in {image_name}")

    def _generate_image_from_embd_simulated(
        self,
        prompt_embeds: torch.tensor,
        pooled_prompt_embeds: torch.tensor,
        image_name: Path | str,
        negative_prompt_embeds: torch.tensor = None,
        negative_pooled_prompt_embeds: torch.tensor = None,
    ) -> None:
        check_tensor_shape(
            tensor=prompt_embeds,
            tensor_name="prompt_embeds",
            correct_shape=[1, 333, 4096],
        )
        check_tensor_shape(
            tensor=pooled_prompt_embeds,
            tensor_name="pooled_prompt_embeds",
            correct_shape=[1, 2048],
        )

        print(f"Image saved in {image_name}")

    def get_embds_text_encoder(
        self,
        prompt: str,
        use_negative_prompts: bool = False,
        seed: int | None = None,
    ):
        if self.simulated:
            return self._get_embds_text_encoder_simulated(
                prompt, use_negative_prompts, seed=seed
            )
        else:
            return self._get_embds_text_encoder(
                prompt, use_negative_prompts, seed=seed
            )

    def _get_embds_text_encoder(
        self,
        prompt: str,
        use_negative_prompts: bool = False,
        seed: int | None = None,
    ) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        setup_seed(0 if seed is None else seed)
        embds = self.pipeline.encode_prompt(prompt, prompt, prompt)
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = embds

        if not use_negative_prompts:
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def _get_embds_text_encoder_simulated(
        self,
        prompt: str,
        use_negative_prompts: bool = False,
        seed: int | None = None,
    ) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        setup_seed(0 if seed is None else seed)

        prompt_embeds = torch.randn(1, 333, 4096, device=self.device)
        pooled_prompt_embeds = torch.randn(1, 2048, device=self.device)
        negative_prompt_embeds = (
            torch.randn(1, 333, 4096, device=self.device)
            if use_negative_prompts
            else None
        )
        negative_pooled_prompt_embeds = (
            torch.randn(1, 2048, device=self.device) if use_negative_prompts else None
        )
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
