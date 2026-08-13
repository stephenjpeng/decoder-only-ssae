"""Tests for the non-Turbo SD3.5 Large backbone."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from backbones import get_backbone, list_backbones
from backbones.sd35_large import Sd35LargeBackbone


class TestSd35LargeBackbone(unittest.TestCase):
    def test_registry_and_stream_metadata(self) -> None:
        """The public registry exposes the fixed SD3.5 stream contract."""
        backbone = get_backbone("sd35_large", device="cpu")

        self.assertIsInstance(backbone, Sd35LargeBackbone)
        self.assertIn("sd35_large", list_backbones())
        self.assertEqual(
            [(spec.name, spec.shape) for spec in backbone.stream_specs],
            [("seq", (333, 4096)), ("pooled", (2048,))],
        )

    def test_model_id_and_non_turbo_defaults(self) -> None:
        """The preset uses Stability AI's non-Turbo model-card defaults."""
        backbone = Sd35LargeBackbone(device="cpu")

        self.assertEqual(
            backbone.model_id, "stabilityai/stable-diffusion-3.5-large"
        )
        self.assertEqual(backbone.DEFAULT_NUM_INFERENCE_STEPS, 28)
        self.assertEqual(backbone.DEFAULT_GUIDANCE_SCALE, 3.5)
        self.assertEqual(backbone.DEFAULT_MAX_SEQUENCE_LENGTH, 256)
        self.assertEqual(
            (backbone.DEFAULT_HEIGHT, backbone.DEFAULT_WIDTH), (1024, 1024)
        )

    def test_load_uses_low_memory_cpu_offload_for_cuda(self) -> None:
        """CUDA loading never moves the complete pipeline onto one GPU."""
        pipeline = MagicMock()
        pipeline_factory = MagicMock(return_value=pipeline)
        fake_diffusers = types.SimpleNamespace(
            StableDiffusion3Pipeline=types.SimpleNamespace(
                from_pretrained=pipeline_factory
            )
        )
        backbone = Sd35LargeBackbone(device="cuda:1")

        with patch.dict(sys.modules, {"diffusers": fake_diffusers}):
            backbone.load()

        pipeline_factory.assert_called_once_with(
            "stabilityai/stable-diffusion-3.5-large",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        pipeline.enable_vae_tiling.assert_called_once_with()
        pipeline.enable_model_cpu_offload.assert_called_once_with(gpu_id=1)
        pipeline.to.assert_not_called()

    def test_encode_passes_prompt_to_all_three_text_encoders(self) -> None:
        """Encoding fixes the T5 context length that yields 333 tokens."""
        pipeline = MagicMock()
        pipeline.encode_prompt.return_value = (
            torch.ones(1, 333, 4096),
            None,
            torch.ones(1, 2048),
            None,
        )
        backbone = Sd35LargeBackbone(device="cpu")
        backbone.pipeline = pipeline
        backbone._loaded = True

        streams = backbone.encode("a red cube")

        pipeline.encode_prompt.assert_called_once_with(
            prompt="a red cube",
            prompt_2="a red cube",
            prompt_3="a red cube",
            max_sequence_length=256,
        )
        self.assertEqual(streams["seq"].shape, (333, 4096))
        self.assertEqual(streams["pooled"].shape, (2048,))

    def test_decode_forwards_defaults_and_seeded_cpu_generator(self) -> None:
        """Decode forwards non-Turbo settings and constructs a repeatable seed."""
        image = MagicMock()
        pipeline = MagicMock(return_value=types.SimpleNamespace(images=[image]))
        pipeline._execution_device = torch.device("cpu")
        backbone = Sd35LargeBackbone(device="cpu")
        backbone.pipeline = pipeline
        backbone._loaded = True
        streams = {
            "seq": torch.zeros(333, 4096),
            "pooled": torch.zeros(2048),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "render.png"
            result = backbone.decode(streams, output_path, seed=31415)

        self.assertEqual(result, output_path)
        image.save.assert_called_once_with(output_path)
        kwargs = pipeline.call_args.kwargs
        self.assertIsNone(kwargs["prompt"])
        self.assertEqual(kwargs["prompt_embeds"].shape, (1, 333, 4096))
        self.assertEqual(kwargs["pooled_prompt_embeds"].shape, (1, 2048))
        self.assertEqual(kwargs["num_inference_steps"], 28)
        self.assertEqual(kwargs["guidance_scale"], 3.5)
        self.assertEqual(kwargs["max_sequence_length"], 256)
        self.assertEqual((kwargs["height"], kwargs["width"]), (1024, 1024))
        self.assertEqual(kwargs["generator"].device.type, "cpu")
        self.assertEqual(kwargs["generator"].initial_seed(), 31415)

    def test_decode_forwards_overrides_and_optional_seed(self) -> None:
        """Call-specific generation settings override the preset defaults."""
        image = MagicMock()
        pipeline = MagicMock(return_value=types.SimpleNamespace(images=[image]))
        pipeline._execution_device = torch.device("cpu")
        backbone = Sd35LargeBackbone(device="cpu")
        backbone.pipeline = pipeline
        backbone._loaded = True
        streams = {
            "seq": torch.zeros(1, 333, 4096),
            "pooled": torch.zeros(1, 2048),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            backbone.decode(
                streams,
                Path(tmpdir) / "render.png",
                num_inference_steps=35,
                guidance_scale=4.0,
                max_sequence_length=256,
                height=768,
                width=640,
            )

        kwargs = pipeline.call_args.kwargs
        self.assertEqual(kwargs["num_inference_steps"], 35)
        self.assertEqual(kwargs["guidance_scale"], 4.0)
        self.assertEqual((kwargs["height"], kwargs["width"]), (768, 640))
        self.assertIsNone(kwargs["generator"])

    def test_decode_places_streams_on_pipeline_execution_device(self) -> None:
        """Positive embeddings move to the device used for CFG embeddings."""
        image = MagicMock()
        pipeline = MagicMock(return_value=types.SimpleNamespace(images=[image]))
        pipeline._execution_device = torch.device("cuda:1")
        backbone = Sd35LargeBackbone(device="cuda:1")
        backbone.pipeline = pipeline
        backbone._loaded = True
        seq = MagicMock()
        pooled = MagicMock()
        seq.dim.return_value = 3
        pooled.dim.return_value = 2
        placed_seq = MagicMock()
        placed_pooled = MagicMock()
        seq.to.return_value = placed_seq
        pooled.to.return_value = placed_pooled

        with tempfile.TemporaryDirectory() as tmpdir:
            backbone.decode(
                {"seq": seq, "pooled": pooled},
                Path(tmpdir) / "render.png",
            )

        seq.to.assert_called_once_with(
            device=torch.device("cuda:1"), dtype=torch.bfloat16
        )
        pooled.to.assert_called_once_with(
            device=torch.device("cuda:1"), dtype=torch.bfloat16
        )
        self.assertIs(pipeline.call_args.kwargs["prompt_embeds"], placed_seq)
        self.assertIs(
            pipeline.call_args.kwargs["pooled_prompt_embeds"], placed_pooled
        )

    def test_decode_rejects_context_length_outside_stream_contract(self) -> None:
        """A different T5 context cannot match a stored 333-token stream."""
        pipeline = MagicMock()
        backbone = Sd35LargeBackbone(device="cpu")
        backbone.pipeline = pipeline
        backbone._loaded = True

        with self.assertRaisesRegex(ValueError, "must be 256"):
            backbone.decode(
                {"seq": torch.zeros(1), "pooled": torch.zeros(1)},
                "unused.png",
                max_sequence_length=512,
            )

        pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
