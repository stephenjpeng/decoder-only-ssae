"""Tests for rendering ladder controls: native, exact round-trip, and packed top-k."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call
import torch

from inference.image_generation.image_generator import ImageGenerator
from evaluation.method_labels import (
    CONDITIONING,
    CONDITIONING_DETAIL,
    METHOD_LABEL,
    METHOD_LABEL_SHORT,
    BASELINE_METHODS,
)


class TestRenderingLadderControls(unittest.TestCase):
    """Test suite for the rendering ladder: native, exact round-trip, packed top-k."""

    def test_native_prompt_in_baseline_methods(self):
        """native_prompt must be registered as a baseline method."""
        self.assertIn("native_prompt", BASELINE_METHODS)

    def test_prompt_only_in_baseline_methods(self):
        """prompt_only key is kept for cache compatibility."""
        self.assertIn("prompt_only", BASELINE_METHODS)

    def test_gt_embed_in_baseline_methods(self):
        """gt_embed remains a baseline method."""
        self.assertIn("gt_embed", BASELINE_METHODS)

    def test_native_prompt_conditioning_is_direct_text(self):
        """native_prompt conditioning must be direct_text."""
        self.assertEqual(CONDITIONING["native_prompt"], "direct_text")
        self.assertEqual(
            CONDITIONING_DETAIL["native_prompt"], "direct_text_pipeline_conditioning"
        )

    def test_prompt_only_conditioning_is_full_embedding(self):
        """prompt_only conditioning is full_embedding, not native/direct_text."""
        self.assertEqual(CONDITIONING["prompt_only"], "full_embedding")
        self.assertEqual(
            CONDITIONING_DETAIL["prompt_only"], "exact_full_embedding_round_trip"
        )

    def test_gt_embed_conditioning_is_packed(self):
        """gt_embed conditioning is packed top-k."""
        self.assertEqual(CONDITIONING["gt_embed"], "packed")
        self.assertIn("topk", CONDITIONING_DETAIL["gt_embed"])

    def test_native_prompt_label_clarity(self):
        """native_prompt label must not say round-trip or packed."""
        label = METHOD_LABEL["native_prompt"]
        self.assertIn("native", label.lower())
        self.assertNotIn("round-trip", label.lower())
        self.assertNotIn("packed", label.lower())

    def test_prompt_only_label_clarity(self):
        """prompt_only label must say round-trip or full embedding, not native."""
        label = METHOD_LABEL["prompt_only"]
        self.assertTrue(
            "round-trip" in label.lower() or "full" in label.lower(),
            f"prompt_only label '{label}' must mention round-trip or full embedding",
        )

    def test_gt_embed_label_mentions_packed_or_oracle(self):
        """gt_embed label must mention packed or oracle."""
        label = METHOD_LABEL["gt_embed"]
        self.assertTrue(
            "packed" in label.lower() or "oracle" in label.lower(),
            f"gt_embed label '{label}' must mention packed or oracle",
        )

    def test_short_labels_distinct(self):
        """Short labels for native_prompt, prompt_only, gt_embed must be distinct."""
        short_native = METHOD_LABEL_SHORT["native_prompt"]
        short_prompt_only = METHOD_LABEL_SHORT["prompt_only"]
        short_gt = METHOD_LABEL_SHORT["gt_embed"]

        self.assertNotEqual(short_native, short_prompt_only)
        self.assertNotEqual(short_native, short_gt)
        self.assertNotEqual(short_prompt_only, short_gt)

    def test_generate_image_from_prompt_calls_encode_prompt(self):
        """generate_image_from_prompt (prompt_only path) calls encode_prompt."""
        gen = ImageGenerator(simulated=True, device="cpu")

        with patch.object(gen, "get_embds_text_encoder") as mock_encode:
            mock_encode.return_value = (
                torch.randn(1, 333, 4096),
                None,
                torch.randn(1, 2048),
                None,
            )
            with patch.object(gen, "generate_image_from_embd") as mock_decode:
                gen.generate_image_from_prompt(
                    "test prompt", "out.png", use_negative_prompts=False, seed=42
                )

                # must call encode_prompt (via get_embds_text_encoder)
                mock_encode.assert_called_once_with(
                    prompt="test prompt", use_negative_prompts=False, seed=42
                )
                # must call generate_image_from_embd
                self.assertEqual(mock_decode.call_count, 1)

    def test_generate_image_from_prompt_native_no_encode_prompt(self):
        """generate_image_from_prompt_native does NOT call encode_prompt."""
        gen = ImageGenerator(simulated=True, device="cpu")

        with patch.object(gen, "get_embds_text_encoder") as mock_encode:
            with patch.object(gen, "_generate_image_from_prompt_native_simulated") as mock_native:
                gen.generate_image_from_prompt_native(
                    "test prompt", "out.png", use_negative_prompts=False, seed=42
                )

                # must NOT call encode_prompt
                mock_encode.assert_not_called()
                # must call native simulated
                mock_native.assert_called_once()

    def test_native_prompt_real_pipeline_receives_text_not_embeddings(self):
        """Native path sends text to pipeline.__call__, not precomputed embeddings."""
        with patch.object(ImageGenerator, "load_pipeline"):
            gen = ImageGenerator(simulated=False, device="cpu")
            gen.pipeline = MagicMock()
            gen.pipeline.return_value = Mock(images=[Mock(save=Mock())])
            # must not call encode_prompt when sending text to pipeline
            gen.pipeline.encode_prompt = MagicMock()

            gen._generate_image_from_prompt_native(
                "test prompt", "out.png", use_negative_prompts=False
            )

            # pipeline must be called with prompt text, not prompt_embeds
            call_kwargs = gen.pipeline.call_args[1]
            self.assertEqual(call_kwargs["prompt"], "test prompt")
            self.assertEqual(call_kwargs["prompt_2"], "test prompt")
            self.assertEqual(call_kwargs["prompt_3"], "test prompt")
            self.assertNotIn("prompt_embeds", call_kwargs)
            self.assertNotIn("pooled_prompt_embeds", call_kwargs)
            # encode_prompt must NOT be called by native path
            gen.pipeline.encode_prompt.assert_not_called()

    def test_native_prompt_real_pipeline_negative_prompt_handling(self):
        """Native path passes negative_prompt=None when not requested."""
        with patch.object(ImageGenerator, "load_pipeline"):
            gen = ImageGenerator(simulated=False, device="cpu")
            gen.pipeline = MagicMock()
            gen.pipeline.return_value = Mock(images=[Mock(save=Mock())])

            gen._generate_image_from_prompt_native(
                "test prompt", "out.png", use_negative_prompts=False
            )

            call_kwargs = gen.pipeline.call_args[1]
            self.assertIsNone(call_kwargs["negative_prompt"])
            self.assertIsNone(call_kwargs["negative_prompt_2"])
            self.assertIsNone(call_kwargs["negative_prompt_3"])

    def test_native_prompt_real_pipeline_negative_prompt_enabled(self):
        """Native path passes empty string negative_prompt when requested."""
        with patch.object(ImageGenerator, "load_pipeline"):
            gen = ImageGenerator(simulated=False, device="cpu")
            gen.pipeline = MagicMock()
            gen.pipeline.return_value = Mock(images=[Mock(save=Mock())])

            gen._generate_image_from_prompt_native(
                "test prompt", "out.png", use_negative_prompts=True
            )

            call_kwargs = gen.pipeline.call_args[1]
            self.assertEqual(call_kwargs["negative_prompt"], "")
            self.assertEqual(call_kwargs["negative_prompt_2"], "")
            self.assertEqual(call_kwargs["negative_prompt_3"], "")

    def test_round_trip_real_pipeline_receives_embeddings_not_text(self):
        """Round-trip path (generate_image_from_embd) receives embeddings, not text."""
        with patch.object(ImageGenerator, "load_pipeline"):
            gen = ImageGenerator(simulated=False, device="cpu")
            gen.pipeline = MagicMock()
            gen.pipeline.return_value = Mock(images=[Mock(save=Mock())])

            prompt_embeds = torch.randn(1, 333, 4096)
            pooled_prompt_embeds = torch.randn(1, 2048)

            gen._generate_image_from_embd(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                image_name="out.png",
            )

            # pipeline must be called with embeddings, prompt=None
            call_kwargs = gen.pipeline.call_args[1]
            self.assertIsNone(call_kwargs["prompt"])
            self.assertIsNotNone(call_kwargs["prompt_embeds"])
            self.assertIsNotNone(call_kwargs["pooled_prompt_embeds"])
            torch.testing.assert_close(call_kwargs["prompt_embeds"], prompt_embeds)
            torch.testing.assert_close(call_kwargs["pooled_prompt_embeds"], pooled_prompt_embeds)

    def test_native_and_round_trip_use_same_seed(self):
        """Native and round-trip paths honor the same seed."""
        gen = ImageGenerator(simulated=True, device="cpu")

        with patch("inference.image_generation.image_generator.setup_seed") as mock_seed:
            gen.generate_image_from_prompt_native(
                "test", "out1.png", use_negative_prompts=False, seed=123
            )
            # setup_seed called once
            self.assertEqual(mock_seed.call_count, 1)
            mock_seed.assert_called_with(123)

        with patch("inference.image_generation.image_generator.setup_seed") as mock_seed:
            gen.generate_image_from_prompt(
                "test", "out2.png", use_negative_prompts=False, seed=123
            )
            # setup_seed called once in get_embds_text_encoder, once in generate_image_from_embd
            self.assertEqual(mock_seed.call_count, 2)
            mock_seed.assert_has_calls([call(123), call(123)])

    def test_native_simulated_rejects_non_string_prompt(self):
        """Simulated native path validates prompt is str."""
        gen = ImageGenerator(simulated=True, device="cpu")

        with self.assertRaises(TypeError):
            gen._generate_image_from_prompt_native_simulated(
                prompt=123,  # wrong type
                image_name="out.png",
                use_negative_prompts=False,
            )

    def test_generate_image_from_prompt_docstring_mentions_round_trip(self):
        """generate_image_from_prompt docstring clarifies it is round-trip."""
        doc = ImageGenerator.generate_image_from_prompt.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(
            "round-trip" in doc.lower() or "encode" in doc.lower(),
            "generate_image_from_prompt docstring must mention round-trip or encode",
        )

    def test_generate_image_from_prompt_native_docstring_mentions_native(self):
        """generate_image_from_prompt_native docstring clarifies it is true native."""
        doc = ImageGenerator.generate_image_from_prompt_native.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(
            "native" in doc.lower() or "directly" in doc.lower(),
            "generate_image_from_prompt_native docstring must mention native or directly",
        )
        self.assertTrue(
            "without" in doc.lower() and "encode" in doc.lower(),
            "generate_image_from_prompt_native docstring must say 'without encode'",
        )

    def test_benchmark_can_distinguish_three_paths(self):
        """Benchmark metadata can distinguish native, round-trip, and packed."""
        # three distinct conditioning values
        self.assertNotEqual(CONDITIONING["native_prompt"], CONDITIONING["prompt_only"])
        self.assertNotEqual(CONDITIONING["native_prompt"], CONDITIONING["gt_embed"])
        self.assertNotEqual(CONDITIONING["prompt_only"], CONDITIONING["gt_embed"])

        # three distinct detail strings
        self.assertNotEqual(
            CONDITIONING_DETAIL["native_prompt"], CONDITIONING_DETAIL["prompt_only"]
        )
        self.assertNotEqual(
            CONDITIONING_DETAIL["native_prompt"], CONDITIONING_DETAIL["gt_embed"]
        )
        self.assertNotEqual(
            CONDITIONING_DETAIL["prompt_only"], CONDITIONING_DETAIL["gt_embed"]
        )


if __name__ == "__main__":
    unittest.main()
