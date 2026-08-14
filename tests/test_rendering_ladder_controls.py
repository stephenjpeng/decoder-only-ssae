"""Tests for rendering ladder controls: native, exact round-trip, and packed top-k."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import torch

from evaluation.baseline_cache import BaselineCache, compute_dataset_id
from evaluation.method_labels import (
    CONDITIONING,
    CONDITIONING_DETAIL,
    METHOD_LABEL,
    METHOD_LABEL_SHORT,
    BASELINE_METHODS,
    conditioning_map,
)
from evaluation.run_image_benchmark import (
    _generate_prompt_conditioned_image,
    _populated_cache_methods,
    _shared_baseline_cache_enabled,
)
from inference.image_generation.image_generator import ImageGenerator


class TestRenderingLadderControls(unittest.TestCase):
    """Test suite for the rendering ladder: native, exact round-trip, packed top-k."""

    def test_native_prompt_in_baseline_methods(self):
        """native_prompt must be registered as a baseline method."""
        self.assertIn("native_prompt", BASELINE_METHODS)

    def test_prompt_only_in_baseline_methods(self):
        """prompt_only key is kept for method-name compatibility."""
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

    def test_runtime_conditioning_records_fill_and_truncation(self):
        """Packed manifest detail reflects the active fill and top-k settings."""
        metadata = conditioning_map(
            ("native_prompt", "prompt_only", "prompt_modified_packed", "gt_embed"),
            fill_policy="source_prompt",
            truncate_embds_topk=512,
            t5_max_sequence_length=256,
        )

        self.assertEqual(metadata["native_prompt"]["t5_max_sequence_length"], 256)
        self.assertIsNone(metadata["native_prompt"]["fill_policy"])
        self.assertEqual(metadata["prompt_only"]["t5_max_sequence_length"], 256)
        self.assertEqual(
            metadata["prompt_modified_packed"]["detail"],
            "topk_512_reencoded_prompt_source_prompt_fill",
        )
        self.assertEqual(
            metadata["prompt_modified_packed"]["truncate_embds_topk"], 512
        )
        self.assertEqual(metadata["gt_embed"]["fill_policy"], "source_prompt")

    def test_untruncated_conditioning_records_no_fill(self):
        """Full embeddings do not claim a fill policy when no truncation is active."""
        metadata = conditioning_map(
            ("gt_embed", "prompt_modified_packed"),
            fill_policy="source_prompt",
            truncate_embds_topk=None,
            t5_max_sequence_length=256,
        )

        for method in metadata:
            self.assertIsNone(metadata[method]["fill_policy"])
            self.assertIsNone(metadata[method]["truncate_embds_topk"])
            self.assertIn("full_untruncated", metadata[method]["detail"])
            self.assertIn("no_fill", metadata[method]["detail"])

    def test_simulated_runs_bypass_shared_baseline_cache(self):
        """Placeholder renders cannot read or write the shared real-render cache."""
        self.assertFalse(
            _shared_baseline_cache_enabled(
                use_baseline_cache=True, simulated=True
            )
        )
        self.assertTrue(
            _shared_baseline_cache_enabled(
                use_baseline_cache=True, simulated=False
            )
        )

    def test_gt_embed_conditioning_is_packed(self):
        """gt_embed conditioning is packed top-k."""
        self.assertEqual(CONDITIONING["gt_embed"], "packed")
        metadata = conditioning_map(
            ("gt_embed",),
            fill_policy="train_mean",
            truncate_embds_topk=512,
            t5_max_sequence_length=256,
        )
        self.assertIn("topk_512", metadata["gt_embed"]["detail"])

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

    def test_benchmark_dispatch_routes_native_and_round_trip_paths(self):
        """Non-simulated benchmark dispatch keeps direct text separate from round-trip."""
        gen = MagicMock(spec=ImageGenerator)

        _generate_prompt_conditioned_image(
            gen, "native_prompt", "test prompt", Path("native.png"), 17
        )
        gen.generate_image_from_prompt_native.assert_called_once_with(
            "test prompt",
            Path("native.png"),
            use_negative_prompts=False,
            seed=17,
        )
        gen.generate_image_from_prompt.assert_not_called()

        gen.reset_mock()
        _generate_prompt_conditioned_image(
            gen, "prompt_only", "test prompt", Path("round-trip.png"), 17
        )
        gen.generate_image_from_prompt.assert_called_once_with(
            "test prompt",
            Path("round-trip.png"),
            use_negative_prompts=False,
            seed=17,
        )
        gen.generate_image_from_prompt_native.assert_not_called()

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
            self.assertEqual(call_kwargs["max_sequence_length"], 256)
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

    def test_round_trip_encoder_uses_native_t5_context_length(self):
        """Exact round-trip and direct text use the same 256-token T5 context."""
        with patch.object(ImageGenerator, "load_pipeline"):
            gen = ImageGenerator(simulated=False, device="cpu")
            gen.pipeline = MagicMock()
            gen.pipeline.encode_prompt.return_value = (
                torch.randn(1, 333, 4096),
                None,
                torch.randn(1, 2048),
                None,
            )

            gen._get_embds_text_encoder("test prompt")

        gen.pipeline.encode_prompt.assert_called_once_with(
            prompt="test prompt",
            prompt_2="test prompt",
            prompt_3="test prompt",
            max_sequence_length=256,
        )
        self.assertEqual(ImageGenerator.MAX_SEQUENCE_LENGTH, 256)
        self.assertEqual(ImageGenerator.fingerprint()["max_sequence_length"], 256)

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

    def test_cache_identity_hashes_ordered_topk_indices(self):
        """Equal top-k counts with different coordinate orders get different cache IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            holdout = Path(tmpdir) / "holdout"
            holdout.mkdir()
            (holdout / "prompts.json").write_text(
                json.dumps([{"prompt": "test"}]), encoding="utf-8"
            )
            common = {
                "holdout_folder": holdout,
                "train_x": torch.zeros(1, 2),
                "train_mask": torch.ones(1, 1),
                "base_seed": 0,
                "ridge_lambda": 0.01,
                "sd3_fingerprint": ImageGenerator.fingerprint(),
                "packer_fingerprint": {"truncate_embds_topk": 3},
            }

            first_id, first_key = compute_dataset_id(
                **common, truncate_embds_topk_indices=[7, 2, 5]
            )
            reordered_id, reordered_key = compute_dataset_id(
                **common, truncate_embds_topk_indices=[2, 7, 5]
            )

            self.assertNotEqual(first_id, reordered_id)
            self.assertEqual(
                first_key["truncate_embds_topk_indices_fingerprint"]["count"], 3
            )
            self.assertNotEqual(
                first_key["truncate_embds_topk_indices_fingerprint"]["sha256"],
                reordered_key["truncate_embds_topk_indices_fingerprint"]["sha256"],
            )

    def test_cache_identity_and_manifest_record_render_contract(self):
        """Cache identity and manifest expose the current render semantics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            holdout = root / "holdout"
            holdout.mkdir()
            (holdout / "prompts.json").write_text(
                json.dumps([{"prompt": "test"}]), encoding="utf-8"
            )
            common = {
                "holdout_folder": holdout,
                "train_x": torch.zeros(1, 2),
                "train_mask": torch.ones(1, 1),
                "base_seed": 0,
                "ridge_lambda": 0.01,
                "sd3_fingerprint": ImageGenerator.fingerprint(),
                "truncate_embds_topk_indices": list(range(512)),
            }
            current_packer = {
                "conditioning_contract_version": 2,
                "fill_policy": "train_mean",
                "truncate_embds_topk": 512,
            }
            old_packer = {**current_packer, "conditioning_contract_version": 1}
            current_id, key = compute_dataset_id(
                **common, packer_fingerprint=current_packer
            )
            old_id, _ = compute_dataset_id(
                **common, packer_fingerprint=old_packer
            )
            self.assertNotEqual(current_id, old_id)
            self.assertEqual(key["baseline_cache_identity_version"], 2)

            metadata = conditioning_map(
                ("native_prompt", "prompt_modified_packed"),
                fill_policy="train_mean",
                truncate_embds_topk=512,
                t5_max_sequence_length=256,
            )
            cache = BaselineCache(root / "cache", current_id, key)
            cache.write(
                methods=("native_prompt", "prompt_modified_packed"),
                extra_manifest={
                    "render_mode": "real",
                    "method_conditioning": metadata,
                    "truncate_embds_topk": 512,
                    "truncate_embds_topk_indices_fingerprint": key[
                        "truncate_embds_topk_indices_fingerprint"
                    ],
                },
            )
            manifest = json.loads((cache.dir / "manifest.json").read_text())
            self.assertEqual(manifest["render_mode"], "real")
            self.assertEqual(manifest["truncate_embds_topk"], 512)
            self.assertEqual(manifest["method_conditioning"], metadata)
            self.assertEqual(
                manifest["truncate_embds_topk_indices_fingerprint"],
                key["truncate_embds_topk_indices_fingerprint"],
            )
            self.assertEqual(
                manifest["key"]["truncate_embds_topk_indices_fingerprint"],
                key["truncate_embds_topk_indices_fingerprint"],
            )

    def test_subset_cache_write_preserves_conditioning_for_existing_rows(self):
        """A subset run keeps metadata for every method already stored in the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = BaselineCache(Path(tmpdir), "dataset", {"contract": 2})
            cache.upsert_row({"method": "native_prompt", "sample_idx": 0})
            cache.upsert_row({"method": "prompt_only", "sample_idx": 0})

            populated = _populated_cache_methods(cache)
            self.assertEqual(populated, ("native_prompt", "prompt_only"))
            metadata = conditioning_map(
                populated,
                fill_policy="train_mean",
                truncate_embds_topk=512,
                t5_max_sequence_length=256,
            )
            # mirror a later benchmark that requests only native_prompt
            cache.write(
                methods=("native_prompt",),
                extra_manifest={"method_conditioning": metadata},
            )

            manifest = json.loads((cache.dir / "manifest.json").read_text())
            self.assertEqual(
                set(manifest["methods"]), {"native_prompt", "prompt_only"}
            )
            self.assertEqual(
                set(manifest["method_conditioning"]),
                {"native_prompt", "prompt_only"},
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
