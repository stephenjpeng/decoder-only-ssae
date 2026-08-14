"""Tests for native-model qualification planning and execution."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backbones.base import StreamSpec
from backbones.flux import FluxDevBackbone
from evaluation.image_qualification import (
    Context,
    PromptSpec,
    Property,
    RenderRow,
    build_audit_plan,
    build_bakeoff_plan,
    validate_coverage,
)
from evaluation.run_native_model_qualification import (
    build_model_configuration,
    fingerprint_model_configuration,
    load_or_create_blinding_mapping,
    manifest_success_matches,
    write_scoring_sheet,
)

SPEC_PATH = "evaluation/config/image_validity_prompts.yaml"


class FakeQualificationBackbone:
    """Small observable backbone used to exercise resume without model weights."""

    model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
    revision = "test-revision"
    device = "cuda"
    dtype = "bfloat16"
    stream_specs = [
        StreamSpec("seq", (333, 4096), "float16", "embds.h5"),
        StreamSpec("pooled", (2048,), "float16", "embds_pooled.h5"),
    ]

    def __init__(self) -> None:
        self.load = MagicMock()
        self.unload = MagicMock()
        self.encode = MagicMock(return_value={"seq": "seq", "pooled": "pooled"})
        self.decode_calls: list[tuple[dict[str, str], Path, int | None]] = []

    def decode(
        self,
        streams: dict[str, str],
        output_path: Path,
        seed: int | None = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        max_sequence_length: int = 512,
    ) -> Path:
        """Record a successful render with the production default signature."""
        self.decode_calls.append((streams, output_path, seed))
        output_path.write_bytes(b"fake image")
        return output_path


class TestPropertyAndContext(unittest.TestCase):
    def test_property_from_dict(self) -> None:
        data = {
            "id": "hair_blond",
            "category": "hair",
            "original_phrase": "A blond girl",
            "rendering_phrase": "with blonde hair",
            "critical": False,
            "high_risk": True,
        }
        prop = Property.from_dict(data)
        self.assertEqual(prop.id, "hair_blond")
        self.assertEqual(prop.rendering_phrase, "with blonde hair")
        self.assertTrue(prop.high_risk)

    def test_context_render(self) -> None:
        context = Context("ctx", "subject", "A person {property}")
        self.assertEqual(context.render("with blue eyes"), "A person with blue eyes")


class TestPromptSpec(unittest.TestCase):
    def test_loads_frozen_inventory(self) -> None:
        spec = PromptSpec.load(SPEC_PATH)
        self.assertEqual(len(spec.properties), 26)
        self.assertEqual(len(spec.audit_design.context_ids), 2)
        self.assertEqual(len(spec.audit_design.seeds), 2)
        self.assertEqual(len(spec.bakeoff_design.property_ids), 12)
        self.assertEqual(len(spec.bakeoff_design.context_ids), 4)
        self.assertEqual(len(spec.bakeoff_design.seeds), 4)
        self.assertEqual(len(spec.bakeoff_design.model_backbones), 3)

    def test_fingerprint_is_stable_and_content_sensitive(self) -> None:
        first = PromptSpec.load(SPEC_PATH)
        second = PromptSpec.load(SPEC_PATH)
        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertEqual(len(first.fingerprint()), 12)

        second.properties[0].rendering_phrase = "changed phrase"
        self.assertNotEqual(first.fingerprint(), second.fingerprint())

    def test_rejects_duplicate_property_ids(self) -> None:
        spec = PromptSpec.load(SPEC_PATH)
        spec.properties[1].id = spec.properties[0].id
        with self.assertRaisesRegex(ValueError, "duplicate property IDs"):
            spec.validate()

    def test_rejects_duplicate_context_ids(self) -> None:
        spec = PromptSpec.load(SPEC_PATH)
        spec.contexts[1].id = spec.contexts[0].id
        with self.assertRaisesRegex(ValueError, "duplicate context IDs"):
            spec.validate()

    def test_rejects_unknown_audit_context(self) -> None:
        spec = PromptSpec.load(SPEC_PATH)
        spec.audit_design.context_ids[0] = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown context"):
            spec.validate()

    def test_rejects_unknown_bakeoff_property(self) -> None:
        spec = PromptSpec.load(SPEC_PATH)
        spec.bakeoff_design.property_ids[0] = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown property"):
            spec.validate()

    def test_rejects_non_high_risk_bakeoff_property(self) -> None:
        spec = PromptSpec.load(SPEC_PATH)
        selected = spec.get_property(spec.bakeoff_design.property_ids[0])
        selected.high_risk = False
        with self.assertRaisesRegex(ValueError, "high_risk=true"):
            spec.validate()

    def test_rejects_changed_design_factor_count(self) -> None:
        spec = PromptSpec.load(SPEC_PATH)
        spec.bakeoff_design.seeds.pop()
        with self.assertRaisesRegex(ValueError, "bakeoff seeds must contain exactly 4"):
            spec.validate()

    def test_rejects_changed_property_count(self) -> None:
        spec = PromptSpec.load(SPEC_PATH)
        spec.properties.pop()
        with self.assertRaisesRegex(ValueError, "exactly 26 properties"):
            spec.validate()


class TestPlanBuilding(unittest.TestCase):
    def test_exact_plan_counts_and_unique_ids(self) -> None:
        spec = PromptSpec.load(SPEC_PATH)
        audit_rows = build_audit_plan(spec)
        bakeoff_rows = build_bakeoff_plan(spec)
        validate_coverage(audit_rows, bakeoff_rows, spec)

        self.assertEqual(len(audit_rows), 104)
        self.assertEqual(len(bakeoff_rows), 576)
        row_ids = [row.row_id for row in audit_rows + bakeoff_rows]
        self.assertEqual(len(row_ids), len(set(row_ids)))

    def test_validate_coverage_rejects_short_plan(self) -> None:
        spec = PromptSpec.load(SPEC_PATH)
        with self.assertRaisesRegex(ValueError, "audit plan has 103"):
            validate_coverage(
                build_audit_plan(spec)[:-1], build_bakeoff_plan(spec), spec
            )

    def test_validate_coverage_rejects_changed_factor_count(self) -> None:
        spec = PromptSpec.load(SPEC_PATH)
        spec.audit_design.seeds.pop()
        with self.assertRaisesRegex(ValueError, "audit seeds must contain exactly 2"):
            validate_coverage(build_audit_plan(spec), build_bakeoff_plan(spec), spec)

    def test_validate_coverage_rejects_changed_property_count(self) -> None:
        spec = PromptSpec.load(SPEC_PATH)
        spec.properties.pop()
        with self.assertRaisesRegex(ValueError, "exactly 26 properties"):
            validate_coverage(build_audit_plan(spec), build_bakeoff_plan(spec), spec)

    def test_materialized_prompts_do_not_duplicate_subjects(self) -> None:
        spec = PromptSpec.load(SPEC_PATH)
        rows = build_audit_plan(spec) + build_bakeoff_plan(spec)
        for row in rows:
            self.assertNotIn("A person A person", row.prompt)
            self.assertNotIn("person person", row.prompt.lower())

    def test_plan_order_and_prompts_are_deterministic(self) -> None:
        spec = PromptSpec.load(SPEC_PATH)
        first = [row.to_dict() for row in build_bakeoff_plan(spec)]
        second = [row.to_dict() for row in build_bakeoff_plan(spec)]
        self.assertEqual(first, second)


class TestBlinding(unittest.TestCase):
    def test_mapping_uses_random_ids_paths_and_randomized_order(self) -> None:
        rows = build_audit_plan(PromptSpec.load(SPEC_PATH))[:12]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            mapping = load_or_create_blinding_mapping(output_dir, rows)
            original_order = [row.row_id for row in rows]

            mapping_mode = stat.S_IMODE(
                (output_dir / "blinded_row_mapping.json").stat().st_mode
            )
            self.assertEqual(mapping_mode, 0o600)
            self.assertNotEqual(mapping["scoring_order"], original_order)
            for row in rows:
                entry = mapping["rows"][row.row_id]
                enumerable_hash = hashlib.sha256(row.row_id.encode()).hexdigest()[:12]
                self.assertNotEqual(entry["blinded_row_id"], enumerable_hash)
                self.assertNotIn(row.property_id, entry["blinded_row_id"])
                self.assertNotIn(row.model_backbone, entry["image_path"])
                self.assertEqual(
                    Path(entry["image_path"]).parent.name, "blinded_images"
                )

    def test_resume_repairs_mapping_permissions(self) -> None:
        rows = build_audit_plan(PromptSpec.load(SPEC_PATH))[:2]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            original = load_or_create_blinding_mapping(output_dir, rows)
            mapping_path = output_dir / "blinded_row_mapping.json"
            mapping_path.chmod(0o644)

            resumed = load_or_create_blinding_mapping(output_dir, rows)

            self.assertEqual(resumed, original)
            self.assertEqual(stat.S_IMODE(mapping_path.stat().st_mode), 0o600)

    def test_sheet_contains_only_blinded_locators_and_scores(self) -> None:
        rows = build_audit_plan(PromptSpec.load(SPEC_PATH))[:4]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            mapping = load_or_create_blinding_mapping(output_dir, rows)
            write_scoring_sheet(output_dir, mapping)

            with (output_dir / "scoring_sheet.csv").open(newline="") as file:
                sheet_rows = list(csv.DictReader(file))
            self.assertEqual(
                list(sheet_rows[0]),
                [
                    "blinded_row_id",
                    "image_path",
                    "target_present",
                    "target_visible",
                    "prompt_ambiguous",
                    "notes",
                ],
            )
            serialized = json.dumps(sheet_rows)
            for row in rows:
                self.assertNotIn(row.property_id, serialized)
                self.assertNotIn(row.model_backbone, serialized)
                self.assertNotIn(row.row_id, serialized)

    def test_resume_preserves_mapping_and_scored_sheet(self) -> None:
        rows = build_audit_plan(PromptSpec.load(SPEC_PATH))[:3]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            first_mapping = load_or_create_blinding_mapping(output_dir, rows)
            write_scoring_sheet(output_dir, first_mapping)
            mapping_bytes = (output_dir / "blinded_row_mapping.json").read_bytes()
            scoring_path = output_dir / "scoring_sheet.csv"
            scoring_path.write_text("completed scores\n")

            second_mapping = load_or_create_blinding_mapping(output_dir, rows)
            write_scoring_sheet(output_dir, second_mapping)
            self.assertEqual(second_mapping, first_mapping)
            self.assertEqual(
                (output_dir / "blinded_row_mapping.json").read_bytes(), mapping_bytes
            )
            self.assertEqual(scoring_path.read_text(), "completed scores\n")


class TestModelFingerprint(unittest.TestCase):
    def test_configuration_records_required_model_contract(self) -> None:
        backbone = FakeQualificationBackbone()
        configuration = build_model_configuration("sd35_large_turbo", backbone)
        self.assertEqual(configuration["model_id"], backbone.model_id)
        self.assertEqual(configuration["revision"], "test-revision")
        self.assertEqual(configuration["render_parameters"]["num_inference_steps"], 4)
        self.assertEqual(
            configuration["stream_contract"],
            [stream.to_dict() for stream in backbone.stream_specs],
        )
        self.assertEqual(
            configuration["configuration"]["transformer_quantization"], "nf4"
        )
        self.assertEqual(configuration["runtime"]["device"], "cuda")
        self.assertEqual(
            configuration["runtime"]["effective_inference_dtype"], "bfloat16"
        )

    def test_each_required_section_changes_fingerprint(self) -> None:
        configuration = build_model_configuration(
            "sd35_large_turbo", FakeQualificationBackbone()
        )
        original = fingerprint_model_configuration(configuration)
        mutations = [
            ("model_id", "different/model"),
            ("revision", "different-revision"),
        ]
        for key, value in mutations:
            changed = copy.deepcopy(configuration)
            changed[key] = value
            self.assertNotEqual(fingerprint_model_configuration(changed), original)

        for section in (
            "configuration",
            "loaded_model_markers",
            "runtime",
            "render_parameters",
            "stream_contract",
        ):
            changed = copy.deepcopy(configuration)
            if section == "stream_contract":
                changed[section][0]["dtype"] = "float32"
            else:
                changed[section]["mutation"] = True
            self.assertNotEqual(fingerprint_model_configuration(changed), original)

    def test_loaded_commit_resolves_missing_instance_revision(self) -> None:
        backbone = FakeQualificationBackbone()
        backbone.revision = None
        backbone.pipeline = SimpleNamespace(
            config={"_commit_hash": "pipeline-commit", "_diffusers_version": "1.0"}
        )
        configuration = build_model_configuration("sd35_large_turbo", backbone)
        self.assertEqual(configuration["revision"], "pipeline-commit")
        self.assertEqual(
            configuration["loaded_model_markers"]["pipeline"]["_commit_hash"],
            "pipeline-commit",
        )

    def test_flux_dev_resolves_internal_render_defaults(self) -> None:
        cpu_configuration = build_model_configuration(
            "flux_dev", FluxDevBackbone(device="cpu")
        )
        cuda_configuration = build_model_configuration(
            "flux_dev", FluxDevBackbone(device="cuda")
        )
        self.assertEqual(cpu_configuration["revision"], "default")
        self.assertEqual(
            cpu_configuration["render_parameters"]["num_inference_steps"], 50
        )
        self.assertEqual(cpu_configuration["render_parameters"]["guidance_scale"], 3.5)
        self.assertEqual(
            cpu_configuration["render_parameters"]["max_sequence_length"], 512
        )
        self.assertEqual(cpu_configuration["runtime"]["device"], "cpu")
        self.assertEqual(
            cpu_configuration["runtime"]["effective_inference_dtype"], "float32"
        )
        self.assertEqual(cuda_configuration["runtime"]["device"], "cuda")
        self.assertEqual(
            cuda_configuration["runtime"]["effective_inference_dtype"], "bfloat16"
        )
        self.assertNotEqual(
            fingerprint_model_configuration(cpu_configuration),
            fingerprint_model_configuration(cuda_configuration),
        )

    def test_sd35_large_uses_backbone_context_contract(self) -> None:
        configuration = build_model_configuration(
            "sd35_large", FakeQualificationBackbone()
        )
        self.assertEqual(configuration["render_parameters"]["num_inference_steps"], 28)
        self.assertEqual(configuration["render_parameters"]["guidance_scale"], 3.5)
        self.assertEqual(configuration["render_parameters"]["max_sequence_length"], 256)

    def test_saved_success_requires_current_spec_and_model_fingerprints(self) -> None:
        row = build_audit_plan(PromptSpec.load(SPEC_PATH))[0]
        configuration = build_model_configuration(
            "sd35_large_turbo", FakeQualificationBackbone()
        )
        model_fingerprint = fingerprint_model_configuration(configuration)
        manifest_row = {
            **row.to_dict(),
            "spec_fingerprint": "current-spec",
            "model_configuration": configuration,
            "model_fingerprint": model_fingerprint,
            "status": "success",
        }
        self.assertTrue(
            manifest_success_matches(
                manifest_row,
                row,
                "current-spec",
                configuration,
                model_fingerprint,
            )
        )

        stale_spec = {**manifest_row, "spec_fingerprint": "stale-spec"}
        self.assertFalse(
            manifest_success_matches(
                stale_spec,
                row,
                "current-spec",
                configuration,
                model_fingerprint,
            )
        )
        stale_model = {**manifest_row, "model_fingerprint": "stale-model"}
        self.assertFalse(
            manifest_success_matches(
                stale_model,
                row,
                "current-spec",
                configuration,
                model_fingerprint,
            )
        )


class TestCLI(unittest.TestCase):
    @patch("evaluation.run_native_model_qualification.get_backbone")
    def test_dry_run_does_not_load_a_backbone(self, get_backbone: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            argv = [
                "run_native_model_qualification.py",
                "--output_dir",
                directory,
                "--dry_run",
            ]
            with patch.object(sys, "argv", argv):
                from evaluation.run_native_model_qualification import main

                main()

            get_backbone.assert_not_called()
            with (Path(directory) / "plan.json").open("r") as file:
                plan = json.load(file)
            self.assertEqual(plan["audit_count"], 104)
            self.assertEqual(plan["bakeoff_count"], 576)
            self.assertEqual(plan["total_count"], 680)

    def test_all_success_resume_canonicalizes_duplicate_ids(self) -> None:
        row = RenderRow(
            "audit_hair_blond_ctx_simple_s42",
            "native",
            "sd35_large_turbo",
            "hair_blond",
            "ctx_simple",
            42,
            "with blonde hair",
        )
        backbone = FakeQualificationBackbone()
        model_configuration = build_model_configuration("sd35_large_turbo", backbone)
        manifest_row = {
            **row.to_dict(),
            "status": "success",
            "spec_fingerprint": "spec-fingerprint",
            "model_configuration": model_configuration,
            "model_fingerprint": fingerprint_model_configuration(model_configuration),
            "error": None,
        }
        fake_spec = MagicMock()
        fake_spec.fingerprint.return_value = "spec-fingerprint"

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            legacy_path = output_dir / "legacy.png"
            legacy_path.write_bytes(b"existing image")
            manifest_row["image_path"] = str(legacy_path)
            manifest_path = output_dir / "manifest.jsonl"
            with manifest_path.open("w") as file:
                file.write(json.dumps(manifest_row) + "\n")
                file.write(json.dumps(manifest_row) + "\n")

            argv = ["run_native_model_qualification.py", "--output_dir", directory]
            with (
                patch.object(sys, "argv", argv),
                patch(
                    "evaluation.run_native_model_qualification.PromptSpec.load",
                    return_value=fake_spec,
                ),
                patch(
                    "evaluation.run_native_model_qualification.build_audit_plan",
                    return_value=[row],
                ),
                patch(
                    "evaluation.run_native_model_qualification.build_bakeoff_plan",
                    return_value=[],
                ),
                patch("evaluation.run_native_model_qualification.validate_coverage"),
                patch(
                    "evaluation.run_native_model_qualification.list_backbones",
                    return_value=["sd35_large_turbo"],
                ),
                patch(
                    "evaluation.run_native_model_qualification.get_backbone",
                    return_value=backbone,
                ),
            ):
                from evaluation.run_native_model_qualification import main

                main()

            final_rows = [
                json.loads(line)
                for line in manifest_path.read_text().splitlines()
                if line
            ]
            self.assertEqual(len(final_rows), 1)
            self.assertEqual(final_rows[0]["row_id"], row.row_id)
            blinded_path = Path(final_rows[0]["image_path"])
            self.assertTrue(blinded_path.is_file())
            self.assertEqual(blinded_path.read_bytes(), b"existing image")

        backbone.encode.assert_not_called()
        self.assertEqual(backbone.decode_calls, [])
        backbone.load.assert_called_once_with()
        backbone.unload.assert_called_once_with()

    def test_stale_spec_and_model_successes_are_rerendered(self) -> None:
        rows = [
            RenderRow(
                "audit_hair_blond_ctx_simple_s42",
                "native",
                "sd35_large_turbo",
                "hair_blond",
                "ctx_simple",
                42,
                "with blonde hair",
            ),
            RenderRow(
                "audit_hair_brunette_ctx_simple_s42",
                "native",
                "sd35_large_turbo",
                "hair_brunette",
                "ctx_simple",
                42,
                "with brown hair",
            ),
        ]
        backbone = FakeQualificationBackbone()
        model_configuration = build_model_configuration("sd35_large_turbo", backbone)
        model_fingerprint = fingerprint_model_configuration(model_configuration)
        manifest_rows = [
            {
                **rows[0].to_dict(),
                "status": "success",
                "spec_fingerprint": "stale-spec",
                "model_configuration": model_configuration,
                "model_fingerprint": model_fingerprint,
                "error": None,
            },
            {
                **rows[1].to_dict(),
                "status": "success",
                "spec_fingerprint": "spec-fingerprint",
                "model_configuration": model_configuration,
                "model_fingerprint": "stale-model",
                "error": None,
            },
        ]
        fake_spec = MagicMock()
        fake_spec.fingerprint.return_value = "spec-fingerprint"

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            manifest_path = output_dir / "manifest.jsonl"
            for index, manifest_row in enumerate(manifest_rows):
                legacy_path = output_dir / f"legacy-{index}.png"
                legacy_path.write_bytes(b"stale image")
                manifest_row["image_path"] = str(legacy_path)
            with manifest_path.open("w") as file:
                for manifest_row in manifest_rows:
                    file.write(json.dumps(manifest_row) + "\n")

            argv = ["run_native_model_qualification.py", "--output_dir", directory]
            with (
                patch.object(sys, "argv", argv),
                patch(
                    "evaluation.run_native_model_qualification.PromptSpec.load",
                    return_value=fake_spec,
                ),
                patch(
                    "evaluation.run_native_model_qualification.build_audit_plan",
                    return_value=rows,
                ),
                patch(
                    "evaluation.run_native_model_qualification.build_bakeoff_plan",
                    return_value=[],
                ),
                patch("evaluation.run_native_model_qualification.validate_coverage"),
                patch(
                    "evaluation.run_native_model_qualification.list_backbones",
                    return_value=["sd35_large_turbo"],
                ),
                patch(
                    "evaluation.run_native_model_qualification.get_backbone",
                    return_value=backbone,
                ),
            ):
                from evaluation.run_native_model_qualification import main

                main()

            final_rows = [
                json.loads(line)
                for line in manifest_path.read_text().splitlines()
                if line
            ]
            self.assertEqual(len(final_rows), 2)
            self.assertTrue(
                all(row["spec_fingerprint"] == "spec-fingerprint" for row in final_rows)
            )
            self.assertTrue(
                all(row["model_fingerprint"] == model_fingerprint for row in final_rows)
            )
            self.assertTrue(
                all(
                    Path(row["image_path"]).read_bytes() == b"fake image"
                    for row in final_rows
                )
            )

        self.assertEqual(len(backbone.decode_calls), 2)

    def test_resume_skips_success_and_replaces_failed_record(self) -> None:
        successful = RenderRow(
            "audit_hair_blond_ctx_simple_s42",
            "native",
            "sd35_large_turbo",
            "hair_blond",
            "ctx_simple",
            42,
            "with blonde hair",
        )
        failed = RenderRow(
            "audit_hair_brunette_ctx_simple_s42",
            "native",
            "sd35_large_turbo",
            "hair_brunette",
            "ctx_simple",
            42,
            "with brown hair",
        )
        model_configuration = build_model_configuration(
            "sd35_large_turbo", FakeQualificationBackbone()
        )
        existing_success = {
            **successful.to_dict(),
            "status": "success",
            "image_path": None,
            "spec_fingerprint": "spec-fingerprint",
            "model_configuration": model_configuration,
            "model_fingerprint": fingerprint_model_configuration(model_configuration),
            "sentinel": "preserve",
        }
        existing_failure = {
            "row_id": failed.row_id,
            "status": "error",
            "image_path": None,
            "error": "previous failure",
        }
        backbone = FakeQualificationBackbone()
        fake_spec = MagicMock()
        fake_spec.fingerprint.return_value = "spec-fingerprint"

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            manifest_path = output_dir / "manifest.jsonl"
            legacy_image_path = output_dir / "legacy-success.png"
            legacy_image_path.write_bytes(b"existing image")
            existing_success["image_path"] = str(legacy_image_path)
            with manifest_path.open("w") as file:
                file.write(json.dumps(existing_success) + "\n")
                file.write(json.dumps(existing_success) + "\n")
                file.write(json.dumps(existing_failure) + "\n")

            argv = ["run_native_model_qualification.py", "--output_dir", directory]
            with (
                patch.object(sys, "argv", argv),
                patch(
                    "evaluation.run_native_model_qualification.PromptSpec.load",
                    return_value=fake_spec,
                ),
                patch(
                    "evaluation.run_native_model_qualification.build_audit_plan",
                    return_value=[successful, failed],
                ),
                patch(
                    "evaluation.run_native_model_qualification.build_bakeoff_plan",
                    return_value=[],
                ),
                patch("evaluation.run_native_model_qualification.validate_coverage"),
                patch(
                    "evaluation.run_native_model_qualification.list_backbones",
                    return_value=["sd35_large_turbo"],
                ),
                patch(
                    "evaluation.run_native_model_qualification.get_backbone",
                    return_value=backbone,
                ),
            ):
                from evaluation.run_native_model_qualification import main

                main()

            with manifest_path.open("r") as file:
                final_rows = [json.loads(line) for line in file if line.strip()]
            with (output_dir / "blinded_row_mapping.json").open("r") as file:
                mapping = json.load(file)
            scorer_paths = [
                output_dir / mapping["rows"][row.row_id]["image_path"]
                for row in (successful, failed)
            ]
            self.assertTrue(all(path.is_file() for path in scorer_paths))
            self.assertEqual(scorer_paths[0].read_bytes(), b"existing image")

        self.assertEqual(len(final_rows), 2)
        self.assertEqual(len({row["row_id"] for row in final_rows}), 2)
        final_by_id = {row["row_id"]: row for row in final_rows}
        self.assertEqual(final_by_id[successful.row_id]["status"], "success")
        self.assertEqual(final_by_id[successful.row_id]["sentinel"], "preserve")
        self.assertEqual(final_by_id[failed.row_id]["status"], "success")
        self.assertIsNone(final_by_id[failed.row_id]["error"])
        backbone.encode.assert_called_once_with(failed.prompt)
        self.assertEqual(len(backbone.decode_calls), 1)
        self.assertEqual(backbone.decode_calls[0][2], failed.seed)
        backbone.load.assert_called_once_with()
        backbone.unload.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
