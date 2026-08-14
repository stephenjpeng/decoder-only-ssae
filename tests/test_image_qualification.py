"""Tests for native-model qualification planning and execution."""

from __future__ import annotations

import copy
import csv
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
from backbones.sd35_large_turbo import Sd35LargeTurboBackbone
from evaluation.image_qualification import (
    Context,
    PromptSpec,
    Property,
    RenderRow,
    build_audit_plan,
    build_bakeoff_plan,
    validate_coverage,
)
from evaluation.image_validity import analyze_image_validity
from evaluation.run_native_model_qualification import (
    QUALIFICATION_RENDER_CONTRACT_VERSION,
    build_model_configuration,
    build_scoring_package,
    fingerprint_model_configuration,
    manifest_success_matches,
    prune_manifest_rows,
    qualification_identity,
    qualification_repository_revision,
    write_manifest,
)

REPOSITORY_REVISION = "a" * 40

SPEC_PATH = "evaluation/config/image_validity_prompts.yaml"


class FakeQualificationBackbone:
    """Small observable backbone used to exercise resume without model weights."""

    model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
    revision = "test-revision"
    device = "cuda"
    dtype = "bfloat16"
    max_length = 512
    stream_specs = [
        StreamSpec("seq", (333, 4096), "float16", "embds.h5"),
        StreamSpec("pooled", (2048,), "float16", "embds_pooled.h5"),
    ]

    def __init__(self) -> None:
        self.load = MagicMock()
        self.unload = MagicMock()
        self.generate_calls: list[tuple[str, Path, int | None]] = []

    def generate(
        self,
        prompt: str,
        output_path: Path,
        seed: int | None = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        max_sequence_length: int = 256,
    ) -> Path:
        """Record a successful native direct-text render."""
        self.generate_calls.append((prompt, output_path, seed))
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
        self.assertTrue(
            all(
                row.design == "audit"
                and row.conditioning == "direct_text"
                and row.effective_context == 256
                for row in audit_rows
            )
        )
        self.assertTrue(
            all(
                row.design == "bakeoff"
                and row.conditioning == "direct_text"
                and row.effective_context
                == (512 if row.model_backbone == "flux_dev" else 256)
                for row in bakeoff_rows
            )
        )

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
    def _successful_manifest(
        self, output_dir: Path, rows: list[RenderRow], identity: str = "identity"
    ) -> dict[str, dict[str, object]]:
        manifest = {}
        private = output_dir / "private_images"
        private.mkdir()
        for row in rows:
            path = private / f"{row.row_id}.png"
            path.write_bytes(b"pixels")
            manifest[row.row_id] = {
                **row.to_dict(),
                "status": "success",
                "image_path": str(path),
                "qualification_identity": f"{identity}-{row.row_id}",
            }
        return manifest

    def test_package_is_opaque_complete_and_private(self) -> None:
        rows = build_bakeoff_plan(PromptSpec.load(SPEC_PATH))[:4]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            manifest = self._successful_manifest(output_dir, rows)
            mapping = build_scoring_package(output_dir, rows, manifest)
            write_manifest(output_dir / "manifest.jsonl", manifest)

            self.assertEqual(
                stat.S_IMODE((output_dir / "blinded_row_mapping.json").stat().st_mode),
                0o600,
            )
            self.assertEqual(
                stat.S_IMODE((output_dir / "manifest.jsonl").stat().st_mode), 0o600
            )
            with (output_dir / "scoring_sheet.csv").open(newline="") as file:
                sheet_rows = list(csv.DictReader(file))
            self.assertEqual(len(sheet_rows), len(rows))
            self.assertEqual(
                list(sheet_rows[0]),
                [
                    "blinded_row_id",
                    "image_path",
                    "target_phrase",
                    "prompt",
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
                self.assertIn(row.prompt, serialized)
                self.assertIn(row.target_phrase, serialized)
                entry = mapping["rows"][row.row_id]
                blinded_path = output_dir / entry["image_path"]
                self.assertTrue(blinded_path.is_file())
                self.assertEqual(blinded_path.stat().st_mtime_ns, 0)

    def test_public_files_are_created_outside_plan_and_model_order(self) -> None:
        rows = build_bakeoff_plan(PromptSpec.load(SPEC_PATH))[:6]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            manifest = self._successful_manifest(output_dir, rows)
            source_order: list[str] = []

            def record_copy(source: Path, destination: Path) -> None:
                source_order.append(str(source))
                Path(destination).write_bytes(Path(source).read_bytes())

            no_shuffle = SimpleNamespace(shuffle=lambda values: None)
            with (
                patch(
                    "evaluation.run_native_model_qualification.secrets.SystemRandom",
                    return_value=no_shuffle,
                ),
                patch(
                    "evaluation.run_native_model_qualification.shutil.copyfile",
                    side_effect=record_copy,
                ),
                patch(
                    "evaluation.run_native_model_qualification.secrets.randbelow",
                    return_value=2,
                ) as randbelow,
            ):
                build_scoring_package(output_dir, rows, manifest)

            planned_sources = [manifest[row.row_id]["image_path"] for row in rows]
            self.assertEqual(source_order, planned_sources[4:] + planned_sources[:4])
            randbelow.assert_called_once_with(4)
            copied_models = [
                next(
                    row.model_backbone
                    for row in rows
                    if manifest[row.row_id]["image_path"] == source
                )
                for source in source_order
            ]
            planned_models = [row.model_backbone for row in rows]
            self.assertEqual(copied_models, planned_models[4:] + planned_models[:4])
            self.assertNotEqual(copied_models, planned_models)

    def test_resume_does_not_touch_retained_public_files_or_scores(self) -> None:
        rows = build_bakeoff_plan(PromptSpec.load(SPEC_PATH))[:3]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            manifest = self._successful_manifest(output_dir, rows)
            first_mapping = build_scoring_package(output_dir, rows, manifest)
            scoring_path = output_dir / "scoring_sheet.csv"
            with scoring_path.open(newline="") as file:
                scored = list(csv.DictReader(file))
            for score in scored:
                score["target_present"] = "true"
                score["target_visible"] = "true"
                score["prompt_ambiguous"] = "false"
            with scoring_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(scored[0]))
                writer.writeheader()
                writer.writerows(scored)
            scored_bytes = scoring_path.read_bytes()
            public_paths = [
                output_dir / entry["image_path"]
                for entry in first_mapping["rows"].values()
            ]
            metadata = {
                path: (path.stat().st_ctime_ns, path.stat().st_mtime_ns)
                for path in public_paths
            }

            with (
                patch(
                    "evaluation.run_native_model_qualification.shutil.copyfile"
                ) as copyfile,
                patch("evaluation.run_native_model_qualification.os.utime") as utime,
            ):
                second_mapping = build_scoring_package(output_dir, rows, manifest)

            copyfile.assert_not_called()
            utime.assert_not_called()
            self.assertEqual(second_mapping, first_mapping)
            self.assertEqual(scoring_path.read_bytes(), scored_bytes)
            self.assertEqual(
                {
                    path: (path.stat().st_ctime_ns, path.stat().st_mtime_ns)
                    for path in public_paths
                },
                metadata,
            )

    def test_failed_render_is_not_sent_to_scorer(self) -> None:
        rows = build_bakeoff_plan(PromptSpec.load(SPEC_PATH))[:2]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            manifest = self._successful_manifest(output_dir, rows)
            manifest[rows[1].row_id] = {
                **rows[1].to_dict(),
                "status": "error",
                "image_path": None,
                "qualification_identity": "failed",
            }
            mapping = build_scoring_package(output_dir, rows, manifest)
            self.assertEqual(set(mapping["rows"]), {rows[0].row_id})
            with (output_dir / "scoring_sheet.csv").open(newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 1)

    def test_changed_identity_replaces_opaque_id_and_drops_score(self) -> None:
        rows = build_bakeoff_plan(PromptSpec.load(SPEC_PATH))[:1]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            manifest = self._successful_manifest(output_dir, rows, "first")
            first = build_scoring_package(output_dir, rows, manifest)
            opaque_id = first["rows"][rows[0].row_id]["blinded_row_id"]
            scoring_path = output_dir / "scoring_sheet.csv"
            with scoring_path.open(newline="") as file:
                scored = list(csv.DictReader(file))
            scored[0]["target_present"] = "true"
            with scoring_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(scored[0]))
                writer.writeheader()
                writer.writerows(scored)

            manifest[rows[0].row_id]["qualification_identity"] = "second"
            second = build_scoring_package(output_dir, rows, manifest)
            self.assertNotEqual(
                second["rows"][rows[0].row_id]["blinded_row_id"], opaque_id
            )
            with scoring_path.open(newline="") as file:
                refreshed = list(csv.DictReader(file))
            self.assertEqual(refreshed[0]["target_present"], "")


class TestDirectNativeBackbones(unittest.TestCase):
    def test_turbo_and_flux_send_text_without_encode_dispatch(self) -> None:
        for backbone, expected_context in (
            (Sd35LargeTurboBackbone(device="cpu"), 256),
            (FluxDevBackbone(device="cpu"), 512),
        ):
            image = MagicMock()
            pipeline = MagicMock(return_value=SimpleNamespace(images=[image]))
            pipeline.device = "cpu"
            pipeline.encode_prompt = MagicMock()
            backbone.pipeline = pipeline
            backbone._loaded = True
            with tempfile.TemporaryDirectory() as directory:
                backbone.generate(
                    "native prompt", Path(directory) / "image.png", seed=5
                )
            pipeline.encode_prompt.assert_not_called()
            kwargs = pipeline.call_args.kwargs
            self.assertEqual(kwargs["prompt"], "native prompt")
            self.assertNotIn("prompt_embeds", kwargs)
            self.assertEqual(kwargs["max_sequence_length"], expected_context)


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
        self.assertIsNone(cpu_configuration["revision"])
        self.assertFalse(cpu_configuration["revision_is_concrete"])
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

    def test_saved_success_requires_current_render_identity(self) -> None:
        row = build_audit_plan(PromptSpec.load(SPEC_PATH))[0]
        configuration = build_model_configuration(
            "sd35_large_turbo", FakeQualificationBackbone()
        )
        model_fingerprint = fingerprint_model_configuration(configuration)
        contract_identity = qualification_identity(
            row,
            "current-spec",
            model_fingerprint,
            REPOSITORY_REVISION,
            QUALIFICATION_RENDER_CONTRACT_VERSION,
        )
        manifest_row = {
            **row.to_dict(),
            "spec_fingerprint": "current-spec",
            "model_configuration": configuration,
            "model_fingerprint": model_fingerprint,
            "repository_code_revision": REPOSITORY_REVISION,
            "qualification_render_contract_version": QUALIFICATION_RENDER_CONTRACT_VERSION,
            "render_contract_fingerprint": contract_identity,
            "render_attempt_id": "attempt",
            "qualification_identity": f"{contract_identity}-attempt",
            "status": "success",
        }

        def matches(candidate: dict[str, object], revision: str, version: int) -> bool:
            return manifest_success_matches(
                candidate,
                row,
                "current-spec",
                configuration,
                model_fingerprint,
                revision,
                version,
            )

        self.assertTrue(matches(manifest_row, REPOSITORY_REVISION, 1))
        self.assertFalse(
            matches(
                {**manifest_row, "spec_fingerprint": "stale-spec"},
                REPOSITORY_REVISION,
                1,
            )
        )
        self.assertFalse(
            matches(
                {**manifest_row, "model_fingerprint": "stale-model"},
                REPOSITORY_REVISION,
                1,
            )
        )
        self.assertFalse(matches(manifest_row, "b" * 40, 1))
        self.assertFalse(matches(manifest_row, REPOSITORY_REVISION, 2))


class TestQualificationIntegrity(unittest.TestCase):
    def test_real_render_requires_clean_concrete_revision(self) -> None:
        clean = {
            "sha": REPOSITORY_REVISION,
            "branch": "test",
            "dirty": False,
            "describe": REPOSITORY_REVISION,
        }
        with patch(
            "evaluation.run_native_model_qualification.git_provenance",
            return_value=clean,
        ):
            self.assertEqual(qualification_repository_revision(), REPOSITORY_REVISION)

        for provenance in (
            {**clean, "dirty": True},
            {**clean, "dirty": None},
            {**clean, "sha": None},
        ):
            with (
                self.subTest(provenance=provenance),
                patch(
                    "evaluation.run_native_model_qualification.git_provenance",
                    return_value=provenance,
                ),
                self.assertRaises(SystemExit),
            ):
                qualification_repository_revision()

    def test_code_and_contract_change_render_identity(self) -> None:
        row = build_audit_plan(PromptSpec.load(SPEC_PATH))[0]
        current = qualification_identity(row, "spec", "model", "a" * 40, 1)
        changed_code = qualification_identity(row, "spec", "model", "b" * 40, 1)
        changed_contract = qualification_identity(row, "spec", "model", "a" * 40, 2)
        self.assertNotEqual(current, changed_code)
        self.assertNotEqual(current, changed_contract)

    def test_obsolete_manifest_rows_and_private_artifacts_are_pruned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            obsolete_path = output_dir / "obsolete.png"
            obsolete_path.write_bytes(b"obsolete")
            manifest = {
                "current": {"row_id": "current", "status": "success"},
                "obsolete": {
                    "row_id": "obsolete",
                    "status": "success",
                    "image_path": str(obsolete_path),
                },
            }
            retained, pruned = prune_manifest_rows(manifest, {"current"}, output_dir)

            self.assertEqual(set(retained), {"current"})
            self.assertEqual(pruned, 1)
            self.assertFalse(obsolete_path.exists())

    def test_invalid_obsolete_set_cannot_partially_delete_artifacts(self) -> None:
        with (
            tempfile.TemporaryDirectory() as output_directory,
            tempfile.TemporaryDirectory() as external_directory,
        ):
            output_dir = Path(output_directory)
            internal_path = output_dir / "internal.png"
            internal_path.write_bytes(b"internal")
            external_path = Path(external_directory) / "external.png"
            external_path.write_bytes(b"external")
            manifest = {
                "first-internal": {
                    "row_id": "first-internal",
                    "status": "success",
                    "image_path": str(internal_path),
                },
                "later-external": {
                    "row_id": "later-external",
                    "status": "success",
                    "image_path": str(external_path),
                },
            }

            with self.assertRaisesRegex(ValueError, "outside qualification output"):
                prune_manifest_rows(manifest, set(), output_dir)
            self.assertTrue(internal_path.exists())
            self.assertTrue(external_path.exists())

    def test_obsolete_symlink_is_unlinked_without_deleting_target(self) -> None:
        with (
            tempfile.TemporaryDirectory() as output_directory,
            tempfile.TemporaryDirectory() as external_directory,
        ):
            output_dir = Path(output_directory)
            external_path = Path(external_directory) / "external.png"
            external_path.write_bytes(b"external")
            symlink_path = output_dir / "obsolete.png"
            symlink_path.symlink_to(external_path)
            manifest = {
                "obsolete": {
                    "row_id": "obsolete",
                    "status": "success",
                    "image_path": str(symlink_path),
                }
            }

            retained, pruned = prune_manifest_rows(manifest, set(), output_dir)
            self.assertEqual(retained, {})
            self.assertEqual(pruned, 1)
            self.assertFalse(symlink_path.exists())
            self.assertFalse(symlink_path.is_symlink())
            self.assertTrue(external_path.exists())

    def test_retained_row_protects_shared_artifact_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            shared_path = output_dir / "shared.png"
            shared_path.write_bytes(b"shared")
            manifest = {
                "current": {"row_id": "current", "image_path": str(shared_path)},
                "obsolete": {"row_id": "obsolete", "image_path": str(shared_path)},
            }

            retained, pruned = prune_manifest_rows(manifest, {"current"}, output_dir)
            self.assertEqual(set(retained), {"current"})
            self.assertEqual(pruned, 1)
            self.assertTrue(shared_path.exists())


class TestCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.revision_patch = patch(
            "evaluation.run_native_model_qualification.qualification_repository_revision",
            return_value=REPOSITORY_REVISION,
        )
        self.revision_patch.start()

    def tearDown(self) -> None:
        self.revision_patch.stop()

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
        model_fingerprint = fingerprint_model_configuration(model_configuration)
        contract_identity = qualification_identity(
            row,
            "spec-fingerprint",
            model_fingerprint,
            REPOSITORY_REVISION,
            QUALIFICATION_RENDER_CONTRACT_VERSION,
        )
        manifest_row = {
            **row.to_dict(),
            "status": "success",
            "spec_fingerprint": "spec-fingerprint",
            "model_configuration": model_configuration,
            "model_fingerprint": model_fingerprint,
            "repository_code_revision": REPOSITORY_REVISION,
            "qualification_render_contract_version": QUALIFICATION_RENDER_CONTRACT_VERSION,
            "render_contract_fingerprint": contract_identity,
            "render_attempt_id": "attempt",
            "qualification_identity": f"{contract_identity}-attempt",
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
            obsolete_path = output_dir / "obsolete.png"
            obsolete_path.write_bytes(b"obsolete image")
            obsolete_row = {
                **manifest_row,
                "row_id": "obsolete-row",
                "image_path": str(obsolete_path),
            }
            with manifest_path.open("w") as file:
                file.write(json.dumps(manifest_row) + "\n")
                file.write(json.dumps(manifest_row) + "\n")
                file.write(json.dumps(obsolete_row) + "\n")

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
            self.assertEqual(stat.S_IMODE(blinded_path.stat().st_mode), 0o600)
            self.assertFalse(obsolete_path.exists())

        self.assertEqual(backbone.generate_calls, [])
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

        self.assertEqual(len(backbone.generate_calls), 2)

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
        model_fingerprint = fingerprint_model_configuration(model_configuration)
        contract_identity = qualification_identity(
            successful,
            "spec-fingerprint",
            model_fingerprint,
            REPOSITORY_REVISION,
            QUALIFICATION_RENDER_CONTRACT_VERSION,
        )
        existing_success = {
            **successful.to_dict(),
            "status": "success",
            "image_path": None,
            "spec_fingerprint": "spec-fingerprint",
            "model_configuration": model_configuration,
            "model_fingerprint": model_fingerprint,
            "repository_code_revision": REPOSITORY_REVISION,
            "qualification_render_contract_version": QUALIFICATION_RENDER_CONTRACT_VERSION,
            "render_contract_fingerprint": contract_identity,
            "render_attempt_id": "attempt",
            "qualification_identity": f"{contract_identity}-attempt",
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
            self.assertEqual(mapping["rows"], {})
            self.assertEqual(legacy_image_path.read_bytes(), b"existing image")
            generated_path = Path(
                next(
                    row["image_path"]
                    for row in final_rows
                    if row["row_id"] == failed.row_id
                )
            )
            self.assertEqual(generated_path.read_bytes(), b"fake image")
            self.assertEqual(stat.S_IMODE(generated_path.stat().st_mode), 0o600)
            self.assertEqual(
                stat.S_IMODE((output_dir / "private_images").stat().st_mode), 0o700
            )

        self.assertEqual(len(final_rows), 2)
        self.assertEqual(len({row["row_id"] for row in final_rows}), 2)
        final_by_id = {row["row_id"]: row for row in final_rows}
        self.assertEqual(final_by_id[successful.row_id]["status"], "success")
        self.assertEqual(final_by_id[successful.row_id]["sentinel"], "preserve")
        self.assertEqual(final_by_id[failed.row_id]["status"], "success")
        self.assertIsNone(final_by_id[failed.row_id]["error"])
        self.assertEqual(len(backbone.generate_calls), 1)
        self.assertEqual(backbone.generate_calls[0][0], failed.prompt)
        self.assertEqual(backbone.generate_calls[0][2], failed.seed)
        backbone.load.assert_called_once_with()
        backbone.unload.assert_called_once_with()

    def test_mocked_render_blind_score_unblind_and_bakeoff_analysis(self) -> None:
        rows = [
            RenderRow(
                "audit-turbo",
                "native",
                "sd35_large_turbo",
                "audit-property",
                "context",
                11,
                "audit prompt",
                design="audit",
                target_phrase="audit phrase",
            )
        ] + [
            RenderRow(
                f"bakeoff-{model}",
                "native",
                model,
                "target-property",
                "context",
                17,
                "full target prompt",
                design="bakeoff",
                effective_context=512 if model == "flux_dev" else 256,
                target_phrase="target phrase",
            )
            for model in ("sd35_large_turbo", "sd35_large", "flux_dev")
        ]
        audit_rows = rows[:1]
        bakeoff_rows = rows[1:]
        backbone = FakeQualificationBackbone()
        fake_spec = MagicMock()
        fake_spec.fingerprint.return_value = "spec-fingerprint"

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            argv = ["run_native_model_qualification.py", "--output_dir", directory]
            with (
                patch.object(sys, "argv", argv),
                patch(
                    "evaluation.run_native_model_qualification.PromptSpec.load",
                    return_value=fake_spec,
                ),
                patch(
                    "evaluation.run_native_model_qualification.build_audit_plan",
                    return_value=audit_rows,
                ),
                patch(
                    "evaluation.run_native_model_qualification.build_bakeoff_plan",
                    return_value=bakeoff_rows,
                ),
                patch("evaluation.run_native_model_qualification.validate_coverage"),
                patch(
                    "evaluation.run_native_model_qualification.list_backbones",
                    return_value=[
                        "sd35_large_turbo",
                        "sd35_large",
                        "flux_dev",
                    ],
                ),
                patch(
                    "evaluation.run_native_model_qualification.get_backbone",
                    return_value=backbone,
                ),
            ):
                from evaluation.run_native_model_qualification import main

                main()

            scoring_path = output_dir / "scoring_sheet.csv"
            with scoring_path.open(newline="") as file:
                scored = list(csv.DictReader(file))
            self.assertEqual(len(scored), 3)
            with (output_dir / "blinded_row_mapping.json").open() as file:
                scoring_mapping = json.load(file)
            self.assertNotIn("audit-turbo", scoring_mapping["rows"])
            for entry in scoring_mapping["rows"].values():
                self.assertEqual(
                    (output_dir / entry["image_path"]).stat().st_mtime_ns, 0
                )
            for row in scored:
                row["target_present"] = "true"
                row["target_visible"] = "true"
                row["prompt_ambiguous"] = "false"
            with scoring_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(scored[0]))
                writer.writeheader()
                writer.writerows(scored)

            canonical_scores = output_dir / "scores.internal.csv"
            results = analyze_image_validity(
                output_dir / "manifest.jsonl",
                scoring_path,
                blinding_mapping_path=output_dir / "blinded_row_mapping.json",
                unblinded_scores_output=canonical_scores,
            )

            self.assertEqual(results["overall"]["n_total"], 3)
            self.assertEqual(
                set(results["per_model"]),
                {
                    "sd35_large_turbo",
                    "sd35_large",
                    "flux_dev",
                },
            )
            self.assertEqual(results["per_model"]["sd35_large_turbo"]["n_total"], 1)
            self.assertEqual(
                results["per_model_property"]["flux_dev"]["target-property"]["n_valid"],
                1,
            )
            self.assertEqual(stat.S_IMODE(canonical_scores.stat().st_mode), 0o600)

            manifest_path = output_dir / "manifest.jsonl"
            manifest_rows = [
                json.loads(line) for line in manifest_path.read_text().splitlines()
            ]
            for manifest_row in manifest_rows:
                if manifest_row["row_id"] == "bakeoff-flux_dev":
                    manifest_row["qualification_identity"] = "replaced-pixels"
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in manifest_rows)
            )
            manifest_path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "identity does not match"):
                analyze_image_validity(
                    manifest_path,
                    scoring_path,
                    blinding_mapping_path=output_dir / "blinded_row_mapping.json",
                )

        self.assertEqual(len(backbone.generate_calls), 4)


if __name__ == "__main__":
    unittest.main()
