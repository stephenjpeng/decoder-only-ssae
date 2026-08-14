"""
Unit tests for evaluation.image_validity.
"""

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from evaluation.image_validity import (
    EligibilityStats,
    ManifestRow,
    ScoringRow,
    ValidityStats,
    analyze_image_validity,
    calculate_eligibility,
    calculate_per_model_validity,
    calculate_per_property_validity,
    calculate_validity,
    load_manifest_jsonl,
    load_scoring_csv,
    parse_bool_strict,
    wilson_confidence_interval,
    write_results_json,
    write_summary_csv,
)
from evaluation.run_image_validity_analysis import main as validity_main


class TestParseBoolStrict(unittest.TestCase):
    """Test strict boolean parsing"""

    def test_valid_true(self):
        self.assertTrue(parse_bool_strict("true", "field", "row1"))
        self.assertTrue(parse_bool_strict("True", "field", "row1"))
        self.assertTrue(parse_bool_strict("TRUE", "field", "row1"))
        self.assertTrue(parse_bool_strict(" true ", "field", "row1"))

    def test_valid_false(self):
        self.assertFalse(parse_bool_strict("false", "field", "row1"))
        self.assertFalse(parse_bool_strict("False", "field", "row1"))
        self.assertFalse(parse_bool_strict("FALSE", "field", "row1"))
        self.assertFalse(parse_bool_strict(" false ", "field", "row1"))

    def test_invalid_values(self):
        for invalid in ["1", "0", "yes", "no", "y", "n", "maybe"]:
            with self.assertRaises(ValueError) as ctx:
                parse_bool_strict(invalid, "test_field", "row123")
            self.assertIn("invalid boolean", str(ctx.exception))
            self.assertIn("test_field", str(ctx.exception))
            self.assertIn("row123", str(ctx.exception))

    def test_none_value(self):
        with self.assertRaises(ValueError) as ctx:
            parse_bool_strict(None, "test_field", "row123")
        self.assertIn("missing value", str(ctx.exception))
        self.assertIn("test_field", str(ctx.exception))

    def test_empty_string(self):
        with self.assertRaises(ValueError) as ctx:
            parse_bool_strict("", "test_field", "row123")
        self.assertIn("empty value", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            parse_bool_strict("  ", "test_field", "row123")
        self.assertIn("empty value", str(ctx.exception))


class TestWilsonConfidenceInterval(unittest.TestCase):
    """Test Wilson score confidence interval calculation"""

    def test_zero_total(self):
        # edge case: no trials
        lower, upper = wilson_confidence_interval(0, 0)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 0.0)

    def test_all_failures(self):
        # edge case: all failures
        lower, upper = wilson_confidence_interval(0, 10)
        self.assertGreaterEqual(lower, 0.0)
        self.assertLess(upper, 0.5)
        # CI should not be [0, 0]
        self.assertGreater(upper, 0.0)

    def test_all_successes(self):
        # edge case: all successes
        lower, upper = wilson_confidence_interval(10, 10)
        self.assertGreater(lower, 0.5)
        self.assertLessEqual(upper, 1.0)
        # CI should not be [1, 1]
        self.assertLess(lower, 1.0)

    def test_fifty_percent(self):
        # 50% success rate
        lower, upper = wilson_confidence_interval(50, 100)
        self.assertGreater(lower, 0.0)
        self.assertLess(upper, 1.0)
        # should be reasonably centered around 0.5
        self.assertLess(abs((lower + upper) / 2 - 0.5), 0.1)

    def test_ninety_percent_boundary(self):
        # test around 90% threshold
        lower_90, upper_90 = wilson_confidence_interval(90, 100)
        self.assertGreater(lower_90, 0.8)
        self.assertLess(upper_90, 1.0)

        # 89/100 should have lower CI below 90%
        lower_89, upper_89 = wilson_confidence_interval(89, 100)
        self.assertLess(lower_89, 0.90)

    def test_confidence_bounds(self):
        # bounds must be [0, 1]
        for n_success in [0, 5, 10]:
            for n_total in [10, 50, 100]:
                if n_success > n_total:
                    continue
                lower, upper = wilson_confidence_interval(n_success, n_total)
                self.assertGreaterEqual(lower, 0.0)
                self.assertLessEqual(upper, 1.0)
                self.assertLessEqual(lower, upper)


class TestLoadScoringCSV(unittest.TestCase):
    """Test scoring CSV loading and validation"""

    def test_valid_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("row_id,target_present,target_visible,prompt_ambiguous,notes\n")
            f.write("row1,true,true,false,looks good\n")
            f.write("row2,false,false,false,\n")
            f.write("row3,true,false,true,target small\n")
            csv_path = Path(f.name)

        try:
            scores = load_scoring_csv(csv_path)
            self.assertEqual(len(scores), 3)

            self.assertTrue(scores["row1"].target_present)
            self.assertTrue(scores["row1"].target_visible)
            self.assertFalse(scores["row1"].prompt_ambiguous)
            self.assertEqual(scores["row1"].notes, "looks good")

            self.assertFalse(scores["row2"].target_present)
            self.assertFalse(scores["row2"].target_visible)

            self.assertTrue(scores["row3"].target_present)
            self.assertFalse(scores["row3"].target_visible)
            self.assertTrue(scores["row3"].prompt_ambiguous)
        finally:
            csv_path.unlink()

    def test_duplicate_row_id(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("row_id,target_present,target_visible,prompt_ambiguous,notes\n")
            f.write("row1,true,true,false,\n")
            f.write("row1,false,false,false,\n")
            csv_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_scoring_csv(csv_path)
            self.assertIn("duplicate row_id", str(ctx.exception))
            self.assertIn("row1", str(ctx.exception))
        finally:
            csv_path.unlink()

    def test_invalid_boolean(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("row_id,target_present,target_visible,prompt_ambiguous,notes\n")
            f.write("row1,yes,true,false,\n")
            csv_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_scoring_csv(csv_path)
            self.assertIn("invalid boolean", str(ctx.exception))
        finally:
            csv_path.unlink()

    def test_missing_column(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("row_id,target_present,target_visible\n")
            f.write("row1,true,true\n")
            csv_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_scoring_csv(csv_path)
            self.assertIn("missing required columns", str(ctx.exception))
        finally:
            csv_path.unlink()

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_scoring_csv(Path("/nonexistent/scores.csv"))

    def test_empty_row_id(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("row_id,target_present,target_visible,prompt_ambiguous,notes\n")
            f.write(",true,true,false,\n")
            csv_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_scoring_csv(csv_path)
            self.assertIn("empty row_id", str(ctx.exception))
        finally:
            csv_path.unlink()

    def test_truncated_csv_row(self):
        """CSV row missing trailing notes column should load cleanly"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("row_id,target_present,target_visible,prompt_ambiguous,notes\n")
            # row with explicit empty notes
            f.write("row1,true,true,false,\n")
            # row missing notes column entirely (csv.DictReader sets to None)
            f.write("row2,true,false,true\n")
            csv_path = Path(f.name)

        try:
            scores = load_scoring_csv(csv_path)
            self.assertEqual(len(scores), 2)
            self.assertEqual(scores["row1"].notes, "")
            self.assertEqual(scores["row2"].notes, "")
            self.assertTrue(scores["row2"].prompt_ambiguous)
        finally:
            csv_path.unlink()

    def test_missing_required_boolean(self):
        """Truncated row missing a required boolean should raise clear error"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("row_id,target_present,target_visible,prompt_ambiguous,notes\n")
            f.write("row1,true,true\n")  # missing prompt_ambiguous
            csv_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_scoring_csv(csv_path)
            self.assertIn("missing value", str(ctx.exception))
            self.assertIn("prompt_ambiguous", str(ctx.exception))
        finally:
            csv_path.unlink()

    def test_surplus_csv_value(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("row_id,target_present,target_visible,prompt_ambiguous,notes\n")
            f.write("row1,true,true,false,note,unexpected\n")
            csv_path = Path(f.name)

        try:
            with self.assertRaisesRegex(ValueError, "more values than columns"):
                load_scoring_csv(csv_path)
        finally:
            csv_path.unlink()

    def test_duplicate_csv_header(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(
                "row_id,target_present,target_visible,prompt_ambiguous,target_visible\n"
            )
            f.write("row1,true,true,false,true\n")
            csv_path = Path(f.name)

        try:
            with self.assertRaisesRegex(ValueError, "duplicate column names"):
                load_scoring_csv(csv_path)
        finally:
            csv_path.unlink()


class TestLoadManifestJSONL(unittest.TestCase):
    """Test manifest JSONL loading and validation"""

    def test_valid_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "row_id": "r1",
                        "model_backbone": "sd35_large_turbo",
                        "property_id": "red_sphere",
                        "context_id": "c1",
                        "seed": 42,
                        "stage": "native",
                        "image_path": "/path/to/img1.png",
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "row_id": "r2",
                        "model_backbone": "flux_dev",
                        "property_id": "blue_cube",
                        "context_id": "c2",
                        "seed": 123,
                        "stage": "edit",
                        "image_path": "/path/to/img2.png",
                        "source_row_id": "r1",
                    }
                )
                + "\n"
            )
            jsonl_path = Path(f.name)

        try:
            rows = load_manifest_jsonl(jsonl_path)
            self.assertEqual(len(rows), 2)

            self.assertEqual(rows[0].row_id, "r1")
            self.assertEqual(rows[0].model_backbone, "sd35_large_turbo")
            self.assertEqual(rows[0].seed, 42)
            self.assertEqual(rows[0].stage, "native")
            self.assertIsNone(rows[0].source_row_id)

            self.assertEqual(rows[1].row_id, "r2")
            self.assertEqual(rows[1].source_row_id, "r1")
        finally:
            jsonl_path.unlink()

    def test_duplicate_row_id_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "row_id": "r1",
                        "model_backbone": "m1",
                        "property_id": "p1",
                        "context_id": "c1",
                        "seed": 1,
                        "stage": "native",
                        "image_path": "/img1.png",
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "row_id": "r1",
                        "model_backbone": "m2",
                        "property_id": "p2",
                        "context_id": "c2",
                        "seed": 2,
                        "stage": "native",
                        "image_path": "/img2.png",
                    }
                )
                + "\n"
            )
            jsonl_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_manifest_jsonl(jsonl_path)
            self.assertIn("duplicate row_id", str(ctx.exception))
        finally:
            jsonl_path.unlink()

    def test_missing_required_field(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "row_id": "r1",
                        "model_backbone": "m1",
                        # missing property_id
                        "context_id": "c1",
                        "seed": 1,
                        "stage": "native",
                        "image_path": "/img.png",
                    }
                )
                + "\n"
            )
            jsonl_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_manifest_jsonl(jsonl_path)
            self.assertIn("missing required fields", str(ctx.exception))
        finally:
            jsonl_path.unlink()

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("not valid json\n")
            jsonl_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_manifest_jsonl(jsonl_path)
            self.assertIn("invalid JSON", str(ctx.exception))
        finally:
            jsonl_path.unlink()

    def test_manifest_row_must_be_object(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("[]\n")
            jsonl_path = Path(f.name)

        try:
            with self.assertRaisesRegex(ValueError, "must be a JSON object"):
                load_manifest_jsonl(jsonl_path)
        finally:
            jsonl_path.unlink()

    def test_manifest_rejects_empty_identifier(self):
        row = {
            "row_id": " ",
            "model_backbone": "m1",
            "property_id": "p1",
            "context_id": "c1",
            "seed": 1,
            "stage": "native",
            "image_path": "/img.png",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(row) + "\n")
            jsonl_path = Path(f.name)

        try:
            with self.assertRaisesRegex(ValueError, "row_id.*nonblank string"):
                load_manifest_jsonl(jsonl_path)
        finally:
            jsonl_path.unlink()

    def test_manifest_rejects_unknown_stage(self):
        row = {
            "row_id": "r1",
            "model_backbone": "m1",
            "property_id": "p1",
            "context_id": "c1",
            "seed": 1,
            "stage": "naitve",
            "image_path": "/img.png",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(row) + "\n")
            jsonl_path = Path(f.name)

        try:
            with self.assertRaisesRegex(ValueError, "unsupported stage"):
                load_manifest_jsonl(jsonl_path)
        finally:
            jsonl_path.unlink()


class TestCalculateValidity(unittest.TestCase):
    """Test validity calculation and gate application"""

    def test_basic_validity(self):
        manifest = [
            ManifestRow("r1", "m1", "p1", "c1", 1, "native", "/img1.png", None),
            ManifestRow("r2", "m1", "p1", "c1", 2, "native", "/img2.png", None),
            ManifestRow("r3", "m1", "p1", "c1", 3, "native", "/img3.png", None),
        ]
        scores = {
            "r1": ScoringRow("r1", True, True, False, ""),
            "r2": ScoringRow("r2", True, False, False, ""),
            "r3": ScoringRow("r3", False, False, False, ""),
        }

        stats = calculate_validity(manifest, scores, gate_threshold=0.90)

        self.assertEqual(stats.n_total, 3)
        self.assertEqual(stats.n_valid, 2)
        self.assertAlmostEqual(stats.rate, 2 / 3)
        # 2/3 < 0.90, should not pass gate
        self.assertFalse(stats.passes_gate)

    def test_ninety_percent_boundary(self):
        # test exact 90% boundary
        manifest = [
            ManifestRow(f"r{i}", "m1", "p1", "c1", i, "native", f"/img{i}.png", None)
            for i in range(10)
        ]
        scores = {
            f"r{i}": ScoringRow(f"r{i}", i < 9, True, False, "")  # 9 out of 10 valid
            for i in range(10)
        }

        stats = calculate_validity(manifest, scores, gate_threshold=0.90)

        self.assertEqual(stats.n_total, 10)
        self.assertEqual(stats.n_valid, 9)
        self.assertEqual(stats.rate, 0.90)
        # exactly 90% should pass
        self.assertTrue(stats.passes_gate)

        # CI lower bound should be less than observed rate
        self.assertLess(stats.ci_lower, stats.rate)

    def test_eighty_nine_percent_fails_gate(self):
        # 89/100 should not pass 90% gate
        manifest = [
            ManifestRow(f"r{i}", "m1", "p1", "c1", i, "native", f"/img{i}.png", None)
            for i in range(100)
        ]
        scores = {
            f"r{i}": ScoringRow(f"r{i}", i < 89, True, False, "") for i in range(100)
        }

        stats = calculate_validity(manifest, scores, gate_threshold=0.90)

        self.assertEqual(stats.n_valid, 89)
        self.assertEqual(stats.rate, 0.89)
        # 89% < 90%, should fail
        self.assertFalse(stats.passes_gate)

    def test_missing_score(self):
        manifest = [
            ManifestRow("r1", "m1", "p1", "c1", 1, "native", "/img1.png", None),
        ]
        scores = {}

        with self.assertRaises(ValueError) as ctx:
            calculate_validity(manifest, scores)
        self.assertIn("has no corresponding score", str(ctx.exception))

    def test_empty_manifest(self):
        stats = calculate_validity([], {})
        self.assertEqual(stats.n_total, 0)
        self.assertEqual(stats.n_valid, 0)
        self.assertEqual(stats.rate, 0.0)
        self.assertFalse(stats.passes_gate)


class TestPerModelValidity(unittest.TestCase):
    """Test per-model validity calculation"""

    def test_multiple_models(self):
        manifest = [
            ManifestRow("r1", "model_a", "p1", "c1", 1, "native", "/img1.png", None),
            ManifestRow("r2", "model_a", "p1", "c1", 2, "native", "/img2.png", None),
            ManifestRow("r3", "model_b", "p1", "c1", 3, "native", "/img3.png", None),
            ManifestRow("r4", "model_b", "p1", "c1", 4, "native", "/img4.png", None),
        ]
        scores = {
            "r1": ScoringRow("r1", True, True, False, ""),
            "r2": ScoringRow("r2", False, False, False, ""),
            "r3": ScoringRow("r3", True, True, False, ""),
            "r4": ScoringRow("r4", True, True, False, ""),
        }

        results = calculate_per_model_validity(manifest, scores)

        self.assertIn("model_a", results)
        self.assertIn("model_b", results)

        # model_a: 1/2 = 50%
        self.assertEqual(results["model_a"].n_valid, 1)
        self.assertEqual(results["model_a"].n_total, 2)
        self.assertFalse(results["model_a"].passes_gate)

        # model_b: 2/2 = 100%
        self.assertEqual(results["model_b"].n_valid, 2)
        self.assertEqual(results["model_b"].n_total, 2)
        self.assertTrue(results["model_b"].passes_gate)


class TestPerPropertyValidity(unittest.TestCase):
    """Test per-property validity calculation"""

    def test_multiple_properties(self):
        manifest = [
            ManifestRow("r1", "m1", "red_sphere", "c1", 1, "native", "/img1.png", None),
            ManifestRow("r2", "m1", "red_sphere", "c1", 2, "native", "/img2.png", None),
            ManifestRow("r3", "m1", "blue_cube", "c1", 3, "native", "/img3.png", None),
        ]
        scores = {
            "r1": ScoringRow("r1", True, True, False, ""),
            "r2": ScoringRow("r2", True, True, False, ""),
            "r3": ScoringRow("r3", False, False, False, ""),
        }

        results = calculate_per_property_validity(manifest, scores)

        self.assertIn("red_sphere", results)
        self.assertIn("blue_cube", results)

        # red_sphere: 2/2 = 100%
        self.assertEqual(results["red_sphere"].n_valid, 2)
        self.assertEqual(results["red_sphere"].n_total, 2)
        self.assertTrue(results["red_sphere"].passes_gate)

        # blue_cube: 0/1 = 0%
        self.assertEqual(results["blue_cube"].n_valid, 0)
        self.assertEqual(results["blue_cube"].n_total, 1)
        self.assertFalse(results["blue_cube"].passes_gate)


class TestCalculateEligibility(unittest.TestCase):
    """Test edit eligibility calculation"""

    def test_basic_eligibility(self):
        manifest = [
            ManifestRow("e1", "m1", "p1", "c1", 1, "edit", "/img1.png", "src1"),
            ManifestRow("e2", "m1", "p1", "c1", 2, "edit", "/img2.png", "src2"),
            ManifestRow("e3", "m1", "p1", "c1", 3, "edit", "/img3.png", "src1"),
        ]
        scores = {
            "src1": ScoringRow("src1", True, True, False, ""),  # eligible
            "src2": ScoringRow("src2", False, False, False, ""),  # not eligible
        }

        eligibility = calculate_eligibility(manifest, scores)

        self.assertEqual(eligibility.n_attempted, 3)
        self.assertEqual(eligibility.n_eligible, 2)  # e1 and e3 both reference src1
        self.assertAlmostEqual(eligibility.rate, 2 / 3)

    def test_missing_source_row(self):
        manifest = [
            ManifestRow("e1", "m1", "p1", "c1", 1, "edit", "/img1.png", "src1"),
            ManifestRow("e2", "m1", "p1", "c1", 2, "edit", "/img2.png", "src_missing"),
        ]
        scores = {
            "src1": ScoringRow("src1", True, True, False, ""),
        }

        with self.assertRaisesRegex(ValueError, "source without a score"):
            calculate_eligibility(manifest, scores)

    def test_no_source_row_id(self):
        manifest = [
            ManifestRow("e1", "m1", "p1", "c1", 1, "edit", "/img1.png", None),
        ]

        with self.assertRaisesRegex(ValueError, "has no source_row_id"):
            calculate_eligibility(manifest, {})

    def test_non_edit_row_rejected(self):
        manifest = [
            ManifestRow("r1", "m1", "p1", "c1", 1, "native", "/img1.png", None),
        ]

        with self.assertRaisesRegex(ValueError, "must have stage edit"):
            calculate_eligibility(manifest, {})

    def test_empty_manifest_eligibility(self):
        eligibility = calculate_eligibility([], {})
        self.assertEqual(eligibility.n_attempted, 0)
        self.assertEqual(eligibility.n_eligible, 0)
        self.assertEqual(eligibility.rate, 0.0)


class TestEndToEnd(unittest.TestCase):
    """Test full analysis pipeline"""

    def test_full_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # create manifest
            manifest_path = tmpdir / "manifest.jsonl"
            with manifest_path.open("w") as f:
                for i in range(10):
                    f.write(
                        json.dumps(
                            {
                                "row_id": f"r{i}",
                                "model_backbone": "sd35_large_turbo",
                                "property_id": "red_sphere",
                                "context_id": "c1",
                                "seed": i,
                                "stage": "native",
                                "image_path": f"/img{i}.png",
                            }
                        )
                        + "\n"
                    )

            # create scores (9/10 valid for exactly 90%)
            scores_path = tmpdir / "scores.csv"
            with scores_path.open("w") as f:
                f.write("row_id,target_present,target_visible,prompt_ambiguous,notes\n")
                for i in range(10):
                    present = "true" if i < 9 else "false"
                    f.write(f"r{i},{present},true,false,\n")

            # run analysis
            results = analyze_image_validity(manifest_path, scores_path)

            # check overall
            self.assertEqual(results["overall"]["n_total"], 10)
            self.assertEqual(results["overall"]["n_valid"], 9)
            self.assertEqual(results["overall"]["rate"], 0.90)
            self.assertTrue(results["overall"]["passes_gate"])

            # check CI bounds exist and are valid
            self.assertGreater(results["overall"]["ci_lower"], 0.0)
            self.assertLess(results["overall"]["ci_upper"], 1.0)
            self.assertLess(results["overall"]["ci_lower"], results["overall"]["rate"])

            # check per-model
            self.assertIn("sd35_large_turbo", results["per_model"])

            # check per-property
            self.assertIn("red_sphere", results["per_property"])

            # write outputs and verify they exist
            json_path = tmpdir / "results.json"
            csv_path = tmpdir / "summary.csv"

            write_results_json(results, json_path)
            write_summary_csv(results, csv_path)

            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())

            # verify JSON is valid
            with json_path.open() as f:
                loaded = json.load(f)
                self.assertEqual(loaded["overall"]["n_valid"], 9)

    def test_with_edits_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # create manifest with native and edit stages
            manifest_path = tmpdir / "manifest.jsonl"
            with manifest_path.open("w") as f:
                # 2 native rows
                f.write(
                    json.dumps(
                        {
                            "row_id": "n1",
                            "model_backbone": "m1",
                            "property_id": "p1",
                            "context_id": "c1",
                            "seed": 1,
                            "stage": "native",
                            "image_path": "/n1.png",
                        }
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "row_id": "n2",
                            "model_backbone": "m1",
                            "property_id": "p1",
                            "context_id": "c1",
                            "seed": 2,
                            "stage": "native",
                            "image_path": "/n2.png",
                        }
                    )
                    + "\n"
                )
                # 2 edit rows
                f.write(
                    json.dumps(
                        {
                            "row_id": "e1",
                            "model_backbone": "m1",
                            "property_id": "p1",
                            "context_id": "c1",
                            "seed": 3,
                            "stage": "edit",
                            "image_path": "/e1.png",
                            "source_row_id": "n1",
                        }
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "row_id": "e2",
                            "model_backbone": "m1",
                            "property_id": "p1",
                            "context_id": "c1",
                            "seed": 4,
                            "stage": "edit",
                            "image_path": "/e2.png",
                            "source_row_id": "n2",
                        }
                    )
                    + "\n"
                )

            # create scores with ambiguity and visibility variation
            scores_path = tmpdir / "scores.csv"
            with scores_path.open("w") as f:
                f.write("row_id,target_present,target_visible,prompt_ambiguous,notes\n")
                f.write("n1,true,true,false,\n")  # eligible source
                f.write(
                    "n2,false,false,true,\n"
                )  # not eligible (not present), ambiguous
                f.write("e1,true,false,false,present but not visible\n")
                f.write("e2,false,false,false,\n")

            results = analyze_image_validity(manifest_path, scores_path)

            # check overall (native-only: 2 native rows, 1 valid)
            self.assertEqual(results["overall"]["n_total"], 2)
            self.assertEqual(results["overall"]["n_valid"], 1)  # n1 only

            # check ambiguity
            self.assertEqual(results["ambiguity"]["n_total"], 4)
            self.assertEqual(results["ambiguity"]["n_ambiguous"], 1)  # n2

            # check visibility (across all rows: n1 and e1 both present)
            self.assertEqual(results["visibility"]["n_total"], 2)  # n1 and e1 present
            self.assertEqual(
                results["visibility"]["n_invisible"], 1
            )  # e1 present but not visible

            # check eligibility
            self.assertIsNotNone(results["eligibility"])
            self.assertEqual(results["eligibility"]["n_attempted"], 2)
            self.assertEqual(
                results["eligibility"]["n_eligible"], 1
            )  # only e1, since n1 is present

    def test_missing_scores_rejected(self):
        """Manifest row without a score should raise clear error"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # create manifest with 3 rows (2 native, 1 edit)
            manifest_path = tmpdir / "manifest.jsonl"
            with manifest_path.open("w") as f:
                f.write(
                    json.dumps(
                        {
                            "row_id": "r1",
                            "model_backbone": "m1",
                            "property_id": "p1",
                            "context_id": "c1",
                            "seed": 1,
                            "stage": "native",
                            "image_path": "/img1.png",
                        }
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "row_id": "r2",
                            "model_backbone": "m1",
                            "property_id": "p1",
                            "context_id": "c1",
                            "seed": 2,
                            "stage": "native",
                            "image_path": "/img2.png",
                        }
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "row_id": "e1",
                            "model_backbone": "m1",
                            "property_id": "p1",
                            "context_id": "c1",
                            "seed": 3,
                            "stage": "edit",
                            "image_path": "/e1.png",
                            "source_row_id": "r1",
                        }
                    )
                    + "\n"
                )

            # create scores missing the edit row
            scores_path = tmpdir / "scores.csv"
            with scores_path.open("w") as f:
                f.write("row_id,target_present,target_visible,prompt_ambiguous,notes\n")
                f.write("r1,true,true,false,\n")
                f.write("r2,true,true,false,\n")
                # e1 is missing

            with self.assertRaises(ValueError) as ctx:
                analyze_image_validity(manifest_path, scores_path)
            self.assertIn("without scores", str(ctx.exception))
            self.assertIn("e1", str(ctx.exception))

    def test_unknown_scores_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # create manifest with 2 rows
            manifest_path = tmpdir / "manifest.jsonl"
            with manifest_path.open("w") as f:
                f.write(
                    json.dumps(
                        {
                            "row_id": "r1",
                            "model_backbone": "m1",
                            "property_id": "p1",
                            "context_id": "c1",
                            "seed": 1,
                            "stage": "native",
                            "image_path": "/img1.png",
                        }
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "row_id": "r2",
                            "model_backbone": "m1",
                            "property_id": "p1",
                            "context_id": "c1",
                            "seed": 2,
                            "stage": "native",
                            "image_path": "/img2.png",
                        }
                    )
                    + "\n"
                )

            # create scores with extra unknown row
            scores_path = tmpdir / "scores.csv"
            with scores_path.open("w") as f:
                f.write("row_id,target_present,target_visible,prompt_ambiguous,notes\n")
                f.write("r1,true,true,false,\n")
                f.write("r2,true,true,false,\n")
                f.write("r999,true,true,false,stale/mistyped row\n")

            with self.assertRaises(ValueError) as ctx:
                analyze_image_validity(manifest_path, scores_path)
            self.assertIn("not in manifest", str(ctx.exception))
            self.assertIn("r999", str(ctx.exception))

    def test_native_vs_edit_validity_separation(self):
        """Native gate should only apply to native-stage rows, not edits"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # 10 native rows with 90% validity, 10 edit rows with 0% validity
            manifest_path = tmpdir / "manifest.jsonl"
            with manifest_path.open("w") as f:
                for i in range(10):
                    f.write(
                        json.dumps(
                            {
                                "row_id": f"n{i}",
                                "model_backbone": "m1",
                                "property_id": "p1",
                                "context_id": "c1",
                                "seed": i,
                                "stage": "native",
                                "image_path": f"/n{i}.png",
                            }
                        )
                        + "\n"
                    )
                for i in range(10):
                    f.write(
                        json.dumps(
                            {
                                "row_id": f"e{i}",
                                "model_backbone": "m1",
                                "property_id": "p1",
                                "context_id": "c1",
                                "seed": i + 100,
                                "stage": "edit",
                                "image_path": f"/e{i}.png",
                                "source_row_id": "n0",
                            }
                        )
                        + "\n"
                    )

            # native: 9/10 valid (90%), edit: 0/10 valid
            scores_path = tmpdir / "scores.csv"
            with scores_path.open("w") as f:
                f.write("row_id,target_present,target_visible,prompt_ambiguous,notes\n")
                for i in range(10):
                    present = "true" if i < 9 else "false"
                    f.write(f"n{i},{present},true,false,\n")
                for i in range(10):
                    f.write(f"e{i},false,false,false,\n")

            results = analyze_image_validity(manifest_path, scores_path)

            # overall should be 9/10 (native only)
            self.assertEqual(results["overall"]["n_total"], 10)
            self.assertEqual(results["overall"]["n_valid"], 9)
            self.assertEqual(results["overall"]["rate"], 0.90)
            self.assertTrue(results["overall"]["passes_gate"])

            # per-model should also be native-only
            self.assertEqual(results["per_model"]["m1"]["n_total"], 10)
            self.assertEqual(results["per_model"]["m1"]["n_valid"], 9)
            self.assertTrue(results["per_model"]["m1"]["passes_gate"])

    def test_edit_requires_source_row_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.jsonl"
            rows = [
                {
                    "row_id": "n1",
                    "model_backbone": "m1",
                    "property_id": "p1",
                    "context_id": "c1",
                    "seed": 1,
                    "stage": "native",
                    "image_path": "/n1.png",
                },
                {
                    "row_id": "e1",
                    "model_backbone": "m1",
                    "property_id": "p1",
                    "context_id": "c1",
                    "seed": 2,
                    "stage": "edit",
                    "image_path": "/e1.png",
                },
            ]
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            scores_path = root / "scores.csv"
            scores_path.write_text(
                "row_id,target_present,target_visible,prompt_ambiguous,notes\n"
                "n1,true,true,false,\n"
                "e1,true,true,false,\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "has no source_row_id"):
                analyze_image_validity(manifest_path, scores_path)

    def test_edit_source_must_be_native(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.jsonl"
            rows = [
                {
                    "row_id": "n1",
                    "model_backbone": "m1",
                    "property_id": "p1",
                    "context_id": "c1",
                    "seed": 1,
                    "stage": "native",
                    "image_path": "/n1.png",
                },
                {
                    "row_id": "e1",
                    "model_backbone": "m1",
                    "property_id": "p1",
                    "context_id": "c1",
                    "seed": 2,
                    "stage": "edit",
                    "image_path": "/e1.png",
                    "source_row_id": "n1",
                },
                {
                    "row_id": "e2",
                    "model_backbone": "m1",
                    "property_id": "p1",
                    "context_id": "c1",
                    "seed": 3,
                    "stage": "edit",
                    "image_path": "/e2.png",
                    "source_row_id": "e1",
                },
            ]
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            scores_path = root / "scores.csv"
            scores_path.write_text(
                "row_id,target_present,target_visible,prompt_ambiguous,notes\n"
                "n1,true,true,false,\n"
                "e1,true,true,false,\n"
                "e2,true,true,false,\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must reference a native row"):
                analyze_image_validity(manifest_path, scores_path)


class TestCLI(unittest.TestCase):
    """Test the image validity command-line boundary"""

    def test_cli_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "row_id": "n1",
                        "model_backbone": "m1",
                        "property_id": "p1",
                        "context_id": "c1",
                        "seed": 1,
                        "stage": "native",
                        "image_path": "/n1.png",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            scores_path = root / "scores.csv"
            scores_path.write_text(
                "row_id,target_present,target_visible,prompt_ambiguous,notes\n"
                "n1,true,true,false,\n",
                encoding="utf-8",
            )
            json_path = root / "results.json"
            csv_path = root / "summary.csv"
            argv = [
                "run_image_validity_analysis",
                "--manifest",
                str(manifest_path),
                "--scores",
                str(scores_path),
                "--output-json",
                str(json_path),
                "--output-csv",
                str(csv_path),
            ]

            stdout = StringIO()
            with patch("sys.argv", argv), redirect_stdout(stdout):
                exit_code = validity_main()

            self.assertEqual(exit_code, 0)
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            self.assertIn("passes 90% gate: True", stdout.getvalue())

    def test_cli_returns_error_for_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            argv = [
                "run_image_validity_analysis",
                "--manifest",
                str(root / "missing.jsonl"),
                "--scores",
                str(root / "missing.csv"),
                "--output-json",
                str(root / "results.json"),
                "--output-csv",
                str(root / "summary.csv"),
            ]

            stderr = StringIO()
            with patch("sys.argv", argv), redirect_stderr(stderr):
                exit_code = validity_main()

            self.assertEqual(exit_code, 1)
            self.assertIn("error:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
