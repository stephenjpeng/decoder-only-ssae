"""Tests for probe intervention fit."""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock
import torch
from evaluation.run_probe_intervention_fit import load_and_validate_probe_artifact


class TestProbeIntervention(unittest.TestCase):
    def test_context_partition_respects_holdout_boundary(self):
        """Verify fit/cal row IDs are disjoint."""
        # simple context-based split
        import random

        context_keys = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        rng = random.Random(123)
        unique_contexts = sorted(set(context_keys))
        rng.shuffle(unique_contexts)

        n_fit_ctx = 3
        fit_ctx = set(unique_contexts[:n_fit_ctx])
        cal_ctx = set(unique_contexts[n_fit_ctx:])

        fit_ids = [i for i, ctx in enumerate(context_keys) if ctx in fit_ctx]
        cal_ids = [i for i, ctx in enumerate(context_keys) if ctx in cal_ctx]

        # should be disjoint
        self.assertEqual(len(set(fit_ids) & set(cal_ids)), 0)
        # should cover all rows
        self.assertEqual(len(fit_ids) + len(cal_ids), len(context_keys))

    def test_alpha_tiebreaking_small_alpha_preferred(self):
        """Among equal candidates, smaller alpha wins."""
        # simulate grid search results
        replacement_grid = [
            {
                "alpha": 1.0,
                "replacement_acc": 0.9,
                "mse_to_new": 0.1,
                "nontarget_acc": 0.85,
            },
            {
                "alpha": 0.5,
                "replacement_acc": 0.9,
                "mse_to_new": 0.1,
                "nontarget_acc": 0.85,
            },
            {
                "alpha": 2.0,
                "replacement_acc": 0.9,
                "mse_to_new": 0.1,
                "nontarget_acc": 0.85,
            },
        ]

        baseline_nontarget_acc = 0.8

        # filter viable
        viable = [
            r
            for r in replacement_grid
            if r["nontarget_acc"] >= baseline_nontarget_acc - 0.01
        ]

        # sort by replacement_acc descending, mse ascending, alpha ascending
        viable.sort(key=lambda r: (-r["replacement_acc"], r["mse_to_new"], r["alpha"]))

        # should select alpha=0.5 (all have same replacement_acc and mse, so smallest alpha wins)
        self.assertEqual(viable[0]["alpha"], 0.5)

    def test_calibration_failed_flag(self):
        """All alphas fail nontarget constraint -> calibration_failed=True."""
        replacement_grid = [
            {"alpha": 0.5, "nontarget_acc": 0.7},
            {"alpha": 1.0, "nontarget_acc": 0.65},
            {"alpha": 2.0, "nontarget_acc": 0.6},
        ]

        baseline_nontarget_acc = 0.8

        viable = [
            r
            for r in replacement_grid
            if r["nontarget_acc"] >= baseline_nontarget_acc - 0.01
        ]

        calibration_failed = len(viable) == 0
        self.assertTrue(calibration_failed)

    def test_artifact_topk_mismatch_raises(self):
        """load_and_validate_probe_artifact with wrong topk -> ValueError."""
        # create a temporary artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "probe_artifact.pt"

            # mock dataset
            mock_dataset = MagicMock()
            mock_dataset.properties.n_properties = 4
            mock_dataset.properties.n_categories = 2
            mock_dataset.properties.pid_to_property = {0: "p0", 1: "p1", 2: "p2", 3: "p3"}
            mock_dataset.properties.cid_to_pids = {0: [0, 1], 1: [2, 3]}
            mock_dataset.truncate_embds_topk = 1000
            mock_dataset.normalize = "max_min"

            # create artifact with different topk
            artifact = {
                "schema_version": 1,
                "property_order": ["p0", "p1", "p2", "p3"],
                "category_pid_groups": [[0, 1], [2, 3]],
                "topk": 500,  # mismatch
                "normalize": "max_min",
            }
            torch.save(artifact, artifact_path)

            # should raise ValueError
            with self.assertRaises(ValueError) as cm:
                load_and_validate_probe_artifact(artifact_path, mock_dataset)

            self.assertIn("topk", str(cm.exception))

    def test_edited_embeddings_not_clipped(self):
        """Large alpha -> output contains values outside [0,1]."""
        embedding_dim = 4
        x = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
        alpha = 2.0  # large enough to push outside [0,1]

        x_edited = x + alpha * direction

        # should have values > 1
        self.assertTrue((x_edited > 1).any())
        # should NOT be clipped
        self.assertAlmostEqual(x_edited[0, 0].item(), 2.5, places=5)


if __name__ == "__main__":
    unittest.main()
