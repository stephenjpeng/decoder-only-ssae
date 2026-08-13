"""Test interaction budget analysis."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from analysis.interaction_budget import (
    bootstrap_paired_diff,
    compute_gap_closure,
    win_rate,
)
from baselines.run_baselines import fit_ridge, predict_linear

try:
    from baselines.property_design import pairwise_design
    from evaluation.embedding_metrics import embedding_metrics

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


class TestInteractionBudget(unittest.TestCase):
    def test_fvu_formula(self) -> None:
        """Verify embedding_metrics FVU equals mse_per_sample.mean() / target.var(dim=0, unbiased=False).sum()."""
        if not DEPS_AVAILABLE:
            self.skipTest("baselines.property_design or evaluation.embedding_metrics not available")

        pred = torch.randn(100, 50)
        target = torch.randn(100, 50)

        met = embedding_metrics(pred, target)
        # mse_per_sample is the sum of squared errors per sample (summed across dimensions)
        mse_per_sample = ((target - pred) ** 2).sum(dim=1)
        target_var_sum = target.var(dim=0, unbiased=False).sum()
        expected_fvu = mse_per_sample.mean() / target_var_sum

        self.assertAlmostEqual(met.fvu, expected_fvu.item(), places=5)

    def test_additive_synthetic_no_pairwise_advantage(self) -> None:
        """X = M @ W_true. Plain ridge FVU <= pairwise_ridge_fvu * 1.05."""
        if not DEPS_AVAILABLE:
            self.skipTest("baselines.property_design or evaluation.embedding_metrics not available")

        n_train = 200
        n_holdout = 50
        n_props = 10
        d = 30

        M_tr = torch.randn(n_train, n_props)
        M_ho = torch.randn(n_holdout, n_props)
        W_true = torch.randn(n_props, d)
        X_tr = M_tr @ W_true
        X_ho = M_ho @ W_true

        # plain ridge
        W_plain = fit_ridge(M_tr, X_tr, lam=1e-3)
        pred_plain = predict_linear(M_ho, W_plain)
        met_plain = embedding_metrics(pred_plain, X_ho)

        # pairwise ridge (dummy columns for interactions)
        from baselines.property_design import InteractionColumn

        # create fake interaction columns
        columns = [
            InteractionColumn(
                left_pid=i,
                right_pid=i + 1,
                left_property=f"prop_{i}",
                right_property=f"prop_{i+1}",
                left_category=f"cat_{i % 2}",
                right_category=f"cat_{(i+1) % 2}",
            )
            for i in range(5)
        ]
        D_tr = pairwise_design(M_tr, columns)
        D_ho = pairwise_design(M_ho, columns)
        W_pw = fit_ridge(D_tr, X_tr, lam=1e-3)
        pred_pw = predict_linear(D_ho, W_pw)
        met_pw = embedding_metrics(pred_pw, X_ho)

        # plain should be as good or better (no pairwise advantage needed)
        self.assertLessEqual(met_plain.fvu, met_pw.fvu * 1.05)

    def test_pairwise_synthetic_lower_mse(self) -> None:
        """X includes interaction term. Pairwise ridge achieves lower holdout MSE."""
        if not DEPS_AVAILABLE:
            self.skipTest("baselines.property_design or evaluation.embedding_metrics not available")

        n_train = 200
        n_holdout = 50
        n_props = 10
        d = 30

        M_tr = torch.randn(n_train, n_props)
        M_ho = torch.randn(n_holdout, n_props)

        # create X with interaction: X = M @ W + (M[:, 0] * M[:, 1]).unsqueeze(1) @ W_int
        W_base = torch.randn(n_props, d)
        W_int = torch.randn(1, d)
        X_tr = M_tr @ W_base + (M_tr[:, 0] * M_tr[:, 1]).unsqueeze(1) @ W_int
        X_ho = M_ho @ W_base + (M_ho[:, 0] * M_ho[:, 1]).unsqueeze(1) @ W_int

        # plain ridge
        W_plain = fit_ridge(M_tr, X_tr, lam=1e-3)
        pred_plain = predict_linear(M_ho, W_plain)
        met_plain = embedding_metrics(pred_plain, X_ho)

        # pairwise ridge
        from baselines.property_design import InteractionColumn

        columns = [
            InteractionColumn(
                left_pid=0,
                right_pid=1,
                left_property="prop_0",
                right_property="prop_1",
                left_category="cat_0",
                right_category="cat_1",
            )
        ]
        D_tr = pairwise_design(M_tr, columns)
        D_ho = pairwise_design(M_ho, columns)
        W_pw = fit_ridge(D_tr, X_tr, lam=1e-3)
        pred_pw = predict_linear(D_ho, W_pw)
        met_pw = embedding_metrics(pred_pw, X_ho)

        # pairwise should achieve lower MSE
        self.assertLess(met_pw.mse_mean, met_plain.mse_mean)

    def test_lambda_tie_breaking_prefers_larger(self) -> None:
        """Two lambdas with same FVU -> larger selected."""
        # simulate validation loop
        val_grid = [
            {"lambda": 1e-3, "fvu": 0.5},
            {"lambda": 1e-2, "fvu": 0.5},
            {"lambda": 1e-1, "fvu": 0.6},
        ]

        best_lambda = None
        best_fvu = float("inf")
        for entry in val_grid:
            lam = entry["lambda"]
            fvu = entry["fvu"]
            if fvu < best_fvu or (fvu == best_fvu and lam > best_lambda):
                best_fvu = fvu
                best_lambda = lam

        self.assertEqual(best_lambda, 1e-2)

    def test_gap_closure_formula(self) -> None:
        """compute_gap_closure(0.8, 0.6, 0.4) -> pairwise_improvement=0.25, gap_closed=0.5."""
        gap = compute_gap_closure(0.8, 0.6, 0.4)
        self.assertAlmostEqual(gap["pairwise_improvement"], 0.25, places=5)
        self.assertAlmostEqual(gap["h_improvement"], 0.5, places=5)
        self.assertAlmostEqual(gap["gap_closed"], 0.5, places=5)

    def test_gap_closure_null_when_denom_zero(self) -> None:
        """compute_gap_closure(0.8, 0.6, 0.8) -> gap_closed is None."""
        gap = compute_gap_closure(0.8, 0.6, 0.8)
        self.assertIsNone(gap["gap_closed"])

    def test_win_rate(self) -> None:
        """win_rate([0.1, 0.2, 0.3], [0.2, 0.1, 0.4]) -> 2/3."""
        wr = win_rate([0.1, 0.2, 0.3], [0.2, 0.1, 0.4])
        self.assertAlmostEqual(wr, 2.0 / 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
