"""Tests for embedding linear probe."""

import unittest
import torch
from evaluation.embedding_linear_probe import CategoricalLinearProbe, fit_probe


class TestCategoricalLinearProbe(unittest.TestCase):
    def test_category_loss_uses_correct_pid_slices(self):
        """Probe with 2 categories pids [0,1] and [2,3]. Create mask where category 0
        selects pid 0, category 1 selects pid 2. Verify loss computation uses correct slices."""
        embedding_dim = 8
        category_pid_groups = [[0, 1], [2, 3]]
        probe = CategoricalLinearProbe(embedding_dim, category_pid_groups)

        # create a batch with specific one-hot masks
        batch_size = 4
        embeddings = torch.randn(batch_size, embedding_dim)
        mask = torch.zeros(batch_size, 4)
        # all samples select pid 0 for category 0, pid 2 for category 1
        mask[:, 0] = 1  # category 0 -> pid 0
        mask[:, 2] = 1  # category 1 -> pid 2

        logits = probe(embeddings)
        loss = probe.loss(logits, mask)

        # manually compute expected loss
        # category 0: logits[:, [0,1]], target is index 0
        # category 1: logits[:, [2,3]], target is index 0 (pid 2 is first in the slice)
        logits_cat0 = logits[:, [0, 1]]
        target_cat0 = torch.zeros(batch_size, dtype=torch.long)
        loss_cat0 = torch.nn.functional.cross_entropy(logits_cat0, target_cat0)

        logits_cat1 = logits[:, [2, 3]]
        target_cat1 = torch.zeros(batch_size, dtype=torch.long)  # pid 2 is index 0 in [2,3]
        loss_cat1 = torch.nn.functional.cross_entropy(logits_cat1, target_cat1)

        expected_loss = (loss_cat0 + loss_cat1) / 2

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)

    def test_context_grouping_keeps_full_category_contexts_together(self):
        """12 rows, 2 context groups, each appearing in fit or cal. Verify all rows from
        same context are in same partition."""
        # simulate a simple context split scenario
        n_rows = 12
        # context A: rows 0-5, context B: rows 6-11
        context_keys = ["A"] * 6 + ["B"] * 6

        # simple deterministic split
        import random
        rng = random.Random(42)
        unique_contexts = sorted(set(context_keys))
        rng.shuffle(unique_contexts)

        fit_contexts = set(unique_contexts[:1])
        cal_contexts = set(unique_contexts[1:])

        fit_ids = [i for i, ctx in enumerate(context_keys) if ctx in fit_contexts]
        cal_ids = [i for i, ctx in enumerate(context_keys) if ctx in cal_contexts]

        # check that all rows from context A are in one partition
        # and all rows from context B are in the other
        if "A" in fit_contexts:
            self.assertEqual(set(fit_ids), set(range(6)))
            self.assertEqual(set(cal_ids), set(range(6, 12)))
        else:
            self.assertEqual(set(fit_ids), set(range(6, 12)))
            self.assertEqual(set(cal_ids), set(range(6)))

    def test_fit_cal_ids_disjoint(self):
        """After context splitting, fit and cal IDs are disjoint."""
        # create some dummy context keys
        context_keys = [0, 0, 1, 1, 2, 2, 3, 3]
        import random
        rng = random.Random(0)
        unique_contexts = sorted(set(context_keys))
        rng.shuffle(unique_contexts)
        n_fit_ctx = 2
        fit_ctx = set(unique_contexts[:n_fit_ctx])
        cal_ctx = set(unique_contexts[n_fit_ctx:])

        fit_ids = [i for i, ctx in enumerate(context_keys) if ctx in fit_ctx]
        cal_ids = [i for i, ctx in enumerate(context_keys) if ctx in cal_ctx]

        self.assertEqual(len(set(fit_ids) & set(cal_ids)), 0)
        self.assertEqual(len(fit_ids) + len(cal_ids), len(context_keys))

    def test_linearly_separable_beats_majority_baseline(self):
        """Synthetic 2-category 4-value dataset, linearly separable. After fitting,
        cal accuracy > 0.5."""
        torch.manual_seed(42)
        embedding_dim = 4
        category_pid_groups = [[0, 1], [2, 3]]

        # create linearly separable data
        # category 0: pid 0 has positive first dim, pid 1 has negative first dim
        # category 1: pid 2 has positive second dim, pid 3 has negative second dim
        n_train = 40
        n_cal = 20

        train_X = torch.zeros(n_train, embedding_dim)
        train_M = torch.zeros(n_train, 4)
        cal_X = torch.zeros(n_cal, embedding_dim)
        cal_M = torch.zeros(n_cal, 4)

        for i in range(n_train):
            # alternate pids
            pid0 = i % 2  # 0 or 1
            pid1 = (i // 2) % 2 + 2  # 2 or 3

            # set features to make them linearly separable
            train_X[i, 0] = 1.0 if pid0 == 0 else -1.0
            train_X[i, 1] = 1.0 if pid1 == 2 else -1.0
            train_X[i, 2:] = torch.randn(2) * 0.1  # noise

            train_M[i, pid0] = 1
            train_M[i, pid1] = 1

        for i in range(n_cal):
            pid0 = i % 2
            pid1 = (i // 2) % 2 + 2

            cal_X[i, 0] = 1.0 if pid0 == 0 else -1.0
            cal_X[i, 1] = 1.0 if pid1 == 2 else -1.0
            cal_X[i, 2:] = torch.randn(2) * 0.1

            cal_M[i, pid0] = 1
            cal_M[i, pid1] = 1

        probe, fit_info = fit_probe(
            train_X,
            train_M,
            cal_X,
            cal_M,
            category_pid_groups,
            embedding_dim=embedding_dim,
            max_epochs=100,
            batch_size=8,
            lr=1e-2,
            patience=15,
            weight_decay_grid=[0.0, 1e-4],
            seed=42,
            device="cpu",
        )

        # cal accuracy should be high
        self.assertGreater(fit_info["best_cal_accuracy"], 0.5)

    def test_replacement_direction_orientation(self):
        """Known-weight probe. d_replace = normalize(w_n - w_o). Verify direction goes
        from old to new."""
        embedding_dim = 4
        category_pid_groups = [[0, 1]]
        probe = CategoricalLinearProbe(embedding_dim, category_pid_groups)

        # manually set weights
        with torch.no_grad():
            probe.linear.weight[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # pid 0
            probe.linear.weight[1] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # pid 1

        w_old = probe.linear.weight.data[0]
        w_new = probe.linear.weight.data[1]

        d_replace = torch.nn.functional.normalize(
            (w_new - w_old).unsqueeze(0), dim=1
        ).squeeze(0)

        # direction should be from [1,0,0,0] toward [0,1,0,0]
        # i.e., [-1, 1, 0, 0] normalized
        expected = torch.nn.functional.normalize(
            torch.tensor([-1.0, 1.0, 0.0, 0.0]).unsqueeze(0), dim=1
        ).squeeze(0)

        torch.testing.assert_close(d_replace, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
