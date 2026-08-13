"""
Unit tests for dataset_generation.compositional_split.
"""

import json
import tempfile
import unittest
from pathlib import Path

from dataset_generation.compositional_split import (
    build_disjoint_split,
    enumerate_full_combos,
    read_categories,
    _validate_holdout_pair,
)


class TestCompositionalSplit(unittest.TestCase):

    def test_pair_filtering_small(self):
        """3 categories x 2 values = 8 tuples; pair (a_val0, b_val0) -> 2 holdout."""
        categories = {
            "a": ["a_val0", "a_val1"],
            "b": ["b_val0", "b_val1"],
            "c": ["c_val0", "c_val1"],
        }
        train, holdout, stats = build_disjoint_split(
            categories,
            holdout_value_pair=("a_val0", "b_val0"),
            seed=0,
        )

        # 2 x 2 x 2 = 8 total; exactly 2 contain both a_val0 and b_val0
        self.assertEqual(len(holdout), 2)
        self.assertEqual(len(train), 6)

        # verify holdout contains both values
        for combo in holdout:
            self.assertIn("a_val0", combo.values())
            self.assertIn("b_val0", combo.values())

        # verify train does not contain both values simultaneously
        for combo in train:
            values = set(combo.values())
            self.assertFalse("a_val0" in values and "b_val0" in values)

    def test_unknown_property_error(self):
        """Unknown phrase raises ValueError."""
        categories = {"a": ["x", "y"], "b": ["p", "q"]}
        with self.assertRaises(ValueError) as cm:
            _validate_holdout_pair(categories, ("x", "unknown"))
        self.assertIn("Unknown property", str(cm.exception))

    def test_same_category_pair_error(self):
        """Two values from same category raises ValueError."""
        categories = {"a": ["x", "y"], "b": ["p", "q"]}
        with self.assertRaises(ValueError) as cm:
            _validate_holdout_pair(categories, ("x", "y"))
        self.assertIn("different categories", str(cm.exception))

    def test_duplicate_property_error(self):
        """Same phrase twice raises ValueError."""
        categories = {"a": ["x", "y"], "b": ["p", "q"]}
        with self.assertRaises(ValueError) as cm:
            _validate_holdout_pair(categories, ("x", "x"))
        self.assertIn("Duplicate property", str(cm.exception))

    def test_disjoint_certificate(self):
        """build_disjoint_split holdout and train have no tuple overlap."""
        categories = {"a": ["x", "y"], "b": ["p", "q"], "c": ["m", "n"]}
        train, holdout, stats = build_disjoint_split(
            categories,
            n_holdout=3,
            seed=42,
        )

        # no overlap between train and holdout tuples
        train_tuples = {tuple(sorted(c.items())) for c in train}
        holdout_tuples = {tuple(sorted(c.items())) for c in holdout}
        overlap = train_tuples & holdout_tuples
        self.assertEqual(len(overlap), 0, f"Found overlap: {overlap}")

    def test_max_holdout_prompts_rejected_in_pair_mode(self):
        """Both holdout_value_pair and max_holdout_prompts raises ValueError."""
        categories = {"a": ["x", "y"], "b": ["p", "q"]}
        with self.assertRaises(ValueError) as cm:
            build_disjoint_split(
                categories,
                holdout_value_pair=("x", "p"),
                max_holdout_prompts=1,
                seed=0,
            )
        self.assertIn("max_holdout_prompts cannot be used with holdout_value_pair", str(cm.exception))

    def test_real_dict_hat_and_gun(self):
        """Load real categories; hat + gun -> 512 holdout, 3584 train."""
        categories_path = Path("dataset_generation/prompts/input/categories_with_properties.json")
        categories = read_categories(categories_path)

        train, holdout, stats = build_disjoint_split(
            categories,
            holdout_value_pair=("and a hat", "holding a gun"),
            seed=0,
        )

        self.assertEqual(stats["n_holdout_written"], 512, f"Expected 512 holdout, got {stats['n_holdout_written']}")
        self.assertEqual(stats["n_train_written"], 3584, f"Expected 3584 train, got {stats['n_train_written']}")

        # verify all holdout contain both
        for combo in holdout:
            self.assertIn("and a hat", combo.values())
            self.assertIn("holding a gun", combo.values())

        # verify no train contain both
        for combo in train:
            values = set(combo.values())
            self.assertFalse("and a hat" in values and "holding a gun" in values)

    def test_real_dict_blond_and_blue_eyes(self):
        """blond + blue eyes -> 1024 holdout, 3072 train."""
        categories_path = Path("dataset_generation/prompts/input/categories_with_properties.json")
        categories = read_categories(categories_path)

        train, holdout, stats = build_disjoint_split(
            categories,
            holdout_value_pair=("A blond girl", "with blue eyes"),
            seed=0,
        )

        self.assertEqual(stats["n_holdout_written"], 1024, f"Expected 1024 holdout, got {stats['n_holdout_written']}")
        self.assertEqual(stats["n_train_written"], 3072, f"Expected 3072 train, got {stats['n_train_written']}")

        # verify all holdout contain both
        for combo in holdout:
            self.assertIn("A blond girl", combo.values())
            self.assertIn("with blue eyes", combo.values())

        # verify no train contain both
        for combo in train:
            values = set(combo.values())
            self.assertFalse("A blond girl" in values and "with blue eyes" in values)


if __name__ == "__main__":
    unittest.main()
