"""
Unit tests for baselines.property_design.
"""

import json
import unittest
from pathlib import Path

import torch

from baselines.property_design import (
    cross_category_interactions,
    design_fingerprint,
    pairwise_design,
)


class DuckProperties:
    """Duck-typed Properties for testing."""

    def __init__(self, dict_categories_with_properties):
        self.dict_categories_with_properties = dict_categories_with_properties

        # build property_to_pid, pid_to_property
        self.property_to_pid = {}
        self.pid_to_property = {}
        pid = 0
        for props in dict_categories_with_properties.values():
            for prop in props:
                self.property_to_pid[prop] = pid
                self.pid_to_property[pid] = prop
                pid += 1
        self.n_properties = pid

        # build category_to_cid, cid_to_category
        self.category_to_cid = {}
        self.cid_to_category = {}
        cid = 0
        for cat in dict_categories_with_properties.keys():
            self.category_to_cid[cat] = cid
            self.cid_to_category[cid] = cat
            cid += 1
        self.n_categories = cid

        # build cid_to_pids, pid_to_cid
        self.cid_to_pids = {}
        self.pid_to_cid = {}
        for cat, props in dict_categories_with_properties.items():
            cid = self.category_to_cid[cat]
            pids = [self.property_to_pid[p] for p in props]
            self.cid_to_pids[cid] = pids
            for p in props:
                self.pid_to_cid[self.property_to_pid[p]] = cid


class TestPropertyDesign(unittest.TestCase):

    def test_interaction_count_real(self):
        """Load real categories; cross_category_interactions returns 276 columns."""
        categories_path = Path("dataset_generation/prompts/input/categories_with_properties.json")
        with open(categories_path, "r") as f:
            categories = json.load(f)

        props = DuckProperties(categories)
        columns = cross_category_interactions(props)

        self.assertEqual(len(columns), 276, f"Expected 276 columns, got {len(columns)}")

    def test_no_same_category_pair(self):
        """Every InteractionColumn has different left_category and right_category."""
        categories_path = Path("dataset_generation/prompts/input/categories_with_properties.json")
        with open(categories_path, "r") as f:
            categories = json.load(f)

        props = DuckProperties(categories)
        columns = cross_category_interactions(props)

        for col in columns:
            self.assertNotEqual(
                col.left_category,
                col.right_category,
                f"Same-category pair found: {col}",
            )

    def test_column_order_stable(self):
        """Two calls return identical tuples."""
        categories = {"a": ["x", "y"], "b": ["p", "q"], "c": ["m", "n"]}
        props = DuckProperties(categories)

        columns1 = cross_category_interactions(props)
        columns2 = cross_category_interactions(props)

        self.assertEqual(columns1, columns2, "Column order is not stable")

    def test_design_values_equal_mask_products(self):
        """For synthetic mask (5, 26), pairwise_design returns (5, 302) with correct values."""
        categories_path = Path("dataset_generation/prompts/input/categories_with_properties.json")
        with open(categories_path, "r") as f:
            categories = json.load(f)

        props = DuckProperties(categories)
        columns = cross_category_interactions(props)

        # create synthetic mask
        n = 5
        mask = torch.rand(n, 26)

        design = pairwise_design(mask, columns)

        # check shape
        self.assertEqual(design.shape, (5, 302), f"Expected shape (5, 302), got {design.shape}")

        # main effects block should match mask
        torch.testing.assert_close(design[:, :26], mask, msg="Main effects don't match mask")

        # interaction columns should match mask products
        for i, col in enumerate(columns):
            expected = mask[:, col.left_pid] * mask[:, col.right_pid]
            actual = design[:, 26 + i]
            torch.testing.assert_close(
                actual,
                expected,
                msg=f"Interaction column {i} ({col.left_property} * {col.right_property}) doesn't match product",
            )

    def test_held_pair_zero_support(self):
        """Build mask where no row contains both hat and gun pids; verify interaction column is zero."""
        categories_path = Path("dataset_generation/prompts/input/categories_with_properties.json")
        with open(categories_path, "r") as f:
            categories = json.load(f)

        props = DuckProperties(categories)
        columns = cross_category_interactions(props)

        # find pids for "and a hat" and "holding a gun"
        hat_pid = props.property_to_pid["and a hat"]
        gun_pid = props.property_to_pid["holding a gun"]

        # create mask where no row has both
        n = 10
        mask = torch.zeros(n, 26)
        # some rows have hat, some have gun, but never both
        mask[0:3, hat_pid] = 1.0
        mask[4:7, gun_pid] = 1.0

        design = pairwise_design(mask, columns)

        # find the interaction column for hat * gun
        interaction_idx = None
        for i, col in enumerate(columns):
            if {col.left_pid, col.right_pid} == {hat_pid, gun_pid}:
                interaction_idx = i
                break

        self.assertIsNotNone(interaction_idx, "Hat * gun interaction column not found")

        # verify it's all zeros
        interaction_col = design[:, 26 + interaction_idx]
        self.assertTrue(
            torch.all(interaction_col == 0),
            f"Expected all zeros, got {interaction_col}",
        )


if __name__ == "__main__":
    unittest.main()
