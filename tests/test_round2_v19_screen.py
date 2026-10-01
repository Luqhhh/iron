import pytest
pytest.importorskip("torch", reason="Optional neural dependency is absent from the locked baseline environment")

import math
import unittest

import numpy as np
import pandas as pd

from bf_tap_r2.v19_screen import paired_lcb95, screen_candidate


class PairedLcbTests(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(paired_lcb95([1.0, 1.0, 1.0, 1.0]), 1.0, places=12)
        value = paired_lcb95([0.004, 0.006, 0.005, 0.007])
        self.assertAlmostEqual(value, 0.0055 - 3.182 * 0.0012909944487358056 / 2.0, places=12)
        self.assertLess(value, 0.0055)

    def test_requires_two_observations(self):
        with self.assertRaisesRegex(ValueError, "two paired"):
            paired_lcb95([1.0])


class ScreenTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        rows = 40
        self.frame = pd.DataFrame({"tap_iron": rng.normal(1000, 50, rows)})
        self.folds = {42: np.tile(np.arange(5), 8), 3407: np.repeat(np.arange(5), 8)}
        base = {42: np.full(rows, 1000.0), 3407: np.full(rows, 1000.0)}
        # A member that is genuinely closer to the truth on every row.
        truth = self.frame["tap_iron"].to_numpy()
        closer = {seed: 0.5 * (base[seed] + truth) for seed in self.folds}
        self.base = base
        self.closer = closer
        self.grid = [index / 10 for index in range(11)]

    def test_detects_a_real_improvement(self):
        result = screen_candidate(self.frame, self.folds, self.base, self.closer, "tap_iron", self.grid)
        self.assertGreater(result["mean_incremental_gain"], 0.0)
        self.assertTrue(result["all_seeds_positive"])
        self.assertEqual(result["positive_cells"], 10)
        self.assertTrue(all(alpha == 1.0 for alpha in result["alphas"].values()))

    def test_identical_member_gives_zero(self):
        result = screen_candidate(self.frame, self.folds, self.base, self.base, "tap_iron", self.grid)
        self.assertEqual(result["mean_incremental_gain"], 0.0)
        self.assertFalse(result["all_seeds_positive"])
        self.assertEqual(result["positive_cells"], 0)

    def test_fold_cells_are_descriptive_and_counted(self):
        result = screen_candidate(self.frame, self.folds, self.base, self.closer, "tap_iron", self.grid)
        self.assertEqual(len(result["cells"]), 10)
        self.assertEqual({cell["seed"] for cell in result["cells"]}, {42, 3407})
        self.assertTrue(all(cell["rows"] == 8 for cell in result["cells"]))

    def test_weight_is_selected_on_the_other_seed(self):
        # Seed 3407 rewards the member, seed 42 does not; the nested alpha for
        # seed 42 must come from 3407 only.
        truth = self.frame["tap_iron"].to_numpy()
        member = {42: self.base[42] + 200.0, 3407: truth.copy()}
        result = screen_candidate(self.frame, self.folds, self.base, member, "tap_iron", self.grid)
        self.assertEqual(result["alphas"]["42"], 1.0)
        self.assertLess(result["seed_deltas"]["42"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
