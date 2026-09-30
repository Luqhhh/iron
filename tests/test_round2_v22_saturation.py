from pathlib import Path
import unittest

import numpy as np

from bf_tap_r2.v22_saturation import nested_gain, run

ROOT = Path(__file__).resolve().parents[1]


class NestedGainTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.full(40, 100.0)
        self.incumbent = {42: np.full(40, 107.5), 3407: np.full(40, 107.5)}
        self.grid = [index / 10 for index in range(11)]

    def test_rewards_a_member_closer_to_truth(self):
        member = {seed: self.truth.copy() for seed in self.incumbent}
        gain, alphas = nested_gain(self.truth, self.incumbent, member, self.grid)
        self.assertGreater(gain, 0.0)
        self.assertEqual(alphas, [1.0, 1.0])

    def test_identical_member_gives_zero(self):
        gain, alphas = nested_gain(self.truth, self.incumbent,
                                   {seed: self.incumbent[seed] for seed in self.incumbent}, self.grid)
        self.assertEqual(gain, 0.0)
        self.assertEqual(alphas, [0.0, 0.0])

    def test_weight_comes_from_the_other_seed_only(self):
        member = {42: self.incumbent[42] + 50.0, 3407: self.truth.copy()}
        gain, alphas = nested_gain(self.truth, self.incumbent, member, self.grid)
        self.assertEqual(alphas[0], 1.0)      # seed 42 held out -> weight from 3407
        self.assertEqual(alphas[1], 0.0)      # seed 3407 held out -> weight from 42
        self.assertLess(gain, 0.0)


class GuardTests(unittest.TestCase):
    def test_refuses_public_output(self):
        with self.assertRaises(ValueError):
            run(ROOT, "configs/round2_v22/SPEC.yaml", "docs/should-not-write")

    def test_refuses_missing_spec(self):
        with self.assertRaises(FileNotFoundError):
            run(ROOT, "configs/round2_v22/NOPE.yaml", "local/runs/round2-v22/x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
