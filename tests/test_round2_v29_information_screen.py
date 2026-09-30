from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from bf_tap_r2.data import FEATURES
from bf_tap_r2.v29_information_screen import residual_screen, run

ROOT = Path(__file__).resolve().parents[1]


def _frame(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({feature: rng.normal(size=rows) for feature in FEATURES})
    frame["spout_no"] = rng.integers(1, 3, rows)
    frame["sample_id"] = [f"s{index}" for index in range(rows)]
    return frame


def _folds(rows: int):
    return {seed: np.tile(np.arange(5), rows // 5) for seed in (42, 3407)}


class ResidualScreenTests(unittest.TestCase):
    def test_no_signal_when_the_target_is_independent_noise(self):
        rows = 2000
        frame = _frame(rows, 7)
        rng = np.random.default_rng(8)
        frame["tap_time_len"] = rng.normal(size=rows)
        frame["tap_iron"] = rng.normal(size=rows)
        folds = _folds(rows)
        zeros = {seed: np.zeros(rows) for seed in folds}
        report = residual_screen(frame, folds, {"tap_time_len": zeros, "tap_iron": zeros}, {})
        self.assertEqual(report["candidates"], 630)
        self.assertLess(report["tap_time_len"]["maximum_absolute_correlation"], 0.15)
        self.assertLess(report["tap_iron"]["maximum_absolute_correlation"], 0.15)

    def test_detects_an_injected_pairwise_interaction(self):
        rows = 2000
        frame = _frame(rows, 11)
        interaction = frame["air_volume"].to_numpy() * frame["oxygen"].to_numpy()
        frame["tap_time_len"] = interaction
        frame["tap_iron"] = np.zeros(rows)
        folds = _folds(rows)
        zeros = {seed: np.zeros(rows) for seed in folds}
        report = residual_screen(frame, folds, {"tap_time_len": zeros, "tap_iron": zeros}, {})
        self.assertGreater(report["tap_time_len"]["maximum_absolute_correlation"], 0.99)
        self.assertGreater(report["tap_time_len"]["candidates_above_null_scale"], 0)
        self.assertEqual(report["tap_time_len"]["top"][0]["name"], "air_volume*oxygen")


class GuardTests(unittest.TestCase):
    def test_refuses_public_output(self):
        with self.assertRaises(ValueError):
            run(ROOT, "configs/round2_v29/SPEC.yaml", "docs/should-not-write")


if __name__ == "__main__":
    unittest.main(verbosity=2)
