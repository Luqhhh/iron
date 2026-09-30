import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from bf_tap_r2.v20_release import design_check, read_predictions, run

SPEC20 = {"blend": {"alpha": 0.325}, "gates": {"local_working_gate": 96.25}}
SPEC17 = {"split_seeds": [42, 3407], "confirmation_seeds": [7777, 12011]}
THREAD_ENV = {name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
ROOT = Path(__file__).resolve().parents[1]


def _fixture(rows: int = 40):
    seeds = [42, 3407, 7777, 12011]
    folds = {seed: np.tile(np.arange(5), rows // 5) for seed in seeds}
    frame = pd.DataFrame({"tap_iron": np.full(rows, 100.0), "tap_time_len": np.full(rows, 100.0)})
    return frame, folds


class ReadPredictionsTests(unittest.TestCase):
    def test_requires_complete_and_finite_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, folds = _fixture()
            directory = root / "preds"
            directory.mkdir()
            for fold in range(5):
                values = np.full(int((folds[42] == fold).sum()), 1.0)
                np.save(directory / f"P_LL_T-s42-f{fold}.npy", values)
            read_predictions(directory, "P_LL_T", frame, folds, [42])
            np.save(directory / "P_LL_T-s42-f2.npy", np.full(8, np.nan))
            with self.assertRaisesRegex(ValueError, "Invalid V17 prediction"):
                read_predictions(directory, "P_LL_T", frame, folds, [42])
            with self.assertRaises(FileNotFoundError):
                read_predictions(directory, "P_LL_T", frame, folds, [3407])


class DesignCheckTests(unittest.TestCase):
    def _prepare(self, root: Path, b0_time_offset: float, member_offset: float,
                 iron_offset: float = 0.0):
        frame, folds = _fixture()
        for seed in (42, 3407, 7777, 12011):
            directory = root / "local/runs/round2-v17" / (
                "development-r1" if seed in (42, 3407) else "confirmation-r1")
            directory.mkdir(parents=True, exist_ok=True)
            for fold in range(5):
                values = np.full(int((folds[seed] == fold).sum()), 100.0 + member_offset)
                np.save(directory / f"P_LL_T-s{seed}-f{fold}.npy", values)
        b0 = {seed: {"tap_iron": np.full(len(frame), 100.0 + iron_offset),
                     "tap_time_len": np.full(len(frame), 100.0 + b0_time_offset)}
              for seed in (42, 3407, 7777, 12011)}
        return frame, folds, b0

    def test_promotes_a_gate_clearing_positive_design(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, folds, b0 = self._prepare(root, b0_time_offset=7.5, member_offset=0.0)
            result = design_check(root, SPEC20, SPEC17, frame, folds, b0)
            self.assertTrue(result["all_seeds_positive"])
            self.assertTrue(result["paired_lcb95_positive"])
            self.assertTrue(result["local_working_gate_met"])
            self.assertTrue(result["promoted"])
            self.assertEqual(len(result["seed_results"]), 4)
            self.assertAlmostEqual(result["alpha"], 0.325)

    def test_rejects_a_worse_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, folds, b0 = self._prepare(root, b0_time_offset=0.0, member_offset=100.0)
            result = design_check(root, SPEC20, SPEC17, frame, folds, b0)
            self.assertFalse(result["all_seeds_positive"])
            self.assertFalse(result["local_working_gate_met"])
            self.assertFalse(result["promoted"])

    def test_gate_is_binding_even_when_gains_are_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, folds, b0 = self._prepare(root, b0_time_offset=7.5, member_offset=0.0,
                                             iron_offset=8.0)
            result = design_check(root, SPEC20, SPEC17, frame, folds, b0)
            self.assertTrue(result["all_seeds_positive"])
            self.assertLess(result["development_mean_candidate_score"], 96.25)
            self.assertFalse(result["promoted"])


class RunGuardTests(unittest.TestCase):
    def test_refuses_public_output(self):
        with mock.patch.dict(os.environ, THREAD_ENV):
            with self.assertRaises(ValueError):
                run(ROOT, "configs/round2_v20/SPEC.yaml", "docs/should-not-write")

    def test_refuses_non_v20_spec(self):
        with mock.patch.dict(os.environ, THREAD_ENV):
            with self.assertRaises(ValueError):
                run(ROOT, "configs/round2_v18/SPEC.yaml", "local/runs/round2-v20/x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
