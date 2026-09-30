from pathlib import Path
import unittest

import numpy as np

from bf_tap_r2.v26_logtarget import inverse, screen, transform

ROOT = Path(__file__).resolve().parents[1]


class TransformTests(unittest.TestCase):
    def test_log_round_trip(self):
        y = np.array([62.0, 128.0, 223.5])
        np.testing.assert_allclose(inverse(transform(y, "log"), "log"), y, rtol=0, atol=1e-9)

    def test_sqrt_round_trip_and_nonnegative_clip(self):
        y = np.array([62.0, 128.0, 223.5])
        np.testing.assert_allclose(inverse(transform(y, "sqrt"), "sqrt"), y, rtol=0, atol=1e-9)
        clipped = inverse(np.array([-3.0, 2.0]), "sqrt")
        self.assertEqual(clipped[0], 0.0)
        self.assertAlmostEqual(clipped[1], 4.0)

    def test_unknown_transform_rejected(self):
        with self.assertRaises(ValueError):
            transform(np.array([1.0]), "nope")
        with self.assertRaises(ValueError):
            inverse(np.array([1.0]), "nope")


class GuardTests(unittest.TestCase):
    def test_refuses_public_output(self):
        with self.assertRaises(ValueError):
            screen(ROOT, "configs/round2_v26/SPEC.yaml",
                   "docs/should-not-write", ["L_TIME"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
