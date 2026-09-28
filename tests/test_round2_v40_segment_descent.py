import unittest
from pathlib import Path

from bf_tap_r2.v40_segment_descent import MEASURED, _segment, analysis, build_probes


class MeasuredTableTests(unittest.TestCase):
    def test_three_points_lie_on_the_a60_v7m_segment(self):
        self.assertEqual(_segment(0.25), (0.10, 0.15, 0.75))
        self.assertEqual(_segment(0.50), (0.20, 0.30, 0.50))
        self.assertEqual(_segment(0.75), (0.30, 0.45, 0.25))
        self.assertEqual(MEASURED["I2"][0], _segment(0.25))
        self.assertEqual(MEASURED["I1"][0], _segment(0.50))
        self.assertEqual(MEASURED["I5"][0], _segment(0.75))

    def test_reported_scores(self):
        self.assertAlmostEqual(MEASURED["I1"][1], 96.3727, places=4)
        self.assertAlmostEqual(MEASURED["I5"][1], 96.3727, places=4)
        self.assertAlmostEqual(MEASURED["I2"][1], 96.3629, places=4)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.report = analysis(step=0.05)

    def test_raised_bounded_ceiling_breaks_the_unreachable_verdict(self):
        self.assertAlmostEqual(self.report["simplex_max_bound_with_V12_iron"], 96.38818, places=4)
        self.assertEqual(self.report["simplex_argmax_coordinates"], [0.0, 0.45, 0.55])
        self.assertAlmostEqual(self.report["bounded_ceiling_with_iron_endpoint"], 96.40418, places=4)
        self.assertGreater(self.report["ceiling_vs_target"], 0.0)
        self.assertFalse(self.report["target_provably_unreachable"])

    def test_grid_growth(self):
        self.assertEqual(self.report["grid_points"], 231)
        self.assertEqual(self.report["bounded_grid_points"], 39)
        self.assertEqual(self.report["unbounded_grid_points"], 192)

    def test_plateau_bracket(self):
        bracket = self.report["ridge"]["lambda_0625_bracket"]
        self.assertAlmostEqual(bracket["lower"], 96.3727, places=4)
        self.assertAlmostEqual(bracket["upper"], 96.3776, places=4)
        self.assertLess(bracket["lower"], bracket["upper"])

    def test_measured_best_is_the_first_interior_probe(self):
        self.assertEqual(self.report["measured_best"]["point"], "I1")
        self.assertAlmostEqual(self.report["measured_best"]["score"], 96.3727, places=4)


class GuardTests(unittest.TestCase):
    def test_refuses_public_probe_output(self):
        with self.assertRaises(ValueError):
            build_probes(Path("/home/lux1/iron"), "configs/round2_v40/PROBES.yaml",
                         "docs/should-not-write")

    def test_refuses_other_probe_spec(self):
        with self.assertRaises(ValueError):
            build_probes(Path("/home/lux1/iron"), "configs/round2_v32/PROBES.yaml",
                         "local/runs/round2-v40/x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
