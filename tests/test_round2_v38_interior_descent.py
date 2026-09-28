import unittest

from bf_tap_r2.v38_interior_descent import MEASURED, PENDING, analysis, design
from bf_tap_r2.v32_family_ceiling import secant_bound


class MeasuredTableTests(unittest.TestCase):
    def test_interior_points_are_recorded_with_their_results(self):
        self.assertAlmostEqual(MEASURED["I1"][1], 96.3727, places=4)
        self.assertAlmostEqual(MEASURED["I2"][1], 96.3629, places=4)
        self.assertEqual(MEASURED["I1"][0], (0.20, 0.30, 0.50))
        self.assertEqual(MEASURED["I2"][0], (0.10, 0.15, 0.75))

    def test_previous_best_is_still_in_the_table(self):
        self.assertAlmostEqual(MEASURED["B0"][1], 96.3679, places=4)
        self.assertLess(MEASURED["B0"][1], MEASURED["I1"][1])


class PendingBoundTests(unittest.TestCase):
    def setUp(self):
        self.report = analysis(step=0.05)
        self.table = design()

    def test_v7m_endpoint_is_provably_dead(self):
        entry = self.report["pending_packages"]["3_TIME_V100"]
        self.assertAlmostEqual(entry["upper_bound"], 96.3531, places=4)
        self.assertLess(entry["upper_bound"], self.report["measured_best"]["score"])
        self.assertEqual(entry["bound_source"]["from"], "I1")
        self.assertEqual(entry["bound_source"]["to"], "I2")
        self.assertAlmostEqual(entry["bound_source"]["lambda"], 2.0, places=9)

    def test_highest_remaining_bound_is_the_pending_interior_point(self):
        entry = self.report["pending_packages"]["5_INTERIOR_A60V7_25"]
        self.assertAlmostEqual(entry["upper_bound"], 96.3825, places=4)
        self.assertEqual(self.report["simplex_argmax_coordinates"], [0.3, 0.45, 0.25])
        self.assertAlmostEqual(self.report["simplex_max_bound_with_V12_iron"], 96.3825, places=4)

    def test_iron_endpoint_uses_the_iron_line_bound(self):
        entry = self.report["pending_packages"]["4_IRON_W100"]
        self.assertIsNone(entry["upper_bound"])
        self.assertAlmostEqual(entry["iron_line_upper_bound"], 96.3839, places=4)

    def test_ceiling_and_slope(self):
        self.assertAlmostEqual(self.report["family_ceiling_with_iron_endpoint"], 96.3985, places=4)
        self.assertAlmostEqual(self.report["ceiling_below_target"], 0.0015, places=6)
        self.assertAlmostEqual(self.report["n_direction_slope_at_v7m_half"], 0.0384, places=4)

    def test_bounded_set_grew_when_the_interior_points_arrived(self):
        self.assertEqual(self.report["grid_points"], 231)
        self.assertEqual(self.report["bounded_grid_points"], 33)
        self.assertEqual(self.report["unbounded_grid_points"], 198)


class GuardTests(unittest.TestCase):
    def test_refuses_public_output(self):
        with self.assertRaises(ValueError):
            from pathlib import Path
            from bf_tap_r2.v38_interior_descent import run
            run(Path("/home/lux1/iron"), "configs/round2_v38/SPEC.yaml", "docs/should-not-write")


if __name__ == "__main__":
    unittest.main(verbosity=2)
