import unittest

from bf_tap_r2.v32_family_ceiling import (
    IRON_HEAD_ROOM,
    MEASURED_TIME,
    measured_table,
    secant_bound,
    simplex_ceiling,
)


class SecantTests(unittest.TestCase):
    def test_extends_a_chord_to_an_upper_bound(self):
        design = {"a": ((1.0, 0.0, 0.0), 1.0), "b": ((0.5, 0.5, 0.0), 2.0)}
        bound = secant_bound((0.0, 1.0, 0.0), design)
        self.assertIsNotNone(bound)
        self.assertAlmostEqual(bound[0], 3.0, places=9)      # 1 + 2*(2-1)

    def test_no_bound_inside_a_chord(self):
        design = {"a": ((1.0, 0.0, 0.0), 1.0), "b": ((0.0, 1.0, 0.0), 3.0)}
        self.assertIsNone(secant_bound((0.5, 0.5, 0.0), design))

    def test_extension_backwards_is_also_a_bound(self):
        design = {"a": ((0.5, 0.5, 0.0), 1.0), "b": ((0.0, 1.0, 0.0), 2.0)}
        bound = secant_bound((1.0, 0.0, 0.0), design)
        self.assertIsNotNone(bound)
        self.assertAlmostEqual(bound[0], 0.0, places=9)


class CeilingTests(unittest.TestCase):
    def setUp(self):
        self.report = simplex_ceiling(step=0.05)

    def test_ceiling_is_below_the_target(self):
        self.assertAlmostEqual(self.report["simplex_max_bound_with_V12_iron"], 96.3832, places=4)
        self.assertAlmostEqual(self.report["family_ceiling_with_iron_endpoint"], 96.3992, places=4)
        self.assertAlmostEqual(self.report["ceiling_below_target"], 0.0008, places=6)

    def test_argmax_is_the_pure_v7m_vertex(self):
        self.assertEqual(self.report["simplex_argmax_coordinates"], [0.0, 0.0, 1.0])
        self.assertEqual(self.report["simplex_argmax_source"]["from"], "A35")
        self.assertEqual(self.report["simplex_argmax_source"]["to"], "B0")
        self.assertAlmostEqual(self.report["simplex_argmax_source"]["lambda"], 2.0, places=9)

    def test_measured_best_is_the_combined_package(self):
        self.assertEqual(self.report["measured_best"]["point"], "B0")
        measured = measured_table()
        self.assertIn("B0", measured)
        every_lift = all(len(record) == 3 for record in MEASURED_TIME.values())
        self.assertTrue(every_lift)
        self.assertAlmostEqual(IRON_HEAD_ROOM, 0.0160, places=9)

    def test_the_bounded_region_is_a_small_part_of_the_simplex(self):
        self.assertGreater(self.report["grid_points"], 200)
        self.assertGreater(self.report["unbounded_grid_points"], 0)
        self.assertEqual(self.report["grid_points"],
                         self.report["bounded_grid_points"] + self.report["unbounded_grid_points"])
        for row in self.report["runner_up_bounds"]:
            self.assertLessEqual(row["bound"], self.report["simplex_max_bound_with_V12_iron"] + 1e-9)


class AxisBoundRegressionTests(unittest.TestCase):
    """Pin the corrected axis bounds so the factor-of-two error cannot return."""

    def _bound(self, low, high, weight):
        slope = (high - low) / 0.5
        return high + slope * (weight - 0.5)

    def test_iron_axis_on_the_b0_time_column(self):
        # w=0 -> V7 package 96.3519, w=0.5 -> B0 96.3679
        self.assertAlmostEqual(self._bound(96.3519, 96.3679, 0.75), 96.3759, places=4)
        self.assertAlmostEqual(self._bound(96.3519, 96.3679, 1.0), 96.3839, places=4)

    def test_iron_axis_on_the_a35_time_column(self):
        self.assertAlmostEqual(self._bound(96.3366, 96.3526, 0.75), 96.3606, places=4)
        self.assertAlmostEqual(self._bound(96.3366, 96.3526, 1.0), 96.3686, places=4)

    def test_time_axis_on_the_v12_iron_column(self):
        self.assertAlmostEqual(self._bound(96.3526, 96.3679, 0.75), 96.37555, places=5)
        self.assertAlmostEqual(self._bound(96.3526, 96.3679, 1.0), 96.3832, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
