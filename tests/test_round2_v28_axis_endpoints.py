from pathlib import Path
import unittest

import yaml

from bf_tap_r2.v18_compose import END_NAMES
from bf_tap_r2.v28_axis_endpoints import run

ROOT = Path("/home/lux1/iron")


class SpecTests(unittest.TestCase):
    def setUp(self):
        self.spec = yaml.safe_load((ROOT / "configs/round2_v28/SPEC.yaml").read_text())

    def test_designs_are_convex_and_bounded(self):
        endpoints = set(END_NAMES)
        self.assertEqual(len(self.spec["designs"]), 3)
        for name, design in self.spec["designs"].items():
            for kind in ("iron", "time"):
                column = design[kind]
                self.assertEqual(column["kind"], "convex", name)
                self.assertAlmostEqual(sum(column["weights"].values()), 1.0, places=12, msg=name)
                for endpoint, weight in column["weights"].items():
                    self.assertIn(endpoint, endpoints, f"{name}:{endpoint}")
                    self.assertGreaterEqual(weight, 0.0, name)
                    self.assertLessEqual(weight, 1.0, name)

    def test_iron_endpoint_is_the_pure_member(self):
        for name in ("V28_IRON_W100", "V28_IRON_W100_TIME_V75", "V28_IRON_W100_TIME_V100"):
            self.assertEqual(self.spec["designs"][name]["iron"]["weights"], {"v12_member_iron": 1.0})


class GuardTests(unittest.TestCase):
    def test_refuses_public_output(self):
        with self.assertRaises(ValueError):
            run(ROOT, "configs/round2_v28/SPEC.yaml", "docs/should-not-write")

    def test_refuses_other_spec_directory(self):
        with self.assertRaises(ValueError):
            run(ROOT, "configs/round2_v18/SPEC.yaml", "local/runs/round2-v28/x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
