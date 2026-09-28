from pathlib import Path
import unittest

import yaml

from bf_tap_r2.v23_capacity import base_trial, probe_trial

ROOT = Path("/home/lux1/iron")
SPEC = yaml.safe_load((ROOT / "configs/round2_v23/SPEC.yaml").read_text())


class TrialTests(unittest.TestCase):
    def test_base_recipe_is_the_frozen_time_winner(self):
        trial = base_trial(ROOT, SPEC)
        self.assertEqual(trial["trial_id"], "v36-s1-N-0048")
        self.assertEqual(trial["target"], "tap_time_len")
        self.assertEqual(trial["structure"], "raw_tabm")
        self.assertEqual(trial["capacity_name"], "large")
        self.assertEqual(trial["parameters"]["k"], 32)
        self.assertEqual(trial["parameters"]["d_block"], 512)

    def test_probe_only_changes_capacity(self):
        wide = probe_trial(ROOT, SPEC, "P_WIDE")
        base = base_trial(ROOT, SPEC)
        self.assertEqual(wide["target"], base["target"])
        self.assertEqual(wide["parameters"]["d_block"], 768)
        self.assertEqual(wide["parameters"]["n_blocks"], 3)
        self.assertEqual(wide["parameters"]["loss"], base["parameters"]["loss"])
        self.assertEqual(wide["parameters"]["optimizer"], base["parameters"]["optimizer"])
        self.assertEqual(wide["parameters"]["max_epochs"], base["parameters"]["max_epochs"])
        deep = probe_trial(ROOT, SPEC, "P_DEEP")
        self.assertEqual(deep["parameters"]["n_blocks"], 5)
        self.assertEqual(deep["parameters"]["d_block"], 512)

    def test_unknown_probe_rejected(self):
        with self.assertRaises(KeyError):
            probe_trial(ROOT, SPEC, "P_NOPE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
