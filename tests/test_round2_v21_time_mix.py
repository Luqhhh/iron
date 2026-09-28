import csv
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bf_tap_r2.v18_compose import SourceTable
from bf_tap_r2.v21_time_mix import _build_payload, _verify, load_member, run


def _table(ids, iron_text, time_text):
    return SourceTable(
        name="parent", path="parent.zip", zip_sha256="0" * 64, ids=tuple(ids),
        iron=np.asarray([float(v) for v in iron_text]), iron_text=tuple(iron_text),
        time=np.asarray([float(v) for v in time_text]), time_text=tuple(time_text),
        other_text={"sample_id": tuple(ids), "pred_tap_iron": tuple(iron_text),
                    "pred_tap_time_len": tuple(time_text)},
    )


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["s1", "s2", "s3"]
        self.iron = ["1000.0", "1100.5", "1200.25"]
        self.parent = _table(self.ids, self.iron, ["100.0", "110.0", "120.0"])
        self.endpoints = {"v36_time": np.array([100.0, 110.0, 120.0]),
                          "pll_time": np.array([90.0, 100.0, 130.0])}

    def test_copies_iron_strings_and_blends_time(self):
        design = {"time": {"v36_time": 0.6, "pll_time": 0.4}}
        payload = _build_payload(self.ids, self.parent.iron_text,
                                 0.6 * self.endpoints["v36_time"] + 0.4 * self.endpoints["pll_time"])
        rows = list(csv.DictReader(io.StringIO(payload.decode())))
        self.assertEqual([row["pred_tap_iron"] for row in rows], self.iron)
        self.assertAlmostEqual(float(rows[0]["pred_tap_time_len"]), 96.0, places=9)
        report = _verify(payload, self.parent, design, self.endpoints, 1e-12)
        self.assertEqual(report["iron_string_mismatches"], 0)
        self.assertEqual(report["time_blend_relative_difference"], 0.0)

    def test_detects_a_changed_iron_string(self):
        design = {"time": {"v36_time": 1.0}}
        payload = _build_payload(self.ids, ["1000.0", "1100.5", "999.0"], self.endpoints["v36_time"])
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            _verify(payload, self.parent, design, self.endpoints, 1e-12)

    def test_detects_a_wrong_blend(self):
        design = {"time": {"v36_time": 0.5, "pll_time": 0.5}}
        payload = _build_payload(self.ids, self.parent.iron_text, self.endpoints["v36_time"])
        with self.assertRaisesRegex(ValueError, "read-back failed"):
            _verify(payload, self.parent, design, self.endpoints, 1e-12)


class MemberTests(unittest.TestCase):
    def test_rejects_hash_mismatch_and_bad_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "member.npy"
            np.save(path, np.array([1.0, 2.0]))
            load_member(path, None, 2)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_member(path, "0" * 64, 2)
            with self.assertRaisesRegex(ValueError, "Invalid P-LL member"):
                load_member(path, None, 3)


class RunGuardTests(unittest.TestCase):
    def test_refuses_public_output(self):
        with self.assertRaises(ValueError):
            run(Path("/home/lux1/iron"), "configs/round2_v21/SPEC.yaml", "docs/should-not-write")

    def test_refuses_wrong_spec_directory(self):
        with self.assertRaises(ValueError):
            run(Path("/home/lux1/iron"), "configs/round2_v18/SPEC.yaml", "local/runs/round2-v21/x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
