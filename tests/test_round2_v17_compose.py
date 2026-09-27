import csv
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile
from bf_tap_r2.compose_v12_v7 import audit_and_compose


class ComposerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = {
            "a35": [["s1", "10.000", "20.0"], ["s2", "12.0", "22.000"], ["s3", "14.0", "24.0"]],
            "v12": [["s1", "11.123400", "20.0"], ["s2", "12.0", "22.000"], ["s3", "13.2", "24.0"]],
            "v7": [["s1", "10.000", "21.123400"], ["s2", "12.0", "23.1"], ["s3", "14.0", "24.0"]],
        }

    def tearDown(self):
        self.tmp.cleanup()

    def run_case(self, overrides=None):
        paths, hashes = {}, {}
        for name, rows in self.rows.items():
            stream = io.StringIO(newline="")
            writer = csv.writer(stream)
            writer.writerow(["sample_id", "pred_tap_iron", "pred_tap_time_len"])
            writer.writerows(rows)
            paths[name] = self.root / (name + ".zip")
            with zipfile.ZipFile(paths[name], "w") as archive:
                archive.writestr("result.csv", stream.getvalue().encode())
            hashes[name] = hashlib.sha256(paths[name].read_bytes()).hexdigest()
        if overrides:
            hashes.update(overrides)
        return audit_and_compose(paths, expected_hashes=hashes, expected_rows=3)

    def test_preserves_field_strings_and_arithmetic(self):
        payload, report = self.run_case()
        result = list(csv.DictReader(io.StringIO(payload.decode())))
        self.assertEqual(result[0]["pred_tap_iron"], "11.123400")
        self.assertEqual(result[0]["pred_tap_time_len"], "21.123400")
        self.assertEqual(report["conditional_score_arithmetic"], "96.3679")
        self.assertEqual(report["remaining_to_96_4_arithmetic"], "0.0321")
        self.assertIsNone(report["actual_combined_platform_score"])
        self.assertIsNone(report["output_zip"])

    def test_rejects_hash_mismatch(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_case({"v12": "0" * 64})

    def test_rejects_nonisolated_v12(self):
        self.rows["v12"][0][2] = "20.0001"
        with self.assertRaisesRegex(ValueError, "isolated iron"):
            self.run_case()

    def test_rejects_nonisolated_v7(self):
        self.rows["v7"][0][1] = "10.01"
        with self.assertRaisesRegex(ValueError, "isolated time"):
            self.run_case()

    def test_rejects_duplicate_ids(self):
        self.rows["v12"][1][0] = "s1"
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            self.run_case()

    def test_rejects_order_mismatch(self):
        self.rows["v7"] = list(reversed(self.rows["v7"]))
        with self.assertRaisesRegex(ValueError, "order differs"):
            self.run_case()

    def test_rejects_nonfinite(self):
        self.rows["v12"][0][1] = "NaN"
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.run_case()

    def test_rejects_negative(self):
        self.rows["v12"][0][1] = "-1.0"
        with self.assertRaisesRegex(ValueError, "negative"):
            self.run_case()

    def test_rejects_wrong_count(self):
        self.rows["v12"].pop()
        with self.assertRaisesRegex(ValueError, "expected 3 rows"):
            self.run_case()


if __name__ == "__main__":
    unittest.main(verbosity=2)
