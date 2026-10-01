import pytest
pytest.importorskip("torch", reason="Optional neural dependency is absent from the locked baseline environment")

import csv
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from bf_tap_r2.v18_compose import (
    build_design,
    read_sources,
    recover_endpoints,
    verify_payload,
)
from bf_tap_r2.v18_surface import concavity_bound, platform_bounds

COLUMNS = ("sample_id", "pred_tap_iron", "pred_tap_time_len")
ALPHAS = {"a35": 0.35, "a45": 0.45, "a60": 0.60, "a72": 0.72, "a85": 0.85, "a100": 1.0}
IDS = ["s1", "s2", "s3"]
V36_TIME = np.array([100.0, 110.0, 120.0])
N_TIME = np.array([90.0, 100.0, 111.0])
V36_IRON = np.array([1000.0, 1100.0, 1200.0])
V12_MEMBER = np.array([900.0, 1050.0, 1300.0])
V7_MEMBER = np.array([95.0, 108.0, 125.0])


def _rows(iron, time):
    return [[sid, format(float(i), ".17g"), format(float(t), ".17g")]
            for sid, i, t in zip(IDS, iron, time)]


class ComposeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        a35_time = (1 - 0.35) * V36_TIME + 0.35 * N_TIME
        v7_time = 0.5 * a35_time + 0.5 * V7_MEMBER
        v12_iron = 0.5 * V36_IRON + 0.5 * V12_MEMBER
        tables = {}
        for name, alpha in ALPHAS.items():
            tables[name] = {"iron": V36_IRON, "time": (1 - alpha) * V36_TIME + alpha * N_TIME}
        tables["v12"] = {"iron": v12_iron, "time": a35_time}
        tables["v7"] = {"iron": V36_IRON, "time": v7_time}
        self.paths, self.hashes = {}, {}
        for name, column in tables.items():
            stream = io.StringIO(newline="")
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(list(COLUMNS))
            writer.writerows(_rows(column["iron"], column["time"]))
            path = self.root / f"{name}.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("result.csv", stream.getvalue().encode())
            self.paths[name] = path
            self.hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.spec = {
            "sources": {
                name: {"path": str(self.paths[name]), "sha256": self.hashes[name],
                       "role": "n_line_crosscheck" if name in ("a45", "a72", "a85", "a100") else "composition_parent",
                       **({"n_alpha": alpha} if name in ALPHAS else {})}
                for name, alpha in ALPHAS.items()
            } | {
                "v12": {"path": str(self.paths["v12"]), "sha256": self.hashes["v12"], "role": "composition_parent"},
                "v7": {"path": str(self.paths["v7"]), "sha256": self.hashes["v7"], "role": "composition_parent"},
            },
            "composition": {"crosscheck_tolerance": 1e-9},
            "verification": {"blend_relative_tolerance": 1e-12},
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_recovers_endpoints_and_crosschecks_n_line(self):
        tables = read_sources(self.root, self.spec)
        endpoints, report = recover_endpoints(tables, self.spec)
        self.assertLessEqual(report["n_line_worst_absolute_difference"], 1e-9)
        np.testing.assert_allclose(endpoints["v36_time"], V36_TIME, rtol=0, atol=1e-9)
        np.testing.assert_allclose(endpoints["n_time"], N_TIME, rtol=0, atol=1e-9)
        np.testing.assert_allclose(endpoints["v12_member_iron"], V12_MEMBER, rtol=0, atol=1e-9)
        np.testing.assert_allclose(endpoints["v7_member_time"], V7_MEMBER, rtol=0, atol=1e-9)

    def test_copy_design_preserves_field_text(self):
        tables = read_sources(self.root, self.spec)
        endpoints, _ = recover_endpoints(tables, self.spec)
        payload, meta = build_design({"iron": {"kind": "copy", "source": "v12"},
                                      "time": {"kind": "copy", "source": "v7"}},
                                     endpoints, tables)
        rows = list(csv.DictReader(io.StringIO(payload.decode())))
        self.assertEqual(rows[0]["pred_tap_iron"], format(float(0.5 * 1000 + 0.5 * 900), ".17g"))
        self.assertEqual(rows[0]["pred_tap_time_len"], format(float(0.5 * ((1 - 0.35) * 100 + 0.35 * 90) + 0.5 * 95), ".17g"))
        self.assertTrue(meta["iron_unchanged_by_copy"])
        self.assertTrue(meta["time_unchanged_by_copy"])
        verify_payload(payload, {"iron": {"kind": "copy", "source": "v12"},
                                 "time": {"kind": "copy", "source": "v7"}},
                       endpoints, tables, 1e-12)

    def test_convex_design_matches_expected_column(self):
        tables = read_sources(self.root, self.spec)
        endpoints, _ = recover_endpoints(tables, self.spec)
        design = {"iron": {"kind": "copy", "source": "v12"},
                  "time": {"kind": "convex", "weights": {"v36_time": 0.25, "v7_member_time": 0.75}}}
        payload, _ = build_design(design, endpoints, tables)
        rows = list(csv.DictReader(io.StringIO(payload.decode())))
        expected = 0.25 * V36_TIME + 0.75 * V7_MEMBER
        observed = np.asarray([float(row["pred_tap_time_len"]) for row in rows])
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-9)
        verify_payload(payload, design, endpoints, tables, 1e-12)

    def test_rejects_hash_mismatch(self):
        self.spec["sources"]["v12"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            read_sources(self.root, self.spec)

    def test_rejects_order_mismatch(self):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(list(COLUMNS))
        writer.writerows(list(reversed(_rows(V36_IRON, (1 - 0.6) * V36_TIME + 0.6 * N_TIME))))
        path = self.root / "a60.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("result.csv", stream.getvalue().encode())
        self.spec["sources"]["a60"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "order differs"):
            read_sources(self.root, self.spec)

    def test_rejects_non_convex_weights(self):
        tables = read_sources(self.root, self.spec)
        endpoints, _ = recover_endpoints(tables, self.spec)
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            build_design({"iron": {"kind": "copy", "source": "v12"},
                          "time": {"kind": "convex", "weights": {"v36_time": 0.5}}}, endpoints, tables)
        with self.assertRaisesRegex(ValueError, "Unknown endpoint"):
            build_design({"iron": {"kind": "copy", "source": "v12"},
                          "time": {"kind": "convex",
                                   "weights": {"a35_time": 1.0, "v7_member_time": 1.0}}}, endpoints, tables)

    def test_rejects_negative_composition(self):
        tables = read_sources(self.root, self.spec)
        endpoints, _ = recover_endpoints(tables, self.spec)
        endpoints["v12_member_iron"] = np.array([-1.0, 10.0, 10.0])
        with self.assertRaisesRegex(ValueError, "negative rows"):
            build_design({"iron": {"kind": "convex", "weights": {"v12_member_iron": 1.0}},
                          "time": {"kind": "copy", "source": "v7"}}, endpoints, tables)

    def test_spec_designs_are_convex_and_bounded(self):
        import yaml

        spec = yaml.safe_load(Path("configs/round2_v18/SPEC.yaml").read_text())
        for name, design in spec["designs"].items():
            for kind in ("iron", "time"):
                column = design[kind]
                if column["kind"] == "convex":
                    self.assertAlmostEqual(sum(column["weights"].values()), 1.0, places=12, msg=name)
                    for weight in column["weights"].values():
                        self.assertGreaterEqual(weight, 0.0, msg=name)
                        self.assertLessEqual(weight, 1.0, msg=name)


class SurfaceTests(unittest.TestCase):
    def test_secant_extension_is_an_upper_bound(self):
        low, high = Fraction(963366, 10000), Fraction(963526, 10000)
        bound = concavity_bound(low, high, Fraction(0), Fraction(1, 2), Fraction(1))
        self.assertEqual(bound, Fraction(963686, 10000))

    def test_platform_ceiling_is_below_target(self):
        bounds = platform_bounds()
        self.assertEqual(bounds["conditional_combined_B0"], "963679/10000")
        self.assertEqual(bounds["two_line_joint_ceiling"], "120499/1250")
        self.assertEqual(bounds["two_line_joint_ceiling_below_96_4"], "1/1250")
        self.assertLess(Fraction(bounds["two_line_joint_ceiling"]), Fraction(964, 10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
