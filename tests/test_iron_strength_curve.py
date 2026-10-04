from pathlib import Path

import numpy as np
import pytest

from bf_tap_r2.iron_strength_curve import wmape
from bf_tap_r2.slot_screen import read_json


def test_wmape_matches_hand_value():
    assert wmape(np.array([10.0, 20.0]), np.array([9.0, 22.0])) == pytest.approx(3.0 / 30.0)


def test_spec_freezes_a_zero_fit_review():
    spec = read_json(Path("configs/iron_strength_curve/SPEC.json"))
    assert spec["new_fits"] == 0 and spec["new_packages"] == 0
    assert spec["agent_uploads"] == 0 and spec["desktop_writes"] == 0
    assert spec["splits"] == [42, 3407] and spec["folds"] == 5
    assert 0.5 in spec["amplification_grid"] and 1.0 in spec["amplification_grid"]
    assert len(spec["seed_keys"]) == 3


def test_release_spec_freezes_budget_and_identity_gate():
    spec = read_json(Path("configs/iron_strength_release/SPEC.json"))
    assert spec["status"].startswith("frozen_")
    assert spec["budget"]["new_scientific_fits"] == 0
    assert spec["budget"]["new_optimizers"] == 0
    assert spec["budget"]["packages"] == 1
    assert spec["budget"]["desktop_writes"] == 0 and spec["budget"]["agent_uploads"] == 0
    assert spec["identity_gate"]["reconstruct_q0.5_equals_parent_iron_strings"] is True
    assert spec["identity_gate"]["time_column_strings_unchanged"] is True
    assert spec["candidate"] == "DE3_IRON_STRENGTH_100"
    assert spec["classification"].startswith("exploration")


def test_curve_report_reproduces_the_frozen_findings():
    report = Path("local/runs/iron-strength-curve-20261005/review-r1/report.json")
    if not report.exists():
        pytest.skip("private review report not present")
    saved = read_json(report)
    curve = saved["amplification_curve"]
    assert curve["pooled_best_q"] == 1.0
    assert curve["folds_improved_0.5_to_1.0"] == 8 and curve["folds_total"] == 10
    gains = curve["score_gain_vs_0.5"]
    assert gains["1.0"] == pytest.approx(0.003000793063932458)
    assert gains["0.0"] < 0 and gains["0.25"] < 0
    assert saved["seed_averaging_curve"]["pooled_score_gain"]["2_to_3"] > 0


def test_release_spec_records_the_pre_execution_amendment():
    spec = read_json(Path("configs/iron_strength_release/SPEC.json"))
    assert spec["status"] == "frozen_amended_before_execution"
    assert spec["amendment"]["before_any_package_written"] is True
    assert spec["budget"]["cold_inferences"] == 0 and spec["budget"]["model_loads"] == 0
    assert len(spec["inputs"]) == 5
    for entry in spec["inputs"].values():
        assert len(entry["sha256"]) == 64 and entry["path"]


def test_release_report_and_audit_are_consistent():
    base = Path("local/runs/iron-strength-release-20261005/release-r1")
    report, audit = base / "report.json", base / "independent-audit.json"
    if not report.exists():
        pytest.skip("private release evidence not present")
    saved, checked = read_json(report), read_json(audit)
    assert saved["candidate"] == "DE3_IRON_STRENGTH_100"
    assert saved["rows"] == 322 and saved["unique_ids"] == 322
    assert saved["verification"]["reconstructed_q05_matches_incumbent"] == 0.0
    assert saved["verification"]["derived_mean3_matches_recorded_release_ensemble"] < 1e-9
    assert saved["pre_upload_screen"]["slot_rule_passed"] is True
    assert saved["pre_upload_screen"]["fitted_functional"]["predicted_sign"] == "positive"
    assert saved["budget_actual"] == {"new_fits": 0, "new_optimizers": 0, "model_loads": 0,
                                      "cold_inferences": 0, "packages": 1, "desktop_writes": 0,
                                      "agent_uploads": 0}
    assert checked["passed"] is True and checked["checks"]["zip_sha256"] == saved["zip_sha256"]
