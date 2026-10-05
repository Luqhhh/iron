from pathlib import Path

import pytest

from bf_tap_r2.slot_screen import read_json


def test_release_spec_freezes_budget_and_weights():
    spec = read_json(Path("configs/ibatch_release/SPEC.json"))
    assert spec["status"] == "frozen_before_execution"
    assert spec["time_weight"] == 0.2 and spec["iron_weight"] == 0.1
    assert spec["time_seeds"] == [42, 1042, 2042]
    assert spec["iron_seeds"] == [42, 104729, 130363]
    assert spec["budget"]["new_full_fits"] == 6 and spec["budget"]["new_optimizers"] == 12
    assert spec["budget"]["new_cv_fits"] == 0 and spec["budget"]["new_confirmation_seeds"] == 0
    assert spec["budget"]["packages"] == 2 and spec["budget"]["desktop_writes"] == 0
    assert spec["budget"]["agent_uploads"] == 0
    assert len(spec["inputs"]) == 8
    for entry in spec["inputs"].values():
        assert len(entry["sha256"]) == 64


def test_reports_and_audits_are_consistent_when_present():
    run = Path("local/runs/ibatch-release-20261005/release-r1")
    if not (run / "report-IBATCH_TIME_A20.json").exists():
        pytest.skip("private release evidence not present")
    for name in ("IBATCH_TIME_A20", "JOINT_IBATCH_IRON_A20"):
        report = read_json(run / f"report-{name}.json")
        audit = read_json(run / f"independent-audit-{name}.json")
        assert report["zip_sha256"] == audit["checks"]["zip_sha256"]
        assert audit["passed"] is True
        assert audit["checks"]["rows"] == 322 and audit["checks"]["unique_ids"] == 322
        assert audit["checks"]["official_order"] and audit["checks"]["finite_nonnegative"]
        assert audit["checks"]["unchanged_column_verbatim"] is True
        assert report["pre_upload_screen"]["rho"] < 0.01
        assert report["pre_upload_screen"]["slot_rule_passed"] is False  # slope is slightly positive
    delivery = read_json(run / "desktop-delivery.json")
    assert delivery["agent_uploads"] == 0
    for entry in delivery["copies"].values():
        assert entry["bytes_equal"] is True and entry["crc"] is True
        assert entry["rows"] == 322 and entry["official_order"] is True
