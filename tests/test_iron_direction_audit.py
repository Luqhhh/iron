from pathlib import Path

from bf_tap_r2.slot_screen import read_json


def test_spec_is_a_zero_fit_closure_check():
    spec = read_json(Path("configs/iron_direction_audit/SPEC.json"))
    assert spec["new_fits"] == 0 and spec["model_loads"] == 0
    assert spec["new_packages"] == 0 and spec["agent_uploads"] == 0
    assert spec["desktop_writes"] == 0 and spec["training_label_reads"] == 0
    assert spec["rounding_levels"] == [4, 3, 2, 1, 0]
    assert spec["splits"] == [42, 3407] and spec["grid_points"] == 41
    assert spec["material_gain_threshold"] == 0.002


def test_audit_report_reproduces_the_closure():
    report = Path("local/runs/iron-direction-audit-20261005/review-r1/report.json")
    if not report.exists():
        import pytest
        pytest.skip("private audit evidence not present")
    saved = read_json(report)
    support = saved["duplicate_and_support"]
    assert support["within_train_duplicate_feature_rows"] == 0
    assert support["test_cells_outside_train_range"] == 0
    for digits, entry in support["rounding_levels"].items():
        assert entry["test_rows_matching_a_train_row"] == 0, digits
    directional = saved["directional_optimality"]
    assert directional["no_direction_improves_both_splits_materially"] is True
    assert directional["best_gain_min_over_splits"] < 0.002
