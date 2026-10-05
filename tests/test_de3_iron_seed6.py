from pathlib import Path

import numpy as np
import pytest

from bf_tap_r2.slot_screen import read_json


def test_spec_freezes_budget_and_distinct_seeds():
    spec = read_json(Path("configs/de3_iron_seed6/SPEC.json"))
    assert spec["new_seeds"] == [999983, 1000003, 1000033]
    assert not set(spec["new_seeds"]) & set(spec["existing_seeds"])
    assert spec["budget"]["new_estimators"] == 30 and spec["budget"]["new_optimizers"] == 60
    assert spec["budget"]["full_fits"] == 0 and spec["budget"]["packages"] == 0
    assert spec["budget"]["desktop_writes"] == 0 and spec["budget"]["agent_uploads"] == 0
    assert spec["automatic_retries"] is False and spec["time_budget_seconds"] is None


def test_expected_effect_basis_reproduces_the_one_over_r_fit():
    spec = read_json(Path("configs/de3_iron_seed6/SPEC.json"))
    basis = spec["expected_effect_basis"]
    A, c = basis["fit"]["A"], basis["fit"]["c"]
    observed = basis["observed"]
    for seeds, key in ((1, "r1"), (2, "r2"), (3, "r3")):
        assert A + c / seeds == pytest.approx(observed[key], abs=5e-6)
    raw = 50 * ((A + c / 3) - (A + c / 6))
    assert raw == pytest.approx(basis["predicted_r6_gain_raw_column"], abs=1e-4)
    assert 0.5 * raw == pytest.approx(basis["predicted_r6_gain_amplified"], abs=1e-4)


def test_recorded_folds_load_with_expected_sizes():
    spec = read_json(Path("configs/de3_iron_seed6/SPEC.json"))
    from bf_tap_r2.de3_iron_seed6 import fold_queries
    layout = fold_queries(spec)
    assert sorted(layout) == [42, 3407]
    for split in layout:
        sizes = [len(fold) for fold in layout[split]]
        assert sizes == [551, 551, 551, 551, 550]
        assert len({sample_id for fold in layout[split] for sample_id in fold}) == 2754


def test_release_spec_is_conditional_and_zero_cv():
    spec = read_json(Path("configs/de3_iron_seed6_release/SPEC.json"))
    assert spec["status"] == "frozen_conditional_on_development_gate"
    assert spec["budget"]["new_cv_fits"] == 0 and spec["budget"]["packages"] == 1
    assert spec["budget"]["desktop_writes"] == 0 and spec["budget"]["agent_uploads"] == 0
    assert spec["new_seeds"] == [999983, 1000003, 1000033]
    assert len(spec["inputs"]) == 4
    for entry in spec["inputs"].values():
        assert len(entry["sha256"]) == 64


def test_development_report_reproduces_the_gate_when_present():
    report = Path("local/runs/de3-iron-seed6-20261005/development-r1/report.json")
    if not report.exists():
        pytest.skip("development report not present")
    saved = read_json(report)
    evaluation = saved["evaluation"]
    assert set(evaluation["splits"]) == {"42", "3407"}
    for entry in evaluation["splits"].values():
        assert entry["rows"] == 2754
        assert np.isfinite(entry["score_gain"])
    assert evaluation["both_splits_positive"] == all(
        entry["score_gain"] > 0 for entry in evaluation["splits"].values())
    assert saved["budget_actual"]["new_estimators"] == 30
    assert saved["budget_actual"]["packages"] == 0
