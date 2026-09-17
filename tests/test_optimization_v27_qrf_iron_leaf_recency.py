import numpy as np
import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.optimization.qrf_iron_leaf_recency_v27 import (
    CANDIDATE_A,
    CANDIDATE_B,
    PARENT,
    PRIMARY_AGGREGATION,
    POOLED_AGGREGATION,
    exposure_pooled_summary,
    isolate_iron,
    isolate_time,
    macro_origin_summary,
    recency60_weights,
    submission_bytes_preserving_other_target,
    verify_isolated_deltas,
)
from bf_tap.optimization.qrf_iron_leaf_recency_v27_run import registration


def scorecard():
    algorithms = [PARENT, CANDIDATE_A, CANDIDATE_B]
    units = [
        ("O1_H1", "O1", 1, 1, 10.0),
        ("O2_H1", "O2", 1, 9, 90.0),
        ("O3_H2", "O3", 2, 2, 20.0),
        ("O4_H3", "O4", 3, 3, 30.0),
        ("O5_H4", "O5", 4, 4, 40.0),
        ("DEV_LONG", "DEV", np.nan, 5, 50.0),
        ("DEV_SHORT", "DEV", np.nan, 6, 60.0),
    ]
    rows = []
    for algorithm in algorithms:
        for unit, cutoff, horizon, n, denominator in units:
            iron = 0.10 + (0.02 if algorithm == CANDIDATE_A else 0.0)
            time = 0.20 - (0.01 if algorithm == CANDIDATE_B else 0.0)
            e = 0.5 * (iron + time)
            for target, wmape in (("tap_iron", iron), ("tap_time_len", time)):
                rows.append({
                    "algorithm": algorithm,
                    "unit": unit,
                    "cutoff": cutoff,
                    "horizon": horizon,
                    "target": target,
                    "n": n,
                    "target_sum": denominator,
                    "absolute_error_sum": denominator * wmape,
                    "signed_error_sum": 0.0,
                    "signed_bias": 0.0,
                    "wmape": wmape,
                    "E": e,
                })
    return pd.DataFrame(rows)


def test_registration_freezes_two_experiments_and_budget():
    value = registration()
    assert value["candidates"] == {"A": CANDIDATE_A, "B": CANDIDATE_B}
    assert value["parent"] == PARENT
    assert value["budget"]["A_total_forest_fits"] == 7
    assert value["budget"]["B_forest_fits"] == 0
    assert value["budget"]["new_candidate_platform_tests"] == 2
    assert value["platform_feedback_may_change_second_candidate"] is False


def test_recency_weights_use_reference_age_and_half_life():
    day = 86400 * 10**9
    cutoff = 200 * day
    got = recency60_weights([cutoff, cutoff - 60 * day, cutoff - 120 * day], cutoff + 1)
    expected = np.exp2(-np.asarray([1 / day, 60 + 1 / day, 120 + 1 / day]) / 60)
    assert np.allclose(got, expected, rtol=0, atol=1e-15)
    with pytest.raises(ContractError, match="before cutoff"):
        recency60_weights([cutoff], cutoff)


def test_target_isolation_helpers():
    parent = pd.DataFrame({"sample_id": ["a", "b"], "pred_tap_iron": [1.0, 2.0], "pred_tap_time_len": [3.0, 4.0]})
    a = isolate_iron(parent, [5.1234566, 6.0])
    b = isolate_time(parent, [7.1234566, 8.0])
    assert a.pred_tap_iron.tolist() == [5.123457, 6.0]
    assert np.array_equal(a.pred_tap_time_len, parent.pred_tap_time_len)
    assert b.pred_tap_time_len.tolist() == [7.123457, 8.0]
    assert np.array_equal(b.pred_tap_iron, parent.pred_tap_iron)


@pytest.mark.parametrize("changed", ["tap_iron", "tap_time_len"])
def test_submission_preserves_other_parent_string(changed):
    parent = pd.DataFrame({
        "sample_id": ["a", "b"],
        "pred_tap_iron": ["1.230000", "2.000000"],
        "pred_tap_time_len": ["3.000000", "4.000000"],
    })
    payload = submission_bytes_preserving_other_target(parent, ["a", "b"], [5.1234566, 6], changed_target=changed)
    rows = payload.decode().splitlines()
    if changed == "tap_iron":
        assert rows[1:] == ["a,5.123457,3.000000", "b,6.000000,4.000000"]
    else:
        assert rows[1:] == ["a,1.230000,5.123457", "b,2.000000,6.000000"]


def test_macro_pooled_and_two_target_isolation():
    value = scorecard()
    macro = macro_origin_summary(value)
    pooled = exposure_pooled_summary(value)
    assert set(macro.aggregation) == {PRIMARY_AGGREGATION}
    assert set(pooled.aggregation) == {POOLED_AGGREGATION}
    audit = verify_isolated_deltas(value, macro, {CANDIDATE_A: "tap_iron", CANDIDATE_B: "tap_time_len"}, parent=PARENT)
    assert audit["checked_cells"] == 14
    assert audit["checked_macro_summaries"] == 24
    assert audit["maximum_absolute_delta_identity_residual"] <= 1e-12


def test_bad_parent_or_target_is_rejected():
    parent = pd.DataFrame({"sample_id": ["a"], "pred_tap_iron": ["1.000000"], "pred_tap_time_len": ["2.000000"]})
    with pytest.raises(ContractError, match="registered"):
        submission_bytes_preserving_other_target(parent, ["a"], [3], changed_target="bad")
    parent.loc[0, "pred_tap_time_len"] = "2"
    with pytest.raises(ContractError, match="six-decimal"):
        submission_bytes_preserving_other_target(parent, ["a"], [3], changed_target="tap_iron")
