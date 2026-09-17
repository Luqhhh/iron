import pandas as pd
import numpy as np
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.optimization.qrf_partition_v26 import (
    CANDIDATE_A,
    CANDIDATE_B,
    PARENT,
    PRIMARY_AGGREGATION,
    POOLED_AGGREGATION,
    exposure_pooled_summary,
    isolate_time,
    macro_origin_summary,
    roundtrip_six,
    submission_bytes_preserving_iron,
    verify_isolated_deltas,
)


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
            iron = 0.10
            time = 0.20
            if algorithm == CANDIDATE_A:
                time += 0.02
            if algorithm == CANDIDATE_B:
                time -= 0.01
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


def test_macro_and_pooled_are_explicitly_separate():
    value = scorecard()
    macro = macro_origin_summary(value)
    pooled = exposure_pooled_summary(value)
    assert set(macro.aggregation) == {PRIMARY_AGGREGATION}
    assert set(pooled.aggregation) == {POOLED_AGGREGATION}
    parent_macro = macro[(macro.algorithm == PARENT) & (macro.scope == "H1")].iloc[0]
    parent_pooled = pooled[(pooled.algorithm == PARENT) & (pooled.scope == "H1")].iloc[0]
    assert parent_macro.n_origins == parent_pooled.n_origins == 2
    assert parent_macro.wmape_time == pytest.approx(0.2)
    assert parent_pooled.wmape_time == pytest.approx(0.2)


def test_exposure_pooling_differs_when_cell_wmapes_differ():
    value = scorecard()
    mask = (value.algorithm == PARENT) & (value.unit == "O1_H1") & (value.target == "tap_time_len")
    value.loc[mask, "wmape"] = 0.4
    value.loc[mask, "absolute_error_sum"] = value.loc[mask, "target_sum"] * 0.4
    unit_mask = (value.algorithm == PARENT) & (value.unit == "O1_H1")
    value.loc[unit_mask, "E"] = 0.25
    macro = macro_origin_summary(value)
    pooled = exposure_pooled_summary(value)
    assert macro.query("algorithm == @PARENT and scope == 'H1'").wmape_time.iloc[0] == pytest.approx(0.3)
    assert pooled.query("algorithm == @PARENT and scope == 'H1'").wmape_time.iloc[0] == pytest.approx(0.22)


def test_isolated_delta_identity_covers_cells_horizons_j_and_dev():
    value = scorecard()
    summary = macro_origin_summary(value)
    audit = verify_isolated_deltas(value, summary, {CANDIDATE_A: "tap_time_len", CANDIDATE_B: "tap_time_len"})
    assert audit["checked_cells"] == 14
    assert audit["checked_macro_summaries"] == 24
    assert audit["maximum_absolute_delta_identity_residual"] <= 1e-12


def test_j_is_four_horizon_mean_and_excludes_dev():
    summary = macro_origin_summary(scorecard())
    part = summary[(summary.algorithm == PARENT) & summary.scope.isin(["H1", "H2", "H3", "H4"])]
    j = summary[(summary.algorithm == PARENT) & (summary.scope == "J")].iloc[0]
    assert j.E == pytest.approx(part.E.mean())
    assert j.scope_type == "GRID"


def test_roundtrip_and_time_isolation():
    parent = pd.DataFrame({"sample_id": ["a", "b"], "pred_tap_iron": [1.0, 2.0], "pred_tap_time_len": [3.0, 4.0]})
    got = isolate_time(parent, [5.1234566, 6.0])
    assert got.pred_tap_time_len.tolist() == [5.123457, 6.0]
    assert np.array_equal(got.pred_tap_iron, parent.pred_tap_iron)
    with pytest.raises(ContractError, match="nonnegative"):
        roundtrip_six([-1])


def test_submission_copies_parent_iron_strings_verbatim():
    parent = pd.DataFrame({
        "sample_id": ["a", "b"],
        "pred_tap_iron": ["001.230000", "2.000000"],
        "pred_tap_time_len": ["3.000000", "4.000000"],
    })
    with pytest.raises(ContractError, match="six-decimal"):
        submission_bytes_preserving_iron(parent, ["a", "b"], [5, 6])
    parent.loc[0, "pred_tap_iron"] = "1.230000"
    payload = submission_bytes_preserving_iron(parent, ["a", "b"], [5.1234566, 6])
    assert payload.decode().splitlines() == [
        "sample_id,pred_tap_iron,pred_tap_time_len",
        "a,1.230000,5.123457",
        "b,2.000000,6.000000",
    ]


def test_scorecard_rejects_mixed_e_or_bad_denominator_math():
    value = scorecard()
    value.loc[0, "E"] += 0.1
    with pytest.raises(ContractError, match="cell E"):
        macro_origin_summary(value)
    value = scorecard()
    value.loc[0, "absolute_error_sum"] += 1
    with pytest.raises(ContractError, match="numerator"):
        exposure_pooled_summary(value)
