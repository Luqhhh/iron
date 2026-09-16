import numpy as np
import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.optimization.burden_lag_v24 import FEATURE_COLUMNS, append_burden_lag, build_burden_lag, normalize_events
from bf_tap.optimization.dual_burden_v24 import CANDIDATE_A, CANDIDATE_B, compose_iron, compose_time, isolate


TZ = "Asia/Shanghai"


def events():
    return pd.DataFrame({
        "cal_time": ["2024-01-01 00:00", "2024-01-03 00:00", "2024-01-03 18:00", "2024-01-03 19:00", "2024-01-04 00:00"],
        "pig": [99, 24, 6, 0, 1], "all_quality": [99, 24, 6, 0, 1],
        "consumption": [99, 24, 6, 0, 1], "fuel_rate": [99, 24, 6, 0, 1], "coke_rate": [99, 24, 6, np.nan, 1],
    })


def samples():
    return pd.DataFrame({"sample_id": ["x"], "reference_time": pd.to_datetime(["2024-01-04 00:00"]).tz_localize(TZ)})


def test_registered_30_columns_and_window_equalities():
    out = build_burden_lag(samples(), events())
    assert list(out) == list(FEATURE_COLUMNS) and len(out.columns) == 30
    assert out.at[0, "burden_lag__pig__0_6h__mean"] == .5  # t and exactly t-5h
    assert out.at[0, "burden_lag__pig__0_6h__valid_count"] == 2
    assert out.at[0, "burden_lag__pig__6_24h__mean"] == 6  # exactly t-6h belongs here
    assert out.at[0, "burden_lag__pig__24_72h__mean"] == 24  # exactly t-24h belongs here
    assert out.at[0, "burden_lag__pig__24_72h__valid_count"] == 1  # exactly t-72h excluded
    assert out.at[0, "burden_lag__coke_rate__0_6h__valid_count"] == 1  # zero row is missing only for this field


def test_future_availability_empty_zero_and_no_fill():
    e = normalize_events(events())
    e.loc[e.event_time == pd.Timestamp("2024-01-04", tz=TZ), "available_at"] += pd.Timedelta(hours=1)
    out = build_burden_lag(samples(), e)
    assert out.at[0, "burden_lag__pig__0_6h__mean"] == 0
    assert out.at[0, "burden_lag__pig__0_6h__valid_count"] == 1
    early = samples(); early.reference_time = pd.Timestamp("2023-12-31", tz=TZ)
    empty = build_burden_lag(early, e)
    assert np.isnan(empty.filter(like="__mean").to_numpy()).all()
    assert (empty.filter(like="valid_count").to_numpy() == 0).all()


def test_duplicates_order_timezone_and_conflicts():
    e = events(); duplicate = pd.concat([e, e.iloc[[2]]], ignore_index=True).sample(frac=1, random_state=3)
    first = build_burden_lag(samples(), e); second = build_burden_lag(samples(), duplicate)
    pd.testing.assert_frame_equal(first, second)
    aware = e.copy(); aware.cal_time = pd.to_datetime(aware.cal_time).dt.tz_localize(TZ).dt.tz_convert("UTC")
    pd.testing.assert_frame_equal(first, build_burden_lag(samples(), aware))
    conflict = pd.concat([e, e.iloc[[2]].assign(pig=123)], ignore_index=True)
    with pytest.raises(ContractError, match="conflicting"): build_burden_lag(samples(), conflict)


def test_append_preserves_every_old_cell_dtype_and_order():
    old = pd.DataFrame({"spout_no": pd.Series(["1"], dtype="string"), "x": [2.0]})
    lag = build_burden_lag(samples(), events()); result = append_burden_lag(old, lag)
    pd.testing.assert_frame_equal(result.iloc[:, :2], old)
    assert list(result.columns[-30:]) == list(FEATURE_COLUMNS)


def test_A_rate_fallback_and_beta_identity():
    got, audit = compose_iron([10, 10], [20, 20], [0, 2], [100, 100], .25)
    assert got[0] == 12 and got[1] == 59 and audit["rate_unusable"] == 1


def test_B_uses_fixed_gate_and_six_decimal_qrf_roundtrip():
    value = compose_time([10.1234566, 20], np.array([True, False]), [14, 99])
    assert value[0] == .75 * 10.123457 + .25 * 14 and value[1] == 20


def test_target_isolation_and_unknown_candidate():
    parent = pd.DataFrame({"sample_id": ["x"], "pred_tap_iron": [1.], "pred_tap_time_len": [2.]})
    a = isolate(parent, iron=[3.], candidate=CANDIDATE_A); b = isolate(parent, time=[4.], candidate=CANDIDATE_B)
    assert a.pred_tap_time_len.tolist() == [2.] and b.pred_tap_iron.tolist() == [1.]
    with pytest.raises(ContractError): isolate(parent, iron=[3.], candidate="wrong")

