import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.features.temporal import build_temporal_features


def t(value):
    return pd.Timestamp(value, tz="Asia/Shanghai")


def test_window_left_open_right_closed_staleness_and_future_perturbation():
    samples = pd.DataFrame({"sample_id": ["s"], "spout_no": [1], "reference_time": [t("2024-01-02 00:00")]})
    events = pd.DataFrame(
        {
            "event_time": [t("2024-01-01 18:00"), t("2024-01-02 00:00"), t("2024-01-02 00:01")],
            "available_at": [t("2024-01-01 18:00"), t("2024-01-02 00:00"), t("2024-01-02 00:01")],
            "v": [1.0, 3.0, 999.0],
        }
    )
    x, audit = build_temporal_features(samples, events, prefix="op", event_time="event_time", available_at="available_at", value_columns=["v"], stale_hours=24, windows_hours=[6])
    assert x.loc[0, "op__v__latest"] == 3
    assert x.loc[0, "op__v__6h__mean"] == 3  # 18:00 is excluded at the left boundary
    assert audit.loc[0, "max_available_at"] == t("2024-01-02 00:00")
    events.loc[2, "v"] = -123456
    changed, _ = build_temporal_features(samples, events, prefix="op", event_time="event_time", available_at="available_at", value_columns=["v"], stale_hours=24, windows_hours=[6])
    pd.testing.assert_frame_equal(x, changed)


def test_latest_value_is_missing_beyond_stale_boundary_but_age_remains():
    samples = pd.DataFrame({"reference_time": [t("2024-01-02 00:01")]})
    events = pd.DataFrame({"event_time": [t("2024-01-01")], "available_at": [t("2024-01-01")], "v": [7.0]})
    x, _ = build_temporal_features(samples, events, prefix="op", event_time="event_time", available_at="available_at", value_columns=["v"], stale_hours=24)
    assert pd.isna(x.loc[0, "op__v__latest"])
    assert x.loc[0, "op__stale"] == 1
    assert x.loc[0, "op__event_age_hours"] > 24


def test_conflicting_same_timestamp_fails_closed():
    samples = pd.DataFrame({"reference_time": [t("2024-01-01 01:00")]})
    events = pd.DataFrame(
        {
            "event_time": [t("2024-01-01"), t("2024-01-01")],
            "available_at": [t("2024-01-01"), t("2024-01-01")],
            "v": [1.0, 2.0],
        }
    )
    with pytest.raises(ContractError, match="conflicting"):
        build_temporal_features(samples, events, prefix="op", event_time="event_time", available_at="available_at", value_columns=["v"], stale_hours=24)


def test_process_source_after_fit_origin_but_before_sample_is_visible_and_order_invariant():
    samples = pd.DataFrame({"reference_time": [t("2024-02-01 12:00")]})
    events = pd.DataFrame(
        {
            "event_time": [t("2024-01-31"), t("2024-02-01 11:00")],
            "available_at": [t("2024-01-31"), t("2024-02-01 11:30")],
            "v": [1.0, 2.0],
        }
    )
    first, _ = build_temporal_features(samples, events, prefix="op", event_time="event_time", available_at="available_at", value_columns=["v"], stale_hours=24)
    shuffled, _ = build_temporal_features(samples, events.sample(frac=1, random_state=3), prefix="op", event_time="event_time", available_at="available_at", value_columns=["v"], stale_hours=24)
    assert first.loc[0, "op__v__latest"] == 2
    pd.testing.assert_frame_equal(first, shuffled)


def test_exact_stale_boundary_remains_available():
    samples = pd.DataFrame({"reference_time": [t("2024-01-02")]})
    events = pd.DataFrame({"event_time": [t("2024-01-01")], "available_at": [t("2024-01-01")], "v": [7.0]})
    x, _ = build_temporal_features(samples, events, prefix="op", event_time="event_time", available_at="available_at", value_columns=["v"], stale_hours=24)
    assert x.loc[0, "op__v__latest"] == 7
    assert x.loc[0, "op__stale"] == 0
