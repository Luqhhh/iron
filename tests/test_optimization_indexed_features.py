import numpy as np
import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.optimization.indexed_features import known_index_features, burden_event_changes, load_known_index


def ts(s):
    return pd.Timestamp(s, tz="Asia/Shanghai")


def index():
    return pd.DataFrame({"sample_id": ["a", "b", "c", "d", "e"], "spout_no": [1, 2, 1, 1, 2],
                         "reference_time": [ts("2024-06-01 00:00"), ts("2024-06-01 06:00"),
                                            ts("2024-06-01 07:00"), ts("2024-06-01 07:00"), ts("2024-06-02 07:00")]})


def test_known_index_strict_ties_boundaries_and_batch_invariance():
    meta = index()
    result = known_index_features(meta, meta)
    assert np.isnan(result.loc[0, "known_index__all__previous_interval_minutes"])
    assert result.loc[1, "known_index__all__6h__known_count"] == 0  # lower boundary excluded
    assert result.loc[2, "known_index__all__previous_interval_minutes"] == 60
    assert result.loc[2, "known_index__spout__previous_interval_minutes"] == 420
    assert result.loc[2, "known_index__all__6h__known_count"] == 1
    pd.testing.assert_series_equal(result.loc[2], result.loc[3], check_names=False)
    pd.testing.assert_frame_equal(result, known_index_features(meta, meta.sample(frac=1, random_state=1)))
    chunks = pd.concat([known_index_features(meta.iloc[:2], meta), known_index_features(meta.iloc[2:], meta)])
    pd.testing.assert_frame_equal(result, chunks)
    changed = meta.copy()
    changed.loc[4, "spout_no"] = 999
    pd.testing.assert_frame_equal(result.iloc[:4], known_index_features(meta.iloc[:4], changed))


def test_known_index_rejects_targets_closing_times_and_identity_conflicts(tmp_path):
    meta = index()
    for column in ("tap_iron", "tap_time_len", "tap_end_time", "ladle_count"):
        with pytest.raises(ContractError):
            known_index_features(meta, meta.assign(**{column: 123}))
    with pytest.raises(ContractError):
        known_index_features(meta, meta.iloc[:2])
    with pytest.raises(ContractError):
        known_index_features(meta, pd.concat([meta, meta.iloc[:1]]))
    path = tmp_path / "synthetic_index.csv"
    meta.assign(tap_iron="DO_NOT_CONVERT_OR_USE", tap_time_len="DO_NOT_USE").to_csv(path, index=False)
    loaded, audit = load_known_index([path])
    assert set(loaded) == {"sample_id", "reference_time", "spout_no"}
    assert audit["targets_or_closing_times_read"] is False
    pd.testing.assert_frame_equal(known_index_features(meta, loaded), known_index_features(meta, meta))
    with pytest.raises(ContractError):
        load_known_index([])


def test_burden_event_changes_visibility_missing_and_duplicate_policy():
    ref = ts("2024-06-04")
    events = pd.DataFrame({"event_time": [ref - pd.Timedelta(hours=h) for h in (72, 24, 1, .5)],
                           "available_at": [ref] * 3 + [ref + pd.Timedelta(hours=1)], "x": [1., 4., 7., 999.]})
    samples = pd.DataFrame({"reference_time": [ref]})
    result = burden_event_changes(samples, events, ["x"])
    assert result.iloc[0]["burden_event__x__previous_delta"] == 3
    assert result.iloc[0]["burden_event__24h__count"] == 1
    assert result.iloc[0]["burden_event__72h__count"] == 2
    duplicate = pd.concat([events, events.iloc[:1]])
    pd.testing.assert_frame_equal(result, burden_event_changes(samples, duplicate, ["x"]))
    changed = events.copy()
    changed.loc[3, "x"] = -999.
    pd.testing.assert_frame_equal(result, burden_event_changes(samples, changed, ["x"]))
    changed.loc[2, "x"] = np.nan
    assert np.isnan(burden_event_changes(samples, changed, ["x"]).iloc[0]["burden_event__x__previous_delta"])
    empty = burden_event_changes(samples, events.iloc[:0], ["x"])
    assert empty.filter(like="count").sum().sum() == 0
    assert empty.filter(like="delta").isna().all().all()
    with pytest.raises(ContractError):
        burden_event_changes(samples, pd.concat([events, events.iloc[:1].assign(x=2.)]), ["x"])
