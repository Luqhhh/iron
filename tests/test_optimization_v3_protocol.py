import numpy as np
import pandas as pd
import pytest
import json
import subprocess
import sys

from bf_tap.exceptions import ContractError
from bf_tap.optimization.alignment import calendar_horizon, stage_alignment
from bf_tap.optimization.calibration import fit_time_calibration, inner_blocks
from bf_tap.optimization.search_models import SingleTargetModel, search_configurations
from bf_tap.optimization.temporal_increment import segmented_operation


def ts(value):
    return pd.Timestamp(value, tz="Asia/Shanghai")


def test_calendar_mapping_preserves_h5():
    assert calendar_horizon(ts("2024-11-01"), ts("2024-12-01")) == 2
    assert calendar_horizon(ts("2024-12-01"), ts("2024-12-01")) == 1
    assert calendar_horizon(ts("2024-11-01"), ts("2025-03-01")) == 5
    result = stage_alignment(component="E09", stage="test_c", fit_cutoff=ts("2024-11-01"),
                             history_cutoff=ts("2024-11-01"), label_available_cutoff=ts("2024-11-01"),
                             reference_times=pd.Series([ts("2025-03-01")]),
                             identity=dict(source="s", data="d", candidate="c", component="e"),
                             development_cells={4: ["O202407_H4"]})
    assert result["uncovered_horizons"] == [5]
    with pytest.raises(ContractError):
        calendar_horizon(ts("2024-12-01"), ts("2024-11-30"))


def calibration_rows(n=100):
    return pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)],
                         "reference_time": ts("2024-08-10"), "label_available_at": ts("2024-08-11"),
                         "pred_tap_time_len": 102., "tap_time_len": 100.,
                         "prediction_fit_cutoff": ts("2024-08-04"), "history_cutoff": ts("2024-08-04")})


def calibrate(rows, **kwargs):
    return fit_time_calibration(rows, outer_cutoff=ts("2024-09-01"), candidate="E12-CVcal",
                                source_run="synthetic", origin_id="O202409", pipeline_identity="p",
                                outer_sample_ids=kwargs.get("outer_sample_ids", ["outer"]))


def test_calibration_zero_fallback_and_illegal_sources():
    assert calibrate(calibration_rows())["median_prediction_minus_actual"]["tap_time_len"] == 2
    assert calibrate(calibration_rows(99))["median_prediction_minus_actual"]["tap_time_len"] == 0
    for col, value in (("label_available_at", ts("2024-09-01")),
                       ("history_cutoff", ts("2024-08-05")),
                       ("prediction_fit_cutoff", ts("2024-08-05")),
                       ("reference_time", ts("2024-09-02"))):
        rows = calibration_rows()
        rows.loc[0, col] = value
        with pytest.raises(ContractError):
            calibrate(rows)
    with pytest.raises(ContractError):
        calibrate(calibration_rows(), outer_sample_ids=["s0"])
    with pytest.raises(ContractError):
        calibrate(pd.concat([calibration_rows(), calibration_rows()]))


def test_inner_selection_availability_and_disjoint_blocks():
    cutoff = ts("2024-09-01")
    first, second = cutoff - pd.Timedelta(days=56), cutoff - pd.Timedelta(days=28)
    rows = pd.DataFrame({"sample_id": ["a", "b", "c", "d", "e"],
                         "reference_time": [first - pd.Timedelta(days=1), first, second, second, cutoff],
                         "label_available_at": [first, second, cutoff - pd.Timedelta(seconds=1), cutoff, cutoff]})
    blocks = inner_blocks(rows, cutoff)
    assert list(blocks["selection_train"].sample_id) == ["a"]
    assert list(blocks["selection_eval"].sample_id) == ["b"]
    assert list(blocks["calibration_eval"].sample_id) == ["c"]
    assert not set(blocks["selection_eval"].sample_id) & set(blocks["calibration_eval"].sample_id)


def test_segments_boundaries_delays_duplicates_and_future_invariance():
    ref = ts("2024-09-01")
    events = pd.DataFrame({"event_time": [ref - pd.Timedelta(hours=h) for h in [6, 12, 24, 48, 7]],
                           "available_at": [ref] * 4 + [ref + pd.Timedelta(hours=1)],
                           "x": [6., 12., 24., 48., 999.]})
    samples = pd.DataFrame({"reference_time": [ref]})
    x = segmented_operation(samples, events, ["x"])
    assert x.iloc[0]["operation_segment__x__6_12h__mean"] == 6
    assert x.iloc[0]["operation_segment__x__12_24h__mean"] == 12
    assert x.iloc[0]["operation_segment__x__24_48h__mean"] == 24
    duplicate = pd.concat([events, events.iloc[:1]])
    pd.testing.assert_frame_equal(x, segmented_operation(samples, duplicate, ["x"]))
    events.loc[4, "x"] = -999
    pd.testing.assert_frame_equal(x, segmented_operation(samples, events, ["x"]))
    empty = segmented_operation(samples, events.iloc[:0], ["x"])
    assert empty.filter(like="count").to_numpy().sum() == 0
    assert empty.filter(like="mean").isna().all().all()


@pytest.mark.parametrize("model_type", ["tunable_catboost_l1", "lightgbm_l1"])
def test_model_inner_selection_and_bundle_roundtrip(tmp_path, model_type):
    config = next(c for c in search_configurations("configs/optimization_v0_3/models") if c["model_type"] == model_type)
    params = {**config["parameters"], **({"iterations": 8} if "catboost" in model_type else {"num_boost_round": 8, "min_data_in_leaf": 2})}
    X = pd.DataFrame({"spout_no": ["a", "b"] * 30, "x": np.arange(60, dtype=float)})
    y = pd.Series(np.arange(60, dtype=float) + 100)
    model = SingleTargetModel(model_type, params, "tap_time_len", patience=3)
    model.fit(X.iloc[:40], y.iloc[:40], inner_validation=(X.iloc[40:], y.iloc[40:]))
    assert 1 <= model.selected_iterations <= 8
    unknown = X.iloc[-4:].copy()
    unknown["spout_no"] = ["new", "a", None, "b"]
    identity = {k: "synthetic" for k in ("fit_cutoff", "history_cutoff", "label_available_cutoff", "train_samples_sha256", "inner_split", "source_sha256", "data_sha256")}
    model.save(tmp_path / "bundle", identity=identity)
    restored = SingleTargetModel.load(tmp_path / "bundle")
    np.testing.assert_array_equal(model.predict(unknown), restored.predict(unknown))
    np.testing.assert_array_equal(restored.predict(unknown), np.concatenate([restored.predict(unknown.iloc[:2]), restored.predict(unknown.iloc[2:])]))
    assert restored.mapping == {"spout_no": ["a", "b"]}
    result = subprocess.run([sys.executable, "-c",
                             "import json,sys,pandas as pd; from bf_tap.optimization.search_models import SingleTargetModel; "
                             "m=SingleTargetModel.load(sys.argv[1]); "
                             "print(json.dumps(m.predict(pd.DataFrame(json.loads(sys.argv[2]))).tolist()))",
                             str(tmp_path / "bundle"), json.dumps(unknown.to_dict("list"))],
                            check=True, capture_output=True, text=True)
    np.testing.assert_array_equal(model.predict(unknown), json.loads(result.stdout))
    with pytest.raises(ContractError):
        restored.predict(unknown[["x", "spout_no"]])


def test_fifteen_distinct_registered_configurations():
    from bf_tap.optimization.v3_config import validate_registration
    assert validate_registration()["calibration"]["minimum_samples"] == 100
    configs = search_configurations("configs/optimization_v0_3/models")
    assert len(configs) == len({c["id"] for c in configs}) == 15
    assert sum(c["model_type"] == "tunable_catboost_l1" for c in configs) == 9


def test_paired_calendar_week_resampling_keeps_repeated_samples_together():
    from bf_tap.optimization.v3_evidence import paired_week_intervals
    rows = []
    for candidate, error in (("new", 1.), ("ref", 2.)):
        for horizon in range(1, 5):
            for origin in ("o1", "o2"):
                for week in range(5):
                    rows.append(dict(candidate=candidate, origin=f"{origin}-{horizon}", horizon=horizon,
                                     sample_id=f"s{week}", week=f"2024-W{week:02d}", tap_iron=10., tap_time_len=10.,
                                     abs_error_tap_iron=error, abs_error_tap_time_len=error))
    frame = pd.DataFrame(rows)
    result = paired_week_intervals(frame, "new", "ref", repetitions=20)
    assert result["deltas"]["J"]["median"] == pytest.approx(-.1)
    assert result["deltas"]["J"]["p025"] == pytest.approx(-.1)
    frame.loc[0, "week"] = "wrong-week"
    with pytest.raises(ContractError):
        paired_week_intervals(frame, "new", "ref", repetitions=2)
