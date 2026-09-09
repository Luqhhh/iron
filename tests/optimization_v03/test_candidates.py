from pathlib import Path

import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.optimization_v03.candidates import (
    derive_stage1_predictions,
    derive_stage2_prediction,
    recent_target_medians,
)
from bf_tap.optimization_v03.config import load_drift_experiment


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = load_drift_experiment(
    ROOT / "configs/optimization_v0_3/experiment.yaml"
)
TZ = "Asia/Shanghai"


def ts(value):
    return pd.Timestamp(value, tz=TZ)


def prediction(ids, iron, time):
    return pd.DataFrame(
        {
            "sample_id": ids,
            "pred_tap_iron": iron,
            "pred_tap_time_len": time,
        }
    )


def test_recent_medians_use_left_closed_window_and_ready_labels_only():
    cutoff = ts("2024-06-01")
    samples = pd.DataFrame(
        {
            "sample_id": ["left", "inside", "older", "late", "at_cutoff"],
            "reference_time": [
                cutoff - pd.Timedelta(days=30),
                cutoff - pd.Timedelta(days=1),
                cutoff - pd.Timedelta(days=30, seconds=1),
                cutoff - pd.Timedelta(days=2),
                cutoff,
            ],
            "label_available_at": [
                cutoff,
                cutoff - pd.Timedelta(hours=1),
                cutoff - pd.Timedelta(days=1),
                cutoff + pd.Timedelta(seconds=1),
                cutoff,
            ],
            "tap_iron": [10.0, 30.0, 999.0, 777.0, 555.0],
            "tap_time_len": [100.0, 140.0, 999.0, 777.0, 555.0],
        }
    )

    medians, audit = recent_target_medians(samples, cutoff, 30)

    assert medians == {"tap_iron": 20.0, "tap_time_len": 120.0}
    assert audit["eligible_rows"] == 2
    assert audit["window_start"] == str(cutoff - pd.Timedelta(days=30))
    assert audit["fit_cutoff"] == str(cutoff)
    assert audit["excluded_label_unavailable_rows"] == 1


def test_stage1_and_stage2_follow_preregistered_target_routes():
    eval_samples = pd.DataFrame({"sample_id": ["a", "b"]})
    e00 = prediction(["a", "b"], [100.0, 200.0], [1000.0, 2000.0])
    b1 = prediction(["a", "b"], [80.0, 160.0], [10.0, 20.0])
    e09 = prediction(["a", "b"], [70.0, 140.0], [30.0, 40.0])

    stage1 = derive_stage1_predictions(
        eval_samples,
        e00,
        b1,
        e09,
        {"tap_iron": 500.0, "tap_time_len": 120.0},
        EXPERIMENT,
    )

    assert stage1["M1_FROZEN"].to_dict("list") == {
        "sample_id": ["a", "b"],
        "pred_tap_iron": [90.0, 180.0],
        "pred_tap_time_len": [10.0, 20.0],
    }
    assert stage1["C1_RECENT30_TIME"].to_dict("list") == {
        "sample_id": ["a", "b"],
        "pred_tap_iron": [90.0, 180.0],
        "pred_tap_time_len": [120.0, 120.0],
    }
    assert stage1["C2_PROCESS_CHANGE"].equals(e09)

    stage2 = derive_stage2_prediction(
        stage1["C1_RECENT30_TIME"],
        stage1["C2_PROCESS_CHANGE"],
        EXPERIMENT,
    )
    assert stage2.to_dict("list") == {
        "sample_id": ["a", "b"],
        "pred_tap_iron": [70.0, 140.0],
        "pred_tap_time_len": [120.0, 120.0],
    }


@pytest.mark.parametrize(
    "bad",
    [
        prediction(["a", "c"], [70.0, 140.0], [30.0, 40.0]),
        prediction(["a", "a"], [70.0, 140.0], [30.0, 40.0]),
    ],
)
def test_prediction_key_mismatch_or_duplicates_are_rejected(bad):
    eval_samples = pd.DataFrame({"sample_id": ["a", "b"]})
    good = prediction(["a", "b"], [1.0, 2.0], [3.0, 4.0])

    with pytest.raises(ContractError, match="sample_id"):
        derive_stage1_predictions(
            eval_samples,
            good,
            good,
            bad,
            {"tap_iron": 10.0, "tap_time_len": 20.0},
            EXPERIMENT,
        )


def test_empty_recent_window_is_rejected():
    cutoff = ts("2024-06-01")
    samples = pd.DataFrame(
        {
            "sample_id": ["old"],
            "reference_time": [cutoff - pd.Timedelta(days=31)],
            "label_available_at": [cutoff - pd.Timedelta(days=1)],
            "tap_iron": [10.0],
            "tap_time_len": [20.0],
        }
    )

    with pytest.raises(ContractError, match="empty"):
        recent_target_medians(samples, cutoff, 30)
