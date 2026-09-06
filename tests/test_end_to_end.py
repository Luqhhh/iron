import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")

from bf_tap.features import build_features
from bf_tap.models.baseline import FROZEN_PARAMETERS
from bf_tap.predict import predict_with_ids
from bf_tap.submission import validate_submission
from bf_tap.train import fit_baseline

TZ = "Asia/Shanghai"


@pytest.mark.model
def test_synthetic_feature_train_predict_contract():
    reference = pd.date_range("2024-01-02", periods=36, freq="2h", tz=TZ)
    samples = pd.DataFrame(
        {
            "sample_id": [f"S{i:03d}" for i in range(36)],
            "tap_no": np.arange(36),
            "spout_no": 1 + np.arange(36) % 2,
            "reference_time": reference,
            "tap_iron": 450 + np.arange(36) % 9,
            "tap_time_len": 100 + np.arange(36) % 6,
        }
    )
    operation = pd.DataFrame(
        {
            "event_time": pd.date_range("2024-01-01", periods=100, freq="h", tz=TZ),
            "available_at": pd.date_range("2024-01-01", periods=100, freq="h", tz=TZ),
            "air": np.linspace(1, 2, 100),
        }
    )
    burden = pd.DataFrame(
        {
            "event_time": pd.date_range("2024-01-01", periods=5, freq="D", tz=TZ),
            "available_at": pd.date_range("2024-01-01", periods=5, freq="D", tz=TZ),
            "pig": np.linspace(30, 34, 5),
        }
    )
    history = samples.copy()
    history["available_at"] = history["reference_time"] + pd.Timedelta(hours=1)
    cfg = {
        "categorical_missing": "__MISSING__",
        "operation": {"value_columns": ["air"], "windows_hours": [6, 24], "latest_max_event_age_hours": 24},
        "burden": {"value_columns": ["pig"], "latest_max_event_age_hours": 72},
        "history": {"last_k_mean": [3, 10]},
    }
    cutoff = reference[28]
    train_samples = samples.iloc[:28].copy()
    eval_samples = samples.iloc[28:].drop(columns=["tap_iron", "tap_time_len"]).copy()
    train_features = build_features(train_samples, operation=operation, burden=burden, history=history, fit_cutoff=cutoff, config=cfg)
    eval_features = build_features(eval_samples, operation=operation, burden=burden, history=history, fit_cutoff=cutoff, config=cfg)
    assert list(train_features.X.columns) == list(eval_features.X.columns)
    model = fit_baseline(train_samples, train_features.X, dict(FROZEN_PARAMETERS))
    result = predict_with_ids(model, eval_samples, eval_features.X)
    validate_submission(result, eval_samples.sample_id)
    assert len(result) == 8
