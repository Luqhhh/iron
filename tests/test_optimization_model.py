import numpy as np
import pandas as pd
import pytest

from bf_tap.models.baseline import DualTargetBaseline, FROZEN_PARAMETERS
from bf_tap.optimization.models import FrozenBaselineAdapter


@pytest.mark.model
def test_e00_adapter_exactly_matches_frozen_baseline_raw_predictions():
    rows = 32
    X = pd.DataFrame(
        {
            "spout_no": pd.Series(1 + np.arange(rows) % 2, dtype="string"),
            "x": np.linspace(0, 1, rows),
        }
    )
    y = pd.DataFrame(
        {
            "tap_iron": 400 + np.arange(rows) % 5,
            "tap_time_len": 80 + np.arange(rows) % 7,
        }
    )
    baseline = DualTargetBaseline(dict(FROZEN_PARAMETERS)).fit(X, y)
    adapter = FrozenBaselineAdapter(dict(FROZEN_PARAMETERS)).fit(X, y)
    pd.testing.assert_frame_equal(
        baseline.predict_raw(X), adapter.predict_raw(X), check_exact=True
    )


@pytest.mark.model
def test_adapter_accepts_positive_aligned_sample_weights():
    rows = 32
    X = pd.DataFrame(
        {
            "spout_no": pd.Series(1 + np.arange(rows) % 2, dtype="string"),
            "x": np.linspace(0, 1, rows),
        }
    )
    y = pd.DataFrame(
        {
            "tap_iron": 400 + np.arange(rows) % 5,
            "tap_time_len": 80 + np.arange(rows) % 7,
        }
    )
    model = FrozenBaselineAdapter(dict(FROZEN_PARAMETERS)).fit(
        X, y, sample_weight=pd.Series(np.full(rows, 0.2))
    )
    prediction = model.predict_raw(X)
    assert prediction.shape == (rows, 2)
    assert np.isfinite(prediction.to_numpy()).all()
