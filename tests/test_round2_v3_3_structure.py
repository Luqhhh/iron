from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.v3_1_models import V31Regressor
from bf_tap_r2.v3_3_residual import FullRecipeResidualRegressor


def synthetic(n=80):
    rng = np.random.default_rng(33)
    frame = pd.DataFrame({name: np.abs(rng.normal(size=n)) + 0.2 for name in FEATURES})
    frame["spout_no"] = np.tile([1, 2, 3, 4], n // 4)
    frame["sample_id"] = [f"R2S_TRAIN_{i:03X}" for i in range(n)]
    frame["tap_iron"] = 500 + 20 * frame.air_volume + 3 * frame.oxygen + rng.normal(0, 2, n)
    frame["tap_time_len"] = 120 + 10 * frame.air_volume + rng.normal(0, 1, n)
    return frame


def base_trial(feature_set="raw", target_transform="identity"):
    return {
        "family": "catboost",
        "target": "tap_iron",
        "feature_set": feature_set,
        "target_transform": target_transform,
        "parameters": {
            "task_type": "CPU", "loss_function": "RMSE", "depth": 2, "iterations": 25,
            "learning_rate": 0.10, "l2_leaf_reg": 3.0, "random_seed": 42,
            "thread_count": 1, "cat_features": ["spout_no"], "allow_writing_files": False,
            "verbose": False, "bootstrap_type": "MVS", "subsample": 0.8,
        },
    }


def residual_trial(alpha=0.25, feature_set="raw", target_transform="identity"):
    return {
        "family": "full_residual",
        "target": "tap_iron",
        "parameters": {
            "base_trial": base_trial(feature_set, target_transform),
            "residual": {
                "kind": "ridge",
                "params": {"alpha": 10.0},
            },
            "alpha": alpha,
            "inner_seed": 123,
            "inner_splits": 3,
        },
    }


@pytest.mark.parametrize("feature_set,target_transform", [("raw", "identity"), ("four", "log1p")])
def test_v33_alpha_zero_equals_parent_recipe(feature_set, target_transform):
    frame = synthetic()
    trial = residual_trial(alpha=0.0, feature_set=feature_set, target_transform=target_transform)
    wrapper = FullRecipeResidualRegressor(trial)
    wrapper.fit(frame, frame["tap_iron"].to_numpy())
    parent = V31Regressor(trial["parameters"]["base_trial"])
    parent.fit(frame, frame["tap_iron"].to_numpy())
    np.testing.assert_allclose(wrapper.predict(frame), parent.predict(frame), rtol=1e-12, atol=1e-10)


def test_v33_signed_residual_can_move_both_directions_and_stays_finite():
    frame = synthetic()
    trial = residual_trial(alpha=0.5)
    model = FullRecipeResidualRegressor(trial).fit(frame, frame["tap_iron"].to_numpy())
    pred = model.predict(frame)
    assert pred.shape == (len(frame),)
    assert np.isfinite(pred).all()
    # The residual head must be allowed to take negative residual labels.
    residual = frame["tap_iron"].to_numpy() - model.base_model_.predict(frame)
    assert residual.min() < 0 < residual.max()


def test_v33_rejects_incomplete_base_trial():
    trial = residual_trial()
    del trial["parameters"]["base_trial"]
    with pytest.raises(ValueError, match="base_trial"):
        FullRecipeResidualRegressor(trial)
