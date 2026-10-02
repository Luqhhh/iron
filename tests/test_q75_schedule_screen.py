import numpy as np
import pandas as pd
import pytest
import torch

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.q75_schedule_screen import CONFIGS, SCHEDULES, ScheduledJointRegressor
from bf_tap_r2.v12_joint import JointRegressor


def frame(rows=60, seed=0):
    rng = np.random.default_rng(seed)
    data = {name: rng.normal(size=rows) for name in FEATURES}
    data["sample_id"] = [f"S{index:05d}" for index in range(rows)]
    data["spout_no"] = rng.integers(1, 3, size=rows)
    data["tap_iron"] = 500 + 40 * data["air_volume"] + rng.normal(scale=5, size=rows)
    data["tap_time_len"] = 120 + 10 * data["oxygen"] + rng.normal(scale=3, size=rows)
    return pd.DataFrame(data)


def settings():
    return {"width": 16, "blocks": 1, "tabm_k": 2, "dropout": 0.1, "embedding_dim": 8,
            "n_frequencies": 4, "lite": True, "loss": "mse", "optimizer": "adamw",
            "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 16,
            "max_epochs": 3, "patience": 2, "min_delta": 1e-5, "inner_seed": 42,
            "random_seed": 42}


def test_configuration_table_is_declared_and_schedules_known():
    assert set(CONFIGS) >= {"CONTROL", "COS", "COS_MAE"}
    assert SCHEDULES == ("constant", "cosine")
    for name, override in CONFIGS.items():
        unknown = set(override) - {"lr_schedule", "loss", "learning_rate", "warmup_epochs",
                                   "dropout", "weight_decay", "max_epochs"}
        assert not unknown, (name, unknown)


def test_control_override_delegates_to_parent_training():
    data = frame()
    recipe = {"backbone": "tabm", "frequency": 0.01}
    torch.manual_seed(0)
    parent = JointRegressor(recipe, settings()).fit(data, data[list(TARGETS)].to_numpy())
    torch.manual_seed(0)
    child = ScheduledJointRegressor(recipe, settings()).fit(data, data[list(TARGETS)].to_numpy())
    np.testing.assert_array_equal(parent.predict(data.drop(columns=list(TARGETS))),
                                  child.predict(data.drop(columns=list(TARGETS))))


def test_cosine_factor_shape_and_warmup():
    model = ScheduledJointRegressor({}, {"lr_schedule": "cosine", "warmup_epochs": 2})
    assert model._lr_factor(1, 10) == pytest.approx(1 / 3)
    assert model._lr_factor(2, 10) == pytest.approx(2 / 3)
    assert model._lr_factor(3, 10) == pytest.approx(0.9619397662556434)
    assert model._lr_factor(6, 10) == pytest.approx(0.5, abs=1e-12)
    assert model._lr_factor(10, 10) == pytest.approx(0.0, abs=1e-12)
    factors = [model._lr_factor(epoch, 10) for epoch in range(1, 11)]
    assert factors[:3] == pytest.approx([1 / 3, 2 / 3, 0.9619397662556434])
    assert all(b <= a for a, b in zip(factors[2:], factors[3:]))  # decays after warmup
    assert ScheduledJointRegressor({}, {})._lr_factor(5, 10) == 1.0
    assert ScheduledJointRegressor({}, {"lr_schedule": "cosine"})._lr_factor(10, 10) == 0.0


@pytest.mark.parametrize("override", ({"lr_schedule": "cosine"}, {"loss": "mae"},
                                      {"lr_schedule": "cosine", "loss": "mae"}))
def test_scheduled_training_produces_finite_predictions(override):
    data = frame(rows=40)
    unit = settings()
    unit.update(override)
    model = ScheduledJointRegressor({"backbone": "tabm", "frequency": None}, unit).fit(
        data, data[list(TARGETS)].to_numpy())
    prediction = model.predict(data.drop(columns=list(TARGETS)))
    assert prediction.shape == (len(data), 2) and np.isfinite(prediction).all()
    assert 1 <= model.metadata_["selected_epoch"] <= unit["max_epochs"]


def test_unknown_schedule_and_loss_are_rejected():
    data = frame(rows=30)
    for override in ({"lr_schedule": "linear"}, {"loss": "huber"}):
        unit = settings()
        unit.update(override)
        model = ScheduledJointRegressor({"backbone": "tabm", "frequency": None}, unit)
        with pytest.raises(ValueError):
            model.fit(data, data[list(TARGETS)].to_numpy())
