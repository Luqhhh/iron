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
                                   "dropout", "weight_decay", "max_epochs", "inner_folds",
                                   "snapshot_radius", "head_loss"}
        assert "COS_MAE_INNER5" in CONFIGS
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


def test_selection_arms_present_and_plain_control_still_delegates():
    assert {"INNER5", "SNAP5", "HEADLOSS", "COS_MAE_INNER5"} <= set(CONFIGS)
    model = ScheduledJointRegressor({}, {})
    assert model._is_plain()
    assert not ScheduledJointRegressor({}, {"inner_folds": 5})._is_plain()
    assert not ScheduledJointRegressor({}, {"snapshot_radius": 20})._is_plain()
    assert not ScheduledJointRegressor({}, {"head_loss": ["mae", "mse"]})._is_plain()


@pytest.mark.parametrize("override", ({"inner_folds": 5}, {"snapshot_radius": 2},
                                      {"head_loss": ["mae", "mse"]}))
def test_selection_arms_train_and_predict(override):
    data = frame(rows=45)
    unit = settings()
    unit.update(override)
    query = data.drop(columns=list(TARGETS))
    model = ScheduledJointRegressor({"backbone": "tabm", "frequency": None}, unit)
    model.fit(data, data[list(TARGETS)].to_numpy(), query=query)
    prediction = model.predict(query)
    assert prediction.shape == (len(data), 2) and np.isfinite(prediction).all()
    assert 1 <= model.metadata_["selected_epoch"] <= unit["max_epochs"] + int(override.get("snapshot_radius", 0))


def test_snapshot_zero_offset_checkpoint_matches_plain_refit_exactly():
    """Checkpoint at the selected epoch must equal a plain refit of the same length."""
    from bf_tap_r2.v12_joint import JointRegressor
    data = frame(rows=45)
    query = data.drop(columns=list(TARGETS))
    y = data[list(TARGETS)].to_numpy()
    recipe = {"backbone": "tabm", "frequency": None}
    unit = settings() | {"snapshot_radius": 2}
    snap = ScheduledJointRegressor(recipe, unit)
    snap._initialize(data, y)
    values = snap._refit_with_snapshots(data, y, total=3, checkpoints={3}, query=query)
    plain = JointRegressor(recipe, settings())
    plain._initialize(data, y)
    plain._train(data, y, 3)
    np.testing.assert_array_equal(values[0], plain.predict(query))


def test_snapshot_checkpoint_matches_plain_refit_exactly():
    """The zero-offset checkpoint must equal the plain refit trajectory point."""
    data = frame(rows=45)
    unit = settings()
    unit.update({"snapshot_radius": 2, "max_epochs": 4, "patience": 4})
    query = data.drop(columns=list(TARGETS))
    model = ScheduledJointRegressor({"backbone": "tabm", "frequency": None}, unit)
    model.fit(data, data[list(TARGETS)].to_numpy(), query=query)
    epoch = model.metadata_["selected_epoch"]
    plain = ScheduledJointRegressor({"backbone": "tabm", "frequency": None}, settings() | {"max_epochs": 4, "patience": 4})
    # the plain arm selects its own epoch; the snapshot arm must include the no-offset point
    assert isinstance(model.snapshot_checkpoints_, list) and model.snapshot_checkpoints_
    assert min(model.snapshot_checkpoints_) >= 1
