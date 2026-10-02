import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.hard_tree_model import HardTreeRegressor
from bf_tap_r2.q75_feature_augmentation import (
    ENGINEERED, AugmentedJointRegressor, ExtendedPreprocessor, augment, engineered_columns)
from bf_tap_r2.v37_hard_tree import HardTreeEnsemble


def frame(rows=120, seed=3):
    rng = np.random.default_rng(seed)
    values = {name: np.abs(rng.normal(loc=100, scale=10, size=rows)) + 1 for name in FEATURES}
    data = pd.DataFrame(values)
    data["sample_id"] = [f"S{index:05d}" for index in range(rows)]
    data["spout_no"] = rng.integers(1, 3, size=rows)
    data["tap_iron"] = 500 + 20 * (data["air_volume"] - data["air_volume"].mean()) / 10 + rng.normal(0, 5, rows)
    data["tap_time_len"] = 120 + 8 * (data["oxygen"] - data["oxygen"].mean()) / 10 + rng.normal(0, 3, rows)
    return data


def tree_settings():
    spec = yaml.safe_load(open("configs/round2_v37/SPEC.yaml"))
    settings = {"model": dict(spec["model"]), "training": dict(spec["training"]), "inner_seed": 42}
    settings["model"].update({"n_estimators": 8, "depth": 3})
    settings["training"].update({"max_epochs": 4, "patience": 4, "batch_size": 32})
    return settings


def test_hard_tree_regressor_trains_and_predicts_cold():
    data = frame()
    settings = tree_settings()
    model = HardTreeRegressor("GLOBAL", settings).fit(data, data["tap_iron"].to_numpy())
    prediction = model.predict(data.drop(columns=list(TARGETS)))
    assert prediction.shape == (len(data),) and np.isfinite(prediction).all()
    assert 1 <= model.metadata_["selected_epoch"] <= 4
    assert model.metadata_["optimizer_runs"] == 2
    # row order must not matter for a forward-only model
    reversed_prediction = model.predict(data.drop(columns=list(TARGETS)).iloc[::-1])[::-1]
    np.testing.assert_allclose(prediction, reversed_prediction, rtol=0, atol=1e-9)


def test_hard_tree_input_uses_spout_one_hot_and_train_only_scaling():
    data = frame()
    model = HardTreeRegressor("GLOBAL", tree_settings())
    model._initialize(data, np.linspace(400.0, 600.0, len(data)))
    x = model._build(data)
    assert x.shape == (len(data), len(FEATURES) + 3)
    assert set(np.unique(x[:, len(FEATURES):])) <= {0.0, 1.0}
    assert np.allclose(x[:, len(FEATURES):].sum(1), 1.0)
    scaled = model._build_scaled(data)
    assert np.allclose(scaled[:, :len(FEATURES)].mean(0), 0.0, atol=1e-6)


def test_engineered_columns_are_deterministic_and_label_free():
    data = frame(rows=30)
    first, second = augment(data), augment(data.copy())
    assert len(ENGINEERED) == 23
    assert tuple(engineered_columns()) == ENGINEERED
    for name in ENGINEERED:
        assert name in first.columns
        np.testing.assert_array_equal(first[name].to_numpy(), second[name].to_numpy())
    # engineered values depend only on features, never on the targets
    changed = data.copy()
    changed["tap_iron"] = changed["tap_iron"] * 3
    np.testing.assert_array_equal(augment(changed)["log_oxygen"].to_numpy(),
                                  first["log_oxygen"].to_numpy())


def test_extended_preprocessor_uses_explicit_feature_list():
    data = augment(frame(rows=40))
    names = (*FEATURES, *ENGINEERED)
    pre = ExtendedPreprocessor(names).fit(data)
    assert pre.feature_names_ == names
    numeric, categorical = pre.transform_tabm(data)
    assert numeric.shape[1] == len(names)
    assert categorical.shape[0] == len(data)


@pytest.mark.parametrize("mixup", (0.0, 0.4))
def test_augmented_regressor_runs_with_and_without_mixup(mixup):
    data = augment(frame(rows=60))
    names = (*FEATURES, *ENGINEERED)
    settings = {"width": 16, "blocks": 1, "tabm_k": 2, "dropout": 0.1, "embedding_dim": 8,
                "n_frequencies": 4, "lite": True, "loss": "mse", "optimizer": "adamw",
                "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 16,
                "max_epochs": 3, "patience": 2, "min_delta": 1e-5, "inner_seed": 42,
                "random_seed": 42, "mixup_alpha": mixup}
    model = AugmentedJointRegressor({"backbone": "tabm", "frequency": None}, settings, names)
    model.fit(data, data[list(TARGETS)].to_numpy())
    prediction = model.predict(data.drop(columns=list(TARGETS)))
    assert prediction.shape == (len(data), 2) and np.isfinite(prediction).all()


def test_hard_tree_uses_the_optimised_routing_class():
    model = HardTreeRegressor("GLOBAL", tree_settings())
    model._initialize(frame(rows=20), np.arange(20, dtype=float))
    assert isinstance(model.model_, HardTreeEnsemble)
    torch.manual_seed(0)
    x = torch.randn(4, len(FEATURES) + 3)
    paths, weights = model.model_.routing_weights(x)
    assert torch.all((paths == 0) | (paths == 1))
    assert np.allclose(weights.sum(1).detach().numpy(), 1.0, atol=1e-6)
