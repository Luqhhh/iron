import numpy as np
import pandas as pd
import pytest
import torch

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.q75_hpo_screen import CONFIGS
from bf_tap_r2.q75_schedule_screen import ScheduledJointRegressor


def frame(rows=40, seed=0):
    rng = np.random.default_rng(seed)
    data = {name: rng.normal(size=rows) + 100 for name in FEATURES}
    data["sample_id"] = [f"S{index:05d}" for index in range(rows)]
    data["spout_no"] = rng.integers(1, 3, size=rows)
    data["tap_iron"] = 500 + 10 * rng.normal(size=rows)
    data["tap_time_len"] = 120 + 5 * rng.normal(size=rows)
    return pd.DataFrame(data)


def settings():
    return {"width": 16, "blocks": 1, "tabm_k": 2, "dropout": 0.1, "embedding_dim": 8,
            "n_frequencies": 4, "lite": True, "loss": "mse", "optimizer": "adamw",
            "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 16,
            "max_epochs": 2, "patience": 2, "min_delta": 1e-5, "inner_seed": 42,
            "random_seed": 42}


def test_config_table_is_single_factor_except_named_combinations():
    assert "CONTROL" in CONFIGS and "K32" in CONFIGS
    assert CONFIGS["K32"][0] == {"tabm_k": 32} and CONFIGS["K32"][1] == {}
    combination_names = {"COS_L1_LR3", "COS_L1_DROP0", "COS_L1_LR3_DROP0",
                          "K32_COS_MAE", "K32_COS_MAE_FREQ001", "K32_FREQ001",
                          "K32_LR3_COS_MAE", "K32_DROP0_COS_MAE", "K32_LR3_DROP0_COS_MAE"}
    for name, (overrides, recipe) in CONFIGS.items():
        assert set(overrides) <= {"dropout", "learning_rate", "weight_decay", "batch_size",
                                  "n_frequencies", "embedding_dim", "tabm_k", "blocks",
                                  "width", "lr_schedule", "loss", "inner_folds",
                                  "snapshot_radius", "head_loss", "mixup_alpha",
                                  "max_epochs", "patience"}, name
        assert set(recipe) <= {"frequency"}, name
        if name in combination_names or name == "CONTROL":
            continue
        if name == "K32":
            assert overrides == {"tabm_k": 32} and recipe == {}
        elif name.startswith("K32"):
            # neighbourhood arms keep tabm_k=32 and change one further factor
            # (K32_LONG bundles the epoch cap with its patience)
            assert overrides.get("tabm_k") == 32, name
            assert len(overrides) + len(recipe) in (2, 3), name
        else:
            assert len(overrides) + len(recipe) == 1, name


@pytest.mark.parametrize("name", ("K32", "FREQ001", "K32_COS_MAE"))
def test_selected_arms_train_and_predict(name):
    data = frame()
    overrides, recipe_overrides = CONFIGS[name]
    unit = settings()
    unit.update(overrides)
    recipe = {"backbone": "tabm", "frequency": 0.01}
    recipe.update(recipe_overrides)
    model = ScheduledJointRegressor(recipe, unit).fit(data, data[list(TARGETS)].to_numpy())
    prediction = model.predict(data.drop(columns=list(TARGETS)))
    assert prediction.shape == (len(data), 2) and np.isfinite(prediction).all()
    assert 1 <= model.metadata_["selected_epoch"] <= unit["max_epochs"]


def test_k32_changes_parameter_count():
    data = frame()
    torch.manual_seed(0)
    base = ScheduledJointRegressor({"backbone": "tabm", "frequency": None}, settings())
    base._initialize(data, data[list(TARGETS)].to_numpy())
    wide = ScheduledJointRegressor({"backbone": "tabm", "frequency": None},
                                   settings() | {"tabm_k": 32})
    wide._initialize(data, data[list(TARGETS)].to_numpy())
    base_parameters = sum(p.numel() for p in base.model_.parameters())
    wide_parameters = sum(p.numel() for p in wide.model_.parameters())
    assert wide_parameters > base_parameters
