from importlib import import_module
from importlib.util import find_spec

import pandas as pd
import pytest

from bf_tap.exceptions import ContractError


def _blend_module():
    module = find_spec("bf_tap.models.optimization_v02.m1_blend.model")
    assert module is not None, "M1 blend model must exist"
    return import_module("bf_tap.models.optimization_v02.m1_blend.model")


def _oof_frames():
    actual = pd.DataFrame(
        {
            "fold_id": ["A", "B"],
            "sample_id": ["s1", "s2"],
            "tap_iron": [100.0, 100.0],
            "tap_time_len": [50.0, 50.0],
        }
    )
    catboost = pd.DataFrame(
        {
            "fold_id": ["A", "B"],
            "sample_id": ["s1", "s2"],
            "pred_tap_iron": [100.0, 100.0],
            "pred_tap_time_len": [60.0, 60.0],
        }
    )
    b1 = pd.DataFrame(
        {
            "fold_id": ["A", "B"],
            "sample_id": ["s1", "s2"],
            "pred_tap_iron": [120.0, 120.0],
            "pred_tap_time_len": [50.0, 50.0],
        }
    )
    return actual, catboost, b1


def test_selects_independent_target_weights_from_pooled_oof():
    blend = _blend_module()
    actual, catboost, b1 = _oof_frames()

    selected = blend.select_blend_weights(
        actual,
        catboost,
        b1,
        weights=[0.0, 0.5, 1.0],
        tie_tolerance=1e-12,
    )

    assert selected["weights"] == {"tap_iron": 1.0, "tap_time_len": 0.0}
    assert selected["metrics"]["pooled"]["loss"] == pytest.approx(0.0)


def test_equal_losses_choose_the_lower_catboost_weight():
    blend = _blend_module()
    actual, catboost, _ = _oof_frames()

    selected = blend.select_blend_weights(
        actual,
        catboost,
        catboost.copy(),
        weights=[1.0, 0.5, 0.0],
        tie_tolerance=1e-12,
    )

    assert selected["weights"] == {"tap_iron": 0.0, "tap_time_len": 0.0}


def test_blend_aligns_rows_by_keys_instead_of_position():
    blend = _blend_module()
    _, catboost, b1 = _oof_frames()
    b1 = b1.iloc[::-1].reset_index(drop=True)
    b1.loc[b1["sample_id"] == "s1", "pred_tap_iron"] = 200.0
    b1.loc[b1["sample_id"] == "s2", "pred_tap_iron"] = 300.0

    result = blend.blend_predictions(
        catboost,
        b1,
        {"tap_iron": 0.5, "tap_time_len": 1.0},
    )

    assert result["sample_id"].tolist() == ["s1", "s2"]
    assert result["pred_tap_iron"].tolist() == [150.0, 200.0]
    assert result["pred_tap_time_len"].tolist() == [60.0, 60.0]


def test_blend_rejects_mismatched_keys_and_invalid_weights():
    blend = _blend_module()
    _, catboost, b1 = _oof_frames()
    b1.loc[0, "sample_id"] = "other"

    with pytest.raises(ContractError, match="keys do not match"):
        blend.blend_predictions(
            catboost,
            b1,
            {"tap_iron": 0.5, "tap_time_len": 0.5},
        )
    with pytest.raises(ContractError, match="exactly"):
        blend.blend_predictions(
            catboost,
            catboost,
            {"tap_iron": 0.5},
        )
