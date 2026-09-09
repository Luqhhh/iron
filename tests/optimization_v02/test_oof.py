from importlib import import_module
from importlib.util import find_spec

import pandas as pd
import pytest

from bf_tap.exceptions import ContractError


def _oof_module():
    module = find_spec("bf_tap.models.optimization_v02.oof")
    assert module is not None, "OOF scoring contract must exist"
    return import_module("bf_tap.models.optimization_v02.oof")


def _frames():
    actual = pd.DataFrame(
        {
            "fold_id": ["A", "B"],
            "sample_id": ["s1", "s2"],
            "tap_iron": [100.0, 900.0],
            "tap_time_len": [100.0, 900.0],
        }
    )
    predicted = pd.DataFrame(
        {
            "fold_id": ["A", "B"],
            "sample_id": ["s1", "s2"],
            "pred_tap_iron": [110.0, 900.0],
            "pred_tap_time_len": [110.0, 900.0],
        }
    )
    return actual, predicted


def test_oof_uses_pooled_denominators_not_mean_fold_loss():
    oof = _oof_module()
    actual, predicted = _frames()

    result = oof.score_oof_predictions(actual, predicted)

    assert result["folds"]["A"]["loss"] == pytest.approx(0.1)
    assert result["folds"]["B"]["loss"] == pytest.approx(0.0)
    assert result["pooled"]["iron"]["wmape"] == pytest.approx(0.01)
    assert result["pooled"]["time"]["wmape"] == pytest.approx(0.01)
    assert result["pooled"]["loss"] == pytest.approx(0.01)


def test_oof_rejects_prediction_assigned_to_wrong_fold():
    oof = _oof_module()
    actual, predicted = _frames()
    predicted.loc[0, "fold_id"] = "B"

    with pytest.raises(ContractError, match="fold/sample keys"):
        oof.score_oof_predictions(actual, predicted)


def test_oof_rejects_sample_reused_across_rolling_folds():
    oof = _oof_module()
    actual, predicted = _frames()
    actual.loc[1, "sample_id"] = "s1"
    predicted.loc[1, "sample_id"] = "s1"

    with pytest.raises(ContractError, match="sample_id appears in multiple folds"):
        oof.score_oof_predictions(actual, predicted)
