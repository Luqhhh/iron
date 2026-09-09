from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

from bf_tap.exceptions import ContractError


def module():
    path = "bf_tap.models.optimization_v02.m4_ensemble.model"
    assert find_spec(path) is not None, "M4 gating implementation must exist"
    return import_module(path)


def frames():
    actual = pd.DataFrame({
        "fold_id": ["f1", "f1", "f2", "f2", "f3", "f3"],
        "sample_id": list("abcdef"),
        "tap_iron": [100.] * 6, "tap_time_len": [100.] * 6,
    })
    def pred(errors):
        value = actual[["fold_id", "sample_id"]].copy()
        for target in ("tap_iron", "tap_time_len"):
            value["pred_" + target] = 100. + np.array(errors)
        return value
    return actual, pred([10] * 6), pred([2, -2] * 3), pred([-2, 2] * 3)


def test_complementary_stable_members_enable_m4_and_frozen_weights_replay():
    m = module()
    actual, control, p1, p2 = frames()
    selected = m.select_ensemble(actual, {"M1_BLEND": p1, "M2_RECENCY": p2}, control, control)
    assert selected["status"] == "ENABLED"
    for weights in selected["weights"].values():
        assert sorted(weights.values()) == [.5, .5]
    result = m.apply_ensemble({"M1_BLEND": p1.iloc[::-1], "M2_RECENCY": p2}, selected)
    np.testing.assert_allclose(result[["pred_tap_iron", "pred_tap_time_len"]], 100.)
    assert selected["metrics"]["pooled"]["loss"] == 0.


def test_identical_or_unstable_members_cannot_enable_m4():
    m = module()
    actual, control, p1, p2 = frames()
    assert m.select_ensemble(actual, {"M1_BLEND": p1}, control, control)["status"] == "SKIPPED"
    assert m.select_ensemble(actual, {"M1_BLEND": p1, "M2_RECENCY": p1}, control, control)["status"] == "SKIPPED"
    assert m.select_ensemble(actual, {"M1_BLEND": control, "M2_RECENCY": p2}, control, control)["status"] == "SKIPPED"


def test_m4_rejects_remote_members_and_mismatched_prediction_keys():
    m = module()
    actual, control, p1, p2 = frames()
    with pytest.raises(ContractError):
        m.select_ensemble(actual, {"E09": p1, "M2_RECENCY": p2}, control, control)
    selected = m.select_ensemble(actual, {"M1_BLEND": p1, "M2_RECENCY": p2}, control, control)
    with pytest.raises(ContractError):
        m.apply_ensemble({"M1_BLEND": p1.iloc[:-1], "M2_RECENCY": p2}, selected)
