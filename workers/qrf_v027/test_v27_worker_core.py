from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qrf_v015"))

from leaf_recency import (
    _exact_cdf_at,
    assert_all_ones_reproduces,
    distribution_weights_recency,
    lower_median_recency,
    predict_with_recency,
    recency60_weights,
)
from target_forest import PARAMETERS, PROTOCOL, TARGET, UNIT, IronTargetForest, make_estimator


def matrix(rows=40):
    rng = np.random.default_rng(2027)
    return rng.normal(size=(rows, 6)).astype(np.float32)


def test_iron_factory_and_protocol_are_target_specific():
    estimator = make_estimator()
    assert type(estimator) is RandomForestRegressor
    assert estimator.criterion == "absolute_error"
    assert estimator.bootstrap is True
    assert estimator.n_estimators == 256
    assert PARAMETERS["min_samples_leaf"] == 10
    assert PROTOCOL == "QRF_FULLTRAIN_TARGET_v027"
    assert TARGET == "tap_iron"
    assert UNIT == "tonne"


def test_iron_forest_uses_raw_response_leaf_distribution(monkeypatch):
    x = matrix()
    y = np.arange(len(x), dtype=float) + 450.0
    ids = [f"x{index}" for index in range(len(x))]
    model = IronTargetForest().fit(x, y, ids)
    assert np.array_equal(model.y, y)
    assert model.training_partition_audit["trees"] == 256
    assert all(tree.criterion == "absolute_error" and tree.splitter == "best" for tree in model.forest.estimators_)
    monkeypatch.setattr(model.forest, "predict", lambda *_: (_ for _ in ()).throw(AssertionError("mean predict forbidden")))
    median, mean, diagnostics = model.predict(x[:4])
    assert median.shape == mean.shape == (4,)
    assert (median >= y.min()).all() and (median <= y.max()).all()
    assert all(abs(item["weight_mass"] - 1.0) <= 1e-12 for item in diagnostics)


def test_iron_contract_rejects_bad_response_or_matrix():
    x = matrix(20)
    ids = [str(index) for index in range(20)]
    with pytest.raises(ValueError, match="float32"):
        IronTargetForest().fit(x.astype(np.float64), np.arange(20), ids)
    with pytest.raises(ValueError, match="nonnegative"):
        IronTargetForest().fit(x, np.arange(20) - 1, ids)


def test_leaf_recency_normalizes_each_tree_not_globally():
    members = [np.asarray([0, 1]), np.asarray([1, 2, 3])]
    recency = np.asarray([1.0, 2.0, 4.0, 8.0])
    weights = distribution_weights_recency(members, recency, 4)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[[0, 1]].sum() >= 0.5
    first_tree = recency[[0, 1]] / recency[[0, 1]].sum() / 2
    second_tree = recency[[1, 2, 3]] / recency[[1, 2, 3]].sum() / 2
    expected = np.zeros(4)
    expected[[0, 1]] += first_tree
    expected[[1, 2, 3]] += second_tree
    assert np.array_equal(weights, expected)


def test_all_ones_recovers_original_distribution_exactly():
    members = [np.asarray([0, 1]), np.asarray([1, 2, 3])]
    got = distribution_weights_recency(members, np.ones(4), 4)
    expected = np.asarray([0.25, 0.25 + 1 / 6, 1 / 6, 1 / 6])
    assert np.array_equal(got, expected)


def test_exact_binary64_fraction_oracle_selects_lower_median():
    y = np.asarray([10.0, 20.0, 30.0, 40.0])
    members = [np.asarray([0, 1]), np.asarray([2, 3])]
    recency = np.ones(4)
    weights = distribution_weights_recency(members, recency, 4)
    assert _exact_cdf_at(y, members, recency, 20.0) == Fraction(1, 2)
    assert lower_median_recency(y, weights, members, recency) == 20.0


def test_recency_half_life_and_cutoff_contract():
    day = 86400 * 10**9
    cutoff = 200 * day
    got = recency60_weights(np.asarray([cutoff - day, cutoff - 61 * day]), cutoff)
    assert got[1] / got[0] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="precede"):
        recency60_weights([cutoff], cutoff)


def test_frozen_forest_recency_prediction_is_batch_invariant():
    x = matrix()
    y = np.arange(len(x), dtype=float) + 80.0
    ids = [f"x{index}" for index in range(len(x))]
    model = IronTargetForest().fit(x, y, ids)
    recency = np.exp2(-np.arange(len(x), dtype=float) / 60.0)
    whole = predict_with_recency(model, x[:8], recency)
    reverse = predict_with_recency(model, x[:8][::-1], recency)
    assert np.array_equal(whole[0], reverse[0][::-1])
    assert assert_all_ones_reproduces(model, x[:8])
