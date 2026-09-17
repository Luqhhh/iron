import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from partition_forest import CANDIDATES, PARAMETERS, PartitionForest, make_estimator


def matrix(rows=40):
    rng = np.random.default_rng(2026)
    return rng.normal(size=(rows, 6)).astype(np.float32)


def test_factory_fixes_only_registered_partition_choices():
    a = make_estimator("A")
    b = make_estimator("B")
    assert type(a) is RandomForestRegressor
    assert type(b) is ExtraTreesRegressor
    assert a.criterion == "absolute_error"
    assert b.criterion == "squared_error"
    assert a.bootstrap is b.bootstrap is True
    assert a.n_estimators == b.n_estimators == 256
    with pytest.raises(ValueError, match="candidate"):
        make_estimator("C")


@pytest.mark.parametrize("candidate", ["A", "B"])
def test_full_original_response_leaf_distribution_and_invariance(candidate, monkeypatch):
    x = matrix()
    y = np.arange(len(x), dtype=float) + 80.0
    ids = [f"x{i}" for i in range(len(x))]
    model = PartitionForest(candidate).fit(x, y, ids)
    assert np.array_equal(model.y, y)
    assert model.training_partition_audit["trees"] == 256
    assert model.training_partition_audit["all_original_rows_projected_per_tree"]
    expected = CANDIDATES[candidate]
    assert all(tree.criterion == expected["criterion"] and tree.splitter == expected["splitter"]
               for tree in model.forest.estimators_)
    monkeypatch.setattr(model.forest, "predict", lambda *_: (_ for _ in ()).throw(AssertionError("mean predict forbidden")))
    q, mean, diagnostics = model.predict(x[:5])
    assert q.shape == mean.shape == (5,)
    assert (q >= y.min()).all() and (q <= y.max()).all()
    assert all(abs(item["weight_mass"] - 1.0) <= 1e-12 for item in diagnostics)
    reverse = model.predict(x[:5][::-1])[0][::-1]
    assert np.array_equal(reverse, q)


def test_raw_response_and_float32_contracts():
    x = matrix(20)
    ids = [str(i) for i in range(20)]
    with pytest.raises(ValueError, match="float32"):
        PartitionForest("A").fit(x.astype(np.float64), np.arange(20), ids)
    with pytest.raises(ValueError, match="nonnegative"):
        PartitionForest("A").fit(x, np.arange(20) - 1, ids)
    with pytest.raises(ValueError, match="alignment"):
        PartitionForest("A").fit(x, np.arange(20), ids[:-1])


def test_parameter_dictionaries_explicitly_retain_bootstrap():
    assert PARAMETERS["A"]["bootstrap"] is True
    assert PARAMETERS["B"]["bootstrap"] is True
    assert PARAMETERS["A"]["criterion"] == "absolute_error"
    assert PARAMETERS["B"]["criterion"] == "squared_error"
