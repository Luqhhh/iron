"""Pure arrays and fake structures; no actual sklearn tree fitting calls."""
from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest

import bf_tap_r2.laplace_leaf_median as original_module
import bf_tap_r2.laplace_leaf_partition as partition_module
from bf_tap_r2.laplace_leaf_median import best_epoch, leaf_medians
from bf_tap_r2.laplace_leaf_partition import ResidualL1PartitionRegressor


class FakeStructureTree:
    """Record structure inputs and route deterministically without any fitting."""

    def __init__(self, *, max_depth, min_samples_leaf, criterion, random_state):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.tree_ = SimpleNamespace(node_count=3)

    def fit(self, x, y):
        self.fit_x = np.asarray(x).copy()
        self.structure_target = np.asarray(y).copy()
        return self

    def apply(self, x):
        return np.where(np.asarray(x)[:, 0] < 0, 1, 2)


@pytest.fixture(autouse=True)
def no_actual_tree_fits(monkeypatch):
    monkeypatch.setattr(original_module, "DecisionTreeRegressor", FakeStructureTree)
    monkeypatch.setattr(partition_module, "DecisionTreeRegressor", FakeStructureTree)


def arrays():
    x = np.column_stack((np.arange(-4., 4.), np.arange(8.) % 3))
    y = 100.+3*x[:, 0]+2*x[:, 1]
    vx = np.array([[-2., .5], [2., 1.], [1., 0.]])
    vy = np.array([95., 108., 103.])
    return x, y, vx, vy


def test_each_structure_sees_full_current_residual_not_its_sign():
    x, y, vx, vy = arrays()
    model = ResidualL1PartitionRegressor().fit(x, y, epochs=300, validation=(vx, vy))
    z = (x-model.x_mean_)/model.x_scale_
    sy = (y-model.y_median_)/model.y_scale_
    mu = np.zeros(len(y))
    vm = np.zeros(len(vy))
    previous = model.initial_train_mae_
    for tree, values, event in zip(model.trees_, model.leaf_values_, model.history_, strict=True):
        assert (tree.max_depth, tree.min_samples_leaf, tree.criterion, tree.random_state) == (3, 20, "absolute_error", 42)
        residual = sy-mu
        np.testing.assert_array_equal(tree.fit_x, z)
        np.testing.assert_array_equal(tree.structure_target, residual)
        assert not np.array_equal(tree.structure_target, np.sign(residual))
        np.testing.assert_array_equal(values, leaf_medians(residual, tree.apply(z), 3))
        mu += .05*values[tree.apply(z)]
        expected_mae = float(np.abs(sy-mu).mean())
        assert event["train_mae_normalized"] == expected_mae
        assert expected_mae <= previous+1e-12
        vm += .05*values[tree.apply(model._x(vx))]
        assert event["calibration_mae"] == float(np.abs(vy-(vm*model.y_scale_+model.y_median_)).mean())
        previous = expected_mae
    assert model.selected_epoch_ == best_epoch(model.history_, model.initial_calibration_mae_)
    assert model.history_[-1]["train_mae_normalized"] < model.initial_train_mae_


def test_calibration_labels_and_features_do_not_change_training_structure():
    x, y, vx, vy = arrays()
    first = ResidualL1PartitionRegressor().fit(x, y, epochs=200, validation=(vx, vy))
    second = ResidualL1PartitionRegressor().fit(x, y, epochs=200, validation=(vx+1e6, vy-1e6))
    for a, b, va, vb in zip(first.trees_, second.trees_, first.leaf_values_, second.leaf_values_, strict=True):
        np.testing.assert_array_equal(a.fit_x, b.fit_x)
        np.testing.assert_array_equal(a.structure_target, b.structure_target)
        np.testing.assert_array_equal(va, vb)
    np.testing.assert_array_equal(first.x_mean_, x.mean(axis=0))
    np.testing.assert_array_equal(first.x_scale_, x.std(axis=0))
    assert first.y_median_ == np.median(y)
    assert first.y_scale_ == y.std()
    np.testing.assert_array_equal(first.predict(vx, epoch=200), second.predict(vx, epoch=200))
    assert all(a["train_mae_normalized"] == b["train_mae_normalized"]
               for a, b in zip(first.history_, second.history_, strict=True))
    assert first.history_[0]["calibration_mae"] != second.history_[0]["calibration_mae"]


def test_full_trajectory_and_callback_boundaries_without_calibration():
    x, y, vx, _ = arrays()
    calls = []
    model = ResidualL1PartitionRegressor().fit(x, y, epochs=301, on_epoch=lambda: calls.append(1))
    assert len(model.trees_) == len(model.leaf_values_) == len(model.history_) == 301
    assert [h["epoch"] for h in model.history_] == list(range(1, 302))
    assert len(calls) == 3
    assert model.selected_epoch_ == 301
    assert all("calibration_mae" not in h for h in model.history_)
    assert model.metadata()["fitted_tree_count"] == model.metadata()["complete_epochs"] == 301
    assert model.metadata()["depth"] == 3
    p = model.predict(vx)
    np.testing.assert_array_equal(model.predict(vx[::-1])[::-1], p)
    np.testing.assert_array_equal(np.r_[model.predict(vx[:1]), model.predict(vx[1:])], p)
    assert model.predict(vx[:0]).shape == (0,)
    np.testing.assert_array_equal(model.predict(vx, epoch=0), np.full(len(vx), np.median(y)))


@pytest.mark.parametrize("validation", [False, True])
def test_zero_epochs_and_fresh_instance_gate(validation):
    x, y, vx, vy = arrays()
    calls = []
    model = ResidualL1PartitionRegressor().fit(x, y, epochs=0,
        validation=(vx, vy) if validation else None, on_epoch=lambda: calls.append(1))
    assert model.depth == 3
    assert model.selected_epoch_ == 0
    assert model.history_ == model.trees_ == model.leaf_values_ == []
    assert calls == []
    np.testing.assert_array_equal(model.predict(vx), np.full(len(vx), np.median(y)))
    with pytest.raises(ValueError, match="Fresh model"):
        model.fit(x, y, epochs=1)


def test_default_is_full_twelve_thousand_and_upper_bound_is_accepted():
    assert signature(ResidualL1PartitionRegressor.fit).parameters["epochs"].default == 12000
    x, y, _, _ = arrays()
    model = ResidualL1PartitionRegressor().fit(x, y, epochs=12000)
    assert model.selected_epoch_ == len(model.trees_) == len(model.history_) == 12000


@pytest.mark.parametrize("epochs", [-1, 12001, True, 1.5, np.int64(300)])
def test_invalid_epoch_counts(epochs):
    x, y, _, _ = arrays()
    model = ResidualL1PartitionRegressor()
    with pytest.raises(ValueError):
        model.fit(x, y, epochs=epochs)
    assert not hasattr(model, "trees_")


@pytest.mark.parametrize("kind", ["empty", "shape", "nonfinite", "features"])
def test_original_calibration_input_gates_are_preserved(kind):
    x, y, vx, vy = arrays()
    if kind == "empty":
        vx, vy = vx[:0], vy[:0]
    elif kind == "shape":
        vy = vy[:1]
    elif kind == "nonfinite":
        vy = np.array([np.nan, 1., 2.])
    else:
        vx = np.zeros((3, 1))
    with pytest.raises(ValueError):
        ResidualL1PartitionRegressor().fit(x, y, epochs=1, validation=(vx, vy))


@pytest.mark.parametrize("kind", ["rows", "shape", "nonfinite_x", "nonfinite_y"])
def test_original_training_input_gates_are_preserved(kind):
    x, y, _, _ = arrays()
    if kind == "rows":
        x, y = x[:1], y[:1]
    elif kind == "shape":
        y = y[:, None]
    elif kind == "nonfinite_x":
        x[0, 0] = np.inf
    else:
        y[0] = np.nan
    with pytest.raises(ValueError):
        ResidualL1PartitionRegressor().fit(x, y, epochs=1)


def test_constant_target_zero_scale_and_exact_ties_select_zero():
    x, _, vx, _ = arrays()
    y = np.full(len(x), 100.)
    model = ResidualL1PartitionRegressor().fit(x, y, epochs=10,
        validation=(vx, np.full(len(vx), 100.)))
    assert model.y_scale_ == 1.
    assert model.selected_epoch_ == 0
    assert model.metadata()["exactly_zero_update_epochs"] == 10
    np.testing.assert_array_equal(model.predict(vx), np.full(len(vx), 100.))


@pytest.mark.parametrize("failure", ["increase", "nonfinite"])
def test_training_l1_failure_is_not_hidden(monkeypatch, failure):
    x, y, _, _ = arrays()
    if failure == "increase":
        monkeypatch.setattr(partition_module, "leaf_medians", lambda *args: -leaf_medians(*args))
    else:
        monkeypatch.setattr(partition_module, "leaf_medians", lambda *args: np.full(3, np.nan))
    with pytest.raises(ValueError, match="Training L1 descent invariant"):
        ResidualL1PartitionRegressor().fit(x, y, epochs=1)
