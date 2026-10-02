"""Long-prefix pure-array and fake-tree tests: no actual sklearn fitting."""
from types import SimpleNamespace

import numpy as np
import pytest

import bf_tap_r2.laplace_leaf_long as long_module
import bf_tap_r2.laplace_leaf_median as original_module
from bf_tap_r2.laplace_leaf_long import LeafMedianLongRegressor
from bf_tap_r2.laplace_leaf_median import LeafMedianRegressor, best_epoch


class FakeStructureTree:
    """Deterministic partition witness, never an actual fitted sklearn tree."""

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
    monkeypatch.setattr(long_module, "DecisionTreeRegressor", FakeStructureTree)


def arrays():
    x = np.column_stack((np.arange(-4., 4.), np.arange(8.) % 3))
    y = 100.+3*x[:, 0]+2*x[:, 1]
    vx = np.array([[-2., .5], [2., 1.], [1., 0.]])
    vy = np.array([95., 108., 103.])
    return x, y, vx, vy


def assert_same_trees(first, second, epochs):
    for a, b, va, vb in zip(first.trees_[:epochs], second.trees_[:epochs],
                             first.leaf_values_[:epochs], second.leaf_values_[:epochs], strict=True):
        assert (a.max_depth, a.min_samples_leaf, a.criterion, a.random_state) == (3, 20, "squared_error", 42)
        assert (b.max_depth, b.min_samples_leaf, b.criterion, b.random_state) == (3, 20, "squared_error", 42)
        np.testing.assert_array_equal(a.fit_x, b.fit_x)
        np.testing.assert_array_equal(a.structure_target, b.structure_target)
        np.testing.assert_array_equal(va, vb)


def test_three_hundred_prefix_is_exact_and_continues_without_reset():
    x, y, vx, vy = arrays()
    original = LeafMedianRegressor(3).fit(x, y, epochs=300, validation=(vx, vy))
    callbacks = []
    model = LeafMedianLongRegressor().fit(x, y, epochs=600, prefix_epochs=300,
                                        validation=(vx, vy), on_epoch=lambda: callbacks.append(1))
    assert model.history_[:300] == original.history_
    assert_same_trees(original, model, 300)
    np.testing.assert_array_equal(model.predict(vx, epoch=300), original.predict(vx, epoch=300))
    assert len(model.trees_) == len(model.history_) == 600
    assert [h["epoch"] for h in model.history_] == list(range(1, 601))
    assert len(callbacks) == 6
    assert model.selected_epoch_ == best_epoch(model.history_, model.initial_calibration_mae_)
    # An independent fake-tree trajectory confirms that continuation uses the
    # existing prefix accumulator rather than restarting its residuals.
    z = (x-model.x_mean_)/model.x_scale_
    mu = np.zeros(len(y))
    sy = (y-model.y_median_)/model.y_scale_
    for tree, values, event in zip(model.trees_, model.leaf_values_, model.history_, strict=True):
        np.testing.assert_array_equal(tree.structure_target, np.sign(sy-mu))
        mu += .05*values[tree.apply(z)]
        assert float(np.abs(sy-mu).mean()) == event["train_mae_normalized"]


def test_scientific_prefix_is_exact_with_fake_trees():
    x, y, vx, vy = arrays()
    original = LeafMedianRegressor(3).fit(x, y, epochs=3000, validation=(vx, vy))
    model = LeafMedianLongRegressor().fit(x, y, epochs=3001, prefix_epochs=3000, validation=(vx, vy))
    assert model.history_[:3000] == original.history_
    assert_same_trees(original, model, 3000)
    np.testing.assert_array_equal(model.predict(vx, epoch=3000), original.predict(vx, epoch=3000))
    assert len(model.trees_) == 3001


@pytest.mark.parametrize("validation", [False, True])
def test_zero_epochs_preserve_original_initial_prediction(validation):
    x, y, vx, vy = arrays()
    calibration = (vx, vy) if validation else None
    model = LeafMedianLongRegressor().fit(x, y, epochs=0, validation=calibration)
    assert model.depth == 3
    assert model.selected_epoch_ == 0
    assert model.history_ == model.trees_ == model.leaf_values_ == []
    np.testing.assert_array_equal(model.predict(vx), np.full(len(vx), np.median(y)))
    assert model.predict(vx[:0]).shape == (0,)


def test_refit_uses_requested_epoch_and_keeps_prediction_order_independence():
    x, y, vx, _ = arrays()
    model = LeafMedianLongRegressor().fit(x, y, epochs=600, prefix_epochs=300)
    assert model.selected_epoch_ == 600
    p = model.predict(vx)
    np.testing.assert_array_equal(model.predict(vx[::-1])[::-1], p)
    np.testing.assert_array_equal(np.r_[model.predict(vx[:1]), model.predict(vx[1:])], p)


def test_calibration_labels_never_change_training_structure_or_leaf_values():
    x, y, vx, vy = arrays()
    first = LeafMedianLongRegressor().fit(x, y, epochs=600, prefix_epochs=300, validation=(vx, vy))
    second = LeafMedianLongRegressor().fit(x, y, epochs=600, prefix_epochs=300,
                                         validation=(vx, vy+1e6))
    assert_same_trees(first, second, 600)
    np.testing.assert_array_equal(first.x_mean_, x.mean(axis=0))
    np.testing.assert_array_equal(first.x_scale_, x.std(axis=0))
    np.testing.assert_array_equal(first.predict(vx, epoch=600), second.predict(vx, epoch=600))
    assert all(a["train_mae_normalized"] == b["train_mae_normalized"]
               for a, b in zip(first.history_, second.history_, strict=True))
    assert first.history_[0]["calibration_mae"] != second.history_[0]["calibration_mae"]


@pytest.mark.parametrize("epochs", [-1, 12001, True, 1.5, np.int64(300)])
def test_invalid_epochs(epochs):
    x, y, _, _ = arrays()
    with pytest.raises(ValueError):
        LeafMedianLongRegressor().fit(x, y, epochs=epochs)


@pytest.mark.parametrize("prefix", [True, 0, 299, 301, 500, 3000.])
def test_invalid_prefix(prefix):
    x, y, _, _ = arrays()
    with pytest.raises(ValueError):
        LeafMedianLongRegressor().fit(x, y, epochs=0, prefix_epochs=prefix)


def test_fresh_model_required_even_after_zero_epoch_fit():
    x, y, _, _ = arrays()
    model = LeafMedianLongRegressor().fit(x, y, epochs=0)
    with pytest.raises(ValueError, match="Fresh model"):
        model.fit(x, y, epochs=600, prefix_epochs=300)


@pytest.mark.parametrize("kind", ["empty", "shape", "nonfinite", "features"])
def test_calibration_validation_matches_original_gate(kind):
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
        LeafMedianLongRegressor().fit(x, y, epochs=600, prefix_epochs=300, validation=(vx, vy))
