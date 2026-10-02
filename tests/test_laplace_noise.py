"""Engineering and analytic checks, not competition model-quality evidence."""
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import laplace

from bf_tap_r2.laplace_noise import (
    BOUNDS, LaplaceTreeRegressor, advance, fisher_diagonal, log_score,
    natural_gradient, score_gradient, training_step,
)
from bf_tap_r2.laplace_noise_g0 import SPEC, synthetic_data, validate_spec


def example():
    rng = np.random.default_rng(701)
    x = rng.normal(size=(180, 4))
    y = 10.+4*x[:, 0]+np.sin(x[:, 1])+.1*rng.laplace(size=len(x))
    return x, y


def test_score_equals_scipy_definition():
    params = np.array([[.2, -.8], [2., 1.], [-.9, 0.]])
    y = np.array([1., -2., .4])
    np.testing.assert_allclose(log_score(y, params), -laplace.logpdf(y, loc=params[:, 0], scale=np.exp(params[:, 1])), atol=1e-14)


def test_score_derivative_finite_difference():
    params = np.array([[.2, -.8], [2., 1.], [-.9, 0.]])
    y = np.array([1., -2., .4])
    calculated = score_gradient(y, params)
    for dim in range(2):
        delta = np.zeros_like(params)
        delta[:, dim] = 1e-6
        finite_difference = (log_score(y, params+delta)-log_score(y, params-delta))/2e-6
        np.testing.assert_allclose(calculated[:, dim], finite_difference, atol=1e-9)


def test_natural_gradient_analytic_fisher_inverse_and_nonsmooth_origin():
    params = np.array([[.2, -.8], [2., 1.], [-.9, 0.]])
    y = np.array([1., 2., .4])
    np.testing.assert_allclose(natural_gradient(y, params)*fisher_diagonal(params), score_gradient(y, params), atol=1e-14)
    assert natural_gradient(y, params)[1, 0] == 0


def test_fisher_expectation_from_fixed_laplace_sample():
    # Empirical verification of our analytic calculation, not the training metric.
    y = np.random.default_rng(702).laplace(.4, 1.7, size=200000)
    params = np.tile((.4, np.log(1.7)), (len(y), 1))
    g = score_gradient(y, params)
    np.testing.assert_allclose(g.T@g/len(y), np.diag((1/1.7**2, 1.)), atol=.015)


def test_fixed_scale_score_is_affine_absolute_error():
    y = np.array([1., 3., -5.])
    p = np.column_stack((np.array([2., 1., -3.]), np.repeat(np.log(2.), 3)))
    np.testing.assert_allclose(log_score(y, p), np.log(4.)+np.abs(y-p[:, 0])/2.)


@pytest.mark.parametrize("recipe", ["FIXED", "ADAPTIVE"])
def test_training_score_monotone_and_ledger(recipe):
    x, y = example()
    model = LaplaceTreeRegressor(recipe).fit(x, y, epochs=12)
    values = np.r_[model.initial_nll_, [v["train_nll"] for v in model.history_]]
    assert np.all(np.diff(values) <= 0)
    assert model.fitted_tree_count_ == 12*(1 if recipe == "FIXED" else 2)
    assert len(model.trees_) == 12
    assert np.abs(model.predict(x)-y).mean() < np.abs(np.median(y)-y).mean()
    if recipe == "FIXED":
        assert np.all(model.predict_params(x)[:, 1] == model.initial_log_scale_)


@pytest.mark.parametrize("recipe", ["FIXED", "ADAPTIVE"])
def test_cold_pickle_order_chunk_empty_and_positive_scale(recipe):
    x, y = example()
    model = LaplaceTreeRegressor(recipe).fit(x, y, epochs=12)
    cold = pickle.loads(pickle.dumps(model, protocol=5))
    q = x[::3]
    pred = cold.predict(q)
    assert np.array_equal(pred, model.predict(q))
    assert np.array_equal(pred, cold.predict(q[::-1])[::-1])
    assert np.array_equal(pred, np.concatenate([cold.predict(q[i:i+7]) for i in range(0, len(q), 7)]))
    assert cold.predict(q[:0]).shape == (0,)
    assert np.all(cold.predict_scale(q) > 0)


def test_preprocessing_only_uses_fit_arrays_and_calibration_cannot_change_updates():
    x, y = example()
    query = np.full((5, 4), 1000.)
    a = LaplaceTreeRegressor("ADAPTIVE").fit(x, y, epochs=5, validation=(query, np.ones(5)))
    b = LaplaceTreeRegressor("ADAPTIVE").fit(x, y, epochs=5, validation=(query, np.ones(5)*10000.))
    assert np.array_equal(a.x_mean_, x.mean(axis=0))
    assert a.y_median_ == np.median(y)
    assert [h["train_nll"] for h in a.history_] == [h["train_nll"] for h in b.history_]
    assert [h["step"] for h in a.history_] == [h["step"] for h in b.history_]
    assert a.fitted_tree_count_ == b.fitted_tree_count_ == 10


def test_selected_prefix_equals_fresh_train_at_selected_epoch():
    x, y = example()
    selector = LaplaceTreeRegressor("ADAPTIVE").fit(x[:120], y[:120], epochs=20, validation=(x[120:], y[120:]), patience=5)
    fresh = LaplaceTreeRegressor("ADAPTIVE").fit(x[:120], y[:120], epochs=selector.selected_epoch_)
    assert np.array_equal(fresh.predict(x[120:]), selector.predict(x[120:]))


def test_constant_target_and_initial_constant_epoch_are_valid():
    x, _ = example()
    model = LaplaceTreeRegressor("ADAPTIVE").fit(x, np.repeat(3., len(x)), epochs=2)
    assert np.array_equal(model.predict(x), np.repeat(3., len(x)))
    assert np.all(model.predict_scale(x) > 0)
    initial = LaplaceTreeRegressor("FIXED").fit(x, np.arange(len(x)), epochs=0)
    assert initial.fitted_tree_count_ == 0
    assert np.all(initial.predict(x) == np.median(np.arange(len(x))))


def test_zero_fallback_and_scale_constraint():
    y = np.zeros(3)
    params = np.column_stack((np.zeros(3), np.zeros(3)))
    step, result, _ = training_step(y, params, np.column_stack((np.ones(3), np.zeros(3))))
    assert step == 0
    assert np.array_equal(result, params)
    extreme = advance(params, np.column_stack((np.zeros(3), np.repeat(1e6, 3))), 1.)
    assert np.all(extreme[:, 1] == BOUNDS[0])
    with pytest.raises(ValueError):
        training_step(y, params, np.full_like(params, np.nan))


@pytest.mark.parametrize("kind", ["x_nan", "y_nan", "shape", "epochs", "single", "empty_columns"])
def test_bad_fit_inputs_fail(kind):
    x, y = example()
    epochs = 2
    if kind == "x_nan":
        x[0, 0] = np.nan
    elif kind == "y_nan":
        y[0] = np.inf
    elif kind == "shape":
        y = y[:-1]
    elif kind == "epochs":
        epochs = 501
    elif kind == "single":
        x, y = x[:1], y[:1]
    elif kind == "empty_columns":
        x = x[:, :0]
    with pytest.raises(ValueError):
        LaplaceTreeRegressor("FIXED").fit(x, y, epochs=epochs)


def test_bad_recipe_refit_and_query_rejected():
    with pytest.raises(ValueError):
        LaplaceTreeRegressor("GAUSSIAN_CRPS")
    x, y = example()
    model = LaplaceTreeRegressor("FIXED").fit(x, y, epochs=1)
    with pytest.raises(ValueError):
        model.fit(x, y, epochs=1)
    with pytest.raises(ValueError):
        model.predict(x[:, :2])
    with pytest.raises(ValueError):
        model.predict(np.full((2, 4), np.nan))


def test_frozen_synthetic_recipe_is_deterministic_full_shape():
    first, second = synthetic_data(), synthetic_data()
    assert [v.shape for v in first] == [(2204, 23), (2204,), (550, 23), (550,)]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert np.all(first[0][:, -2:].sum(axis=1) == 1)


def test_frozen_spec_rejects_changes():
    spec = json.loads(Path(SPEC).read_text(encoding="utf-8"))
    validate_spec(spec)
    spec["epochs"] = 499
    with pytest.raises(ValueError):
        validate_spec(spec)
