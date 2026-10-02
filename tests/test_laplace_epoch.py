"""Checks for the new complete trajectory, without modifying old G0 tests."""
import json
from pathlib import Path

import numpy as np
import pytest

from bf_tap_r2.laplace_epoch import LaplaceEpochRegressor, selected_epoch
from bf_tap_r2.laplace_noise import LaplaceTreeRegressor
from bf_tap_r2.laplace_epoch_experiment import SPEC, independent_params, synthetic_arrays, validate_spec


@pytest.mark.parametrize("recipe", ["FIXED", "ADAPTIVE"])
def test_complete_trajectory_retains_all_and_matches_old_prefix(recipe):
    x, y = synthetic_arrays()
    new = LaplaceEpochRegressor(recipe).fit(x[:108], y[:108], epochs=501, validation=(x[108:144], y[108:144]))
    old = LaplaceTreeRegressor(recipe).fit(x[:108], y[:108], epochs=500)
    assert len(new.history_) == len(new.trees_) == 501
    assert new.fitted_tree_count_ == 501*(1 if recipe == "FIXED" else 2)
    assert np.array_equal(new.predict(x[144:], epoch=500), old.predict(x[144:]))
    assert [v["train_nll"] for v in new.history_[:500]] == [v["train_nll"] for v in old.history_]
    assert new.best_epoch(500) <= new.best_epoch(501)
    p, maes = independent_params(new, x[108:144], 501, y[108:144])
    assert np.array_equal(p, new.predict_params(x[108:144], epoch=501))
    assert maes == [h["calibration_mae"] for h in new.history_]


@pytest.mark.parametrize("recipe", ["FIXED", "ADAPTIVE"])
def test_calibration_does_not_change_complete_training_updates(recipe):
    x, y = synthetic_arrays()
    a = LaplaceEpochRegressor(recipe).fit(x[:108], y[:108], epochs=8, validation=(x[108:144], y[108:144]))
    b = LaplaceEpochRegressor(recipe).fit(x[:108], y[:108], epochs=8, validation=(x[108:144], y[108:144]+1000))
    assert np.array_equal(a.predict(x[144:], epoch=8), b.predict(x[144:], epoch=8))
    assert [h["train_nll"] for h in a.history_] == [h["train_nll"] for h in b.history_]


def test_selection_includes_initial_constant_exact_ties_and_horizon():
    history = [{"calibration_mae": 3.}, {"calibration_mae": 2.}, {"calibration_mae": 2.}, {"calibration_mae": 1.}]
    assert selected_epoch(history, 2., 3) == 0
    assert selected_epoch(history, 4., 3) == 2
    assert selected_epoch(history, 4., 4) == 4
    with pytest.raises(ValueError):
        selected_epoch(history, 4., 5)
    with pytest.raises(ValueError):
        selected_epoch(history, float("nan"), 1)


@pytest.mark.parametrize("bad", [-1, 3001, True, 1.5])
def test_invalid_epoch_limits_rejected(bad):
    x, y = synthetic_arrays()
    with pytest.raises(ValueError):
        LaplaceEpochRegressor("FIXED").fit(x, y, epochs=bad)


def test_empty_order_chunk_and_initial_checkpoint():
    x, y = synthetic_arrays()
    m = LaplaceEpochRegressor("ADAPTIVE").fit(x[:108], y[:108], epochs=8)
    q = x[144:]
    assert np.array_equal(m.predict(q, epoch=0), np.repeat(np.median(y[:108]), len(q)))
    assert np.array_equal(m.predict(q), m.predict(q[::-1])[::-1])
    assert np.array_equal(m.predict(q), np.concatenate([m.predict(q[:3]), m.predict(q[3:])]))
    assert m.predict(q[:0]).shape == (0,)
    with pytest.raises(ValueError):
        m.predict(q, epoch=9)
    with pytest.raises(ValueError):
        m.fit(x, y)


def test_new_spec_separate_and_frozen():
    spec = json.loads(Path(SPEC).read_text(encoding="utf-8"))
    validate_spec(spec)
    spec["split_seeds"] = [42]
    with pytest.raises(ValueError):
        validate_spec(spec)


@pytest.mark.parametrize("recipe", ["FIXED", "ADAPTIVE"])
def test_fresh_refit_short_checkpoint_matches_separate_short_fit(recipe):
    x, y = synthetic_arrays()
    long = LaplaceEpochRegressor(recipe).fit(x[:144], y[:144], epochs=16)
    short = LaplaceEpochRegressor(recipe).fit(x[:144], y[:144], epochs=7)
    assert np.array_equal(long.predict(x[144:], epoch=7), short.predict(x[144:]))
