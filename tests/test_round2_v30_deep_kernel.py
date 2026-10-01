import pytest
pytest.importorskip("torch", reason="Optional neural dependency is absent from the locked baseline environment")

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.v30_deep_kernel import (
    DeepKernelRegressor, SparseKernel, collapsed_bound, farthest_rows, fit_partition, select_weight,
)
from bf_tap_r2.v30_run import partitions, verified_unit


def settings():
    result = yaml.safe_load(Path("configs/round2_v30/SPEC.yaml").read_text())["training"]
    return dict(result, inducing_points=12, max_epochs=4, patience=3, hidden_widths=[8], latent_dim=3)


def frame(n=70):
    rng = np.random.default_rng(851)
    x = rng.normal(size=(n, len(FEATURES)))
    result = pd.DataFrame(x, columns=FEATURES)
    result["sample_id"] = [f"syn-{i:04d}" for i in range(n)]
    result["spout_no"] = 1 + np.arange(n) % 2
    result["tap_iron"] = 100 + 5 * np.sin(x[:, 0]) + .5 * x[:, 1]
    result["tap_time_len"] = 30 + 2 * np.sin(x[:, 2])
    return result


def test_sparse_bound_matches_independent_dense_covariance_and_posterior():
    torch.manual_seed(32)
    x, z = torch.randn(13, 3, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
    def kernel(a, b):
        return torch.exp(-torch.cdist(a, b).square() / 2)
    kmm, kmn = kernel(z, z), kernel(z, x)
    y, noise, jitter = torch.randn(13, dtype=torch.float64), torch.tensor(.13), 1e-6
    bound, coefficient = collapsed_bound(kmm, kmn, y, noise, torch.tensor(13.), jitter)
    stable = kmm + jitter * torch.eye(len(z))
    q = kmn.T @ torch.linalg.solve(stable, kmn)
    covariance = q + noise * torch.eye(len(x))
    exact = -torch.distributions.MultivariateNormal(torch.zeros_like(y), covariance_matrix=covariance).log_prob(y)
    exact += (13 - q.trace()) / (2 * noise)
    assert torch.allclose(bound, exact, atol=1e-6, rtol=1e-7)
    query = torch.randn(4, 3, dtype=torch.float64)
    expected = kernel(query, z) @ torch.linalg.solve(stable, kmn) @ torch.linalg.solve(covariance, y)
    assert torch.allclose(kernel(query, z) @ coefficient, expected, atol=1e-8)


@pytest.mark.parametrize("recipe", ["GP_ARD", "DKL_RAW", "DKL_PLR"])
def test_finite_kernel_and_encoder_gradients_and_cold_predictions(recipe, tmp_path):
    data, options = frame(), settings()
    model = DeepKernelRegressor(recipe, options).initialize(data.iloc[:50], data.tap_iron.to_numpy()[:50])
    loss, _ = model.model_.objective(model.x_train_, model.inducing_, model.y_train_, options["jitter"])
    loss.backward()
    for param in model.model_.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()
    if recipe != "GP_ARD":
        assert sum(float(p.grad.abs().sum()) for p in model.model_.encoder.parameters()) > 0
    model.train(3)
    query = data.iloc[50:].drop(columns=list(TARGETS))
    expected = model.predict(query)
    destination = tmp_path / "model.pt"
    model.save(destination)
    loaded = DeepKernelRegressor.load(destination)
    assert np.allclose(loaded.predict(query), expected, atol=1e-9, rtol=0)
    assert np.allclose(loaded.predict(query.iloc[::-1])[::-1], expected, atol=1e-9, rtol=0)
    singles = np.concatenate([loaded.predict(query.iloc[i:i+1]) for i in range(len(query))])
    assert np.allclose(singles, expected, atol=1e-9, rtol=0)
    with pytest.raises(FileExistsError): model.save(destination)
    state = torch.load(destination, weights_only=True)
    assert "y_train" not in state and "optimizer" not in state


def test_exact_inducing_gp_limit_and_nonnegative_trace_penalty():
    torch.manual_seed(31)
    x = torch.randn(9, 2, dtype=torch.float64)
    k = torch.exp(-torch.cdist(x, x).square() / 2)
    y = torch.randn(9, dtype=torch.float64)
    noise = torch.tensor(.07, dtype=torch.float64)
    bound, coefficient = collapsed_bound(k, k, y, noise, k.trace(), 1e-10)
    covariance = k + noise * torch.eye(len(x))
    expected = -torch.distributions.MultivariateNormal(torch.zeros_like(y), covariance_matrix=covariance).log_prob(y)
    assert torch.allclose(bound, expected, atol=1e-7, rtol=0)
    assert torch.allclose(k @ coefficient, k @ torch.linalg.solve(covariance, y), atol=1e-7, rtol=0)


def test_partition_leaves_query_labels_out_and_calibration_is_disjoint():
    data = frame(100)
    folds = np.arange(len(data)) % 5
    spec = yaml.safe_load(Path("configs/round2_v30/SPEC.yaml").read_text())
    training, query, fitting, calibration = partitions(data, folds, 0, spec)
    assert not set(TARGETS) & set(query.columns)
    assert not set(fitting.sample_id) & set(calibration.sample_id)
    modified = data.copy()
    modified.loc[folds == 0, list(TARGETS)] = 1e10
    altered = partitions(modified, folds, 0, spec)
    for a, b in zip((training, query, fitting, calibration), altered):
        pd.testing.assert_frame_equal(a, b)


def test_calibration_and_outer_query_do_not_fit_preprocessing():
    data, options = frame(), settings()
    fitting, calibration, query = data.iloc[:40].copy(), data.iloc[40:55].copy(), data.iloc[55:].copy()
    calibration.loc[:, list(FEATURES)] += 100
    query.loc[:, list(FEATURES)] += 300
    training = pd.concat([fitting, calibration])
    base = np.full(len(calibration), fitting.tap_iron.mean())
    final, values, meta, _ = fit_partition(fitting, calibration, training, query.drop(columns=list(TARGETS)),
        "tap_iron", "GP_ARD", options, base, [0., .5, 1.])
    assert np.allclose(meta["calibration"]["preprocessing"]["means"], fitting[list(FEATURES)].mean())
    assert np.allclose(meta["refit"]["preprocessing"]["means"], training[list(FEATURES)].mean())
    assert np.isfinite(values).all()
    with pytest.raises(ValueError, match="Query labels"):
        fit_partition(fitting, calibration, training, query, "tap_iron", "GP_ARD", options, base, [0., 1.])


def test_weight_selection_ties_and_invalid_predictions():
    assert select_weight([1, 3], [1, 3], [1, 3], [0, .5, 1])[0] == 0
    assert select_weight([1, 3], [0, 0], [1, 3], [0, .5, 1])[0] == 1
    with pytest.raises(ValueError): select_weight([1], [0], [np.nan], [0, 1])
    with pytest.raises(ValueError): select_weight([1], [0], [1], [1, 0])


def test_inducing_subset_is_unique_and_completed_units_are_immutable(tmp_path):
    x = np.zeros((12, 3))
    assert len(set(farthest_rows(x, 10, 42))) == 10
    assert np.array_equal(farthest_rows(x, 10, 42), farthest_rows(x, 10, 42))
    unit = tmp_path / "failed"
    unit.mkdir()
    with pytest.raises(ValueError, match="preserved"): verified_unit(unit, "identity")
