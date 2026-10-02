from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.dcn_cross import (CrossNetwork, CrossPreprocessor, CrossRegressor,
                                clean, fit_partition)
from bf_tap_r2.dcn_cross_verify import independent_prediction, verify_model
from bf_tap_r2.v7_periodic import file_hash


def settings():
    s = yaml.safe_load(Path("configs/dcn_cross_preparation/SPEC.yaml").read_text())["training"]
    s.update(deep_widths=[8, 8], max_epochs=4, batch_size=16, dropout=.1)
    return s


def frame(n=48):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(n, len(FEATURES)))
    f = pd.DataFrame(x, columns=FEATURES)
    f["sample_id"] = [f"synthetic-dcn-{i}" for i in range(n)]
    f["spout_no"] = np.arange(n) % 2 + 1
    for t in TARGETS:
        f[t] = 10 + 2 * x[:, 0] + x[:, 1] * x[:, 2]
    return f


def scalar_cross(x, layers, arm):
    h = x.copy()
    for weight, bias in layers:
        update = np.array([bias[i] + sum(weight[i, j] * h[j] for j in range(len(h)))
                           for i in range(len(h))])
        h = h + (x * update if arm == "CROSS" else update)
    return h


@pytest.mark.parametrize("arm", ["ADDITIVE", "CROSS"])
def test_scalar_operator_and_finite_difference_gradients(arm):
    s = settings()
    n = CrossNetwork(3, s, arm)
    rng = np.random.default_rng(5)
    layers = []
    with torch.no_grad():
        for layer in n.cross:
            w, b = rng.normal(0, .2, (3, 3)), rng.normal(0, .1, 3)
            layer.weight.copy_(torch.tensor(w)); layer.bias.copy_(torch.tensor(b))
            layers.append((w, b))
    a = rng.normal(size=3)
    x = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    actual = n.cross_branch(x)
    np.testing.assert_allclose(actual.detach(), scalar_cross(a, layers, arm), rtol=0, atol=1e-14)
    gradient = torch.autograd.grad(actual.square().sum(), x)[0].numpy()
    eps = 1e-6
    expected = []
    for j in range(3):
        delta = np.eye(3)[j] * eps
        expected.append((np.square(scalar_cross(a + delta, layers, arm)).sum()
                         - np.square(scalar_cross(a - delta, layers, arm)).sum()) / (2 * eps))
    np.testing.assert_allclose(gradient, expected, rtol=1e-7, atol=1e-8)


def test_cross_has_degree_four_and_control_stays_affine():
    s = settings()
    x = torch.tensor([[.2], [1.3], [-.7]], dtype=torch.float64)
    for arm in ["ADDITIVE", "CROSS"]:
        n = CrossNetwork(1, s, arm)
        polynomial = np.array([0., 1.])
        with torch.no_grad():
            for i, layer in enumerate(n.cross):
                w, b = .1 * (i + 1), .02
                layer.weight.fill_(w); layer.bias.fill_(b)
                update = polynomial * w
                update[0] += b
                if arm == "CROSS":
                    update = np.r_[0., update]
                polynomial = np.polynomial.polynomial.polyadd(polynomial, update)
        np.testing.assert_allclose(n.cross_branch(x).detach().numpy()[:, 0],
                                   np.polynomial.polynomial.polyval(x.numpy()[:, 0], polynomial), atol=1e-14)
        assert len(polynomial) == (5 if arm == "CROSS" else 2)
        assert polynomial[-1] != 0


def test_matched_initialization_and_nonzero_cross_gradients():
    s = settings()
    a, c = [CrossNetwork(23, s, arm).eval() for arm in ["ADDITIVE", "CROSS"]]
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, c.state_dict()[name], rtol=0, atol=0)
    x = torch.linspace(-1, 1, 5 * 23, dtype=torch.float64).reshape(5, 23)
    torch.testing.assert_close(a(x), c(x), rtol=0, atol=0)
    c(x).square().sum().backward()
    assert all(torch.count_nonzero(layer.weight.grad) > 0 for layer in c.cross)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in c.parameters())


def test_preprocessing_train_only_constant_unknown_and_ids():
    f = clean(frame())
    f[FEATURES[0]] = 7.
    p = CrossPreprocessor().fit(f)
    snapshot = deepcopy(p.metadata())
    q = f.iloc[:3].copy()
    q[FEATURES[1]] = 1e9; q.spout_no = 999
    values = p.transform(q)
    assert np.count_nonzero(values[:, 0]) == 0
    assert np.count_nonzero(values[:, -2:]) == 0
    assert p.metadata() == snapshot
    with pytest.raises(ValueError, match="targets"):
        p.transform(q.assign(tap_time_len=1))
    with pytest.raises(ValueError, match="identities"):
        CrossPreprocessor().fit(pd.concat([f, f.iloc[:1]]))


@pytest.mark.parametrize("arm", ["ADDITIVE", "CROSS"])
def test_cold_numpy_order_chunk_and_external_hash(tmp_path, arm):
    f = frame()
    s = settings()
    model = CrossRegressor(arm, s).initialize(f.iloc[:30], f.tap_iron.to_numpy()[:30])
    model.train(s["max_epochs"], (clean(f.iloc[30:40]), f.tap_iron.to_numpy()[30:40]))
    path = tmp_path / "model.json"
    sha = model.save(path)
    q = clean(f.iloc[40:])
    expected = model.predict(q)
    cold = CrossRegressor.load(path, sha)
    for actual in [cold.predict(q), cold.predict(q.iloc[::-1])[::-1],
                   np.concatenate([cold.predict(q.iloc[i:i+1]) for i in range(len(q))]),
                   independent_prediction(path, q, sha)]:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    verify_model(path, f.iloc[:30], f.tap_iron.to_numpy()[:30], arm, s, sha, model.metadata())
    with pytest.raises(FileExistsError):
        model.save(path)
    with pytest.raises(ValueError, match="anchored"):
        CrossRegressor.load(path, "0" * 64)
    with pytest.raises(ValueError, match="repeated fit"):
        model.train(1)


def test_independent_audit_rejects_updated_hash_with_forged_metadata(tmp_path):
    f, s = frame(), settings()
    m = CrossRegressor("CROSS", s).initialize(f, f.tap_iron.to_numpy())
    m.train(2)
    original = tmp_path / "original.json"
    m.save(original)
    for field, value, match in [("fit_rows", 47, "preprocessing"),
                                ("target_mean", 0, "scale"),
                                ("selected_epoch", 1, "epoch")]:
        payload = json.loads(original.read_text())
        payload["metadata"][field] = value
        bad = tmp_path / f"{field}.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match=match):
            verify_model(bad, f, f.tap_iron.to_numpy(), "CROSS", s, file_hash(bad))
    payload = json.loads(original.read_text())
    payload["state"]["cross.0.bias"][0] += .1
    bad = tmp_path / "state.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="parameter state"):
        CrossRegressor.load(bad, file_hash(bad))


def test_partition_fresh_refit_identity_and_query_rejection():
    f, s = frame(), settings()
    outer = f.iloc[:40].copy()
    q = clean(f.iloc[40:])
    m, selector, pred, cp = fit_partition(outer.iloc[:30], outer.iloc[30:], outer,
                                         q, "tap_iron", "CROSS", s)
    assert m.preprocessor_.fit_digest_ != selector.preprocessor_.fit_digest_
    assert m.initial_state_digest_ == selector.initial_state_digest_
    assert m.selected_epoch_ == selector.selected_epoch_
    assert np.array_equal(m.preprocessor_.mean_, outer[list(FEATURES)].to_numpy().mean(0))
    assert pred.shape == (8,) and cp.shape == (10,)
    for query in [f.iloc[40:], clean(outer.iloc[:1])]:
        with pytest.raises(ValueError, match="labels|partition"):
            fit_partition(outer.iloc[:30], outer.iloc[30:], outer, query, "tap_iron", "CROSS", s)
    altered = outer.copy(); altered.iloc[0, 0] += 1
    with pytest.raises(ValueError, match="row contents"):
        fit_partition(outer.iloc[:30], outer.iloc[30:], altered, q, "tap_iron", "CROSS", s)


def test_training_validation_budget_and_rng_isolation():
    f, s = frame(), settings()
    before = torch.random.get_rng_state().clone()
    m = CrossRegressor("CROSS", s).initialize(f, f.tap_iron.to_numpy())
    assert torch.equal(before, torch.random.get_rng_state())
    with pytest.raises(ValueError, match="budget"):
        m.train(s["max_epochs"] + 1)
    with pytest.raises(ValueError, match="overlaps"):
        m.train(2, (clean(f.iloc[:3]), np.array([1, 2, 3])))
    m.train(2)
    assert torch.equal(before, torch.random.get_rng_state())
