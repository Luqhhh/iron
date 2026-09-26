from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("tabm")
pytest.importorskip("rtdl_num_embeddings")

from bf_tap_r2.data import FEATURES
from bf_tap_r2.v3_4_bags import group_safe_inner_folds
from bf_tap_r2.v7_periodic import PeriodicRegressor, evaluate_fold, make_network, write_new


@pytest.fixture
def sample():
    rng = np.random.default_rng(18)
    x = rng.normal(size=(100, len(FEATURES)))
    frame = pd.DataFrame(x, columns=FEATURES)
    frame["sample_id"] = [f"synthetic-{i:04}" for i in range(len(frame))]
    frame["spout_no"] = np.arange(len(frame)) % 3 + 1
    frame["tap_iron"] = 500 + 15 * np.sin(x[:, 0]) + x[:, 1] * 10
    settings = yaml.safe_load(Path("configs/round2_v7/SPEC.yaml").read_text())["training"]
    settings.update(width=16, tabm_k=2, max_epochs=3, patience=2, embedding_dim=4, n_frequencies=4)
    return frame, settings


@pytest.mark.parametrize("backbone", ["mlp", "tabm"])
def test_frequency_parameters_receive_gradients(sample, backbone):
    frame, settings = sample
    net = make_network({"backbone": backbone, "frequency": 0.01}, settings, 4)
    pred = net(torch.tensor(frame[list(FEATURES)].values, dtype=torch.float32),
               torch.tensor(frame[["spout_no"]].values, dtype=torch.long))
    pred.square().mean().backward()
    frequency = [p for name, p in net.named_parameters() if "periodic" in name]
    assert frequency
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in frequency)


@pytest.mark.parametrize("backbone", ["mlp", "tabm"])
@pytest.mark.parametrize("frequency", [None, 0.01])
def test_refit_is_deterministic_and_inference_is_row_independent(sample, backbone, frequency):
    frame, settings = sample
    recipe = {"backbone": backbone, "frequency": frequency}
    model = PeriodicRegressor(recipe, settings).fit(frame, frame.tap_iron.values)
    again = PeriodicRegressor(recipe, settings).fit(frame, frame.tap_iron.values)
    expected = model.predict(frame)
    np.testing.assert_array_equal(expected, again.predict(frame))
    np.testing.assert_allclose(expected, model.predict(frame.iloc[::-1])[::-1], atol=1e-4, rtol=0)
    chunked = np.concatenate([model.predict(chunk) for chunk in [frame.iloc[:30], frame.iloc[30:]]])
    np.testing.assert_allclose(expected, chunked, atol=1e-4, rtol=0)
    np.testing.assert_allclose(expected[:1], model.predict(frame.iloc[:1]), atol=1e-4, rtol=0)
    folds = group_safe_inner_folds(frame, seed=settings["inner_seed"])["fold"]
    np.testing.assert_allclose(model.metadata_["inner_feature_means"], frame.loc[folds != 0, list(FEATURES)].mean())
    np.testing.assert_allclose(model.metadata_["outer_feature_means"], frame[list(FEATURES)].mean())
    assert model.metadata_["optimizer_runs"] == 2


def test_outer_labels_and_features_cannot_change_fitted_model(sample):
    frame, settings = sample
    folds = np.arange(len(frame)) % 5
    recipe = {"backbone": "mlp", "frequency": 0.01}
    pred, metadata = evaluate_fold(frame, folds, "tap_iron", recipe, settings, 0)
    changed = frame.copy()
    changed.loc[folds == 0, "tap_iron"] += 100000
    altered, same_metadata = evaluate_fold(changed, folds, "tap_iron", recipe, settings, 0)
    np.testing.assert_array_equal(pred, altered)
    assert metadata == same_metadata
    changed.loc[folds == 0, list(FEATURES)] += 100
    _, new_metadata = evaluate_fold(changed, folds, "tap_iron", recipe, settings, 0)
    assert metadata == new_metadata


def test_existing_evidence_is_not_overwritten(tmp_path):
    path = tmp_path / "evidence.json"
    write_new(path, {"first": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_new(path, {"second": True})
    assert path.read_bytes() == before
