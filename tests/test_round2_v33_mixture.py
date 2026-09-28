from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("rtdl_num_embeddings")
from scipy.optimize import brentq
from scipy.stats import norm

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.v33_mixture import MixtureRegressor, fit_partition, mixture_median, mixture_nll
from bf_tap_r2.v33_run import partitions, read_reference, select_finalists, verified_unit


def options():
    s = yaml.safe_load(Path("configs/round2_v33/SPEC.yaml").read_text())["training"]
    return dict(s, hidden_widths=[12], max_epochs=3, patience=2)


def synthetic(n=90):
    x = np.random.default_rng(533).normal(size=(n, len(FEATURES)))
    data = pd.DataFrame(x, columns=FEATURES)
    data["sample_id"] = [f"synthetic-{i}" for i in range(n)]
    data["spout_no"] = 1 + np.arange(n) % 2
    data["tap_iron"] = 100 + 5*x[:, 0] + np.sin(x[:, 1])
    data["tap_time_len"] = 30 + 2*x[:, 2]
    return data


def test_likelihood_matches_independent_mixture_density_and_has_gradients():
    y = torch.tensor([-.2, 1.7], dtype=torch.float64)
    logits = torch.tensor([[.3, -.4, .8], [-.2, .7, -.1]], dtype=torch.float64, requires_grad=True)
    means = torch.tensor([[-2., 0., 2.], [0., 1., 3.]], dtype=torch.float64, requires_grad=True)
    scales = torch.tensor([[.3, 1., .6], [.5, .2, 2.]], dtype=torch.float64, requires_grad=True)
    loss = mixture_nll(y, logits, means, scales)
    weights = torch.softmax(logits, 1).detach().numpy()
    expected = -np.log((weights * norm.pdf(y.numpy()[:, None], means.detach().numpy(), scales.detach().numpy())).sum(1)).mean()
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-12)
    loss.backward()
    for value in (logits, means, scales):
        assert torch.isfinite(value.grad).all() and float(value.grad.abs().sum()) > 0


def test_median_matches_scalar_cdf_root_not_mixture_mean():
    w, m, s = np.array([[.8, .2]]), np.array([[-1., 5.]]), np.array([[.3, 2.]])
    median = mixture_median(w, m, s)[0]
    root = brentq(lambda x: sum(w[0] * norm.cdf(x, m[0], s[0])) - .5, -10, 20)
    assert median == pytest.approx(root, abs=1e-10)
    assert abs(median - (w*m).sum()) > 1
    assert mixture_median([[1]], [[7]], [[.2]])[0] == 7
    assert mixture_median([[.5,.5]], [[-3,3]], [[1,1]])[0] == pytest.approx(0, abs=1e-10)
    with pytest.raises(ValueError): mixture_median([[1]], [[0]], [[0]])


@pytest.mark.parametrize("recipe", ["GAUSS1", "MDN3"])
def test_training_only_scaling_gradients_and_saved_query_invariance(recipe, tmp_path):
    data = synthetic()
    training, query = data.iloc[:65], data.iloc[65:].drop(columns=list(TARGETS)).copy()
    query.loc[:, list(FEATURES)] += 20
    model = MixtureRegressor(recipe, options()).initialize(training, training.tap_iron.to_numpy())
    assert np.allclose(model.preprocessor_.means_, training[list(FEATURES)].mean())
    mixture_nll(model.y_train_, *model.model_(model.x_train_)).backward()
    assert sum(float(p.grad.abs().sum()) for p in model.model_.embedding.parameters()) > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.model_.parameters())
    model.train(2)
    expected = model.predict(query)
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = MixtureRegressor.load(path)
    assert np.allclose(expected, loaded.predict(query), atol=1e-8, rtol=0)
    assert np.allclose(expected, loaded.predict(query.iloc[::-1])[::-1], atol=1e-8, rtol=0)
    assert np.allclose(expected, np.concatenate([loaded.predict(query.iloc[i:i+1]) for i in range(len(query))]), atol=1e-8, rtol=0)
    with pytest.raises(FileExistsError): model.save(path)
    payload = torch.load(path, weights_only=True)
    assert not {"y_train", "optimizer", "query_labels"} & payload.keys()


def test_outer_labels_cannot_affect_partitions_and_labeled_query_is_rejected():
    data = synthetic(100)
    folds = np.arange(len(data)) % 5
    spec = yaml.safe_load(Path("configs/round2_v33/SPEC.yaml").read_text())
    expected = partitions(data, folds, 0, spec)
    changed = data.copy()
    changed.loc[folds == 0, list(TARGETS)] = 1e8
    for a,b in zip(expected, partitions(changed, folds, 0, spec)):
        pd.testing.assert_frame_equal(a,b)
    train, query, fit, cal = expected
    query["tap_iron"] = 100
    with pytest.raises(ValueError, match="Query labels"):
        fit_partition(fit, cal, train, query, "tap_iron", "MDN3", options(), np.zeros(len(cal)), [0.,1.])


def test_failed_units_are_not_reused(tmp_path):
    tmp_path.joinpath("failed").mkdir()
    with pytest.raises(ValueError, match="preserved"):
        verified_unit(tmp_path / "failed", "fake")


def test_reference_readback_rejects_wrong_training_identity(tmp_path):
    import json
    (tmp_path / "metadata.json").write_text(json.dumps({"fit_ids_digest":"incorrect"}))
    data = synthetic()
    with pytest.raises(ValueError, match="fit rows"):
        read_reference(tmp_path, data.iloc[:50], data.iloc[50:], {})


def test_only_positive_mixture_beating_control_can_consume_confirmation():
    spec = yaml.safe_load(Path("configs/round2_v33/SPEC.yaml").read_text())
    rows = [dict(target="tap_iron",recipe="GAUSS1",both_seeds_positive=True,paired_seed_summary={"mean":.02}),
            dict(target="tap_iron",recipe="MDN3",both_seeds_positive=True,paired_seed_summary={"mean":.01})]
    assert select_finalists(rows,spec) == {"tap_iron":None,"tap_time_len":None}
    rows[1]["paired_seed_summary"]["mean"] = .03
    assert select_finalists(rows,spec)["tap_iron"] == "MDN3"
    rows[1]["both_seeds_positive"] = False
    assert select_finalists(rows,spec)["tap_iron"] is None
