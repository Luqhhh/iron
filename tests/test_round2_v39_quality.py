import json
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from bf_tap_r2.data import FEATURES
from bf_tap_r2.v7_periodic import digest
from bf_tap_r2.v39_regressor import TreePreprocessor, TreeRegressor, fit_partition
from bf_tap_r2.v37_hard_tree import HardTreeEnsemble
from bf_tap_r2.v39_run import read_reference, select_finalists


def frame(n=20):
    result = pd.DataFrame(np.random.default_rng(7).normal(size=(n, len(FEATURES))), columns=FEATURES)
    result["sample_id"] = [f"s{i}" for i in range(n)]
    result["spout_no"] = np.arange(n)%2+1
    return result


def test_train_only_quantiles_unknown_categories_and_persistence(tmp_path):
    train = frame()
    preprocessor = TreePreprocessor().fit(train)
    quantiles = preprocessor.transformer_.quantiles_.copy()
    query = frame(4)
    query.loc[:, list(FEATURES)] = 10000
    query["spout_no"] = 99
    encoded = preprocessor.transform(query)
    assert not encoded[:, len(FEATURES):].any()
    assert np.array_equal(quantiles, preprocessor.transformer_.quantiles_)
    restored = TreePreprocessor.from_metadata(preprocessor.metadata())
    assert np.array_equal(encoded, restored.transform(query))
    settings = {"model": dict(n_estimators=8, depth=3, dropout=.2, selected_variables=.8),
                "prediction_chunk_rows": 256}
    model = TreeRegressor("INSTANCE", settings)
    model.preprocessor_, model.mean_, model.std_ = preprocessor, 100., 20.
    model.model_ = HardTreeEnsemble(len(FEATURES)+2, "INSTANCE", settings["model"])
    model.save(tmp_path/"model.pt")
    cold = TreeRegressor.load(tmp_path/"model.pt")
    assert np.array_equal(model.predict(query), cold.predict(query))
    with pytest.raises(FileExistsError):
        model.save(tmp_path/"model.pt")


def test_current_reference_is_reweighted_from_verified_old_components(tmp_path):
    train, query = frame(4), frame(3)
    components = {"v36_iron": np.array([1., 2., 3.]), "v12_iron": np.array([2., 4., 6.]),
                  "v36_time": np.array([10., 20., 30.]), "n_time": np.array([20., 40., 60.]),
                  "v7_time": np.array([5., 10., 15.])}
    old = .325*components["v36_time"]+.175*components["n_time"]+.5*components["v7_time"]
    np.savez(tmp_path/"predictions.npz", **components,
             tap_iron=.5*components["v36_iron"]+.5*components["v12_iron"], tap_time_len=old,
             query_ids=query.sample_id.to_numpy(dtype=str))
    (tmp_path/"metadata.json").write_text(json.dumps({"fit_ids_digest": digest(train.sample_id.tolist())}))
    values, _ = read_reference(tmp_path, train, query, {})
    assert np.array_equal(values["historical_b0_time"], old)
    assert np.array_equal(values["tap_time_len"], .2*components["v36_time"]+.3*components["n_time"]+.5*components["v7_time"])
    assert not np.array_equal(values["tap_time_len"], old)


def test_query_labels_are_rejected_before_an_optimizer_can_start():
    full = frame(10)
    query = full.iloc[8:].copy(); query["tap_iron"] = 1
    with pytest.raises(ValueError, match="Query labels"):
        fit_partition(full.iloc[:6], full.iloc[6:8], full.iloc[:8], query,
                      "tap_iron", "INSTANCE", {}, None, [0, 1])


def test_control_and_small_mean_cannot_be_promoted():
    spec = {"promotion": {"eligible_recipes": ["INSTANCE"], "development_mean_gain_ge": .01},
            "tie_preference_by_target": {t: ["GLOBAL", "INSTANCE"] for t in ("tap_iron", "tap_time_len")}}
    rows = [{"target": "tap_iron", "recipe": r, "both_seeds_positive": True,
             "paired_seed_summary": {"mean": v}} for r, v in [("GLOBAL", .004), ("INSTANCE", .009)]]
    assert select_finalists(rows, spec)["tap_iron"] is None
    rows[1]["paired_seed_summary"]["mean"] = .011
    assert select_finalists(rows, spec)["tap_iron"] == "INSTANCE"
    rows[0]["paired_seed_summary"]["mean"] = .012
    assert select_finalists(rows, spec)["tap_iron"] is None
