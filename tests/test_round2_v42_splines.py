import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bf_tap_r2.data import FEATURES
from bf_tap_r2.v42_splines import (SplinePreprocessor, SplineRegressor, basis_matrix,
                                  deletion_costs, fit_partition, least_squares, pair_gains)
from bf_tap_r2.v42_run import select_finalists


def settings():
    s = yaml.safe_load((Path(__file__).parents[1]/"configs/round2_v42/SPEC.yaml").read_text())["training"]
    s.update(max_terms=9, min_side_rows=3, selection_counts=[1, 5, 9])
    return s


def frame(n=80):
    x = np.random.default_rng(41).normal(size=(n, len(FEATURES)))
    result = pd.DataFrame(x, columns=FEATURES)
    result["sample_id"] = [f"s{i}" for i in range(n)]
    result["spout_no"] = np.arange(n)%2+1
    return result


@pytest.mark.parametrize("redundant", [False, True])
def test_pair_gain_and_deletion_match_independent_lstsq(redundant):
    rng = np.random.default_rng(33)
    b = np.column_stack([np.ones(50), rng.normal(size=(50, 4))])
    if redundant:
        b = np.column_stack([b, b[:, 1]+b[:, 2]])
    y = rng.normal(size=50)
    coef, rss, q, singular, right = least_squares(b, y, 1e-10)
    a, c = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    c[:, 1] = a[:, 1]  # A rank-one pair.
    gains = pair_gains(q, y-b@coef, a, c, 1e-10)
    for k in range(3):
        full = np.column_stack([b, a[:, k], c[:, k]])
        fitted = full@np.linalg.lstsq(full, y, rcond=1e-10)[0]
        assert gains[k] == pytest.approx(rss-np.sum((y-fitted)**2), abs=1e-10)
    costs = deletion_costs(coef, singular, right)
    for k in range(1, b.shape[1]):
        reduced = np.delete(b, k, axis=1)
        fitted = reduced@np.linalg.lstsq(reduced, y, rcond=1e-10)[0]
        assert costs[k] == pytest.approx(np.sum((y-fitted)**2)-rss, abs=1e-10)


def test_product_basis_has_real_interaction():
    x = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    values = basis_matrix(x, [[], [[0, .2, 1], [1, .3, 1]]])[:, 1]
    assert values[-1] == pytest.approx(.56)
    assert np.array_equal(values[:3], np.zeros(3))


def test_train_only_preprocessing_and_unknown_categories():
    train, query = frame(), frame(4)
    pre = SplinePreprocessor().fit(train)
    frozen = json.dumps(pre.metadata(), sort_keys=True)
    query.loc[:, list(FEATURES)] = 10000
    query["spout_no"] = 99
    assert not pre.transform(query)[:, len(FEATURES):].any()
    assert json.dumps(pre.metadata(), sort_keys=True) == frozen
    assert np.array_equal(pre.transform(query), SplinePreprocessor.restore(pre.metadata()).transform(query))


def test_path_selection_refit_and_roundtrip(tmp_path):
    data = frame(120)
    y = 10+3*np.maximum(data[FEATURES[0]].to_numpy(), 0)
    model = SplineRegressor("PAIR", settings()).fit(data.iloc[:80], y[:80], validation=(data.iloc[80:], y[80:]))
    best = min(model.selection_, key=lambda r: (r["calibration_mae"], r["count"]))
    assert model.selected_["count"] == best["count"]
    assert all(a["rss"] <= b["rss"]+1e-9 for a,b in zip(model.path_, model.path_[1:]))
    assert all(0 in row["active"] for row in model.path_)
    model.save(tmp_path/"model.json")
    restored = SplineRegressor.load(tmp_path/"model.json")
    assert np.array_equal(model.predict(data), restored.predict(data))
    assert np.allclose(model.predict(data), model.predict(data.iloc[::-1])[::-1], atol=1e-12)
    with pytest.raises(FileExistsError): model.save(tmp_path/"model.json")
    assert np.mean(np.abs(y[80:]-model.predict(data.iloc[80:]))) < .5


def test_query_labels_rejected_before_fitting():
    data=frame(20); query=data.iloc[18:].copy();query["tap_iron"]=1
    with pytest.raises(ValueError, match="Query labels"):
        fit_partition(data.iloc[:12], data.iloc[12:18], data.iloc[:18], query,
                      "tap_iron", "PAIR", {}, None, [0, 1])


def test_control_and_small_gain_cannot_be_selected():
    spec={"promotion":{"eligible_recipes":["PAIR"],"development_mean_gain_ge":.01},
          "tie_preference_by_target":{t:["ADDITIVE","PAIR"] for t in ["tap_iron","tap_time_len"]}}
    rows=[{"target":"tap_iron","recipe":r,"both_seeds_positive":True,
           "paired_seed_summary":{"mean":v}} for r,v in [("ADDITIVE",.009),("PAIR",.008)]]
    assert select_finalists(rows,spec)["tap_iron"] is None
    rows[1]["paired_seed_summary"]["mean"]=.02
    assert select_finalists(rows,spec)["tap_iron"] == "PAIR"
