from copy import deepcopy
import csv
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.low_signal_cv import sensitivity
from bf_tap_r2.models import inputs
from bf_tap_r2.v2_refinement import (L15Regressor, choose_c2, combine_same_fold,
    fit_once, isolated_payload, new_model, spec_at)

ROOT = Path(__file__).resolve().parents[1]


def metric(value):
    return {"wmape": value, "by_fold": {str(i): value for i in range(5)}, "by_spout": {"1": value, "2": value}}


def test_c2_gate_does_not_promote_merely_beating_median():
    spec = spec_at(ROOT)
    scores = {r: {str(s): {t: metric(.10 if r != "C2" else .04) for t in TARGETS} for s in spec["split_seeds"]} for r in spec["routes"]}
    assert set(choose_c2(scores, spec)[1].values()) == {"C2"}
    for s in spec["split_seeds"]:
        scores["B3_C2_SEED_ENSEMBLE"][str(s)][TARGETS[0]] = metric(.03)
        scores["D4_CATBOOST_SHALLOW"][str(s)][TARGETS[0]] = metric(.0300005)
    # Within the frozen tolerance, lower estimated inference cost wins.
    assert choose_c2(scores, spec)[1][TARGETS[0]] == "D4_CATBOOST_SHALLOW"
    scores["D4_CATBOOST_SHALLOW"]["42"][TARGETS[0]]["by_spout"]["2"] = .042
    assert choose_c2(scores, spec)[1][TARGETS[0]] == "B3_C2_SEED_ENSEMBLE"


def test_ensemble_rejects_cross_fold_models():
    members = {"a": (0, [1., 2.]), "b": (0, [3., 4.]), "c": (0, [5., 6.])}
    np.testing.assert_array_equal(combine_same_fold(members, list(members), 0), [3., 4.])
    members["c"] = (1, [5., 6.])
    with pytest.raises(ValueError, match="same held-out fold"):
        combine_same_fold(members, list(members), 0)


@pytest.mark.parametrize("target", TARGETS)
def test_isolated_package_preserves_other_column_strings(target):
    ids = [f"R2S2_TEST_{i:012X}" for i in range(322)]
    text = "sample_id,pred_tap_iron,pred_tap_time_len\n" + "".join(f"{sid},5.2300000000000004e+02,00124.12345678901234\n" for sid in ids)
    payload = isolated_payload(text.encode(), ids, target, np.arange(322)/7)
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    col = "pred_" + next(t for t in TARGETS if t != target)
    originals = list(csv.DictReader(io.StringIO(text)))
    assert [r[col] for r in rows] == [r[col] for r in originals]
    assert [float(r["pred_"+target]) for r in rows] == list(np.arange(322)/7)


def test_seed_and_d4_overrides_leave_legacy_config_untouched():
    spec = spec_at(ROOT)
    legacy = yaml.safe_load((ROOT / spec["legacy_catboost_config"]).read_text())
    before = deepcopy(legacy)
    seed = new_model("C2_2026", spec, legacy)
    d4 = new_model("D4", spec, legacy)
    assert seed.config["execution"]["model_seed"] == 2026
    assert seed.config["models"]["C2"] == legacy["models"]["C2"]
    assert d4.config["models"]["C2"]["depth"] == 4
    assert d4.config["models"]["C2"]["iterations"] == 3000
    assert legacy == before


def synthetic():
    rng = np.random.default_rng(2026)
    frame = pd.DataFrame({f: rng.normal(size=120) for f in FEATURES})
    frame["spout_no"] = np.tile([1, 2], 60)
    frame["sample_id"] = [f"synthetic_{i}" for i in range(120)]
    frame["tap_iron"] = 500 + 40*frame.air_volume + 20*frame.oxygen
    frame["tap_time_len"] = 100 + frame.air_volume
    return frame


def test_lightgbm_training_categories_persist_and_order_is_invariant(tmp_path):
    # Synthetic engineering fit 1/2; deliberately small, outside real-data budget.
    frame = synthetic()
    config = {**spec_at(ROOT)["lightgbm"], "n_estimators": 30, "min_child_samples": 5}
    model = L15Regressor(config).fit(frame.iloc[:90], frame.tap_iron.iloc[:90])
    assert model.categories_ == [1, 2]
    test = frame.iloc[90:].copy()
    before = test.copy(deep=True)
    prediction = model.predict(test)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(prediction, joblib.load(path).predict(test.iloc[::-1])[::-1])
    test.loc[test.index[0], "spout_no"] = 3
    assert model.transform(test).spout_no.isna().iloc[0]
    assert model.categories_ == [1, 2]
    assert np.isfinite(model.predict(test)).all()
    pd.testing.assert_frame_equal(frame.iloc[90:], before)


def test_catboost_tree_limit_full_matches_normal_prediction():
    # Synthetic engineering fit 2/2; no CV truncation selection.
    spec = spec_at(ROOT)
    legacy = yaml.safe_load((ROOT / spec["legacy_catboost_config"]).read_text())
    legacy["models"]["C2"]["iterations"] = 30
    legacy["models"]["C2"]["depth"] = 2
    frame = synthetic()
    model = new_model("C2_42", spec, legacy).fit(frame.iloc[:90], frame.tap_iron.iloc[:90])
    valid = frame.iloc[90:]
    np.testing.assert_array_equal(model.predict(valid), model.estimator_.predict(inputs(valid, True), ntree_end=30))
    assert not np.array_equal(model.predict(valid), model.estimator_.predict(inputs(valid, True), ntree_end=10))


def test_bootstrap_uses_identical_sample_indices_across_split_seeds():
    frame = synthetic()
    p = frame.tap_iron.to_numpy() + np.arange(len(frame))/5
    baseline = frame.tap_iron.to_numpy() + 5
    distribution, _ = sensitivity(frame, "tap_iron", {42: p, 3407: p}, {42: baseline, 3407: baseline}, {"seed": 42, "repetitions": 20})
    np.testing.assert_array_equal(distribution["42"], distribution["3407"])


def test_budget_counts_failed_starts(tmp_path):
    (tmp_path / "fit_ledger.jsonl").write_text(json.dumps({"stage": "cv", "event": "start"}) + "\n")
    with pytest.raises(ValueError, match="budget"):
        fit_once(None, None, None, tmp_path / "absent.joblib", tmp_path, "cv", {}, 1)
