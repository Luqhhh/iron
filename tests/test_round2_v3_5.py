from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES
from bf_tap_r2.v3_1_search import canonical_trial_hash
from bf_tap_r2.v3_4_bags import protocol_bag_hash
from bf_tap_r2.v3_5_composition import (
    apply_s0_rule,
    s0_cap_records,
    select_g2_expert,
    select_s0_rule,
)
from bf_tap_r2.v3_5_models import (
    V35EBMRegressor,
    V35GlobalSpoutEBMRegressor,
    V35Regressor,
    evaluate_v35_outer_folds,
)
from bf_tap_r2.v3_5_run import v35_trial_identity
from bf_tap_r2.v3_5_sampler import (
    RAW_EBM_COLUMNS,
    build_conditional_extension_trials,
    mark_duplicate_slots,
    resolve_explicit_interactions,
    sample_v35,
    schedule_summary,
)


def _frame(n: int = 120, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(index=range(n))
    frame["sample_id"] = [f"S{i:06d}" for i in range(n)]
    frame["spout_no"] = np.asarray([(i % 4) + 1 for i in range(n)], dtype=int)
    for j, name in enumerate(FEATURES):
        frame[name] = 1.0 + rng.normal(0.0, 0.1, n) + 0.03 * j
    frame["tap_iron"] = 100.0 + 2.0 * rng.normal(size=n)
    frame["tap_time_len"] = 30.0 + 1.0 * rng.normal(size=n)
    if (frame[["tap_iron", "tap_time_len"]] <= 0).any().any():
        raise AssertionError("Synthetic labels must be positive for log1p")
    return frame


def _parent_ebm(target: str = "tap_iron", trial_id: str = "v34-s1-ebm_boundary-0016",
                leaf: int = 60, interactions: int = 10) -> dict:
    return {
        "trial_id": trial_id,
        "kind": "ebm_boundary",
        "target": target,
        "target_transform": "log1p",
        "feature_set": "raw",
        "parameters": {
            "max_bins": 128,
            "min_samples_leaf": int(leaf),
            "interactions": int(interactions),
            "max_interaction_bins": 32,
            "max_leaves": 2,
            "objective": "rmse",
            "learning_rate": 0.03,
            "outer_bags": 4,
            "inner_bags": 0,
            "max_rounds": 6000,
            "early_stopping_rounds": 100,
            "random_state": 42,
            "n_jobs": 1,
        },
    }


def _ebm_trial(target: str = "tap_iron", *, feature_set: str = "raw",
               interactions: int | list[tuple[int, int]] = 0,
               leaf: int = 5, max_rounds: int = 25) -> dict:
    return {
        "trial_id": f"v35-test-{target}-{feature_set}",
        "kind": "ebm_regularized",
        "target": target,
        "target_transform": "log1p",
        "feature_set": feature_set,
        "parameters": {
            "max_bins": 16,
            "min_samples_leaf": int(leaf),
            "interactions": interactions,
            "max_interaction_bins": 8,
            "max_leaves": 2,
            "objective": "rmse",
            "learning_rate": 0.2,
            "outer_bags": 4,
            "inner_bags": 0,
            "max_rounds": int(max_rounds),
            "early_stopping_rounds": 5,
            "random_state": 42,
            "n_jobs": 1,
        },
        "protocol": {"bags": "group-safe-bags-v1", "inner_splits": 5, "bag_seed": 42},
    }


def _fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "configs/round2_v3_5").mkdir(parents=True)
    shutil.copy("configs/round2_v3_5/search.yaml", root / "configs/round2_v3_5/search.yaml")
    ledger = root / "local/runs/round2-v3.4-ebm-and-constrained-composition/refine-r1/seed-42/fit_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    trials = [
        _parent_ebm("tap_iron", "v34-s1-ebm_boundary-0016", leaf=60, interactions=10),
        _parent_ebm("tap_iron", "v34-s1-ebm_boundary-0020", leaf=60, interactions=20),
        _parent_ebm("tap_time_len", "v34-s1-ebm_boundary-0089", leaf=60, interactions=10),
        _parent_ebm("tap_time_len", "v34-s1-ebm_boundary-0081", leaf=30, interactions=10),
    ]
    with ledger.open("w", encoding="utf-8") as handle:
        for trial in trials:
            handle.write(json.dumps({"event": "complete", "trial_id": trial["trial_id"], "trial": trial}) + "\n")
    return root


def test_v35_sampler_budget_counts_indices_and_na_rules(tmp_path):
    root = _fake_root(tmp_path)
    trials = sample_v35(root)
    summary = schedule_summary(trials)
    assert summary["slots"] == 256
    assert summary["available"] == 256
    assert summary["not_applicable"] == 0
    assert summary["duplicate"] == 0
    assert summary["line_counts"] == {"R": 144, "I": 64, "L": 48}
    assert {line: sum(t["line"] == line and t["target"] == "tap_iron" for t in trials)
            for line in ("R", "I", "L")} == {"R": 72, "I": 32, "L": 24}

    r_iron = [t for t in trials if t["line"] == "R" and t["target"] == "tap_iron"]
    assert {t["parameters"]["max_bins"] for t in r_iron} == {64, 128, 256}
    assert {t["parameters"]["min_samples_leaf"] for t in r_iron} == {60, 100, 160}
    assert {t["parameters"]["interactions"] for t in r_iron} == {2, 5, 10, 20}
    assert {t["parameters"]["max_interaction_bins"] for t in r_iron} == {16, 32}
    r_time = [t for t in trials if t["line"] == "R" and t["target"] == "tap_time_len"]
    assert {t["parameters"]["max_bins"] for t in r_time} == {128, 256, 512}

    i2 = [t for t in trials if t.get("interaction_strategy") == "I2"
          and t["feature_set"] == "raw" and t["target"] == "tap_iron"]
    assert i2
    expected_i2 = resolve_explicit_interactions(RAW_EBM_COLUMNS, [
        ("spout_no", "air_volume"), ("spout_no", "total_press_diff"),
        ("spout_no", "hot_air_temp"), ("spout_no", "fuel_rate"), ("spout_no", "pig"),
    ])
    assert i2[0]["parameters"]["interactions"] == expected_i2
    i3 = [t for t in trials if t.get("interaction_strategy") == "I3"
          and t["feature_set"] == "four" and t["target"] == "tap_iron"]
    assert i3
    assert len(i3[0]["parameters"]["interactions"]) == 10

    l_trials = [t for t in trials if t["line"] == "L"]
    assert len({t["parameters"]["component_key"] for t in l_trials}) == 16
    for beta in (0.15, 0.30, 0.50):
        group = [t for t in l_trials if t["beta"] == beta]
        assert len(group) == 16

    duplicate = deepcopy(trials[0])
    duplicate.pop("trial_id", None)
    marked = mark_duplicate_slots([trials[0], duplicate])
    assert marked[1]["status"] == "duplicate"
    assert marked[1]["duplicate_of"] == marked[0]["trial_id"]


def test_v35_explicit_interactions_resolve_dedupe_and_feature_order():
    columns = RAW_EBM_COLUMNS
    pairs = [("pig", "spout_no"), ("spout_no", "pig"), ("air_volume", "oxygen")]
    resolved = resolve_explicit_interactions(columns, pairs)
    assert resolved == sorted({tuple(sorted((columns.index("pig"), columns.index("spout_no")))),
                               tuple(sorted((columns.index("air_volume"), columns.index("oxygen"))))})
    with pytest.raises(KeyError):
        resolve_explicit_interactions(columns, [("missing", "pig")])


def test_s0_cap_records_constraints_and_selection():
    rng = np.random.default_rng(1234)
    n = 90
    y = {"42": 10.0 + rng.normal(size=n), "3407": 12.0 + rng.normal(size=n)}
    l1 = {seed: y[seed] + 0.5 * rng.normal(size=n) for seed in y}
    experts = {
        "a": {seed: y[seed] + 0.40 * rng.normal(size=n) for seed in y},
        "b": {seed: y[seed] + 0.55 * rng.normal(size=n) for seed in y},
    }
    result = s0_cap_records(y, l1, experts, g1_names=["a", "b"])
    assert len(result["rules"]) == 6
    assert len(result["old_cap_controls"]) == 2
    assert set(result["caps"]) == {0.65, 0.80, 1.00}
    for rule in result["rules"]:
        w = np.asarray(rule["weights"], dtype=float)
        assert (w >= -1e-9).all()
        assert abs(w.sum() - 1.0) < 1e-7
        assert rule["new_weight_total"] <= rule["cap"] + 1e-7
    best = select_s0_rule(result["rules"])
    assert best["group"] in {"G1", "G2"}
    applied = apply_s0_rule(l1, experts, best)
    assert set(applied) == set(y)
    assert applied["42"].shape == (n,)
    assert select_g2_expert(y, experts, ["a", "b"])["selected"] in {"a", "b"}


def test_v35_ebm_fit_serialization_order_and_explicit_interactions():
    frame = _frame(100)
    trial = _ebm_trial("tap_iron", feature_set="raw", interactions=[(0, 21), (0, 3)], max_rounds=20)
    model = V35EBMRegressor(trial).fit(frame, frame["tap_iron"].to_numpy(dtype=float))
    pred = model.predict(frame)
    assert pred.shape == (len(frame),)
    assert np.isfinite(pred).all()
    assert model.fit_meta_["interactions"] == [(0, 21), (0, 3)]
    restored = pickle.loads(pickle.dumps(model))
    assert np.allclose(pred, restored.predict(frame), rtol=0.0, atol=1e-12)
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    shuffled_pred = model.predict(shuffled)
    aligned = pd.Series(shuffled_pred, index=shuffled["sample_id"]).loc[frame["sample_id"]].to_numpy()
    assert np.allclose(pred, aligned, rtol=0.0, atol=1e-12)

    four = V35EBMRegressor(_ebm_trial("tap_iron", feature_set="four", interactions=0, max_rounds=15))
    four.fit(frame, frame["tap_iron"].to_numpy(dtype=float))
    assert four.input_columns_[-4:] == (
        "oxygen_per_air_volume", "pressure_per_air_volume", "thermal_difference", "upper_pressure_fraction"
    )
    assert np.isfinite(four.predict(frame)).all()


def test_v35_global_local_ebm_beta_zero_fallback_and_serialization():
    frame = _frame(120)
    frame["spout_no"] = np.repeat([1, 2, 3, 4], [50, 40, 20, 10])
    y = frame["tap_iron"].to_numpy(dtype=float)
    parent = _parent_ebm("tap_iron", leaf=5, interactions=0)
    common = {
        "kind": "global_spout_ebm",
        "target": "tap_iron",
        "parameters": {
            "parent_trial": parent,
            "local_min_samples_leaf": 5,
            "local_interactions": 0,
            "beta": 0.30,
            "min_spout_samples": 25,
            "local_include_spout": True,
            "feature_set": "raw",
        },
    }
    model = V35GlobalSpoutEBMRegressor(common).fit(frame, y)
    # Spout 4 has 10 rows, below the 25-row threshold, and must fall back.
    assert 4 not in model.local_models_
    assert model.fit_meta_["local_errors"][4] == "below_min_spout_samples"
    pred = model.predict(frame)
    assert np.isfinite(pred).all()
    unseen = frame.iloc[[0]].copy().reset_index(drop=True)
    unseen["spout_no"] = 99
    assert np.allclose(model.predict(unseen), model.global_model_.predict(unseen), rtol=0.0, atol=1e-12)
    roundtrip = pickle.loads(pickle.dumps(model))
    assert np.allclose(pred, roundtrip.predict(frame), rtol=0.0, atol=1e-12)

    beta_zero = deepcopy(common)
    beta_zero["parameters"]["beta"] = 0.0
    zero = V35GlobalSpoutEBMRegressor(beta_zero).fit(frame, y)
    assert np.allclose(zero.predict(frame), zero.global_model_.predict(frame), rtol=0.0, atol=1e-12)

    dispatcher = V35Regressor(common)
    with pytest.raises(RuntimeError):
        dispatcher.predict(frame)


def test_v35_outer_folds_prediction_no_fit_and_identity_components():
    frame = _frame(90)
    folds = np.asarray([i % 5 for i in range(len(frame))], dtype=int)
    trial = _ebm_trial("tap_iron", interactions=0, max_rounds=15)
    result = evaluate_v35_outer_folds(frame, folds, trial, fold_ids=(0, 1))
    assert result["pooled_wmape"] > 0
    assert set(result["fold_scores"]) == {"0", "1"}
    assert len(result["fit_meta"]) == 2
    assert np.isfinite(result["predictions"][np.isin(folds, [0, 1])]).all()
    assert np.isnan(result["predictions"][~np.isin(folds, [0, 1])]).all()

    frame_for_bag = _frame(60)
    bag = protocol_bag_hash(frame_for_bag, n_outer_bags=4, n_inner_splits=5, seed=42)
    kwargs = dict(batch_id="b", data_hash="d", fold_hash="f", fold_ids=(0, 1),
                  model_seed=42, stage="coarse", code_version="c", source_hash="s")
    first = v35_trial_identity(trial, bag_hash=bag, **kwargs)
    second = v35_trial_identity(trial, bag_hash="other", **kwargs)
    assert first != second


def test_conditional_extension_budget_shape():
    centers = [_ebm_trial("tap_iron", interactions=5)]
    centers[0]["trial_id"] = "center-1"
    slots = build_conditional_extension_trials(centers, max_slots=64, slots_per_direction=32)
    assert 0 < len(slots) <= 64
    assert all(slot.get("extension_of") == "center-1" for slot in slots)
