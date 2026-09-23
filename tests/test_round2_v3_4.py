from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.v3_1_search import canonical_trial_hash
from bf_tap_r2.v3_4_bags import (
    build_group_safe_bags,
    group_safe_inner_folds,
    protocol_bag_hash,
)
from bf_tap_r2.v3_4_composition import (
    constrained_forward_select,
    simple_mix_alphas,
    s0_rule_records,
)
from bf_tap_r2.v3_4_models import (
    ShrunkSpoutRegressor,
    V34EBMRegressor,
    V34ResidualRegressor,
    evaluate_v34_outer_folds,
)
from bf_tap_r2.v3_4_sampler import (
    build_conditional_extension_trials,
    mark_duplicate_slots,
    sample_v34,
    schedule_summary,
)
from bf_tap_r2.v3_4_run import v34_trial_identity


def _frame(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(index=range(n))
    frame["sample_id"] = [f"S{i:06d}" for i in range(n)]
    frame["spout_no"] = np.asarray([(i % 4) + 1 for i in range(n)], dtype=int)
    for j, name in enumerate(FEATURES):
        frame[name] = 1.0 + rng.normal(0.0, 0.1, n) + 0.03 * j
    frame["tap_iron"] = 100.0 + 2.0 * rng.normal(size=n)
    frame["tap_time_len"] = 30.0 + 1.0 * rng.normal(size=n)
    if (frame["tap_iron"] <= 0).any() or (frame["tap_time_len"] <= 0).any():
        raise AssertionError("Synthetic labels must stay positive for target transforms")
    return frame


def _catboost_params(iterations: int = 20, depth: int = 2) -> dict:
    return {
        "task_type": "CPU",
        "loss_function": "RMSE",
        "depth": int(depth),
        "iterations": int(iterations),
        "learning_rate": 0.2,
        "l2_leaf_reg": 3.0,
        "random_strength": 0.0,
        "random_seed": 42,
        "thread_count": 1,
        "cat_features": ["spout_no"],
        "allow_writing_files": False,
        "verbose": False,
        "bootstrap_type": "No",
    }


def _parent_trial(target: str = "tap_iron", iterations: int = 20) -> dict:
    return {
        "trial_id": f"v-test-{target}",
        "family": "catboost",
        "target": target,
        "feature_set": "raw",
        "target_transform": "identity",
        "parameters": _catboost_params(iterations=iterations),
    }


def _ebm_trial(target: str = "tap_iron", *, transform: str = "identity",
               max_rounds: int = 25, outer_bags: int = 4) -> dict:
    return {
        "trial_id": f"v-test-ebm-{target}",
        "kind": "ebm_boundary",
        "target": target,
        "target_transform": transform,
        "parameters": {
            "max_bins": 16,
            "min_samples_leaf": 5,
            "interactions": 0,
            "max_interaction_bins": 8,
            "max_leaves": 2,
            "objective": "rmse",
            "learning_rate": 0.2,
            "outer_bags": int(outer_bags),
            "inner_bags": 0,
            "max_rounds": int(max_rounds),
            "early_stopping_rounds": 5,
            "random_state": 42,
            "n_jobs": 1,
        },
        "protocol": {"bags": "group-safe-bags-v1", "inner_splits": 5, "bag_seed": 42},
    }


def test_group_safe_bags_shape_isolation_and_order_independence():
    frame = _frame(120)
    info = build_group_safe_bags(frame)
    bags = info["bags"]
    assert bags.shape == (4, len(frame))
    assert bags.dtype == np.int8
    assert set(np.unique(bags).tolist()) == {-1, 1}
    assert all((bags[i] == 1).any() and (bags[i] == -1).any() for i in range(4))
    groups = info["group_id"]
    for row in range(4):
        for group in set(groups):
            values = bags[row][groups == group]
            assert len(set(values.tolist())) == 1
    # Fold 4 is never an internal validation fold in this four-bag protocol.
    assert all((bags[i][info["fold"] == 4] == 1).all() for i in range(4))
    assert len(info["bag_hash"]) == 64

    shuffled = frame.sample(frac=1.0, random_state=11).reset_index(drop=True)
    shuffled_info = build_group_safe_bags(shuffled)
    left = pd.Series(info["fold"], index=frame["sample_id"])
    right = pd.Series(shuffled_info["fold"], index=shuffled["sample_id"])
    assert left.sort_index().equals(right.sort_index())


def test_ebm_group_safe_bags_fit_predict_order_and_serialization():
    frame = _frame(100)
    y = frame["tap_iron"].to_numpy(dtype=float)
    model = V34EBMRegressor(_ebm_trial("tap_iron")).fit(frame, y)
    pred = model.predict(frame)
    assert pred.shape == (len(frame),)
    assert np.isfinite(pred).all()
    assert len(model.bag_hash_) == 64
    assert model.fit_meta_["bag_validation_counts"] == [20, 20, 20, 20]
    assert "best_iteration" in model.fit_meta_
    assert model.fit_meta_["configured_max_rounds"] == 25

    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    shuffled_pred = model.predict(shuffled)
    restored = pd.Series(shuffled_pred, index=shuffled["sample_id"]).loc[frame["sample_id"]].to_numpy()
    assert np.allclose(pred, restored, rtol=0.0, atol=1e-12)

    roundtrip = pickle.loads(pickle.dumps(model))
    assert np.allclose(pred, roundtrip.predict(frame), rtol=0.0, atol=1e-12)


def test_ebm_residual_alpha_zero_parent_restore_and_inverse_transform():
    frame = _frame(90)
    y = frame["tap_iron"].to_numpy(dtype=float)
    base = _ebm_trial("tap_iron", transform="mean")
    base_snapshot = deepcopy(base)
    trial = {
        "kind": "ebm_residual",
        "target": "tap_iron",
        "parameters": {
            "base_trial": deepcopy(base),
            "coordinate": "original_unit",
            "corrector": {"kind": "ridge", "name": "ridge_10", "params": {"alpha": 10.0}},
            "alpha": 0.0,
            "inner_seed": 7,
            "inner_splits": 2,
        },
    }
    model = V34ResidualRegressor(trial)
    assert model.base_trial == base_snapshot
    fitted = model.fit(frame, y)
    alpha_zero = fitted.predict(frame)
    full_base = fitted.base_model_.predict(frame)
    assert np.allclose(alpha_zero, full_base, rtol=0.0, atol=1e-12)
    assert np.isfinite(alpha_zero).all()
    assert "mean" in fitted.base_model_.target_state_


def test_shrink_spout_fallback_and_beta_zero():
    frame = _frame(120)
    # Force a small spout that must fall back to the global model.
    frame["spout_no"] = np.repeat([1, 2, 3, 4], [40, 40, 30, 10])
    y = frame["tap_iron"].to_numpy(dtype=float)
    parent = _parent_trial("tap_iron", iterations=15)
    common = {
        "kind": "global_spout_shrink",
        "target": "tap_iron",
        "parameters": {
            "parent_trial": deepcopy(parent),
            "local_l2_multiplier": 1.0,
            "beta": 0.25,
            "min_spout_samples": 25,
            "local_include_spout": True,
        },
    }
    model = ShrunkSpoutRegressor(common).fit(frame, y)
    assert 4 not in model.local_models_
    assert model.local_available_[4] is False
    assert set(model.local_models_) == {1, 2, 3}
    pred = model.predict(frame)
    assert np.isfinite(pred).all()
    unseen = frame.iloc[[0]].copy().reset_index(drop=True)
    unseen["spout_no"] = 9
    global_unseen = model.global_model_.predict(unseen)
    assert np.allclose(model.predict(unseen), global_unseen, rtol=0.0, atol=1e-12)

    beta_zero = deepcopy(common)
    beta_zero["parameters"]["beta"] = 0.0
    zero_model = ShrunkSpoutRegressor(beta_zero).fit(frame, y)
    assert np.allclose(zero_model.predict(frame), zero_model.global_model_.predict(frame),
                       rtol=0.0, atol=1e-12)


def test_constrained_composition_and_s0_records():
    rng = np.random.default_rng(123)
    y = {
        "42": 10.0 + rng.normal(size=60),
        "3407": 12.0 + rng.normal(size=60),
    }
    l1 = {seed: y[seed] + 0.4 * rng.normal(size=60) for seed in y}
    ebm = {}
    for name, scale in (("a", 0.5), ("b", 0.6), ("c", 0.8), ("d", 1.0)):
        ebm[name] = {seed: y[seed] + scale * rng.normal(size=60) for seed in y}
    result = constrained_forward_select(
        y,
        {seed: np.column_stack([l1[seed], ebm["a"][seed], ebm["b"][seed], ebm["c"][seed], ebm["d"][seed]])
         for seed in y},
        names=["L1", "a", "b", "c", "d"],
        l1_index=0,
        candidate_indices=[1, 2, 3, 4],
        max_new_experts=2,
        max_new_weight=0.5,
    )
    weights = np.asarray(result["weights"], dtype=float)
    assert result["columns"][0] == 0
    assert len(result["new_experts"]) <= 2
    assert abs(weights.sum() - 1.0) < 1e-7
    assert (weights >= -1e-12).all()
    assert result["new_weight_total"] <= 0.5 + 1e-7
    assert "baseline_l1_objective" in result

    scan = simple_mix_alphas(y, l1, ebm["a"], alphas=[0.0, 0.5])
    assert scan[0]["mean_wmape"] == pytest.approx(np.mean([
        np.abs(y[s] - l1[s]).sum() / np.abs(y[s]).sum() for s in y
    ]))
    records = s0_rule_records(y, l1, ebm, alphas=(0.1, 0.25, 0.5))
    assert records
    assert all(0.0 <= row["alpha"] <= 1.0 for row in records)
    assert max(row["mean_wmape"] for row in records) <= max(
        np.abs(y[s] - l1[s]).sum() / np.abs(y[s]).sum() for s in y
    ) + 1e-6


def _write_complete_ledger(path: Path, trial: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"event": "complete", "trial_id": trial["trial_id"], "trial": trial}
    path.write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")


def _fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "configs/round2_v3_4").mkdir(parents=True)
    shutil.copy("configs/round2_v3_4/search.yaml", root / "configs/round2_v3_4/search.yaml")

    def parent(trial_id: str, target: str) -> dict:
        return {
            "trial_id": trial_id,
            "family": "catboost",
            "target": target,
            "feature_set": "raw",
            "target_transform": "identity",
            "parameters": _catboost_params(iterations=5),
        }

    v31_trials = [
        parent("v31-s1-expr-iron-0018", "tap_iron"),
        parent("v31-s1-time-0021-0050", "tap_time_len"),
    ]
    v32_trials = [
        parent("v32-s1-iron_log_expression-0019", "tap_iron"),
        parent("v32-s1-time_neighborhood-0126", "tap_time_len"),
    ]
    v31_path = root / "local/runs/round2-v3.1-directed-search/s1-coarse-r1/fit_ledger.jsonl"
    v32_path = root / "local/runs/round2-v3.2-ensemble-and-target-search/coarse-r1/fit_ledger.jsonl"
    v31_path.parent.mkdir(parents=True, exist_ok=True)
    v32_path.parent.mkdir(parents=True, exist_ok=True)
    with v31_path.open("w", encoding="utf-8") as handle:
        for trial in v31_trials:
            handle.write(json.dumps({"event": "complete", "trial_id": trial["trial_id"], "trial": trial}) + "\n")
    with v32_path.open("w", encoding="utf-8") as handle:
        for trial in v32_trials:
            handle.write(json.dumps({"event": "complete", "trial_id": trial["trial_id"], "trial": trial}) + "\n")

    def ebm_spec(trial_id: str, target: str, transform: str, *, leaf: int = 30) -> dict:
        return {
            "trial_id": trial_id,
            "kind": "ebm",
            "target": target,
            "target_transform": transform,
            "parameters": {
                "max_bins": 128,
                "min_samples_leaf": leaf,
                "interactions": 10,
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

    iron_first = ebm_spec("v33-s1-ebm-0095", "tap_iron", "log1p", leaf=30)
    iron_second = ebm_spec("v33-s1-ebm-0087", "tap_iron", "log1p", leaf=10)
    time_first = ebm_spec("v33-s1-ebm-0127", "tap_time_len", "log1p", leaf=30)
    refine_dir = root / "local/runs/round2-v3.3-structure-search/refine-r1"
    refine_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        {"trial_id": iron_first["trial_id"], "line": "ebm", "target": "tap_iron", "mean_wmape": 0.038},
        {"trial_id": iron_second["trial_id"], "line": "ebm", "target": "tap_iron", "mean_wmape": 0.039},
        {"trial_id": time_first["trial_id"], "line": "ebm", "target": "tap_time_len", "mean_wmape": 0.040},
    ]
    selection = {
        "tap_iron": [iron_first, iron_second],
        "tap_time_len": [time_first],
    }
    (refine_dir / "refine_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (refine_dir / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    return root


def test_v34_sampler_schedule_centers_na_and_duplicate_rules(tmp_path):
    root = _fake_root(tmp_path)
    trials = sample_v34(root)
    summary = schedule_summary(trials)
    assert summary["slots"] == 144
    assert summary["available"] == 136
    assert summary["not_applicable"] == 8
    assert summary["duplicate"] == 0
    assert summary["line_counts"] == {"ebm_boundary": 96, "ebm_residual": 32, "global_spout_shrink": 16}
    time_boundary = [t for t in trials if t["line"] == "ebm_boundary" and t["target"] == "tap_time_len"]
    assert time_boundary and all(t["target_transform"] == "log1p" for t in time_boundary)
    iron_boundary = [t for t in trials if t["line"] == "ebm_boundary" and t["target"] == "tap_iron"]
    assert {t["target_transform"] for t in iron_boundary} == {"log1p", "mean"}
    na = [t for t in trials if t.get("status") == "not_applicable"]
    assert len(na) == 8
    assert all(t["target"] == "tap_time_len" for t in na)

    duplicate = deepcopy(trials[0])
    duplicate.pop("trial_id", None)
    marked = mark_duplicate_slots([trials[0], duplicate])
    assert marked[1]["status"] == "duplicate"
    assert marked[1]["duplicate_of"] == marked[0]["trial_id"]


def test_cache_identity_changes_with_bag_protocol_and_source():
    trial = _ebm_trial("tap_iron")
    kwargs = dict(batch_id="b", data_hash="d", fold_hash="f", fold_ids=(0, 1),
                  model_seed=42, stage="coarse", code_version="c", source_hash="s")
    first = v34_trial_identity(trial, bag_hash="bag-a", **kwargs)
    second = v34_trial_identity(trial, bag_hash="bag-b", **kwargs)
    assert first != second
    mutated = deepcopy(trial)
    mutated["parameters"]["learning_rate"] = 0.99
    third = v34_trial_identity(mutated, bag_hash="bag-a", **kwargs)
    assert first != third
    assert canonical_trial_hash(trial) != canonical_trial_hash(mutated)




def test_conditional_extension_changes_only_unscanned_dimensions():
    center = _ebm_trial("tap_iron")
    center["trial_id"] = "center-1"
    center["parameters"]["learning_rate"] = 0.03
    center["parameters"]["max_rounds"] = 6000
    slots = build_conditional_extension_trials([center])
    assert len(slots) == 8
    base_params = center["parameters"]
    for slot in slots:
        params = slot["parameters"]
        assert params["max_bins"] == base_params["max_bins"]
        assert params["interactions"] == base_params["interactions"]
        assert params["max_interaction_bins"] == base_params["max_interaction_bins"]
        changed = (params["learning_rate"] != base_params["learning_rate"]
                   or params["max_rounds"] != base_params["max_rounds"])
        assert changed


def test_outer_fold_evaluation_predictions_are_full_length():
    frame = _frame(80)
    fold = np.asarray([(i % 5) for i in range(len(frame))], dtype=int)
    result = evaluate_v34_outer_folds(frame, fold, _ebm_trial("tap_iron", max_rounds=15), fold_ids=(0, 1))
    predictions = result["predictions"]
    assert predictions.shape == (len(frame),)
    assert np.isfinite(predictions[(fold == 0) | (fold == 1)]).all()
    assert np.isnan(predictions[(fold == 2) | (fold == 3) | (fold == 4)]).all()
    assert len(result["fit_meta"]) == 2
    assert all("bag_hash" in meta for meta in result["fit_meta"])
