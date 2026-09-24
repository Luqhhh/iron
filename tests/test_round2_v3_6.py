from __future__ import annotations

import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES
from bf_tap_r2.v3_6_models import V36EBMRegressor, V36Regressor, evaluate_v36_outer_folds
from bf_tap_r2.v3_6_networks import V36NetworkRegressor
from bf_tap_r2.v3_6_sampler import (
    TARGET_LINE_COUNTS,
    load_v36_config,
    resolve_parent_centers,
    sample_v36,
    schedule_summary,
)


def _frame(n: int = 90, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(index=range(n))
    frame["sample_id"] = [f"S{i:06d}" for i in range(n)]
    frame["spout_no"] = np.asarray([(i % 3) + 1 for i in range(n)], dtype=int)
    for j, name in enumerate(FEATURES):
        frame[name] = 1.0 + rng.normal(0.0, 0.1, n) + 0.02 * j
    frame["tap_iron"] = 100.0 + 2.0 * rng.normal(size=n)
    frame["tap_time_len"] = 30.0 + 1.0 * rng.normal(size=n)
    if (frame[["tap_iron", "tap_time_len"]] <= 0).any().any():
        raise AssertionError("Synthetic labels must be positive")
    return frame


def _parent_trial(target: str, trial_id: str) -> dict:
    return {
        "trial_id": trial_id,
        "kind": "ebm_boundary",
        "target": target,
        "target_transform": "log1p",
        "feature_set": "raw",
        "parameters": {
            "max_bins": 32,
            "min_samples_leaf": 5,
            "interactions": 2,
            "max_interaction_bins": 8,
            "max_leaves": 2,
            "objective": "rmse",
            "learning_rate": 0.05,
            "outer_bags": 4,
            "inner_bags": 0,
            "max_rounds": 50,
            "early_stopping_rounds": 5,
            "random_state": 42,
            "n_jobs": 1,
        },
    }


def _fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "configs/round2_v3_6").mkdir(parents=True)
    shutil.copy("configs/round2_v3_6/search.yaml", root / "configs/round2_v3_6/search.yaml")
    ledgers = {
        root / "local/runs/round2-v3.4-ebm-and-constrained-composition/refine-r1/seed-42/fit_ledger.jsonl": [
            ("tap_time_len", "v34-s1-ebm_boundary-0081"),
            ("tap_time_len", "v34-s1-ebm_boundary-0089"),
        ],
        root / "local/runs/round2-v3.5-regularized-ebm-and-composition/refine-r1/seed-42/fit_ledger.jsonl": [
            ("tap_iron", "v35-s1-R-0036"),
            ("tap_iron", "v35-s1-R-0041"),
            ("tap_time_len", "v35-s1-I-0181"),
            ("tap_time_len", "v35-s1-R-0109"),
        ],
    }
    for path, rows in ledgers.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for target, trial_id in rows:
                trial = _parent_trial(target, trial_id)
                handle.write(json.dumps({"event": "complete", "trial_id": trial_id, "trial": trial}) + "\n")
    return root


def test_v36_sampler_counts_and_pre_registered_grid(tmp_path):
    root = _fake_root(tmp_path)
    config = load_v36_config(root)
    trials = sample_v36(root)
    summary = schedule_summary(trials)
    assert summary["slots"] == 224
    assert summary["line_counts"] == {"O": 64, "D": 96, "N": 64}
    assert summary["duplicate_trial_hashes"] == 0
    for line, expected in TARGET_LINE_COUNTS.items():
        for target, count in expected.items():
            assert summary["target_counts"][line][target] == count
    o = [trial for trial in trials if trial["line"] == "O"]
    assert {trial["target_transform"] for trial in o} == {"mean", "log1p"}
    assert {trial["parameters"]["objective"] for trial in o} == {
        "rmse", "pseudo_huber:delta=0.01", "pseudo_huber:delta=0.03", "pseudo_huber:delta=0.10"
    }
    assert {trial["parameters"]["reg_lambda"] for trial in o} == {0.0, 10.0}
    d = [trial for trial in trials if trial["line"] == "D"]
    assert {trial["training_mode"] for trial in d} == {"greedy_original", "cyclic_only"}
    assert {trial["smoothing_mode"] for trial in d} == {"off", "on"}
    assert {trial["parameters"]["max_leaves"] for trial in d} == {2, 3}
    assert {trial["parameters"]["learning_rate"] for trial in d} == {0.01, 0.05}
    n = [trial for trial in trials if trial["line"] == "N"]
    assert {trial["structure"] for trial in n} == {"raw_mlp", "ple_mlp", "raw_tabm", "ple_tabm"}
    assert len({trial["training_setting"] for trial in n}) == 4


def test_v36_ebm_requested_and_effective_parameters_recorded():
    frame = _frame(100)
    trial = {
        "kind": "ebm_loss",
        "target": "tap_iron",
        "target_transform": "mean",
        "feature_set": "raw",
        "parameters": {
            "max_bins": 16,
            "min_samples_leaf": 5,
            "interactions": 0,
            "max_interaction_bins": 8,
            "max_leaves": 2,
            "objective": "pseudo_huber:delta=0.10",
            "learning_rate": 0.2,
            "outer_bags": 4,
            "inner_bags": 0,
            "max_rounds": 20,
            "early_stopping_rounds": 5,
            "random_state": 42,
            "n_jobs": 1,
            "reg_lambda": 10.0,
            "greedy_ratio": 0.0,
            "cyclic_progress": True,
            "smoothing_rounds": 50,
            "interaction_smoothing_rounds": 10,
        },
        "protocol": {"bags": "group-safe-bags-v1", "inner_splits": 5, "bag_seed": 42},
    }
    model = V36EBMRegressor(trial).fit(frame, frame["tap_iron"].to_numpy(dtype=float))
    assert model.requested_params_["objective"] == "pseudo_huber:delta=0.10"
    assert model.effective_params_["reg_lambda"] == 10.0
    assert model.effective_params_["greedy_ratio"] == 0.0
    assert model.effective_params_["cyclic_progress"] is True
    assert model.fit_meta_["ignored_params"] == []
    assert all(check["effective"] for check in model.parameter_checks_.values())
    roundtrip = pickle.loads(pickle.dumps(model))
    assert np.allclose(model.predict(frame), roundtrip.predict(frame), rtol=0.0, atol=1e-12)


def _network_trial(structure: str, target: str = "tap_iron") -> dict:
    params = {
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "random_seed": 42,
        "n_bins": 8,
        "d_embedding": 4,
        "batch_size": 32,
        "max_epochs": 2,
        "early_stopping_patience": 2,
        "min_delta": 1e-5,
        "inner_validation_folds": 5,
        "inner_validation_seed": 42,
    }
    if "tabm" in structure:
        params.update({"k": 2, "n_blocks": 1, "d_block": 8, "dropout": 0.0})
    else:
        params.update({"hidden": [8], "dropout": 0.0})
    return {
        "trial_id": f"v36-test-{structure}-{target}",
        "kind": "numeric_tabm" if "tabm" in structure else "numeric_mlp",
        "structure": structure,
        "target": target,
        "target_transform": "train_mean_std",
        "parameters": params,
    }


@pytest.mark.parametrize("structure", ["raw_mlp", "ple_mlp", "raw_tabm", "ple_tabm"])
def test_v36_network_structures_fit_and_predict(structure):
    frame = _frame(80)
    model = V36NetworkRegressor(_network_trial(structure)).fit(frame, frame["tap_iron"].to_numpy(dtype=float))
    pred = model.predict(frame)
    assert pred.shape == (len(frame),)
    assert np.isfinite(pred).all()
    assert "ple_bins_lengths" in model.fit_meta_["preprocessing"]
    restored = pickle.loads(pickle.dumps(model))
    assert np.allclose(pred, restored.predict(frame), rtol=0.0, atol=1e-10)


def test_v36_outer_fold_dispatch_and_network_no_label_leak_shape():
    frame = _frame(90)
    folds = np.asarray([i % 5 for i in range(len(frame))], dtype=int)
    trial = _network_trial("raw_mlp")
    result = evaluate_v36_outer_folds(frame, folds, trial, fold_ids=(0,))
    assert set(result["fold_scores"]) == {"0"}
    assert len(result["fit_meta"]) == 1
    assert np.isfinite(result["predictions"][folds == 0]).all()
    assert np.isnan(result["predictions"][folds != 0]).all()
    dispatcher = V36Regressor(trial)
    with pytest.raises(RuntimeError):
        dispatcher.predict(frame)


def test_v36_a_development_replay_if_private_evidence_present():
    root = Path(".")
    if not (root / "local/runs/round2-v3.4-ebm-and-constrained-composition/l1-oof-r1/seed-42").exists():
        pytest.skip("private V3.4 development OOF evidence is unavailable")
    from bf_tap_r2.v3_6_reference import build_a_development_replay

    result = build_a_development_replay(root)
    assert abs(result["a_dev_package_score"] - 96.19464675636728) < 1e-9
    assert abs(result["v35_dev_package_score"] - 96.20173099861380) < 1e-9
    assert result["delta_v35_vs_a_dev"] > 0.0
