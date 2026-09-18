from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "qrf_v026"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qrf_v015"))

from append_forest import (  # noqa: E402
    APPENDED_PARAMETERS,
    NEW_TREES,
    PARENT_PARAMETERS,
    PARENT_TREES,
    PROTOCOL,
    TOTAL_TREES,
    append_forest,
    bootstrap_draws,
    forest_state_sha256,
    ordered_training_identity,
    tree_state_sha256,
    validate_parent,
    verify_appended,
)
from partition_forest import PARAMETERS as V26_PARAMETERS  # noqa: E402


def fixture():
    rng = np.random.default_rng(2026)
    rows, features = 90, 5
    x = rng.normal(size=(rows, features)).astype(np.float32)
    y = np.abs(rng.normal(loc=120.0, scale=25.0, size=rows))
    ids = [f"BF4_{index:06d}" for index in range(rows)]
    parent = RandomForestRegressor(**PARENT_PARAMETERS).fit(x, y)
    return parent, x, y, ids


@pytest.fixture(scope="module")
def toy():
    parent, x, y, ids = fixture()
    forest, certificate = append_forest(parent, ids, x, y)
    return parent, x, y, ids, forest, certificate


def test_registered_parameters_equal_v026(toy):
    assert V26_PARAMETERS["A"] == PARENT_PARAMETERS
    assert APPENDED_PARAMETERS["n_estimators"] == TOTAL_TREES
    assert APPENDED_PARAMETERS["warm_start"] is True
    assert PROTOCOL == "QRF_TIME_WARM_START_APPEND_1024_v030"


def test_append_preserves_parent_and_prefix(toy):
    parent, x, y, ids, forest, certificate = toy
    before = forest_state_sha256(parent)
    assert before == forest_state_sha256(parent)
    assert len(parent.estimators_) == PARENT_TREES
    assert len(forest.estimators_) == TOTAL_TREES
    assert certificate["new_trees"] == NEW_TREES
    assert certificate["prefix"]["prefix_exact"] is True
    assert certificate["parent_candidate_id"] == "V26A_QRF_ABSOLUTE_SPLIT_TIME"
    assert certificate["parent_trees"] == PARENT_TREES and certificate["total_trees"] == TOTAL_TREES
    assert certificate["protocol"] == PROTOCOL and certificate["warm_start_registered"] is True
    assert certificate["tree_state_sha256"][:PARENT_TREES] == before
    assert np.array_equal(bootstrap_draws(parent), bootstrap_draws(forest)[:PARENT_TREES])
    assert len(set(certificate["new_tree_random_states"])) == NEW_TREES


def test_append_matches_one_shot_1024_toy_training(toy):
    parent, x, y, ids, forest, _ = toy
    one_shot = RandomForestRegressor(**APPENDED_PARAMETERS).fit(x, y)
    assert [tree.random_state for tree in forest.estimators_] == [tree.random_state for tree in one_shot.estimators_]
    assert forest_state_sha256(forest) == forest_state_sha256(one_shot)
    assert np.array_equal(bootstrap_draws(forest), bootstrap_draws(one_shot))


def test_serialized_append_recovers_draws_and_prefix(tmp_path, toy):
    parent, x, y, ids, forest, certificate = toy
    path = tmp_path / "appended.joblib"
    joblib.dump(forest, path, compress=3)
    restored = joblib.load(path)
    result = verify_appended(parent, restored, x)
    assert result["prefix_exact"] is True
    assert len(result["tree_state_sha256"]) == TOTAL_TREES
    assert np.array_equal(bootstrap_draws(forest), bootstrap_draws(restored))
    assert tree_state_sha256(restored.estimators_[-1]) == certificate["new_tree_state_sha256"][-1]


def test_ordered_training_identity_detects_reordering(toy):
    parent, x, y, ids, forest, certificate = toy
    first = ordered_training_identity(ids, x, y)
    second = ordered_training_identity(ids, x[::-1].copy(), y[::-1].copy())
    assert first["ids"] == second["ids"]
    assert first["matrix"] != second["matrix"]
    assert first["response"] != second["response"]


def test_rejects_non_registered_parent_parameters():
    parent, x, y, ids = fixture()
    parent.set_params(criterion="squared_error")
    with pytest.raises(ValueError, match="parameters"):
        validate_parent(parent, ids, x, y)


def test_rejects_warm_start_parent_flag():
    parent, x, y, ids = fixture()
    parent.set_params(warm_start=True)
    with pytest.raises(ValueError, match="parameters"):
        validate_parent(parent, ids, x, y)


def test_rejects_unfitted_clone():
    parent, x, y, ids = fixture()
    from sklearn.base import clone

    with pytest.raises(ValueError, match="256"):
        validate_parent(clone(parent), ids, x, y)


def test_rejects_duplicate_ids_and_bad_dtype():
    parent, x, y, ids = fixture()
    duplicated = list(ids)
    duplicated[1] = duplicated[0]
    with pytest.raises(ValueError, match="alignment"):
        validate_parent(parent, duplicated, x, y)
    with pytest.raises(ValueError, match="float32"):
        validate_parent(parent, ids, x.astype(np.float64), y)


def test_rejects_negative_response():
    parent, x, y, ids = fixture()
    bad = y.copy()
    bad[0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        validate_parent(parent, ids, x, bad)


def test_append_requires_deep_copy_parent(toy):
    parent, x, y, ids, forest, certificate = toy
    assert forest is not parent
    assert all(left is not right for left, right in zip(parent.estimators_, forest.estimators_))
    snapshot = deepcopy(parent)
    assert forest_state_sha256(snapshot) == forest_state_sha256(parent)


def test_fit_ledger_counts_adopted_and_trained_fits(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("qrf_v030_worker_ledger_test", HERE / "worker.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    fit_ledger = module.fit_ledger

    for slot in ("6",):
        folder = tmp_path / "models" / "B" / slot
        folder.mkdir(parents=True)
        (folder / "adoption_receipt.json").write_text(json.dumps({"new_trees": 768}), encoding="utf-8")
    trained = tmp_path / "models" / "B" / "7"
    trained.mkdir(parents=True)
    (trained / "append_intent.json").write_text(json.dumps({"new_trees": 768}), encoding="utf-8")
    (trained / "fit_record.json").write_text(json.dumps({"new_trees": 768}), encoding="utf-8")
    ledger = fit_ledger(tmp_path)
    assert ledger["fit_attempted"] == 2
    assert ledger["fit_completed"] == 2
    assert ledger["trained_fits"] == 1
    assert ledger["adopted_fits"] == [6]
    assert ledger["new_trees_completed"] == 1536
    assert ledger["new_trees_trained_here"] == 768
    assert ledger["new_trees_adopted"] == 768
    assert ledger["slots"] == [6, 7]
