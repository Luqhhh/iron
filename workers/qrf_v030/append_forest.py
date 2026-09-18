"""Warm-start-only 256 -> 1024 expansion of the certified v0.26 A time forest.

The append protocol never retrains the parent.  A deep copy of the fitted
256-tree ``RandomForestRegressor`` is switched to ``n_estimators=1024`` and
``warm_start=True`` and receives exactly one additional ``fit`` call on the
certified X/y/ID order.  Everything before that fit either proves that the
restored object still is the registered v0.26 A endpoint or raises.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
from sklearn.ensemble import RandomForestRegressor


PROTOCOL = "QRF_TIME_WARM_START_APPEND_1024_v030"
PARENT_PROTOCOL = "QRF_FULLTRAIN_PARTITION_v026"
PARENT_CANDIDATE_ID = "V26A_QRF_ABSOLUTE_SPLIT_TIME"
TARGET = "tap_time_len"
UNIT = "minutes"
PARENT_TREES = 256
TOTAL_TREES = 1024
NEW_TREES = TOTAL_TREES - PARENT_TREES
WARM_START = True

# Frozen copy of workers/qrf_v026/partition_forest.py PARAMETERS["A"]; the
# adapter cross-checks this copy against the imported v0.26 registration.
PARENT_PARAMETERS = {
    "n_estimators": 256,
    "criterion": "absolute_error",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 10,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.7,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "bootstrap": True,
    "max_samples": None,
    "oob_score": False,
    "random_state": 2026,
    "n_jobs": 8,
    "verbose": 0,
    "warm_start": False,
    "ccp_alpha": 0.0,
    "monotonic_cst": None,
}
APPENDED_PARAMETERS = {**PARENT_PARAMETERS, "n_estimators": TOTAL_TREES, "warm_start": WARM_START}
TREE_ARRAYS = (
    "children_left", "children_right", "feature", "threshold", "impurity",
    "n_node_samples", "weighted_n_node_samples", "value", "capacity",
    "missing_go_to_left",
)


def array_identity(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": array.dtype.str, "shape": array.shape}, sort_keys=True).encode())
    digest.update(array.tobytes())
    return {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": digest.hexdigest()}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def tree_state(estimator) -> dict:
    tree = estimator.tree_
    arrays = {}
    for name in TREE_ARRAYS:
        if hasattr(tree, name):
            arrays[name] = array_identity(getattr(tree, name))
    return {
        "random_state": int(estimator.random_state),
        "criterion": str(estimator.criterion),
        "splitter": str(estimator.splitter),
        "n_features": int(estimator.n_features_in_),
        "n_outputs": int(estimator.n_outputs_),
        "node_count": int(tree.node_count),
        "max_depth": int(tree.max_depth),
        "arrays": arrays,
    }


def tree_state_sha256(estimator) -> str:
    return _digest(tree_state(estimator))


def forest_state_sha256(forest) -> list:
    return [tree_state_sha256(estimator) for estimator in forest.estimators_]


def bootstrap_draws(forest) -> np.ndarray:
    draws = np.asarray(forest.estimators_samples_, dtype=np.int32)
    if draws.ndim != 2 or draws.shape[0] != len(forest.estimators_):
        raise ValueError("registered bootstrap draw shape differs")
    return draws


def bootstrap_sha256(draws: np.ndarray) -> str:
    return array_identity(np.asarray(draws, dtype=np.int32))["sha256"]


def ordered_training_identity(ids, matrix, response) -> dict:
    """Ordered X/y/ID digests for the certified append training identity."""
    matrix = np.asarray(matrix)
    response = np.asarray(response)
    labels = [str(item) for item in ids]
    if matrix.dtype != np.float32 or matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("append training requires a finite float32 matrix")
    if response.ndim != 1 or not np.isfinite(response).all() or (response < 0).any():
        raise ValueError("append training requires finite nonnegative raw responses")
    if len(labels) != len(matrix) or len(labels) != len(response) or not labels or len(set(labels)) != len(labels):
        raise ValueError("append training ID/response/matrix alignment differs")
    return {
        "rows": len(labels),
        "feature_columns": int(matrix.shape[1]),
        "ids": array_identity(np.asarray(labels, dtype="U")),
        "matrix": array_identity(matrix),
        "response": array_identity(response.astype(np.float64, copy=False)),
        "ordered": True,
    }


def partition_sha256(forest, matrix) -> list:
    """Per-tree full-leaf partition identity of the certified training rows."""
    matrix = np.asarray(matrix)
    result = []
    for estimator in forest.estimators_:
        assigned = estimator.apply(matrix)
        if assigned.shape != (len(matrix),):
            raise ValueError("append training leaf assignment shape differs")
        mapping = {int(leaf): np.flatnonzero(assigned == leaf).astype(np.int64) for leaf in np.unique(assigned)}
        members = np.concatenate(list(mapping.values())) if mapping else np.asarray([], dtype=np.int64)
        if len(members) != len(matrix) or not np.array_equal(np.sort(members), np.arange(len(matrix))):
            raise ValueError("append training rows are not partitioned exactly once")
        digest = hashlib.sha256()
        for leaf in sorted(mapping):
            digest.update(np.asarray([leaf], dtype=np.int64).tobytes())
            current = np.ascontiguousarray(mapping[leaf], dtype=np.int64)
            digest.update(np.asarray(current.shape, dtype=np.int64).tobytes())
            digest.update(current.tobytes())
        result.append(digest.hexdigest())
    return result


def full_leaf_mapping(forest, matrix) -> list:
    mapping = []
    matrix = np.asarray(matrix)
    for estimator in forest.estimators_:
        assigned = estimator.apply(matrix)
        mapping.append({int(leaf): np.flatnonzero(assigned == leaf) for leaf in np.unique(assigned)})
    return mapping


def validate_parent(forest, ids, matrix, response, *, expected_parameters=None) -> dict:
    """Prove the restored object still is the registered v0.26 A endpoint."""
    parameters = PARENT_PARAMETERS if expected_parameters is None else dict(expected_parameters)
    if parameters != PARENT_PARAMETERS:
        raise ValueError("append parent registration differs from the frozen v0.26 A parameters")
    if not isinstance(forest, RandomForestRegressor):
        raise ValueError("append parent must be a RandomForestRegressor")
    actual = forest.get_params(deep=False)
    if set(actual) != set(parameters) or any(actual.get(name) != value for name, value in parameters.items()):
        raise ValueError("restored parent parameters differ from the registered v0.26 A contract")
    if not hasattr(forest, "estimators_") or len(forest.estimators_) != PARENT_TREES:
        raise ValueError("restored parent must hold exactly 256 fitted trees")
    if any(str(tree.criterion) != "absolute_error" or str(tree.splitter) != "best" for tree in forest.estimators_):
        raise ValueError("restored parent tree criterion/splitter differs")
    training = ordered_training_identity(ids, matrix, response)
    labels = np.asarray(matrix)
    draws = bootstrap_draws(forest)
    if draws.shape != (PARENT_TREES, len(labels)) or (draws < 0).any() or (draws >= len(labels)).any():
        raise ValueError("restored parent bootstrap positions differ from the certified training rows")
    partition = partition_sha256(forest, matrix)
    return {
        "protocol": PARENT_PROTOCOL,
        "candidate_id": PARENT_CANDIDATE_ID,
        "target": TARGET,
        "unit": UNIT,
        "parameters": parameters,
        "trees": PARENT_TREES,
        "training_identity": training,
        "tree_state_sha256": forest_state_sha256(forest),
        "bootstrap_sha256": bootstrap_sha256(draws),
        "partition_sha256": partition,
    }


def verified_prefix(parent, appended) -> dict:
    """Compare the first 256 trees, draws and leaf partitions of both forests."""
    if len(parent.estimators_) != PARENT_TREES or len(appended.estimators_) != TOTAL_TREES:
        raise ValueError("append prefix comparison requires a 256/1024 tree pair")
    before = forest_state_sha256(parent)
    after = forest_state_sha256(appended)[:PARENT_TREES]
    if before != after:
        raise ValueError("appended forest changed a parent tree state")
    if any(left is right for left, right in zip(parent.estimators_, appended.estimators_)):
        raise ValueError("appended forest must hold deep-copied estimators")
    parent_draws = bootstrap_draws(parent)
    appended_draws = bootstrap_draws(appended)
    if not np.array_equal(parent_draws, appended_draws[:PARENT_TREES]):
        raise ValueError("appended forest changed a certified parent bootstrap draw")
    return {
        "prefix_trees": PARENT_TREES,
        "prefix_tree_state_sha256": after,
        "prefix_bootstrap_sha256": bootstrap_sha256(appended_draws[:PARENT_TREES]),
        "parent_bootstrap_sha256": bootstrap_sha256(parent_draws),
        "prefix_exact": True,
    }


def append_forest(parent, ids, matrix, response, *, fit_hook=None):
    """Deep-copy the certified parent, append 768 trees with one warm-start fit."""
    certificate = validate_parent(parent, ids, matrix, response)
    before_parent = forest_state_sha256(parent)
    clone = deepcopy(parent)
    if clone is parent or clone.estimators_ is parent.estimators_:
        raise ValueError("append requires a deep copy of the certified parent")
    clone.set_params(n_estimators=TOTAL_TREES, warm_start=WARM_START)
    actual = clone.get_params(deep=False)
    if any(actual.get(name) != value for name, value in APPENDED_PARAMETERS.items()):
        raise ValueError("append parameters differ from the registered warm-start contract")
    if len(clone.estimators_) != PARENT_TREES:
        raise ValueError("deep-copied parent tree count differs before the single append fit")
    features = np.asarray(matrix)
    targets = np.asarray(response, dtype=np.float64)
    if fit_hook is None:
        clone.fit(features, targets)
    else:
        fit_hook(clone, features, targets)
    if len(clone.estimators_) != TOTAL_TREES:
        raise ValueError("warm-start append did not reach exactly 1024 trees")
    if forest_state_sha256(parent) != before_parent:
        raise ValueError("append modified the certified parent object")
    prefix = verified_prefix(parent, clone)
    states = forest_state_sha256(clone)
    new_states = states[PARENT_TREES:]
    if len(new_states) != NEW_TREES or len(set(new_states)) != NEW_TREES:
        raise ValueError("appended forest must hold 768 distinct new trees")
    if any(str(tree.criterion) != "absolute_error" or str(tree.splitter) != "best" for tree in clone.estimators_[PARENT_TREES:]):
        raise ValueError("appended trees changed the registered criterion/splitter")
    draws = bootstrap_draws(clone)
    if draws.shape != (TOTAL_TREES, len(features)):
        raise ValueError("appended bootstrap draw matrix differs")
    return clone, {
        **certificate,
        "protocol": PROTOCOL,
        "parent_protocol": PARENT_PROTOCOL,
        "parent_trees": PARENT_TREES,
        "total_trees": TOTAL_TREES,
        "new_trees": NEW_TREES,
        "parameters": APPENDED_PARAMETERS,
        "prefix": prefix,
        "tree_state_sha256": states,
        "bootstrap_sha256": bootstrap_sha256(draws),
        "new_tree_random_states": [int(tree.random_state) for tree in clone.estimators_[PARENT_TREES:]],
        "new_tree_state_sha256": new_states,
        "deep_copied_parent": True,
        "clone_of_unfitted_parent_forbidden": True,
        "single_warm_start_fit": True,
        "warm_start_registered": True,
        "old_prefix_unchanged": True,
        "new_trees_are_not_copies": True,
    }


def verify_appended(parent, appended, matrix=None) -> dict:
    """Cold/audit comparison of a persisted appended forest against its parent."""
    result = verified_prefix(parent, appended)
    states = forest_state_sha256(appended)
    if len(states) != TOTAL_TREES:
        raise ValueError("persisted appended forest tree count differs")
    if matrix is not None:
        partition = partition_sha256(appended, matrix)
        if len(partition) != TOTAL_TREES:
            raise ValueError("persisted appended forest partition count differs")
        result["partition_sha256"] = partition
    result["tree_state_sha256"] = states
    result["bootstrap_sha256"] = bootstrap_sha256(bootstrap_draws(appended))
    result["new_tree_random_states"] = [int(tree.random_state) for tree in appended.estimators_[PARENT_TREES:]]
    return result


class AppendedTimeForest:
    """Frozen 1024-tree wrapper; responses always come from the certified raw y."""

    protocol = PROTOCOL
    target = TARGET
    unit = UNIT
    appended_new_trees = NEW_TREES
    warm_start_registered = True

    def __init__(self, forest, ids, response, training_months, leaves, certificate):
        if len(forest.estimators_) != TOTAL_TREES or len(leaves) != TOTAL_TREES:
            raise ValueError("appended time forest requires exactly 1024 tree leaves")
        if len(ids) != len(response) or len(leaves) != len(forest.estimators_):
            raise ValueError("appended time forest training identity differs")
        self.forest = forest
        self.ids = list(ids)
        self.y = np.asarray(response, dtype=np.float64)
        self.training_months = training_months
        self.leaves = leaves
        self.certificate = certificate
        self.prefix_verified = bool(certificate.get("prefix", {}).get("prefix_exact"))
        self.new_tree_random_states = list(certificate["new_tree_random_states"])

    def prediction_support(self):
        return [float(self.y.min()), float(self.y.max())]


__all__ = [
    "PROTOCOL", "PARENT_PROTOCOL", "PARENT_CANDIDATE_ID", "TARGET", "UNIT",
    "PARENT_TREES", "TOTAL_TREES", "NEW_TREES", "WARM_START",
    "PARENT_PARAMETERS", "APPENDED_PARAMETERS", "TREE_ARRAYS",
    "array_identity", "tree_state", "tree_state_sha256", "forest_state_sha256",
    "bootstrap_draws", "bootstrap_sha256", "ordered_training_identity",
    "partition_sha256", "full_leaf_mapping", "validate_parent", "verified_prefix",
    "append_forest", "verify_appended", "AppendedTimeForest",
]
