"""Frozen v0.33 time-forest definitions.

The estimator partitions the certified v0.15 transformed matrix.  Responses
are retained separately so the versioned OOB adapter can estimate each query
from selected leaf members rather than ``RandomForestRegressor.predict``.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


PROTOCOL = "QRF_TIME_FEATURE_AND_ROW_SAMPLING_FOREST_v033"
CANDIDATES = {
    "A": "V33A_OOB_TIME_FEATURE_THIRD",
    "B": "V33B_OOB_TIME_HALF_BOOTSTRAP",
}
COMMON_PARAMETERS = {
    "n_estimators": 256,
    "criterion": "absolute_error",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 10,
    "min_weight_fraction_leaf": 0.0,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "bootstrap": True,
    "oob_score": False,
    "random_state": 2026,
    "n_jobs": 8,
    "verbose": 0,
    "warm_start": False,
    "ccp_alpha": 0.0,
    "monotonic_cst": None,
}
PARAMETERS = {
    "A": {**COMMON_PARAMETERS, "max_features": 1.0 / 3.0, "max_samples": None},
    "B": {**COMMON_PARAMETERS, "max_features": 0.7, "max_samples": 0.5},
}


def expected_draws(candidate: str, rows: int) -> int:
    if candidate not in CANDIDATES or rows < 1:
        raise ValueError("registered candidate and positive training rows required")
    value = PARAMETERS[candidate]["max_samples"]
    return rows if value is None else max(round(float(value) * rows), 1)


def make_estimator(candidate: str) -> RandomForestRegressor:
    if candidate not in CANDIDATES:
        raise ValueError("registered v0.33 candidate required")
    estimator = RandomForestRegressor(**PARAMETERS[candidate])
    actual = estimator.get_params(deep=False)
    if any(actual.get(name) != value for name, value in PARAMETERS[candidate].items()):
        raise ValueError("constructed v0.33 estimator parameters differ")
    if candidate == "A" and actual["max_features"] != 0.3333333333333333:
        raise ValueError("candidate A must use the binary64 value of 1/3")
    return estimator


class SampledTimeForest:
    """New from-scratch RF whose leaves are projected over all unique rows."""

    def __init__(self, candidate: str):
        if candidate not in CANDIDATES:
            raise ValueError("registered v0.33 candidate required")
        self.candidate = candidate

    def fit(self, x, y, ids):
        matrix = np.asarray(x)
        response = np.asarray(y, dtype=np.float64)
        identifiers = list(ids)
        if matrix.dtype != np.float32 or matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("v0.33 forest requires finite float32 two-dimensional input")
        if len(matrix) != len(response) or len(response) != len(identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("v0.33 training matrix/response/ID alignment differs")
        if not np.isfinite(response).all() or (response < 0).any():
            raise ValueError("raw tap_time_len responses must be finite nonnegative")
        self.forest = make_estimator(self.candidate)
        self.forest.fit(matrix, response)
        if len(self.forest.estimators_) != 256:
            raise ValueError("v0.33 tree count differs")
        if any(tree.criterion != "absolute_error" or tree.splitter != "best" for tree in self.forest.estimators_):
            raise ValueError("v0.33 fitted tree criterion/splitter differs")
        self.y = response
        self.ids = identifiers
        self.leaves = []
        leaf_sizes, node_counts, depths = [], [], []
        for estimator in self.forest.estimators_:
            assigned = estimator.apply(matrix)
            mapping = {int(node): np.flatnonzero(assigned == node).astype(np.int32) for node in np.unique(assigned)}
            projected = np.concatenate(list(mapping.values()))
            if len(projected) != len(matrix) or not np.array_equal(np.sort(projected), np.arange(len(matrix))):
                raise ValueError("all original rows must project exactly once per tree")
            self.leaves.append(mapping)
            leaf_sizes.extend(map(len, mapping.values()))
            node_counts.append(int(estimator.tree_.node_count))
            depths.append(int(estimator.tree_.max_depth))
        draws = np.asarray(self.forest.estimators_samples_, dtype=np.int32)
        expected = expected_draws(self.candidate, len(matrix))
        if draws.shape != (256, expected):
            raise ValueError("actual bootstrap draw length differs from registered N/m")
        self.training_partition_audit = {
            "trees": 256,
            "training_rows": len(matrix),
            "draws_per_tree": expected,
            "leaf_count": len(leaf_sizes),
            "minimum_full_leaf_size": int(min(leaf_sizes)),
            "maximum_full_leaf_size": int(max(leaf_sizes)),
            "node_count_minimum": int(min(node_counts)),
            "node_count_maximum": int(max(node_counts)),
            "depth_minimum": int(min(depths)),
            "depth_maximum": int(max(depths)),
            "all_original_rows_projected_per_tree": True,
            "bootstrap_multiplicity_used_only_for_tree_construction": True,
        }
        return self

