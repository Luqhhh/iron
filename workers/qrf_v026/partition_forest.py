"""v0.26 forests: partition changes only, frozen full-training leaf response."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from qrf_model import distribution_weights, lower_median


PROTOCOL = "QRF_FULLTRAIN_PARTITION_v026"
PARENT_PROTOCOL = "QRF_FULLTRAIN_LEAF_v1"
CANDIDATES = {
    "A": {
        "candidate_id": "V26A_QRF_ABSOLUTE_SPLIT_TIME",
        "estimator_class": "RandomForestRegressor",
        "criterion": "absolute_error",
        "splitter": "best",
    },
    "B": {
        "candidate_id": "V26B_EXTRA_RANDOM_SPLIT_TIME",
        "estimator_class": "ExtraTreesRegressor",
        "criterion": "squared_error",
        "splitter": "random",
    },
}
COMMON_PARAMETERS = {
    "n_estimators": 256,
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
PARAMETERS = {
    key: {**COMMON_PARAMETERS, "criterion": definition["criterion"]}
    for key, definition in CANDIDATES.items()
}


def make_estimator(candidate: str):
    if candidate not in CANDIDATES:
        raise ValueError("registered v0.26 candidate A or B required")
    estimator_type = RandomForestRegressor if candidate == "A" else ExtraTreesRegressor
    estimator = estimator_type(**PARAMETERS[candidate])
    actual = estimator.get_params(deep=False)
    if any(actual.get(name) != value for name, value in PARAMETERS[candidate].items()):
        raise ValueError("constructed estimator parameters differ from registration")
    if candidate == "B" and actual["bootstrap"] is not True:
        raise ValueError("v0.26 ExtraTrees must explicitly retain bootstrap=True")
    return estimator


class PartitionForest:
    """Use the estimator only to partition X; leaf responses always use raw y."""

    def __init__(self, candidate: str):
        if candidate not in CANDIDATES:
            raise ValueError("registered v0.26 candidate A or B required")
        self.candidate = candidate

    def fit(self, x, y, ids):
        if len(set(ids)) != len(ids) or len(y) != len(ids) or len(x) != len(ids):
            raise ValueError("training response/ID alignment differs")
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("forest requires finite float32 input")
        response = np.asarray(y, dtype=np.float64)
        if not np.isfinite(response).all() or (response < 0).any():
            raise ValueError("raw tap_time_len responses must be finite nonnegative")
        self.forest = make_estimator(self.candidate)
        self.forest.fit(x, response)
        expected = CANDIDATES[self.candidate]
        if type(self.forest).__name__ != expected["estimator_class"]:
            raise ValueError("actual estimator class differs")
        if len(self.forest.estimators_) != PARAMETERS[self.candidate]["n_estimators"]:
            raise ValueError("tree count differs")
        if any(tree.criterion != expected["criterion"] or tree.splitter != expected["splitter"]
               for tree in self.forest.estimators_):
            raise ValueError("fitted tree splitter/criterion differs")
        self.y = response
        self.ids = list(ids)
        self.leaves = []
        leaf_sizes = []
        for tree in self.forest.estimators_:
            leaf_ids = tree.apply(x)
            mapping = {int(leaf): np.flatnonzero(leaf_ids == leaf) for leaf in np.unique(leaf_ids)}
            members = np.concatenate(list(mapping.values()))
            if not np.array_equal(np.sort(members), np.arange(len(ids))) or len(set(members.tolist())) != len(ids):
                raise ValueError("original training rows not partitioned exactly once")
            self.leaves.append(mapping)
            leaf_sizes.extend(len(indices) for indices in mapping.values())
        self.training_partition_audit = {
            "trees": len(self.leaves),
            "leaf_count": len(leaf_sizes),
            "minimum_leaf_size": int(min(leaf_sizes)),
            "maximum_leaf_size": int(max(leaf_sizes)),
            "all_original_rows_projected_per_tree": True,
            "bootstrap_multiplicity_used_only_for_tree_construction": True,
        }
        return self

    def predict(self, x, training_months=None):
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("prediction requires finite float32 input")
        leaf_ids = np.column_stack([tree.apply(x) for tree in self.forest.estimators_])
        median, mean, diagnostics = [], [], []
        for row in leaf_ids:
            members = [self.leaves[index][int(leaf)] for index, leaf in enumerate(row)]
            weights = distribution_weights(members, len(self.y))
            value = lower_median(self.y, weights, members)
            if value < self.y.min() or value > self.y.max():
                raise ValueError("leaf median left raw training response support")
            median.append(value)
            mean.append(float(np.sum(self.y * weights)))
            diagnostic = {
                "effective_neighbors": float(1.0 / np.sum(weights * weights)),
                "maximum_weight": float(weights.max()),
                "weight_mass": float(weights.sum()),
            }
            if training_months is not None:
                months = np.asarray(training_months)
                diagnostic["training_month_weights"] = {
                    str(month): float(weights[months == month].sum()) for month in sorted(set(months))
                }
            diagnostics.append(diagnostic)
        return np.asarray(median), np.asarray(mean), diagnostics
