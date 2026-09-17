"""v0.27 target-aware absolute-error QRF for direct iron prediction."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from qrf_model import distribution_weights, lower_median


PROTOCOL = "QRF_FULLTRAIN_TARGET_v027"
TARGET = "tap_iron"
UNIT = "tonne"
PARAMETERS = {
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


def make_estimator():
    estimator = RandomForestRegressor(**PARAMETERS)
    if any(estimator.get_params(deep=False).get(key) != value for key, value in PARAMETERS.items()):
        raise ValueError("constructed v0.27 iron estimator parameters differ")
    return estimator


class IronTargetForest:
    """Partition with raw iron and predict its full-training leaf lower median."""

    protocol = PROTOCOL
    target = TARGET
    unit = UNIT

    def fit(self, x, y, ids):
        if len(set(ids)) != len(ids) or len(y) != len(ids) or len(x) != len(ids):
            raise ValueError("iron response/ID alignment differs")
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("iron forest requires finite float32 input")
        response = np.asarray(y, dtype=np.float64)
        if not np.isfinite(response).all() or (response < 0).any():
            raise ValueError("raw tap_iron tonnes must be finite nonnegative")
        self.forest = make_estimator()
        self.forest.fit(x, response)
        if type(self.forest).__name__ != "RandomForestRegressor" or len(self.forest.estimators_) != 256:
            raise ValueError("v0.27 iron estimator class/tree count differs")
        if any(tree.criterion != "absolute_error" or tree.splitter != "best" for tree in self.forest.estimators_):
            raise ValueError("v0.27 iron tree splitter/criterion differs")
        self.y = response
        self.ids = list(ids)
        self.leaves = []
        sizes = []
        for tree in self.forest.estimators_:
            assigned = tree.apply(x)
            mapping = {int(leaf): np.flatnonzero(assigned == leaf) for leaf in np.unique(assigned)}
            members = np.concatenate(list(mapping.values()))
            if not np.array_equal(np.sort(members), np.arange(len(ids))) or len(set(members.tolist())) != len(ids):
                raise ValueError("original iron training rows not partitioned exactly once")
            self.leaves.append(mapping)
            sizes.extend(len(indices) for indices in mapping.values())
        self.training_partition_audit = {
            "trees": len(self.leaves),
            "leaf_count": len(sizes),
            "minimum_leaf_size": int(min(sizes)),
            "maximum_leaf_size": int(max(sizes)),
            "all_original_rows_projected_per_tree": True,
            "bootstrap_multiplicity_used_only_for_tree_construction": True,
        }
        return self

    def predict(self, x, training_months=None):
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("iron prediction requires finite float32 input")
        routed = np.column_stack([tree.apply(x) for tree in self.forest.estimators_])
        median, mean, diagnostics = [], [], []
        for row in routed:
            members = [self.leaves[index][int(leaf)] for index, leaf in enumerate(row)]
            weights = distribution_weights(members, len(self.y))
            value = lower_median(self.y, weights, members)
            if value < self.y.min() or value > self.y.max():
                raise ValueError("iron leaf median left training response support")
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
