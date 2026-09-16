"""Signed-response QRF using the frozen v0.15 tree/distribution convention."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from qrf_model import PARAMETERS, distribution_weights, lower_median


PROTOCOL = "QRF_FULLTRAIN_SIGNED_RESPONSE_v025"
PARENT_PROTOCOL = "QRF_FULLTRAIN_LEAF_v1"


class SignedQRF:
    def fit(self, x, signed_response, ids):
        if len(set(ids)) != len(ids) or len(signed_response) != len(ids) or len(x) != len(ids):
            raise ValueError("signed training response/ID alignment differs")
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("forest requires finite float32 input")
        response = np.asarray(signed_response, dtype=np.float64)
        if not np.isfinite(response).all() or not (response < 0).any() or not (response > 0).any():
            raise ValueError("signed response requires finite negative and positive values")
        self.forest = RandomForestRegressor(**PARAMETERS)
        self.forest.fit(x, response)
        self.y = response
        self.ids = list(ids)
        self.leaves = []
        for tree in self.forest.estimators_:
            leaf_ids = tree.apply(x)
            mapping = {int(leaf): np.flatnonzero(leaf_ids == leaf) for leaf in np.unique(leaf_ids)}
            if not np.array_equal(np.sort(np.concatenate(list(mapping.values()))), np.arange(len(ids))):
                raise ValueError("original training rows not partitioned exactly once")
            self.leaves.append(mapping)
        return self

    def predict(self, x, training_months=None):
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("prediction requires finite float32 input")
        leaf_ids = np.column_stack([tree.apply(x) for tree in self.forest.estimators_])
        median, mean, diagnostics = [], [], []
        for row in leaf_ids:
            members = [self.leaves[index][int(leaf)] for index, leaf in enumerate(row)]
            weights = distribution_weights(members, len(self.y))
            median.append(lower_median(self.y, weights, members))
            mean.append(float(np.sum(self.y * weights)))
            diagnostic = {"effective_neighbors": float(1. / np.sum(weights * weights)),
                          "maximum_weight": float(weights.max())}
            if training_months is not None:
                diagnostic["training_month_weights"] = {
                    str(month): float(weights[np.asarray(training_months) == month].sum())
                    for month in sorted(set(training_months))
                }
            diagnostics.append(diagnostic)
        return np.asarray(median), np.asarray(mean), diagnostics

