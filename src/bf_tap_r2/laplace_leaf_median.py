"""Explicit LAD TreeBoost leaf-median updates; an existing idea, not new theory."""
from __future__ import annotations

import math
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def leaf_medians(residual, leaves, node_count):
    residual, leaves = np.asarray(residual, float), np.asarray(leaves)
    if (residual.ndim != 1 or not len(residual) or residual.shape != leaves.shape
            or not np.isfinite(residual).all() or not np.issubdtype(leaves.dtype, np.integer)
            or not isinstance(node_count, int) or node_count < 1
            or (leaves < 0).any() or (leaves >= node_count).any()):
        raise ValueError("Invalid residual leaf partition")
    values = np.full(node_count, np.nan)
    for leaf in np.unique(leaves):
        values[leaf] = np.median(residual[leaves == leaf])
    return values


def best_epoch(history, initial_mae):
    values = [float(initial_mae), *[float(h["calibration_mae"]) for h in history]]
    if not np.isfinite(values).all():
        raise ValueError("Invalid selection history")
    return min(range(len(values)), key=lambda k: (values[k], k))


class LeafMedianRegressor:
    def __init__(self, depth):
        if isinstance(depth, bool) or depth not in (3, 6):
            raise ValueError("Only two preregistered depths allowed")
        self.depth = depth

    def _x(self, x):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != len(self.x_mean_) or not np.isfinite(x).all():
            raise ValueError("Invalid feature matrix")
        z = (x-self.x_mean_)/self.x_scale_
        if not np.isfinite(z).all():
            raise ValueError("Invalid scaled feature matrix")
        return z

    def fit(self, x, y, *, epochs=3000, validation=None, on_epoch=None):
        if hasattr(self, "trees_"):
            raise ValueError("Fresh model required")
        x, y = np.asarray(x, float), np.asarray(y, float)
        if (x.ndim != 2 or x.shape[1] < 1 or len(x) < 2 or y.shape != (len(x),)
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or not isinstance(epochs, int) or isinstance(epochs, bool) or not 0 <= epochs <= 3000):
            raise ValueError("Invalid fit arrays or epoch count")
        self.x_mean_, self.x_scale_ = x.mean(axis=0), x.std(axis=0)
        self.x_scale_ = np.where(self.x_scale_ > 0, self.x_scale_, 1.)
        self.y_median_, self.y_scale_ = float(np.median(y)), float(y.std()) or 1.
        sy, z = (y-self.y_median_)/self.y_scale_, self._x(x)
        mu = np.zeros(len(y))
        self.trees_, self.leaf_values_, self.history_ = [], [], []
        self.fit_rows_ = len(y)
        self.initial_train_mae_ = float(np.abs(sy).mean())
        self.initial_calibration_mae_ = None
        if validation is not None:
            vx, vy = self._x(validation[0]), np.asarray(validation[1], float)
            if not len(vx) or vy.shape != (len(vx),) or not np.isfinite(vy).all():
                raise ValueError("Invalid calibration arrays")
            vm = np.zeros(len(vy))
            self.initial_calibration_mae_ = float(np.abs(vy-self.y_median_).mean())
        previous = self.initial_train_mae_
        for epoch in range(1, epochs+1):
            residual = sy-mu
            tree = DecisionTreeRegressor(max_depth=self.depth, min_samples_leaf=20,
                                         criterion="squared_error", random_state=42)
            tree.fit(z, np.sign(residual))
            leaves = tree.apply(z)
            values = leaf_medians(residual, leaves, tree.tree_.node_count)
            update = .05*values[leaves]
            mu += update
            mae = float(np.abs(sy-mu).mean())
            if not math.isfinite(mae) or mae > previous+1e-12:
                raise ValueError("Training L1 descent invariant failed")
            entry = dict(epoch=epoch, train_mae_normalized=mae,
                         maximum_abs_update=float(np.max(np.abs(update))),
                         nonzero_update_rows=int(np.count_nonzero(update)))
            if validation is not None:
                vm += .05*values[tree.apply(vx)]
                entry["calibration_mae"] = float(np.abs(vy-(vm*self.y_scale_+self.y_median_)).mean())
            self.trees_.append(tree)
            self.leaf_values_.append(values)
            self.history_.append(entry)
            previous = mae
            if on_epoch is not None and epoch % 100 == 0:
                on_epoch()
        self.selected_epoch_ = best_epoch(self.history_, self.initial_calibration_mae_) if validation is not None else epochs
        return self

    def predict(self, x, *, epoch=None):
        if epoch is None:
            epoch = self.selected_epoch_
        if not isinstance(epoch, int) or isinstance(epoch, bool) or not 0 <= epoch <= len(self.trees_):
            raise ValueError("Invalid prediction epoch")
        z = self._x(x)
        mu = np.zeros(len(z))
        if len(z):
            for tree, values in zip(self.trees_[:epoch], self.leaf_values_[:epoch], strict=True):
                mu += .05*values[tree.apply(z)]
        p = mu*self.y_scale_+self.y_median_
        if not np.isfinite(p).all():
            raise ValueError("Invalid prediction; no clipping")
        return p

    def metadata(self):
        return dict(depth=self.depth, fit_rows=self.fit_rows_, feature_count=len(self.x_mean_),
                    selected_epoch=self.selected_epoch_, complete_epochs=len(self.trees_),
                    fitted_tree_count=len(self.trees_), initial_train_mae=self.initial_train_mae_,
                    initial_calibration_mae=self.initial_calibration_mae_,
                    exactly_zero_update_epochs=sum(h["nonzero_update_rows"] == 0 for h in self.history_))
