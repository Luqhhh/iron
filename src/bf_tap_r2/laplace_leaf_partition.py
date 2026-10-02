"""Shallow boosting with residual-magnitude L1 partitions and leaf medians.

This changes both the structure target and its split criterion relative to the
sign-gradient recipe. It is an existing empirical L1 partition idea, not a
claim that the original L1 gradients are wrong or that this recipe is optimal.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from .laplace_leaf_median import LeafMedianRegressor, best_epoch, leaf_medians


class ResidualL1PartitionRegressor(LeafMedianRegressor):
    """Fit each shallow structure to current residuals under absolute error."""

    def __init__(self):
        super().__init__(3)

    def fit(self, x, y, *, epochs=12000, validation=None, on_epoch=None):
        if (not isinstance(epochs, int) or isinstance(epochs, bool)
                or not 0 <= epochs <= 12000):
            raise ValueError("Invalid residual-L1-partition epoch count")

        # Reuse the frozen fit-only normalizers, input/calibration validation,
        # freshness gate and initial predictions without fitting a sign tree.
        super().fit(x, y, epochs=0, validation=validation, on_epoch=on_epoch)
        x, y = np.asarray(x, float), np.asarray(y, float)
        z = self._x(x)
        sy = (y-self.y_median_)/self.y_scale_
        mu = np.zeros(len(y))
        if validation is not None:
            vx, vy = self._x(validation[0]), np.asarray(validation[1], float)
            vm = np.zeros(len(vy))

        previous = self.initial_train_mae_
        for epoch in range(1, epochs+1):
            residual = sy-mu
            tree = DecisionTreeRegressor(max_depth=self.depth, min_samples_leaf=20,
                                         criterion="absolute_error", random_state=42)
            tree.fit(z, residual)
            leaves = tree.apply(z)
            # Keep the response estimator explicit and independently replayable;
            # the tree supplies its partition, not its native leaf prediction.
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

        self.selected_epoch_ = (best_epoch(self.history_, self.initial_calibration_mae_)
                                if validation is not None else epochs)
        return self
