"""Long shallow LAD trajectories with an unchanged, replayable frozen prefix."""
from __future__ import annotations

import math

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from .laplace_leaf_median import LeafMedianRegressor, best_epoch, leaf_medians


class LeafMedianLongRegressor(LeafMedianRegressor):
    """Run the original shallow prefix, then continue its exact update recipe."""

    def __init__(self):
        super().__init__(3)

    def fit(self, x, y, *, epochs=3000, validation=None, on_epoch=None,
            prefix_epochs=3000):
        if (not isinstance(epochs, int) or isinstance(epochs, bool)
                or not 0 <= epochs <= 12000):
            raise ValueError("Invalid long-trajectory epoch count")
        if (not isinstance(prefix_epochs, int) or isinstance(prefix_epochs, bool)
                or prefix_epochs not in (300, 3000)):
            raise ValueError("Only engineering or scientific prefix lengths allowed")

        # This validates fit/calibration arrays and freshness using the frozen
        # implementation, and produces exactly its original prefix trajectory.
        prefix = min(epochs, prefix_epochs)
        super().fit(x, y, epochs=prefix, validation=validation, on_epoch=on_epoch)
        self.prefix_epochs_ = prefix_epochs
        if epochs == prefix:
            return self

        x, y = np.asarray(x, float), np.asarray(y, float)
        z = self._x(x)
        sy = (y-self.y_median_)/self.y_scale_
        mu = np.zeros(len(y))
        if validation is not None:
            vx, vy = self._x(validation[0]), np.asarray(validation[1], float)
            vm = np.zeros(len(vy))
        # Reproduce the in-fit accumulator operation by operation, rather than
        # round-tripping raw predictions through the target normalization.
        for tree, values in zip(self.trees_, self.leaf_values_, strict=True):
            mu += .05*values[tree.apply(z)]
            if validation is not None:
                vm += .05*values[tree.apply(vx)]

        previous = float(np.abs(sy-mu).mean())
        if previous != self.history_[-1]["train_mae_normalized"]:
            raise ValueError("Frozen prefix accumulator reconstruction differs")
        if validation is not None:
            calibration_mae = float(np.abs(vy-(vm*self.y_scale_+self.y_median_)).mean())
            if calibration_mae != self.history_[-1]["calibration_mae"]:
                raise ValueError("Frozen prefix calibration accumulator reconstruction differs")
        for epoch in range(prefix+1, epochs+1):
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

        self.selected_epoch_ = (best_epoch(self.history_, self.initial_calibration_mae_)
                                if validation is not None else epochs)
        return self
