"""Complete Laplace trajectories; separate from the immutable 500-round G0."""
from __future__ import annotations

import math
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from .laplace_noise import BOUNDS, LaplaceTreeRegressor, advance, log_score, natural_gradient, training_step


def selected_epoch(history, initial_mae, horizon):
    if (not isinstance(horizon, int) or isinstance(horizon, bool)
            or not 0 <= horizon <= len(history) or not math.isfinite(initial_mae)):
        raise ValueError("Invalid selection horizon")
    values = [float(initial_mae), *[float(h["calibration_mae"]) for h in history[:horizon]]]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite selection values")
    return min(range(len(values)), key=lambda k: (values[k], k))


class LaplaceEpochRegressor(LaplaceTreeRegressor):
    def fit(self, x, y, *, epochs=3000, validation=None):
        if hasattr(self, "trees_"):
            raise ValueError("Use a fresh instance")
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if (x.ndim != 2 or x.shape[1] < 1 or len(x) < 2 or y.shape != (len(x),)
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or not isinstance(epochs, int) or isinstance(epochs, bool)
                or not 0 <= epochs <= 3000):
            raise ValueError("Invalid complete-trajectory fit")
        self.x_mean_, self.x_scale_ = x.mean(axis=0), x.std(axis=0)
        self.x_scale_ = np.where(self.x_scale_ > 0, self.x_scale_, 1.)
        self.y_median_, self.y_scale_ = float(np.median(y)), float(y.std())
        self.y_scale_ = self.y_scale_ if self.y_scale_ > 0 else 1.
        sy = (y-self.y_median_)/self.y_scale_
        self.initial_log_scale_ = float(np.clip(np.log(max(float(np.abs(sy).mean()), math.exp(BOUNDS[0]))), *BOUNDS))
        x = self._x(x)
        params = self._initial(len(y))
        self.fit_rows_ = len(y)
        self.trees_, self.steps_, self.history_ = [], [], []
        self.fitted_tree_count_ = 0
        self.initial_nll_ = float(log_score(sy, params).mean())
        self.initial_calibration_mae_ = None
        if validation is not None:
            vx, vy = self._x(validation[0]), np.asarray(validation[1], dtype=float)
            if not len(vx) or vy.shape != (len(vx),) or not np.isfinite(vy).all():
                raise ValueError("Invalid calibration arrays")
            vp = self._initial(len(vy))
            self.initial_calibration_mae_ = float(np.abs(vy-self.y_median_).mean())
        dimensions = 1 if self.recipe == "FIXED" else 2
        for epoch in range(1, epochs+1):
            gradient = natural_gradient(sy, params)
            if not np.isfinite(gradient).all():
                raise ValueError("Nonfinite gradient")
            trees, direction = [], np.zeros_like(params)
            for dim in range(dimensions):
                tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20,
                                             criterion="squared_error", random_state=42)
                tree.fit(x, gradient[:, dim])
                direction[:, dim] = tree.predict(x)
                trees.append(tree)
                self.fitted_tree_count_ += 1
            step, params, loss = training_step(sy, params, direction)
            self.trees_.append(trees)
            self.steps_.append(step)
            entry = {"epoch": epoch, "train_nll": loss, "step": step}
            if validation is not None:
                vd = np.zeros_like(vp)
                for dim, tree in enumerate(trees):
                    vd[:, dim] = tree.predict(vx)
                vp = advance(vp, vd, step)
                entry["calibration_mae"] = float(np.abs(vy-(vp[:, 0]*self.y_scale_+self.y_median_)).mean())
            self.history_.append(entry)
        self.selected_epoch_ = self.best_epoch(epochs) if validation is not None else epochs
        return self

    def best_epoch(self, horizon):
        if self.initial_calibration_mae_ is None:
            raise ValueError("Calibration required for selection")
        return selected_epoch(self.history_, self.initial_calibration_mae_, horizon)

    def predict_params(self, x, *, epoch=None):
        if epoch is None:
            epoch = self.selected_epoch_
        if (not isinstance(epoch, int) or isinstance(epoch, bool)
                or not 0 <= epoch <= len(self.trees_)):
            raise ValueError("Invalid prediction checkpoint")
        x = self._x(x)
        params = self._initial(len(x))
        if not len(x):
            return params
        for trees, step in zip(self.trees_[:epoch], self.steps_[:epoch], strict=True):
            direction = np.zeros_like(params)
            for dim, tree in enumerate(trees):
                direction[:, dim] = tree.predict(x)
            params = advance(params, direction, step)
        if not np.isfinite(params).all():
            raise ValueError("Nonfinite prediction")
        return params

    def predict(self, x, *, epoch=None):
        return self.predict_params(x, epoch=epoch)[:, 0]*self.y_scale_+self.y_median_

    def metadata(self):
        return {"recipe": self.recipe, "fit_rows": self.fit_rows_, "feature_count": len(self.x_mean_),
                "selected_epoch": self.selected_epoch_, "complete_epochs": len(self.history_),
                "fitted_tree_count": self.fitted_tree_count_, "retained_tree_count": sum(map(len, self.trees_)),
                "zero_steps": self.steps_.count(0.), "initial_nll": self.initial_nll_,
                "initial_calibration_mae": self.initial_calibration_mae_}
