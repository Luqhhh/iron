"""Independent Laplace-NLL tree recipe, not an NGBoost-library reproduction.

This prototype accepts arrays, not competition tables. Official partition and
reference binding must be implemented/frozen separately before scientific use.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.tree import DecisionTreeRegressor

STEPS = (1., .5, .25, .125, .0625, .03125, 0.)
BOUNDS = (-6., 3.)
LEARNING_RATE = .05


def checked_distribution(y, params):
    y, params = np.asarray(y, dtype=float), np.asarray(params, dtype=float)
    if (y.ndim != 1 or params.shape != (len(y), 2) or not len(y)
            or not np.isfinite(y).all() or not np.isfinite(params).all()
            or np.any(params[:, 1] < BOUNDS[0]) or np.any(params[:, 1] > BOUNDS[1])):
        raise ValueError("Invalid bounded Laplace parameters or targets")
    return y, params


def log_score(y, params):
    y, params = checked_distribution(y, params)
    return math.log(2.) + params[:, 1] + np.abs(y-params[:, 0])/np.exp(params[:, 1])


def score_gradient(y, params):
    y, params = checked_distribution(y, params)
    b = np.exp(params[:, 1])
    return np.column_stack((np.sign(params[:, 0]-y)/b, 1.-np.abs(y-params[:, 0])/b))


def fisher_diagonal(params):
    params = np.asarray(params, dtype=float)
    checked_distribution(np.zeros(len(params)), params)
    return np.column_stack((np.exp(-2.*params[:, 1]), np.ones(len(params))))


def natural_gradient(y, params):
    # Analytic Fisher inverse; no Monte Carlo metric approximation.
    y, params = checked_distribution(y, params)
    b = np.exp(params[:, 1])
    return np.column_stack((b*np.sign(params[:, 0]-y), 1.-np.abs(y-params[:, 0])/b))


def advance(params, direction, step):
    result = np.asarray(params, dtype=float) - LEARNING_RATE*step*direction
    result[:, 1] = np.clip(result[:, 1], *BOUNDS)
    return result


def training_step(y, params, direction):
    direction = np.asarray(direction, dtype=float)
    if direction.shape != params.shape or not np.isfinite(direction).all():
        raise ValueError("Nonfinite or mismatched training direction")
    previous = float(log_score(y, params).mean())
    for step in STEPS:
        proposed = advance(params, direction, step)
        value = float(log_score(y, proposed).mean())
        if np.isfinite(value) and value <= previous:
            return step, proposed, value
    raise ValueError("Even the frozen zero fallback failed")


class LaplaceTreeRegressor:
    def __init__(self, recipe):
        if recipe not in ("FIXED", "ADAPTIVE"):
            raise ValueError("Unknown Laplace recipe")
        self.recipe = recipe

    def _x(self, x):
        x = np.asarray(x, dtype=float)
        if (x.ndim != 2 or x.shape[1] != len(self.x_mean_)
                or not np.isfinite(x).all()):
            raise ValueError("Invalid query features")
        result = (x-self.x_mean_)/self.x_scale_
        if not np.isfinite(result).all():
            raise ValueError("Invalid standardized features")
        return result

    def _initial(self, n):
        return np.tile((0., self.initial_log_scale_), (n, 1))

    def fit(self, x, y, *, epochs=500, validation=None, patience=60):
        if hasattr(self, "trees_"):
            raise ValueError("Use a fresh instance for every fit, no implicit continuation")
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if (x.ndim != 2 or x.shape[1] < 1 or len(x) < 2 or y.shape != (len(x),)
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or not isinstance(epochs, int) or not 0 <= epochs <= 500
                or not isinstance(patience, int) or patience < 1):
            raise ValueError("Invalid fit arrays or frozen iteration range")
        self.x_mean_, self.x_scale_ = x.mean(axis=0), x.std(axis=0)
        self.x_scale_ = np.where(self.x_scale_ > 0, self.x_scale_, 1.)
        self.y_median_, self.y_scale_ = float(np.median(y)), float(y.std())
        self.y_scale_ = self.y_scale_ if self.y_scale_ > 0 else 1.
        standardized = (y-self.y_median_)/self.y_scale_
        self.initial_log_scale_ = float(np.clip(np.log(max(
            float(np.abs(standardized).mean()), math.exp(BOUNDS[0]))), *BOUNDS))
        x = self._x(x)
        params = self._initial(len(y))
        self.fit_rows_ = len(y)
        self.trees_, self.steps_, self.history_ = [], [], []
        self.fitted_tree_count_ = 0
        self.initial_nll_ = float(log_score(standardized, params).mean())
        dimensions = 1 if self.recipe == "FIXED" else 2
        best_epoch, stale, best_mae = 0, 0, float("inf")
        if validation is not None:
            vx, vy = self._x(validation[0]), np.asarray(validation[1], dtype=float)
            if vy.shape != (len(vx),) or not len(vy) or not np.isfinite(vy).all():
                raise ValueError("Invalid calibration arrays")
            vp = self._initial(len(vy))
            best_mae = float(np.abs(vy-self.y_median_).mean())
        for epoch in range(1, epochs+1):
            gradient = natural_gradient(standardized, params)
            if not np.isfinite(gradient).all():
                raise ValueError("Nonfinite natural gradient")
            trees, direction = [], np.zeros_like(params)
            for dim in range(dimensions):
                tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20,
                                             criterion="squared_error", random_state=42)
                tree.fit(x, gradient[:, dim])
                direction[:, dim] = tree.predict(x)
                trees.append(tree)
                self.fitted_tree_count_ += 1
            step, params, value = training_step(standardized, params, direction)
            self.trees_.append(trees)
            self.steps_.append(step)
            entry = {"epoch": epoch, "train_nll": value, "step": step}
            if validation is not None:
                vd = np.zeros_like(vp)
                for dim, tree in enumerate(trees):
                    vd[:, dim] = tree.predict(vx)
                vp = advance(vp, vd, step)
                mae = float(np.abs(vy-(vp[:, 0]*self.y_scale_+self.y_median_)).mean())
                entry["calibration_mae"] = mae
                if mae < best_mae:
                    best_mae, best_epoch, stale = mae, epoch, 0
                else:
                    stale += 1
            self.history_.append(entry)
            if validation is not None and stale >= patience:
                break
        self.selected_epoch_ = best_epoch if validation is not None else epochs
        self.trees_ = self.trees_[:self.selected_epoch_]
        self.steps_ = self.steps_[:self.selected_epoch_]
        return self

    def predict_params(self, x):
        x = self._x(x)
        params = self._initial(len(x))
        if not len(x):
            return params
        for trees, step in zip(self.trees_, self.steps_, strict=True):
            direction = np.zeros_like(params)
            for dim, tree in enumerate(trees):
                direction[:, dim] = tree.predict(x)
            params = advance(params, direction, step)
        if not np.isfinite(params).all():
            raise ValueError("Nonfinite distribution prediction")
        return params

    def predict(self, x):
        return self.predict_params(x)[:, 0]*self.y_scale_+self.y_median_

    def predict_scale(self, x):
        return np.exp(self.predict_params(x)[:, 1])*self.y_scale_

    def metadata(self):
        return {"recipe": self.recipe, "fit_rows": self.fit_rows_,
                "feature_count": len(self.x_mean_), "selected_epoch": self.selected_epoch_,
                "fitted_tree_count": self.fitted_tree_count_,
                "retained_tree_count": sum(map(len, self.trees_)),
                "zero_steps": self.steps_.count(0.), "initial_nll": self.initial_nll_}
