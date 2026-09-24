"""V4 projection-pursuit mechanisms (P line).

The core model follows the PPR form

    prediction = sum_r g_r(a_r^T z(x))

where ``z``, the projection directions ``a_r`` and the one-dimensional smooth
functions ``g_r`` are all fitted on the rows passed to :meth:`fit`.  The module
also provides a fully nested residual wrapper: the base model's inner OOF
predictions are produced before the projection model sees a residual, so the
outer-validation labels are never used to train the second stage.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import SplineTransformer, StandardScaler


def _as_2d_float(x: Any, name: str = "X") -> np.ndarray:
    array = np.asarray(x, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty 2-D array")
    return array


def _as_1d_float(y: Any, name: str = "y") -> np.ndarray:
    array = np.asarray(y, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty 1-D array")
    return array


class ProjectionPursuitRegressor:
    """Small-capacity PPR using PLS coordinates and ridge-smoothed splines.

    PLS learns a few supervised linear projections; a univariate spline is then
    fitted on each projection score.  The result is the sum of those smooth
    one-dimensional functions, which is the PPR functional form.  It is not an
    axis-aligned EBM and it is not a full RBF grid.
    """

    def __init__(
        self,
        *,
        n_components: int = 2,
        n_knots: int = 4,
        degree: int = 3,
        alpha: float = 1.0,
        standardize: bool = True,
        fit_intercept: bool = True,
    ) -> None:
        if int(n_components) < 1:
            raise ValueError("n_components must be positive")
        if int(n_knots) < 2 or int(degree) < 1:
            raise ValueError("n_knots and degree must be positive")
        if float(alpha) < 0:
            raise ValueError("alpha must be nonnegative")
        self.n_components = int(n_components)
        self.n_knots = int(n_knots)
        self.degree = int(degree)
        self.alpha = float(alpha)
        self.standardize = bool(standardize)
        self.fit_intercept = bool(fit_intercept)

    def fit(self, x: Any, y: Any) -> "ProjectionPursuitRegressor":
        x_array = _as_2d_float(x)
        y_array = _as_1d_float(y)
        if len(x_array) != len(y_array):
            raise ValueError("X and y must have the same number of rows")
        if self.n_components > min(x_array.shape):
            raise ValueError(
                f"n_components={self.n_components} exceeds min(n_samples, n_features)={min(x_array.shape)}"
            )
        self.scaler_ = StandardScaler() if self.standardize else None
        scaled = self.scaler_.fit_transform(x_array) if self.scaler_ is not None else x_array
        self.pls_ = PLSRegression(n_components=self.n_components, scale=False)
        self.pls_.fit(scaled, y_array)
        score = np.asarray(self.pls_.transform(scaled), dtype=float)
        if score.ndim != 2 or score.shape[1] != self.n_components:
            raise RuntimeError("PLS transform did not return the requested projection count")
        self.spline_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            include_bias=False,
            extrapolation="constant",
        )
        self.spline_.fit(score)
        basis = np.asarray(self.spline_.transform(score), dtype=float)
        self.ridge_ = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        self.ridge_.fit(basis, y_array)
        self.n_features_in_ = int(x_array.shape[1])
        self.fit_meta_ = {
            "mechanism": "v4_projection_pursuit",
            "n_components": self.n_components,
            "n_knots": self.n_knots,
            "degree": self.degree,
            "alpha": self.alpha,
            "standardize": self.standardize,
            "n_projection_scores": int(score.shape[1]),
            "n_spline_columns": int(basis.shape[1]),
            "solver": "PLSRegression + univariate SplineTransformer + Ridge",
            "selection_data": "training_rows_only",
        }
        return self

    def _scores(self, x: np.ndarray) -> np.ndarray:
        scaled = self.scaler_.transform(x) if self.scaler_ is not None else x
        return np.asarray(self.pls_.transform(scaled), dtype=float)

    def predict(self, x: Any) -> np.ndarray:
        if not hasattr(self, "ridge_"):
            raise RuntimeError("ProjectionPursuitRegressor is not fitted")
        x_array = _as_2d_float(x, "X")
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count than fit")
        basis = np.asarray(self.spline_.transform(self._scores(x_array)), dtype=float)
        prediction = np.asarray(self.ridge_.predict(basis), dtype=float)
        if prediction.shape != (len(x_array),) or not np.isfinite(prediction).all():
            raise ValueError("Invalid projection-pursuit prediction")
        return prediction


class CrossFittedResidualProjectionRegressor:
    """Fit a PPR model to fully nested residuals of a base estimator.

    The base model is cross-fitted inside the current training rows.  Its inner
    OOF predictions are used to construct the residual target for the second
    stage.  A malicious base estimator that simply returns labels it saw at fit
    time is therefore forced to return its fallback prediction for inner-fold
    rows; see ``tests/test_round2_v4.py``.
    """

    def __init__(
        self,
        base_estimator: Any,
        *,
        n_splits: int = 5,
        random_state: int = 42,
        projection_factory: Any | None = None,
    ) -> None:
        if int(n_splits) < 2:
            raise ValueError("n_splits must be at least 2")
        self.base_estimator = base_estimator
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.projection_factory = projection_factory or ProjectionPursuitRegressor

    def fit(self, x: Any, y: Any, groups: Iterable[Any] | None = None) -> "CrossFittedResidualProjectionRegressor":
        x_array = _as_2d_float(x)
        y_array = _as_1d_float(y)
        if len(x_array) != len(y_array):
            raise ValueError("X and y must have the same number of rows")
        group_array = None if groups is None else np.asarray(list(groups))
        if group_array is not None and len(group_array) != len(x_array):
            raise ValueError("groups must align with X")
        if group_array is None:
            splitter: Any = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            split_iterator = splitter.split(x_array)
        else:
            n_groups = len(set(group_array.tolist()))
            if n_groups < self.n_splits:
                raise ValueError("Not enough groups for the requested number of splits")
            splitter = GroupKFold(n_splits=min(self.n_splits, n_groups))
            split_iterator = splitter.split(x_array, y_array, group_array)
        inner_oof = np.full(len(y_array), np.nan, dtype=float)
        for train_index, valid_index in split_iterator:
            model = deepcopy(self.base_estimator)
            model.fit(x_array[train_index], y_array[train_index])
            inner_oof[valid_index] = np.asarray(model.predict(x_array[valid_index]), dtype=float)
        if not np.isfinite(inner_oof).all():
            raise RuntimeError("Inner cross-fitting left an uncovered row")
        residual = y_array - inner_oof
        self.projection_model_ = self.projection_factory()
        self.projection_model_.fit(x_array, residual)
        # Refit the base model on all training rows for the final additive prediction.
        self.base_model_ = deepcopy(self.base_estimator)
        self.base_model_.fit(x_array, y_array)
        self.n_features_in_ = int(x_array.shape[1])
        self.inner_oof_ = inner_oof
        self.residual_target_ = residual
        self.fit_meta_ = {
            "mechanism": "v4_cross_fitted_residual_projection",
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "inner_oof_finite": True,
            "inner_residual_rmse": float(np.sqrt(np.mean(residual * residual))),
            "selection_data": "training_rows_only",
        }
        return self

    def predict(self, x: Any) -> np.ndarray:
        if not hasattr(self, "base_model_"):
            raise RuntimeError("CrossFittedResidualProjectionRegressor is not fitted")
        x_array = _as_2d_float(x, "X")
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count than fit")
        prediction = np.asarray(self.base_model_.predict(x_array), dtype=float) + np.asarray(
            self.projection_model_.predict(x_array), dtype=float
        )
        if prediction.shape != (len(x_array),) or not np.isfinite(prediction).all():
            raise ValueError("Invalid residual-projection prediction")
        return prediction


from .v4_smooth_models import LinearControlRegressor

FixedLinearCoordinateRegressor = LinearControlRegressor

__all__ = [
    "CrossFittedResidualProjectionRegressor",
    "FixedLinearCoordinateRegressor",
    "ProjectionPursuitRegressor",
]
