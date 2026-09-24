"""V4 cross-target chaining and shared/private mechanisms (J line).

At inference time the second target is predicted from the input features and a
*model prediction* of the first target.  No true value of the first target is
ever passed to the second stage.  During training, the first-stage input is a
strict inner OOF prediction produced by cross-fitting inside the current
training rows.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Sequence

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


def _cross_fitted_predictions(
    estimator: Any,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    if groups is None:
        splitter: Any = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
        iterator = splitter.split(x)
    else:
        n_groups = len(set(groups.tolist()))
        if n_groups < 2:
            raise ValueError("Cross-target cross-fitting requires at least two groups")
        splitter = GroupKFold(n_splits=min(int(n_splits), n_groups))
        iterator = splitter.split(x, y, groups)
    oof = np.full(len(y), np.nan, dtype=float)
    for train_index, valid_index in iterator:
        model = deepcopy(estimator)
        model.fit(x[train_index], y[train_index])
        oof[valid_index] = np.asarray(model.predict(x[valid_index]), dtype=float)
    if not np.isfinite(oof).all():
        raise RuntimeError("Cross-target inner cross-fitting left an uncovered row")
    return oof


class CrossTargetChainRegressor:
    """Asymmetric chain: X -> predicted source -> target.

    ``direction`` is descriptive metadata only; the caller passes the actual
    ``source`` and ``target`` arrays.  The second-stage design matrix is always
    ``[X, inner_oof_source]`` during fit and ``[X, stage1.predict(X)]`` during
    prediction.  The true source target is never a feature.
    """

    def __init__(
        self,
        source_estimator: Any,
        target_estimator: Any,
        *,
        direction: str,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        if direction not in {"iron_to_time", "time_to_iron"}:
            raise ValueError("direction must be 'iron_to_time' or 'time_to_iron'")
        if int(n_splits) < 2:
            raise ValueError("n_splits must be at least 2")
        self.source_estimator = source_estimator
        self.target_estimator = target_estimator
        self.direction = str(direction)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)

    def fit(
        self,
        x: Any,
        source_target: Any,
        target_target: Any,
        *,
        groups: Iterable[Any] | None = None,
    ) -> "CrossTargetChainRegressor":
        x_array = _as_2d_float(x)
        source_array = _as_1d_float(source_target, "source_target")
        target_array = _as_1d_float(target_target, "target_target")
        if not (len(x_array) == len(source_array) == len(target_array)):
            raise ValueError("X, source_target and target_target must align")
        group_array = None if groups is None else np.asarray(list(groups))
        if group_array is not None and len(group_array) != len(x_array):
            raise ValueError("groups must align with X")
        self.inner_source_oof_ = _cross_fitted_predictions(
            self.source_estimator,
            x_array,
            source_array,
            group_array,
            self.n_splits,
            self.random_state,
        )
        augmented = np.column_stack([x_array, self.inner_source_oof_])
        self.target_model_ = deepcopy(self.target_estimator)
        self.target_model_.fit(augmented, target_array)
        self.source_model_ = deepcopy(self.source_estimator)
        self.source_model_.fit(x_array, source_array)
        self.n_features_in_ = int(x_array.shape[1])
        self.fit_meta_ = {
            "mechanism": "v4_cross_target_chain",
            "direction": self.direction,
            "stage2_input_columns": [f"x{i}" for i in range(x_array.shape[1])] + ["inner_oof_source"],
            "inner_oof_source_rmse": float(
                np.sqrt(np.mean((source_array - self.inner_source_oof_) ** 2))
            ),
            "true_source_target_used_as_feature": False,
            "selection_data": "training_rows_only",
        }
        return self

    def predict(self, x: Any) -> np.ndarray:
        if not hasattr(self, "source_model_"):
            raise RuntimeError("CrossTargetChainRegressor is not fitted")
        x_array = _as_2d_float(x, "X")
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count than fit")
        source_prediction = np.asarray(self.source_model_.predict(x_array), dtype=float)
        augmented = np.column_stack([x_array, source_prediction])
        prediction = np.asarray(self.target_model_.predict(augmented), dtype=float)
        if prediction.shape != (len(x_array),) or not np.isfinite(prediction).all():
            raise ValueError("Invalid cross-target chain prediction")
        return prediction


class SharedPrivateCrossTargetRegressor:
    """Shared low-dimensional projections with target-private linear functions.

    A multi-target PLS model learns shared projection directions.  Each target
    then receives a target-specific smooth response over the shared scores plus
    a target-private ridge term on the standardised raw features.  The model is
    intentionally low-capacity; it is not a replacement for the full joint
    CatBoost/MultiRMSE members.
    """

    def __init__(
        self,
        *,
        n_components: int = 2,
        n_knots: int = 3,
        degree: int = 2,
        shared_alpha: float = 1.0,
        private_alpha: float = 10.0,
        standardize: bool = True,
    ) -> None:
        if int(n_components) < 1:
            raise ValueError("n_components must be positive")
        if int(n_knots) < 2 or int(degree) < 1:
            raise ValueError("n_knots and degree must be positive")
        if float(shared_alpha) < 0 or float(private_alpha) < 0:
            raise ValueError("ridge penalties must be nonnegative")
        self.n_components = int(n_components)
        self.n_knots = int(n_knots)
        self.degree = int(degree)
        self.shared_alpha = float(shared_alpha)
        self.private_alpha = float(private_alpha)
        self.standardize = bool(standardize)

    def fit(self, x: Any, y_iron: Any, y_time: Any) -> "SharedPrivateCrossTargetRegressor":
        x_array = _as_2d_float(x)
        y_iron_array = _as_1d_float(y_iron, "y_iron")
        y_time_array = _as_1d_float(y_time, "y_time")
        if not (len(x_array) == len(y_iron_array) == len(y_time_array)):
            raise ValueError("X and both targets must align")
        if self.n_components > min(x_array.shape):
            raise ValueError("n_components exceeds data rank")
        self.scaler_ = StandardScaler() if self.standardize else None
        scaled = self.scaler_.fit_transform(x_array) if self.scaler_ is not None else x_array
        joint_targets = np.column_stack([y_iron_array, y_time_array])
        self.shared_pls_ = PLSRegression(n_components=self.n_components, scale=False)
        self.shared_pls_.fit(scaled, joint_targets)
        shared_scores = np.asarray(self.shared_pls_.transform(scaled), dtype=float)
        self.shared_spline_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            include_bias=False,
            extrapolation="constant",
        )
        self.shared_spline_.fit(shared_scores)
        shared_basis = np.asarray(self.shared_spline_.transform(shared_scores), dtype=float)
        design = np.column_stack([shared_basis, scaled])
        self.iron_model_ = Ridge(alpha=self.private_alpha).fit(design, y_iron_array)
        self.time_model_ = Ridge(alpha=self.private_alpha).fit(design, y_time_array)
        self.n_features_in_ = int(x_array.shape[1])
        self.fit_meta_ = {
            "mechanism": "v4_shared_private_cross_target",
            "n_components": self.n_components,
            "shared": "multi-target PLS projections plus univariate spline responses",
            "private": "target-specific ridge coefficients on standardised features",
            "n_shared_columns": int(shared_basis.shape[1]),
            "n_private_columns": int(scaled.shape[1]),
            "selection_data": "training_rows_only",
        }
        return self

    def _design(self, x: np.ndarray) -> np.ndarray:
        scaled = self.scaler_.transform(x) if self.scaler_ is not None else x
        scores = np.asarray(self.shared_pls_.transform(scaled), dtype=float)
        shared_basis = np.asarray(self.shared_spline_.transform(scores), dtype=float)
        return np.column_stack([shared_basis, scaled])

    def predict(self, x: Any) -> dict[str, np.ndarray]:
        if not hasattr(self, "iron_model_"):
            raise RuntimeError("SharedPrivateCrossTargetRegressor is not fitted")
        x_array = _as_2d_float(x, "X")
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count than fit")
        design = self._design(x_array)
        iron = np.asarray(self.iron_model_.predict(design), dtype=float)
        time = np.asarray(self.time_model_.predict(design), dtype=float)
        if not (np.isfinite(iron).all() and np.isfinite(time).all()):
            raise ValueError("Invalid shared/private cross-target prediction")
        return {"tap_iron": iron, "tap_time_len": time}


class SharedOnlyCrossTargetRegressor:
    """Control: shared projections and target responses, no private raw features."""

    def __init__(
        self,
        *,
        n_components: int = 2,
        n_knots: int = 3,
        degree: int = 2,
        shared_alpha: float = 1.0,
        standardize: bool = True,
    ) -> None:
        if int(n_components) < 1:
            raise ValueError("n_components must be positive")
        if int(n_knots) < 2 or int(degree) < 1:
            raise ValueError("n_knots and degree must be positive")
        if float(shared_alpha) < 0:
            raise ValueError("shared_alpha must be nonnegative")
        self.n_components = int(n_components)
        self.n_knots = int(n_knots)
        self.degree = int(degree)
        self.shared_alpha = float(shared_alpha)
        self.standardize = bool(standardize)

    def fit(self, x: Any, y_iron: Any, y_time: Any) -> "SharedOnlyCrossTargetRegressor":
        x_array = _as_2d_float(x)
        y_iron_array = _as_1d_float(y_iron, "y_iron")
        y_time_array = _as_1d_float(y_time, "y_time")
        if not (len(x_array) == len(y_iron_array) == len(y_time_array)):
            raise ValueError("X and both targets must align")
        if self.n_components > min(x_array.shape):
            raise ValueError("n_components exceeds data rank")
        self.scaler_ = StandardScaler() if self.standardize else None
        scaled = self.scaler_.fit_transform(x_array) if self.scaler_ is not None else x_array
        joint_targets = np.column_stack([y_iron_array, y_time_array])
        self.shared_pls_ = PLSRegression(n_components=self.n_components, scale=False)
        self.shared_pls_.fit(scaled, joint_targets)
        shared_scores = np.asarray(self.shared_pls_.transform(scaled), dtype=float)
        self.shared_spline_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            include_bias=False,
            extrapolation="constant",
        )
        self.shared_spline_.fit(shared_scores)
        design = np.asarray(self.shared_spline_.transform(shared_scores), dtype=float)
        self.iron_model_ = Ridge(alpha=self.shared_alpha).fit(design, y_iron_array)
        self.time_model_ = Ridge(alpha=self.shared_alpha).fit(design, y_time_array)
        self.n_features_in_ = int(x_array.shape[1])
        self.fit_meta_ = {
            "mechanism": "v4_shared_only_cross_target",
            "n_components": self.n_components,
            "n_shared_columns": int(design.shape[1]),
            "private_features_used": False,
            "selection_data": "training_rows_only",
        }
        return self

    def _design(self, x: np.ndarray) -> np.ndarray:
        scaled = self.scaler_.transform(x) if self.scaler_ is not None else x
        scores = np.asarray(self.shared_pls_.transform(scaled), dtype=float)
        return np.asarray(self.shared_spline_.transform(scores), dtype=float)

    def predict(self, x: Any) -> dict[str, np.ndarray]:
        if not hasattr(self, "iron_model_"):
            raise RuntimeError("SharedOnlyCrossTargetRegressor is not fitted")
        x_array = _as_2d_float(x, "X")
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count than fit")
        design = self._design(x_array)
        iron = np.asarray(self.iron_model_.predict(design), dtype=float)
        time = np.asarray(self.time_model_.predict(design), dtype=float)
        if not (np.isfinite(iron).all() and np.isfinite(time).all()):
            raise ValueError("Invalid shared-only cross-target prediction")
        return {"tap_iron": iron, "tap_time_len": time}


__all__ = [
    "CrossTargetChainRegressor",
    "SharedOnlyCrossTargetRegressor",
    "SharedPrivateCrossTargetRegressor",
]
