"""V4.1 E-line residual correctors.

All correctors model a signed residual ``y - B_OOF``.  They never apply
log1p, nonnegativity or clipping to the correction target.  ``alpha`` is not
fitted here; it is selected separately inside the nested protocol.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from .data import FEATURES
from .v4_leaf_estimators import fit_lightgbm_linear_tree

__all__ = [
    "ConstantMedianResidual",
    "LADSplineResidual",
    "LinearLeafResidual",
    "LocalWeightedMedianResidual",
    "exact_l1_alpha",
    "weighted_median",
]


def weighted_median(values: Any, weights: Any) -> float:
    """Return a deterministic weighted median with lower-median tie handling."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape or len(values) == 0:
        raise ValueError("weighted_median expects aligned nonempty 1-D arrays")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weighted_median inputs must be finite and weights nonnegative")
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    cumulative = np.cumsum(weights)
    index = int(np.searchsorted(cumulative, 0.5 * total, side="left"))
    index = min(index, len(values) - 1)
    return float(values[index])


def exact_l1_alpha(y: Any, baseline: Any, direction: Any) -> float:
    """Minimise ``sum(abs(y - baseline - alpha*direction))`` on [0, 1].

    For nonzero direction entries the constrained optimum is the weighted
    median of ``(y - baseline) / direction`` with weights ``abs(direction)``.
    """
    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if y.ndim != 1 or baseline.shape != y.shape or direction.shape != y.shape or len(y) == 0:
        raise ValueError("exact_l1_alpha expects aligned nonempty 1-D arrays")
    if not (np.isfinite(y).all() and np.isfinite(baseline).all() and np.isfinite(direction).all()):
        raise ValueError("Nonfinite alpha inputs")
    active = direction != 0.0
    if not active.any():
        return 0.0
    ratio = (y[active] - baseline[active]) / direction[active]
    q = weighted_median(np.clip(ratio, 0.0, 1.0), np.abs(direction[active]))
    return float(min(1.0, max(0.0, q)))


def _feature_frame(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        columns = [*FEATURES, "spout_no"]
        missing = [name for name in columns if name not in x.columns]
        if missing:
            raise ValueError(f"Residual feature frame is missing columns: {missing}")
        return x.loc[:, columns].reset_index(drop=True)
    array = np.asarray(x, dtype=float)
    if array.ndim != 2 or array.shape[1] != len(FEATURES):
        raise ValueError("Residual numpy input must be the 21 registered numeric columns")
    return pd.DataFrame(array, columns=list(FEATURES))


class ConstantMedianResidual:
    """E0: global median residual with an explicit zero-correction fallback."""

    def fit(self, x: Any, residual: Any) -> "ConstantMedianResidual":
        residual = np.asarray(residual, dtype=float)
        if residual.ndim != 1 or not len(residual) or not np.isfinite(residual).all():
            raise ValueError("E0 requires a finite residual vector")
        self.median_ = float(np.median(residual))
        self.zero_fallback_ = self.median_
        return self

    def predict(self, x: Any) -> np.ndarray:
        length = len(x)
        return np.full(length, self.median_, dtype=float)


class LADSplineResidual:
    """E1: low-degree additive spline plus category one-hot, LAD location."""

    def __init__(self, *, n_knots: int = 4, degree: int = 2, alpha: float = 0.01) -> None:
        if int(n_knots) < 2 or int(degree) < 1:
            raise ValueError("Invalid spline basis")
        self.n_knots = int(n_knots)
        self.degree = int(degree)
        self.alpha = float(alpha)
        if self.alpha < 0:
            raise ValueError("alpha must be nonnegative")

    def fit(self, x: Any, residual: Any) -> "LADSplineResidual":
        x_frame = _feature_frame(x)
        residual = np.asarray(residual, dtype=float)
        if residual.shape != (len(x_frame),) or not np.isfinite(residual).all():
            raise ValueError("E1 residual shape/finite check failed")
        numeric = list(FEATURES)
        self.preprocessor_ = ColumnTransformer([
            ("spline", SplineTransformer(n_knots=self.n_knots, degree=self.degree,
                                         include_bias=False, extrapolation="constant"), numeric),
            ("spout", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["spout_no"]),
        ], remainder="drop")
        self.model_ = Pipeline([
            ("pre", self.preprocessor_),
            ("lad", QuantileRegressor(quantile=0.5, alpha=self.alpha, solver="highs")),
        ])
        self.model_.fit(x_frame, residual)
        return self

    def predict(self, x: Any) -> np.ndarray:
        x_frame = _feature_frame(x)
        prediction = np.asarray(self.model_.predict(x_frame), dtype=float)
        if prediction.shape != (len(x_frame),) or not np.isfinite(prediction).all():
            raise ValueError("E1 produced invalid predictions")
        return prediction


class LinearLeafResidual:
    """E2: LightGBM linear-leaf model on the signed residual."""

    def __init__(self, *, n_estimators: int = 300, num_leaves: int = 15,
                 learning_rate: float = 0.03, min_child_samples: int = 20,
                 random_state: int = 42) -> None:
        self.n_estimators = int(n_estimators)
        self.num_leaves = int(num_leaves)
        self.learning_rate = float(learning_rate)
        self.min_child_samples = int(min_child_samples)
        self.random_state = int(random_state)

    def _numeric_frame(self, x: Any) -> np.ndarray:
        x_frame = _feature_frame(x)
        return x_frame.loc[:, list(FEATURES)].to_numpy(dtype=float)

    def fit(self, x: Any, residual: Any) -> "LinearLeafResidual":
        x_array = self._numeric_frame(x)
        residual = np.asarray(residual, dtype=float)
        if residual.shape != (len(x_array),) or not np.isfinite(residual).all():
            raise ValueError("E2 residual shape/finite check failed")
        self.model_ = fit_lightgbm_linear_tree(
            x_array,
            residual,
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            verbose=-1,
        )
        return self

    def predict(self, x: Any) -> np.ndarray:
        x_array = self._numeric_frame(x)
        prediction = np.asarray(self.model_.predict(x_array), dtype=float)
        if prediction.shape != (len(x_array),) or not np.isfinite(prediction).all():
            raise ValueError("E2 produced invalid predictions")
        return prediction


class LocalWeightedMedianResidual:
    """E3: local weighted median in a fixed prediction-state space.

    The state vector is supplied by the nested protocol.  Scaling uses only
    the training bank.  The neighbour count is fixed to
    ``min(128, n_bank)``; if the bank is smaller than the pre-declared support
    floor, the corrector shrinks to zero.
    """

    def __init__(self, *, min_bank: int = 32, max_neighbors: int = 128,
                 eps: float = 1e-12) -> None:
        self.min_bank = int(min_bank)
        self.max_neighbors = int(max_neighbors)
        self.eps = float(eps)
        if self.min_bank < 1 or self.max_neighbors < 1:
            raise ValueError("Invalid E3 support parameters")

    def fit(self, state: Any, residual: Any) -> "LocalWeightedMedianResidual":
        state = np.asarray(state, dtype=float)
        residual = np.asarray(residual, dtype=float)
        if state.ndim != 2 or residual.shape != (state.shape[0],) or not len(residual):
            raise ValueError("E3 state/residual shape mismatch")
        if not (np.isfinite(state).all() and np.isfinite(residual).all()):
            raise ValueError("E3 state/residual must be finite")
        self.bank_state_ = state
        self.bank_residual_ = residual
        self.mean_ = state.mean(axis=0)
        scale = state.std(axis=0)
        self.scale_ = np.where(scale > self.eps, scale, 1.0)
        self.n_bank_ = int(len(residual))
        self.k_ = int(min(self.max_neighbors, self.n_bank_))
        return self

    def predict(self, state: Any) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        if state.ndim != 2 or state.shape[1] != self.bank_state_.shape[1] or not len(state):
            raise ValueError("E3 query state shape mismatch")
        if not np.isfinite(state).all():
            raise ValueError("E3 query state contains nonfinite values")
        if self.n_bank_ < self.min_bank:
            return np.zeros(len(state), dtype=float)
        bank = (self.bank_state_ - self.mean_) / self.scale_
        query = (state - self.mean_) / self.scale_
        output = np.zeros(len(query), dtype=float)
        for row, q in enumerate(query):
            distances = np.sqrt(np.sum((bank - q) ** 2, axis=1))
            order = np.argsort(distances, kind="mergesort")[: self.k_]
            local_distances = distances[order]
            h = float(local_distances[-1])
            if h <= self.eps:
                weights = np.ones(len(order), dtype=float)
            else:
                u = local_distances / h
                weights = np.clip(1.0 - u ** 3, 0.0, None) ** 3
                if weights.sum() <= 0:
                    weights = np.ones(len(order), dtype=float)
            output[row] = weighted_median(self.bank_residual_[order], weights)
        if not np.isfinite(output).all():
            raise ValueError("E3 produced nonfinite predictions")
        return output
