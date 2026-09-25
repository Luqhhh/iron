"""State-space E2 linear-leaf residual adapter for the V4.1-r2 core.

The original V4.1 E2 recipe was a LightGBM linear-tree model on the 21 raw
features.  The r2 nested core deliberately carries only the frozen
prediction-state vector, so this adapter applies the same linear-leaf residual
mechanism to the r2 numeric state columns.  It is declared explicitly as
``E2_state_linear_leaf`` and is not a claimed reproduction of the original
raw-feature E2.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .v4_leaf_estimators import fit_lightgbm_linear_tree

NUMERIC_STATE = ("pred_iron", "pred_time", "gap_iron", "gap_time")

__all__ = ["StateLinearLeafResidual"]


class StateLinearLeafResidual:
    """LightGBM linear-tree residual regression on r2 prediction state."""

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        num_leaves: int = 15,
        learning_rate: float = 0.03,
        min_child_samples: int = 20,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.num_leaves = int(num_leaves)
        self.learning_rate = float(learning_rate)
        self.min_child_samples = int(min_child_samples)
        self.random_state = int(random_state)

    def _frame(self, state: Any) -> np.ndarray:
        if not hasattr(state, "columns") or any(name not in state.columns for name in NUMERIC_STATE):
            raise ValueError(f"E2-state requires columns {NUMERIC_STATE}")
        values = np.asarray(state.loc[:, NUMERIC_STATE].to_numpy(dtype=float), dtype=float)
        if values.ndim != 2 or values.shape[1] != len(NUMERIC_STATE) or not np.isfinite(values).all():
            raise ValueError("Invalid E2-state numeric frame")
        return values

    def fit(self, state: Any, residual: Any) -> "StateLinearLeafResidual":
        x = self._frame(state)
        residual = np.asarray(residual, dtype=float)
        if residual.shape != (len(x),) or not np.isfinite(residual).all():
            raise ValueError("E2-state residual shape/finite check failed")
        self.model_ = fit_lightgbm_linear_tree(
            x,
            residual,
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            verbose=-1,
        )
        return self

    def predict(self, state: Any) -> np.ndarray:
        x = self._frame(state)
        prediction = np.asarray(self.model_.predict(x), dtype=float)
        if prediction.shape != (len(x),) or not np.isfinite(prediction).all():
            raise ValueError("E2-state produced invalid predictions")
        return prediction
