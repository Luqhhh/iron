from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ..exceptions import ContractError
from ..models.baseline import DualTargetBaseline
from ..models.baseline import TARGETS


class OptimizationModel(Protocol):
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: pd.Series | None = None,
        inner_validation: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    ) -> "OptimizationModel": ...

    def predict_raw(self, X: pd.DataFrame) -> pd.DataFrame: ...


class FrozenBaselineAdapter:
    """v0.2 interface backed by the immutable baseline estimator."""

    model_type = "frozen_baseline_catboost"

    def __init__(self, parameters: dict, categorical: tuple[str, ...] = ("spout_no",)):
        self._model = DualTargetBaseline(parameters, categorical)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: pd.Series | None = None,
        inner_validation: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    ) -> "FrozenBaselineAdapter":
        if inner_validation is not None:
            raise ValueError("frozen baseline adapter does not use inner validation")
        if sample_weight is None:
            self._model.fit(X, y)
            return self
        weights = np.asarray(sample_weight, dtype=float)
        values = y[list(TARGETS)].to_numpy(dtype=float)
        if (
            len(X) == 0
            or len(X) != len(y)
            or len(X) != len(weights)
            or not np.isfinite(values).all()
            or not np.isfinite(weights).all()
            or (weights <= 0).any()
        ):
            raise ContractError("weighted training rows, targets and weights must be finite and aligned")
        if set(self._model.categorical) - set(X.columns):
            raise ContractError("categorical feature is missing")
        if (values < 0).any() or X[list(self._model.categorical)].isna().any().any():
            raise ContractError("weighted training values violate the baseline contract")
        self._model.feature_names_ = list(X.columns)
        self._model.feature_schema_ = [
            {
                "name": column,
                "dtype": str(X[column].dtype),
                "categorical": column in self._model.categorical,
            }
            for column in X.columns
        ]
        self._model.models_ = {}
        model_type = self._model._catboost_regressor()
        for target in TARGETS:
            model = model_type(**self._model.parameters)
            model.fit(
                X,
                y[target],
                cat_features=list(self._model.categorical),
                sample_weight=weights,
            )
            self._model.models_[target] = model
        return self

    def predict_raw(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.predict_raw(X)

    def save(
        self,
        directory: str | Path,
        *,
        metadata: dict,
        history_snapshot: pd.DataFrame,
    ) -> None:
        self._model.save(directory, metadata=metadata, history_snapshot=history_snapshot)
