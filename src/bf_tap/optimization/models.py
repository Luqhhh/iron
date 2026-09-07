from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from ..models.baseline import DualTargetBaseline


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
        if sample_weight is not None or inner_validation is not None:
            raise ValueError("frozen baseline adapter does not accept v0.2-only fit controls")
        self._model.fit(X, y)
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
