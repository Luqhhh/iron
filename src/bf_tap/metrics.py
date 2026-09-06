from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .exceptions import ContractError


@dataclass(frozen=True)
class TargetMetrics:
    n: int
    abs_error_sum: float
    actual_sum: float
    mae: float
    wmape: float


def target_metrics(actual: np.ndarray, predicted: np.ndarray) -> TargetMetrics:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ContractError("metric inputs must be non-empty aligned vectors")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ContractError("metric inputs must be finite")
    denominator = float(y.sum())
    if denominator <= 0:
        raise ContractError("WMAPE denominator must be positive")
    numerator = float(np.abs(p - y).sum())
    return TargetMetrics(len(y), numerator, denominator, numerator / len(y), numerator / denominator)


def score_predictions(actual: pd.DataFrame, predicted: pd.DataFrame) -> dict[str, object]:
    id_col = "sample_id"
    if actual[id_col].isna().any() or predicted[id_col].isna().any():
        raise ContractError("null sample_id")
    if actual[id_col].duplicated().any() or predicted[id_col].duplicated().any():
        raise ContractError("duplicate sample_id")
    expected = set(actual[id_col].astype(str))
    received = set(predicted[id_col].astype(str))
    if expected != received:
        raise ContractError("prediction sample_id set does not match labels")
    aligned = actual[[id_col, "tap_iron", "tap_time_len"]].merge(
        predicted[[id_col, "pred_tap_iron", "pred_tap_time_len"]],
        on=id_col,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    iron = target_metrics(aligned["tap_iron"], aligned["pred_tap_iron"])
    time = target_metrics(aligned["tap_time_len"], aligned["pred_tap_time_len"])
    loss = 0.5 * iron.wmape + 0.5 * time.wmape
    return {
        "iron": asdict(iron),
        "time": asdict(time),
        "loss": loss,
        "score": max(0.0, 100.0 * (1.0 - loss)),
    }
