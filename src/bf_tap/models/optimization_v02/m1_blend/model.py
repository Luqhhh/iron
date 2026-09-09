from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from ....exceptions import ContractError
from ..oof import score_oof_predictions


_TARGETS = ("tap_iron", "tap_time_len")
_PREDICTION_COLUMNS = tuple(f"pred_{target}" for target in _TARGETS)


def _validated_prediction(
    frame: pd.DataFrame,
    *,
    scope: str,
    include_fold: bool,
) -> pd.DataFrame:
    keys = ["sample_id"]
    if include_fold:
        keys.insert(0, "fold_id")
    required = set(keys) | set(_PREDICTION_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ContractError(f"{scope} missing columns: {sorted(missing)}")
    result = frame[keys + list(_PREDICTION_COLUMNS)].copy()
    for key in keys:
        if result[key].isna().any():
            raise ContractError(f"{scope} contains null keys")
        result[key] = result[key].astype("string")
    if result[keys].duplicated().any():
        raise ContractError(f"{scope} contains duplicate keys")
    if not np.isfinite(result[list(_PREDICTION_COLUMNS)].to_numpy(dtype=float)).all():
        raise ContractError(f"{scope} contains non-finite predictions")
    return result


def _validate_target_weights(weights: Mapping[str, float]) -> dict[str, float]:
    if set(weights) != set(_TARGETS):
        raise ContractError("blend weights must contain exactly both targets")
    result = {target: float(weights[target]) for target in _TARGETS}
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in result.values()):
        raise ContractError("blend weights must be finite values in [0, 1]")
    return result


def blend_predictions(
    catboost: pd.DataFrame,
    b1: pd.DataFrame,
    weights: Mapping[str, float],
) -> pd.DataFrame:
    has_fold = "fold_id" in catboost.columns
    if has_fold != ("fold_id" in b1.columns):
        raise ContractError("blend prediction key shapes differ")
    keys = ["fold_id", "sample_id"] if has_fold else ["sample_id"]
    left = _validated_prediction(catboost, scope="CatBoost prediction", include_fold=has_fold)
    right = _validated_prediction(b1, scope="B1 prediction", include_fold=has_fold)
    left_keys = set(left[keys].itertuples(index=False, name=None))
    right_keys = set(right[keys].itertuples(index=False, name=None))
    if left_keys != right_keys:
        raise ContractError("blend prediction keys do not match")
    target_weights = _validate_target_weights(weights)

    left["_row_order"] = np.arange(len(left))
    merged = left.merge(
        right,
        on=keys,
        how="inner",
        validate="one_to_one",
        sort=False,
        suffixes=("_catboost", "_b1"),
    ).sort_values("_row_order", kind="stable")
    result = merged[keys].copy()
    for target in _TARGETS:
        column = f"pred_{target}"
        weight = target_weights[target]
        result[column] = (
            weight * merged[f"{column}_catboost"]
            + (1.0 - weight) * merged[f"{column}_b1"]
        )
    return result.reset_index(drop=True)


def _validate_grid(weights: Iterable[float], tie_tolerance: float) -> list[float]:
    values = [float(value) for value in weights]
    if (
        not values
        or len(values) != len(set(values))
        or not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
    ):
        raise ContractError("blend weight grid must contain unique finite values in [0, 1]")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ContractError("tie_tolerance must be finite and nonnegative")
    return sorted(values)


def select_blend_weights(
    actual: pd.DataFrame,
    catboost: pd.DataFrame,
    b1: pd.DataFrame,
    *,
    weights: Iterable[float],
    tie_tolerance: float,
) -> dict[str, object]:
    if "fold_id" not in actual.columns:
        raise ContractError("OOF actual must include fold_id")
    grid = _validate_grid(weights, tie_tolerance)
    rows: dict[str, list[dict[str, float]]] = {target: [] for target in _TARGETS}
    for weight in grid:
        prediction = blend_predictions(
            catboost,
            b1,
            {target: weight for target in _TARGETS},
        )
        metrics = score_oof_predictions(actual, prediction)["pooled"]
        for target, metric_name in (("tap_iron", "iron"), ("tap_time_len", "time")):
            rows[target].append(
                {"weight": weight, "wmape": float(metrics[metric_name]["wmape"])}
            )

    selected: dict[str, float] = {}
    for target in _TARGETS:
        minimum = min(row["wmape"] for row in rows[target])
        selected[target] = min(
            row["weight"]
            for row in rows[target]
            if row["wmape"] <= minimum + tie_tolerance
        )
    final_prediction = blend_predictions(catboost, b1, selected)
    return {
        "weights": selected,
        "metrics": score_oof_predictions(actual, final_prediction),
        "grid": rows,
        "predictions": final_prediction,
    }
