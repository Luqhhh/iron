from __future__ import annotations

import pandas as pd

from ..exceptions import ContractError


def shrink_toward_b1(
    learned: pd.DataFrame,
    b1: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    required = {"sample_id", "pred_tap_iron", "pred_tap_time_len"}
    if set(learned.columns) != required or set(b1.columns) != required:
        raise ContractError("shrinkage inputs must have the canonical prediction columns")
    merged = learned.merge(b1, on="sample_id", suffixes=("_learned", "_b1"), validate="one_to_one")
    if len(merged) != len(learned) or len(merged) != len(b1):
        raise ContractError("shrinkage prediction IDs differ")
    result = pd.DataFrame({"sample_id": merged["sample_id"].astype("string")})
    for target in ("tap_iron", "tap_time_len"):
        weight = float(weights[target])
        if not 0 <= weight <= 1:
            raise ContractError("shrinkage weights must be in [0, 1]")
        result[f"pred_{target}"] = (
            weight * merged[f"pred_{target}_learned"]
            + (1.0 - weight) * merged[f"pred_{target}_b1"]
        )
    return result


def convex_blend(
    predictions: dict[str, pd.DataFrame],
    weights: dict[str, dict[str, float]],
) -> pd.DataFrame:
    if not predictions:
        raise ContractError("convex blend requires predictions")
    model_ids = set(predictions)
    required = {"sample_id", "pred_tap_iron", "pred_tap_time_len"}
    first_id = next(iter(predictions))
    first = predictions[first_id]
    if set(first.columns) != required:
        raise ContractError("convex blend inputs must have canonical prediction columns")
    result = pd.DataFrame({"sample_id": first["sample_id"].astype("string")})
    for model_id, frame in predictions.items():
        if set(frame.columns) != required:
            raise ContractError("convex blend inputs must have canonical prediction columns")
        if frame["sample_id"].astype(str).tolist() != result["sample_id"].astype(str).tolist():
            raise ContractError(f"convex blend prediction IDs or order differ: {model_id}")
    for target in ("tap_iron", "tap_time_len"):
        target_weights = weights.get(target)
        if not isinstance(target_weights, dict) or set(target_weights) != model_ids:
            raise ContractError(f"convex blend weights do not match models for {target}")
        numeric = {name: float(weight) for name, weight in target_weights.items()}
        if any(weight < 0 for weight in numeric.values()) or abs(sum(numeric.values()) - 1.0) > 1e-12:
            raise ContractError(f"convex blend weights must be nonnegative and sum to one for {target}")
        result[f"pred_{target}"] = sum(
            numeric[name] * predictions[name][f"pred_{target}"].to_numpy()
            for name in sorted(model_ids)
        )
    return result


def apply_global_residual_calibration(
    prediction: pd.DataFrame,
    median_prediction_minus_actual: dict[str, float],
    *,
    lower_bound: float = 0.0,
) -> pd.DataFrame:
    required = {"sample_id", "pred_tap_iron", "pred_tap_time_len"}
    if set(prediction.columns) != required:
        raise ContractError("calibration input must have canonical prediction columns")
    if set(median_prediction_minus_actual) != {"tap_iron", "tap_time_len"}:
        raise ContractError("calibration requires both target residuals")
    result = prediction.copy()
    for target, residual in median_prediction_minus_actual.items():
        result[f"pred_{target}"] = (
            result[f"pred_{target}"] - float(residual)
        ).clip(lower=lower_bound)
    return result
