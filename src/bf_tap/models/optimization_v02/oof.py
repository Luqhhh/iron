from __future__ import annotations

import pandas as pd

from ...exceptions import ContractError
from ...metrics import score_predictions


_ACTUAL_COLUMNS = {"fold_id", "sample_id", "tap_iron", "tap_time_len"}
_PREDICTION_COLUMNS = {
    "fold_id",
    "sample_id",
    "pred_tap_iron",
    "pred_tap_time_len",
}


def _validate_frame(frame: pd.DataFrame, required: set[str], scope: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ContractError(f"{scope} missing columns: {sorted(missing)}")
    if frame[["fold_id", "sample_id"]].isna().any().any():
        raise ContractError(f"{scope} contains null fold/sample keys")
    keys = frame[["fold_id", "sample_id"]].astype(str)
    if keys.duplicated().any():
        raise ContractError(f"{scope} contains duplicate fold/sample keys")
    if keys["sample_id"].duplicated().any():
        raise ContractError("sample_id appears in multiple folds")


def score_oof_predictions(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
) -> dict[str, object]:
    _validate_frame(actual, _ACTUAL_COLUMNS, "OOF actual")
    _validate_frame(predicted, _PREDICTION_COLUMNS, "OOF prediction")
    actual_keys = set(
        actual[["fold_id", "sample_id"]].astype(str).itertuples(index=False, name=None)
    )
    predicted_keys = set(
        predicted[["fold_id", "sample_id"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if actual_keys != predicted_keys:
        raise ContractError("OOF fold/sample keys do not match")

    actual_view = actual[list(_ACTUAL_COLUMNS)].copy()
    predicted_view = predicted[list(_PREDICTION_COLUMNS)].copy()
    actual_view["fold_id"] = actual_view["fold_id"].astype(str)
    actual_view["sample_id"] = actual_view["sample_id"].astype("string")
    predicted_view["fold_id"] = predicted_view["fold_id"].astype(str)
    predicted_view["sample_id"] = predicted_view["sample_id"].astype("string")

    folds: dict[str, object] = {}
    for fold_id in sorted(actual_view["fold_id"].unique()):
        actual_part = actual_view.loc[actual_view["fold_id"] == fold_id].drop(
            columns="fold_id"
        )
        predicted_part = predicted_view.loc[
            predicted_view["fold_id"] == fold_id
        ].drop(columns="fold_id")
        folds[fold_id] = score_predictions(actual_part, predicted_part)

    return {
        "pooled": score_predictions(
            actual_view.drop(columns="fold_id"),
            predicted_view.drop(columns="fold_id"),
        ),
        "folds": folds,
    }
