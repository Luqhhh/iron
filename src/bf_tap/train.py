from __future__ import annotations

import pandas as pd

from .exceptions import ContractError
from .models.baseline import DualTargetBaseline


def fit_baseline(
    samples: pd.DataFrame,
    X: pd.DataFrame,
    parameters: dict,
    *,
    categorical: tuple[str, ...] = ("spout_no",),
) -> DualTargetBaseline:
    if not samples.index.equals(X.index):
        raise ContractError("sample metadata and features are not index-aligned")
    required = {"reference_time", "tap_no", "sample_id", "tap_iron", "tap_time_len"}
    missing = required - set(samples.columns)
    if missing:
        raise ContractError(f"training metadata missing columns: {sorted(missing)}")
    ordered_index = samples.sort_values(
        ["reference_time", "tap_no", "sample_id"], kind="mergesort"
    ).index
    ordered_x = X.loc[ordered_index].reset_index(drop=True)
    ordered_y = samples.loc[ordered_index, ["tap_iron", "tap_time_len"]].reset_index(drop=True)
    return DualTargetBaseline(parameters, categorical).fit(ordered_x, ordered_y)
