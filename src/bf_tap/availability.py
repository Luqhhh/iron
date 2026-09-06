from __future__ import annotations

import pandas as pd

from .exceptions import ContractError


def eligible_asof(
    frame: pd.DataFrame,
    reference_time: pd.Timestamp,
    *,
    available_at: str = "available_at",
) -> pd.DataFrame:
    if available_at not in frame:
        raise ContractError(f"missing availability column: {available_at}")
    return frame.loc[frame[available_at] <= reference_time]


def freeze_history_origin(
    history: pd.DataFrame,
    fit_cutoff: pd.Timestamp,
    *,
    available_at: str = "available_at",
) -> pd.DataFrame:
    required = {"reference_time", available_at, "tap_iron", "tap_time_len"}
    missing = required - set(history.columns)
    if missing:
        raise ContractError(f"history missing columns: {sorted(missing)}")
    if (history["reference_time"] > history[available_at]).any():
        raise ContractError("history reference_time cannot follow available_at")
    return history.loc[
        (history["reference_time"] < fit_cutoff)
        & (history[available_at] <= fit_cutoff)
        & history[["tap_iron", "tap_time_len"]].notna().all(axis=1)
    ].copy()


def assert_no_future(
    audit: pd.DataFrame,
    *,
    reference_column: str = "reference_time",
    max_available_column: str = "max_available_at",
) -> None:
    present = audit[max_available_column].notna()
    if (audit.loc[present, max_available_column] > audit.loc[present, reference_column]).any():
        raise ContractError("feature audit contains future availability")
