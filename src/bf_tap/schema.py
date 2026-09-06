from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .exceptions import ContractError
from .io import TARGETS, ensure_unique_nonnull

SAMPLE_COLUMNS = ("sample_id", "tap_no", "spout_no", "reference_time")


@dataclass(frozen=True)
class SchemaResult:
    rows: int
    columns: tuple[str, ...]


def validate_samples(frame: pd.DataFrame, *, labeled: bool) -> SchemaResult:
    expected = set(SAMPLE_COLUMNS) | (set(TARGETS) if labeled else set())
    missing = expected - set(frame.columns)
    if missing:
        raise ContractError(f"sample table missing columns: {sorted(missing)}")
    ensure_unique_nonnull(frame, ["sample_id"], "sample table")
    ensure_unique_nonnull(frame, ["tap_no"], "sample table")
    if frame["reference_time"].isna().any():
        raise ContractError("sample table has null reference_time")
    if labeled:
        values = frame[list(TARGETS)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ContractError("labels must be finite")
        if (values < 0).any():
            raise ContractError("labels must be nonnegative")
    return SchemaResult(len(frame), tuple(frame.columns))


def validate_event_source(
    frame: pd.DataFrame,
    *,
    event_time: str,
    available_at: str,
    value_columns: list[str],
) -> SchemaResult:
    required = {event_time, available_at, *value_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ContractError(f"event source missing columns: {sorted(missing)}")
    if frame[[event_time, available_at]].isna().any().any():
        raise ContractError("event source has null timing keys")
    if (frame[available_at] < frame[event_time]).any():
        raise ContractError("available_at precedes event_time")
    values = frame[value_columns].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ContractError("event source contains non-finite numeric values")
    return SchemaResult(len(frame), tuple(frame.columns))


def _validate_history_metadata_frame(
    frame: pd.DataFrame,
    *,
    end_time: str = "tap_end_time",
    available_at: str = "available_at",
) -> None:
    required = {"sample_id", "tap_no", "spout_no", "reference_time", end_time, available_at}
    missing = required - set(frame.columns)
    if missing:
        raise ContractError(f"history metadata missing columns: {sorted(missing)}")
    ensure_unique_nonnull(frame, ["sample_id"], "history")
    ensure_unique_nonnull(frame, ["tap_no"], "history")
    if frame[["spout_no", "reference_time", end_time, available_at]].isna().any().any():
        raise ContractError("history metadata contains null values")
    if (frame["reference_time"] > frame[end_time]).any():
        raise ContractError("history must satisfy reference_time <= tap_end_time")
    if (frame[end_time] > frame[available_at]).any():
        raise ContractError("history must satisfy tap_end_time <= available_at")


def validate_history(
    frame: pd.DataFrame,
    *,
    end_time: str = "tap_end_time",
    available_at: str = "available_at",
) -> SchemaResult:
    _validate_history_metadata_frame(frame, end_time=end_time, available_at=available_at)
    missing = set(TARGETS) - set(frame.columns)
    if missing:
        raise ContractError(f"history missing targets: {sorted(missing)}")
    values = frame[list(TARGETS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ContractError("history targets must be finite")
    if (values < 0).any():
        raise ContractError("history targets must be nonnegative")
    return SchemaResult(len(frame), tuple(frame.columns))


def validate_cross_table_metadata(samples: pd.DataFrame, history: pd.DataFrame) -> dict[str, int]:
    sample_columns = ["sample_id", "tap_no", "spout_no", "reference_time"]
    missing = set(sample_columns) - set(samples.columns)
    if missing:
        raise ContractError(f"sample metadata missing columns: {sorted(missing)}")
    ensure_unique_nonnull(samples, ["sample_id"], "sample metadata")
    _validate_history_metadata_frame(history)
    merged = samples[sample_columns].merge(
        history[sample_columns], on="sample_id", how="inner", suffixes=("_sample", "_history"), validate="one_to_one"
    )
    for column in ("tap_no", "spout_no", "reference_time"):
        if not (merged[f"{column}_sample"] == merged[f"{column}_history"]).all():
            raise ContractError(f"cross-table metadata mismatch: {column}")
    return {
        "sample_rows": len(samples),
        "history_rows": len(history),
        "matched_rows": len(merged),
        "sample_only_rows": len(samples) - len(merged),
        "history_only_rows": len(history) - len(merged),
    }


def validate_cross_table_consistency(
    samples: pd.DataFrame,
    history: pd.DataFrame,
    *,
    target_atol: float = 1e-9,
) -> dict[str, int]:
    evidence = validate_cross_table_metadata(samples, history)
    required = set(TARGETS)
    if required - set(samples.columns) or required - set(history.columns):
        raise ContractError("cross-table target comparison requires both targets")
    merged = samples[["sample_id", *TARGETS]].merge(
        history[["sample_id", *TARGETS]],
        on="sample_id",
        how="inner",
        suffixes=("_sample", "_history"),
        validate="one_to_one",
    )
    for target in TARGETS:
        left = merged[f"{target}_sample"].to_numpy(dtype=float)
        right = merged[f"{target}_history"].to_numpy(dtype=float)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ContractError("cross-table targets must be finite")
        if not np.allclose(left, right, rtol=0.0, atol=target_atol):
            raise ContractError(f"cross-table target mismatch: {target}")
    return evidence
