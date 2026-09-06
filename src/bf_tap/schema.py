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
    return SchemaResult(len(frame), tuple(frame.columns))
