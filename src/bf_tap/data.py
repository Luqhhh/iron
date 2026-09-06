from __future__ import annotations

import pandas as pd

from .exceptions import ContractError
from .io import parse_local_time


def normalize_event_source(
    frame: pd.DataFrame,
    *,
    event_time_column: str,
    available_at_column: str | None,
    value_columns: list[str],
) -> pd.DataFrame:
    if not available_at_column:
        raise ContractError("available_at mapping is unresolved; refusing real feature build")
    required = {event_time_column, available_at_column, *value_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ContractError(f"source mapping columns missing: {sorted(missing)}")
    result = pd.DataFrame(index=frame.index)
    result["event_time"] = parse_local_time(frame[event_time_column], event_time_column)
    result["available_at"] = parse_local_time(frame[available_at_column], available_at_column)
    for column in value_columns:
        result[column] = pd.to_numeric(frame[column], errors="coerce")
    return result
