from __future__ import annotations

import numpy as np
import pandas as pd

from .exceptions import ContractError
from .io import parse_local_time


def normalize_event_source(
    frame: pd.DataFrame,
    *,
    event_time_column: str,
    available_at_column: str | None,
    value_columns: list[str],
    missing_markers: list[str] | None = None,
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
    markers = set(missing_markers or [])
    parse_audit: dict[str, dict[str, int]] = {}
    for column in value_columns:
        raw = frame[column]
        normalized = raw.mask(raw.astype("string").isin(markers)) if markers else raw
        numeric = pd.to_numeric(normalized, errors="coerce")
        invalid = normalized.notna() & numeric.isna()
        if invalid.any():
            examples = normalized.loc[invalid].astype(str).drop_duplicates().head(3).tolist()
            raise ContractError(
                f"{column}: {int(invalid.sum())} invalid numeric values, examples={examples}"
            )
        finite = numeric.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            raise ContractError(f"{column}: non-finite numeric values")
        result[column] = numeric.astype(float)
        parse_audit[column] = {
            "known_missing": int(normalized.isna().sum()),
            "invalid_conversion": 0,
            "nonfinite": 0,
        }
    result.attrs["parse_audit"] = parse_audit
    return result
