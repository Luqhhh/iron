from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .exceptions import ContractError, ProtectedLabelError

TZ = "Asia/Shanghai"
TARGETS = ("tap_iron", "tap_time_len")


def parse_local_time(series: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise ContractError(f"{name}: {bad} unparseable timestamps")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(TZ, ambiguous="raise", nonexistent="raise")
    return parsed.dt.tz_convert(TZ)


def read_csv(
    path: str | Path,
    *,
    time_columns: Iterable[str] = (),
    required_columns: Iterable[str] = (),
    usecols: Iterable[str] | None = None,
) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise ContractError(f"missing input file: {source}")
    requested = list(usecols) if usecols is not None else None
    frame = pd.read_csv(source, dtype={"sample_id": "string"}, usecols=requested)
    absent = set(required_columns) - set(frame.columns)
    if absent:
        raise ContractError(f"{source}: missing columns {sorted(absent)}")
    for column in time_columns:
        if column in frame:
            frame[column] = parse_local_time(frame[column], column)
    return frame


def read_development_labels(path: str | Path, cutoff: str) -> pd.DataFrame:
    """Read labels only for rows strictly before a declared development cutoff.

    The timestamp-only pass selects safe row numbers before target columns are read.
    This prevents accidental loading of protected November targets into a development
    process. The CSV header is line 0, so pandas skiprows receives data-line offsets.
    """
    source = Path(path)
    meta = read_csv(
        source,
        usecols=["sample_id", "reference_time"],
        required_columns=["sample_id", "reference_time"],
        time_columns=["reference_time"],
    )
    boundary = pd.Timestamp(cutoff)
    if boundary.tzinfo is None:
        boundary = boundary.tz_localize(TZ)
    else:
        boundary = boundary.tz_convert(TZ)
    safe = meta["reference_time"] < boundary
    protected_rows = set((meta.index[~safe] + 1).tolist())
    frame = pd.read_csv(
        source,
        dtype={"sample_id": "string"},
        skiprows=lambda line: line in protected_rows,
    )
    frame["reference_time"] = parse_local_time(frame["reference_time"], "reference_time")
    if (frame["reference_time"] >= boundary).any():
        raise ProtectedLabelError("protected labels crossed development cutoff")
    return frame


def read_development_history(
    path: str | Path,
    cutoff: str,
    *,
    available_at_column: str,
) -> pd.DataFrame:
    """Load only the history-label rows authorized at a development origin."""
    source = Path(path)
    meta_columns = ["sample_id", "reference_time", available_at_column]
    meta = read_csv(
        source,
        usecols=meta_columns,
        required_columns=meta_columns,
        time_columns=["reference_time", available_at_column],
    )
    boundary = pd.Timestamp(cutoff)
    if boundary.tzinfo is None:
        boundary = boundary.tz_localize(TZ)
    else:
        boundary = boundary.tz_convert(TZ)
    safe = (meta["reference_time"] < boundary) & (meta[available_at_column] <= boundary)
    protected_rows = set((meta.index[~safe] + 1).tolist())
    frame = pd.read_csv(
        source,
        dtype={"sample_id": "string"},
        skiprows=lambda line: line in protected_rows,
    )
    frame["reference_time"] = parse_local_time(frame["reference_time"], "reference_time")
    frame[available_at_column] = parse_local_time(
        frame[available_at_column], available_at_column
    )
    if (
        (frame["reference_time"] >= boundary)
        | (frame[available_at_column] > boundary)
    ).any():
        raise ProtectedLabelError("history labels crossed development origin")
    return frame


def ensure_unique_nonnull(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    if frame[columns].isna().any().any():
        raise ContractError(f"{source}: null value in key {columns}")
    duplicated = frame.duplicated(columns, keep=False)
    if duplicated.any():
        raise ContractError(f"{source}: duplicate key {columns} ({int(duplicated.sum())} rows)")
