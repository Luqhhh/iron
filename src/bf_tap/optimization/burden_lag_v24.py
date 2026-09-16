"""Frozen event-record burden summaries for optimization v0.24."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..exceptions import ContractError
from ..io import parse_local_time


FIELDS = ("pig", "all_quality", "consumption", "fuel_rate", "coke_rate")
WINDOWS = (
    ("0_6h", pd.Timedelta(hours=6), pd.Timedelta(0)),
    ("6_24h", pd.Timedelta(hours=24), pd.Timedelta(hours=6)),
    ("24_72h", pd.Timedelta(hours=72), pd.Timedelta(hours=24)),
)
STATISTICS = ("mean", "valid_count")
FEATURE_COLUMNS = tuple(
    f"burden_lag__{field}__{name}__{stat}"
    for field in FIELDS for name, _, _ in WINDOWS for stat in STATISTICS
)


def normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    """Normalize the published burden source under its existing ASSUMED clock contract."""
    value = events.copy()
    if "event_time" not in value and "cal_time" in value:
        value = value.rename(columns={"cal_time": "event_time"})
    required = {"event_time", *FIELDS}
    if required - set(value):
        raise ContractError("burden source columns differ from the frozen v0.24 contract")
    value["event_time"] = parse_local_time(value.event_time, "burden event_time")
    if "available_at" not in value:
        value["available_at"] = value.event_time
    else:
        value["available_at"] = parse_local_time(value.available_at, "burden available_at")
    if value.event_time.isna().any() or value.available_at.isna().any() or (value.available_at < value.event_time).any():
        raise ContractError("invalid burden event/availability time")
    for field in FIELDS:
        value[field] = pd.to_numeric(value[field], errors="coerce")
    relevant = ["event_time", "available_at", *FIELDS]
    duplicated = value.duplicated(["event_time"], keep=False)
    if duplicated.any():
        groups = value.loc[duplicated, relevant].groupby("event_time", dropna=False, sort=False)
        if any(len(group.drop_duplicates()) > 1 for _, group in groups):
            raise ContractError("conflicting burden rows share the same event_time")
    value = value.drop_duplicates(relevant).sort_values(
        ["event_time", "available_at"], kind="mergesort"
    ).reset_index(drop=True)
    return value[relevant]


def build_burden_lag(samples: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Return the registered 30 columns without changing sample order or index."""
    required = {"sample_id", "reference_time"}
    if required - set(samples) or samples.empty or samples.sample_id.isna().any() or samples.sample_id.astype(str).duplicated().any():
        raise ContractError("unique complete burden-lag query rows required")
    refs = parse_local_time(samples.reference_time, "burden-lag reference_time")
    ordered = normalize_events(events)
    rows: list[dict[str, float]] = []
    for ref in refs:
        visible = ordered.loc[(ordered.event_time <= ref) & (ordered.available_at <= ref)]
        row: dict[str, float] = {}
        for field in FIELDS:
            for name, outer, inner in WINDOWS:
                values = visible.loc[
                    (visible.event_time > ref - outer)
                    & (visible.event_time <= ref - inner),
                    field,
                ].dropna()
                stem = f"burden_lag__{field}__{name}"
                row[f"{stem}__mean"] = float(values.mean()) if len(values) else np.nan
                row[f"{stem}__valid_count"] = float(len(values))
        rows.append(row)
    result = pd.DataFrame(rows, index=samples.index, columns=FEATURE_COLUMNS, dtype=float)
    if list(result) != list(FEATURE_COLUMNS) or np.isinf(result.to_numpy()).any():
        raise ContractError("burden-lag feature schema or numeric values differ")
    return result


def append_burden_lag(original: pd.DataFrame, lag: pd.DataFrame) -> pd.DataFrame:
    if not original.index.equals(lag.index) or original.columns.duplicated().any() or lag.columns.duplicated().any():
        raise ContractError("old and burden-lag feature rows/schema are misaligned")
    if set(original) & set(lag) or list(lag) != list(FEATURE_COLUMNS):
        raise ContractError("burden-lag columns collide or differ")
    before = original.copy(deep=True)
    result = pd.concat([original, lag], axis=1)
    if not result.iloc[:, : len(before.columns)].equals(before):
        raise ContractError("existing feature values/dtypes/order changed")
    return result

