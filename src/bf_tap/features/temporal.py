from __future__ import annotations

import numpy as np
import pandas as pd

from ..exceptions import ContractError
from ..schema import validate_event_source


def build_temporal_features(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    *,
    prefix: str,
    event_time: str,
    available_at: str,
    value_columns: list[str],
    stale_hours: float,
    windows_hours: list[int] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_event_source(
        events,
        event_time=event_time,
        available_at=available_at,
        value_columns=value_columns,
    )
    duplicate_times = events.duplicated([event_time], keep=False)
    if duplicate_times.any():
        relevant = [event_time, available_at, *value_columns]
        conflicting = any(
            len(group.drop_duplicates()) > 1
            for _, group in events.loc[duplicate_times, relevant].groupby(event_time, dropna=False)
        )
        if conflicting:
            raise ContractError("conflicting rows share the same event_time")
        events = events.drop_duplicates(relevant)
    ordered = events.sort_values([event_time, available_at], kind="mergesort")
    rows: list[dict[str, float]] = []
    audits: list[dict[str, object]] = []
    for sample in samples.itertuples(index=False):
        ref = sample.reference_time
        visible = ordered.loc[
            (ordered[event_time] <= ref) & (ordered[available_at] <= ref)
        ]
        row: dict[str, float] = {}
        audit: dict[str, object] = {
            "reference_time": ref,
            "max_available_at": visible[available_at].max() if len(visible) else pd.NaT,
            f"{prefix}__visible_count": len(visible),
        }
        if len(visible):
            last = visible.iloc[-1]
            age_hours = (ref - last[event_time]).total_seconds() / 3600.0
        else:
            last = None
            age_hours = np.nan
        row[f"{prefix}__event_age_hours"] = age_hours
        stale = bool(np.isfinite(age_hours) and age_hours > stale_hours)
        row[f"{prefix}__stale"] = float(stale)
        for column in value_columns:
            value = np.nan if last is None or stale else float(last[column])
            row[f"{prefix}__{column}__latest"] = value
            row[f"{prefix}__{column}__missing"] = float(pd.isna(value))
            for hours in windows_hours:
                lower = ref - pd.Timedelta(hours=hours)
                window = visible.loc[(visible[event_time] > lower) & (visible[event_time] <= ref), column]
                valid = pd.to_numeric(window, errors="coerce").dropna()
                stem = f"{prefix}__{column}__{hours}h"
                row[f"{stem}__mean"] = float(valid.mean()) if len(valid) else np.nan
                row[f"{stem}__std"] = float(valid.std(ddof=0)) if len(valid) else np.nan
                row[f"{stem}__count"] = float(len(valid))
        rows.append(row)
        audits.append(audit)
    return pd.DataFrame(rows, index=samples.index), pd.DataFrame(audits, index=samples.index)
