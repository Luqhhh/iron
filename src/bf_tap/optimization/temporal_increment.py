"""F-B uses event-time segments with the unchanged availability contract."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..exceptions import ContractError
from ..schema import validate_event_source


def segmented_operation(samples: pd.DataFrame, events: pd.DataFrame,
                        value_columns: list[str]) -> pd.DataFrame:
    validate_event_source(events, event_time="event_time", available_at="available_at",
                          value_columns=value_columns)
    relevant = ["event_time", "available_at", *value_columns]
    events = events[relevant].drop_duplicates()
    if events.event_time.duplicated().any():
        raise ContractError("conflicting rows share the same event_time")
    events = events.sort_values(["event_time", "available_at"], kind="mergesort")
    rows = []
    for ref in samples.reference_time:
        row = {}
        for a, b in ((6, 12), (12, 24), (24, 48)):
            part = events.loc[(events.event_time > ref - pd.Timedelta(hours=b))
                              & (events.event_time <= ref - pd.Timedelta(hours=a))
                              & (events.available_at <= ref)]
            for column in value_columns:
                values = part[column].dropna()
                stem = f"operation_segment__{column}__{a}_{b}h"
                row[f"{stem}__mean"] = float(values.mean()) if len(values) else np.nan
                row[f"{stem}__count"] = float(len(values))
        rows.append(row)
    return pd.DataFrame(rows, index=samples.index)
