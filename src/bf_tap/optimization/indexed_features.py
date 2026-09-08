"""OPT-09 F-C/F-D: known opening metadata and visible burden events only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import file_identities, stable_digest, verify_file_identities
from ..exceptions import ContractError
from ..io import read_csv
from ..schema import validate_event_source
from .alignment import aware

INDEX_COLUMNS = ["sample_id", "spout_no", "reference_time"]


def normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    if set(frame.columns) != set(INDEX_COLUMNS) or frame[INDEX_COLUMNS].isna().any().any():
        raise ContractError("known index accepts only non-null sample_id, spout_no, reference_time")
    result = frame[INDEX_COLUMNS].copy()
    result["sample_id"] = result.sample_id.astype("string")
    result["spout_no"] = result.spout_no.astype("string")
    result["reference_time"] = result.reference_time.map(aware)
    return result


def load_known_index(paths: list[str | Path]) -> tuple[pd.DataFrame, dict]:
    if not paths or len({str(Path(p).resolve()) for p in paths}) != len(paths):
        raise ContractError("explicit unique known-index source paths are required")
    identities = file_identities({str(Path(p)): p for p in paths})
    parts = [normalize_index(read_csv(p, usecols=INDEX_COLUMNS, required_columns=INDEX_COLUMNS,
                                     time_columns=["reference_time"])) for p in paths]
    combined = pd.concat(parts, ignore_index=True)
    unique = combined.drop_duplicates(INDEX_COLUMNS)
    if unique.sample_id.duplicated().any():
        raise ContractError("known-index sample identity conflicts between declared inputs")
    unique = unique.sort_values(["reference_time", "sample_id"], kind="mergesort").reset_index(drop=True)
    verify_file_identities(identities)
    return unique, {"schema_version": "known-index-v1", "sources": identities,
                    "columns_read": INDEX_COLUMNS, "rows": len(unique),
                    "exact_duplicate_rows_removed": len(combined) - len(unique),
                    "metadata_sha256": stable_digest(unique.astype(str).to_dict("records")),
                    "reference_min": str(unique.reference_time.min()), "reference_max": str(unique.reference_time.max()),
                    "completeness": "relative_to_declared_inputs_not_full_production",
                    "targets_or_closing_times_read": False}


def known_index_features(samples: pd.DataFrame, known_index: pd.DataFrame) -> pd.DataFrame:
    query, index = normalize_index(samples), normalize_index(known_index)
    if query.sample_id.duplicated().any() or index.sample_id.duplicated().any():
        raise ContractError("known-index query and source IDs must be unique")
    joined = query.merge(index, on="sample_id", suffixes=("_query", "_index"), validate="one_to_one")
    if (len(joined) != len(query)
            or (joined.spout_no_query != joined.spout_no_index).any()
            or (joined.reference_time_query != joined.reference_time_index).any()):
        raise ContractError("query metadata is missing from or disagrees with declared index sources")
    all_times = np.sort(index.reference_time.map(lambda t: t.value).to_numpy(np.int64))
    spout_times = {str(s): np.sort(part.reference_time.map(lambda t: t.value).to_numpy(np.int64))
                   for s, part in index.groupby("spout_no", sort=True)}
    rows = []
    for sample in query.itertuples(index=False):
        ref = sample.reference_time.value
        row = {}
        for group, times in (("all", all_times), ("spout", spout_times[str(sample.spout_no)])):
            end = int(np.searchsorted(times, ref, side="left"))  # all same-time rows excluded
            stem = f"known_index__{group}"
            row[f"{stem}__previous_interval_minutes"] = (ref - times[end - 1]) / 60e9 if end else np.nan
            for hours in (6, 24):
                start = int(np.searchsorted(times, ref - hours * 3600 * 10**9, side="right"))
                row[f"{stem}__{hours}h__known_count"] = float(max(0, end - start))
        rows.append(row)
    return pd.DataFrame(rows, index=samples.index)


def burden_event_changes(samples: pd.DataFrame, events: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    validate_event_source(events, event_time="event_time", available_at="available_at", value_columns=value_columns)
    events = events[["event_time", "available_at", *value_columns]].drop_duplicates()
    if events.event_time.duplicated().any():
        raise ContractError("conflicting burden rows share event_time")
    events = events.sort_values(["event_time", "available_at"], kind="mergesort")
    rows = []
    for ref in samples.reference_time:
        visible = events.loc[(events.event_time <= ref) & (events.available_at <= ref)]
        row = {}
        for column in value_columns:
            row[f"burden_event__{column}__previous_delta"] = (
                float(visible.iloc[-1][column] - visible.iloc[-2][column]) if len(visible) >= 2 else np.nan)
        for hours in (24, 72):
            row[f"burden_event__{hours}h__count"] = float((visible.event_time > ref - pd.Timedelta(hours=hours)).sum())
        # The frozen baseline already supplies burden__event_age_hours.
        rows.append(row)
    return pd.DataFrame(rows, index=samples.index)
