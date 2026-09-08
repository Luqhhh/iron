"""Calendar identities for OPT-07; never infer a horizon from a score."""
from __future__ import annotations

import pandas as pd

from ..exceptions import ContractError


def aware(value: object) -> pd.Timestamp:
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ContractError("time identity must be a non-null timezone-aware timestamp")
    return value.tz_convert("Asia/Shanghai")


def calendar_horizon(cutoff: object, reference_time: object) -> int:
    cutoff, reference = aware(cutoff), aware(reference_time)
    if reference < cutoff:
        raise ContractError("prediction precedes model cutoff")
    return 12 * (reference.year - cutoff.year) + reference.month - cutoff.month + 1


def stage_alignment(*, component: str, stage: str, fit_cutoff: object,
                    history_cutoff: object, label_available_cutoff: object,
                    reference_times: pd.Series, identity: dict,
                    development_cells: dict[int, list[str]]) -> dict:
    if not component or not stage or reference_times.empty:
        raise ContractError("stage alignment requires component, stage and prediction times")
    if set(identity) != {"source", "data", "candidate", "component"} or not all(identity.values()):
        raise ContractError("stage alignment requires complete source/data/candidate/component identities")
    fit, history, labels = map(aware, (fit_cutoff, history_cutoff, label_available_cutoff))
    if history > fit or labels > fit:
        raise ContractError("history/label cutoff exceeds model cutoff")
    times = reference_times.map(aware)
    horizons = sorted({calendar_horizon(fit, time) for time in times})
    return {
        "schema_version": "stage-alignment-v1", "component": component, "stage": stage,
        "fit_cutoff": str(fit), "history_cutoff": str(history),
        "label_available_cutoff": str(labels), "identity": identity,
        "prediction_start": str(times.min()), "prediction_end_inclusive": str(times.max()),
        "calendar_horizons": horizons,
        "elapsed_days": {"min": (times.min() - fit).total_seconds() / 86400,
                         "max": (times.max() - fit).total_seconds() / 86400},
        "development_cells": {f"H{h}": development_cells.get(h, []) for h in horizons},
        "uncovered_horizons": [h for h in horizons if not development_cells.get(h)],
    }
