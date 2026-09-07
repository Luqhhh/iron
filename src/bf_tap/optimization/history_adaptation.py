from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..artifacts import stable_digest
from ..exceptions import ContractError


@dataclass(frozen=True)
class HistoryViewSet:
    samples: pd.DataFrame
    targets: pd.DataFrame
    sample_weight: pd.Series
    audit: pd.DataFrame


def build_history_views(
    samples: pd.DataFrame,
    ages_days: tuple[int, ...],
    *,
    inner_validation_fraction: float = 0.2,
) -> HistoryViewSet:
    """Create grouped, equally weighted stale-history views of labeled samples."""
    required = {
        "sample_id",
        "reference_time",
        "tap_iron",
        "tap_time_len",
    }
    missing = required - set(samples)
    if missing:
        raise ContractError(f"history adaptation input missing columns: {sorted(missing)}")
    if samples.empty or samples["sample_id"].duplicated().any():
        raise ContractError("history adaptation requires non-empty unique original samples")
    if (
        not ages_days
        or tuple(sorted(set(ages_days))) != ages_days
        or ages_days[0] != 0
        or any(isinstance(age, bool) or not isinstance(age, int) or age < 0 for age in ages_days)
    ):
        raise ContractError("history view ages must be sorted unique nonnegative integers starting at 0")
    if not 0.0 < inner_validation_fraction < 0.5:
        raise ContractError("inner validation fraction must be in (0, 0.5)")

    originals = samples.sort_values(["reference_time", "sample_id"], kind="mergesort")
    split_at = max(1, min(len(originals) - 1, int(np.floor(len(originals) * (1.0 - inner_validation_fraction)))))
    inner_validation_ids = set(originals.iloc[split_at:]["sample_id"].astype(str))
    view_count = len(ages_days)
    frames: list[pd.DataFrame] = []
    for age in ages_days:
        frame = samples.copy()
        original_ids = frame["sample_id"].astype("string")
        frame["original_sample_id"] = original_ids
        frame["history_age_days"] = age
        frame["history_origin"] = frame["reference_time"] - pd.Timedelta(days=age)
        frame["view_id"] = original_ids + f"::history_age_{age}d"
        frame["sample_id"] = frame["view_id"].astype("string")
        frame["sample_weight"] = 1.0 / view_count
        frame["inner_partition"] = np.where(
            original_ids.astype(str).isin(inner_validation_ids),
            "inner_validation",
            "inner_train",
        )
        frames.append(frame)
    views = pd.concat(frames, ignore_index=True).sort_values(
        ["reference_time", "original_sample_id", "history_age_days"], kind="mergesort"
    ).reset_index(drop=True)
    if views["view_id"].duplicated().any():
        raise ContractError("synthetic history view IDs are not unique")
    grouped = views.groupby("original_sample_id", sort=False)
    if not (grouped.size() == view_count).all():
        raise ContractError("each original sample must have every synthetic history view")
    if not np.allclose(grouped["sample_weight"].sum().to_numpy(dtype=float), 1.0):
        raise ContractError("synthetic history view weights must total one per original sample")
    if (grouped["inner_partition"].nunique() != 1).any():
        raise ContractError("views of an original sample crossed inner partitions")
    if (views["history_origin"] > views["reference_time"]).any():
        raise ContractError("synthetic history origin follows sample reference time")

    targets = views[["tap_iron", "tap_time_len"]].copy()
    weights = views["sample_weight"].astype(float).copy()
    audit_columns = [
        "original_sample_id",
        "view_id",
        "history_age_days",
        "history_origin",
        "sample_weight",
        "inner_partition",
    ]
    audit = views[audit_columns].copy()
    audit["view_contract_sha256"] = stable_digest(
        {
            "ages_days": list(ages_days),
            "inner_validation_fraction": inner_validation_fraction,
            "weight_per_view": 1.0 / view_count,
        }
    )
    return HistoryViewSet(views, targets, weights, audit)
