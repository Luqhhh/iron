from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..availability import assert_no_future
from ..exceptions import ContractError
from .history import build_history_features
from .sample import build_sample_features
from .temporal import build_temporal_features


@dataclass(frozen=True)
class FeatureResult:
    X: pd.DataFrame
    audit: dict[str, pd.DataFrame]
    categorical: tuple[str, ...]


def build_features(
    samples: pd.DataFrame,
    *,
    operation: pd.DataFrame,
    burden: pd.DataFrame,
    history: pd.DataFrame,
    fit_cutoff: pd.Timestamp,
    config: dict,
) -> FeatureResult:
    if samples["sample_id"].duplicated().any():
        raise ContractError("feature samples contain duplicate sample_id")
    sample_x = build_sample_features(samples, config.get("categorical_missing", "__MISSING__"))
    op_cfg = config["operation"]
    op_x, op_audit = build_temporal_features(
        samples,
        operation,
        prefix="operation",
        event_time="event_time",
        available_at="available_at",
        value_columns=list(op_cfg["value_columns"]),
        stale_hours=float(op_cfg["latest_max_event_age_hours"]),
        windows_hours=list(op_cfg["windows_hours"]),
    )
    burden_cfg = config["burden"]
    burden_x, burden_audit = build_temporal_features(
        samples,
        burden,
        prefix="burden",
        event_time="event_time",
        available_at="available_at",
        value_columns=list(burden_cfg["value_columns"]),
        stale_hours=float(burden_cfg["latest_max_event_age_hours"]),
    )
    history_x, history_audit = build_history_features(
        samples,
        history,
        fit_cutoff=fit_cutoff,
        available_at="available_at",
        last_k=tuple(config["history"]["last_k_mean"]),
    )
    for audit in (op_audit, burden_audit, history_audit):
        assert_no_future(audit)
    combined = pd.concat([sample_x, op_x, burden_x, history_x], axis=1)
    if combined.columns.duplicated().any():
        raise ContractError("feature builder produced duplicate columns")
    return FeatureResult(
        X=combined,
        audit={"operation": op_audit, "burden": burden_audit, "history": history_audit},
        categorical=("spout_no",),
    )
