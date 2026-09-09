from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ..exceptions import ContractError
from .config import DriftExperiment


_TARGETS = ("tap_iron", "tap_time_len")
_PREDICTION_COLUMNS = (
    "sample_id",
    "pred_tap_iron",
    "pred_tap_time_len",
)


def recent_target_medians(
    samples: pd.DataFrame,
    fit_cutoff: pd.Timestamp,
    window_days: int,
) -> tuple[dict[str, float], dict[str, object]]:
    required = {
        "sample_id",
        "reference_time",
        "label_available_at",
        *_TARGETS,
    }
    missing = required - set(samples)
    if missing:
        raise ContractError(f"recent-window samples missing columns: {sorted(missing)}")
    if window_days != 30:
        raise ContractError("optimization-v0.3 recent window must be exactly 30 days")
    cutoff = pd.Timestamp(fit_cutoff)
    if cutoff.tzinfo is None:
        raise ContractError("fit_cutoff must be timezone-aware")
    window_start = cutoff - pd.Timedelta(days=window_days)
    in_window = (samples["reference_time"] >= window_start) & (
        samples["reference_time"] < cutoff
    )
    ready = samples["label_available_at"] <= cutoff
    eligible = samples.loc[in_window & ready]
    if eligible.empty:
        raise ContractError("recent 30-day eligible label window is empty")
    values = eligible[list(_TARGETS)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ContractError("recent 30-day targets must be finite")
    medians = {target: float(values[target].median()) for target in _TARGETS}
    return medians, {
        "window_days": window_days,
        "window_start": str(window_start),
        "fit_cutoff": str(cutoff),
        "candidate_rows": int(in_window.sum()),
        "eligible_rows": int((in_window & ready).sum()),
        "excluded_label_unavailable_rows": int((in_window & ~ready).sum()),
    }


def _aligned(
    eval_samples: pd.DataFrame,
    prediction: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    if "sample_id" not in eval_samples:
        raise ContractError("evaluation samples missing sample_id")
    missing = set(_PREDICTION_COLUMNS) - set(prediction)
    if missing:
        raise ContractError(f"{scope} prediction missing columns: {sorted(missing)}")
    order = eval_samples[["sample_id"]].copy()
    order["sample_id"] = order["sample_id"].astype("string")
    values = prediction[list(_PREDICTION_COLUMNS)].copy()
    values["sample_id"] = values["sample_id"].astype("string")
    if (
        order["sample_id"].isna().any()
        or values["sample_id"].isna().any()
        or order["sample_id"].duplicated().any()
        or values["sample_id"].duplicated().any()
    ):
        raise ContractError(f"{scope} sample_id values must be unique and non-null")
    try:
        aligned = order.merge(
            values,
            on="sample_id",
            how="left",
            sort=False,
            validate="one_to_one",
            indicator=True,
        )
    except pd.errors.MergeError as exc:
        raise ContractError(f"{scope} sample_id values are not one-to-one") from exc
    if (
        len(aligned) != len(values)
        or (aligned["_merge"] != "both").any()
    ):
        raise ContractError(f"{scope} sample_id set differs from evaluation samples")
    aligned = aligned.drop(columns="_merge")
    numeric = aligned[["pred_tap_iron", "pred_tap_time_len"]].apply(
        pd.to_numeric, errors="coerce"
    )
    aligned["sample_id"] = eval_samples["sample_id"].to_numpy(copy=True)
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ContractError(f"{scope} predictions must be finite")
    aligned[["pred_tap_iron", "pred_tap_time_len"]] = numeric
    return aligned


def derive_stage1_predictions(
    eval_samples: pd.DataFrame,
    e00: pd.DataFrame,
    b1: pd.DataFrame,
    e09: pd.DataFrame,
    recent: Mapping[str, float],
    experiment: DriftExperiment,
) -> dict[str, pd.DataFrame]:
    if set(recent) != set(_TARGETS):
        raise ContractError("recent medians must contain exactly both targets")
    if not all(np.isfinite(float(recent[target])) for target in _TARGETS):
        raise ContractError("recent medians must be finite")
    learned = _aligned(eval_samples, e00, "E00")
    control = _aligned(eval_samples, b1, "B1")
    process_change = _aligned(eval_samples, e09, "E09")

    m1 = learned[["sample_id"]].copy()
    for target in _TARGETS:
        column = f"pred_{target}"
        weight = experiment.m1_catboost_weights[target]
        m1[column] = (
            weight * learned[column]
            + (1.0 - weight) * control[column]
        ).clip(lower=0.0)

    c1 = m1.copy()
    c1["pred_tap_time_len"] = max(0.0, float(recent["tap_time_len"]))
    c2 = process_change.copy()
    for column in ("pred_tap_iron", "pred_tap_time_len"):
        c2[column] = c2[column].clip(lower=0.0)

    return {
        experiment.candidate_ids["m1"]: m1,
        experiment.candidate_ids["c1"]: c1,
        experiment.candidate_ids["c2"]: c2,
    }


def derive_stage2_prediction(
    c1: pd.DataFrame,
    c2: pd.DataFrame,
    experiment: DriftExperiment,
) -> pd.DataFrame:
    ids = c1[["sample_id"]].copy()
    first = _aligned(ids, c1, experiment.candidate_ids["c1"])
    second = _aligned(ids, c2, experiment.candidate_ids["c2"])
    result = first[["sample_id"]].copy()
    result["pred_tap_iron"] = second["pred_tap_iron"].clip(lower=0.0)
    result["pred_tap_time_len"] = first["pred_tap_time_len"].clip(lower=0.0)
    return result
