"""Pre-origin calibration protocol. Outer labels are never inputs to fitting."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..artifacts import stable_digest
from ..exceptions import ContractError
from .alignment import aware


def inner_blocks(samples: pd.DataFrame, outer_cutoff: object) -> dict[str, pd.DataFrame]:
    cutoff = aware(outer_cutoff)
    first, second = cutoff - pd.Timedelta(days=56), cutoff - pd.Timedelta(days=28)
    required = {"sample_id", "reference_time", "label_available_at"}
    if required - set(samples) or samples["sample_id"].duplicated().any():
        raise ContractError("inner blocks require unique samples and label availability")
    t, a = samples.reference_time, samples.label_available_at
    if t.isna().any() or a.isna().any():
        raise ContractError("missing inner temporal identity")
    return {
        "selection_train": samples.loc[(t < first) & (a <= first)].copy(),
        # Selection labels must be available when calibration's estimator is fitted.
        "selection_eval": samples.loc[(t >= first) & (t < second) & (a <= second)].copy(),
        "calibration_train": samples.loc[(t < second) & (a <= second)].copy(),
        "calibration_eval": samples.loc[(t >= second) & (t < cutoff) & (a < cutoff)].copy(),
        "outer_train": samples.loc[(t < cutoff) & (a <= cutoff)].copy(),
    }


def fit_time_calibration(predictions: pd.DataFrame, *, outer_cutoff: object,
                         candidate: str, source_run: str, origin_id: str,
                         pipeline_identity: str, outer_sample_ids: list[str]) -> dict:
    cutoff = aware(outer_cutoff)
    start = cutoff - pd.Timedelta(days=28)
    required = {"sample_id", "reference_time", "label_available_at", "pred_tap_time_len",
                "tap_time_len", "prediction_fit_cutoff", "history_cutoff"}
    if required - set(predictions):
        raise ContractError("calibration predictions lack provenance columns")
    if not all((candidate, source_run, origin_id, pipeline_identity)):
        raise ContractError("calibration source identity is required")
    if predictions.sample_id.isna().any() or predictions.sample_id.astype(str).duplicated().any():
        raise ContractError("calibration cannot count repeated sample IDs as independent samples")
    if set(predictions.sample_id.astype(str)) & set(map(str, outer_sample_ids)):
        raise ContractError("calibration_fit_overlap with outer evaluation")
    for column in ("reference_time", "label_available_at", "prediction_fit_cutoff", "history_cutoff"):
        if not predictions.empty:
            predictions = predictions.copy()
            predictions[column] = predictions[column].map(aware)
    if not predictions.empty and (
        (predictions.reference_time < start).any()
        or (predictions.reference_time >= cutoff).any()
        or (predictions.label_available_at >= cutoff).any()
        or (predictions.prediction_fit_cutoff != start).any()
        or (predictions.history_cutoff != start).any()
    ):
        raise ContractError("calibration violates frozen inner block or pre-origin availability")
    residual = predictions.pred_tap_time_len.to_numpy(float) - predictions.tap_time_len.to_numpy(float)
    if not np.isfinite(residual).all():
        raise ContractError("calibration residuals must be finite")
    ordered = predictions.sort_values("sample_id").astype(str).to_dict("records")
    return {
        "schema_version": "calibration-provenance-v1", "candidate": candidate,
        "source_run": source_run, "origin_id": origin_id, "pipeline_identity": pipeline_identity,
        "outer_cutoff": str(cutoff), "inner_fit_cutoff": str(start),
        "origin_ids": [origin_id], "sample_count": len(predictions),
        "sample_ids_sha256": stable_digest(sorted(predictions.sample_id.astype(str))),
        "prediction_records_sha256": stable_digest(ordered),
        "label_available_min": str(predictions.label_available_at.min()) if len(predictions) else None,
        "label_available_max": str(predictions.label_available_at.max()) if len(predictions) else None,
        "residual_definition": "prediction_minus_actual", "sample_weight": "uniform_unique_sample",
        "scenario_weight": "single_inner_block", "minimum_samples": 100,
        "fallback": "zero_below_100" if len(predictions) < 100 else None,
        "median_prediction_minus_actual": {
            "tap_iron": 0.0, "tap_time_len": float(np.median(residual)) if len(predictions) >= 100 else 0.0},
        "independent_evaluation_sample_ids_sha256": stable_digest(sorted(map(str, outer_sample_ids))),
        "calibration_fit_overlap": False, "calibration_available_after_origin": False,
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "development_selection_not_untouched_holdout",
    }
