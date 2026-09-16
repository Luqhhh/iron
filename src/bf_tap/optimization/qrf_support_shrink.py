"""Causal support-normalized shrinkage for the frozen QRF time branch.

This module contains no model fitting and no stage-specific paths.  It keeps
the V22 mathematics, temporal checks, scalar LAD certificate, and prediction
serialization independent from the lifecycle runner.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..artifacts import atomic_write_json, stable_digest
from ..exceptions import ContractError
from .structural import lad_coefficient


CANDIDATE = "V22_CAUSAL_H2_QRF_SHRINK"
REFERENCE = "V10"
REPLAY_CONTROL = "V21_REPLAY"
PROTOCOL = "QRF_FULLTRAIN_LEAF_v1"
SUPPORT_ABSOLUTE_TOLERANCE = 1e-8
SUPPORT_RELATIVE_TOLERANCE = 1e-12
MINIMUM_OOF_ROWS = 100
PREDICTION_COLUMNS = ["pred_tap_iron", "pred_tap_time_len"]
DIRECTION_REQUIRED = [
    "sample_id", "spout_no", "reference_time",
    "model_cutoff", "training_identity_sha256", "model_bundle_sha256",
    "Q", "effective_neighbors", "N", "u", "M", "median_source_sha256", "d",
]
BANK_REQUIRED = [*DIRECTION_REQUIRED[:3], "label_available_at", *DIRECTION_REQUIRED[3:]]


def _aware(value, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ContractError(f"{name} must be timezone-aware")
    return result.tz_convert("Asia/Shanghai")


def _times(values, name: str) -> pd.Series:
    try:
        result = pd.to_datetime(values, utc=True).dt.tz_convert("Asia/Shanghai")
    except (TypeError, ValueError, AttributeError) as exc:
        raise ContractError(f"invalid {name}") from exc
    if result.isna().any():
        raise ContractError(f"null {name}")
    return result


def normalized_support(
    effective_neighbors: Sequence[float] | np.ndarray,
    training_rows: int,
    *,
    absolute_tolerance: float = SUPPORT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = SUPPORT_RELATIVE_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``u=n_eff/N`` and ``a=1-u`` after a narrow boundary check.

    The denominator is a scalar certified count of unique original training
    rows.  Values outside [1, N] by more than max(abs_tol, rel_tol*N) fail;
    only values inside that numerical tolerance are snapped to the boundary.
    """
    if isinstance(training_rows, bool) or not isinstance(training_rows, (int, np.integer)) or training_rows < 1:
        raise ContractError("N must be a positive integer unique-training-row count")
    if not np.isfinite([absolute_tolerance, relative_tolerance]).all() or absolute_tolerance < 0 or relative_tolerance < 0:
        raise ContractError("support tolerances must be finite and nonnegative")
    values = np.asarray(effective_neighbors, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ContractError("effective-neighbor support must be a finite nonempty vector")
    tolerance = max(float(absolute_tolerance), float(relative_tolerance) * training_rows)
    if (values < 1.0 - tolerance).any() or (values > training_rows + tolerance).any():
        raise ContractError("effective-neighbor support is materially outside [1,N]")
    normalized = values.copy()
    normalized[normalized < 1.0] = 1.0
    normalized[normalized > training_rows] = float(training_rows)
    u = normalized / float(training_rows)
    a = 1.0 - u
    if (u < 0).any() or (u > 1).any() or (a < 0).any() or (a > 1).any():
        raise ContractError("normalized support invariant failed")
    return u, a


def _history(history: pd.DataFrame) -> pd.DataFrame:
    if "label_available_at" not in history and "available_at" in history:
        history = history.rename(columns={"available_at": "label_available_at"})
    required = {"sample_id", "spout_no", "reference_time", "label_available_at", "tap_time_len"}
    if required - set(history) or history.empty or history.sample_id.isna().any() or history.sample_id.duplicated().any():
        raise ContractError("unique certified training history is required")
    result = history[list(required)].copy()
    result["sample_id"] = result.sample_id.astype(str)
    result["spout_no"] = result.spout_no.astype(str)
    result["reference_time"] = _times(result.reference_time, "history reference_time")
    result["label_available_at"] = _times(result.label_available_at, "history label_available_at")
    values = result.tap_time_len.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ContractError("history responses must be finite and nonnegative")
    return result


def fixed_cutoff_median(
    history: pd.DataFrame,
    cutoff,
    spout: str,
    *,
    window_days: int = 60,
) -> dict:
    """Create a source-bound ordinary-median certificate for one cutoff/spout."""
    if isinstance(window_days, bool) or not isinstance(window_days, int) or window_days != 60:
        raise ContractError("V22 fixes the median window at 60 days")
    h = _history(history)
    c = _aware(cutoff, "model cutoff")
    s = str(spout)
    legal = h.loc[
        (h.spout_no == s)
        & (h.reference_time < c)
        & (h.label_available_at <= c)
    ].sort_values(["reference_time", "sample_id"], kind="mergesort")
    recent = legal.loc[legal.reference_time >= c - pd.Timedelta(days=window_days)]
    if not recent.empty:
        selected, source = recent, "recent60"
    elif not legal.empty:
        selected, source = legal, "all_legal_same_spout_fallback"
    else:
        selected, source = legal, "no_legal_same_spout_history"
    value = None if selected.empty else float(selected.tap_time_len.median())
    rows = selected[["sample_id", "reference_time", "label_available_at", "tap_time_len"]]
    certificate = {
        "kind": "V22_FIXED_CUTOFF_SPOUT_MEDIAN_v1",
        "cutoff": c.isoformat(),
        "spout_no": s,
        "window_days": window_days,
        "source": source,
        "fallback": source != "recent60",
        "eligible_rows": int(len(legal)),
        "selected_rows": int(len(selected)),
        "selected_ids_sha256": stable_digest(selected.sample_id.tolist()),
        "selected_rows_sha256": stable_digest(rows.astype({"sample_id": str}).to_dict("records")),
        "median": value,
        "statistic": "pandas_ordinary_median",
    }
    certificate["certificate_sha256"] = stable_digest(certificate)
    return certificate


def fixed_cutoff_medians(
    history: pd.DataFrame,
    cutoff,
    spouts: Sequence[str] = ("1", "2"),
) -> dict[str, dict]:
    keys = [str(s) for s in spouts]
    if not keys or len(set(keys)) != len(keys):
        raise ContractError("unique declared spouts are required")
    return {s: fixed_cutoff_median(history, cutoff, s) for s in keys}


def verify_median_certificate(history: pd.DataFrame, certificate: Mapping) -> bool:
    if certificate.get("kind") != "V22_FIXED_CUTOFF_SPOUT_MEDIAN_v1":
        raise ContractError("unknown median certificate")
    expected = fixed_cutoff_median(
        history,
        certificate["cutoff"],
        str(certificate["spout_no"]),
        window_days=int(certificate["window_days"]),
    )
    if dict(certificate) != expected:
        raise ContractError("stored cutoff median or its source certificate differs")
    return True


def build_prediction_directions(
    queries: pd.DataFrame,
    qrf_median: Sequence[float],
    effective_neighbors: Sequence[float],
    training_rows: int,
    medians: Mapping[str, Mapping],
    *,
    model_cutoff,
    training_identity_sha256: str,
    model_bundle_sha256: str,
) -> pd.DataFrame:
    """Build target-free V22 directions for arbitrary prediction metadata."""
    required = {"sample_id", "spout_no", "reference_time"}
    if required - set(queries) or queries.empty or queries.sample_id.isna().any() or queries.sample_id.duplicated().any():
        raise ContractError("unique complete prediction query metadata required")
    if not training_identity_sha256 or not model_bundle_sha256:
        raise ContractError("certified model and training identities required")
    q = np.asarray(qrf_median, dtype=np.float64)
    neff = np.asarray(effective_neighbors, dtype=np.float64)
    if q.shape != (len(queries),) or neff.shape != q.shape or not np.isfinite(q).all() or (q < 0).any():
        raise ContractError("aligned finite nonnegative QRF medians required")
    u, attenuation = normalized_support(neff, training_rows)
    c = _aware(model_cutoff, "model cutoff")
    result = queries[["sample_id", "spout_no", "reference_time"]].copy()
    result["sample_id"] = result.sample_id.astype(str)
    result["spout_no"] = result.spout_no.astype(str)
    result["reference_time"] = _times(result.reference_time, "query reference_time")
    if (result.reference_time < c).any():
        raise ContractError("query precedes model cutoff")
    median_values, median_sources, missing = [], [], []
    for spout in result.spout_no:
        cert = medians.get(spout)
        if cert is None or cert.get("median") is None:
            median_values.append(float("nan"))
            median_sources.append("NO_CERTIFIED_SAME_SPOUT_MEDIAN")
            missing.append(True)
        else:
            if pd.Timestamp(cert["cutoff"]) != c or str(cert["spout_no"]) != spout:
                raise ContractError("median certificate cutoff/spout differs from query model")
            if stable_digest({k: v for k, v in cert.items() if k != "certificate_sha256"}) != cert["certificate_sha256"]:
                raise ContractError("median certificate self-digest differs")
            median_values.append(float(cert["median"]))
            median_sources.append(str(cert["certificate_sha256"]))
            missing.append(False)
    m = np.asarray(median_values, dtype=np.float64)
    missing_array = np.asarray(missing, dtype=bool)
    d = np.zeros(len(result), dtype=np.float64)
    d[~missing_array] = attenuation[~missing_array] * (m[~missing_array] - q[~missing_array])
    if not np.isfinite(d).all():
        raise ContractError("nonfinite support-shrink direction")
    result["model_cutoff"] = c
    result["training_identity_sha256"] = training_identity_sha256
    result["model_bundle_sha256"] = model_bundle_sha256
    result["Q"] = q
    result["effective_neighbors"] = neff
    result["N"] = int(training_rows)
    result["u"] = u
    result["M"] = m
    result["median_source_sha256"] = median_sources
    result["d"] = d
    return result[DIRECTION_REQUIRED]


def build_direction_bank(
    queries: pd.DataFrame,
    qrf_median: Sequence[float],
    effective_neighbors: Sequence[float],
    training_rows: int,
    medians: Mapping[str, Mapping],
    *,
    model_cutoff,
    training_identity_sha256: str,
    model_bundle_sha256: str,
) -> pd.DataFrame:
    """Build a target-free H2 bank with label-availability metadata."""
    if "label_available_at" not in queries:
        raise ContractError("H2 bank requires label_available_at metadata")
    directions = build_prediction_directions(
        queries, qrf_median, effective_neighbors, training_rows, medians,
        model_cutoff=model_cutoff,
        training_identity_sha256=training_identity_sha256,
        model_bundle_sha256=model_bundle_sha256,
    )
    label_available = _times(queries.label_available_at, "query label_available_at")
    if (label_available < directions.reference_time).any():
        raise ContractError("query label precedes reference_time")
    directions.insert(3, "label_available_at", label_available.to_numpy())
    return directions[BANK_REQUIRED]


def verify_prediction_directions(value: pd.DataFrame) -> bool:
    if set(DIRECTION_REQUIRED) - set(value) or value.empty or value.sample_id.isna().any() or value.sample_id.duplicated().any():
        raise ContractError("complete unique V22 prediction directions required")
    nonnullable = [name for name in DIRECTION_REQUIRED if name != "M"]
    if value[nonnullable].isna().any().any():
        raise ContractError("null V22 prediction direction evidence")
    if {"tap_iron", "tap_time_len", "pred_tap_iron", "pred_tap_time_len"} & set(value):
        raise ContractError("V22 directions must not contain targets or composed predictions")
    checked = value.copy()
    for name in ("reference_time", "model_cutoff"):
        checked[name] = _times(checked[name], f"direction {name}")
    numeric = checked[["Q", "effective_neighbors", "N", "u", "d"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (checked.Q < 0).any() or (checked.reference_time < checked.model_cutoff).any():
        raise ContractError("invalid numeric or temporal prediction directions")
    for _, group in checked.groupby(["model_cutoff", "training_identity_sha256", "model_bundle_sha256"], sort=True):
        counts = group.N.unique()
        if len(counts) != 1:
            raise ContractError("support denominator varies with prediction batch")
        if not float(counts[0]).is_integer():
            raise ContractError("support denominator must be an integer training count")
        n = int(counts[0])
        expected_u, _ = normalized_support(group.effective_neighbors.to_numpy(), n)
        if not np.array_equal(expected_u, group.u.to_numpy(dtype=float)):
            raise ContractError("stored normalized support differs")
        known = group.M.notna().to_numpy()
        expected_d = np.zeros(len(group))
        expected_d[known] = (1.0 - expected_u[known]) * (group.M.to_numpy()[known] - group.Q.to_numpy()[known])
        if not np.allclose(expected_d, group.d.to_numpy(), rtol=0, atol=1e-12):
            raise ContractError("stored shrink direction differs")
    return True


def verify_h2_bank(bank: pd.DataFrame, expected_training_rows: Mapping[int, int] | None = None) -> bool:
    if set(BANK_REQUIRED) - set(bank) or bank.empty or bank.sample_id.isna().any() or bank.sample_id.duplicated().any():
        raise ContractError("complete unique V22 H2 bank required")
    verify_prediction_directions(bank)
    value = bank.copy()
    for name in ("reference_time", "label_available_at", "model_cutoff"):
        value[name] = _times(value[name], f"bank {name}")
    if not ((value.model_cutoff < value.reference_time) & (value.label_available_at >= value.reference_time)).all():
        raise ContractError("H2 bank temporal boundary differs")
    for cutoff, group in value.groupby("model_cutoff", sort=True):
        starts = cutoff + pd.DateOffset(months=1)
        ends = cutoff + pd.DateOffset(months=2)
        if not ((group.reference_time >= starts) & (group.reference_time < ends)).all():
            raise ContractError("bank is not a genuine calendar H2 forecast")
        n = int(group.N.iloc[0])
        if expected_training_rows is not None and expected_training_rows.get(cutoff.month) != n:
            raise ContractError("certified training row count differs")
    return True


def select_outer_oof(
    bank: pd.DataFrame,
    certified_history: pd.DataFrame,
    outer_cutoff,
    spout: str,
    *,
    minimum: int = MINIMUM_OOF_ROWS,
) -> tuple[pd.DataFrame, dict]:
    """Attach only labels available at the outer cutoff for one spout."""
    verify_h2_bank(bank)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum != MINIMUM_OOF_ROWS:
        raise ContractError("V22 fixes the per-spout minimum at 100")
    c = _aware(outer_cutoff, "outer cutoff")
    b = bank.copy()
    for name in ("reference_time", "label_available_at", "model_cutoff"):
        b[name] = _times(b[name], f"bank {name}")
    selected = b.loc[
        (b.spout_no.astype(str) == str(spout))
        & (b.reference_time < c)
        & (b.model_cutoff < c)
        & (b.label_available_at <= c)
    ].sort_values(["reference_time", "sample_id"], kind="mergesort")
    h = _history(certified_history).set_index("sample_id")
    if not set(selected.sample_id.astype(str)) <= set(h.index):
        raise ContractError("OOF label absent from certified outer history")
    if not selected.empty:
        labels = h.loc[selected.sample_id.astype(str)]
        if (
            not np.array_equal(labels.spout_no.to_numpy(), selected.spout_no.astype(str).to_numpy())
            or not np.array_equal(labels.reference_time.to_numpy(), selected.reference_time.to_numpy())
            or not np.array_equal(labels.label_available_at.to_numpy(), selected.label_available_at.to_numpy())
            or (labels.reference_time >= c).any()
            or (labels.label_available_at > c).any()
        ):
            raise ContractError("certified OOF label metadata or outer boundary differs")
        selected = selected.copy()
        selected["tap_time_len"] = labels.tap_time_len.to_numpy(dtype=float)
    status = {
        "outer_cutoff": c.isoformat(),
        "spout_no": str(spout),
        "eligible_rows": int(len(selected)),
        "minimum_rows": minimum,
        "fit_eligible": bool(len(selected) >= minimum),
        "selected_ids_sha256": stable_digest(selected.sample_id.astype(str).tolist()),
    }
    return selected, status


def lambda_certificate(selected: pd.DataFrame, value: float, *, minimum: int = MINIMUM_OOF_ROWS) -> dict:
    required = {"sample_id", "Q", "d", "tap_time_len"}
    if required - set(selected) or selected.sample_id.duplicated().any():
        raise ContractError("aligned unique scalar-fit rows required")
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ContractError("lambda must be in [0,1]")
    q = selected.Q.to_numpy(dtype=float)
    d = selected.d.to_numpy(dtype=float)
    y = selected.tap_time_len.to_numpy(dtype=float)
    if not all(np.isfinite(v).all() for v in (q, d, y)):
        raise ContractError("nonfinite scalar-fit data")
    if len(selected) < minimum:
        if value != 0:
            raise ContractError("insufficient rows require lambda=0")
        return {
            "lambda": 0.0,
            "rows": int(len(selected)),
            "fallback": "INSUFFICIENT_OOF_ROWS",
            "fit_performed": False,
            "ids_sha256": stable_digest(selected.sample_id.astype(str).tolist()),
        }
    if not np.any(d != 0):
        if value != 0:
            raise ContractError("zero direction requires lambda=0")
        return {
            "lambda": 0.0,
            "rows": int(len(selected)),
            "fallback": "ZERO_DIRECTION",
            "fit_performed": False,
            "ids_sha256": stable_digest(selected.sample_id.astype(str).tolist()),
            "labels_sha256": stable_digest(selected[["sample_id", "tap_time_len"]].to_dict("records")),
            "inputs_sha256": stable_digest(selected[["sample_id", "Q", "d"]].to_dict("records")),
            "objective": float(np.abs(y - q).sum()),
            "smallest_minimizer_verified": True,
        }
    optimum = lad_coefficient(y, q, d)
    if value != optimum:
        raise ContractError("stored lambda is not the constrained LAD smallest minimizer")
    residual = y - q - value * d
    return {
        "lambda": float(value),
        "rows": int(len(selected)),
        "fallback": None,
        "fit_performed": True,
        "ids_sha256": stable_digest(selected.sample_id.astype(str).tolist()),
        "labels_sha256": stable_digest(selected[["sample_id", "tap_time_len"]].to_dict("records")),
        "inputs_sha256": stable_digest(selected[["sample_id", "Q", "d"]].to_dict("records")),
        "objective": float(np.abs(residual).sum()),
        "smallest_minimizer_verified": True,
    }


def fit_lambda(selected: pd.DataFrame, *, minimum: int = MINIMUM_OOF_ROWS) -> dict:
    if len(selected) < minimum:
        return lambda_certificate(selected, 0.0, minimum=minimum)
    value = lad_coefficient(
        selected.tap_time_len.to_numpy(dtype=float),
        selected.Q.to_numpy(dtype=float),
        selected.d.to_numpy(dtype=float),
    )
    return lambda_certificate(selected, value, minimum=minimum)


def verify_lambda_certificate(
    selected: pd.DataFrame,
    certificate: Mapping,
    *,
    minimum: int = MINIMUM_OOF_ROWS,
) -> bool:
    """Verify a saved LAD optimum without invoking any fitting primitive."""
    required = {"sample_id", "Q", "d", "tap_time_len"}
    if required - set(selected) or selected.sample_id.duplicated().any():
        raise ContractError("aligned unique scalar-certificate rows required")
    value = float(certificate.get("lambda", float("nan")))
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ContractError("invalid saved lambda certificate")
    q = selected.Q.to_numpy(dtype=float)
    d = selected.d.to_numpy(dtype=float)
    y = selected.tap_time_len.to_numpy(dtype=float)
    if not all(np.isfinite(x).all() for x in (q, d, y)):
        raise ContractError("nonfinite scalar-certificate data")
    normalized_labels = selected[["sample_id", "tap_time_len"]].copy()
    normalized_labels["tap_time_len"] = normalized_labels.tap_time_len.astype(float)
    if len(selected) < minimum:
        expected = {
            "lambda": 0.0, "rows": int(len(selected)), "fallback": "INSUFFICIENT_OOF_ROWS",
            "fit_performed": False, "ids_sha256": stable_digest(selected.sample_id.astype(str).tolist()),
        }
    elif not np.any(d != 0):
        expected = {
            "lambda": 0.0, "rows": int(len(selected)), "fallback": "ZERO_DIRECTION",
            "fit_performed": False, "ids_sha256": stable_digest(selected.sample_id.astype(str).tolist()),
            "labels_sha256": stable_digest(normalized_labels.to_dict("records")),
            "inputs_sha256": stable_digest(selected[["sample_id", "Q", "d"]].to_dict("records")),
            "objective": float(np.abs(y - q).sum()), "smallest_minimizer_verified": True,
        }
    else:
        nonzero = d != 0
        ratios = (y[nonzero] - q[nonzero]) / d[nonzero]
        weights = np.abs(d[nonzero])
        order = np.argsort(ratios, kind="mergesort")
        position = np.searchsorted(np.cumsum(weights[order]), 0.5 * weights.sum(), side="left")
        optimum = float(np.clip(ratios[order[position]], 0.0, 1.0))
        expected = {
            "lambda": optimum, "rows": int(len(selected)), "fallback": None, "fit_performed": True,
            "ids_sha256": stable_digest(selected.sample_id.astype(str).tolist()),
            "labels_sha256": stable_digest(normalized_labels.to_dict("records")),
            "inputs_sha256": stable_digest(selected[["sample_id", "Q", "d"]].to_dict("records")),
            "objective": float(np.abs(y - q - optimum * d).sum()), "smallest_minimizer_verified": True,
        }
    if dict(certificate) != expected:
        raise ContractError("stored lambda optimality certificate differs")
    return True


class LambdaBudget:
    """Persist at most one per-spout scalar derivation for each outer origin."""
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def derive(self, outer_month: int, spout: str, selected: pd.DataFrame) -> dict:
        if outer_month not in range(6, 12) or str(spout) not in ("1", "2"):
            raise ContractError("only the 12 registered origin/spout lambda slots are allowed")
        key = f"outer-{outer_month}-spout-{spout}"
        result_path = self.root / f"{key}.json"
        selected_identity = stable_digest(selected.to_dict("records"))
        if result_path.exists():
            import json
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if saved["selected_sha256"] != selected_identity:
                raise ContractError("lambda slot already derived from different OOF rows")
            if saved["certificate"] != lambda_certificate(selected, saved["lambda"]):
                raise ContractError("stored lambda certificate differs")
            return saved
        completed = [p for p in self.root.glob("outer-*-spout-*.json") if not p.name.endswith(".intent.json")]
        if len(completed) >= 12:
            raise ContractError("lambda parameter-slot budget exceeded")
        intent_path = self.root / f"{key}.intent.json"
        eligible = len(selected) >= MINIMUM_OOF_ROWS and np.any(selected.d.to_numpy(dtype=float) != 0)
        if eligible:
            atomic_write_json(intent_path, {
                "outer_month": outer_month,
                "spout_no": str(spout),
                "selected_sha256": selected_identity,
                "primitive": "constrained_LAD_smallest_minimizer",
            })
        certificate = fit_lambda(selected)
        saved = {
            "outer_month": outer_month,
            "spout_no": str(spout),
            "lambda": certificate["lambda"],
            "selected_sha256": selected_identity,
            "certificate": certificate,
        }
        atomic_write_json(result_path, saved)
        return saved

    def counts(self) -> dict:
        results = [p for p in self.root.glob("outer-*-spout-*.json") if not p.name.endswith(".intent.json")]
        intents = list(self.root.glob("outer-*-spout-*.intent.json"))
        return {
            "parameter_slots_completed": len(results),
            "fit_attempts": len(intents),
            "zero_fallbacks": sum(
                1 for p in results
                if __import__("json").loads(p.read_text(encoding="utf-8"))["certificate"]["fit_performed"] is False
            ),
        }


def apply_shrink(
    v10: pd.DataFrame,
    bank: pd.DataFrame,
    lambdas: Mapping[str, float],
) -> pd.DataFrame:
    """Copy V10 iron exactly and replace time with certified V22 time."""
    required = {"sample_id", *PREDICTION_COLUMNS}
    if required - set(v10) or v10.empty or v10.sample_id.isna().any() or v10.sample_id.duplicated().any():
        raise ContractError("complete unique V10 predictions required")
    if "label_available_at" in bank:
        verify_h2_bank(bank)
    else:
        verify_prediction_directions(bank)
    if set(v10.sample_id.astype(str)) != set(bank.sample_id.astype(str)):
        raise ContractError("V10 and V22 bank sample IDs differ")
    b = bank.assign(sample_id=bank.sample_id.astype(str)).set_index("sample_id").loc[v10.sample_id.astype(str)]
    base = v10[PREDICTION_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(base).all() or (base < 0).any():
        raise ContractError("finite nonnegative V10 predictions required")
    q = b.Q.to_numpy(dtype=float)
    if not np.array_equal(base[:, 1], q):
        raise ContractError("V10 time must be the exact QRF median Q")
    missing = {str(s) for s in b.spout_no if str(s) in ("1", "2") and str(s) not in lambdas}
    if missing:
        raise ContractError(f"missing certified lambda for known spouts: {sorted(missing)}")
    coefficients = np.asarray([float(lambdas.get(str(s), 0.0)) for s in b.spout_no], dtype=float)
    if not np.isfinite(coefficients).all() or ((coefficients < 0) | (coefficients > 1)).any():
        raise ContractError("all spout lambdas must be in [0,1]")
    time = q + coefficients * b.d.to_numpy(dtype=float)
    if not np.isfinite(time).all() or (time < 0).any():
        raise ContractError("finite nonnegative V22 time required")
    known = b.M.notna().to_numpy()
    lo = np.minimum(q[known], b.M.to_numpy(dtype=float)[known]) - 1e-12
    hi = np.maximum(q[known], b.M.to_numpy(dtype=float)[known]) + 1e-12
    if ((time[known] < lo) | (time[known] > hi)).any():
        raise ContractError("V22 prediction is outside the Q/M convex hull")
    result = v10[["sample_id"]].copy()
    result["pred_tap_iron"] = v10.pred_tap_iron.to_numpy(copy=True)
    result["pred_tap_time_len"] = time
    return result


def serialize_six_decimals(prediction: pd.DataFrame) -> bytes:
    if list(prediction.columns) != ["sample_id", *PREDICTION_COLUMNS]:
        raise ContractError("official prediction column order required")
    values = prediction[PREDICTION_COLUMNS].to_numpy(dtype=float)
    if prediction.sample_id.isna().any() or prediction.sample_id.duplicated().any() or not np.isfinite(values).all() or (values < 0).any():
        raise ContractError("valid unique nonnegative predictions required")
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(["sample_id", *PREDICTION_COLUMNS])
    for row in prediction.itertuples(index=False):
        writer.writerow([str(row.sample_id), f"{row.pred_tap_iron:.6f}", f"{row.pred_tap_time_len:.6f}"])
    return handle.getvalue().encode("utf-8")


def replay_v21(
    v10_six_decimal: pd.DataFrame,
    queries: pd.DataFrame,
    history: pd.DataFrame,
    effective_neighbors: Sequence[float],
    *,
    cutoff,
) -> tuple[pd.DataFrame, dict]:
    """Replay the original moving-query-window V21 rule without redefining it."""
    if list(v10_six_decimal.columns) != ["sample_id", *PREDICTION_COLUMNS]:
        raise ContractError("V21 replay requires original V10 three-column input")
    required_query = {"sample_id", "spout_no", "reference_time"}
    required_history = {"spout_no", "reference_time", "tap_end_time", "tap_time_len"}
    if required_query - set(queries) or required_history - set(history):
        raise ContractError("V21 replay metadata/history columns missing")
    if v10_six_decimal.sample_id.astype(str).tolist() != queries.sample_id.astype(str).tolist():
        raise ContractError("V10/query order differs for V21 replay")
    neff = np.asarray(effective_neighbors, dtype=float)
    if neff.shape != (len(queries),) or not np.isfinite(neff).all():
        raise ContractError("aligned V21 effective-neighbor diagnostics required")
    c = _aware(cutoff, "V21 cutoff")
    q = queries.copy()
    q["reference_time"] = _times(q.reference_time, "V21 query reference_time")
    h = history.copy()
    h["reference_time"] = _times(h.reference_time, "V21 history reference_time")
    h["tap_end_time"] = _times(h.tap_end_time, "V21 history tap_end_time")
    h["spout_no"] = h.spout_no.astype(str)
    medians, recent_counts, fallback = [], [], []
    for row in q.itertuples(index=False):
        same = h.spout_no == str(row.spout_no)
        recent = h.loc[
            same & (h.reference_time < row.reference_time)
            & (h.reference_time >= row.reference_time - pd.Timedelta(days=60))
            & (h.tap_end_time <= c)
        ]
        recent_counts.append(int(len(recent)))
        used_fallback = recent.empty
        selected = h.loc[same & (h.tap_end_time <= c)] if used_fallback else recent
        if selected.empty:
            raise ContractError("V21 replay has no same-spout completed history")
        medians.append(float(selected.tap_time_len.median()))
        fallback.append(used_fallback)
    current = v10_six_decimal.pred_tap_time_len.to_numpy(dtype=float)
    gate = (q.spout_no.astype(str).to_numpy() == "1") & (neff < 500.0)
    adjusted = current.copy()
    m = np.asarray(medians)
    adjusted[gate] = 0.75 * current[gate] + 0.25 * m[gate]
    result = v10_six_decimal.copy()
    result["pred_tap_time_len"] = adjusted
    return result, {
        "changed_rows": int(gate.sum()),
        "changed_fraction": float(gate.mean()),
        "recent_window_eligible_counts": recent_counts,
        "fallback_rows": int(np.count_nonzero(fallback)),
        "fallback_fraction": float(np.mean(fallback)),
        "rule": {"spout": "1", "effective_neighbors_lt": 500.0, "shrink": 0.25},
    }


def acceptance(metrics: Mapping, summary: Mapping, *, engineering: bool, iron_exact: bool) -> dict:
    """Apply the preregistered V22 gates to same-precision metric mappings."""
    cells: dict[int, np.ndarray] = {}
    for horizon in range(1, 5):
        cells[horizon] = np.asarray([
            value["candidates"][CANDIDATE]["overall"]["loss"]
            - value["candidates"][REFERENCE]["overall"]["loss"]
            for key, value in metrics.items()
            if key.startswith("O") and value["horizon"] == horizon
        ], dtype=float)
    if [len(cells[h]) for h in range(1, 5)] != [6, 5, 4, 3]:
        raise ContractError("complete 18-cell origin/horizon grid required")
    h2_units = [
        (key, value) for key, value in metrics.items()
        if key.startswith("O") and value["horizon"] == 2
    ]
    recent = [
        value["candidates"][CANDIDATE]["overall"]["loss"]
        - value["candidates"][REFERENCE]["overall"]["loss"]
        for key, value in h2_units
        if int(key[5:7]) + 1 in (9, 10, 11)
    ]
    v21_delta = float(np.mean([
        value["candidates"][CANDIDATE]["overall"]["loss"]
        - value["candidates"][REPLAY_CONTROL]["overall"]["loss"]
        for _, value in h2_units
    ]))
    delta_j = float(summary[CANDIDATE]["J"] - summary[REFERENCE]["J"])
    dev = {
        name: float(metrics[name]["candidates"][CANDIDATE]["overall"]["loss"]
                    - metrics[name]["candidates"][REFERENCE]["overall"]["loss"])
        for name in ("DEV_LONG", "DEV_SHORT")
    }
    gates = {
        "engineering_causal_source_budget": bool(engineering),
        "iron_exact": bool(iron_exact),
        "H2_mean_E": float(cells[2].mean()) <= -0.0005,
        "H2_improved_origins": int((cells[2] < 0).sum()) >= 4,
        "recent_H2_improved": int(sum(v < 0 for v in recent)) >= 2,
        "H2_single_origin": float(cells[2].max()) <= 0.0010,
        "H2_not_worse_than_V21_replay": v21_delta <= 0.0,
        "J_guardrail": delta_j <= 0.0002,
        "H1_guardrail": float(cells[1].mean()) <= 0.0005,
        "H3_guardrail": float(cells[3].mean()) <= 0.0005,
        "H4_guardrail": float(cells[4].mean()) <= 0.0005,
        "DEV_LONG_guardrail": dev["DEV_LONG"] <= 0.0007,
        "DEV_SHORT_guardrail": dev["DEV_SHORT"] <= 0.0007,
    }
    passed = all(gates.values())
    return {
        "status": "DEV_ACCEPTED_PENDING_OFFICIAL_IDENTITY" if passed else "FAIL_RETAIN_V21_AND_V10",
        "gates": gates,
        "H2_delta_E": float(cells[2].mean()),
        "H2_improved_origins": int((cells[2] < 0).sum()),
        "recent_H2_improved": int(sum(v < 0 for v in recent)),
        "H2_single_origin_max_delta_E": float(cells[2].max()),
        "H2_minus_V21_REPLAY": v21_delta,
        "delta_J": delta_j,
        "horizon_delta_E": {str(h): float(v.mean()) for h, v in cells.items()},
        "DEV_delta_E": dev,
        "G0": "PASS" if engineering and iron_exact else "FAIL",
        "G1": "PASS" if passed else "FAIL",
        "ready_for_platform": False,
        "platform_uploads": 0,
    }
