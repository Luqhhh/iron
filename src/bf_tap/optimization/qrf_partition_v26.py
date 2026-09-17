"""Pure contracts for the two preregistered v0.26 QRF partition tests."""
from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..exceptions import ContractError


CANDIDATE_A = "V26A_QRF_ABSOLUTE_SPLIT_TIME"
CANDIDATE_B = "V26B_EXTRA_RANDOM_SPLIT_TIME"
PARENT = "V21_REPLAY"
TARGETS = ("tap_iron", "tap_time_len")
PREDICTION_COLUMNS = ("pred_tap_iron", "pred_tap_time_len")
PRIMARY_AGGREGATION = "macro_origin_mean_wmape"
POOLED_AGGREGATION = "exposure_pooled_wmape"
TOLERANCE = 1e-12


def roundtrip_six(values: Sequence[float]) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or not np.isfinite(raw).all() or (raw < 0).any():
        raise ContractError("finite nonnegative one-dimensional predictions required")
    return np.asarray([float(f"{value:.6f}") for value in raw], dtype=np.float64)


def isolate_time(parent: pd.DataFrame, time: Sequence[float]) -> pd.DataFrame:
    expected = ["sample_id", *PREDICTION_COLUMNS]
    if list(parent) != expected or parent.sample_id.isna().any() or parent.sample_id.astype(str).duplicated().any():
        raise ContractError("valid ordered three-column V21 parent required")
    result = parent.copy()
    result["pred_tap_time_len"] = roundtrip_six(time)
    if not np.array_equal(result.pred_tap_iron.to_numpy(), parent.pred_tap_iron.to_numpy()):
        raise ContractError("time-only candidate changed V21 iron")
    return result


def submission_bytes_preserving_iron(parent_strings: pd.DataFrame, sample_ids: Sequence[str], time: Sequence[float]) -> bytes:
    """Serialize six-decimal time while copying the parent's iron text verbatim."""
    expected = ["sample_id", *PREDICTION_COLUMNS]
    if list(parent_strings) != expected or parent_strings.isna().any().any():
        raise ContractError("complete string-valued V21 result required")
    ids = [str(value) for value in sample_ids]
    if parent_strings.sample_id.astype(str).tolist() != ids or len(set(ids)) != len(ids):
        raise ContractError("candidate/V21 submission IDs or order differ")
    iron = parent_strings.pred_tap_iron.astype(str).tolist()
    six_decimal = re.compile(r"^(?:0|[1-9][0-9]*)\.[0-9]{6}$")
    if any(six_decimal.fullmatch(value) is None for value in iron):
        raise ContractError("V21 iron strings must already be finite nonnegative six-decimal values")
    q = roundtrip_six(time)
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(expected)
    for sample_id, iron_text, duration in zip(ids, iron, q, strict=True):
        writer.writerow([sample_id, iron_text, f"{duration:.6f}"])
    return handle.getvalue().encode("utf-8")


def _validate_scorecard(scorecard: pd.DataFrame) -> pd.DataFrame:
    required = {
        "algorithm", "unit", "cutoff", "horizon", "target", "n", "target_sum",
        "absolute_error_sum", "wmape", "E",
    }
    if required - set(scorecard) or scorecard.empty:
        raise ContractError("complete canonical cell scorecard required")
    value = scorecard.copy()
    if value.duplicated(["algorithm", "unit", "target"]).any():
        raise ContractError("duplicate algorithm/unit/target scorecard row")
    if set(value.target) != set(TARGETS):
        raise ContractError("canonical scorecard must contain exactly both targets")
    numeric = value[["n", "target_sum", "absolute_error_sum", "wmape", "E"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (value.n <= 0).any() or (value.target_sum <= 0).any() or (value.absolute_error_sum < 0).any():
        raise ContractError("invalid canonical scorecard metric")
    recomputed = value.absolute_error_sum.to_numpy(dtype=float) / value.target_sum.to_numpy(dtype=float)
    if not np.allclose(recomputed, value.wmape, rtol=0, atol=TOLERANCE):
        raise ContractError("scorecard WMAPE numerator/denominator mismatch")
    for (_, _), part in value.groupby(["algorithm", "unit"], sort=False):
        if len(part) != 2 or set(part.target) != set(TARGETS):
            raise ContractError("each cell requires exactly two target rows")
        expected_e = float(part.set_index("target").loc[list(TARGETS), "wmape"].mean())
        if not np.allclose(part.E.to_numpy(dtype=float), expected_e, rtol=0, atol=TOLERANCE):
            raise ContractError("cell E is not the equal-target WMAPE mean")
        if part.n.nunique() != 1 or part.cutoff.nunique(dropna=False) != 1 or part.horizon.nunique(dropna=False) != 1:
            raise ContractError("cell target rows disagree on identity")
    return value


def _wide_cell_rows(scorecard: pd.DataFrame) -> pd.DataFrame:
    value = _validate_scorecard(scorecard)
    rows: list[dict] = []
    for (algorithm, unit), part in value.groupby(["algorithm", "unit"], sort=False):
        by_target = part.set_index("target")
        rows.append({
            "algorithm": algorithm,
            "unit": unit,
            "cutoff": part.cutoff.iloc[0],
            "horizon": part.horizon.iloc[0],
            "wmape_iron": float(by_target.loc["tap_iron", "wmape"]),
            "wmape_time": float(by_target.loc["tap_time_len", "wmape"]),
            "E": float(part.E.iloc[0]),
        })
    return pd.DataFrame(rows)


def macro_origin_summary(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Aggregate equal-origin target WMAPEs; DEV never enters J."""
    cells = _wide_cell_rows(scorecard)
    rows: list[dict] = []
    for algorithm, part in cells.groupby("algorithm", sort=False):
        origins = part.loc[part.unit.astype(str).str.startswith("O")].copy()
        for row in origins.itertuples(index=False):
            rows.append({
                "algorithm": algorithm, "scope": row.unit, "scope_type": "CELL",
                "aggregation": PRIMARY_AGGREGATION, "n_origins": 1,
                "wmape_iron": row.wmape_iron, "wmape_time": row.wmape_time, "E": row.E,
            })
        horizon_rows: list[dict] = []
        for horizon in range(1, 5):
            selected = origins.loc[origins.horizon == horizon]
            if selected.empty:
                raise ContractError(f"missing registered H{horizon} cells")
            item = {
                "algorithm": algorithm, "scope": f"H{horizon}", "scope_type": "HORIZON_MEAN",
                "aggregation": PRIMARY_AGGREGATION, "n_origins": int(len(selected)),
                "wmape_iron": float(selected.wmape_iron.mean()),
                "wmape_time": float(selected.wmape_time.mean()),
                "E": float(selected.E.mean()),
            }
            if abs(item["E"] - .5 * (item["wmape_iron"] + item["wmape_time"])) > TOLERANCE:
                raise ContractError("macro target WMAPEs and E use different origin weighting")
            rows.append(item)
            horizon_rows.append(item)
        rows.append({
            "algorithm": algorithm, "scope": "J", "scope_type": "GRID",
            "aggregation": PRIMARY_AGGREGATION, "n_origins": 4,
            "wmape_iron": float(np.mean([row["wmape_iron"] for row in horizon_rows])),
            "wmape_time": float(np.mean([row["wmape_time"] for row in horizon_rows])),
            "E": float(np.mean([row["E"] for row in horizon_rows])),
        })
        for unit in ("DEV_LONG", "DEV_SHORT"):
            selected = part.loc[part.unit == unit]
            if len(selected) != 1:
                raise ContractError(f"exactly one {unit} cell required")
            row = selected.iloc[0]
            rows.append({
                "algorithm": algorithm, "scope": unit, "scope_type": "DEVELOPMENT",
                "aggregation": PRIMARY_AGGREGATION, "n_origins": 1,
                "wmape_iron": float(row.wmape_iron), "wmape_time": float(row.wmape_time), "E": float(row.E),
            })
    return pd.DataFrame(rows)


def exposure_pooled_summary(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Pool error exposure inside each horizon; this is diagnostic only."""
    value = _validate_scorecard(scorecard)
    rows: list[dict] = []
    for algorithm, part in value.groupby("algorithm", sort=False):
        origin_rows = part.loc[part.unit.astype(str).str.startswith("O")]
        for horizon in range(1, 5):
            selected = origin_rows.loc[origin_rows.horizon == horizon]
            metrics = {}
            for target in TARGETS:
                target_rows = selected.loc[selected.target == target]
                if target_rows.empty:
                    raise ContractError(f"missing pooled H{horizon}/{target}")
                metrics[target] = float(target_rows.absolute_error_sum.sum() / target_rows.target_sum.sum())
            rows.append({
                "algorithm": algorithm, "scope": f"H{horizon}", "scope_type": "HORIZON_POOLED_DIAGNOSTIC",
                "aggregation": POOLED_AGGREGATION,
                "n_origins": int(selected.unit.nunique()),
                "wmape_iron": metrics["tap_iron"], "wmape_time": metrics["tap_time_len"],
                "E": .5 * (metrics["tap_iron"] + metrics["tap_time_len"]),
            })
        for unit in ("DEV_LONG", "DEV_SHORT"):
            selected = part.loc[part.unit == unit]
            metrics = {target: float(selected.loc[selected.target == target, "wmape"].iloc[0]) for target in TARGETS}
            rows.append({
                "algorithm": algorithm, "scope": unit, "scope_type": "DEVELOPMENT_POOLED_DIAGNOSTIC",
                "aggregation": POOLED_AGGREGATION, "n_origins": 1,
                "wmape_iron": metrics["tap_iron"], "wmape_time": metrics["tap_time_len"],
                "E": .5 * (metrics["tap_iron"] + metrics["tap_time_len"]),
            })
    return pd.DataFrame(rows)


def verify_isolated_deltas(
    scorecard: pd.DataFrame,
    summary: pd.DataFrame,
    changed_targets: Mapping[str, str],
    *,
    parent: str = PARENT,
) -> dict:
    """Prove target isolation and ΔE=0.5ΔWMAPE for cells and macro summaries."""
    value = _validate_scorecard(scorecard)
    if set(changed_targets.values()) - set(TARGETS):
        raise ContractError("unknown isolated target")
    checked_cells = 0
    maximum_residual = 0.0
    parent_cells = value.loc[value.algorithm == parent].set_index(["unit", "target"])
    for candidate, changed in changed_targets.items():
        candidate_cells = value.loc[value.algorithm == candidate].set_index(["unit", "target"])
        if set(candidate_cells.index) != set(parent_cells.index):
            raise ContractError("candidate/parent cell grid differs")
        unchanged = "tap_time_len" if changed == "tap_iron" else "tap_iron"
        for unit in sorted(set(value.loc[value.algorithm == parent, "unit"])):
            for target in TARGETS:
                left = candidate_cells.loc[(unit, target)]
                right = parent_cells.loc[(unit, target)]
                if int(left.n) != int(right.n) or abs(float(left.target_sum) - float(right.target_sum)) > TOLERANCE:
                    raise ContractError("candidate/parent cell n or target denominator differs")
            if abs(float(candidate_cells.loc[(unit, unchanged)].wmape) - float(parent_cells.loc[(unit, unchanged)].wmape)) > TOLERANCE:
                raise ContractError("isolated candidate changed the other target")
            delta_changed = float(candidate_cells.loc[(unit, changed)].wmape - parent_cells.loc[(unit, changed)].wmape)
            delta_e = float(candidate_cells.loc[(unit, changed)].E - parent_cells.loc[(unit, changed)].E)
            residual = abs(delta_e - .5 * delta_changed)
            maximum_residual = max(maximum_residual, residual)
            if residual > TOLERANCE:
                raise ContractError("cell isolated-target delta identity failed")
            checked_cells += 1
    primary = summary.loc[summary.aggregation == PRIMARY_AGGREGATION].set_index(["algorithm", "scope"])
    checked_summaries = 0
    for candidate, changed in changed_targets.items():
        field = "wmape_iron" if changed == "tap_iron" else "wmape_time"
        unchanged_field = "wmape_time" if changed == "tap_iron" else "wmape_iron"
        scopes = set(summary.loc[summary.algorithm == candidate, "scope"])
        for scope in scopes:
            left = primary.loc[(candidate, scope)]
            right = primary.loc[(parent, scope)]
            if abs(float(left[unchanged_field]) - float(right[unchanged_field])) > TOLERANCE:
                raise ContractError("macro isolated candidate changed the other target")
            residual = abs(float(left.E - right.E) - .5 * float(left[field] - right[field]))
            maximum_residual = max(maximum_residual, residual)
            if residual > TOLERANCE:
                raise ContractError("macro isolated-target delta identity failed")
            checked_summaries += 1
    return {
        "checked_cells": checked_cells,
        "checked_macro_summaries": checked_summaries,
        "maximum_absolute_delta_identity_residual": maximum_residual,
        "tolerance": TOLERANCE,
    }
