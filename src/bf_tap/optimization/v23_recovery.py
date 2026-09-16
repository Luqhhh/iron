"""Zero-fit recovery, canonical scoring, and stage composition for v0.23."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..artifacts import file_sha256, stable_digest
from ..exceptions import ContractError
from .qrf_support_shrink import (
    CANDIDATE as V22,
    PREDICTION_COLUMNS,
    apply_shrink,
    build_prediction_directions,
    replay_v21,
    serialize_six_decimals,
)


V1 = "V1"
V10 = "V10"
V21 = "V21_REPLAY"
M_ONLY = "D_M_CUTOFF_ONLY"
FIXED_ALGORITHMS = (V1, V10, V21, V22)
TARGETS = ("tap_iron", "tap_time_len")
PREDICTIONS = ("pred_tap_iron", "pred_tap_time_len")
EXPECTED_HORIZON_CELLS = {1: 6, 2: 5, 3: 4, 4: 3}


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_explicit_offsets(values: pd.Series, name: str) -> None:
    """Reject naive timestamp text before a historical V21 replay."""
    text = values.astype(str)
    explicit = text.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True, na=False)
    if not bool(explicit.all()):
        raise ContractError(f"{name} must preserve an explicit timezone offset")


def historical_v21_replay_audit(
    history_path: Path,
    direction_root: Path,
    prediction_root: Path,
    all_errors: pd.DataFrame,
) -> dict:
    """Re-run all six historical V21 origins and bind them to scoring evidence."""
    history = pd.read_csv(
        history_path,
        dtype={"sample_id": str, "spout_no": str},
        float_precision="round_trip",
    )
    required_history = {"sample_id", "spout_no", "reference_time", "tap_end_time", "tap_time_len"}
    if required_history - set(history):
        raise ContractError("historical V21 replay history schema differs")
    _require_explicit_offsets(history.reference_time, "historical history reference_time")
    _require_explicit_offsets(history.tap_end_time, "historical history tap_end_time")
    rows = []
    for month in range(6, 12):
        origin = f"O2024{month:02d}"
        directions_path = direction_root / f"outer-{month}.csv"
        v10_path = prediction_root / str(month) / "V10.csv"
        v21_path = prediction_root / str(month) / "V21_REPLAY.csv"
        directions = pd.read_csv(
            directions_path,
            dtype={"sample_id": str, "spout_no": str},
            float_precision="round_trip",
        )
        required_direction = {
            "sample_id", "spout_no", "reference_time", "model_cutoff", "effective_neighbors",
        }
        if required_direction - set(directions) or directions.sample_id.duplicated().any():
            raise ContractError(f"historical V21 directions differ for {origin}")
        _require_explicit_offsets(directions.reference_time, f"{origin} reference_time")
        _require_explicit_offsets(directions.model_cutoff, f"{origin} model_cutoff")
        cutoffs = directions.model_cutoff.unique()
        if len(cutoffs) != 1:
            raise ContractError(f"historical V21 cutoff is not fixed for {origin}")
        v10 = pd.read_csv(v10_path, dtype={"sample_id": str}, float_precision="round_trip")
        replayed, diagnostics = replay_v21(
            v10,
            directions[["sample_id", "spout_no", "reference_time"]],
            history,
            directions.effective_neighbors,
            cutoff=pd.Timestamp(cutoffs[0]),
        )
        payload = serialize_six_decimals(replayed)
        if payload != v21_path.read_bytes():
            raise ContractError(f"historical V21 replay differs byte-for-byte for {origin}")

        evidence = all_errors.loc[
            (all_errors.candidate == V21) & (all_errors.origin == origin),
            ["sample_id", *PREDICTIONS],
        ]
        if evidence.empty:
            raise ContractError(f"canonical V21 evidence is absent for {origin}")
        conflicts = evidence.groupby("sample_id", sort=False)[list(PREDICTIONS)].nunique(dropna=False)
        if (conflicts > 1).any().any():
            raise ContractError(f"canonical V21 exposures conflict for {origin}")
        unique_evidence = evidence.drop_duplicates("sample_id").set_index("sample_id")
        expected = replayed.set_index("sample_id")
        if set(unique_evidence.index) != set(expected.index):
            raise ContractError(f"canonical V21 IDs differ from replay for {origin}")
        unique_evidence = unique_evidence.loc[expected.index]
        if any(
            not np.array_equal(
                unique_evidence[column].to_numpy(dtype=float),
                np.asarray([six_decimal_float(value) for value in expected[column]], dtype=float),
            )
            for column in PREDICTIONS
        ):
            raise ContractError(f"canonical V21 predictions differ from replay for {origin}")
        rows.append({
            "origin": origin,
            "rows": int(len(replayed)),
            "scoring_exposures": int(len(evidence)),
            "cutoff": str(cutoffs[0]),
            "v10_sha256": file_sha256(v10_path),
            "v21_sha256": file_sha256(v21_path),
            "directions_sha256": file_sha256(directions_path),
            "byte_exact": True,
            "canonical_evidence_exact": True,
            "changed_rows": diagnostics["changed_rows"],
            "fallback_rows": diagnostics["fallback_rows"],
        })
    return {
        "kind": "V23_HISTORICAL_V21_REPLAY_AUDIT_v1",
        "status": "PASS",
        "history": {"path": str(history_path.resolve()), "sha256": file_sha256(history_path)},
        "timezone_contract": "explicit_offset_required_before_replay",
        "origins": rows,
        "all_six_replays_byte_exact": True,
        "all_scoring_predictions_exact": True,
        "fit_attempts": 0,
    }


def six_decimal_float(value: float) -> float:
    if not np.isfinite(value) or value < 0:
        raise ContractError("predictions must be finite and nonnegative before serialization")
    return float(f"{value:.6f}")


def canonical_errors(source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Align four algorithms and recompute errors after six-decimal prediction text."""
    required = {
        "sample_id", "reference_time", "spout_no", *TARGETS, *PREDICTIONS,
        "candidate", "origin", "horizon", "unit", "week",
    }
    if required - set(source):
        raise ContractError("v0.22 row evidence lacks canonical scoring columns")
    value = source.loc[source.candidate.isin(FIXED_ALGORITHMS)].copy()
    value["sample_id"] = value.sample_id.astype(str)
    value["spout_no"] = value.spout_no.astype(str)
    if set(value.candidate.unique()) != set(FIXED_ALGORITHMS):
        raise ContractError("canonical evidence does not contain all four fixed algorithms")
    identity_columns = ["unit", "sample_id", "reference_time", "spout_no", *TARGETS, "origin", "horizon", "week"]
    reference = (
        value.loc[value.candidate == V1, identity_columns]
        .sort_values(["unit", "sample_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if reference.duplicated(["unit", "sample_id"]).any():
        raise ContractError("duplicate unit/sample exposure in canonical reference")
    for candidate in FIXED_ALGORITHMS:
        part = (
            value.loc[value.candidate == candidate, identity_columns]
            .sort_values(["unit", "sample_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        if not part.equals(reference):
            raise ContractError(f"{candidate} row, target, denominator, or metadata identity differs")
    for column in PREDICTIONS:
        value[column] = value[column].map(six_decimal_float)
    for target, prediction in zip(TARGETS, PREDICTIONS, strict=True):
        value[f"error_{target}"] = value[prediction] - value[target]
        value[f"abs_error_{target}"] = value[f"error_{target}"].abs()
    unit_horizons = value[["unit", "horizon"]].drop_duplicates()
    counts = unit_horizons.dropna().groupby("horizon").size().to_dict()
    normalized_counts = {int(key): int(number) for key, number in counts.items()}
    if normalized_counts != EXPECTED_HORIZON_CELLS:
        raise ContractError(f"canonical grid differs: {normalized_counts}")
    dev = sorted(unit_horizons.loc[unit_horizons.horizon.isna(), "unit"].tolist())
    if dev != ["DEV_LONG", "DEV_SHORT"]:
        raise ContractError("canonical DEV units differ")
    identity = {
        "status": "PASS",
        "algorithms": list(FIXED_ALGORITHMS),
        "rows_per_algorithm": int(len(reference)),
        "exposures_per_algorithm": int(len(reference)),
        "unique_sample_ids": int(reference.sample_id.nunique()),
        "repeated_exposures": int(len(reference) - reference.sample_id.nunique()),
        "unit_count": int(reference.unit.nunique()),
        "grid_cells": int(reference.loc[reference.horizon.notna(), "unit"].nunique()),
        "horizon_cells": {str(key): value for key, value in normalized_counts.items()},
        "development_units": dev,
        "identity_sha256": stable_digest(reference.astype({"sample_id": str, "spout_no": str}).to_dict("records")),
        "predictions_serialized_to_six_decimals": True,
        "labels_rounded": False,
    }
    return value.sort_values(["candidate", "unit", "sample_id"], kind="mergesort").reset_index(drop=True), identity


def scorecard(errors: pd.DataFrame, algorithms: Sequence[str] = FIXED_ALGORITHMS) -> pd.DataFrame:
    rows: list[dict] = []
    for (candidate, unit), part in errors.loc[errors.candidate.isin(algorithms)].groupby(["candidate", "unit"], sort=True):
        horizons = part.horizon.dropna().unique()
        horizon = None if not len(horizons) else int(horizons[0])
        origins = part.origin.unique()
        if len(origins) != 1:
            raise ContractError("one scoring unit must have exactly one origin")
        target_metrics = {}
        for target in TARGETS:
            denominator = float(part[target].sum())
            numerator = float(part[f"abs_error_{target}"].sum())
            if denominator <= 0:
                raise ContractError("canonical target denominator must be positive")
            target_metrics[target] = {
                "n": int(len(part)),
                "target_sum": denominator,
                "absolute_error_sum": numerator,
                "signed_error_sum": float(part[f"error_{target}"].sum()),
                "signed_bias": float(part[f"error_{target}"].mean()),
                "wmape": numerator / denominator,
            }
        e = float(np.mean([target_metrics[target]["wmape"] for target in TARGETS]))
        for target in TARGETS:
            rows.append({
                "algorithm": candidate,
                "unit": unit,
                "cutoff": str(origins[0]),
                "horizon": horizon,
                "target": target,
                **target_metrics[target],
                "E": e,
            })
    return pd.DataFrame(rows).sort_values(["algorithm", "unit", "target"], kind="mergesort").reset_index(drop=True)


def score_summary(card: pd.DataFrame) -> pd.DataFrame:
    """Return cell, horizon, J and DEV E using the registered aggregation."""
    cells = card.drop_duplicates(["algorithm", "unit"])[["algorithm", "unit", "cutoff", "horizon", "E"]]
    rows: list[dict] = []
    for row in cells.itertuples(index=False):
        rows.append({"algorithm": row.algorithm, "scope": row.unit, "scope_type": "CELL" if pd.notna(row.horizon) else "DEV", "E": float(row.E)})
    for candidate, part in cells.loc[cells.horizon.notna()].groupby("algorithm", sort=True):
        horizon_values = {}
        for horizon, group in part.groupby("horizon", sort=True):
            h = int(horizon)
            if len(group) != EXPECTED_HORIZON_CELLS[h]:
                raise ContractError("incomplete horizon during canonical aggregation")
            value = float(group.E.mean())
            horizon_values[h] = value
            rows.append({"algorithm": candidate, "scope": f"H{h}", "scope_type": "HORIZON", "E": value})
        rows.append({"algorithm": candidate, "scope": "J", "scope_type": "GRID", "E": float(np.mean(list(horizon_values.values())))})
    return pd.DataFrame(rows).sort_values(["scope_type", "scope", "algorithm"], kind="mergesort").reset_index(drop=True)


def reference_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot(index=["scope", "scope_type"], columns="algorithm", values="E")
    if set(FIXED_ALGORITHMS) - set(pivot):
        raise ContractError("reference delta table lacks a fixed algorithm")
    rows = []
    for (scope, scope_type), values in pivot.iterrows():
        for candidate in FIXED_ALGORITHMS:
            rows.append({
                "scope": scope,
                "scope_type": scope_type,
                "algorithm": candidate,
                "E": float(values[candidate]),
                "V1_E": float(values[V1]),
                "V10_E": float(values[V10]),
                "delta_vs_V1": float(values[candidate] - values[V1]),
                "delta_vs_V10": float(values[candidate] - values[V10]),
            })
    return pd.DataFrame(rows).sort_values(["scope_type", "scope", "algorithm"], kind="mergesort").reset_index(drop=True)


def _lambda_by_origin(lambda_root: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for month in range(6, 12):
        value = read_json(lambda_root / str(month) / "summary.json")
        if value.get("status") != "PASS" or int(value.get("outer_month", -1)) != month:
            raise ContractError("saved v0.22 lambda summary differs")
        result[f"O2024{month:02d}"] = {
            str(spout): float(row["lambda"]) for spout, row in value["spouts"].items()
        }
    return result


def load_directions(direction_root: Path) -> pd.DataFrame:
    parts = []
    for month in range(6, 12):
        part = pd.read_csv(
            direction_root / f"outer-{month}.csv",
            dtype={"sample_id": str, "spout_no": str},
            float_precision="round_trip",
        )
        part.insert(0, "origin", f"O2024{month:02d}")
        if part.duplicated("sample_id").any():
            raise ContractError("saved v0.22 directions contain duplicate IDs")
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def median_only_errors(canonical: pd.DataFrame, directions: pd.DataFrame) -> pd.DataFrame:
    base = canonical.loc[canonical.candidate == V10].copy()
    lookup = directions.set_index(["origin", "sample_id"])
    index = pd.MultiIndex.from_frame(base[["origin", "sample_id"]])
    try:
        aligned = lookup.loc[index].reset_index()
    except KeyError as exc:
        raise ContractError("saved M directions do not cover canonical V10 rows") from exc
    if not np.array_equal(base.spout_no.to_numpy(), aligned.spout_no.astype(str).to_numpy()):
        raise ContractError("saved M directions differ from scoring spouts")
    q = aligned.Q.to_numpy(dtype=float)
    if not np.array_equal(base.pred_tap_time_len.to_numpy(dtype=float), np.asarray([six_decimal_float(x) for x in q])):
        raise ContractError("canonical V10 time differs from saved Q after six-decimal serialization")
    m = aligned.M.to_numpy(dtype=float)
    known = np.isfinite(m)
    prediction = q.copy()
    prediction[known] = m[known]
    base["candidate"] = M_ONLY
    base["pred_tap_time_len"] = [six_decimal_float(value) for value in prediction]
    for target, predicted in zip(TARGETS, PREDICTIONS, strict=True):
        base[f"error_{target}"] = base[predicted] - base[target]
        base[f"abs_error_{target}"] = base[f"error_{target}"].abs()
    base["median_fallback_to_Q"] = ~known
    return base


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        raise ContractError("diagnostic quantiles require rows")
    return dict(zip(
        ("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"),
        np.quantile(values, [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist(),
    ))


def local_calendar_month(values) -> pd.Series:
    """Return competition-local months, including for offset-aware CSV text."""
    return pd.to_datetime(values, utc=True).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m")


def shrinkage_strength(
    canonical: pd.DataFrame,
    directions: pd.DataFrame,
    lambdas: Mapping[str, Mapping[str, float]],
) -> dict:
    grid = canonical.loc[(canonical.candidate == V22) & canonical.horizon.notna()].copy()
    actual = (
        grid.sort_values(["origin", "sample_id", "unit"], kind="mergesort")
        .drop_duplicates(["origin", "sample_id"])
        .set_index(["origin", "sample_id"])
    )
    direction = directions.set_index(["origin", "sample_id"]).sort_index()
    if not direction.index.equals(actual.sort_index().index):
        raise ContractError("V22 directions and unique grid exposures differ")
    actual = actual.loc[direction.index]
    rows = direction.reset_index()
    coefficients = np.asarray([
        float(lambdas[origin].get(str(spout), 0.0))
        for origin, spout in zip(rows.origin, rows.spout_no, strict=True)
    ])
    rows["lambda"] = coefficients
    rows["r"] = coefficients * (1.0 - rows.u.to_numpy(dtype=float))
    rows["absolute_prediction_change"] = np.abs(coefficients * rows.d.to_numpy(dtype=float))
    rows["full_precision_prediction"] = rows.Q.to_numpy(dtype=float) + coefficients * rows.d.to_numpy(dtype=float)
    rows["tap_time_len"] = actual.tap_time_len.to_numpy(dtype=float)
    rows["residual"] = rows.full_precision_prediction - rows.tap_time_len
    rows["target_month"] = local_calendar_month(rows.reference_time)
    groups = []
    for keys, part in rows.groupby(["origin", "spout_no"], sort=True):
        groups.append({
            "origin": keys[0], "spout_no": str(keys[1]), "n": int(len(part)),
            "lambda": float(part["lambda"].iloc[0]),
            "r_quantiles": _quantiles(part.r.to_numpy(dtype=float)),
            "absolute_prediction_change_quantiles": _quantiles(part.absolute_prediction_change.to_numpy(dtype=float)),
            "residual_mean": float(part.residual.mean()),
            "residual_median": float(part.residual.median()),
            "M_values": sorted(float(value) for value in part.M.dropna().unique()),
            "missing_M_rows": int(part.M.isna().sum()),
        })
    months = []
    for keys, part in rows.groupby(["origin", "target_month", "spout_no"], sort=True):
        months.append({
            "origin": keys[0], "target_month": keys[1], "spout_no": str(keys[2]), "n": int(len(part)),
            "r_mean": float(part.r.mean()),
            "absolute_prediction_change_mean": float(part.absolute_prediction_change.mean()),
            "residual_mean": float(part.residual.mean()),
            "residual_median": float(part.residual.median()),
        })
    return {
        "status": "PASS",
        "definition": "r=lambda*(1-u)",
        "uses_saved_Q_u_M_lambda_only": True,
        "new_parameter_or_median_estimation": False,
        "by_origin_spout": groups,
        "by_origin_target_month_spout": months,
    }


def diagnostic_table(canonical: pd.DataFrame, median_errors: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    combined = pd.concat([
        canonical.loc[canonical.candidate.isin((V10, V22))],
        median_errors,
    ], ignore_index=True, sort=False)
    card = scorecard(combined, algorithms=(V10, V22, M_ONLY))
    summary = score_summary(card)
    pivot = summary.pivot(index=["scope", "scope_type"], columns="algorithm", values="E")
    rows = []
    for (scope, scope_type), value in pivot.iterrows():
        for candidate in (V10, V22, M_ONLY):
            rows.append({
                "scope": scope, "scope_type": scope_type, "algorithm": candidate,
                "E": float(value[candidate]),
                "delta_vs_V10": float(value[candidate] - value[V10]),
                "delta_vs_V22": float(value[candidate] - value[V22]),
            })
    result = pd.DataFrame(rows).sort_values(["scope_type", "scope", "algorithm"], kind="mergesort").reset_index(drop=True)
    lookup = result.set_index(["scope", "algorithm"]).E
    conclusions = {
        "status": "PASS_DIAGNOSTIC_ONLY",
        "new_fit_or_statistic_estimation": False,
        "J": {candidate: float(lookup[("J", candidate)]) for candidate in (V10, V22, M_ONLY)},
        "V22_gain_beyond_direct_saved_M_on_J": float(lookup[("J", M_ONLY)] - lookup[("J", V22)]),
        "V22_strictly_better_than_direct_saved_M_on_J": bool(lookup[("J", V22)] < lookup[("J", M_ONLY)]),
        "H1": {candidate: float(lookup[("H1", candidate)]) for candidate in (V10, V22, M_ONLY)},
        "H2": {candidate: float(lookup[("H2", candidate)]) for candidate in (V10, V22, M_ONLY)},
        "V22_H1_H2_direction_vs_V10_consistent": bool(
            np.sign(lookup[("H1", V22)] - lookup[("H1", V10)])
            == np.sign(lookup[("H2", V22)] - lookup[("H2", V10)])
        ),
        "M_only_H1_H2_direction_vs_V10_consistent": bool(
            np.sign(lookup[("H1", M_ONLY)] - lookup[("H1", V10)])
            == np.sign(lookup[("H2", M_ONLY)] - lookup[("H2", V10)])
        ),
        "interpretation_limit": "diagnostic_only_no_followup_search_or_platform_authorization",
    }
    return result, conclusions


def monthly_bias_table(canonical: pd.DataFrame, median_errors: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([
        canonical.loc[(canonical.candidate.isin((V10, V22))) & canonical.horizon.notna()],
        median_errors.loc[median_errors.horizon.notna()],
    ], ignore_index=True, sort=False)
    combined["target_month"] = local_calendar_month(combined.reference_time)
    rows = []
    for (candidate, origin, month, spout), part in combined.groupby(
        ["candidate", "origin", "target_month", "spout_no"], sort=True
    ):
        for target in TARGETS:
            denominator = float(part[target].sum())
            rows.append({
                "algorithm": candidate, "origin": origin, "target_month": month,
                "spout_no": str(spout), "target": target, "n": int(len(part)),
                "target_sum": denominator,
                "absolute_error_sum": float(part[f"abs_error_{target}"].sum()),
                "wmape": float(part[f"abs_error_{target}"].sum() / denominator),
                "signed_bias": float(part[f"error_{target}"].mean()),
                "residual_median": float(part[f"error_{target}"].median()),
            })
    return pd.DataFrame(rows)


def training_history_from_handoff(train_path: Path, bundle: Mapping) -> pd.DataFrame:
    with np.load(train_path, allow_pickle=False) as data:
        required = {"ids", "spout", "reference_ns", "y", "available_ns"}
        if required - set(data.files):
            raise ContractError("certified final QRF handoff lacks frozen response history")
        ids = data["ids"].astype(str)
        if ids.tolist() != bundle["input"]["ids"] or len(set(ids)) != len(ids):
            raise ContractError("final QRF training handoff identity differs")
        history = pd.DataFrame({
            "sample_id": ids,
            "spout_no": data["spout"].astype(str),
            "reference_time": pd.to_datetime(data["reference_ns"], utc=True).tz_convert("Asia/Shanghai"),
            "tap_end_time": pd.to_datetime(data["available_ns"], utc=True).tz_convert("Asia/Shanghai"),
            "tap_time_len": data["y"].astype(float),
        })
    cutoff = pd.Timestamp(bundle["input"]["cutoff"])
    if history.sample_id.duplicated().any() or (history.reference_time >= cutoff).any() or (history.tap_end_time > cutoff).any():
        raise ContractError("final QRF training history violates cutoff")
    return history


def stage_metadata(path: Path) -> pd.DataFrame:
    value = pd.read_csv(path, dtype={"sample_id": str, "spout_no": str})
    if list(value) != ["sample_id", "tap_no", "spout_no", "reference_time"]:
        raise ContractError("stage metadata schema differs")
    if value.sample_id.isna().any() or value.sample_id.duplicated().any():
        raise ContractError("stage metadata IDs must be complete and unique")
    value["reference_time"] = pd.to_datetime(value.reference_time).dt.tz_localize("Asia/Shanghai")
    return value[["sample_id", "spout_no", "reference_time"]]


def stage_candidates(
    metadata: pd.DataFrame,
    qrf_path: Path,
    final_model: Mapping,
    saved_medians: Mapping,
    saved_lambdas: Mapping,
    history: pd.DataFrame,
    v1_path: Path,
    v10_path: Path,
) -> tuple[dict[str, bytes], dict]:
    """Compose V1/V21/V22 for one target-free stage without estimating state."""
    with np.load(qrf_path, allow_pickle=False) as data:
        if set(data.files) != {"ids", "Q", "effective_neighbors", "N"}:
            raise ContractError("fresh v0.23 QRF support payload differs")
        arrays = {key: data[key] for key in data.files}
    ids = metadata.sample_id.astype(str).tolist()
    if arrays["ids"].astype(str).tolist() != ids or len(set(arrays["N"].tolist())) != 1:
        raise ContractError("fresh QRF support output is not aligned to stage metadata")
    if int(arrays["N"][0]) != int(final_model["unique_training_rows"]):
        raise ContractError("fresh stage support denominator differs from final model")
    v1 = pd.read_csv(v1_path, dtype={"sample_id": str}, float_precision="round_trip")
    v10 = pd.read_csv(v10_path, dtype={"sample_id": str}, float_precision="round_trip")
    for name, frame in ((V1, v1), (V10, v10)):
        if list(frame) != ["sample_id", *PREDICTION_COLUMNS] or frame.sample_id.tolist() != ids:
            raise ContractError(f"{name} stage predictions differ from metadata identity/order")
    if not np.array_equal(v10.pred_tap_time_len.to_numpy(dtype=float), arrays["Q"]):
        raise ContractError("V10 stage time differs from fresh QRF median")
    v10_six = pd.read_csv(io_bytes(serialize_six_decimals(v10)), dtype={"sample_id": str}, float_precision="round_trip")
    v21, v21_diagnostics = replay_v21(
        v10_six, metadata, history, arrays["effective_neighbors"],
        cutoff=pd.Timestamp(final_model["cutoff_ns"], unit="ns", tz="UTC"),
    )
    certificates = saved_medians.get("certificates", {})
    if saved_medians.get("kind") != "V22_FINAL_CUTOFF_MEDIANS_v1" or set(certificates) != {"1", "2"}:
        raise ContractError("saved final V22 median parameters differ")
    parameters = saved_lambdas.get("spouts", {})
    if saved_lambdas.get("kind") != "V22_FINAL_SPOUT_LAMBDAS_v1" or set(parameters) != {"1", "2"}:
        raise ContractError("saved final V22 lambda parameters differ")
    directions = build_prediction_directions(
        metadata,
        arrays["Q"], arrays["effective_neighbors"], int(arrays["N"][0]), certificates,
        model_cutoff=pd.Timestamp(final_model["cutoff_ns"], unit="ns", tz="UTC"),
        training_identity_sha256=final_model["training_identity_sha256"],
        model_bundle_sha256=final_model["bundle_sha256"],
    )
    v22 = apply_shrink(v10, directions, {spout: float(row["lambda"]) for spout, row in parameters.items()})
    payloads = {V1: serialize_six_decimals(v1), V21: serialize_six_decimals(v21), V22: serialize_six_decimals(v22)}
    return payloads, {
        "rows": len(metadata),
        "ids_sha256": stable_digest(ids),
        "fresh_effective_neighbors": True,
        "N": int(arrays["N"][0]),
        "v21": v21_diagnostics,
        "v22_saved_medians": {spout: cert["median"] for spout, cert in certificates.items()},
        "v22_saved_lambdas": {spout: float(row["lambda"]) for spout, row in parameters.items()},
        "new_median_estimates": 0,
        "fit_attempts": 0,
    }


def io_bytes(value: bytes):
    from io import BytesIO
    return BytesIO(value)
