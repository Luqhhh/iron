from __future__ import annotations

import numpy as np
import pandas as pd

from ..exceptions import ContractError
from ..metrics import score_predictions


_TARGETS = ("tap_iron", "tap_time_len")
_PREDICTION_COLUMNS = ("pred_tap_iron", "pred_tap_time_len")
_HOUR_LABELS = ("00-05", "06-11", "12-17", "18-23")


def _prediction_part(predicted: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    wanted = set(ids.astype(str))
    return predicted.loc[
        predicted["sample_id"].astype(str).isin(wanted),
        ["sample_id", *_PREDICTION_COLUMNS],
    ]


def _group_metrics(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    groups: pd.Series,
) -> dict[str, object]:
    normalized = groups.astype("string").fillna("missing")
    result: dict[str, object] = {}
    for group in sorted(normalized.unique()):
        mask = normalized == group
        part = actual.loc[mask]
        result[str(group)] = score_predictions(
            part,
            _prediction_part(predicted, part["sample_id"]),
        )
    return result


def diagnostic_metrics(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
) -> dict[str, object]:
    required = {
        "sample_id",
        "reference_time",
        "spout_no",
        *_TARGETS,
    }
    missing = required - set(actual)
    if missing:
        raise ContractError(f"diagnostic labels missing columns: {sorted(missing)}")
    overall = score_predictions(actual, predicted)
    reference_time = pd.to_datetime(actual["reference_time"], errors="coerce")
    if reference_time.isna().any():
        raise ContractError("diagnostic reference_time must be parseable")
    hour_bucket = pd.cut(
        reference_time.dt.hour,
        bins=[-1, 5, 11, 17, 23],
        labels=list(_HOUR_LABELS),
    ).astype("string")
    quartiles: dict[str, object] = {}
    if len(actual) < 4:
        raise ContractError("actual-quartile diagnostics require at least four rows")
    for target in _TARGETS:
        ranks = pd.to_numeric(actual[target], errors="coerce").rank(method="first")
        if ranks.isna().any():
            raise ContractError(f"{target} values must be numeric for quartiles")
        groups = pd.qcut(
            ranks,
            4,
            labels=["Q1", "Q2", "Q3", "Q4"],
        ).astype("string")
        quartiles[target] = _group_metrics(actual, predicted, groups)
    return {
        "overall": overall,
        "by_month": _group_metrics(
            actual,
            predicted,
            reference_time.dt.strftime("%Y-%m"),
        ),
        "by_spout": _group_metrics(
            actual,
            predicted,
            actual["spout_no"].astype("string"),
        ),
        "by_hour_bucket": _group_metrics(actual, predicted, hour_bucket),
        "by_actual_quartile": quartiles,
    }


def scenario_summary(rows: pd.DataFrame) -> dict[str, object]:
    required = {
        "candidate_id",
        "unit_id",
        "sample_id",
        "reference_time",
        "spout_no",
        *_TARGETS,
        *_PREDICTION_COLUMNS,
    }
    missing = required - set(rows)
    if missing:
        raise ContractError(f"scenario rows missing columns: {sorted(missing)}")
    output: dict[str, object] = {}
    if "horizon" in rows:
        rows = rows.loc[rows["horizon"] >= 1].copy()
    for candidate_id, part in rows.groupby("candidate_id", sort=True):
        part = part.copy()
        if part.duplicated(["unit_id", "sample_id"]).any():
            raise ContractError(
                f"{candidate_id}: duplicate unit_id/sample_id scenario key"
            )
        part["scenario_sample_id"] = (
            part["unit_id"].astype(str)
            + "::"
            + part["sample_id"].astype(str)
        )
        actual = part[
            [
                "scenario_sample_id",
                "reference_time",
                "spout_no",
                *_TARGETS,
            ]
        ].rename(columns={"scenario_sample_id": "sample_id"})
        predicted = part[
            ["scenario_sample_id", *_PREDICTION_COLUMNS]
        ].rename(columns={"scenario_sample_id": "sample_id"})
        reference_time = pd.to_datetime(actual["reference_time"], errors="coerce")
        output[str(candidate_id)] = {
            "scenario_rows": len(part),
            "overall": score_predictions(actual, predicted),
            "by_month": _group_metrics(
                actual,
                predicted,
                reference_time.dt.strftime("%Y-%m"),
            ),
        }
    return output


def _loss(
    iron_abs_error: np.ndarray,
    time_abs_error: np.ndarray,
    iron_actual: np.ndarray,
    time_actual: np.ndarray,
) -> np.ndarray:
    iron_denominator = iron_actual.sum(axis=-1)
    time_denominator = time_actual.sum(axis=-1)
    if (iron_denominator <= 0).any() or (time_denominator <= 0).any():
        raise ContractError("bootstrap WMAPE denominator must be positive")
    return 0.5 * (
        iron_abs_error.sum(axis=-1) / iron_denominator
        + time_abs_error.sum(axis=-1) / time_denominator
    )


def paired_daily_block_bootstrap(
    rows: pd.DataFrame,
    candidate_id: str,
    control_id: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int | str]:
    if repetitions <= 0:
        raise ContractError("bootstrap repetitions must be positive")
    keys = ["unit_id", "sample_id"]
    value_columns = [
        "reference_time",
        "tap_iron",
        "tap_time_len",
        *_PREDICTION_COLUMNS,
    ]
    candidate = rows.loc[
        rows["candidate_id"] == candidate_id,
        [*keys, *value_columns],
    ]
    control = rows.loc[
        rows["candidate_id"] == control_id,
        [*keys, *value_columns],
    ]
    if candidate.empty or control.empty:
        raise ContractError("bootstrap candidate/control rows must be non-empty")
    try:
        paired = candidate.merge(
            control,
            on=keys,
            suffixes=("_candidate", "_control"),
            validate="one_to_one",
        )
    except pd.errors.MergeError as exc:
        raise ContractError("bootstrap scenario keys must be one-to-one") from exc
    if len(paired) != len(candidate) or len(paired) != len(control):
        raise ContractError("bootstrap candidate/control scenario keys differ")
    for column in ("reference_time", *_TARGETS):
        left = paired[f"{column}_candidate"]
        right = paired[f"{column}_control"]
        if column == "reference_time":
            equal = pd.to_datetime(left).astype(str).equals(
                pd.to_datetime(right).astype(str)
            )
        else:
            equal = np.array_equal(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
            )
        if not equal:
            raise ContractError(f"bootstrap actual {column} values differ")

    paired["local_date"] = pd.to_datetime(
        paired["reference_time_candidate"]
    ).dt.strftime("%Y-%m-%d")
    for side in ("candidate", "control"):
        for target in _TARGETS:
            paired[f"{target}_abs_error_{side}"] = (
                paired[f"pred_{target}_{side}"] - paired[f"{target}_candidate"]
            ).abs()
    blocks = paired.groupby(["unit_id", "local_date"], sort=True).agg(
        iron_actual=("tap_iron_candidate", "sum"),
        time_actual=("tap_time_len_candidate", "sum"),
        iron_candidate=("tap_iron_abs_error_candidate", "sum"),
        time_candidate=("tap_time_len_abs_error_candidate", "sum"),
        iron_control=("tap_iron_abs_error_control", "sum"),
        time_control=("tap_time_len_abs_error_control", "sum"),
    )
    if blocks.empty:
        raise ContractError("bootstrap has no local-day blocks")

    def column(name: str) -> np.ndarray:
        return blocks[name].to_numpy(dtype=float)

    actual_iron = column("iron_actual")
    actual_time = column("time_actual")
    candidate_iron = column("iron_candidate")
    candidate_time = column("time_candidate")
    control_iron = column("iron_control")
    control_time = column("time_control")
    point = float(
        _loss(candidate_iron[None, :], candidate_time[None, :],
              actual_iron[None, :], actual_time[None, :])[0]
        - _loss(control_iron[None, :], control_time[None, :],
                actual_iron[None, :], actual_time[None, :])[0]
    )

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(blocks),
        size=(repetitions, len(blocks)),
    )
    deltas = _loss(
        candidate_iron[indices],
        candidate_time[indices],
        actual_iron[indices],
        actual_time[indices],
    ) - _loss(
        control_iron[indices],
        control_time[indices],
        actual_iron[indices],
        actual_time[indices],
    )
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    return {
        "candidate_id": candidate_id,
        "control_id": control_id,
        "point_delta": point,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "repetitions": repetitions,
        "seed": seed,
        "blocks": len(blocks),
    }
