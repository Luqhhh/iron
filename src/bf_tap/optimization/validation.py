from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..exceptions import ContractError
from ..metrics import score_predictions, target_metrics
from .config import EvaluationUnit, Origin
from .ensemble import shrink_toward_b1


def units_for_origin(origin: Origin, train_start: pd.Timestamp) -> list[EvaluationUnit]:
    return [
        EvaluationUnit(
            id=f"{origin.id}_H{horizon}",
            origin_id=origin.id,
            horizon=horizon,
            fit_cutoff=origin.fit_cutoff,
            train_start=train_start,
            eval_start=origin.fit_cutoff + pd.DateOffset(months=horizon - 1),
            eval_end=origin.fit_cutoff + pd.DateOffset(months=horizon),
        )
        for horizon in range(1, origin.horizons + 1)
    ]


def select_partitions(
    samples: pd.DataFrame, unit: EvaluationUnit
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    required = {"sample_id", "reference_time", "label_available_at"}
    missing = required - set(samples)
    if missing:
        raise ContractError(f"optimization split input missing columns: {sorted(missing)}")
    train_time = (samples["reference_time"] >= unit.train_start) & (
        samples["reference_time"] < unit.fit_cutoff
    )
    ready = samples["label_available_at"] <= unit.fit_cutoff
    evaluation = (samples["reference_time"] >= unit.eval_start) & (
        samples["reference_time"] < unit.eval_end
    )
    train = samples.loc[train_time & ready].copy()
    test = samples.loc[evaluation].copy()
    if train.empty or test.empty:
        raise ContractError(f"{unit.id}: empty train or evaluation partition")
    overlap = set(train["sample_id"].astype(str)) & set(test["sample_id"].astype(str))
    if overlap:
        raise ContractError(f"{unit.id}: train/evaluation sample_id overlap")
    return train, test, {
        "candidate_train_rows": int(train_time.sum()),
        "eligible_train_rows": len(train),
        "excluded_label_unavailable_rows": int((train_time & ~ready).sum()),
        "evaluation_rows": len(test),
    }


def history_age_groups(features: pd.DataFrame) -> pd.Series:
    column = "history__all__latest_available_age_minutes"
    if column not in features:
        return pd.Series("history_disabled", index=features.index, dtype="string")
    hours = pd.to_numeric(features[column], errors="coerce") / 60.0
    return pd.cut(
        hours,
        bins=[-np.inf, 24, 24 * 7, 24 * 30, 24 * 60, 24 * 90, np.inf],
        labels=["0-1d", "1-7d", "7-30d", "30-60d", "60-90d", ">90d"],
    ).astype("string").fillna("missing")


def _group_metrics(actual: pd.DataFrame, predicted: pd.DataFrame, groups: pd.Series) -> dict[str, object]:
    result: dict[str, object] = {}
    normalized = groups.astype("string").fillna("missing")
    for group in sorted(normalized.unique()):
        part = actual.loc[normalized == group]
        ids = set(part["sample_id"].astype(str))
        pred = predicted.loc[predicted["sample_id"].astype(str).isin(ids)]
        result[str(group)] = score_predictions(part, pred)
    return result


def diagnostic_metrics(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    full_features: pd.DataFrame,
) -> dict[str, object]:
    months = actual["reference_time"].dt.strftime("%Y-%m")
    stale = (
        (full_features["operation__stale"] > 0)
        | (full_features["burden__stale"] > 0)
    ).map({True: "any_stale", False: "fresh"})
    missing = full_features.isna().any(axis=1).map({True: "any_missing", False: "complete"})
    return {
        "overall": score_predictions(actual, predicted),
        "by_month": _group_metrics(actual, predicted, months),
        "by_spout": _group_metrics(actual, predicted, actual["spout_no"]),
        "by_missing": _group_metrics(actual, predicted, missing),
        "by_stale": _group_metrics(actual, predicted, stale),
        "by_history_age": _group_metrics(actual, predicted, history_age_groups(full_features)),
    }


def error_contributions(actual: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    columns = ["sample_id", "reference_time", "spout_no", "tap_iron", "tap_time_len"]
    result = actual[columns].merge(predicted, on="sample_id", validate="one_to_one")
    for target in ("tap_iron", "tap_time_len"):
        result[f"error_{target}"] = result[f"pred_{target}"] - result[target]
        result[f"abs_error_{target}"] = result[f"error_{target}"].abs()
    result["week"] = result["reference_time"].dt.strftime("%G-W%V")
    return result


def shrinkage_diagnostics(
    actual: pd.DataFrame,
    learned: pd.DataFrame,
    b1: pd.DataFrame,
    grid: list[float],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for target in ("tap_iron", "tap_time_len"):
        target_rows = []
        for weight in grid:
            prediction = shrink_toward_b1(
                learned,
                b1,
                {"tap_iron": weight if target == "tap_iron" else 1.0,
                 "tap_time_len": weight if target == "tap_time_len" else 1.0},
            )
            metric = target_metrics(actual[target], prediction[f"pred_{target}"])
            target_rows.append({"weight": weight, **metric.__dict__})
        result[target] = target_rows
    return result


def aggregate_grid(metrics: dict[str, dict[str, object]]) -> dict[str, object]:
    """Aggregate scenarios without pretending repeated samples are independent OOF rows."""
    by_candidate: dict[str, dict[int, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for unit_id, unit_values in metrics.items():
        horizon = unit_values.get("horizon")
        if not isinstance(horizon, int) or horizon < 1:
            continue
        for candidate, values in unit_values["candidates"].items():
            by_candidate[candidate][horizon].append(values["overall"])
    output: dict[str, object] = {}
    for candidate, horizons in by_candidate.items():
        horizon_summary: dict[str, object] = {}
        horizon_losses = []
        for horizon in range(1, 5):
            cells = horizons.get(horizon, [])
            if not cells:
                raise ContractError(f"{candidate}: grid is missing horizon {horizon}")
            loss = float(np.mean([cell["loss"] for cell in cells]))
            horizon_losses.append(loss)
            horizon_summary[f"H{horizon}"] = {
                "origins": len(cells),
                "mean_loss": loss,
                "iron_mean_wmape": float(np.mean([cell["iron"]["wmape"] for cell in cells])),
                "time_mean_wmape": float(np.mean([cell["time"]["wmape"] for cell in cells])),
            }
        output[candidate] = {
            "horizons": horizon_summary,
            "J": float(np.mean(horizon_losses)),
        }
    return output


def evaluate_acceptance(
    metrics: dict[str, dict[str, object]],
    grid_summary: dict[str, object],
    policy: dict[str, object],
) -> dict[str, object]:
    if not grid_summary or "DEV_LONG" not in metrics or "DEV_SHORT" not in metrics:
        return {
            "status": "INCOMPLETE",
            "reason": "acceptance requires suite=all (screening folds and complete grid)",
        }
    baseline = grid_summary["E00"]
    output: dict[str, object] = {}
    for candidate, summary in grid_summary.items():
        if candidate in {"E00", "B0", "B1"}:
            continue
        horizon_deltas = {
            horizon: float(summary["horizons"][horizon]["mean_loss"])
            - float(baseline["horizons"][horizon]["mean_loss"])
            for horizon in ("H1", "H2", "H3", "H4")
        }
        improved = sum(delta < 0 for delta in horizon_deltas.values())
        target_deltas = {}
        for target in ("iron", "time"):
            key = f"{target}_mean_wmape"
            target_deltas[target] = float(np.mean([
                summary["horizons"][horizon][key]
                - baseline["horizons"][horizon][key]
                for horizon in ("H1", "H2", "H3", "H4")
            ]))
        long_loss = metrics["DEV_LONG"]["candidates"][candidate]["overall"]["loss"]
        better_control = min(
            metrics["DEV_LONG"]["candidates"][control]["overall"]["loss"]
            for control in ("B0", "B1")
        )
        short_delta = (
            metrics["DEV_SHORT"]["candidates"][candidate]["overall"]["loss"]
            - metrics["DEV_SHORT"]["candidates"]["E00"]["overall"]["loss"]
        )
        checks = {
            "J_improvement": float(baseline["J"] - summary["J"])
            >= float(policy["min_J_improvement"]),
            "improved_horizons": improved >= int(policy["min_improved_horizons"]),
            "max_horizon_regression": max(horizon_deltas.values())
            <= float(policy["max_any_horizon_regression"]),
            "dev_long_better_than_control": (
                long_loss < better_control
                if policy["dev_long_must_beat_better_control"]
                else True
            ),
            "dev_short_regression": short_delta <= float(policy["dev_short_max_regression"]),
            "per_target_regression": max(target_deltas.values())
            <= float(policy["per_target_mean_wmape_max_regression"]),
        }
        output[candidate] = {
            "pass": all(checks.values()),
            "checks": checks,
            "J": summary["J"],
            "J_improvement": float(baseline["J"] - summary["J"]),
            "horizon_loss_delta": horizon_deltas,
            "improved_horizons": improved,
            "target_mean_wmape_delta": target_deltas,
            "DEV_LONG_loss": long_loss,
            "DEV_LONG_better_control_loss": better_control,
            "DEV_SHORT_loss_delta": short_delta,
        }
    return {"status": "PASS" if any(row["pass"] for row in output.values()) else "FAIL", "candidates": output}
