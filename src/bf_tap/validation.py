from __future__ import annotations

import json
import platform
from pathlib import Path
from time import perf_counter

import catboost
import numpy as np
import pandas as pd
import yaml

from .audit import sha256_file
from .config import load_yaml, require_real_data_contract
from .data import normalize_event_source
from .exceptions import ContractError
from .features import build_features
from .io import (
    read_csv,
    read_development_history,
    read_development_labels,
)
from .metrics import score_predictions
from .models.sanity import MedianControls
from .splits import select_split, split_from_config
from .submission import write_submission
from .train import fit_baseline


def _dump(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _prediction_frame(ids: pd.Series, values: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ids.astype("string").to_numpy(),
            "pred_tap_iron": values["pred_tap_iron"].to_numpy(),
            "pred_tap_time_len": values["pred_tap_time_len"].to_numpy(),
        }
    )


def _monthly_metrics(actual: pd.DataFrame, predicted: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    months = actual["reference_time"].dt.strftime("%Y-%m")
    for month in sorted(months.unique()):
        part = actual.loc[months == month]
        ids = set(part["sample_id"].astype(str))
        pred = predicted.loc[predicted["sample_id"].astype(str).isin(ids)]
        result[month] = score_predictions(part, pred)
    return result


def _group_metrics(
    actual: pd.DataFrame, predicted: pd.DataFrame, groups: pd.Series
) -> dict[str, object]:
    result: dict[str, object] = {}
    for group in sorted(groups.astype(str).unique()):
        part = actual.loc[groups.astype(str) == group]
        ids = set(part["sample_id"].astype(str))
        pred = predicted.loc[predicted["sample_id"].astype(str).isin(ids)]
        result[group] = score_predictions(part, pred)
    return result


def evaluate_quality(metrics: dict[str, object], policy: dict[str, object]) -> dict[str, object]:
    cat = metrics["catboost"]
    b0 = metrics["B0"]
    b1 = metrics["B1"]
    margin = float(policy["max_loss_relative_to_better_control"])
    tolerance = float(policy["per_target_max_absolute_wmape_regression_vs_B1"])
    combined_limit = min(float(b0["loss"]), float(b1["loss"])) + margin
    combined_pass = float(cat["loss"]) <= combined_limit
    target_pass = {
        target: float(cat[target]["wmape"]) <= float(b1[target]["wmape"]) + tolerance
        for target in ("iron", "time")
    }
    return {
        "pass": combined_pass and all(target_pass.values()),
        "combined_pass": combined_pass,
        "combined_limit": combined_limit,
        "per_target_pass": target_pass,
        "per_target_tolerance": tolerance,
    }


def run_development_validation(
    *,
    data_config_path: str | Path,
    baseline_config_path: str | Path,
    feature_config_path: str | Path,
    split_config_path: str | Path,
    output: str | Path,
) -> Path:
    data_cfg = load_yaml(data_config_path)
    require_real_data_contract(data_cfg)
    baseline_cfg = load_yaml(baseline_config_path)
    feature_cfg = load_yaml(feature_config_path)
    split_cfg = load_yaml(split_config_path)
    acceptance_cfg = load_yaml(Path(baseline_config_path).with_name("acceptance.yaml"))
    folds = [split_from_config(v) for v in split_cfg["folds"] if v["kind"] == "development"]
    if not folds:
        raise ContractError("no development folds configured")

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    try:
        # The development label gate never parses rows whose reference time is
        # in the protected November holdout.
        protection_cutoff = min(fold.eval_end for fold in folds)
        labels = read_development_labels(
            data_cfg["paths"]["train_samples"], str(protection_cutoff)
        )
        history = read_development_history(
            data_cfg["paths"]["tap_history_train"],
            str(protection_cutoff),
            available_at_column=data_cfg["sources"]["history"]["available_at_column"],
        )
        history = history.copy()
        history["available_at"] = history[
            data_cfg["sources"]["history"]["available_at_column"]
        ]
        history_meta = read_csv(
            data_cfg["paths"]["tap_history_train"],
            usecols=["sample_id", data_cfg["supervised_target_available_at"]],
            required_columns=["sample_id", data_cfg["supervised_target_available_at"]],
            time_columns=[data_cfg["supervised_target_available_at"]],
        ).rename(columns={data_cfg["supervised_target_available_at"]: "label_available_at"})
        labels = labels.merge(history_meta, on="sample_id", how="left", validate="one_to_one")
        if labels["label_available_at"].isna().any():
            raise ContractError("training samples lack target availability mapping")

        operation_raw = read_csv(data_cfg["paths"]["operation_hourly"])
        op_map = data_cfg["sources"]["operation"]
        operation = normalize_event_source(
            operation_raw,
            event_time_column=op_map["event_time_column"],
            available_at_column=op_map["available_at_column"],
            value_columns=list(feature_cfg["operation"]["value_columns"]),
        )
        burden_raw = read_csv(data_cfg["paths"]["burden_change"])
        burden_map = data_cfg["sources"]["burden"]
        burden = normalize_event_source(
            burden_raw,
            event_time_column=burden_map["event_time_column"],
            available_at_column=burden_map["available_at_column"],
            value_columns=list(feature_cfg["burden"]["value_columns"]),
        )

        all_metrics: dict[str, object] = {}
        all_quality_pass = True
        for fold in folds:
            fold_started = perf_counter()
            fold_dir = destination / fold.id
            fold_dir.mkdir()
            train_samples, eval_samples, split_evidence = select_split(labels, fold)
            if train_samples.empty or eval_samples.empty:
                raise ContractError(f"{fold.id}: empty train or evaluation partition")
            train_features = build_features(
                train_samples,
                operation=operation,
                burden=burden,
                history=history,
                fit_cutoff=fold.fit_cutoff,
                config=feature_cfg,
            )
            eval_features = build_features(
                eval_samples,
                operation=operation,
                burden=burden,
                history=history,
                fit_cutoff=fold.fit_cutoff,
                config=feature_cfg,
            )
            if list(train_features.X.columns) != list(eval_features.X.columns):
                raise ContractError(f"{fold.id}: train/eval feature columns differ")

            model = fit_baseline(
                train_samples,
                train_features.X,
                dict(baseline_cfg["parameters"]),
                categorical=tuple(baseline_cfg["categorical_features"]),
            )
            raw_values = model.predict_raw(eval_features.X)
            final_values = raw_values.clip(lower=0.0)
            raw_prediction = _prediction_frame(eval_samples["sample_id"], raw_values)
            prediction = _prediction_frame(eval_samples["sample_id"], final_values)
            raw_prediction.to_csv(fold_dir / "predictions_raw.csv", index=False)
            write_submission(
                prediction,
                eval_samples["sample_id"],
                fold_dir / "predictions.csv",
            )
            model.save(fold_dir / "bundle")

            controls = MedianControls(
                min_group_count=int(baseline_cfg["controls"]["per_spout"]["min_group_count"])
            ).fit(train_samples)
            b0 = controls.predict(eval_samples, "B0")
            b1 = controls.predict(eval_samples, "B1")
            metrics = {
                "catboost": score_predictions(eval_samples, prediction),
                "B0": score_predictions(eval_samples, b0),
                "B1": score_predictions(eval_samples, b1),
                "catboost_by_month": _monthly_metrics(eval_samples, prediction),
                "catboost_by_spout": _group_metrics(
                    eval_samples, prediction, eval_samples["spout_no"]
                ),
                "split": split_evidence,
            }
            any_missing = eval_features.X.isna().any(axis=1).map(
                {True: "any_missing", False: "complete"}
            )
            any_stale = (
                (eval_features.X["operation__stale"] > 0)
                | (eval_features.X["burden__stale"] > 0)
            ).map({True: "any_stale", False: "fresh"})
            metrics["catboost_by_missing"] = _group_metrics(
                eval_samples, prediction, any_missing
            )
            metrics["catboost_by_stale"] = _group_metrics(
                eval_samples, prediction, any_stale
            )
            roundtrip = pd.read_csv(
                fold_dir / "predictions.csv", dtype={"sample_id": "string"}
            )
            metrics["catboost_csv_roundtrip"] = score_predictions(eval_samples, roundtrip)
            metrics["quality"] = evaluate_quality(metrics, acceptance_cfg["quality"])
            metrics["walltime_seconds"] = perf_counter() - fold_started
            all_quality_pass = all_quality_pass and bool(metrics["quality"]["pass"])

            errors = eval_samples[["sample_id", "tap_iron", "tap_time_len"]].merge(
                prediction, on="sample_id", validate="one_to_one"
            )
            errors["abs_error_tap_iron"] = np.abs(
                errors["pred_tap_iron"] - errors["tap_iron"]
            )
            errors["abs_error_tap_time_len"] = np.abs(
                errors["pred_tap_time_len"] - errors["tap_time_len"]
            )
            errors.nlargest(20, "abs_error_tap_iron").to_csv(
                fold_dir / "top20_iron_errors.csv", index=False
            )
            errors.nlargest(20, "abs_error_tap_time_len").to_csv(
                fold_dir / "top20_time_errors.csv", index=False
            )
            _dump(fold_dir / "metrics.json", metrics)
            _dump(
                fold_dir / "feature_manifest.json",
                {
                    "columns": list(train_features.X.columns),
                    "categorical": list(train_features.categorical),
                    "train_rows": len(train_features.X),
                    "eval_rows": len(eval_features.X),
                    "disabled_fields": [],
                },
            )
            for source, audit in eval_features.audit.items():
                audit.to_csv(fold_dir / f"feature_audit_{source}.csv", index=False)
            all_metrics[fold.id] = metrics

        data_identity = {
            key: sha256_file(value)
            for key, value in data_cfg["paths"].items()
            if value and Path(value).is_file()
        }
        _dump(
            destination / "run_manifest.json",
            {
                "status": "PASS_EXECUTION",
                "suite": "development",
                "holdout_consumed": False,
                "availability_contract": data_cfg["status"],
                "data_sha256": data_identity,
                "quality_status": "PASS" if all_quality_pass else "FAIL",
                "walltime_seconds": perf_counter() - started,
            },
        )
        _dump(
            destination / "resolved_config.json",
            {
                "data": data_cfg,
                "baseline": baseline_cfg,
                "features": feature_cfg,
                "validation": split_cfg,
                "acceptance": acceptance_cfg,
            },
        )
        _dump(
            destination / "environment.json",
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "catboost": catboost.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyyaml": yaml.__version__,
            },
        )
        _dump(destination / "metrics.json", all_metrics)
    except Exception as exc:
        (destination / "status.json").write_text(
            json.dumps({"status": "FAILED", "error": str(exc)}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        raise
    return destination
