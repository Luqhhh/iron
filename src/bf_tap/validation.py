from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .artifacts import (
    atomic_write_json,
    build_inference_source_contract,
    code_identity,
    file_identities,
    file_sha256,
    runtime_environment,
    stable_digest,
    verify_file_identities,
)
from .availability import freeze_history_origin
from .config import (
    load_yaml,
    validate_acceptance_config,
    validate_baseline_config,
    validate_data_paths,
    validate_feature_config,
    validate_frozen_contracts,
    validate_semantic_contract,
    validate_validation_config,
)
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
from .protection import load_protection_policy, read_access_ledger
from .schema import (
    validate_cross_table_consistency,
    validate_cross_table_metadata,
    validate_history,
    validate_samples,
)
from .splits import select_split, validate_split_definitions
from .submission import write_submission
from .train import fit_baseline


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
    semantic_contract_path: str | Path,
    protection_policy_path: str | Path,
    protection_ledger_path: str | Path | None,
    output: str | Path,
) -> Path:
    data_cfg = load_yaml(data_config_path)
    validate_data_paths(data_cfg, command="development")
    semantic_cfg = load_yaml(semantic_contract_path)
    validate_semantic_contract(semantic_cfg)
    protection = load_protection_policy(protection_policy_path)
    baseline_cfg = load_yaml(baseline_config_path)
    validate_baseline_config(baseline_cfg)
    feature_cfg = load_yaml(feature_config_path)
    validate_feature_config(feature_cfg)
    contract_digests = validate_frozen_contracts(
        baseline_cfg, feature_cfg, semantic_cfg
    )
    split_cfg = load_yaml(split_config_path)
    validate_validation_config(split_cfg, timezone=protection.timezone)
    acceptance_cfg = load_yaml(Path(baseline_config_path).with_name("acceptance.yaml"))
    validate_acceptance_config(acceptance_cfg)
    all_folds = validate_split_definitions(split_cfg["folds"], protection)
    folds = [fold for fold in all_folds if fold.kind == "development"]
    if not folds:
        raise ContractError("no development folds configured")

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    resolved = {
        "data": data_cfg,
        "data_contract": semantic_cfg,
        "protection": {
            "contract_id": protection.contract_id,
            "policy_digest": protection.digest,
        },
        "baseline": baseline_cfg,
        "features": feature_cfg,
        "validation": split_cfg,
        "acceptance": acceptance_cfg,
    }
    atomic_write_json(destination / "resolved_config.json", resolved)
    used_paths = {
        key: data_cfg["paths"][key]
        for key in (
            "train_samples",
            "tap_history_train",
            "operation_hourly",
            "burden_change",
            "data_dictionary",
        )
    }
    try:
        inputs_before = file_identities(used_paths)
        inference_source_contract = build_inference_source_contract(
            inputs_before,
            semantic_contract_sha256=contract_digests[
                "semantic_contract_sha256"
            ],
        )
        code = code_identity(Path.cwd())
        environment = runtime_environment()
        lock_sha256 = file_sha256("uv.lock")
        ledger_status = read_access_ledger(protection_ledger_path, protection)
        atomic_write_json(destination / "environment.json", environment)
        atomic_write_json(
            destination / "run_state.json",
            {
                "status": "RUNNING",
                "suite": "development",
                "inputs_before": inputs_before,
                "code_identity": code,
                "lockfile_sha256": lock_sha256,
                "resolved_config_sha256": stable_digest(resolved),
                "contract_digests": contract_digests,
                "inference_source_contract": inference_source_contract,
                "protection_ledger": ledger_status,
                "cache": {"enabled": False, "reason": "no cache I/O in DEV workflow"},
            },
        )
    except Exception as exc:
        atomic_write_json(
            destination / "final_status.json",
            {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    try:
        # The development label gate never parses rows whose reference time is
        # in the protected November holdout.
        protection_cutoff = max(fold.eval_end for fold in folds)
        train_meta = read_csv(
            data_cfg["paths"]["train_samples"],
            usecols=["sample_id", "tap_no", "spout_no", "reference_time"],
            required_columns=["sample_id", "tap_no", "spout_no", "reference_time"],
            time_columns=["reference_time"],
        )
        history_mapping = semantic_cfg["sources"]["history"]
        history_meta = read_csv(
            data_cfg["paths"]["tap_history_train"],
            usecols=[
                "sample_id",
                "tap_no",
                "spout_no",
                history_mapping["event_time_column"],
                history_mapping["end_time_column"],
            ],
            required_columns=[
                "sample_id",
                "tap_no",
                "spout_no",
                history_mapping["event_time_column"],
                history_mapping["end_time_column"],
            ],
            time_columns=[
                history_mapping["event_time_column"],
                history_mapping["end_time_column"],
            ],
        ).rename(columns={history_mapping["event_time_column"]: "reference_time"})
        history_meta["available_at"] = history_meta[history_mapping["end_time_column"]]
        metadata_consistency = validate_cross_table_metadata(train_meta, history_meta)
        labels = read_development_labels(
            data_cfg["paths"]["train_samples"],
            requested_end=protection_cutoff,
            protection_policy=protection,
        )
        validate_samples(labels, labeled=True)
        history = read_development_history(
            data_cfg["paths"]["tap_history_train"],
            requested_end=protection_cutoff,
            protection_policy=protection,
            available_at_column=history_mapping["available_at_column"],
        )
        history = history.copy()
        history["available_at"] = history[history_mapping["available_at_column"]]
        validate_history(
            history,
            end_time=history_mapping["end_time_column"],
            available_at="available_at",
        )
        target_consistency = validate_cross_table_consistency(labels, history)
        target_available_at = semantic_cfg["targets"]["available_at_column"]
        label_availability = read_csv(
            data_cfg["paths"]["tap_history_train"],
            usecols=["sample_id", target_available_at],
            required_columns=["sample_id", target_available_at],
            time_columns=[target_available_at],
        ).rename(columns={target_available_at: "label_available_at"})
        labels = labels.merge(
            label_availability, on="sample_id", how="left", validate="one_to_one"
        )
        if labels["label_available_at"].isna().any():
            raise ContractError("training samples lack target availability mapping")

        operation_raw = read_csv(data_cfg["paths"]["operation_hourly"])
        op_map = semantic_cfg["sources"]["operation"]
        operation = normalize_event_source(
            operation_raw,
            event_time_column=op_map["event_time_column"],
            available_at_column=op_map["available_at_column"],
            value_columns=list(feature_cfg["operation"]["value_columns"]),
            missing_markers=op_map["missing_markers"],
        )
        burden_raw = read_csv(data_cfg["paths"]["burden_change"])
        burden_map = semantic_cfg["sources"]["burden"]
        burden = normalize_event_source(
            burden_raw,
            event_time_column=burden_map["event_time_column"],
            available_at_column=burden_map["available_at_column"],
            value_columns=list(feature_cfg["burden"]["value_columns"]),
            missing_markers=burden_map["missing_markers"],
        )

        all_metrics: dict[str, object] = {}
        all_quality_pass = True
        artifact_sha256: dict[str, dict[str, str]] = {}
        split_identities: dict[str, dict[str, str]] = {}
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
            history_snapshot = freeze_history_origin(history, fold.fit_cutoff)
            model.save(
                fold_dir / "bundle",
                metadata={
                    "baseline_config": baseline_cfg,
                    "semantic_contract": semantic_cfg,
                    "feature_config": feature_cfg,
                    "contract_digests": contract_digests,
                    "inference_source_contract": inference_source_contract,
                    "training": {
                        "mode": "development-validation",
                        "fold_id": fold.id,
                        "fit_cutoff": str(fold.fit_cutoff),
                        "train_start": str(fold.train_start),
                        "eligible_rows": len(train_samples),
                        "eligible_sample_id_sha256": stable_digest(
                            train_samples["sample_id"].astype(str).tolist()
                        ),
                        "history_rows": len(history_snapshot),
                        "history_origin_sha256": stable_digest(
                            history_snapshot[["sample_id", "available_at"]]
                            .astype(str)
                            .to_dict("records")
                        ),
                    },
                    "code_identity": code,
                    "environment": environment,
                    "lockfile_sha256": lock_sha256,
                },
                history_snapshot=history_snapshot,
            )

            controls = MedianControls(
                min_group_count=int(baseline_cfg["controls"]["per_spout"]["min_group_count"])
            ).fit(train_samples)
            b0 = controls.predict(eval_samples, "B0")
            b1 = controls.predict(eval_samples, "B1")
            b0.to_csv(fold_dir / "predictions_B0.csv", index=False)
            b1.to_csv(fold_dir / "predictions_B1.csv", index=False)
            metrics = {
                "catboost": score_predictions(eval_samples, prediction),
                "B0": score_predictions(eval_samples, b0),
                "B1": score_predictions(eval_samples, b1),
                "split": split_evidence,
                "clipped_counts": {
                    column: int((raw_prediction[column] < 0).sum())
                    for column in ("pred_tap_iron", "pred_tap_time_len")
                },
            }
            any_missing = eval_features.X.isna().any(axis=1).map(
                {True: "any_missing", False: "complete"}
            )
            any_stale = (
                (eval_features.X["operation__stale"] > 0)
                | (eval_features.X["burden__stale"] > 0)
            ).map({True: "any_stale", False: "fresh"})
            predictions_by_model = {"catboost": prediction, "B0": b0, "B1": b1}
            for model_name, model_prediction in predictions_by_model.items():
                metrics[f"{model_name}_by_month"] = _monthly_metrics(
                    eval_samples, model_prediction
                )
                metrics[f"{model_name}_by_spout"] = _group_metrics(
                    eval_samples, model_prediction, eval_samples["spout_no"]
                )
                metrics[f"{model_name}_by_missing"] = _group_metrics(
                    eval_samples, model_prediction, any_missing
                )
                metrics[f"{model_name}_by_stale"] = _group_metrics(
                    eval_samples, model_prediction, any_stale
                )
            roundtrip = pd.read_csv(
                fold_dir / "predictions.csv", dtype={"sample_id": "string"}
            )
            metrics["catboost_csv_roundtrip"] = score_predictions(eval_samples, roundtrip)
            metrics["quality"] = evaluate_quality(metrics, acceptance_cfg["quality"])
            metrics["walltime_seconds"] = perf_counter() - fold_started
            all_quality_pass = all_quality_pass and bool(metrics["quality"]["pass"])

            tracking = pd.DataFrame(index=eval_samples.index)
            for source, audit in eval_features.audit.items():
                for column in audit.columns:
                    if column != "reference_time":
                        tracking[f"{source}__audit__{column}"] = audit[column]
            error_source = pd.concat(
                [
                    eval_samples[
                        [
                            "sample_id",
                            "reference_time",
                            "spout_no",
                            "tap_iron",
                            "tap_time_len",
                        ]
                    ],
                    tracking,
                ],
                axis=1,
            )
            errors = error_source.merge(
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
            atomic_write_json(fold_dir / "metrics.json", metrics)
            atomic_write_json(
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
            artifact_sha256[fold.id] = {
                str(path.relative_to(fold_dir)): file_sha256(path)
                for path in sorted(fold_dir.rglob("*"))
                if path.is_file()
            }
            split_identities[fold.id] = {
                "train_sample_id_sha256": stable_digest(
                    train_samples["sample_id"].astype(str).tolist()
                ),
                "eval_sample_id_sha256": stable_digest(
                    eval_samples["sample_id"].astype(str).tolist()
                ),
                "history_origin_sha256": model.bundle_metadata_["training"][
                    "history_origin_sha256"
                ]
                if hasattr(model, "bundle_metadata_")
                else stable_digest(
                    history_snapshot[["sample_id", "available_at"]]
                    .astype(str)
                    .to_dict("records")
                ),
            }
            all_metrics[fold.id] = metrics

        verify_file_identities(inputs_before)
        atomic_write_json(
            destination / "run_manifest.json",
            {
                "status": "PASS_EXECUTION",
                "suite": "development",
                "protection": {
                    "policy_id": protection.contract_id,
                    "policy_digest": protection.digest,
                    "development_read_end_exclusive": str(protection_cutoff),
                    "ledger": read_access_ledger(protection_ledger_path, protection),
                },
                "availability_contract": {
                    "contract_id": semantic_cfg["contract_id"],
                    "evidence_status": semantic_cfg["evidence_status"],
                },
                "cross_table_consistency": {
                    "metadata": metadata_consistency,
                    "targets": target_consistency,
                },
                "input_identities": inputs_before,
                "inputs_stable": True,
                "code_identity": code,
                "lockfile_sha256": lock_sha256,
                "resolved_config_sha256": stable_digest(resolved),
                "contract_digests": contract_digests,
                "inference_source_contract": inference_source_contract,
                "split_identities": split_identities,
                "artifact_sha256": artifact_sha256,
                "cache": {"enabled": False, "reason": "no cache I/O in DEV workflow"},
                "quality_status": "PASS" if all_quality_pass else "FAIL",
                "walltime_seconds": perf_counter() - started,
            },
        )
        atomic_write_json(destination / "metrics.json", all_metrics)
        atomic_write_json(
            destination / "final_status.json",
            {
                "status": "PASS_EXECUTION",
                "quality_status": "PASS" if all_quality_pass else "FAIL",
                "run_manifest_sha256": file_sha256(destination / "run_manifest.json"),
                "metrics_sha256": file_sha256(destination / "metrics.json"),
            },
        )
    except Exception as exc:
        atomic_write_json(
            destination / "final_status.json",
            {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    return destination
