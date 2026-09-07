from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import (
    atomic_write_json,
    code_identity,
    file_identities,
    file_sha256,
    runtime_environment,
    stable_digest,
    verify_file_identities,
)
from ..availability import freeze_history_origin
from ..config import (
    load_yaml,
    validate_baseline_config,
    validate_data_paths,
    validate_feature_config,
    validate_frozen_contracts,
    validate_semantic_contract,
)
from ..data import normalize_event_source
from ..exceptions import ContractError
from ..features import build_features
from ..io import read_csv, read_development_history, read_development_labels
from ..models.sanity import MedianControls
from ..protection import load_protection_policy, read_access_ledger
from ..schema import (
    validate_cross_table_consistency,
    validate_cross_table_metadata,
    validate_history,
    validate_samples,
)
from ..submission import write_submission
from .config import (
    Candidate,
    EvaluationUnit,
    load_acceptance,
    load_experiment,
    load_feature_selection,
    load_model_config,
    load_validation,
)
from .ensemble import shrink_toward_b1
from .features import select_candidate_features
from .history_adaptation import HistoryViewSet, build_history_views
from .models import FrozenBaselineAdapter
from .process_change import add_process_change_features
from .registry import write_registry
from .validation import (
    aggregate_grid,
    diagnostic_metrics,
    error_contributions,
    evaluate_acceptance,
    select_partitions,
    shrinkage_diagnostics,
    units_for_origin,
)


def _prediction_frame(ids: pd.Series, values: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ids.astype("string").to_numpy(),
            "pred_tap_iron": values["pred_tap_iron"].to_numpy(),
            "pred_tap_time_len": values["pred_tap_time_len"].to_numpy(),
        }
    )


def _prediction_part(prediction: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    order = pd.DataFrame({"sample_id": samples["sample_id"].astype("string")})
    return order.merge(prediction, on="sample_id", validate="one_to_one", sort=False)


def _compare_e00(reference: Path, generated: Path, unit_id: str) -> dict[str, object]:
    old_raw_path = reference / unit_id / "predictions_raw.csv"
    old_csv_path = reference / unit_id / "predictions.csv"
    if not old_raw_path.is_file() or not old_csv_path.is_file():
        raise ContractError(f"E00 reference is missing outputs for {unit_id}")
    old = pd.read_csv(old_raw_path, dtype={"sample_id": "string"})
    new = pd.read_csv(generated / "predictions_raw.csv", dtype={"sample_id": "string"})
    aligned = old.merge(new, on="sample_id", suffixes=("_old", "_new"), validate="one_to_one")
    if len(aligned) != len(old) or len(aligned) != len(new):
        raise ContractError(f"E00 reference sample IDs differ for {unit_id}")
    differences = [
        np.max(np.abs(aligned[f"{column}_old"] - aligned[f"{column}_new"]))
        for column in ("pred_tap_iron", "pred_tap_time_len")
    ]
    max_diff = float(max(differences))
    csv_equal = old_csv_path.read_bytes() == (generated / "predictions.csv").read_bytes()
    return {
        "raw_prediction_max_abs_diff": max_diff,
        "csv_byte_equality": csv_equal,
        "pass": max_diff == 0.0 and csv_equal,
    }


def _load_inputs(
    *,
    data_cfg: dict[str, Any],
    semantic_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
    protection: Any,
    requested_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
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
            "sample_id", "tap_no", "spout_no", history_mapping["event_time_column"],
            history_mapping["end_time_column"],
        ],
        required_columns=[
            "sample_id", "tap_no", "spout_no", history_mapping["event_time_column"],
            history_mapping["end_time_column"],
        ],
        time_columns=[history_mapping["event_time_column"], history_mapping["end_time_column"]],
    ).rename(columns={history_mapping["event_time_column"]: "reference_time"})
    history_meta["available_at"] = history_meta[history_mapping["end_time_column"]]
    metadata_consistency = validate_cross_table_metadata(train_meta, history_meta)

    labels = read_development_labels(
        data_cfg["paths"]["train_samples"],
        requested_end=requested_end,
        protection_policy=protection,
    )
    validate_samples(labels, labeled=True)
    history = read_development_history(
        data_cfg["paths"]["tap_history_train"],
        requested_end=requested_end,
        protection_policy=protection,
        available_at_column=history_mapping["available_at_column"],
    ).copy()
    history["available_at"] = history[history_mapping["available_at_column"]]
    validate_history(history, end_time=history_mapping["end_time_column"], available_at="available_at")
    target_consistency = validate_cross_table_consistency(labels, history)
    target_available_at = semantic_cfg["targets"]["available_at_column"]
    availability = read_csv(
        data_cfg["paths"]["tap_history_train"],
        usecols=["sample_id", target_available_at],
        required_columns=["sample_id", target_available_at],
        time_columns=[target_available_at],
    ).rename(columns={target_available_at: "label_available_at"})
    labels = labels.merge(availability, on="sample_id", how="left", validate="one_to_one")
    if labels["label_available_at"].isna().any():
        raise ContractError("training samples lack target availability mapping")

    op_map = semantic_cfg["sources"]["operation"]
    operation = normalize_event_source(
        read_csv(data_cfg["paths"]["operation_hourly"]),
        event_time_column=op_map["event_time_column"],
        available_at_column=op_map["available_at_column"],
        value_columns=list(feature_cfg["operation"]["value_columns"]),
        missing_markers=op_map["missing_markers"],
    )
    burden_map = semantic_cfg["sources"]["burden"]
    burden = normalize_event_source(
        read_csv(data_cfg["paths"]["burden_change"]),
        event_time_column=burden_map["event_time_column"],
        available_at_column=burden_map["available_at_column"],
        value_columns=list(feature_cfg["burden"]["value_columns"]),
        missing_markers=burden_map["missing_markers"],
    )
    consistency = {"metadata": metadata_consistency, "targets": target_consistency}
    return labels, history, operation, burden, consistency


def _group_units(units: list[EvaluationUnit]) -> list[list[EvaluationUnit]]:
    groups: dict[tuple[pd.Timestamp, pd.Timestamp], list[EvaluationUnit]] = defaultdict(list)
    for unit in units:
        groups[(unit.fit_cutoff, unit.train_start)].append(unit)
    return [sorted(values, key=lambda value: value.eval_start) for values in groups.values()]


def run_optimization_validation(
    *,
    data_config_path: str | Path,
    baseline_config_path: str | Path,
    feature_config_path: str | Path,
    semantic_contract_path: str | Path,
    protection_policy_path: str | Path,
    protection_ledger_path: str | Path | None,
    experiment_config_path: str | Path,
    optimization_feature_config_path: str | Path,
    optimization_validation_config_path: str | Path,
    optimization_acceptance_config_path: str | Path,
    optimization_model_config_path: str | Path,
    suite: str,
    output: str | Path,
    e00_reference: str | Path | None = None,
    candidate_ids: list[str] | None = None,
) -> Path:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    try:
        if suite not in {"screening", "grid", "all"}:
            raise ContractError("optimization suite must be screening, grid, or all")
        data_cfg = load_yaml(data_config_path)
        validate_data_paths(data_cfg, command="development")
        semantic_cfg = load_yaml(semantic_contract_path)
        validate_semantic_contract(semantic_cfg)
        protection = load_protection_policy(protection_policy_path)
        baseline_cfg = load_yaml(baseline_config_path)
        validate_baseline_config(baseline_cfg)
        feature_cfg = load_yaml(feature_config_path)
        validate_feature_config(feature_cfg)
        contract_digests = validate_frozen_contracts(baseline_cfg, feature_cfg, semantic_cfg)
        experiment_cfg, candidates = load_experiment(experiment_config_path)
        if candidate_ids:
            requested_candidates = set(candidate_ids)
            known_candidates = {candidate.id for candidate in candidates}
            unknown_candidates = requested_candidates - known_candidates
            if unknown_candidates:
                raise ContractError(f"unknown optimization candidates: {sorted(unknown_candidates)}")
            if "E00" not in requested_candidates:
                raise ContractError("candidate subset must include E00 as the frozen comparison")
            candidates = [candidate for candidate in candidates if candidate.id in requested_candidates]
            selected_ids = {candidate.id for candidate in candidates}
            for candidate in candidates:
                if candidate.base_candidate and candidate.base_candidate not in selected_ids:
                    raise ContractError(
                        f"{candidate.id}: selected subset omits base candidate {candidate.base_candidate}"
                    )
        selection_cfg = load_feature_selection(optimization_feature_config_path)
        model_cfg = load_model_config(optimization_model_config_path)
        validation_cfg, screening, origins = load_validation(
            optimization_validation_config_path, expected_timezone=protection.timezone
        )
        acceptance_cfg = load_acceptance(optimization_acceptance_config_path)
        declared_base = Path(selection_cfg["base_feature_config"])
        if declared_base.resolve() != Path(feature_config_path).resolve():
            raise ContractError("optimization feature selection points to a different baseline feature config")
        declared_model_base = Path(model_cfg["baseline_config"])
        if declared_model_base.resolve() != Path(baseline_config_path).resolve():
            raise ContractError("optimization model points to a different baseline config")
        if experiment_cfg["baseline_id"] != baseline_cfg["baseline_id"]:
            raise ContractError("optimization experiment baseline_id differs from frozen baseline")
        grid_units = [
            unit
            for origin in origins
            for unit in units_for_origin(origin, pd.Timestamp(validation_cfg["train_start"]))
        ]
        units = (
            screening if suite == "screening" else grid_units if suite == "grid" else [*screening, *grid_units]
        )
        requested_end = max(unit.eval_end for unit in units)
        protection.validate_development_read(requested_end)
        resolved = {
            "data": data_cfg,
            "semantic": semantic_cfg,
            "baseline": baseline_cfg,
            "baseline_features": feature_cfg,
            "experiment": experiment_cfg,
            "optimization_features": selection_cfg,
            "optimization_validation": validation_cfg,
            "optimization_acceptance": acceptance_cfg,
            "optimization_model": model_cfg,
            "suite": suite,
            "selected_candidates": [candidate.id for candidate in candidates],
        }
        atomic_write_json(destination / "resolved_config.json", resolved)
        used_paths = {
            key: data_cfg["paths"][key]
            for key in ("train_samples", "tap_history_train", "operation_hourly", "burden_change", "data_dictionary")
        }
        inputs_before = file_identities(used_paths)
        code = code_identity(Path.cwd())
        environment = runtime_environment()
        ledger = read_access_ledger(protection_ledger_path, protection)
        atomic_write_json(
            destination / "run_state.json",
            {
                "status": "RUNNING",
                "optimization_id": "optimization-v0.2",
                "suite": suite,
                "inputs_before": inputs_before,
                "code_identity": code,
                "environment": environment,
                "lockfile_sha256": file_sha256("uv.lock"),
                "resolved_config_sha256": stable_digest(resolved),
                "contract_digests": contract_digests,
                "protection_ledger": ledger,
                "cache": {"enabled": False, "reason": "OPT-01/02 does not use feature cache"},
            },
        )

        labels, history, operation, burden, consistency = _load_inputs(
            data_cfg=data_cfg,
            semantic_cfg=semantic_cfg,
            feature_cfg=feature_cfg,
            protection=protection,
            requested_end=requested_end,
        )
        metrics: dict[str, dict[str, object]] = {}
        registry: list[dict[str, object]] = []
        long_predictions: list[pd.DataFrame] = []
        e00_checks: dict[str, object] = {}
        model_candidates = [candidate for candidate in candidates if candidate.kind == "model"]
        derived_candidates = [candidate for candidate in candidates if candidate.kind != "model"]

        for grouped_units in _group_units(units):
            first = grouped_units[0]
            train, _, split = select_partitions(labels, first)
            eval_start = min(unit.eval_start for unit in grouped_units)
            eval_end = max(unit.eval_end for unit in grouped_units)
            eval_samples = labels.loc[
                (labels["reference_time"] >= eval_start) & (labels["reference_time"] < eval_end)
            ].copy()
            if eval_samples.empty:
                raise ContractError(f"{first.origin_id}: empty combined evaluation partition")
            train_result = build_features(
                train, operation=operation, burden=burden, history=history,
                fit_cutoff=first.fit_cutoff, config=feature_cfg,
            )
            eval_result = build_features(
                eval_samples, operation=operation, burden=burden, history=history,
                fit_cutoff=first.fit_cutoff, config=feature_cfg,
            )
            train_features = add_process_change_features(
                train_result.X,
                selection_cfg["process_change"],
                baseline_value_columns=list(feature_cfg["operation"]["value_columns"]),
            )
            eval_features = add_process_change_features(
                eval_result.X,
                selection_cfg["process_change"],
                baseline_value_columns=list(feature_cfg["operation"]["value_columns"]),
            )
            controls = MedianControls(
                min_group_count=int(baseline_cfg["controls"]["per_spout"]["min_group_count"])
            ).fit(train)
            b0_all = controls.predict(eval_samples, "B0")
            b1_all = controls.predict(eval_samples, "B1")
            raw_by_candidate: dict[str, pd.DataFrame] = {}
            final_by_candidate: dict[str, pd.DataFrame] = {}
            feature_columns: dict[str, list[str]] = {}
            fit_seconds: dict[str, float] = {}
            fit_rows: dict[str, int] = {}
            training_history_identity: dict[str, str] = {}
            sample_weight_policy: dict[str, str] = {}
            history_snapshot = freeze_history_origin(history, first.fit_cutoff)
            history_origin_sha256 = stable_digest(
                history_snapshot[["sample_id", "available_at"]].astype(str).to_dict("records")
            )
            augmented_cache: dict[
                tuple[int, ...], tuple[HistoryViewSet, pd.DataFrame, str]
            ] = {}
            for candidate in model_candidates:
                if len(candidate.history_view_ages_days) == 1:
                    candidate_train_features = train_features
                    candidate_targets = train[["tap_iron", "tap_time_len"]]
                    weights = None
                    training_identity = history_origin_sha256
                    weight_policy = "uniform_original_samples"
                else:
                    ages = candidate.history_view_ages_days
                    cached = augmented_cache.get(ages)
                    if cached is None:
                        views = build_history_views(train, ages)
                        augmented_result = build_features(
                            views.samples,
                            operation=operation,
                            burden=burden,
                            history=history,
                            fit_cutoff=first.fit_cutoff,
                            config=feature_cfg,
                        )
                        augmented_features = add_process_change_features(
                            augmented_result.X,
                            selection_cfg["process_change"],
                            baseline_value_columns=list(
                                feature_cfg["operation"]["value_columns"]
                            ),
                        )
                        history_audit = augmented_result.audit["history"]
                        if (
                            history_audit["max_available_at"].notna()
                            & (history_audit["max_available_at"] > history_audit["history_origin"])
                        ).any():
                            raise ContractError("synthetic view exposed history after its history_origin")
                        view_audit = views.audit.copy()
                        view_audit["history_visible_count"] = history_audit[
                            "history__visible_count"
                        ].to_numpy()
                        view_audit["history_identity_sha256"] = history_audit[
                            "history_identity_sha256"
                        ].to_numpy()
                        training_identity = stable_digest(
                            view_audit.astype(str).to_dict("records")
                        )
                        audit_dir = destination / "training_views" / first.origin_id
                        audit_dir.mkdir(parents=True, exist_ok=True)
                        ages_name = "ages_" + "_".join(str(age) for age in ages)
                        view_audit.to_csv(audit_dir / f"{ages_name}.csv", index=False)
                        cached = (views, augmented_features, training_identity)
                        augmented_cache[ages] = cached
                    views, candidate_train_features, training_identity = cached
                    candidate_targets = views.targets
                    weights = views.sample_weight
                    weight_policy = "total_one_per_original_sample"
                X_train = select_candidate_features(candidate_train_features, candidate, selection_cfg)
                X_eval = select_candidate_features(eval_features, candidate, selection_cfg)
                if list(X_train.columns) != list(X_eval.columns):
                    raise ContractError(f"{candidate.id}: train/eval feature columns differ")
                fit_started = perf_counter()
                model = FrozenBaselineAdapter(
                    dict(baseline_cfg["parameters"]), tuple(baseline_cfg["categorical_features"])
                ).fit(X_train, candidate_targets, sample_weight=weights)
                fit_seconds[candidate.id] = perf_counter() - fit_started
                fit_rows[candidate.id] = len(X_train)
                training_history_identity[candidate.id] = training_identity
                sample_weight_policy[candidate.id] = weight_policy
                raw = _prediction_frame(eval_samples["sample_id"], model.predict_raw(X_eval))
                final = raw.copy()
                for column in ("pred_tap_iron", "pred_tap_time_len"):
                    final[column] = final[column].clip(lower=0.0)
                raw_by_candidate[candidate.id] = raw
                final_by_candidate[candidate.id] = final
                feature_columns[candidate.id] = list(X_train.columns)
            for candidate in derived_candidates:
                if candidate.kind != "shrink_b1" or candidate.weights is None or candidate.base_candidate is None:
                    raise ContractError(f"unsupported derived candidate: {candidate.id}")
                base = final_by_candidate.get(candidate.base_candidate)
                if base is None:
                    raise ContractError(f"{candidate.id}: base candidate has not been fitted")
                final = shrink_toward_b1(base, b1_all, candidate.weights)
                raw_by_candidate[candidate.id] = final.copy()
                final_by_candidate[candidate.id] = final
                feature_columns[candidate.id] = feature_columns[candidate.base_candidate]
                fit_seconds[candidate.id] = 0.0
                fit_rows[candidate.id] = fit_rows[candidate.base_candidate]
                training_history_identity[candidate.id] = training_history_identity[
                    candidate.base_candidate
                ]
                sample_weight_policy[candidate.id] = sample_weight_policy[candidate.base_candidate]

            for unit in grouped_units:
                _, actual, unit_split = select_partitions(labels, unit)
                full_features = eval_result.X.loc[actual.index]
                unit_dir = destination / "units" / unit.id
                unit_dir.mkdir(parents=True)
                unit_metrics: dict[str, object] = {
                    "origin_id": unit.origin_id,
                    "horizon": unit.horizon,
                    "fit_cutoff": str(unit.fit_cutoff),
                    "eval_start": str(unit.eval_start),
                    "eval_end": str(unit.eval_end),
                    "split": unit_split,
                    "candidates": {},
                }
                b0 = _prediction_part(b0_all, actual)
                b1 = _prediction_part(b1_all, actual)
                for control_id, prediction in (("B0", b0), ("B1", b1)):
                    unit_metrics["candidates"][control_id] = diagnostic_metrics(actual, prediction, full_features)
                for candidate in candidates:
                    candidate_dir = unit_dir / candidate.id
                    candidate_dir.mkdir()
                    raw = _prediction_part(raw_by_candidate[candidate.id], actual)
                    prediction = _prediction_part(final_by_candidate[candidate.id], actual)
                    raw.to_csv(candidate_dir / "predictions_raw.csv", index=False)
                    write_submission(prediction, actual["sample_id"], candidate_dir / "predictions.csv")
                    candidate_metrics = diagnostic_metrics(actual, prediction, full_features)
                    errors = error_contributions(actual, prediction)
                    errors.nlargest(20, "abs_error_tap_iron").to_csv(
                        candidate_dir / "top20_iron_errors.csv", index=False
                    )
                    errors.nlargest(20, "abs_error_tap_time_len").to_csv(
                        candidate_dir / "top20_time_errors.csv", index=False
                    )
                    weekly = errors.groupby("week", sort=True).agg(
                        n=("sample_id", "size"),
                        iron_abs_error_sum=("abs_error_tap_iron", "sum"),
                        iron_signed_error_sum=("error_tap_iron", "sum"),
                        time_abs_error_sum=("abs_error_tap_time_len", "sum"),
                        time_signed_error_sum=("error_tap_time_len", "sum"),
                    ).reset_index()
                    weekly.to_csv(candidate_dir / "errors_by_week.csv", index=False)
                    if candidate.id == "E00":
                        candidate_metrics["shrinkage_grid"] = shrinkage_diagnostics(
                            actual, prediction, b1, list(experiment_cfg["shrinkage_grid"])
                        )
                        if e00_reference is not None and unit.horizon == 0:
                            e00_checks[unit.id] = _compare_e00(
                                Path(e00_reference), candidate_dir, unit.id
                            )
                    unit_metrics["candidates"][candidate.id] = candidate_metrics
                    overall = candidate_metrics["overall"]
                    registry.append(
                        {
                            "candidate_id": candidate.id,
                            "origin_id": unit.origin_id,
                            "unit_id": unit.id,
                            "horizon": unit.horizon,
                            "fit_cutoff": str(unit.fit_cutoff),
                            "history_policy": (
                                "synthetic_frozen_history_views"
                                if len(candidate.history_view_ages_days) > 1
                                else "baseline_origin_freeze"
                            ),
                            "history_view_ages_days": list(candidate.history_view_ages_days),
                            "history_origin_sha256": training_history_identity[candidate.id],
                            "feature_columns_sha256": stable_digest(feature_columns[candidate.id]),
                            "model_parameters_sha256": stable_digest(baseline_cfg["parameters"]),
                            "tap_iron_parameters_sha256": stable_digest(baseline_cfg["parameters"]),
                            "tap_time_len_parameters_sha256": stable_digest(baseline_cfg["parameters"]),
                            "sample_weight_policy": sample_weight_policy[candidate.id],
                            "code_source_snapshot_sha256": code["source_snapshot_sha256"],
                            "input_identities_sha256": stable_digest(inputs_before),
                            "resolved_config_sha256": stable_digest(resolved),
                            "eligible_train_rows": unit_split["eligible_train_rows"],
                            "fit_train_rows": fit_rows[candidate.id],
                            "eval_rows": unit_split["evaluation_rows"],
                            "iron_abs_error_sum": overall["iron"]["abs_error_sum"],
                            "iron_actual_sum": overall["iron"]["actual_sum"],
                            "time_abs_error_sum": overall["time"]["abs_error_sum"],
                            "time_actual_sum": overall["time"]["actual_sum"],
                            "loss": overall["loss"],
                            "fit_seconds_for_origin": fit_seconds[candidate.id],
                        }
                    )
                    long = prediction.copy()
                    long.insert(0, "origin_id", unit.origin_id)
                    long.insert(0, "candidate_id", candidate.id)
                    long_predictions.append(long)
                atomic_write_json(unit_dir / "metrics.json", unit_metrics)
                metrics[unit.id] = unit_metrics

        verify_file_identities(inputs_before)
        write_registry(destination, registry)
        predictions = pd.concat(long_predictions, ignore_index=True)
        if predictions.duplicated(["candidate_id", "origin_id", "sample_id"]).any():
            raise ContractError("prediction primary key (candidate_id, origin_id, sample_id) is not unique")
        predictions.to_csv(destination / "predictions_long.csv", index=False)
        atomic_write_json(destination / "metrics.json", metrics)
        grid_summary = aggregate_grid(metrics) if suite in {"grid", "all"} else {}
        atomic_write_json(destination / "grid_summary.json", grid_summary)
        acceptance = evaluate_acceptance(metrics, grid_summary, acceptance_cfg)
        atomic_write_json(destination / "acceptance.json", acceptance)
        model_quality_status = {
            "PASS": "G1_PASS_OPTIMIZATION_DEVELOPMENT",
            "FAIL": "G1_FAIL_OPTIMIZATION_ACCEPTANCE",
            "INCOMPLETE": "G1_NOT_EVALUATED_FULL_GATE",
        }[acceptance["status"]]
        comparison_rows = []
        for unit_id, unit_values in metrics.items():
            for candidate_id, values in unit_values["candidates"].items():
                overall = values["overall"]
                comparison_rows.append(
                    {
                        "unit_id": unit_id,
                        "origin_id": unit_values["origin_id"],
                        "horizon": unit_values["horizon"],
                        "candidate_id": candidate_id,
                        "n": overall["iron"]["n"],
                        "iron_wmape": overall["iron"]["wmape"],
                        "time_wmape": overall["time"]["wmape"],
                        "loss": overall["loss"],
                        "score": overall["score"],
                    }
                )
        pd.DataFrame(comparison_rows).to_csv(destination / "candidate_comparison.csv", index=False)
        e00_status = {
            "status": "NOT_REQUESTED" if e00_reference is None else "PASS",
            "reference": str(e00_reference) if e00_reference is not None else None,
            "checks": e00_checks,
        }
        if e00_reference is not None and (
            set(e00_checks) != {unit.id for unit in screening if unit in units}
            or not all(check["pass"] for check in e00_checks.values())
        ):
            e00_status["status"] = "FAIL"
            raise ContractError("E00 does not exactly reproduce the supplied baseline run")
        atomic_write_json(destination / "e00_reproduction.json", e00_status)
        manifest = {
            "status": "PASS_EXECUTION",
            "quality_status": acceptance["status"],
            "engineering_status": "G0_OPT_EXECUTION_PASS",
            "model_quality_status": model_quality_status,
            "optimization_id": "optimization-v0.2",
            "suite": suite,
            "evaluation_units": len(units),
            "candidate_count": len(candidates),
            "cross_table_consistency": consistency,
            "input_identities": inputs_before,
            "inputs_stable": True,
            "code_identity": code,
            "resolved_config_sha256": stable_digest(resolved),
            "contract_digests": contract_digests,
            "protection": {
                "policy_id": protection.contract_id,
                "development_read_end_exclusive": str(requested_end),
                "ledger": ledger,
                "protected_labels_read": False,
            },
            "e00_reproduction": e00_status,
            "optimization_acceptance": acceptance,
            "walltime_seconds": perf_counter() - started,
        }
        atomic_write_json(destination / "run_manifest.json", manifest)
        atomic_write_json(
            destination / "final_status.json",
            {
                "status": "PASS_EXECUTION",
                "engineering_status": manifest["engineering_status"],
                "model_quality_status": manifest["model_quality_status"],
                "run_manifest_sha256": file_sha256(destination / "run_manifest.json"),
            },
        )
        return destination
    except Exception as exc:
        if not (destination / "final_status.json").exists():
            atomic_write_json(
                destination / "final_status.json",
                {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)},
            )
        raise
