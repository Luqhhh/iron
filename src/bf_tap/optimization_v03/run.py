from __future__ import annotations

import json
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
from ..config import (
    load_yaml,
    validate_baseline_config,
    validate_feature_config,
    validate_frozen_contracts,
    validate_data_paths,
    validate_semantic_contract,
)
from ..exceptions import ContractError
from ..io import read_csv, read_development_labels
from ..models.sanity import MedianControls
from ..optimization import run_optimization_validation
from ..optimization.config import (
    EvaluationUnit,
    load_acceptance,
    load_experiment,
    load_feature_selection,
    load_model_config,
    load_validation,
)
from ..optimization.validation import (
    aggregate_grid,
    evaluate_acceptance,
    select_partitions,
    units_for_origin,
)
from ..protection import load_protection_policy, read_access_ledger
from ..schema import validate_samples
from .candidates import (
    derive_stage1_predictions,
    derive_stage2_prediction,
    recent_target_medians,
)
from .config import DriftExperiment, load_drift_experiment
from .diagnostics import (
    diagnostic_metrics,
    paired_daily_block_bootstrap,
    scenario_summary,
)

_TARGETS = ("tap_iron", "tap_time_len")
_PREDICTION_COLUMNS = (
    "sample_id",
    "pred_tap_iron",
    "pred_tap_time_len",
)


_CANONICAL_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL_PROTECTION_END = pd.Timestamp("2024-11-01T00:00:00+08:00")
_CANONICAL_DEPENDENCIES = {
    "data_contract": ("configs/data_contract.yaml", "d9b8f064ba4c75e50eef723c3d8304e192c8191d69738581de33c49fda45e7b2"),
    "baseline": ("configs/baseline.yaml", "a4d2a332f0f17cfb9de9dec9a059d337e2fc6ed4d14d99930f7e4b4b2f8ad1b3"),
    "feature": ("configs/features.yaml", "b43a7ca7e295a948f795630b5836f9ade5b3b369b116f531b1ae302a852e65d4"),
    "protection": ("configs/protection.yaml", "4bc88981583c0307de485bb422a38cb760ae0470fb5114a309dce1b1ade7e638"),
    "drift_experiment": ("configs/optimization_v0_3/experiment.yaml", "ff9fc2273328271e15905bc434e7f3a9ab4308398f5bf9255634c3c8b22dca12"),
    "core_experiment": ("configs/optimization_v0_2/experiment.yaml", "118ef07f5f5dbee407b60bfe2e7f47b161c92cc9b37a79d4cf62afebde99087a"),
    "optimization_feature": ("configs/optimization_v0_2/features.yaml", "1cbdb6c2b32f3c07c8135ca103317c222acf2092b945a95b6380b51af96d059a"),
    "optimization_validation": ("configs/optimization_v0_2/validation.yaml", "e94ac091454ebb76d0d8d8e4c2d3682e5762d10d4f42665445bb3460ff0bfbd9"),
    "optimization_acceptance": ("configs/optimization_v0_2/acceptance.yaml", "cb390e274235a9c4a85cb173ebd89977d4d15ed6ab719ccd5624fac41bbb7fc8"),
    "optimization_model": ("configs/optimization_v0_2/models/catboost.yaml", "28e3ca6cf4000c92d5ca3cc4506ad5cce49ce4c54968d9cbce9fc58de021071f"),
}


def _canonical_dependency_path(name: str, supplied: str | Path) -> Path:
    relative, expected_digest = _CANONICAL_DEPENDENCIES[name]
    canonical = (_CANONICAL_ROOT / relative).resolve()
    raw = Path(supplied)
    actual = (raw if raw.is_absolute() else _CANONICAL_ROOT / raw).resolve()
    if actual != canonical:
        raise ContractError(f"canonical {name} path is required")
    if not actual.is_file() or file_sha256(actual) != expected_digest:
        raise ContractError(f"{name} differs from the frozen canonical contract")
    return actual


def validate_v03_dependencies(
    *,
    data_contract_path: str | Path,
    baseline_config_path: str | Path,
    feature_config_path: str | Path,
    protection_policy_path: str | Path,
    drift_experiment_config_path: str | Path,
    core_experiment_config_path: str | Path,
    optimization_feature_config_path: str | Path,
    optimization_validation_config_path: str | Path,
    optimization_acceptance_config_path: str | Path,
    optimization_model_config_path: str | Path,
) -> dict[str, Path]:
    supplied = {
        "data_contract": data_contract_path,
        "baseline": baseline_config_path,
        "feature": feature_config_path,
        "protection": protection_policy_path,
        "drift_experiment": drift_experiment_config_path,
        "core_experiment": core_experiment_config_path,
        "optimization_feature": optimization_feature_config_path,
        "optimization_validation": optimization_validation_config_path,
        "optimization_acceptance": optimization_acceptance_config_path,
        "optimization_model": optimization_model_config_path,
    }
    paths = {
        name: _canonical_dependency_path(name, path)
        for name, path in supplied.items()
    }
    protection = load_protection_policy(paths["protection"])
    if protection.development_label_end_exclusive != _CANONICAL_PROTECTION_END:
        raise ContractError(
            "protection development boundary is not the canonical 2024-11-01 boundary"
        )
    baseline_cfg = load_yaml(paths["baseline"])
    feature_cfg = load_yaml(paths["feature"])
    semantic_cfg = load_yaml(paths["data_contract"])
    validate_frozen_contracts(baseline_cfg, feature_cfg, semantic_cfg)
    load_experiment(paths["core_experiment"])
    optimization_features = load_feature_selection(paths["optimization_feature"])
    if optimization_features["base_feature_config"] != "configs/features.yaml":
        raise ContractError("optimization feature config points to a different feature contract")
    load_validation(
        paths["optimization_validation"], expected_timezone=protection.timezone
    )
    load_acceptance(paths["optimization_acceptance"])
    optimization_model = load_model_config(paths["optimization_model"])
    if optimization_model["baseline_config"] != "configs/baseline.yaml":
        raise ContractError("optimization model config points to a different baseline contract")
    return paths


def normalize_stage1_acceptance(
    acceptance: dict[str, Any],
    experiment: DriftExperiment,
) -> dict[str, Any]:
    rows = acceptance.get("candidates")
    if not isinstance(rows, dict):
        raise ContractError("stage1 acceptance lacks candidate rows")
    stage1_ids = {
        experiment.candidate_ids["c1"],
        experiment.candidate_ids["c2"],
    }
    status = "PASS" if any(
        isinstance(rows.get(candidate_id), dict)
        and rows[candidate_id].get("pass") is True
        for candidate_id in stage1_ids
    ) else "FAIL"
    return {**acceptance, "status": status}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stage2_decision(
    acceptance: dict[str, Any],
    experiment: DriftExperiment,
) -> dict[str, object]:
    rows = acceptance.get("candidates")
    if not isinstance(rows, dict):
        raise ContractError("stage1 acceptance lacks candidate rows")
    failed = [
        candidate_id
        for candidate_id in experiment.stage2_requires_all
        if not isinstance(rows.get(candidate_id), dict)
        or rows[candidate_id].get("pass") is not True
    ]
    return {
        "status": "RUN" if not failed else "SKIPPED",
        "candidate_id": experiment.candidate_ids["stage2"],
        "requires_all": list(experiment.stage2_requires_all),
        "failed_prerequisites": failed,
        "route": {
            "tap_iron": experiment.candidate_ids["c2"],
            "tap_time_len": experiment.candidate_ids["c1"],
        },
    }


def quality_status(
    stage1_acceptance: dict[str, Any],
    experiment: DriftExperiment,
) -> str:
    rows = stage1_acceptance.get("candidates")
    if not isinstance(rows, dict):
        raise ContractError("stage1 acceptance lacks candidate rows")
    passed = any(
        isinstance(rows.get(experiment.candidate_ids[key]), dict)
        and rows[experiment.candidate_ids[key]].get("pass") is True
        for key in ("c1", "c2")


    )
    return (
        "G1_PASS_OPTIMIZATION_DEVELOPMENT"
        if passed
        else "G1_FAIL_OPTIMIZATION_ACCEPTANCE"
    )


def assert_c2_equivalence(
    c2: pd.DataFrame,
    e09: pd.DataFrame,
) -> dict[str, float | int | bool]:
    required = set(_PREDICTION_COLUMNS)
    if required - set(c2) or required - set(e09):
        raise ContractError("C2/E09 prediction columns differ")
    left = c2[list(_PREDICTION_COLUMNS)].copy()
    right = e09[list(_PREDICTION_COLUMNS)].copy()
    left["sample_id"] = left["sample_id"].astype("string")
    right["sample_id"] = right["sample_id"].astype("string")
    if (
        left["sample_id"].isna().any()
        or right["sample_id"].isna().any()
        or left["sample_id"].duplicated().any()
        or right["sample_id"].duplicated().any()
        or set(left["sample_id"]) != set(right["sample_id"])
    ):
        raise ContractError("C2/E09 sample_id sets differ")
    aligned = left.merge(
        right,
        on="sample_id",
        suffixes=("_c2", "_e09"),
        validate="one_to_one",
    )
    differences = [
        np.max(
            np.abs(
                aligned[f"{column}_c2"].to_numpy(dtype=float)
                - aligned[f"{column}_e09"].to_numpy(dtype=float)
            )
        )
        for column in ("pred_tap_iron", "pred_tap_time_len")
    ]
    max_diff = float(max(differences))
    if max_diff != 0.0:
        raise ContractError(f"C2 differs from E09; max_abs_diff={max_diff}")
    return {"rows": len(aligned), "max_abs_diff": max_diff, "pass": True}


def _safe_development_labels(
    *,
    data_cfg: dict[str, Any],
    semantic_cfg: dict[str, Any],
    protection: Any,
    requested_end: pd.Timestamp,
) -> pd.DataFrame:
    protection.validate_development_read(requested_end)
    labels = read_development_labels(
        data_cfg["paths"]["train_samples"],
        requested_end=requested_end,
        protection_policy=protection,
    )
    validate_samples(labels, labeled=True)
    available_at = semantic_cfg["targets"]["available_at_column"]
    availability = read_csv(
        data_cfg["paths"]["tap_history_train"],
        usecols=["sample_id", available_at],
        required_columns=["sample_id", available_at],
        time_columns=[available_at],
    ).rename(columns={available_at: "label_available_at"})
    labels = labels.merge(
        availability,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if labels["label_available_at"].isna().any():
        raise ContractError("development labels lack label availability")
    return labels


def _all_units(
    screening: list[EvaluationUnit],
    origins: list[Any],
    train_start: pd.Timestamp,
) -> list[EvaluationUnit]:
    return [
        *screening,
        *[
            unit
            for origin in origins
            for unit in units_for_origin(origin, train_start)
        ],
    ]


def _group_units(
    units: list[EvaluationUnit],
) -> dict[str, list[EvaluationUnit]]:
    grouped: dict[str, list[EvaluationUnit]] = defaultdict(list)
    for unit in units:
        grouped[unit.origin_id].append(unit)
    return {
        origin_id: sorted(values, key=lambda unit: unit.eval_start)
        for origin_id, values in grouped.items()
    }


def _prediction_part(
    prediction: pd.DataFrame,
    actual: pd.DataFrame,
) -> pd.DataFrame:
    order = actual[["sample_id"]].copy()
    order["sample_id"] = order["sample_id"].astype("string")
    values = prediction[list(_PREDICTION_COLUMNS)].copy()
    values["sample_id"] = values["sample_id"].astype("string")
    result = order.merge(
        values,
        on="sample_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if len(result) != len(values) and len(actual) == len(prediction):
        raise ContractError("prediction rows do not align with evaluation rows")
    if result[["pred_tap_iron", "pred_tap_time_len"]].isna().any().any():
        raise ContractError("prediction sample_id set differs from evaluation rows")
    result["sample_id"] = actual["sample_id"].to_numpy(copy=True)
    return result


def _error_rows(
    actual: pd.DataFrame,
    prediction: pd.DataFrame,
) -> pd.DataFrame:
    result = actual[
        ["sample_id", "reference_time", "spout_no", *_TARGETS]
    ].merge(prediction, on="sample_id", validate="one_to_one")
    for target in _TARGETS:
        result[f"error_{target}"] = (
            result[f"pred_{target}"] - result[target]
        )
        result[f"abs_error_{target}"] = result[f"error_{target}"].abs()
    return result


def _comparison_rows(metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for unit_id, values in metrics.items():
        for candidate_id, candidate in values["candidates"].items():
            overall = candidate["overall"]
            rows.append(
                {
                    "unit_id": unit_id,
                    "origin_id": values["origin_id"],
                    "horizon": values["horizon"],
                    "candidate_id": candidate_id,
                    "n": overall["iron"]["n"],
                    "iron_wmape": overall["iron"]["wmape"],
                    "time_wmape": overall["time"]["wmape"],
                    "loss": overall["loss"],
                    "iron_signed_bias": overall["iron"]["signed_bias"],
                    "time_signed_bias": overall["time"]["signed_bias"],
                }
            )
    return pd.DataFrame(rows)


def run_drift_validation(
    *,
    data_config_path: str | Path,
    baseline_config_path: str | Path,
    feature_config_path: str | Path,
    semantic_contract_path: str | Path,
    protection_policy_path: str | Path,
    protection_ledger_path: str | Path | None,
    drift_experiment_config_path: str | Path,
    core_experiment_config_path: str | Path,
    optimization_feature_config_path: str | Path,
    optimization_validation_config_path: str | Path,
    optimization_acceptance_config_path: str | Path,
    optimization_model_config_path: str | Path,
    output: str | Path,
) -> Path:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    protected_labels_read: bool | str = "UNVERIFIED"
    safety_checkpoint = False
    try:
        experiment = load_drift_experiment(drift_experiment_config_path)
        validate_v03_dependencies(
            data_contract_path=semantic_contract_path,
            baseline_config_path=baseline_config_path,
            feature_config_path=feature_config_path,
            protection_policy_path=protection_policy_path,
            drift_experiment_config_path=drift_experiment_config_path,
            core_experiment_config_path=core_experiment_config_path,
            optimization_feature_config_path=optimization_feature_config_path,
            optimization_validation_config_path=optimization_validation_config_path,
            optimization_acceptance_config_path=optimization_acceptance_config_path,
            optimization_model_config_path=optimization_model_config_path,
        )
        if Path(experiment.core_experiment_config).resolve() != Path(
            core_experiment_config_path
        ).resolve():
            raise ContractError(
                "v0.3 experiment points to a different core experiment config"
            )
        data_cfg = load_yaml(data_config_path)
        validate_data_paths(data_cfg, command="development")
        semantic_cfg = load_yaml(semantic_contract_path)
        validate_semantic_contract(semantic_cfg)
        protection = load_protection_policy(protection_policy_path)
        validation_cfg, screening, origins = load_validation(
            optimization_validation_config_path,
            expected_timezone=protection.timezone,
        )
        acceptance_policy = load_acceptance(
            optimization_acceptance_config_path
        )
        train_start = pd.Timestamp(validation_cfg["train_start"])
        units = _all_units(screening, origins, train_start)
        requested_end = _CANONICAL_PROTECTION_END
        if max(unit.eval_end for unit in units) != requested_end:
            raise ContractError(
                "v0.3 full grid must end at the development protection boundary"
            )
        protection.validate_development_read(requested_end)

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
        inputs_before = file_identities(used_paths)
        code_before = code_identity(Path.cwd())
        ledger_before = read_access_ledger(
            protection_ledger_path,
            protection,
        )
        resolved = {
            "optimization_id": experiment.optimization_id,
            "evidence_scope": experiment.evidence_scope,
            "drift_experiment": load_yaml(drift_experiment_config_path),
            "core_experiment_path": str(core_experiment_config_path),
            "validation": validation_cfg,
            "acceptance": acceptance_policy,
            "requested_end_exclusive": str(requested_end),
        }
        atomic_write_json(destination / "resolved_config.json", resolved)
        atomic_write_json(
            destination / "run_state.json",
            {
                "status": "RUNNING",
                "optimization_id": experiment.optimization_id,
                "inputs_before": inputs_before,
                "code_identity": code_before,
                "environment": runtime_environment(),
                "lockfile_sha256": file_sha256("uv.lock"),
                "resolved_config_sha256": stable_digest(resolved),
                "protection_ledger_before": ledger_before,
                "protected_labels_read": protected_labels_read,
            },
        )

        core_output = destination / "core_v02"
        run_optimization_validation(
            data_config_path=data_config_path,
            baseline_config_path=baseline_config_path,
            feature_config_path=feature_config_path,
            semantic_contract_path=semantic_contract_path,
            protection_policy_path=protection_policy_path,
            protection_ledger_path=protection_ledger_path,
            experiment_config_path=core_experiment_config_path,
            optimization_feature_config_path=optimization_feature_config_path,
            optimization_validation_config_path=optimization_validation_config_path,
            optimization_acceptance_config_path=optimization_acceptance_config_path,
            optimization_model_config_path=optimization_model_config_path,
            suite="all",
            output=core_output,
            candidate_ids=list(experiment.core_candidate_ids),
        )
        core_final = _read_json(core_output / "final_status.json")
        core_manifest = _read_json(core_output / "run_manifest.json")
        if (
            core_final.get("status") != "PASS_EXECUTION"
            or core_manifest.get("protection", {}).get(
                "protected_labels_read"
            )
            is not False
        ):
            raise ContractError("v0.2 core run did not pass safe execution")

        labels = _safe_development_labels(
            data_cfg=data_cfg,
            semantic_cfg=semantic_cfg,
            protection=protection,
            requested_end=requested_end,
        )
        core_predictions = pd.read_csv(
            core_output / "predictions_long.csv",
            dtype={"sample_id": "string"},
        )
        required_core = {
            "candidate_id",
            "origin_id",
            *_PREDICTION_COLUMNS,
        }
        if required_core - set(core_predictions):
            raise ContractError("core predictions_long contract is incomplete")

        metrics: dict[str, dict[str, Any]] = {}
        unit_actual: dict[str, pd.DataFrame] = {}
        unit_predictions: dict[str, dict[str, pd.DataFrame]] = {}
        scenario_rows: list[pd.DataFrame] = []
        recent_audits: dict[str, dict[str, object]] = {}
        equivalence: dict[str, dict[str, float | int | bool]] = {}

        for origin_id, origin_units in _group_units(units).items():
            first = origin_units[0]
            train, _, _ = select_partitions(labels, first)
            eval_start = min(unit.eval_start for unit in origin_units)
            eval_end = max(unit.eval_end for unit in origin_units)
            evaluation = labels.loc[
                (labels["reference_time"] >= eval_start)
                & (labels["reference_time"] < eval_end)
            ].copy()
            controls = MedianControls(
                min_group_count=int(
                    load_yaml(baseline_config_path)["controls"][
                        "per_spout"
                    ]["min_group_count"]
                )
            ).fit(train)
            b0 = controls.predict(evaluation, "B0")
            b1 = controls.predict(evaluation, "B1")
            e00 = core_predictions.loc[
                (core_predictions["candidate_id"] == "E00")
                & (core_predictions["origin_id"] == origin_id),
                list(_PREDICTION_COLUMNS),
            ].copy()
            e09 = core_predictions.loc[
                (
                    core_predictions["candidate_id"]
                    == "E09_PROCESS_CHANGE_E02"
                )
                & (core_predictions["origin_id"] == origin_id),
                list(_PREDICTION_COLUMNS),
            ].copy()
            recent, recent_audit = recent_target_medians(
                labels,
                first.fit_cutoff,
                experiment.recent_window_days,
            )
            recent_audits[origin_id] = recent_audit
            stage1 = derive_stage1_predictions(
                evaluation,
                e00,
                b1,
                e09,
                recent,
                experiment,
            )
            equivalence[origin_id] = assert_c2_equivalence(
                stage1[experiment.candidate_ids["c2"]],
                e09,
            )
            origin_predictions = {
                "B0": b0,
                "B1": b1,
                "E00": e00,
                **stage1,
            }

            for unit in origin_units:
                _, actual, split = select_partitions(labels, unit)
                unit_actual[unit.id] = actual
                candidate_predictions = {
                    candidate_id: _prediction_part(prediction, actual)
                    for candidate_id, prediction in origin_predictions.items()
                }
                unit_predictions[unit.id] = candidate_predictions
                unit_metrics: dict[str, Any] = {
                    "origin_id": unit.origin_id,
                    "horizon": unit.horizon,
                    "fit_cutoff": str(unit.fit_cutoff),
                    "eval_start": str(unit.eval_start),
                    "eval_end": str(unit.eval_end),
                    "split": split,
                    "candidates": {},
                }
                for candidate_id, prediction in candidate_predictions.items():
                    candidate_dir = (
                        destination / "units" / unit.id / candidate_id
                    )
                    candidate_dir.mkdir(parents=True, exist_ok=False)
                    prediction.to_csv(
                        candidate_dir / "predictions.csv",
                        index=False,
                    )
                    errors = _error_rows(actual, prediction)
                    errors.to_csv(candidate_dir / "errors.csv", index=False)
                    unit_metrics["candidates"][candidate_id] = (
                        diagnostic_metrics(actual, prediction)
                    )
                    long = errors[
                        [
                            "sample_id",
                            "reference_time",
                            "spout_no",
                            *_TARGETS,
                            *_PREDICTION_COLUMNS[1:],
                        ]
                    ].copy()
                    long.insert(0, "horizon", unit.horizon)
                    long.insert(0, "origin_id", unit.origin_id)
                    long.insert(0, "unit_id", unit.id)
                    long.insert(0, "candidate_id", candidate_id)
                    scenario_rows.append(long)
                metrics[unit.id] = unit_metrics

        atomic_write_json(
            destination / "recent_window_audit.json",
            recent_audits,
        )
        max_equivalence = max(
            float(value["max_abs_diff"])
            for value in equivalence.values()
        )
        equivalence_summary = {
            "pass": max_equivalence == 0.0,
            "max_abs_diff": max_equivalence,
            "origins": equivalence,
        }
        atomic_write_json(
            destination / "c2_e09_equivalence.json",
            equivalence_summary,
        )

        stage1_grid = aggregate_grid(metrics)
        stage1_acceptance = evaluate_acceptance(
            metrics,
            stage1_grid,
            acceptance_policy,
        )
        stage1_acceptance = normalize_stage1_acceptance(stage1_acceptance, experiment)
        atomic_write_json(
            destination / "stage1_acceptance.json",
            stage1_acceptance,
        )
        decision = stage2_decision(stage1_acceptance, experiment)
        if decision["status"] == "RUN":
            stage2_id = experiment.candidate_ids["stage2"]
            for unit in units:
                prediction = derive_stage2_prediction(
                    unit_predictions[unit.id][
                        experiment.candidate_ids["c1"]
                    ],
                    unit_predictions[unit.id][
                        experiment.candidate_ids["c2"]
                    ],
                    experiment,
                )
                unit_predictions[unit.id][stage2_id] = prediction
                actual = unit_actual[unit.id]
                candidate_dir = (
                    destination / "units" / unit.id / stage2_id
                )
                candidate_dir.mkdir(parents=True, exist_ok=False)
                prediction.to_csv(
                    candidate_dir / "predictions.csv",
                    index=False,
                )
                errors = _error_rows(actual, prediction)
                errors.to_csv(candidate_dir / "errors.csv", index=False)
                metrics[unit.id]["candidates"][stage2_id] = (
                    diagnostic_metrics(actual, prediction)
                )
                long = errors[
                    [
                        "sample_id",
                        "reference_time",
                        "spout_no",
                        *_TARGETS,
                        *_PREDICTION_COLUMNS[1:],
                    ]
                ].copy()
                long.insert(0, "horizon", unit.horizon)
                long.insert(0, "origin_id", unit.origin_id)
                long.insert(0, "unit_id", unit.id)
                long.insert(0, "candidate_id", stage2_id)
                scenario_rows.append(long)
            grid_summary = aggregate_grid(metrics)
            acceptance = evaluate_acceptance(
                metrics,
                grid_summary,
                acceptance_policy,
            )
            decision = {
                **decision,
                "status": "EVALUATED",
                "acceptance": acceptance["candidates"][stage2_id],
            }
        else:
            grid_summary = stage1_grid
            acceptance = stage1_acceptance
        atomic_write_json(destination / "stage2.json", decision)

        for unit_id, values in metrics.items():
            atomic_write_json(
                destination / "units" / unit_id / "metrics.json",
                values,
            )
        rows = pd.concat(scenario_rows, ignore_index=True)
        rows.to_csv(
            destination / "predictions_with_actual.csv",
            index=False,
        )
        atomic_write_json(destination / "metrics.json", metrics)
        atomic_write_json(
            destination / "grid_summary.json",
            grid_summary,
        )
        atomic_write_json(
            destination / "scenario_summary.json",
            scenario_summary(rows),
        )
        atomic_write_json(
            destination / "acceptance.json",
            acceptance,
        )
        _comparison_rows(metrics).to_csv(
            destination / "candidate_comparison.csv",
            index=False,
        )

        bootstrap_rows = rows.loc[rows["horizon"] >= 1]
        bootstrap: dict[str, object] = {
            "scope": "grid_scenario_rows",
            "block": "unit_id_plus_local_calendar_day",
            "comparisons": {},
        }
        stage_candidates = [
            experiment.candidate_ids["c1"],
            experiment.candidate_ids["c2"],
        ]
        if decision["status"] == "EVALUATED":
            stage_candidates.append(experiment.candidate_ids["stage2"])
        for candidate_id in stage_candidates:
            for control_id in ("E00", "B0", "B1"):
                key = f"{candidate_id}__vs__{control_id}"
                bootstrap["comparisons"][key] = (
                    paired_daily_block_bootstrap(
                        bootstrap_rows,
                        candidate_id,
                        control_id,
                        repetitions=experiment.bootstrap_repetitions,
                        seed=experiment.bootstrap_seed,
                    )
                )
        atomic_write_json(destination / "bootstrap.json", bootstrap)

        verify_file_identities(inputs_before)
        code_after = code_identity(Path.cwd())
        if (
            code_after["source_snapshot_sha256"]
            != code_before["source_snapshot_sha256"]
        ):
            raise ContractError(
                "source configuration or implementation changed during run"
            )
        ledger_after = read_access_ledger(
            protection_ledger_path,
            protection,
        )
        if stable_digest(ledger_after) != stable_digest(ledger_before):
            raise ContractError(
                "protected access ledger changed during development run"
            )
        safety_checkpoint = True
        if not safety_checkpoint:
            raise ContractError("protected-label safety checkpoint was not verified")
        protected_labels_read = False
        model_quality_status = quality_status(
            stage1_acceptance,
            experiment,
        )
        manifest = {
            "status": "PASS_EXECUTION",
            "engineering_status": "G0_OPT_EXECUTION_PASS",
            "model_quality_status": model_quality_status,
            "optimization_id": experiment.optimization_id,
            "evidence_scope": experiment.evidence_scope,
            "development_evidence_seen": True,
            "evaluation_units": len(units),
            "scenario_rows": len(rows),
            "stage1_acceptance": stage1_acceptance,
            "stage2": decision,
            "c2_e09_equivalence": equivalence_summary,
            "core_manifest_sha256": file_sha256(
                core_output / "run_manifest.json"
            ),
            "input_identities": inputs_before,
            "inputs_stable": True,
            "code_identity": code_before,
            "source_snapshot_unchanged": True,
            "protection": {
                "policy_id": protection.contract_id,
                "development_read_end_exclusive": str(requested_end),
                "ledger_before": ledger_before,
                "ledger_unchanged": True,
                "protected_labels_read": protected_labels_read,
            },
            "walltime_seconds": perf_counter() - started,
        }
        atomic_write_json(destination / "run_manifest.json", manifest)
        atomic_write_json(
            destination / "final_status.json",
            {
                "status": "PASS_EXECUTION",
                "engineering_status": manifest["engineering_status"],
                "model_quality_status": model_quality_status,
                "stage2_status": decision["status"],
                "protected_labels_read": protected_labels_read,
                "run_manifest_sha256": file_sha256(
                    destination / "run_manifest.json"
                ),
            },
        )
        return destination
    except Exception as exc:
        if not (destination / "final_status.json").exists():
            atomic_write_json(
                destination / "final_status.json",
                {
                    "status": "FAILED",
                    "engineering_status": "G0_OPT_EXECUTION_FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "protected_labels_read": "UNVERIFIED",
                },
            )
        raise
