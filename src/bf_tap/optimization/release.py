from __future__ import annotations

import json
import resource
from pathlib import Path
from time import perf_counter

import pandas as pd

from ..artifacts import (
    atomic_write_json,
    build_inference_source_contract,
    code_identity,
    file_identities,
    file_sha256,
    runtime_environment,
    stable_digest,
    validate_inference_source_contract,
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
from ..exceptions import ContractError
from ..features import build_features
from ..io import read_csv
from ..models.baseline import DualTargetBaseline
from ..offline import _load_process_sources
from ..protection import load_protection_policy, read_access_ledger
from ..schema import validate_history, validate_samples
from ..submission import validate_submission, write_submission
from .ensemble import apply_global_residual_calibration, convex_blend
from .config import load_experiment, load_feature_selection, load_model_config
from .features import select_candidate_features
from .history_adaptation import build_history_views
from .models import FrozenBaselineAdapter
from .process_change import add_process_change_features
from .run import _load_inputs


def _candidate_payload(candidate) -> dict[str, object]:
    return {
        "id": candidate.id,
        "kind": candidate.kind,
        "sources": list(candidate.sources),
        "history_components": list(candidate.history_components),
        "history_view_ages_days": list(candidate.history_view_ages_days),
    }


def _derived_candidate_payload(candidate) -> dict[str, object]:
    return {
        "id": candidate.id,
        "kind": candidate.kind,
        "component_candidates": list(candidate.component_candidates),
        "base_candidate": candidate.base_candidate,
        "blend_weights": candidate.blend_weights,
        "residual_calibration": candidate.residual_calibration,
    }


def run_derived_prediction(
    *,
    experiment_config_path: str | Path,
    baseline_config_path: str | Path,
    candidate_id: str,
    component_prediction_paths: dict[str, str | Path],
    data_config_path: str | Path,
    stage: str,
    output: str | Path,
) -> Path:
    """Create an auditable prediction from frozen component prediction runs."""
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    try:
        data_cfg = load_yaml(data_config_path)
        validate_data_paths(data_cfg, command="pack", stage=stage)
        baseline = load_yaml(baseline_config_path)
        validate_baseline_config(baseline)
        experiment, candidates = load_experiment(experiment_config_path)
        matches = [candidate for candidate in candidates if candidate.id == candidate_id]
        if len(matches) != 1 or matches[0].kind not in {
            "convex_blend", "residual_calibration"
        }:
            raise ContractError("derived prediction requires one registered derived candidate")
        candidate = matches[0]
        dependencies = (
            candidate.component_candidates
            if candidate.kind == "convex_blend"
            else (candidate.base_candidate,)
        )
        if None in dependencies or set(component_prediction_paths) != set(dependencies):
            raise ContractError("component prediction paths do not match candidate dependencies")
        expected = pd.read_csv(
            data_cfg["paths"][f"{stage}_samples"],
            usecols=["sample_id"],
            dtype={"sample_id": "string"},
        )["sample_id"]
        inputs: dict[str, str | Path] = {}
        predictions: dict[str, pd.DataFrame] = {}
        component_manifests: dict[str, object] = {}
        for dependency in dependencies:
            assert dependency is not None
            component_dir = Path(component_prediction_paths[dependency])
            result_path = component_dir / "result.csv"
            manifest_path = component_dir / "prediction_manifest.json"
            if not result_path.is_file() or not manifest_path.is_file():
                raise ContractError(f"component prediction run is incomplete: {dependency}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            result_sha256 = file_sha256(result_path)
            if (
                manifest.get("status") != "PASS"
                or manifest.get("optimization_id") != experiment["optimization_id"]
                or manifest.get("candidate_id") != dependency
                or manifest.get("stage") != stage
                or manifest.get("result_sha256") != result_sha256
            ):
                raise ContractError(f"component prediction manifest mismatch: {dependency}")
            frame = pd.read_csv(result_path, dtype={"sample_id": "string"})
            validate_submission(frame, expected)
            predictions[dependency] = frame
            inputs[f"{dependency}.result"] = result_path
            inputs[f"{dependency}.manifest"] = manifest_path
            component_manifests[dependency] = {
                "prediction_manifest_sha256": file_sha256(manifest_path),
                "result_sha256": result_sha256,
            }
        inputs_before = file_identities(inputs)
        atomic_write_json(
            destination / "run_state.json",
            {
                "status": "RUNNING",
                "optimization_id": experiment["optimization_id"],
                "stage": stage,
                "candidate": _derived_candidate_payload(candidate),
                "component_inputs": inputs_before,
                "code_identity": code_identity(Path.cwd()),
                "environment": runtime_environment(),
            },
        )
        if candidate.kind == "convex_blend" and candidate.blend_weights is not None:
            result = convex_blend(predictions, candidate.blend_weights)
        elif (
            candidate.kind == "residual_calibration"
            and candidate.base_candidate is not None
            and candidate.residual_calibration is not None
        ):
            result = apply_global_residual_calibration(
                predictions[candidate.base_candidate],
                candidate.residual_calibration,
                lower_bound=float(baseline["postprocessing"]["lower_bound"]),
            )
        else:
            raise ContractError("derived candidate configuration is incomplete")
        result.to_csv(destination / "predictions_raw.csv", index=False)
        write_submission(result, expected, destination / "result.csv")
        verify_file_identities(inputs_before)
        atomic_write_json(
            destination / "prediction_manifest.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "stage": stage,
                "optimization_id": experiment["optimization_id"],
                "candidate_id": candidate.id,
                "candidate": _derived_candidate_payload(candidate),
                "components": component_manifests,
                "protected_labels_read": False,
                "rows": len(result),
                "sample_id_sha256": stable_digest(result["sample_id"].astype(str).tolist()),
                "raw_prediction_sha256": file_sha256(destination / "predictions_raw.csv"),
                "result_sha256": file_sha256(destination / "result.csv"),
                "walltime_seconds": perf_counter() - started,
            },
        )
        atomic_write_json(
            destination / "final_status.json",
            {
                "status": "PASS",
                "prediction_manifest_sha256": file_sha256(
                    destination / "prediction_manifest.json"
                ),
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


def run_candidate_training(
    *,
    data_config_path: str | Path,
    semantic_contract_path: str | Path,
    protection_policy_path: str | Path,
    protection_ledger_path: str | Path | None,
    baseline_config_path: str | Path,
    feature_config_path: str | Path,
    experiment_config_path: str | Path,
    optimization_feature_config_path: str | Path,
    optimization_model_config_path: str | Path,
    candidate_id: str,
    train_start: str,
    fit_cutoff: str,
    output: str | Path,
) -> Path:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    try:
        data_cfg = load_yaml(data_config_path)
        validate_data_paths(data_cfg, command="development")
        semantic = load_yaml(semantic_contract_path)
        validate_semantic_contract(semantic)
        protection = load_protection_policy(protection_policy_path)
        baseline = load_yaml(baseline_config_path)
        validate_baseline_config(baseline)
        features = load_yaml(feature_config_path)
        validate_feature_config(features)
        contract_digests = validate_frozen_contracts(baseline, features, semantic)
        experiment, candidates = load_experiment(experiment_config_path)
        selection = load_feature_selection(optimization_feature_config_path)
        model_config = load_model_config(optimization_model_config_path)
        matches = [candidate for candidate in candidates if candidate.id == candidate_id]
        if len(matches) != 1 or matches[0].kind != "model":
            raise ContractError("candidate training requires one registered model candidate")
        candidate = matches[0]
        if Path(selection["base_feature_config"]).resolve() != Path(feature_config_path).resolve():
            raise ContractError("optimization feature selection points to a different baseline feature config")
        if Path(model_config["baseline_config"]).resolve() != Path(baseline_config_path).resolve():
            raise ContractError("optimization model points to a different baseline config")
        if experiment["baseline_id"] != baseline["baseline_id"]:
            raise ContractError("optimization experiment baseline_id differs from frozen baseline")
        fit = protection.validate_development_read(fit_cutoff)
        start = pd.Timestamp(train_start)
        if start.tzinfo is None or not start < fit:
            raise ContractError("train_start must be timezone-aware and earlier than fit_cutoff")

        paths = data_cfg["paths"]
        input_names = {
            key: paths[key]
            for key in (
                "train_samples", "tap_history_train", "operation_hourly",
                "burden_change", "data_dictionary",
            )
        }
        inputs_before = file_identities(input_names)
        inference_contract = build_inference_source_contract(
            inputs_before,
            semantic_contract_sha256=contract_digests["semantic_contract_sha256"],
        )
        code = code_identity(Path.cwd())
        environment = runtime_environment()
        lock_sha = file_sha256("uv.lock")
        resolved = {
            "optimization_id": "optimization-v0.2",
            "candidate": _candidate_payload(candidate),
            "data": data_cfg,
            "semantic": semantic,
            "baseline": baseline,
            "features": features,
            "optimization_features": selection,
            "optimization_model": model_config,
            "train_start": str(start),
            "fit_cutoff": str(fit),
        }
        atomic_write_json(destination / "resolved_config.json", resolved)
        atomic_write_json(
            destination / "run_state.json",
            {
                "status": "RUNNING",
                "purpose": "development_platform_submission",
                "candidate_id": candidate.id,
                "inputs_before": inputs_before,
                "code_identity": code,
                "environment": environment,
                "resolved_config_sha256": stable_digest(resolved),
                "contract_digests": contract_digests,
                "inference_source_contract": inference_contract,
                "protection": {
                    "protected_access": False,
                    "ledger": read_access_ledger(protection_ledger_path, protection),
                },
            },
        )
        labels, history, operation, burden, consistency = _load_inputs(
            data_cfg=data_cfg,
            semantic_cfg=semantic,
            feature_cfg=features,
            protection=protection,
            requested_end=fit,
        )
        eligible = labels.loc[
            (labels["reference_time"] >= start)
            & (labels["reference_time"] < fit)
            & (labels["label_available_at"] <= fit)
        ].copy()
        if eligible.empty:
            raise ContractError("no eligible training rows")
        if len(candidate.history_view_ages_days) > 1:
            views = build_history_views(eligible, candidate.history_view_ages_days)
            fit_samples = views.samples
            fit_targets = views.targets
            sample_weight = views.sample_weight
            weight_policy = "total_one_per_original_sample"
        else:
            views = None
            fit_samples = eligible
            fit_targets = eligible[["tap_iron", "tap_time_len"]]
            sample_weight = None
            weight_policy = "uniform_original_samples"
        built = build_features(
            fit_samples,
            operation=operation,
            burden=burden,
            history=history,
            fit_cutoff=fit,
            config=features,
        )
        fit_features = add_process_change_features(
            built.X,
            selection["process_change"],
            baseline_value_columns=list(features["operation"]["value_columns"]),
        )
        selected = select_candidate_features(fit_features, candidate, selection)
        model = FrozenBaselineAdapter(
            dict(baseline["parameters"]), tuple(baseline["categorical_features"])
        ).fit(selected, fit_targets, sample_weight=sample_weight)
        if views is not None:
            history_audit = built.audit["history"]
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
            view_audit.to_csv(destination / "training_history_views.csv", index=False)
            training_views_sha256 = file_sha256(destination / "training_history_views.csv")
        else:
            training_views_sha256 = None
        history_snapshot = freeze_history_origin(history, fit)
        optimization_metadata = {
            "schema_version": 1,
            "optimization_id": "optimization-v0.2",
            "candidate": _candidate_payload(candidate),
            "feature_selection": selection,
        }
        optimization_metadata["sha256"] = stable_digest(optimization_metadata)
        model.save(
            destination / "bundle",
            metadata={
                "baseline_config": baseline,
                "semantic_contract": semantic,
                "feature_config": features,
                "contract_digests": contract_digests,
                "inference_source_contract": inference_contract,
                "training": {
                    "mode": "optimization-development-submission",
                    "purpose": "development_platform_submission",
                    "candidate_id": candidate.id,
                    "fit_cutoff": str(fit),
                    "train_start": str(start),
                    "eligible_rows": len(eligible),
                    "fit_rows": len(fit_samples),
                    "history_view_ages_days": list(candidate.history_view_ages_days),
                    "sample_weight_policy": weight_policy,
                    "training_history_views_sha256": training_views_sha256,
                    "eligible_sample_id_sha256": stable_digest(
                        eligible["sample_id"].astype(str).tolist()
                    ),
                    "history_rows": len(history_snapshot),
                    "history_origin_sha256": stable_digest(
                        history_snapshot[["sample_id", "available_at"]]
                        .astype(str).to_dict("records")
                    ),
                    "input_identities": inputs_before,
                    "cross_table_consistency": consistency,
                    "protection_access": {
                        "lifecycle": "development",
                        "protected_access": False,
                    },
                },
                "code_identity": code,
                "environment": environment,
                "lockfile_sha256": lock_sha,
                "optimization": optimization_metadata,
            },
            history_snapshot=history_snapshot,
        )
        verify_file_identities(inputs_before)
        atomic_write_json(
            destination / "final_status.json",
            {
                "status": "PASS",
                "candidate_id": candidate.id,
                "protected_labels_read": False,
                "bundle_json_sha256": file_sha256(destination / "bundle" / "bundle.json"),
                "walltime_seconds": perf_counter() - started,
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


def run_candidate_prediction(
    *,
    bundle_path: str | Path,
    data_config_path: str | Path,
    stage: str,
    output: str | Path,
) -> Path:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    try:
        data_cfg = load_yaml(data_config_path)
        validate_data_paths(data_cfg, command="predict", stage=stage)
        paths = data_cfg["paths"]
        required_inputs = {
            f"{stage}_samples": paths[f"{stage}_samples"],
            "operation_hourly": paths["operation_hourly"],
            "burden_change": paths["burden_change"],
        }
        inputs_before = file_identities(required_inputs)
        model = DualTargetBaseline.load(bundle_path)
        metadata = model.bundle_metadata_
        optimization = metadata.get("optimization")
        if not isinstance(optimization, dict) or optimization.get("optimization_id") != "optimization-v0.2":
            raise ContractError("bundle is not an optimization-v0.2 bundle")
        declared_digest = optimization.get("sha256")
        digest_payload = {key: value for key, value in optimization.items() if key != "sha256"}
        if declared_digest != stable_digest(digest_payload):
            raise ContractError("optimization bundle metadata digest mismatch")
        candidate_value = optimization["candidate"]
        if set(candidate_value) != {
            "id", "kind", "sources", "history_components", "history_view_ages_days"
        }:
            raise ContractError("optimization bundle candidate metadata is invalid")
        if candidate_value["kind"] != "model":
            raise ContractError("optimization prediction requires a model candidate")
        from .config import Candidate

        candidate = Candidate(
            candidate_value["id"],
            candidate_value["kind"],
            tuple(candidate_value["sources"]),
            tuple(candidate_value["history_components"]),
            history_view_ages_days=tuple(candidate_value["history_view_ages_days"]),
        )
        semantic = metadata["semantic_contract"]
        features = metadata["feature_config"]
        baseline = metadata["baseline_config"]
        contract_digests = validate_frozen_contracts(baseline, features, semantic)
        if metadata["contract_digests"] != contract_digests:
            raise ContractError("bundle contract digest metadata mismatch")
        validate_inference_source_contract(
            metadata["inference_source_contract"],
            inputs_before,
            semantic_contract_sha256=contract_digests["semantic_contract_sha256"],
        )
        atomic_write_json(
            destination / "run_state.json",
            {
                "status": "RUNNING",
                "stage": stage,
                "optimization_id": "optimization-v0.2",
                "candidate_id": candidate.id,
                "bundle_json_sha256": file_sha256(Path(bundle_path) / "bundle.json"),
                "inputs_before": inputs_before,
                "code_identity": code_identity(Path.cwd()),
                "environment": runtime_environment(),
            },
        )
        samples = read_csv(
            paths[f"{stage}_samples"],
            required_columns=["sample_id", "tap_no", "spout_no", "reference_time"],
            time_columns=["reference_time"],
        )
        validate_samples(samples, labeled=False)
        operation, burden, parse_audit = _load_process_sources(paths, semantic, features)
        history = model.load_history_snapshot()
        validate_history(history)
        built = build_features(
            samples,
            operation=operation,
            burden=burden,
            history=history,
            fit_cutoff=pd.Timestamp(metadata["training"]["fit_cutoff"]),
            config=features,
        )
        prediction_features = add_process_change_features(
            built.X,
            optimization["feature_selection"]["process_change"],
            baseline_value_columns=list(features["operation"]["value_columns"]),
        )
        selected = select_candidate_features(
            prediction_features, candidate, optimization["feature_selection"]
        )
        raw = model.predict_raw(selected)
        raw_frame = pd.DataFrame(
            {
                "sample_id": samples["sample_id"].astype("string").to_numpy(),
                "pred_tap_iron": raw["pred_tap_iron"].to_numpy(),
                "pred_tap_time_len": raw["pred_tap_time_len"].to_numpy(),
            }
        )
        result = raw_frame.copy()
        for column in ("pred_tap_iron", "pred_tap_time_len"):
            result[column] = result[column].clip(lower=float(metadata["postprocess"]["lower_bound"]))
        raw_frame.to_csv(destination / "predictions_raw.csv", index=False)
        write_submission(result, samples["sample_id"], destination / "result.csv")
        for source, audit in built.audit.items():
            audit.to_csv(destination / f"feature_audit_{source}.csv", index=False)
        verify_file_identities(inputs_before)
        atomic_write_json(
            destination / "prediction_manifest.json",
            {
                "status": "PASS",
                "stage": stage,
                "optimization_id": "optimization-v0.2",
                "candidate_id": candidate.id,
                "rows": len(result),
                "sample_id_sha256": stable_digest(result["sample_id"].astype(str).tolist()),
                "bundle_json_sha256": file_sha256(Path(bundle_path) / "bundle.json"),
                "input_identities": inputs_before,
                "inputs_stable": True,
                "parse_audit": parse_audit,
                "raw_prediction_sha256": file_sha256(destination / "predictions_raw.csv"),
                "result_sha256": file_sha256(destination / "result.csv"),
                "walltime_seconds": perf_counter() - started,
                "seconds_per_sample": (perf_counter() - started) / len(result),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "clipped_counts": {
                    column: int((raw_frame[column] < 0).sum())
                    for column in ("pred_tap_iron", "pred_tap_time_len")
                },
            },
        )
        atomic_write_json(
            destination / "final_status.json",
            {
                "status": "PASS",
                "prediction_manifest_sha256": file_sha256(destination / "prediction_manifest.json"),
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
