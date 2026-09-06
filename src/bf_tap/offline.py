from __future__ import annotations

import json
import resource
from pathlib import Path
from time import perf_counter

import pandas as pd

from .artifacts import (
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
from .availability import freeze_history_origin
from .config import (
    load_yaml,
    validate_baseline_config,
    validate_data_paths,
    validate_feature_config,
    validate_frozen_contracts,
    validate_semantic_contract,
)
from .data import normalize_event_source
from .exceptions import ContractError, ProtectedLabelError
from .features import build_features
from .io import read_csv, read_development_history, read_development_labels
from .models.baseline import DualTargetBaseline
from .protection import load_protection_policy, read_access_ledger
from .schema import (
    validate_cross_table_consistency,
    validate_cross_table_metadata,
    validate_history,
    validate_samples,
)
from .submission import write_submission
from .train import fit_baseline


def _load_process_sources(
    paths: dict,
    semantic: dict,
    features: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    op_mapping = semantic["sources"]["operation"]
    operation = normalize_event_source(
        read_csv(paths["operation_hourly"]),
        event_time_column=op_mapping["event_time_column"],
        available_at_column=op_mapping["available_at_column"],
        value_columns=list(features["operation"]["value_columns"]),
        missing_markers=op_mapping["missing_markers"],
    )
    burden_mapping = semantic["sources"]["burden"]
    burden = normalize_event_source(
        read_csv(paths["burden_change"]),
        event_time_column=burden_mapping["event_time_column"],
        available_at_column=burden_mapping["available_at_column"],
        value_columns=list(features["burden"]["value_columns"]),
        missing_markers=burden_mapping["missing_markers"],
    )
    return operation, burden, {
        "operation": operation.attrs.get("parse_audit", {}),
        "burden": burden.attrs.get("parse_audit", {}),
    }


def _metadata_consistency(paths: dict, semantic: dict) -> dict[str, int]:
    history_mapping = semantic["sources"]["history"]
    samples = read_csv(
        paths["train_samples"],
        usecols=["sample_id", "tap_no", "spout_no", "reference_time"],
        required_columns=["sample_id", "tap_no", "spout_no", "reference_time"],
        time_columns=["reference_time"],
    )
    history = read_csv(
        paths["tap_history_train"],
        usecols=["sample_id", "tap_no", "spout_no", "reference_time", history_mapping["end_time_column"]],
        required_columns=["sample_id", "tap_no", "spout_no", "reference_time", history_mapping["end_time_column"]],
        time_columns=["reference_time", history_mapping["end_time_column"]],
    )
    history["available_at"] = history[history_mapping["end_time_column"]]
    return validate_cross_table_metadata(samples, history)


def _authorized_official_access(
    *,
    ledger_path: str | Path | None,
    protection,
    frozen_manifest_path: str | Path | None,
) -> dict:
    if ledger_path is None or frozen_manifest_path is None:
        raise ProtectedLabelError(
            "official-release training requires protection ledger and frozen manifest"
        )
    frozen = Path(frozen_manifest_path)
    if not frozen.is_file():
        raise ProtectedLabelError("frozen manifest does not exist")
    frozen_sha = file_sha256(frozen)
    ledger = read_access_ledger(ledger_path, protection)
    matching = [
        entry
        for entry in ledger["entries"]
        if entry.get("lifecycle") == "final_training"
        and entry.get("frozen_manifest_sha256") == frozen_sha
    ]
    if not matching:
        raise ProtectedLabelError("no matching final_training access ledger entry")
    return {"ledger": ledger, "frozen_manifest_sha256": frozen_sha}


def run_training(
    *,
    data_config_path: str | Path,
    semantic_contract_path: str | Path,
    protection_policy_path: str | Path,
    baseline_config_path: str | Path,
    feature_config_path: str | Path,
    output: str | Path,
    mode: str,
    train_start: str,
    fit_cutoff: str | None = None,
    protection_ledger_path: str | Path | None = None,
    frozen_manifest_path: str | Path | None = None,
) -> Path:
    if mode not in {"development", "official-release"}:
        raise ContractError(f"unsupported training mode: {mode}")
    data_cfg = load_yaml(data_config_path)
    validate_data_paths(
        data_cfg, command="development" if mode == "development" else "train"
    )
    semantic = load_yaml(semantic_contract_path)
    validate_semantic_contract(semantic)
    protection = load_protection_policy(protection_policy_path)
    baseline = load_yaml(baseline_config_path)
    validate_baseline_config(baseline)
    features = load_yaml(feature_config_path)
    validate_feature_config(features)
    contract_digests = validate_frozen_contracts(baseline, features, semantic)
    paths = data_cfg["paths"]

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    resolved = {
        "data": data_cfg,
        "data_contract": semantic,
        "protection": {
            "contract_id": protection.contract_id,
            "policy_digest": protection.digest,
        },
        "baseline": baseline,
        "features": features,
        "mode": mode,
        "train_start": train_start,
        "fit_cutoff": fit_cutoff,
    }
    atomic_write_json(destination / "resolved_config.json", resolved)
    input_names = {
        key: paths[key]
        for key in ("train_samples", "tap_history_train", "operation_hourly", "burden_change", "data_dictionary")
    }
    started = perf_counter()
    try:
        inputs_before = file_identities(input_names)
        inference_source_contract = build_inference_source_contract(
            inputs_before,
            semantic_contract_sha256=contract_digests[
                "semantic_contract_sha256"
            ],
        )
        runtime = runtime_environment()
        code = code_identity(Path.cwd())
        lock_sha = file_sha256("uv.lock") if Path("uv.lock").is_file() else "UNAVAILABLE"
        atomic_write_json(
            destination / "run_state.json",
            {
                "status": "RUNNING",
                "mode": mode,
                "inputs_before": inputs_before,
                "semantic_contract_id": semantic["contract_id"],
                "semantic_evidence_status": semantic["evidence_status"],
                "protection_policy_id": protection.contract_id,
                "protection_policy_digest": protection.digest,
                "code_identity": code,
                "environment": runtime,
                "lockfile_sha256": lock_sha,
                "resolved_config_sha256": stable_digest(resolved),
                "contract_digests": contract_digests,
                "inference_source_contract": inference_source_contract,
                "cache": {"enabled": False, "reason": "baseline implementation has no cache I/O"},
            },
        )
    except Exception as exc:
        atomic_write_json(
            destination / "final_status.json",
            {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    try:
        if mode == "development":
            if fit_cutoff is None:
                raise ContractError("development training requires fit_cutoff")
            fit = protection.validate_development_read(fit_cutoff)
            labels = read_development_labels(
                paths["train_samples"],
                requested_end=fit,
                protection_policy=protection,
            )
            history = read_development_history(
                paths["tap_history_train"],
                requested_end=fit,
                protection_policy=protection,
                available_at_column=semantic["sources"]["history"]["available_at_column"],
            )
            access = {
                "lifecycle": "development",
                "protected_access": False,
                "ledger": read_access_ledger(protection_ledger_path, protection),
            }
        else:
            access = _authorized_official_access(
                ledger_path=protection_ledger_path,
                protection=protection,
                frozen_manifest_path=frozen_manifest_path,
            )
            if not paths.get("test_a_samples"):
                raise ContractError("official-release training requires test_a_samples path")
            test_a = read_csv(
                paths["test_a_samples"],
                usecols=["reference_time"],
                required_columns=["reference_time"],
                time_columns=["reference_time"],
            )
            fit = test_a["reference_time"].min()
            labels = read_csv(
                paths["train_samples"],
                required_columns=["sample_id", "tap_no", "spout_no", "reference_time", "tap_iron", "tap_time_len"],
                time_columns=["reference_time"],
            )
            history = read_csv(
                paths["tap_history_train"],
                time_columns=["reference_time", semantic["sources"]["history"]["end_time_column"]],
            )
        validate_samples(labels, labeled=True)
        history = history.copy()
        history["available_at"] = history[
            semantic["sources"]["history"]["available_at_column"]
        ]
        validate_history(history)
        metadata_consistency = _metadata_consistency(paths, semantic)
        target_consistency = validate_cross_table_consistency(labels, history)
        availability = history[["sample_id", semantic["targets"]["available_at_column"]]].rename(
            columns={semantic["targets"]["available_at_column"]: "label_available_at"}
        )
        labels = labels.merge(availability, on="sample_id", how="left", validate="one_to_one")
        start = pd.Timestamp(train_start)
        if start.tzinfo is None:
            raise ContractError("train_start must be timezone-aware")
        eligible = labels.loc[
            (labels["reference_time"] >= start)
            & (labels["reference_time"] < fit)
            & (labels["label_available_at"] <= fit)
        ].copy()
        if eligible.empty:
            raise ContractError("no eligible training rows")
        operation, burden, parse_audit = _load_process_sources(paths, semantic, features)
        built = build_features(
            eligible,
            operation=operation,
            burden=burden,
            history=history,
            fit_cutoff=fit,
            config=features,
        )
        model = fit_baseline(
            eligible,
            built.X,
            dict(baseline["parameters"]),
            categorical=tuple(baseline["categorical_features"]),
        )
        history_snapshot = freeze_history_origin(history, fit)
        training_metadata = {
            "mode": mode,
            "fit_cutoff": str(fit),
            "train_start": str(start),
            "eligible_rows": len(eligible),
            "eligible_sample_id_sha256": stable_digest(eligible["sample_id"].astype(str).tolist()),
            "history_rows": len(history_snapshot),
            "history_origin_sha256": stable_digest(
                history_snapshot[["sample_id", "available_at"]].astype(str).to_dict("records")
            ),
            "input_identities": inputs_before,
            "cross_table_consistency": {
                "metadata": metadata_consistency,
                "targets": target_consistency,
            },
            "parse_audit": parse_audit,
            "protection_access": access,
        }
        model.save(
            destination / "bundle",
            metadata={
                "baseline_config": baseline,
                "semantic_contract": semantic,
                "feature_config": features,
                "contract_digests": contract_digests,
                "inference_source_contract": inference_source_contract,
                "training": training_metadata,
                "code_identity": code,
                "environment": runtime,
                "lockfile_sha256": lock_sha,
            },
            history_snapshot=history_snapshot,
        )
        verify_file_identities(inputs_before)
        final = {
            "status": "PASS",
            "mode": mode,
            "bundle_json_sha256": file_sha256(destination / "bundle" / "bundle.json"),
            "walltime_seconds": perf_counter() - started,
            "inputs_stable": True,
        }
        atomic_write_json(destination / "final_status.json", final)
    except Exception as exc:
        atomic_write_json(
            destination / "final_status.json",
            {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    return destination


def run_prediction(
    *,
    bundle_path: str | Path,
    data_config_path: str | Path,
    stage: str,
    output: str | Path,
) -> Path:
    data_cfg = load_yaml(data_config_path)
    validate_data_paths(data_cfg, command="predict", stage=stage)
    paths = data_cfg["paths"]
    required_inputs = {
        f"{stage}_samples": paths[f"{stage}_samples"],
        "operation_hourly": paths["operation_hourly"],
        "burden_change": paths["burden_change"],
    }
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    try:
        inputs_before = file_identities(required_inputs)
        model = DualTargetBaseline.load(bundle_path)
        metadata = model.bundle_metadata_
        semantic = metadata["semantic_contract"]
        features = metadata["feature_config"]
        baseline = metadata["baseline_config"]
        contract_digests = validate_frozen_contracts(
            baseline, features, semantic
        )
        if metadata["contract_digests"] != contract_digests:
            raise ContractError("bundle contract digest metadata mismatch")
        validate_inference_source_contract(
            metadata["inference_source_contract"],
            inputs_before,
            semantic_contract_sha256=contract_digests[
                "semantic_contract_sha256"
            ],
        )
        atomic_write_json(
            destination / "run_state.json",
            {
                "status": "RUNNING",
                "stage": stage,
                "bundle_json_sha256": file_sha256(Path(bundle_path) / "bundle.json"),
                "semantic_contract_id": semantic["contract_id"],
                "contract_digests": contract_digests,
                "inference_source_contract_id": metadata[
                    "inference_source_contract"
                ]["contract_id"],
                "inputs_before": inputs_before,
                "code_identity": code_identity(Path.cwd()),
                "environment": runtime_environment(),
            },
        )
    except Exception as exc:
        atomic_write_json(
            destination / "final_status.json",
            {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    try:
        samples = read_csv(
            paths[f"{stage}_samples"],
            required_columns=["sample_id", "tap_no", "spout_no", "reference_time"],
            time_columns=["reference_time"],
        )
        validate_samples(samples, labeled=False)
        operation, burden, parse_audit = _load_process_sources(paths, semantic, features)
        history = model.load_history_snapshot()
        validate_history(history)
        fit_cutoff = pd.Timestamp(metadata["training"]["fit_cutoff"])
        built = build_features(
            samples,
            operation=operation,
            burden=burden,
            history=history,
            fit_cutoff=fit_cutoff,
            config=features,
        )
        raw = model.predict_raw(built.X)
        final = raw.clip(lower=float(metadata["postprocess"]["lower_bound"]))
        raw_frame = pd.DataFrame(
            {
                "sample_id": samples["sample_id"].astype("string").to_numpy(),
                "pred_tap_iron": raw["pred_tap_iron"].to_numpy(),
                "pred_tap_time_len": raw["pred_tap_time_len"].to_numpy(),
            }
        )
        result = raw_frame.copy()
        result[["pred_tap_iron", "pred_tap_time_len"]] = final[
            ["pred_tap_iron", "pred_tap_time_len"]
        ].to_numpy()
        raw_frame.to_csv(destination / "predictions_raw.csv", index=False)
        write_submission(result, samples["sample_id"], destination / "result.csv")
        for source, audit in built.audit.items():
            audit.to_csv(destination / f"feature_audit_{source}.csv", index=False)
        verify_file_identities(inputs_before)
        component_sizes = {
            name: (Path(bundle_path) / name).stat().st_size
            for name in metadata["component_sha256"]
        }
        atomic_write_json(
            destination / "prediction_manifest.json",
            {
                "status": "PASS",
                "stage": stage,
                "rows": len(result),
                "sample_id_sha256": stable_digest(result["sample_id"].astype(str).tolist()),
                "bundle_json_sha256": file_sha256(Path(bundle_path) / "bundle.json"),
                "contract_digests": contract_digests,
                "inference_source_contract": metadata[
                    "inference_source_contract"
                ],
                "input_identities": inputs_before,
                "inputs_stable": True,
                "parse_audit": parse_audit,
                "raw_prediction_sha256": file_sha256(destination / "predictions_raw.csv"),
                "result_sha256": file_sha256(destination / "result.csv"),
                "component_sizes": component_sizes,
                "model_bytes": sum(
                    size for name, size in component_sizes.items() if name.endswith(".cbm")
                ),
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
                "prediction_manifest_sha256": file_sha256(
                    destination / "prediction_manifest.json"
                ),
            },
        )
    except Exception as exc:
        atomic_write_json(
            destination / "final_status.json",
            {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    return destination
