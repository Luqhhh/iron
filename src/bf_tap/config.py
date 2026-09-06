from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml

from .exceptions import ContractError


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"configuration root must be a mapping: {source}")
    if value.get("schema_version") != 1:
        raise ContractError(f"unsupported schema_version in {source}")
    return value


PATH_KEYS = {
    "train_samples",
    "test_a_samples",
    "test_b_samples",
    "test_c_samples",
    "operation_hourly",
    "burden_change",
    "tap_history_train",
    "data_dictionary",
}


def _require_nonempty_strings(mapping: dict[str, Any], names: set[str], scope: str) -> None:
    missing = sorted(name for name in names if name not in mapping)
    empty = sorted(
        name for name in names if name in mapping and (not isinstance(mapping[name], str) or not mapping[name].strip())
    )
    if missing or empty:
        raise ContractError(f"{scope} missing required values: missing={missing}, empty={empty}")


def validate_data_paths(
    config: dict[str, Any], *, command: str, stage: str | None = None
) -> None:
    allowed_root = {"schema_version", "paths"}
    unknown_root = set(config) - allowed_root
    if unknown_root:
        raise ContractError(f"data path config has unknown keys: {sorted(unknown_root)}")
    if config.get("schema_version") != 1 or not isinstance(config.get("paths"), dict):
        raise ContractError("data path config requires schema_version=1 and a paths mapping")
    paths = config["paths"]
    unknown_paths = set(paths) - PATH_KEYS
    if unknown_paths:
        raise ContractError(f"data path config paths has unknown keys: {sorted(unknown_paths)}")
    common = {"operation_hourly", "burden_change"}
    if command in {"development", "train"}:
        required = common | {"train_samples", "tap_history_train", "data_dictionary"}
    elif command == "predict":
        if stage not in {"test_a", "test_b", "test_c"}:
            raise ContractError("predict requires a supported stage")
        required = common | {f"{stage}_samples"}
    elif command in {"check-submission", "pack"}:
        if stage not in {"test_a", "test_b", "test_c"}:
            raise ContractError(f"{command} requires a supported stage")
        required = {f"{stage}_samples"}
    elif command == "audit":
        if not paths:
            raise ContractError("audit data paths must not be empty")
        required = set(paths)
    else:
        raise ContractError(f"unsupported command contract: {command}")
    _require_nonempty_strings(paths, required, f"{command} data paths")


def validate_semantic_contract(config: dict[str, Any]) -> None:
    allowed_root = {
        "schema_version",
        "contract_id",
        "evidence_status",
        "timezone",
        "sources",
        "targets",
        "evidence",
    }
    unknown = set(config) - allowed_root
    missing = allowed_root - set(config)
    if unknown or missing:
        raise ContractError(
            f"semantic contract keys invalid: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if config["schema_version"] != 1:
        raise ContractError("unsupported semantic contract schema_version")
    if config["evidence_status"] not in {"VERIFIED", "ASSUMED"}:
        raise ContractError("evidence_status must be VERIFIED or ASSUMED")
    _require_nonempty_strings(config, {"contract_id", "timezone"}, "semantic contract")
    sources = config["sources"]
    if not isinstance(sources, dict) or set(sources) != {"operation", "burden", "history"}:
        raise ContractError("semantic sources must be exactly operation, burden, history")
    process_allowed = {"event_time_column", "available_at_column", "missing_markers"}
    for name in ("operation", "burden"):
        source = sources[name]
        if not isinstance(source, dict) or set(source) != process_allowed:
            raise ContractError(f"{name} semantic mapping keys are invalid")
        _require_nonempty_strings(
            source, {"event_time_column", "available_at_column"}, f"{name} semantic mapping"
        )
        if not isinstance(source["missing_markers"], list) or not all(
            isinstance(value, str) for value in source["missing_markers"]
        ):
            raise ContractError(f"{name}.missing_markers must be a string list")
    history = sources["history"]
    history_keys = {"event_time_column", "end_time_column", "available_at_column"}
    if not isinstance(history, dict) or set(history) != history_keys:
        raise ContractError("history semantic mapping keys are invalid")
    _require_nonempty_strings(history, history_keys, "history semantic mapping")
    targets = config["targets"]
    if not isinstance(targets, dict) or set(targets) != {"available_at_column"}:
        raise ContractError("targets mapping must contain only available_at_column")
    _require_nonempty_strings(targets, {"available_at_column"}, "targets mapping")
    evidence = config["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != {"assets", "fields", "pending"}:
        raise ContractError(
            "semantic evidence must contain exactly assets, fields and pending"
        )
    assets = evidence["assets"]
    if not isinstance(assets, list) or not assets:
        raise ContractError("semantic evidence assets must be a non-empty list")
    asset_ids: set[str] = set()
    for asset in assets:
        _exact_keys(asset, {"asset_id", "location", "sha256"}, "evidence asset")
        _require_nonempty_strings(
            asset, {"asset_id", "location", "sha256"}, "evidence asset"
        )
        if not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]):
            raise ContractError("evidence asset sha256 must be lowercase hexadecimal")
        asset_ids.add(asset["asset_id"])
    if len(asset_ids) != len(assets):
        raise ContractError("semantic evidence asset_id values must be unique")
    fields = evidence["fields"]
    expected_fields = {
        "operation.clock",
        "burden.cal_time",
        "history.tap_end_time",
        "targets.tap_end_time",
    }
    _exact_keys(fields, expected_fields, "semantic field evidence")
    for name, field in fields.items():
        _exact_keys(
            field,
            {"status", "asset_id", "location", "conclusion"},
            f"semantic field evidence {name}",
        )
        _require_nonempty_strings(
            field,
            {"status", "asset_id", "location", "conclusion"},
            f"semantic field evidence {name}",
        )
        if field["status"] not in {"VERIFIED", "ASSUMED"}:
            raise ContractError(f"invalid semantic evidence status for {name}")
        if field["asset_id"] not in asset_ids:
            raise ContractError(f"unknown evidence asset for {name}")
    if not isinstance(evidence["pending"], list):
        raise ContractError("semantic evidence pending must be a list")


def _exact_keys(mapping: Any, expected: set[str], scope: str) -> None:
    if not isinstance(mapping, dict):
        raise ContractError(f"{scope} must be a mapping")
    missing = expected - set(mapping)
    unknown = set(mapping) - expected
    if missing or unknown:
        raise ContractError(
            f"{scope} keys invalid: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def validate_baseline_config(config: dict[str, Any]) -> None:
    _exact_keys(
        config,
        {
            "schema_version",
            "decision_status",
            "baseline_id",
            "targets",
            "parameters",
            "categorical_features",
            "max_categorical_cardinality",
            "controls",
            "postprocessing",
        },
        "baseline config",
    )
    if config["schema_version"] != 1 or config["decision_status"] != "frozen_by_ai":
        raise ContractError("baseline config must be schema v1 and frozen_by_ai")
    if config["targets"] != ["tap_iron", "tap_time_len"]:
        raise ContractError("baseline targets differ from frozen contract")
    if config["categorical_features"] != ["spout_no"]:
        raise ContractError("baseline categorical features differ from frozen contract")
    _exact_keys(config["controls"], {"global", "per_spout"}, "baseline controls")
    _exact_keys(
        config["controls"]["per_spout"],
        {"id", "min_group_count", "fallback"},
        "per_spout control",
    )
    _exact_keys(
        config["postprocessing"], {"lower_bound", "csv_decimals"}, "postprocessing"
    )
    if config["postprocessing"] != {"lower_bound": 0.0, "csv_decimals": 6}:
        raise ContractError("postprocessing differs from frozen contract")


def validate_feature_config(config: dict[str, Any]) -> None:
    _exact_keys(
        config,
        {
            "schema_version",
            "decision_status",
            "sample",
            "operation",
            "burden",
            "history",
            "categorical_missing",
        },
        "feature config",
    )
    if config["schema_version"] != 1 or config["decision_status"] != "frozen_by_ai":
        raise ContractError("feature config must be schema v1 and frozen_by_ai")
    _exact_keys(config["sample"], {"categorical", "derived"}, "sample features")
    _exact_keys(
        config["operation"],
        {"value_columns", "windows_hours", "latest_max_event_age_hours"},
        "operation features",
    )
    _exact_keys(
        config["burden"], {"value_columns", "latest_max_event_age_hours"}, "burden features"
    )
    _exact_keys(config["history"], {"targets", "last_k_mean", "require_both_targets_available", "exclude_current_tap"}, "history features")
    if config["operation"]["windows_hours"] != [6, 24]:
        raise ContractError("operation windows differ from frozen contract")
    if config["history"]["last_k_mean"] != [3, 10]:
        raise ContractError("history windows differ from frozen contract")
    if config["categorical_missing"] != "__MISSING__":
        raise ContractError("categorical missing token differs from frozen contract")


def validate_acceptance_config(config: dict[str, Any]) -> None:
    _exact_keys(
        config,
        {
            "schema_version",
            "same_bundle_raw_prediction_max_abs_diff",
            "same_bundle_csv_byte_equality",
            "same_environment_retrain_raw_prediction_max_abs_diff",
            "inference_walltime_seconds_per_sample_max",
            "inference_total_seconds_max",
            "quality",
        },
        "acceptance config",
    )
    _exact_keys(
        config["quality"],
        {
            "max_loss_relative_to_better_control",
            "per_target_max_absolute_wmape_regression_vs_B1",
        },
        "acceptance quality",
    )


def validate_validation_config(config: dict[str, Any], *, timezone: str) -> None:
    _exact_keys(
        config, {"schema_version", "timezone", "interval", "folds"}, "validation config"
    )
    if config["schema_version"] != 1:
        raise ContractError("unsupported validation schema_version")
    if config["timezone"] != timezone:
        raise ContractError("validation timezone differs from protection policy")
    if config["interval"] != "left_closed_right_open":
        raise ContractError("validation interval must be left_closed_right_open")
    if not isinstance(config["folds"], list) or not config["folds"]:
        raise ContractError("validation folds must be a non-empty list")
