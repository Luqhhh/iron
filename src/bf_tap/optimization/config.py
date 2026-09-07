from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import load_yaml
from ..exceptions import ContractError

SOURCES = {"sample", "operation", "burden", "history", "process_change"}
HISTORY_COMPONENTS = {"age", "count", "target"}
PROCESS_CHANGE_TRANSFORMS = (
    "latest_minus_6h_mean",
    "latest_minus_24h_mean",
    "mean_6h_minus_24h_mean",
)


def _exact(mapping: Any, expected: set[str], scope: str) -> None:
    if not isinstance(mapping, dict):
        raise ContractError(f"{scope} must be a mapping")
    missing = expected - set(mapping)
    unknown = set(mapping) - expected
    if missing or unknown:
        raise ContractError(
            f"{scope} keys invalid: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


@dataclass(frozen=True)
class Candidate:
    id: str
    kind: str
    sources: tuple[str, ...] = ()
    history_components: tuple[str, ...] = ()
    base_candidate: str | None = None
    weights: dict[str, float] | None = None
    history_view_ages_days: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class EvaluationUnit:
    id: str
    origin_id: str
    horizon: int
    fit_cutoff: pd.Timestamp
    train_start: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp


@dataclass(frozen=True)
class Origin:
    id: str
    fit_cutoff: pd.Timestamp
    horizons: int


def load_experiment(path: str | Path) -> tuple[dict[str, Any], list[Candidate]]:
    config = load_yaml(path)
    _exact(
        config,
        {
            "schema_version",
            "optimization_id",
            "baseline_id",
            "candidate_order",
            "candidates",
            "shrinkage_grid",
        },
        "optimization experiment",
    )
    if config["optimization_id"] != "optimization-v0.2":
        raise ContractError("optimization_id must be optimization-v0.2")
    order = config["candidate_order"]
    candidates = config["candidates"]
    if not isinstance(order, list) or not order or len(order) != len(set(order)):
        raise ContractError("candidate_order must be a non-empty unique list")
    if not isinstance(candidates, dict) or set(candidates) != set(order):
        raise ContractError("candidate_order and candidates must contain identical IDs")
    parsed: list[Candidate] = []
    for candidate_id in order:
        value = candidates[candidate_id]
        if value.get("kind") == "model":
            _exact(
                value,
                {"kind", "sources", "history_components", "history_view_ages_days"},
                candidate_id,
            )
            sources = tuple(value["sources"])
            components = tuple(value["history_components"])
            ages = value["history_view_ages_days"]
            if not sources or len(sources) != len(set(sources)) or set(sources) - SOURCES:
                raise ContractError(f"{candidate_id}: invalid or duplicate sources")
            if "sample" not in sources:
                raise ContractError(f"{candidate_id}: sample source is mandatory")
            if len(components) != len(set(components)) or set(components) - HISTORY_COMPONENTS:
                raise ContractError(f"{candidate_id}: invalid history_components")
            if ("history" in sources) != bool(components):
                raise ContractError(f"{candidate_id}: history source/components disagree")
            if (
                not isinstance(ages, list)
                or not ages
                or ages != sorted(set(ages))
                or ages[0] != 0
                or any(isinstance(age, bool) or not isinstance(age, int) or age < 0 for age in ages)
            ):
                raise ContractError(
                    f"{candidate_id}: history_view_ages_days must be sorted unique nonnegative integers starting at 0"
                )
            if "history" not in sources and ages != [0]:
                raise ContractError(
                    f"{candidate_id}: history-free candidates cannot use synthetic history views"
                )
            parsed.append(
                Candidate(
                    candidate_id,
                    "model",
                    sources,
                    components,
                    history_view_ages_days=tuple(ages),
                )
            )
        elif value.get("kind") == "shrink_b1":
            _exact(value, {"kind", "base_candidate", "weights"}, candidate_id)
            weights = value["weights"]
            _exact(weights, {"tap_iron", "tap_time_len"}, f"{candidate_id}.weights")
            if any(
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not 0 <= float(weight) <= 1
                for weight in weights.values()
            ):
                raise ContractError(f"{candidate_id}: weights must be in [0, 1]")
            if value["base_candidate"] not in candidates:
                raise ContractError(f"{candidate_id}: unknown base_candidate")
            parsed.append(
                Candidate(
                    candidate_id,
                    "shrink_b1",
                    base_candidate=value["base_candidate"],
                    weights={key: float(weight) for key, weight in weights.items()},
                )
            )
        else:
            raise ContractError(f"{candidate_id}: unsupported candidate kind")
    grid = config["shrinkage_grid"]
    if not isinstance(grid, list) or sorted(set(grid)) != list(grid):
        raise ContractError("shrinkage_grid must be sorted and unique")
    if not grid or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 <= x <= 1 for x in grid):
        raise ContractError("shrinkage_grid values must be in [0, 1]")
    return config, parsed


def load_feature_selection(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    _exact(
        config,
        {
            "schema_version",
            "base_feature_config",
            "source_prefixes",
            "history_component_patterns",
            "process_change",
        },
        "optimization feature selection",
    )
    _exact(config["source_prefixes"], SOURCES, "source_prefixes")
    _exact(
        config["history_component_patterns"],
        HISTORY_COMPONENTS,
        "history_component_patterns",
    )
    _exact(
        config["process_change"],
        {"feature_set_id", "value_columns", "transforms"},
        "process_change",
    )
    process_change = config["process_change"]
    if process_change["feature_set_id"] != "signed_operation_level_deltas_v1":
        raise ContractError("unsupported process_change feature_set_id")
    if (
        not isinstance(process_change["value_columns"], list)
        or not process_change["value_columns"]
        or len(process_change["value_columns"]) != len(set(process_change["value_columns"]))
        or not all(isinstance(value, str) and value for value in process_change["value_columns"])
    ):
        raise ContractError("process_change.value_columns must be a non-empty unique string list")
    if tuple(process_change["transforms"]) != PROCESS_CHANGE_TRANSFORMS:
        raise ContractError("process_change.transforms differ from signed delta v1 contract")
    for scope in ("source_prefixes", "history_component_patterns"):
        for name, values in config[scope].items():
            if not isinstance(values, list) or not values or not all(isinstance(x, str) and x for x in values):
                raise ContractError(f"{scope}.{name} must be a non-empty string list")
    return config


def load_acceptance(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    _exact(
        config,
        {
            "schema_version",
            "selection_metric",
            "min_J_improvement",
            "min_improved_horizons",
            "max_any_horizon_regression",
            "dev_short_max_regression",
            "per_target_mean_wmape_max_regression",
            "dev_long_must_beat_better_control",
        },
        "optimization acceptance",
    )
    if config["selection_metric"] != "equal_horizon_mean":
        raise ContractError("optimization selection_metric must be equal_horizon_mean")
    for name in (
        "min_J_improvement",
        "max_any_horizon_regression",
        "dev_short_max_regression",
        "per_target_mean_wmape_max_regression",
    ):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ContractError(f"{name} must be nonnegative")
    if config["min_improved_horizons"] not in {1, 2, 3, 4}:
        raise ContractError("min_improved_horizons must be from 1 to 4")
    if not isinstance(config["dev_long_must_beat_better_control"], bool):
        raise ContractError("dev_long_must_beat_better_control must be boolean")
    return config


def load_model_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    _exact(
        config,
        {"schema_version", "model_type", "baseline_config"},
        "optimization model",
    )
    if config["model_type"] != "frozen_baseline_catboost":
        raise ContractError("OPT-01/02 only supports frozen_baseline_catboost")
    if not isinstance(config["baseline_config"], str) or not config["baseline_config"]:
        raise ContractError("optimization model baseline_config must be a path")
    return config


def _timestamp(value: Any, scope: str, timezone: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ContractError(f"{scope} must be timezone-aware")
    if str(result.tz) != timezone and str(result.tzinfo) != timezone:
        # Numeric +08:00 is accepted when it represents the configured local offset.
        local = result.tz_convert(timezone)
        if local.utcoffset() != result.utcoffset():
            raise ContractError(f"{scope} must use timezone {timezone}")
    return result


def load_validation(path: str | Path, *, expected_timezone: str) -> tuple[dict[str, Any], list[EvaluationUnit], list[Origin]]:
    config = load_yaml(path)
    _exact(
        config,
        {"schema_version", "timezone", "interval", "train_start", "screening_folds", "origins"},
        "optimization validation",
    )
    if config["timezone"] != expected_timezone or config["interval"] != "left_closed_right_open":
        raise ContractError("optimization validation timezone or interval differs from protection contract")
    train_start = _timestamp(config["train_start"], "train_start", expected_timezone)
    screening: list[EvaluationUnit] = []
    for value in config["screening_folds"]:
        _exact(value, {"id", "fit_cutoff", "eval_start", "eval_end"}, "screening fold")
        fit = _timestamp(value["fit_cutoff"], f"{value['id']}.fit_cutoff", expected_timezone)
        start = _timestamp(value["eval_start"], f"{value['id']}.eval_start", expected_timezone)
        end = _timestamp(value["eval_end"], f"{value['id']}.eval_end", expected_timezone)
        if not train_start < fit <= start < end:
            raise ContractError(f"{value['id']}: invalid temporal order")
        screening.append(EvaluationUnit(value["id"], value["id"], 0, fit, train_start, start, end))
    origins: list[Origin] = []
    units: list[EvaluationUnit] = []
    for value in config["origins"]:
        _exact(value, {"id", "fit_cutoff", "horizons"}, "origin")
        fit = _timestamp(value["fit_cutoff"], f"{value['id']}.fit_cutoff", expected_timezone)
        horizons = value["horizons"]
        if isinstance(horizons, bool) or not isinstance(horizons, int) or not 1 <= horizons <= 4:
            raise ContractError(f"{value['id']}: horizons must be an integer from 1 to 4")
        origins.append(Origin(value["id"], fit, horizons))
        for horizon in range(1, horizons + 1):
            start = fit + pd.DateOffset(months=horizon - 1)
            end = fit + pd.DateOffset(months=horizon)
            units.append(
                EvaluationUnit(
                    f"{value['id']}_H{horizon}", value["id"], horizon, fit, train_start, start, end
                )
            )
    ids = [unit.id for unit in [*screening, *units]]
    if len(ids) != len(set(ids)):
        raise ContractError("optimization evaluation IDs must be unique")
    return config, screening, origins
