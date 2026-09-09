from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import pandas as pd

from ...exceptions import ContractError
from ...splits import split_from_config
from .registry import get_candidate_definition


_COMMON_KEYS = {
    "schema_version",
    "optimization_id",
    "baseline_id",
    "feature_contract",
    "development_label_end_exclusive",
    "random_seed",
    "folds",
    "selection",
}
_SELECTION = {
    "metric": "pooled_wmape",
    "per_target_max_absolute_wmape_regression": 0.002,
    "minimum_winning_fold_fraction": 2 / 3,
}
_EXPECTED_PARAMETERS = {
    "M1_BLEND": {
        "weights": [0.0, 0.25, 0.5, 0.75, 1.0],
        "per_target": True,
        "tie_break": "prefer_b1",
        "tie_tolerance": 1e-12,
    },
    "M2_RECENCY": {
        "half_life_days": [30, 60, 120],
        "normalize_mean": True,
        "base_parameters_from": "baseline-v0.1",
    },
    "M3_RESIDUAL": {
        "anchor": "B1",
        "fallback": "B0",
        "min_group_count": 20,
        "train_anchor_mode": "expanding_asof",
        "base_parameters_from": "baseline-v0.1",
    },
    "M4_ENSEMBLE": {
        "enabled": False,
        "activation_gate": "residual_complementarity_required",
    },
}


def _require_exact_keys(value: object, expected: set[str], scope: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ContractError(f"{scope} must be a mapping")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ContractError(
            f"{scope} keys invalid: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return value


def validate_optimization_common_config(config: dict[str, object]) -> None:
    _require_exact_keys(config, _COMMON_KEYS, "common config")
    expected_scalars = {
        "schema_version": 1,
        "optimization_id": "optimization-v0.2",
        "baseline_id": "baseline-v0.1",
        "feature_contract": "baseline-v0.1-asof",
        "random_seed": 2026,
    }
    for key, expected in expected_scalars.items():
        if config[key] != expected:
            raise ContractError(f"common config {key} must be {expected!r}")
    if config["selection"] != _SELECTION:
        raise ContractError("common config selection differs from optimization-v0.2")
    boundary = pd.Timestamp(config["development_label_end_exclusive"])
    if boundary.tzinfo is None:
        raise ContractError("development label boundary must be timezone-aware")
    folds_value = config["folds"]
    if not isinstance(folds_value, list) or len(folds_value) != 3:
        raise ContractError("optimization-v0.2 requires exactly three rolling folds")
    folds = [split_from_config(value) for value in folds_value]
    if [fold.id for fold in folds] != ["ROLL_2024_06", "ROLL_2024_07", "ROLL_2024_08"]:
        raise ContractError("rolling fold ids differ from optimization-v0.2")
    for fold in folds:
        if fold.kind != "development" or fold.fit_cutoff != fold.eval_start:
            raise ContractError("rolling folds must be development origins")
        if fold.eval_end > boundary:
            raise ContractError("rolling fold crosses protected label boundary")
    if any(left.eval_end > right.eval_start for left, right in zip(folds, folds[1:])):
        raise ContractError("rolling evaluation windows overlap")


def validate_optimization_candidate_config(config: dict[str, object]) -> None:
    _require_exact_keys(
        config,
        {"schema_version", "optimization_id", "candidate_id", "family", "parameters"},
        "candidate config",
    )
    if config["schema_version"] != 1 or config["optimization_id"] != "optimization-v0.2":
        raise ContractError("candidate config identity differs from optimization-v0.2")
    candidate_id = config["candidate_id"]
    if not isinstance(candidate_id, str):
        raise ContractError("candidate_id must be a string")
    definition = get_candidate_definition(candidate_id)
    if config["family"] != definition.family:
        raise ContractError("candidate family does not match registry")
    if config["parameters"] != _EXPECTED_PARAMETERS[candidate_id]:
        raise ContractError(f"{candidate_id} parameters differ from registered contract")

def build_rolling_validation_config(
    common_config: dict[str, object],
    *,
    timezone: str,
) -> dict[str, object]:
    validate_optimization_common_config(common_config)
    if not isinstance(timezone, str) or not timezone:
        raise ContractError("validation timezone must be a non-empty string")
    return {
        "schema_version": 1,
        "timezone": timezone,
        "interval": "left_closed_right_open",
        "folds": deepcopy(common_config["folds"]),
    }
