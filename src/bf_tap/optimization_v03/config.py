from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import load_yaml
from ..exceptions import ContractError


_CANONICAL = {
    "schema_version": 1,
    "optimization_id": "optimization-v0.3-drift",
    "baseline_id": "baseline-v0.1",
    "core_experiment_config": "configs/optimization_v0_2/experiment.yaml",
    "core_candidate_ids": ["E00", "E09_PROCESS_CHANGE_E02"],
    "candidate_ids": {
        "m1": "M1_FROZEN",
        "c1": "C1_RECENT30_TIME",
        "c2": "C2_PROCESS_CHANGE",
        "stage2": "C3_PREREG_COMBINED",
    },
    "recent_window_days": 30,
    "m1_catboost_weights": {
        "tap_iron": 0.5,
        "tap_time_len": 0.0,
    },
    "stage2_requires_all": ["C1_RECENT30_TIME", "C2_PROCESS_CHANGE"],
    "bootstrap": {
        "block": "local_calendar_day",
        "repetitions": 2000,
        "seed": 20260908,
    },
    "evidence_scope": "development_seen_through_2024_10",
}


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
class DriftExperiment:
    optimization_id: str
    baseline_id: str
    core_experiment_config: str
    core_candidate_ids: tuple[str, str]
    candidate_ids: dict[str, str]
    recent_window_days: int
    m1_catboost_weights: dict[str, float]
    stage2_requires_all: tuple[str, str]
    bootstrap_repetitions: int
    bootstrap_seed: int
    evidence_scope: str


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            return False
        return all(_strict_equal(actual[key], expected[key]) for key in expected)
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def load_drift_experiment(path: str | Path) -> DriftExperiment:
    config = load_yaml(path)
    _exact(config, set(_CANONICAL), "optimization-v0.3 experiment")
    _exact(config["candidate_ids"], set(_CANONICAL["candidate_ids"]), "candidate_ids")
    _exact(
        config["m1_catboost_weights"],
        set(_CANONICAL["m1_catboost_weights"]),
        "m1_catboost_weights",
    )
    _exact(config["bootstrap"], set(_CANONICAL["bootstrap"]), "bootstrap")
    if not _strict_equal(config, _CANONICAL):
        raise ContractError(
            "optimization-v0.3 experiment differs from the frozen drift contract"
        )
    return DriftExperiment(
        optimization_id=config["optimization_id"],
        baseline_id=config["baseline_id"],
        core_experiment_config=config["core_experiment_config"],
        core_candidate_ids=tuple(config["core_candidate_ids"]),
        candidate_ids=dict(config["candidate_ids"]),
        recent_window_days=config["recent_window_days"],
        m1_catboost_weights={
            key: float(value)
            for key, value in config["m1_catboost_weights"].items()
        },
        stage2_requires_all=tuple(config["stage2_requires_all"]),
        bootstrap_repetitions=config["bootstrap"]["repetitions"],
        bootstrap_seed=config["bootstrap"]["seed"],
        evidence_scope=config["evidence_scope"],
    )
