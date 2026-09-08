from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError, ProtectedLabelError
from bf_tap.optimization.config import (
    Candidate,
    load_experiment,
    load_feature_selection,
    load_model_config,
    load_validation,
)
from bf_tap.optimization.features import select_candidate_features
from bf_tap.optimization.run import run_optimization_validation


ROOT = Path(__file__).parents[1]


def _write(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_registered_first_batch_is_strict_and_has_14_grid_cells(tmp_path):
    config, candidates = load_experiment(ROOT / "configs/optimization_v0_2/experiment.yaml")
    assert [candidate.id for candidate in candidates] == config["candidate_order"]
    assert len(candidates) == 18
    by_id = {candidate.id: candidate for candidate in candidates}
    assert by_id["E07_FROZEN_E02"].history_view_ages_days == (0, 7, 30, 60, 90)
    assert by_id["E08_FROZEN_E04"].history_view_ages_days == (0, 7, 30, 60, 90)
    assert by_id["E12_BLEND_E09_E04_80_20"].component_candidates == (
        "E09_PROCESS_CHANGE_E02", "E04"
    )
    assert by_id["E14_TIMECAL_E09"].residual_calibration == {
        "tap_iron": 0.0,
        "tap_time_len": 1.68610975,
    }
    assert by_id["E15_TARGETWISE_E12_E14"].component_candidates == (
        "E12_BLEND_E09_E04_80_20", "E14_TIMECAL_E09"
    )
    assert by_id["E16_TIMECAL_E12"].base_candidate == "E12_BLEND_E09_E04_80_20"
    _, screening, origins = load_validation(
        ROOT / "configs/optimization_v0_2/validation.yaml",
        expected_timezone="Asia/Shanghai",
    )
    assert [fold.id for fold in screening] == ["DEV_LONG", "DEV_SHORT"]
    assert sum(origin.horizons for origin in origins) == 14
    assert load_model_config(
        ROOT / "configs/optimization_v0_2/models/catboost.yaml"
    )["model_type"] == "frozen_baseline_catboost"

    invalid = deepcopy(config)
    invalid["candidates"]["E00"]["typo"] = True
    path = tmp_path / "invalid.yaml"
    _write(path, invalid)
    with pytest.raises(ContractError, match="unknown"):
        load_experiment(path)

    invalid_ages = deepcopy(config)
    invalid_ages["candidates"]["E07_FROZEN_E02"]["history_view_ages_days"] = [7, 0]
    ages_path = tmp_path / "invalid-ages.yaml"
    _write(ages_path, invalid_ages)
    with pytest.raises(ContractError, match="history_view_ages_days"):
        load_experiment(ages_path)

    invalid_order = deepcopy(config)
    invalid_order["candidate_order"].remove("E15_TARGETWISE_E12_E14")
    invalid_order["candidate_order"].insert(0, "E15_TARGETWISE_E12_E14")
    order_path = tmp_path / "invalid-order.yaml"
    _write(order_path, invalid_order)
    with pytest.raises(ContractError, match="component_candidates"):
        load_experiment(order_path)


def test_feature_ablations_select_only_the_declared_components():
    selection = load_feature_selection(ROOT / "configs/optimization_v0_2/features.yaml")
    frame = pd.DataFrame(
        {
            "spout_no": pd.Series(["1"], dtype="string"),
            "hour_sin": [0.1],
            "hour_cos": [0.2],
            "weekday": [1.0],
            "operation__air__latest": [2.0],
            "burden__pig__latest": [3.0],
            "history__all__latest_available_age_minutes": [4.0],
            "history__all__tap_iron__latest": [5.0],
            "history__all__tap_iron__last3_mean": [6.0],
            "history__all__tap_iron__last3_count": [3.0],
        }
    )
    _, candidates = load_experiment(ROOT / "configs/optimization_v0_2/experiment.yaml")
    by_id = {candidate.id: candidate for candidate in candidates}
    e01 = select_candidate_features(frame, by_id["E01"], selection)
    assert not any(column.startswith("history__") for column in e01)
    e02 = select_candidate_features(frame, by_id["E02"], selection)
    assert "history__all__latest_available_age_minutes" not in e02
    assert "history__all__tap_iron__latest" in e02
    assert "history__all__tap_iron__last3_count" in e02
    e03 = select_candidate_features(frame, by_id["E03"], selection)
    assert "history__all__latest_available_age_minutes" in e03
    assert "history__all__tap_iron__last3_count" in e03
    assert "history__all__tap_iron__latest" not in e03
    assert "history__all__tap_iron__last3_mean" not in e03
    e04 = select_candidate_features(frame, by_id["E04"], selection)
    assert "operation__air__latest" not in e04
    assert "burden__pig__latest" not in e04

    with_changes = frame.assign(
        process_change__air__latest_minus_6h_mean=[0.5]
    )
    e09 = select_candidate_features(
        with_changes, by_id["E09_PROCESS_CHANGE_E02"], selection
    )
    assert "process_change__air__latest_minus_6h_mean" in e09
    unchanged = select_candidate_features(with_changes, by_id["E02"], selection)
    assert "process_change__air__latest_minus_6h_mean" not in unchanged


def test_unclassified_feature_fails_closed():
    selection = load_feature_selection(ROOT / "configs/optimization_v0_2/features.yaml")
    frame = pd.DataFrame({"spout_no": pd.Series(["1"], dtype="string"), "mystery": [1.0]})
    candidate = Candidate("X", "model", ("sample",), ())
    with pytest.raises(ContractError, match="unclassified"):
        select_candidate_features(frame, candidate, selection)


def test_optimization_grid_crossing_protection_boundary_fails_before_file_reads(tmp_path):
    validation = load_yaml(ROOT / "configs/optimization_v0_2/validation.yaml")
    validation["origins"] = [
        {
            "id": "FORBIDDEN",
            "fit_cutoff": "2024-11-01T00:00:00+08:00",
            "horizons": 1,
        }
    ]
    validation_path = tmp_path / "validation.yaml"
    _write(validation_path, validation)
    data_path = tmp_path / "data.yaml"
    _write(
        data_path,
        {
            "schema_version": 1,
            "paths": {
                "train_samples": "/must/not/be/read/train.csv",
                "tap_history_train": "/must/not/be/read/history.csv",
                "operation_hourly": "/must/not/be/read/operation.csv",
                "burden_change": "/must/not/be/read/burden.csv",
                "data_dictionary": "/must/not/be/read/dictionary.xlsx",
            },
        },
    )
    with pytest.raises(ProtectedLabelError, match="protection boundary"):
        run_optimization_validation(
            data_config_path=data_path,
            baseline_config_path=ROOT / "configs/baseline.yaml",
            feature_config_path=ROOT / "configs/features.yaml",
            semantic_contract_path=ROOT / "configs/data_contract.yaml",
            protection_policy_path=ROOT / "configs/protection.yaml",
            protection_ledger_path=None,
            experiment_config_path=ROOT / "configs/optimization_v0_2/experiment.yaml",
            optimization_feature_config_path=ROOT / "configs/optimization_v0_2/features.yaml",
            optimization_validation_config_path=validation_path,
            optimization_acceptance_config_path=ROOT / "configs/optimization_v0_2/acceptance.yaml",
            optimization_model_config_path=ROOT / "configs/optimization_v0_2/models/catboost.yaml",
            suite="grid",
            output=tmp_path / "forbidden-run",
        )
    assert not Path("/must/not/be/read/train.csv").exists()
