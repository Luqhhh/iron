from pathlib import Path

import pytest
import yaml

from bf_tap.exceptions import ContractError
from bf_tap.optimization_v03.config import load_drift_experiment


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/optimization_v0_3/experiment.yaml"


def test_canonical_drift_experiment_loads_frozen_decisions():
    experiment = load_drift_experiment(CONFIG)

    assert experiment.optimization_id == "optimization-v0.3-drift"
    assert experiment.baseline_id == "baseline-v0.1"
    assert experiment.core_candidate_ids == ("E00", "E09_PROCESS_CHANGE_E02")
    assert experiment.recent_window_days == 30
    assert experiment.m1_catboost_weights == {
        "tap_iron": 0.5,
        "tap_time_len": 0.0,
    }
    assert experiment.candidate_ids == {
        "m1": "M1_FROZEN",
        "c1": "C1_RECENT30_TIME",
        "c2": "C2_PROCESS_CHANGE",
        "stage2": "C3_PREREG_COMBINED",
    }
    assert experiment.stage2_requires_all == (
        "C1_RECENT30_TIME",
        "C2_PROCESS_CHANGE",
    )
    assert experiment.bootstrap_repetitions == 2000
    assert experiment.bootstrap_seed == 20260908
    assert experiment.evidence_scope == "development_seen_through_2024_10"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), 2),
        (("optimization_id",), "optimization-v0.2"),
        (("baseline_id",), "baseline-v9"),
        (("core_experiment_config",), "other.yaml"),
        (("core_candidate_ids",), ["E00"]),
        (("candidate_ids", "c1"), "SEARCHED_C1"),
        (("recent_window_days",), 31),
        (("m1_catboost_weights", "tap_iron"), 0.75),
        (("m1_catboost_weights", "tap_time_len"), 0.25),
        (("stage2_requires_all",), ["C1_RECENT30_TIME"]),
        (("bootstrap", "block"), "row"),
        (("bootstrap", "repetitions"), 1999),
        (("bootstrap", "seed"), 1),
        (("evidence_scope",), "independent_validation"),
    ],
)
def test_any_frozen_decision_change_is_rejected(tmp_path, path, value):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    changed = tmp_path / "experiment.yaml"
    changed.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ContractError):
        load_drift_experiment(changed)


def test_unknown_config_key_is_rejected(tmp_path):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["search_grid"] = [14, 30, 60]
    changed = tmp_path / "experiment.yaml"
    changed.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ContractError, match="keys invalid"):
        load_drift_experiment(changed)

@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("recent_window_days",), True),
        (("bootstrap", "repetitions"), True),
        (("m1_catboost_weights", "tap_time_len"), False),
    ],
)
def test_boolean_scalars_cannot_equal_frozen_numeric_scalars(tmp_path, path, value):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    changed = tmp_path / "experiment.yaml"
    changed.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ContractError):
        load_drift_experiment(changed)
