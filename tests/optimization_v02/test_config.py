from copy import deepcopy
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import pytest

from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.models.optimization_v02.registry import candidate_ids


ROOT = Path(__file__).parents[2]
CONFIG_ROOT = ROOT / "configs" / "optimization" / "v0.2"


def _config_module():
    module = find_spec("bf_tap.models.optimization_v02.config")
    assert module is not None, "optimization config contract must exist"
    return import_module("bf_tap.models.optimization_v02.config")


def test_common_config_declares_pre_dev_short_rolling_folds():
    contract = _config_module()
    common = load_yaml(CONFIG_ROOT / "common.yaml")

    contract.validate_optimization_common_config(common)

    assert common["optimization_id"] == "optimization-v0.2"
    assert common["baseline_id"] == "baseline-v0.1"
    assert [fold["id"] for fold in common["folds"]] == [
        "ROLL_2024_06",
        "ROLL_2024_07",
        "ROLL_2024_08",
    ]
    assert all(
        fold["eval_end"] <= "2024-09-01T00:00:00+08:00"
        for fold in common["folds"]
    )
    assert common["selection"] == {
        "metric": "pooled_wmape",
        "per_target_max_absolute_wmape_regression": 0.002,
        "minimum_winning_fold_fraction": 2 / 3,
    }


def test_candidate_configs_match_registry_and_keep_m4_disabled():
    contract = _config_module()
    seen = []

    for path in sorted(CONFIG_ROOT.glob("m*.yaml")):
        config = load_yaml(path)
        contract.validate_optimization_candidate_config(config)
        seen.append(config["candidate_id"])

    assert tuple(seen) == candidate_ids()
    assert load_yaml(CONFIG_ROOT / "m4_ensemble.yaml")["parameters"]["enabled"] is False


def test_optimization_config_contract_rejects_unknown_keys_and_family_mismatch():
    contract = _config_module()
    common = load_yaml(CONFIG_ROOT / "common.yaml")
    invalid_common = {**common, "typo": True}
    with pytest.raises(ContractError, match="common config keys invalid"):
        contract.validate_optimization_common_config(invalid_common)

    candidate = load_yaml(CONFIG_ROOT / "m1_blend.yaml")
    invalid_candidate = deepcopy(candidate)
    invalid_candidate["family"] = "residual"
    with pytest.raises(ContractError, match="does not match registry"):
        contract.validate_optimization_candidate_config(invalid_candidate)

def test_common_config_builds_independent_baseline_validation_config():
    contract = _config_module()
    common = load_yaml(CONFIG_ROOT / "common.yaml")

    generated = contract.build_rolling_validation_config(
        common,
        timezone="Asia/Shanghai",
    )

    assert generated == {
        "schema_version": 1,
        "timezone": "Asia/Shanghai",
        "interval": "left_closed_right_open",
        "folds": common["folds"],
    }
    generated["folds"][0]["id"] = "mutated"
    assert common["folds"][0]["id"] == "ROLL_2024_06"

