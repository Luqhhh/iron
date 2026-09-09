from pathlib import Path

import pytest
import yaml

from bf_tap import cli


def arguments():
    return [
        "optimize-v0.3",
        "--data-config",
        "configs/m234.local.yaml",
        "--output",
        "local/run-v03",
    ]


def test_v03_parser_exposes_only_frozen_all_suite_defaults():
    args = cli._parser().parse_args(arguments())

    assert args.command == "optimize-v0.3"
    assert args.data_config == "configs/m234.local.yaml"
    assert args.baseline_config == "configs/baseline.yaml"
    assert args.feature_config == "configs/features.yaml"
    assert args.data_contract == "configs/data_contract.yaml"
    assert args.protection_policy == "configs/protection.yaml"
    assert args.protection_ledger == "local/manifests/protected_access.json"
    assert args.drift_experiment_config == (
        "configs/optimization_v0_3/experiment.yaml"
    )
    assert args.core_experiment_config == (
        "configs/optimization_v0_2/experiment.yaml"
    )
    assert args.optimization_feature_config == (
        "configs/optimization_v0_2/features.yaml"
    )
    assert args.optimization_validation_config == (
        "configs/optimization_v0_2/validation.yaml"
    )
    assert args.optimization_acceptance_config == (
        "configs/optimization_v0_2/acceptance.yaml"
    )
    assert args.optimization_model_config == (
        "configs/optimization_v0_2/models/catboost.yaml"
    )
    assert not hasattr(args, "suite")
    assert not hasattr(args, "optimization_candidates")


def test_v03_main_forwards_every_frozen_path(monkeypatch, capsys):
    received = {}

    def fake_run(**kwargs):
        received.update(kwargs)
        return Path("local/run-v03")

    monkeypatch.setattr(cli, "run_drift_validation", fake_run)

    assert cli.main(arguments()) == 0
    assert received == {
        "data_config_path": "configs/m234.local.yaml",
        "baseline_config_path": "configs/baseline.yaml",
        "feature_config_path": "configs/features.yaml",
        "semantic_contract_path": "configs/data_contract.yaml",
        "protection_policy_path": "configs/protection.yaml",
        "protection_ledger_path": "local/manifests/protected_access.json",
        "drift_experiment_config_path": (
            "configs/optimization_v0_3/experiment.yaml"
        ),
        "core_experiment_config_path": (
            "configs/optimization_v0_2/experiment.yaml"
        ),
        "optimization_feature_config_path": (
            "configs/optimization_v0_2/features.yaml"
        ),
        "optimization_validation_config_path": (
            "configs/optimization_v0_2/validation.yaml"
        ),
        "optimization_acceptance_config_path": (
            "configs/optimization_v0_2/acceptance.yaml"
        ),
        "optimization_model_config_path": (
            "configs/optimization_v0_2/models/catboost.yaml"
        ),
        "output": "local/run-v03",
    }
    assert capsys.readouterr().out.strip() == "local/run-v03"
def test_cli_rejects_noncanonical_protection_policy(tmp_path):
    config = yaml.safe_load(Path("configs/protection.yaml").read_text())
    config["development_label_end_exclusive"] = "2024-12-01T00:00:00+08:00"
    shifted = tmp_path / "protection.yaml"
    shifted.write_text(yaml.safe_dump(config))

    with pytest.raises(Exception, match="canonical"):
        cli._validate_v03_protection(shifted)
