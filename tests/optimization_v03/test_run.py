from pathlib import Path
import inspect

import pandas as pd
import pytest
import yaml

from bf_tap.exceptions import ContractError
from bf_tap.optimization_v03.config import load_drift_experiment
from bf_tap.optimization_v03.run import (
    assert_c2_equivalence,
    normalize_stage1_acceptance,
    quality_status,
    run_drift_validation,
    stage2_decision,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = load_drift_experiment(
    ROOT / "configs/optimization_v0_3/experiment.yaml"
)


def acceptance(c1, c2, m1=True):
    return {
        "status": "PASS",
        "candidates": {
            "M1_FROZEN": {"pass": m1},
            "C1_RECENT30_TIME": {"pass": c1},
            "C2_PROCESS_CHANGE": {"pass": c2},
        },
    }


@pytest.mark.parametrize(
    ("c1", "c2", "expected_status", "failed"),
    [
        (True, True, "RUN", []),
        (False, True, "SKIPPED", ["C1_RECENT30_TIME"]),
        (True, False, "SKIPPED", ["C2_PROCESS_CHANGE"]),
        (
            False,
            False,
            "SKIPPED",
            ["C1_RECENT30_TIME", "C2_PROCESS_CHANGE"],
        ),
    ],
)
def test_stage2_requires_both_preregistered_stage1_candidates(
    c1, c2, expected_status, failed
):
    decision = stage2_decision(acceptance(c1, c2), EXPERIMENT)

    assert decision["status"] == expected_status
    assert decision["failed_prerequisites"] == failed
    assert decision["candidate_id"] == "C3_PREREG_COMBINED"


def test_quality_status_ignores_passing_m1_control():
    assert quality_status(acceptance(False, False, m1=True), EXPERIMENT) == (
        "G1_FAIL_OPTIMIZATION_ACCEPTANCE"
    )
    assert quality_status(acceptance(True, False, m1=False), EXPERIMENT) == (
        "G1_PASS_OPTIMIZATION_DEVELOPMENT"
    )

def test_stage1_top_level_status_ignores_m1_control():
    value = normalize_stage1_acceptance(acceptance(False, False, m1=True), EXPERIMENT)

    assert value["status"] == "FAIL"
    assert value["candidates"]["M1_FROZEN"]["pass"] is True




def prediction(iron=(1.0, 2.0), time=(3.0, 4.0)):

    return pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "pred_tap_iron": iron,
            "pred_tap_time_len": time,
        }
    )


def test_c2_equivalence_is_exact_on_keys_and_values():
    assert assert_c2_equivalence(prediction(), prediction()) == {
        "rows": 2,
        "max_abs_diff": 0.0,
        "pass": True,
    }
    with pytest.raises(ContractError, match="differs"):
        assert_c2_equivalence(prediction(), prediction(iron=(1.0, 2.000001)))
    with pytest.raises(ContractError, match="sample_id"):
        assert_c2_equivalence(
            prediction(),
            prediction().assign(sample_id=["a", "c"]),
        )


def test_existing_output_is_rejected_before_any_config_read(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        run_drift_validation(
            data_config_path="missing-data.yaml",
            baseline_config_path="missing-baseline.yaml",
            feature_config_path="missing-features.yaml",
            semantic_contract_path="missing-contract.yaml",
            protection_policy_path="missing-protection.yaml",
            protection_ledger_path=None,
            drift_experiment_config_path="missing-drift.yaml",
            core_experiment_config_path="missing-core.yaml",
            optimization_feature_config_path="missing-opt-features.yaml",
            optimization_validation_config_path="missing-validation.yaml",
            optimization_acceptance_config_path="missing-acceptance.yaml",
            optimization_model_config_path="missing-model.yaml",
            output=existing,
        )
def test_runner_rejects_shifted_protection_and_validation_before_label_read(tmp_path, monkeypatch):
    protection = yaml.safe_load((ROOT / "configs/protection.yaml").read_text())
    protection["development_label_end_exclusive"] = "2024-12-01T00:00:00+08:00"
    protection["protected_interval"]["start"] = "2024-12-01T00:00:00+08:00"
    shifted_protection = tmp_path / "protection.yaml"
    shifted_protection.write_text(yaml.safe_dump(protection))
    validation = yaml.safe_load(
        (ROOT / "configs/optimization_v0_2/validation.yaml").read_text()
    )
    for fold in validation["screening_folds"]:
        fold["eval_end"] = "2024-12-01T00:00:00+08:00"
    shifted_validation = tmp_path / "validation.yaml"
    shifted_validation.write_text(yaml.safe_dump(validation))
    monkeypatch.setattr(
        "bf_tap.optimization_v03.run.read_development_labels",
        lambda *args, **kwargs: pytest.fail("labels must not be read"),
    )
    with pytest.raises(ContractError, match="canonical protection"):
        run_drift_validation(
            data_config_path=ROOT / "configs/m234.local.yaml",
            baseline_config_path=ROOT / "configs/baseline.yaml",
            feature_config_path=ROOT / "configs/features.yaml",
            semantic_contract_path=ROOT / "configs/data_contract.yaml",
            protection_policy_path=shifted_protection,
            protection_ledger_path=None,
            drift_experiment_config_path=ROOT / "configs/optimization_v0_3/experiment.yaml",
            core_experiment_config_path=ROOT / "configs/optimization_v0_2/experiment.yaml",
            optimization_feature_config_path=ROOT / "configs/optimization_v0_2/features.yaml",
            optimization_validation_config_path=shifted_validation,
            optimization_acceptance_config_path=ROOT / "configs/optimization_v0_2/acceptance.yaml",
            optimization_model_config_path=ROOT / "configs/optimization_v0_2/models/catboost.yaml",
            output=tmp_path / "run",
        )
def test_runner_failure_keeps_protected_label_state_unverified(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(FileNotFoundError):
        run_drift_validation(
            data_config_path="missing-data.yaml",
            baseline_config_path="missing-baseline.yaml",
            feature_config_path="missing-features.yaml",
            semantic_contract_path="missing-contract.yaml",
            protection_policy_path="missing-protection.yaml",
            protection_ledger_path=None,
            drift_experiment_config_path="missing-drift.yaml",
            core_experiment_config_path="missing-core.yaml",
            optimization_feature_config_path="missing-opt-features.yaml",
            optimization_validation_config_path="missing-validation.yaml",
            optimization_acceptance_config_path="missing-acceptance.yaml",
            optimization_model_config_path="missing-model.yaml",
            output=output,
        )
    assert yaml.safe_load((output / "final_status.json").read_text())[
        "protected_labels_read"
    ] == "UNVERIFIED"
def test_stage1_top_level_status_ignores_m1_control():
    value = normalize_stage1_acceptance(acceptance(False, False, m1=True), EXPERIMENT)

    assert value["status"] == "FAIL"
    assert value["candidates"]["M1_FROZEN"]["pass"] is True

def test_safety_checkpoint_does_not_rewrite_run_state():
    source = inspect.getsource(run_drift_validation)

    assert "atomic_write_json(destination / \"run_state.json\", run_state)" not in source
