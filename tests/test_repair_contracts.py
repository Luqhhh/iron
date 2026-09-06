from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from bf_tap.config import (
    load_yaml,
    validate_data_paths,
    validate_frozen_contracts,
    validate_semantic_contract,
)
from bf_tap.exceptions import ContractError, ProtectedLabelError
from bf_tap.features.history import build_history_features
from bf_tap.io import read_development_labels
from bf_tap.protection import ProtectionPolicy, read_access_ledger, record_protected_access
from bf_tap.schema import (
    validate_cross_table_consistency,
    validate_cross_table_metadata,
    validate_history,
)
from bf_tap.splits import split_from_config, validate_split_definitions


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="Asia/Shanghai")


@pytest.fixture
def protection() -> ProtectionPolicy:
    return ProtectionPolicy.from_config(
        {
            "schema_version": 1,
            "contract_id": "holdout-protection-v1",
            "timezone": "Asia/Shanghai",
            "development_label_end_exclusive": "2024-11-01T00:00:00+08:00",
            "protected_interval": {
                "start": "2024-11-01T00:00:00+08:00",
                "end": "2024-12-01T00:00:00+08:00",
            },
            "allowed_protected_lifecycles": ["holdout_scoring", "final_training"],
        }
    )


def test_split_rejects_overlap_duplicate_id_naive_timezone_and_unknown_kind(protection):
    valid = {
        "id": "DEV",
        "kind": "development",
        "train_start": "2024-03-01T00:00:00+08:00",
        "fit_cutoff": "2024-07-01T00:00:00+08:00",
        "eval_start": "2024-07-01T00:00:00+08:00",
        "eval_end": "2024-11-01T00:00:00+08:00",
    }
    validate_split_definitions([valid], protection)
    with pytest.raises(ContractError, match="temporal order"):
        validate_split_definitions([{**valid, "eval_start": "2024-06-01T00:00:00+08:00"}], protection)
    with pytest.raises(ContractError, match="duplicate fold id"):
        validate_split_definitions([valid, valid], protection)
    with pytest.raises(ContractError, match="timezone-aware"):
        validate_split_definitions([{**valid, "fit_cutoff": "2024-07-01"}], protection)
    with pytest.raises(ContractError, match="unsupported fold kind"):
        validate_split_definitions([{**valid, "kind": "renamed_development"}], protection)
    with pytest.raises(ContractError, match="protected label boundary"):
        validate_split_definitions(
            [{**valid, "eval_end": "2024-12-01T00:00:00+08:00"}], protection
        )


def test_select_split_rejects_sample_id_overlap(protection):
    split = split_from_config(
        {
            "id": "BAD",
            "kind": "development",
            "train_start": "2024-03-01T00:00:00+08:00",
            "fit_cutoff": "2024-07-01T00:00:00+08:00",
            "eval_start": "2024-07-01T00:00:00+08:00",
            "eval_end": "2024-11-01T00:00:00+08:00",
        }
    )
    # Duplicate physical rows put S1 on opposite sides of the time boundary.
    frame = pd.DataFrame(
        {
            "sample_id": ["S1", "S1"],
            "reference_time": [ts("2024-06-30"), ts("2024-07-01")],
            "label_available_at": [ts("2024-06-30 01:00"), ts("2024-07-01 01:00")],
        }
    )
    from bf_tap.splits import select_split

    with pytest.raises(ContractError, match="overlap"):
        select_split(frame, split)


def test_development_label_reader_cannot_cross_independent_protection_boundary(
    tmp_path, protection
):
    source = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "sample_id": ["safe", "protected"],
            "tap_no": [1, 2],
            "spout_no": [1, 2],
            "reference_time": ["2024-10-31 23:00:00", "2024-11-01 00:00:00"],
            "tap_iron": [1.0, 999.0],
            "tap_time_len": [2.0, 999.0],
        }
    ).to_csv(source, index=False)
    with pytest.raises(ProtectedLabelError, match="protection boundary"):
        read_development_labels(
            source,
            requested_end="2024-12-01T00:00:00+08:00",
            protection_policy=protection,
        )


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["S1"],
            "tap_no": [1],
            "spout_no": [2],
            "reference_time": [ts("2024-07-01 10:00")],
            "tap_end_time": [ts("2024-07-01 11:00")],
            "available_at": [ts("2024-07-01 11:00")],
            "tap_iron": [100.0],
            "tap_time_len": [60.0],
        }
    )


def test_history_rejects_time_travel_and_cross_table_conflicts():
    history = _history()
    bad = history.copy()
    bad.loc[0, "reference_time"] = ts("2024-07-02")
    with pytest.raises(ContractError, match="reference_time <= tap_end_time"):
        validate_history(bad)

    samples = history.drop(columns=["tap_end_time", "available_at"]).copy()
    conflict = history.copy()
    conflict.loc[0, "tap_iron"] = 101.0
    with pytest.raises(ContractError, match="target mismatch"):
        validate_cross_table_consistency(samples, conflict, target_atol=1e-9)
    conflict = history.copy()
    conflict.loc[0, "spout_no"] = 1
    with pytest.raises(ContractError, match="metadata mismatch"):
        validate_cross_table_metadata(samples.drop(columns=["tap_iron", "tap_time_len"]), conflict)


def test_history_feature_entry_rejects_time_travel():
    history = _history()
    history.loc[0, "reference_time"] = ts("2024-07-02")
    samples = pd.DataFrame(
        {
            "sample_id": ["E1"],
            "spout_no": [2],
            "reference_time": [ts("2024-07-03")],
        }
    )
    with pytest.raises(ContractError, match="reference_time <= tap_end_time"):
        build_history_features(samples, history, fit_cutoff=ts("2024-07-03"))


def test_strict_config_validation_is_command_scoped():
    paths = {
        "schema_version": 1,
        "paths": {
            "train_samples": "/tmp/train.csv",
            "operation_hourly": "/tmp/op.csv",
            "burden_change": "/tmp/burden.csv",
            "tap_history_train": "/tmp/history.csv",
            "data_dictionary": "/tmp/dict.xlsx",
        },
    }
    validate_data_paths(paths, command="development")
    predict_only = {
        "schema_version": 1,
        "paths": {
            "test_a_samples": "/tmp/a.csv",
            "operation_hourly": "/tmp/op.csv",
            "burden_change": "/tmp/burden.csv",
        },
    }
    validate_data_paths(predict_only, command="predict", stage="test_a")
    validate_data_paths(predict_only, command="check-submission", stage="test_a")
    with pytest.raises(ContractError, match="unknown keys"):
        validate_data_paths({**paths, "typo": True}, command="development")
    with pytest.raises(ContractError, match="missing required"):
        validate_data_paths(
            {"schema_version": 1, "paths": {"train_samples": "/tmp/train.csv"}},
            command="development",
        )
    with pytest.raises(ContractError, match="must not be empty"):
        validate_data_paths(
            {"schema_version": 1, "paths": {}},
            command="audit",
        )

    semantic = {
        "schema_version": 1,
        "contract_id": "competition-timestamp-contract-v1",
        "evidence_status": "ASSUMED",
        "timezone": "Asia/Shanghai",
        "sources": {
            "operation": {"event_time_column": "clock", "available_at_column": "clock", "missing_markers": [""]},
            "burden": {"event_time_column": "cal_time", "available_at_column": "cal_time", "missing_markers": [""]},
            "history": {"event_time_column": "reference_time", "end_time_column": "tap_end_time", "available_at_column": "tap_end_time"},
        },
        "targets": {"available_at_column": "tap_end_time"},
        "evidence": {
            "assets": [
                {
                    "asset_id": "dictionary",
                    "location": "dictionary.xlsx",
                    "sha256": "0" * 64,
                }
            ],
            "fields": {
                name: {
                    "status": "ASSUMED",
                    "asset_id": "dictionary",
                    "location": name,
                    "conclusion": "synthetic field-level evidence",
                }
                for name in (
                    "operation.clock",
                    "burden.cal_time",
                    "history.tap_end_time",
                    "targets.tap_end_time",
                )
            },
            "pending": ["reporting delay"],
        },
    }
    validate_semantic_contract(semantic)


def test_protected_access_ledger_is_explicit_and_policy_bound(tmp_path, protection):
    ledger = tmp_path / "ledger.json"
    assert read_access_ledger(ledger, protection)["consumed"] is False
    with pytest.raises(ProtectedLabelError, match="unauthorized"):
        record_protected_access(
            ledger,
            protection,
            lifecycle="development",
            frozen_manifest_sha256="0" * 64,
        )
    recorded = record_protected_access(
        ledger,
        protection,
        lifecycle="holdout_scoring",
        frozen_manifest_sha256="a" * 64,
    )
    assert recorded["consumed"] is True
    assert read_access_ledger(ledger, protection)["entries"][0]["lifecycle"] == "holdout_scoring"


def test_frozen_contract_digests_reject_every_semantic_family_drift():
    root = Path(__file__).parents[1]
    baseline = load_yaml(root / "configs/baseline.yaml")
    features = load_yaml(root / "configs/features.yaml")
    semantic = load_yaml(root / "configs/data_contract.yaml")
    identities = validate_frozen_contracts(baseline, features, semantic)
    assert set(identities) == {
        "baseline_contract_sha256",
        "feature_contract_sha256",
        "semantic_contract_sha256",
    }

    mutations = []
    changed = deepcopy(features)
    changed["sample"]["derived"] = ["hour_sin", "hour_cos"]
    mutations.append((baseline, changed, semantic))
    changed = deepcopy(features)
    changed["operation"]["value_columns"] = changed["operation"]["value_columns"][:-1]
    mutations.append((baseline, changed, semantic))
    changed = deepcopy(features)
    changed["operation"]["latest_max_event_age_hours"] = 25
    mutations.append((baseline, changed, semantic))
    changed = deepcopy(features)
    changed["burden"]["latest_max_event_age_hours"] = 73
    mutations.append((baseline, changed, semantic))
    changed = deepcopy(features)
    changed["history"]["exclude_current_tap"] = False
    mutations.append((baseline, changed, semantic))
    changed = deepcopy(semantic)
    changed["evidence"]["pending"].append("new unresolved semantic")
    mutations.append((baseline, features, changed))
    changed = deepcopy(baseline)
    changed["max_categorical_cardinality"] = 65
    mutations.append((changed, features, semantic))

    for baseline_value, feature_value, semantic_value in mutations:
        with pytest.raises(ContractError, match="contract SHA256 mismatch"):
            validate_frozen_contracts(
                baseline_value, feature_value, semantic_value
            )
