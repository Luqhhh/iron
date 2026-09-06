import json

import numpy as np
import pandas as pd
import pytest

from bf_tap.models.baseline import DualTargetBaseline
from bf_tap.artifacts import stable_digest
from bf_tap.exceptions import ContractError


@pytest.mark.model
def test_dual_model_roundtrip_is_identical(tmp_path):
    parameters = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 800,
        "depth": 5,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5.0,
        "random_seed": 2026,
        "task_type": "CPU",
        "thread_count": 8,
        "bootstrap_type": "No",
        "random_strength": 0.0,
        "rsm": 1.0,
        "boosting_type": "Plain",
        "has_time": True,
        "one_hot_max_size": 64,
        "nan_mode": "Min",
        "use_best_model": False,
        "allow_writing_files": False,
        "verbose": False,
    }
    X = pd.DataFrame(
        {
            "spout_no": [str(1 + i % 2) for i in range(40)],
            "hour_sin": np.sin(np.arange(40)),
            "feature": np.arange(40, dtype=float),
        }
    )
    y = pd.DataFrame(
        {"tap_iron": 400 + np.arange(40) * 0.5, "tap_time_len": 100 + np.arange(40) % 7}
    )
    model = DualTargetBaseline(parameters).fit(X, y)
    before = model.predict_raw(X)
    bundle = tmp_path / "bundle"
    history = pd.DataFrame(
        {
            "sample_id": ["h"],
            "tap_no": [1],
            "spout_no": [1],
            "reference_time": ["2024-01-01T00:00:00+08:00"],
            "tap_end_time": ["2024-01-01T01:00:00+08:00"],
            "available_at": ["2024-01-01T01:00:00+08:00"],
            "tap_iron": [1.0],
            "tap_time_len": [1.0],
        }
    )
    metadata = {
        "baseline_config": {"baseline_id": "test"},
        "semantic_contract": {"contract_id": "test"},
        "feature_config": {"version": 1},
        "training": {
            "fit_cutoff": "2024-01-02T00:00:00+08:00",
            "history_origin_sha256": stable_digest(
                history[["sample_id", "available_at"]].astype(str).to_dict("records")
            ),
        },
        "code_identity": {"snapshot": "test"},
        "environment": {"python": "test"},
        "lockfile_sha256": "0" * 64,
        "contract_digests": {
            "baseline_contract_sha256": "1" * 64,
            "feature_contract_sha256": "2" * 64,
            "semantic_contract_sha256": "3" * 64,
        },
        "inference_source_contract": {
            "schema_version": 1,
            "contract_id": "synthetic-sources-v1",
            "sources": {
                "operation_hourly": {"sha256": "4" * 64, "bytes": 1},
                "burden_change": {"sha256": "5" * 64, "bytes": 1},
            },
        },
    }
    model.save(bundle, metadata=metadata, history_snapshot=history)
    after = DualTargetBaseline.load(bundle).predict_raw(X)
    assert np.max(np.abs(before.to_numpy() - after.to_numpy())) <= 1e-9
    single = DualTargetBaseline.load(bundle).predict_raw(X.iloc[[7]])
    assert np.max(np.abs(single.to_numpy() - after.iloc[[7]].to_numpy())) <= 1e-9
    metadata = json.loads((bundle / "bundle.json").read_text())
    assert metadata["feature_names"] == list(X.columns)
    assert sorted(p.name for p in bundle.glob("*.cbm")) == ["tap_iron.cbm", "tap_time_len.cbm"]
    assert DualTargetBaseline.load(bundle).load_history_snapshot().sample_id.tolist() == ["h"]
    component = bundle / "tap_iron.cbm"
    component.write_bytes(component.read_bytes() + b"tampered")
    with pytest.raises(ContractError, match="component hash mismatch"):
        DualTargetBaseline.load(bundle)
