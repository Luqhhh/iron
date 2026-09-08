import json

import pandas as pd
import pytest

from bf_tap.artifacts import atomic_write_json, build_inference_source_contract, file_identities
from bf_tap.exceptions import ContractError
from bf_tap.optimization.indexed_features import load_known_index
from bf_tap.optimization import v3_verify_profiles as verifier


def test_profile_verifier_requires_completed_run_before_data_read(tmp_path):
    atomic_write_json(tmp_path / "final_status.json", {"status": "FAILED"})
    with pytest.raises(ContractError, match="completed"):
        verifier.verify_profiles(tmp_path, tmp_path / "out", "nonexistent.yaml", ["nonexistent.csv"])


def test_profile_verifier_all_origins_metadata_only_and_prediction_mismatch(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    atomic_write_json(source / "final_status.json", {"engineering_status": "G0_EXECUTION_PASS"})
    index_path = tmp_path / "index.csv"
    pd.DataFrame({"sample_id": ["a", "b"], "spout_no": [1, 2],
                  "reference_time": ["2024-07-01", "2024-07-02"],
                  "tap_iron": ["DO_NOT_READ", "DO_NOT_READ"]}).to_csv(index_path, index=False)
    _, audit = load_known_index([index_path])
    atomic_write_json(source / "known_index_manifest.json", audit)
    event_path = tmp_path / "events.csv"
    pd.DataFrame({"event_time": ["2024-06-01"], "value": [1.]}).to_csv(event_path, index=False)
    paths = {"operation_hourly": str(event_path), "burden_change": str(event_path)}
    data_path = tmp_path / "data.yaml"
    data_path.write_text(json.dumps({"schema_version": 1, "paths": paths}))
    identities = file_identities(paths)
    mapping = {"event_time_column": "event_time", "available_at_column": "event_time", "missing_markers": []}
    metadata = {
        "inference_source_contract": build_inference_source_contract(identities, semantic_contract_sha256="a" * 64),
        "contract_digests": {"semantic_contract_sha256": "a" * 64},
        "semantic_contract": {"sources": {kind: mapping for kind in ("operation", "burden")}},
        "feature_config": {kind: {"value_columns": ["value"]} for kind in ("operation", "burden")},
    }
    expected = pd.DataFrame({"sample_id": ["a", "b"], "pred_tap_iron": [12., 13.], "pred_tap_time_len": [9., 10.]})
    metrics = {}
    for n in range(7):
        unit, origin = f"unit{n}", f"origin{n}"
        metrics[unit] = {"origin_id": origin}
        for profile in ("F-C", "F-D"):
            bundle = source / "bundles" / origin / profile
            bundle.mkdir(parents=True)
            atomic_write_json(bundle / "bundle.json", {"synthetic": True})
            destination = source / "units" / unit / profile
            destination.mkdir(parents=True)
            expected.to_csv(destination / "predictions.csv", index=False)
    atomic_write_json(source / "candidate_metrics.json", metrics)

    class Predictor:
        def __init__(self, path):
            self.metadata = metadata

        def predict(self, queries, operation, burden, index):
            assert list(queries.columns) == ["sample_id", "spout_no", "reference_time"]
            assert list(index.columns) == list(queries.columns)
            return expected.copy()

    monkeypatch.setattr(verifier, "ProfilePredictor", Predictor)
    verifier.verify_profiles(source, tmp_path / "out", str(data_path), [str(index_path)])
    status = json.loads((tmp_path / "out" / "final_status.json").read_text())
    assert status["bundles_checked"] == 14 and status["max_absolute_difference"] == 0
    assert status["targets_read"] is False
    expected.assign(pred_tap_iron=100.).to_csv(source / "units" / "unit0" / "F-C" / "predictions.csv", index=False)
    with pytest.raises(ContractError, match="differs"):
        verifier.verify_profiles(source, tmp_path / "out2", str(data_path), [str(index_path)])
