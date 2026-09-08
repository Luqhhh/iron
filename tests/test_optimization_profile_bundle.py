import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bf_tap.artifacts import atomic_write_json, file_identities, file_sha256
from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.features import build_features
from bf_tap.optimization.config import load_experiment, load_feature_selection
from bf_tap.optimization.features import select_candidate_features
from bf_tap.optimization.indexed_features import load_known_index
from bf_tap.optimization.models import FrozenBaselineAdapter
from bf_tap.optimization.process_change import add_process_change_features
from bf_tap.optimization.v3_remaining_features import (ProfilePredictor, extend_features, load_profiles, save_profile_model)


@pytest.mark.parametrize("profile", ["F-C", "F-D"])
def test_profile_bundle_cold_process_and_batch_invariance(tmp_path, profile):
    feature_config = load_yaml("configs/features.yaml")
    selection = load_feature_selection("configs/optimization_v0_2/features.yaml")
    baseline = load_yaml("configs/baseline.yaml")
    _, candidates = load_experiment("configs/optimization_v0_2/experiment.yaml")
    candidate = next(c for c in candidates if c.id == "E09_PROCESS_CHANGE_E02")
    times = pd.date_range("2024-07-02", periods=36, freq="2h", tz="Asia/Shanghai")
    samples = pd.DataFrame({"sample_id": [f"s{i:03d}" for i in range(36)], "tap_no": np.arange(36),
                            "spout_no": np.arange(36) % 2 + 1, "reference_time": times,
                            "tap_iron": 450. + np.arange(36) % 9, "tap_time_len": 100. + np.arange(36) % 6})
    samples["label_available_at"] = samples.reference_time + pd.Timedelta(hours=1)
    history = samples.assign(available_at=samples.label_available_at, tap_end_time=samples.label_available_at)
    op_times = pd.date_range("2024-07-01", periods=120, freq="h", tz="Asia/Shanghai")
    operation = pd.DataFrame({"event_time": op_times, "available_at": op_times})
    for i, c in enumerate(feature_config["operation"]["value_columns"]):
        operation[c] = np.arange(len(operation)) + i
    burden = operation.iloc[::8][["event_time", "available_at"]].copy()
    for i, c in enumerate(feature_config["burden"]["value_columns"]):
        burden[c] = np.arange(len(burden)) + i
    op_path, burden_path, index_path, query_path = [tmp_path / f"{n}.csv" for n in ("operation", "burden", "index", "query")]
    operation.to_csv(op_path, index=False)
    burden.to_csv(burden_path, index=False)
    samples[["sample_id", "spout_no", "reference_time"]].to_csv(index_path, index=False)
    known_index, index_audit = load_known_index([index_path])
    train, query = samples.iloc[:24], samples.iloc[24:]
    query[["sample_id", "tap_no", "spout_no", "reference_time"]].to_csv(query_path, index=False)
    cutoff = times[24]
    ctx = SimpleNamespace(features=feature_config, baseline=baseline, semantic=load_yaml("configs/data_contract.yaml"),
                          history=history, code={"synthetic": True}, candidates={"E09": candidate}, selection=selection,
                          profiles=load_profiles(), index_audit=index_audit,
                          inputs=file_identities({"operation_hourly": op_path, "burden_change": burden_path}))

    def X(part):
        base = build_features(part, operation=operation, burden=burden, history=history, fit_cutoff=cutoff, config=feature_config).X
        base = add_process_change_features(base, selection["process_change"], baseline_value_columns=feature_config["operation"]["value_columns"])
        base = select_candidate_features(base, candidate, selection)
        return extend_features(base, part, profile=profile, burden=burden, burden_columns=feature_config["burden"]["value_columns"], index=known_index)

    train_x = X(train)
    model = FrozenBaselineAdapter(baseline["parameters"]).fit(train_x, train[["tap_iron", "tap_time_len"]])
    save_profile_model(model, tmp_path / "bundle", ctx, train, cutoff, "E09", profile, train_x)
    restored = ProfilePredictor(tmp_path / "bundle")
    direct = model.predict_raw(X(query)).clip(lower=0)
    loaded = restored.predict(query, operation, burden, known_index)
    np.testing.assert_array_equal(direct.to_numpy(), loaded[["pred_tap_iron", "pred_tap_time_len"]].to_numpy())
    chunks = pd.concat([restored.predict(q, operation, burden, known_index) for q in (query.iloc[:5], query.iloc[5:])])
    pd.testing.assert_frame_equal(loaded, chunks)
    program = (
        "import json,sys; from bf_tap.io import read_csv; "
        "from bf_tap.optimization.v3_remaining_features import ProfilePredictor; "
        "from bf_tap.optimization.indexed_features import load_known_index; "
        "m=ProfilePredictor(sys.argv[1]); q=read_csv(sys.argv[2],time_columns=['reference_time']); "
        "op=read_csv(sys.argv[3],time_columns=['event_time','available_at']); "
        "b=read_csv(sys.argv[4],time_columns=['event_time','available_at']); "
        "idx,_=load_known_index([sys.argv[5]]); "
        "print(json.dumps(m.predict(q,op,b,idx)[['pred_tap_iron','pred_tap_time_len']].values.tolist()))"
    )
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path / "bundle"), str(query_path), str(op_path), str(burden_path), str(index_path)],
                            text=True, capture_output=True, check=True)
    np.testing.assert_array_equal(direct.to_numpy(), json.loads(result.stdout))
    with pytest.raises(ContractError, match="precede"):
        restored.predict(train, operation, burden, known_index)
    metadata = json.loads((tmp_path / "bundle" / "bundle.json").read_text())
    metadata["training"]["history_cutoff"] = str(cutoff + pd.Timedelta(days=1))
    atomic_write_json(tmp_path / "bundle" / "bundle.json", metadata, overwrite=True)
    atomic_write_json(tmp_path / "bundle" / "profile_identity.json", {"bundle_json_sha256": file_sha256(tmp_path / "bundle" / "bundle.json")}, overwrite=True)
    with pytest.raises(ContractError, match="cutoff"):
        ProfilePredictor(tmp_path / "bundle")
