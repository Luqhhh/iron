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
from bf_tap.optimization.calibration import fit_time_calibration, inner_blocks
from bf_tap.optimization.config import load_experiment, load_feature_selection
from bf_tap.optimization.features import select_candidate_features
from bf_tap.optimization.indexed_features import INDEX_COLUMNS, known_index_features, load_known_index
from bf_tap.optimization.process_change import add_process_change_features
from bf_tap.optimization.search_models import SingleTargetModel, search_configurations
from bf_tap.optimization.v3_followup import model_identity
from bf_tap.optimization.v3_release import ReleasePredictor, predict_stage, save_release


@pytest.fixture
def synthetic_bundle(tmp_path):
    features = load_yaml("configs/features.yaml")
    selection = load_feature_selection("configs/optimization_v0_2/features.yaml")
    _, candidates = load_experiment("configs/optimization_v0_2/experiment.yaml")
    candidate = next(c for c in candidates if c.id == "E09_PROCESS_CHANGE_E02")
    times = pd.date_range("2024-07-02", periods=36, freq="2h", tz="Asia/Shanghai")
    samples = pd.DataFrame({"sample_id": [f"s{i}" for i in range(36)], "tap_no": np.arange(36),
                           "spout_no": np.arange(36) % 2 + 1, "reference_time": times,
                           "tap_iron": 450. + np.arange(36), "tap_time_len": 100. + np.arange(36) % 6})
    samples["label_available_at"] = samples.reference_time + pd.Timedelta(hours=1)
    history = samples.assign(available_at=samples.label_available_at, tap_end_time=samples.label_available_at)
    t = pd.date_range("2024-07-01", periods=120, freq="h", tz="Asia/Shanghai")
    operation = pd.DataFrame({"event_time": t, "available_at": t})
    for n, c in enumerate(features["operation"]["value_columns"]):
        operation[c] = np.arange(len(t)) + n
    burden = operation.iloc[::8][["event_time", "available_at"]].copy()
    for n, c in enumerate(features["burden"]["value_columns"]):
        burden[c] = np.arange(len(burden)) + n
    for name, frame in (("operation", operation), ("burden", burden), ("index", samples[INDEX_COLUMNS]), ("query", samples.iloc[24:][INDEX_COLUMNS])):
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    index, _ = load_known_index([tmp_path / "index.csv"])
    cutoff = times[24]
    ctx = SimpleNamespace(labels=samples, history=history, features=features, selection=selection,
                          baseline=load_yaml("configs/baseline.yaml"), semantic=load_yaml("configs/data_contract.yaml"),
                          candidates={"E09": candidate}, profiles=load_yaml("configs/optimization_v0_3/features_remaining.yaml"),
                          inputs=file_identities({"operation_hourly": tmp_path / "operation.csv", "burden_change": tmp_path / "burden.csv"}),
                          code={"synthetic": True})
    train, query = samples.iloc[:24], samples.iloc[24:]

    def X(part):
        frame = build_features(part, operation=operation, burden=burden, history=history, fit_cutoff=cutoff, config=features).X
        frame = add_process_change_features(frame, selection["process_change"], baseline_value_columns=features["operation"]["value_columns"])
        frame = select_candidate_features(frame, candidate, selection)
        return pd.concat([frame, known_index_features(part[INDEX_COLUMNS], index)], axis=1)

    configs = {c["id"]: c for c in search_configurations("configs/optimization_v0_3/models")}
    expected = query[["sample_id"]].copy()
    models = {}
    for target, cid in (("tap_iron", "CB08"), ("tap_time_len", "CB02")):
        model = SingleTargetModel(configs[cid]["model_type"], configs[cid]["parameters"], target)
        model.fit(X(train), train[target], iterations=3)
        path = tmp_path / "models" / target
        model.save(path, identity=model_identity(ctx, train, cutoff, inner_blocks(samples, cutoff), "F-C", "synthetic-selection"))
        models[target] = path
        expected[f"pred_{target}"] = model.predict(X(query))
    empty = pd.DataFrame(columns=["sample_id", "reference_time", "label_available_at", "tap_time_len", "pred_tap_time_len", "prediction_fit_cutoff", "history_cutoff"])
    provenance = fit_time_calibration(empty, outer_cutoff=cutoff, candidate="CB-FC-CVcal", source_run="synthetic", origin_id="synthetic",
                                      pipeline_identity="synthetic", outer_sample_ids=list(query.sample_id))
    provenance.update(selected_iterations=3, selected_iterations_source="synthetic-selection")
    save_release(ctx, models, provenance, cutoff, tmp_path / "bundle", {}, {})
    return tmp_path, query, operation, burden, index, expected


def test_complete_release_cold_process_batch_and_no_future_history(synthetic_bundle):
    path, query, op, burden, index, expected = synthetic_bundle
    model = ReleasePredictor(path / "bundle")
    actual = model.predict(query, op, burden, index)
    np.testing.assert_array_equal(actual.iloc[:, 1:].to_numpy(), expected.iloc[:, 1:].to_numpy())
    pieces = pd.concat([model.predict(q, op, burden, index) for q in (query.iloc[::2], query.iloc[1::2])]).sort_index()
    pd.testing.assert_frame_equal(actual, pieces)
    program = (
        "import sys,json; from bf_tap.optimization.v3_release import ReleasePredictor; "
        "from bf_tap.optimization.indexed_features import load_known_index; from bf_tap.io import read_csv; "
        "from pathlib import Path; p=Path(sys.argv[1]); m=ReleasePredictor(p/'bundle'); "
        "q=read_csv(p/'query.csv',time_columns=['reference_time']); "
        "op=read_csv(p/'operation.csv',time_columns=['event_time','available_at']); "
        "b=read_csv(p/'burden.csv',time_columns=['event_time','available_at']); idx,_=load_known_index([p/'index.csv']); "
        "print(json.dumps(m.predict(q,op,b,idx).iloc[:,1:].values.tolist()))"
    )
    cold = subprocess.run([sys.executable, "-c", program, str(path)], capture_output=True, text=True, check=True)
    np.testing.assert_array_equal(expected.iloc[:, 1:].to_numpy(), json.loads(cold.stdout))
    assert (model.history.available_at <= model.cutoff).all()
    query = query.copy()
    query.reference_time = model.cutoff - pd.Timedelta(days=1)
    with pytest.raises(ContractError, match="precedes"):
        model.predict(query, op, burden, index)


@pytest.mark.parametrize("field,value", [("history_cutoff", "2024-08-01T00:00:00+08:00"), ("candidate", "E16")])
def test_release_rejects_component_and_cutoff_mismatch(synthetic_bundle, field, value):
    path, *_ = synthetic_bundle
    bundle = path / "bundle"
    m = json.loads((bundle / "bundle.json").read_text())
    m[field] = value
    atomic_write_json(bundle / "bundle.json", m, overwrite=True)
    atomic_write_json(bundle / "bundle_identity.json", {"bundle_json_sha256": file_sha256(bundle / "bundle.json")}, overwrite=True)
    with pytest.raises(ContractError, match="mismatch"):
        ReleasePredictor(bundle)


def test_release_rejects_unregistered_stage_and_damaged_component(synthetic_bundle):
    path, *_ = synthetic_bundle
    config = path / "data.yaml"
    config.write_text(json.dumps({"schema_version": 1, "paths": {}}))
    with pytest.raises(ContractError, match="stage"):
        predict_stage(path / "bundle", str(config), "test_a", [], path / "out")
    (path / "bundle" / "tap_iron" / "model.bin").write_bytes(b"damaged synthetic model")
    with pytest.raises(ContractError, match="identity"):
        ReleasePredictor(path / "bundle")


def test_release_rejects_changed_calibration_origin(synthetic_bundle):
    path, *_ = synthetic_bundle
    root = path / "bundle"
    p = json.loads((root / "calibration_provenance.json").read_text())
    p["outer_cutoff"] = "2024-08-01T00:00:00+08:00"
    atomic_write_json(root / "calibration_provenance.json", p, overwrite=True)
    m = json.loads((root / "bundle.json").read_text())
    m["files"]["calibration_provenance.json"] = file_sha256(root / "calibration_provenance.json")
    atomic_write_json(root / "bundle.json", m, overwrite=True)
    atomic_write_json(root / "bundle_identity.json", {"bundle_json_sha256": file_sha256(root / "bundle.json")}, overwrite=True)
    with pytest.raises(ContractError, match="calibration provenance"):
        ReleasePredictor(root)


def test_release_rejects_stage_input_identity_change(synthetic_bundle):
    path, *_ = synthetic_bundle
    root = path / "bundle"
    paths = {"train_samples": str(path / "index.csv"), "test_a_samples": str(path / "query.csv"),
             "operation_hourly": str(path / "operation.csv"), "burden_change": str(path / "burden.csv")}
    m = json.loads((root / "bundle.json").read_text())
    m["stages"] = {"test_a": {"index_roles": ["train_samples", "test_a_samples"], "inputs": file_identities(paths)}}
    atomic_write_json(root / "bundle.json", m, overwrite=True)
    atomic_write_json(root / "bundle_identity.json", {"bundle_json_sha256": file_sha256(root / "bundle.json")}, overwrite=True)
    config = path / "data.yaml"
    config.write_text(json.dumps({"schema_version": 1, "paths": paths}))
    (path / "operation.csv").write_text("changed source")
    with pytest.raises(ContractError, match="source identity"):
        predict_stage(root, str(config), "test_a", [paths["train_samples"], paths["test_a_samples"]], path / "out")
