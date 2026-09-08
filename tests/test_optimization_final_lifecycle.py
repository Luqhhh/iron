"""Real frozen estimators and synthetic-only file-level protected lifecycle."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from bf_tap.artifacts import atomic_write_json, file_sha256
from bf_tap.exceptions import ContractError, ProtectedLabelError
from bf_tap.optimization import final_lifecycle as fl


def synthetic_files(tmp_path):
    refs = pd.date_range("2024-03-01", "2024-11-29", freq="3D", tz="Asia/Shanghai")
    rows = pd.DataFrame({"sample_id": [f"s{i:04d}" for i in range(len(refs))],
                         "tap_no": np.arange(len(refs)), "spout_no": 1 + np.arange(len(refs)) % 2,
                         "reference_time": refs, "tap_iron": 430. + np.arange(len(refs)) % 29,
                         "tap_time_len": 90. + np.arange(len(refs)) % 17})
    history = rows.copy()
    history["tap_end_time"] = refs + pd.Timedelta(hours=2)
    cutoff = pd.Timestamp("2024-12-01 01:44", tz="Asia/Shanghai")
    history.loc[len(history)-2, "tap_end_time"] = cutoff  # inclusive availability
    history.loc[len(history)-1, "tap_end_time"] = cutoff + pd.Timedelta(minutes=1)
    paths = {}
    for name, value in (("train_samples", rows), ("tap_history_train", history)):
        paths[name] = str(tmp_path / (name + ".csv"))
        value.to_csv(paths[name], index=False)
    for stage, month in (("test_a", 12), ("test_b", 1), ("test_c", 2)):
        samples = rows.iloc[:3][fl.META].copy()
        samples["sample_id"] = [stage + str(i) for i in range(3)]
        samples["reference_time"] = pd.date_range(f"{2024 if month == 12 else 2025}-{month:02d}-01 01:44", periods=3, freq="D", tz="Asia/Shanghai")
        paths[stage + "_samples"] = str(tmp_path / (stage + ".csv"))
        samples.to_csv(paths[stage + "_samples"], index=False)
    a = fl.algorithm()
    for source, timecol in (("operation", "clock"), ("burden", "cal_time")):
        times = pd.date_range("2024-02-28", "2025-03-01", freq="12h", tz="Asia/Shanghai")
        frame = pd.DataFrame({timecol: times})
        for i, col in enumerate(a["features"][source]["value_columns"]):
            frame[col] = i + np.sin(np.arange(len(times)) / 13.)
        key = "operation_hourly" if source == "operation" else "burden_change"
        paths[key] = str(tmp_path / (key + ".csv"))
        frame.to_csv(paths[key], index=False)
    paths["data_dictionary"] = str(tmp_path / "dictionary.xlsx")
    (tmp_path / "dictionary.xlsx").write_bytes(b"synthetic identity only")
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({"schema_version": 1, "paths": paths}))
    return data, paths, rows


def authorization(manifest, lifecycle):
    return {"schema_version": "protected-authorization-v1", "lifecycle": lifecycle,
            "policy_digest": fl.read_json(manifest)["policy_digest"],
            "frozen_manifest_sha256": file_sha256(manifest), "approval_id": "synthetic-" + lifecycle}


def test_final_reader_rejects_before_target_read_and_config_mismatch(tmp_path, monkeypatch):
    data, _, _ = synthetic_files(tmp_path)
    manifest = fl.freeze(data, tmp_path / "freeze")
    ledger = tmp_path / "ledger.jsonl"
    def forbidden(*args, **kwargs):
        pytest.fail("target parser called before authorization")
    monkeypatch.setattr(fl, "_read_eligible_rows", forbidden)
    with pytest.raises(ProtectedLabelError):
        fl.protected_inputs(manifest, None, ledger, "final_training")
    auth = authorization(manifest, "final_training")
    original = fl.algorithm()
    monkeypatch.setattr(fl, "algorithm", lambda: {**original, "weights": {"E04": 1.}})
    with pytest.raises(ProtectedLabelError, match="configuration mismatch"):
        fl.protected_inputs(manifest, auth, ledger, "final_training")
    assert not ledger.exists()
    with pytest.raises(FileExistsError):
        fl.freeze(data, tmp_path / "freeze")


@pytest.mark.model
def test_synthetic_complete_protected_lifecycle_and_cold_prediction(tmp_path, monkeypatch):
    data, paths, rows = synthetic_files(tmp_path)
    manifest = fl.freeze(data, tmp_path / "freeze")
    ledger = tmp_path / "ledger.jsonl"
    prepared, report, final, pred = [tmp_path / x for x in ("prepared", "report", "final", "pred")]
    for p in (prepared, report, final, pred):
        p.mkdir()
    fl.prepare(manifest, prepared)
    assert not ledger.exists()
    fl.report(manifest, prepared, authorization(manifest, "holdout_scoring"), ledger, report)
    first = ledger.read_bytes()
    info = fl.read_json(report / "holdout_report.json")
    assert info["independent_protected_months"] == 1
    assert set(info["metrics"]) == {f"HOLDOUT_H{i}" for i in range(1, 5)}
    assert info["rows"] == sum(rows.reference_time >= pd.Timestamp("2024-11-01", tz="Asia/Shanghai"))-1
    with pytest.raises(ProtectedLabelError, match="already consumed"):
        fl.protected_inputs(manifest, {**authorization(manifest, "holdout_scoring"), "approval_id": "different"}, ledger, "holdout_scoring")
    assert ledger.read_bytes() == first
    fl.final_train(manifest, report / "holdout_report.json", authorization(manifest, "final_training"), ledger, final)
    assert ledger.read_bytes().startswith(first)
    training = fl.read_json(final / "training_manifest.json")["training"]
    assert training["eligible_rows"] == len(rows)-1
    assert training["label_available_max"] == training["fit_cutoff"]
    # Independent prediction must succeed with official-label paths absent and
    # the estimator's fit method forbidden in the current process.
    predict_config = tmp_path / "predict.yaml"
    predict_config.write_text(yaml.safe_dump({"schema_version": 1, "paths": {
        k: v for k, v in paths.items() if k in {"test_a_samples", "operation_hourly", "burden_change"}}}))
    for name in ("train_samples", "tap_history_train"):
        from pathlib import Path
        Path(paths[name]).unlink()
    def forbidden_fit(*args, **kwargs):
        pytest.fail("predict invoked fit")
    monkeypatch.setattr(fl.DualTargetBaseline, "fit", forbidden_fit)
    fl.predict(final / "bundle", predict_config, "test_a", pred)
    cold = tmp_path / "cold"
    subprocess.run([sys.executable, "-m", "bf_tap.optimization.final_lifecycle", "predict",
                    "--bundle", str(final / "bundle"), "--data-config", str(predict_config),
                    "--output", str(cold)], check=True, capture_output=True, text=True)
    assert (pred / "predictions_raw.csv").read_bytes() == (cold / "predictions_raw.csv").read_bytes()
    assert (pred / "result.csv").read_bytes() == (cold / "result.csv").read_bytes()
    # Keep the original failure evidence; a second CLI attempt cannot replace it.
    failed = tmp_path / "failed"
    command = [sys.executable, "-m", "bf_tap.optimization.final_lifecycle", "predict",
               "--bundle", str(tmp_path / "missing"), "--output", str(failed)]
    assert subprocess.run(command, capture_output=True).returncode != 0
    before = (failed / "final_status.json").read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert (failed / "final_status.json").read_bytes() == before


def test_eligibility_excludes_reference_equal_and_late_availability():
    c = pd.Timestamp("2024-12-01 01:44", tz="Asia/Shanghai")
    rows = pd.DataFrame({"sample_id": ["ok", "same_time", "late"],
                         "reference_time": [c-pd.Timedelta(days=1), c, c-pd.Timedelta(days=1)],
                         "label_available_at": [c, c, c+pd.Timedelta(seconds=1)]})
    assert fl.eligible(rows, c).sample_id.tolist() == ["ok"]
