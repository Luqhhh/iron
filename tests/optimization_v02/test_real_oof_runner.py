"""Full real-estimator pipeline with deliberately unparsable protected targets."""
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from bf_tap.config import load_yaml

ROOT = Path(__file__).parents[2]


def dataset(tmp_path):
    times = pd.to_datetime(
        ["2024-03-01 01:00", "2024-03-02 01:00", "2024-03-03 01:00",
         "2024-04-01 01:00", "2024-05-01 01:00", "2024-05-15 01:00",
         "2024-06-05 01:00", "2024-06-20 01:00", "2024-07-05 01:00",
         "2024-07-20 01:00", "2024-08-05 01:00", "2024-08-20 01:00",
         "2024-09-05 01:00", "2024-10-05 01:00", "2024-11-05 01:00"])
    n = len(times)
    data = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)],
                         "tap_no": np.arange(n), "spout_no": 1 + np.arange(n) % 2,
                         "reference_time": times, "tap_iron": 100. + np.arange(n) * 3,
                         "tap_time_len": 20. + np.arange(n)})
    for t in ("tap_iron", "tap_time_len"):
        data[t] = data[t].astype(object)
        data.loc[n-1, t] = "PROTECTED_SENTINEL"
    data.to_csv(tmp_path / "train_samples.csv", index=False)
    history = data.copy()
    history["tap_end_time"] = times + pd.Timedelta(hours=1)
    history.to_csv(tmp_path / "tap_history_train.csv", index=False)
    cfg = load_yaml(ROOT / "configs/features.yaml")
    operation = pd.DataFrame({"clock": times - pd.Timedelta(hours=1)})
    for i, col in enumerate(cfg["operation"]["value_columns"]):
        operation[col] = i + 1. + np.arange(n)
    operation.to_csv(tmp_path / "operation_hourly.csv", index=False)
    burden = pd.DataFrame({"cal_time": times - pd.Timedelta(hours=1)})
    for i, col in enumerate(cfg["burden"]["value_columns"]):
        burden[col] = i + 2. + np.arange(n)
    burden.to_csv(tmp_path / "burden_change.csv", index=False)
    (tmp_path / "data_dictionary.xlsx").write_bytes(b"synthetic identity")
    paths = {name: str(tmp_path / (name + ext)) for name, ext in [
        ("train_samples", ".csv"), ("tap_history_train", ".csv"),
        ("operation_hourly", ".csv"), ("burden_change", ".csv"),
        ("data_dictionary", ".xlsx")]}
    config = tmp_path / "data.local.yaml"
    config.write_text(yaml.safe_dump({"schema_version": 1, "paths": paths}))
    return config


def test_real_oof_runner_excludes_november_and_preserves_failures(tmp_path, monkeypatch):
    path = "bf_tap.models.optimization_v02.workflow"
    assert find_spec(path) is not None, "real OOF workflow must exist"
    m = import_module(path)
    monkeypatch.chdir(ROOT)
    data_config = dataset(tmp_path)
    output = tmp_path / "run"
    m.run_local_oof(data_config, output, confirm=True)
    status = json.loads((output / "final_status.json").read_text())
    assert status["engineering_status"] == "PASS"
    assert status["protected_labels_read"] is False
    for name in ("M2_H30", "M2_H60", "M2_H120", "M3_RESIDUAL", "CatBoost", "B0", "B1"):
        pred = pd.read_csv(output / "oof" / f"{name}.csv")
        assert pred["sample_id"].tolist() == [f"s{i}" for i in range(6, 12)]
        assert len(pred) == 6
    # Offline replay must not need either training label file.
    offline_path = "bf_tap.models.optimization_v02.offline"
    assert find_spec(offline_path) is not None, "offline fold replay must exist"
    offline = import_module(offline_path)
    paths = yaml.safe_load(data_config.read_text())["paths"]
    (tmp_path / "train_samples.csv").unlink()
    (tmp_path / "tap_history_train.csv").unlink()
    query = pd.DataFrame({"sample_id": ["s6", "s7"], "tap_no": [6, 7], "spout_no": [1, 2],
                          "reference_time": pd.to_datetime(
                              ["2024-06-05 01:00", "2024-06-20 01:00"]).tz_localize("Asia/Shanghai")})
    replayed = offline.predict_fold(
        output / "folds/ROLL_2024_06", query, paths,
        output / "frozen_selection.json")
    for name, values in replayed.items():
        expected_path = output / "oof" / f"{name}.csv"
        expected = pd.read_csv(expected_path).iloc[:2]
        np.testing.assert_allclose(values[["pred_tap_iron", "pred_tap_time_len"]],
                                   expected[["pred_tap_iron", "pred_tap_time_len"]],
                                   atol=1e-9, rtol=0)
    frozen = json.loads((output / "frozen_selection.json").read_text())
    confirmation = json.loads((output / "confirmation.json").read_text())
    assert confirmation["selection_sha256"] == frozen["selection_sha256"]
    assert set(confirmation["folds"]) == {"DEV_LONG", "DEV_SHORT"}
    # All estimator bundles must survive load and replay with their stored semantics.
    audit = json.loads((output / "folds/ROLL_2024_06/audit.json").read_text())
    assert all(row["roundtrip_max_abs_diff"] <= 1e-10 for row in audit["models"].values())
    assert audit["models"]["M3_RESIDUAL"]["warmup_rows"] >= 1
    with pytest.raises(FileExistsError):
        m.run_local_oof(data_config, output, confirm=True)
    assert json.loads((output / "final_status.json").read_text()) == status
