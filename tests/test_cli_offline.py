import json
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import yaml

from bf_tap.offline import run_prediction

OPERATION_COLUMNS = [
    "air_volume",
    "cold_air_press",
    "hot_air_press",
    "oxygen",
    "hot_air_temp",
    "coal_rate",
    "humidity",
    "gas_rate",
    "furnace_top_press",
    "upper_press_diff",
    "lower_press_diff",
    "total_press_diff",
    "air_press_ratio",
    "furnace_top_temp_avg",
    "air_speed",
    "furnace_throat_temp",
]
BURDEN_COLUMNS = ["pig", "all_quality", "consumption", "fuel_rate", "coke_rate"]


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _run(root: Path, *args: str) -> subprocess.CompletedProcess:
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    return subprocess.run(
        [sys.executable, "-m", "bf_tap", *args],
        cwd=root,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )


@pytest.mark.model
def test_file_level_train_predict_twice_and_pack_without_training_labels(tmp_path):
    root = Path(__file__).parents[1]
    data = tmp_path / "data"
    data.mkdir()
    reference = pd.date_range("2024-01-01 01:00", periods=36, freq="2h")
    train = pd.DataFrame(
        {
            "sample_id": [f"S{i:03d}" for i in range(36)],
            "tap_no": np.arange(100, 136),
            "spout_no": 1 + np.arange(36) % 2,
            "reference_time": reference,
            "tap_iron": 450 + np.arange(36) % 9,
            "tap_time_len": 100 + np.arange(36) % 6,
        }
    )
    train_path = data / "train_samples.csv"
    train.to_csv(train_path, index=False)
    history = train.copy()
    history["tap_end_time"] = history["reference_time"] + pd.Timedelta(hours=1)
    history.to_csv(data / "tap_history_train.csv", index=False)
    clocks = pd.date_range("2023-12-31", periods=120, freq="h")
    operation = pd.DataFrame({"clock": clocks})
    for index, column in enumerate(OPERATION_COLUMNS):
        operation[column] = np.linspace(index + 1, index + 2, len(clocks))
    operation.to_csv(data / "operation_hourly.csv", index=False)
    changes = pd.date_range("2023-12-31", periods=8, freq="D")
    burden = pd.DataFrame({"cal_time": changes})
    for index, column in enumerate(BURDEN_COLUMNS):
        burden[column] = np.linspace(index + 1, index + 2, len(changes))
    burden.to_csv(data / "burden_change.csv", index=False)
    (data / "data_dictionary.xlsx").write_bytes(b"synthetic dictionary identity only")
    test = train.iloc[30:35][["sample_id", "tap_no", "spout_no", "reference_time"]].copy()
    test["sample_id"] = [f"T{i:03d}" for i in range(len(test))]
    test.to_csv(data / "test_a_samples.csv", index=False)

    train_paths = tmp_path / "train_paths.yaml"
    _write_yaml(
        train_paths,
        {
            "schema_version": 1,
            "paths": {
                "train_samples": str(train_path),
                "operation_hourly": str(data / "operation_hourly.csv"),
                "burden_change": str(data / "burden_change.csv"),
                "tap_history_train": str(data / "tap_history_train.csv"),
                "data_dictionary": str(data / "data_dictionary.xlsx"),
            },
        },
    )
    predict_paths = tmp_path / "predict_paths.yaml"
    _write_yaml(
        predict_paths,
        {
            "schema_version": 1,
            "paths": {
                "test_a_samples": str(data / "test_a_samples.csv"),
                "operation_hourly": str(data / "operation_hourly.csv"),
                "burden_change": str(data / "burden_change.csv"),
            },
        },
    )
    protection = tmp_path / "protection.yaml"
    _write_yaml(
        protection,
        {
            "schema_version": 1,
            "contract_id": "synthetic-holdout-v1",
            "timezone": "Asia/Shanghai",
            "development_label_end_exclusive": "2024-02-01T00:00:00+08:00",
            "protected_interval": {
                "start": "2024-02-01T00:00:00+08:00",
                "end": "2024-03-01T00:00:00+08:00",
            },
            "allowed_protected_lifecycles": ["holdout_scoring", "final_training"],
        },
    )
    train_run = tmp_path / "train-run"
    _run(
        root,
        "train",
        "--data-config",
        str(train_paths),
        "--data-contract",
        str(root / "configs/data_contract.yaml"),
        "--protection-policy",
        str(protection),
        "--config",
        str(root / "configs/baseline.yaml"),
        "--feature-config",
        str(root / "configs/features.yaml"),
        "--mode",
        "development",
        "--train-start",
        "2024-01-01T00:00:00+08:00",
        "--fit-cutoff",
        "2024-01-03T12:00:00+08:00",
        "--output",
        str(train_run),
    )
    train_path.unlink()
    first = tmp_path / "predict-1"
    second = tmp_path / "predict-2"
    for output in (first, second):
        _run(
            root,
            "predict",
            "--bundle",
            str(train_run / "bundle"),
            "--data-config",
            str(predict_paths),
            "--stage",
            "test_a",
            "--output",
            str(output),
        )
    assert (first / "predictions_raw.csv").read_bytes() == (
        second / "predictions_raw.csv"
    ).read_bytes()
    assert (first / "result.csv").read_bytes() == (second / "result.csv").read_bytes()
    manifest = json.loads((first / "prediction_manifest.json").read_text())
    assert manifest["status"] == "PASS"
    assert manifest["rows"] == len(test)
    packed = tmp_path / "packed"
    _run(
        root,
        "pack",
        "--stage",
        "test_a",
        "--team-name",
        "synthetic",
        "--result",
        str(first / "result.csv"),
        "--output-dir",
        str(packed),
        "--data-config",
        str(predict_paths),
    )
    archive = packed / "synthetic_bf_tap_predict_prelim.zip"
    with ZipFile(archive) as handle:
        assert handle.namelist() == ["result.csv"]

    bundle_metadata = json.loads((train_run / "bundle" / "bundle.json").read_text())
    assert bundle_metadata["bundle_schema_version"] == 3
    assert set(bundle_metadata["inference_source_contract"]["sources"]) == {
        "burden_change",
        "operation_hourly",
    }
    operation.to_csv(data / "operation_hourly.csv", index=False, float_format="%.12f")
    mismatched = subprocess.run(
        [
            sys.executable,
            "-m",
            "bf_tap",
            "predict",
            "--bundle",
            str(train_run / "bundle"),
            "--data-config",
            str(predict_paths),
            "--stage",
            "test_a",
            "--output",
            str(tmp_path / "predict-mismatched-source"),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        check=False,
        text=True,
        capture_output=True,
    )
    assert mismatched.returncode == 2
    assert "inference source identity mismatch" in mismatched.stderr


def test_prediction_preflight_failure_records_final_status(tmp_path):
    paths = tmp_path / "missing-paths.yaml"
    _write_yaml(
        paths,
        {
            "schema_version": 1,
            "paths": {
                "test_a_samples": str(tmp_path / "missing-samples.csv"),
                "operation_hourly": str(tmp_path / "missing-operation.csv"),
                "burden_change": str(tmp_path / "missing-burden.csv"),
            },
        },
    )
    output = tmp_path / "failed-prediction"
    with pytest.raises(FileNotFoundError):
        run_prediction(
            bundle_path=tmp_path / "missing-bundle",
            data_config_path=paths,
            stage="test_a",
            output=output,
        )
    status = json.loads((output / "final_status.json").read_text())
    assert status["status"] == "FAILED"
    assert status["error_type"] == "FileNotFoundError"
