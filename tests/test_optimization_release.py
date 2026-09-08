import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bf_tap.exceptions import ContractError
from bf_tap.optimization.release import run_derived_prediction


ROOT = Path(__file__).parents[1]


def _write_component(path: Path, candidate_id: str, iron: list[float], time: list[float]) -> None:
    path.mkdir()
    result = pd.DataFrame(
        {
            "sample_id": pd.Series(["001", "002"], dtype="string"),
            "pred_tap_iron": iron,
            "pred_tap_time_len": time,
        }
    )
    result.to_csv(path / "result.csv", index=False)
    result_sha256 = hashlib.sha256((path / "result.csv").read_bytes()).hexdigest()
    (path / "prediction_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "optimization_id": "optimization-v0.2",
                "candidate_id": candidate_id,
                "stage": "test_a",
                "result_sha256": result_sha256,
            }
        ),
        encoding="utf-8",
    )


def test_registered_convex_blend_prediction_is_auditable(tmp_path):
    samples = tmp_path / "samples.csv"
    pd.DataFrame({"sample_id": ["001", "002"]}).to_csv(samples, index=False)
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "paths": {"test_a_samples": str(samples)}}
        ),
        encoding="utf-8",
    )
    e09 = tmp_path / "e09"
    e04 = tmp_path / "e04"
    _write_component(e09, "E09_PROCESS_CHANGE_E02", [10.0, 20.0], [30.0, 40.0])
    _write_component(e04, "E04", [20.0, 40.0], [50.0, 70.0])

    output = run_derived_prediction(
        experiment_config_path=ROOT / "configs/optimization_v0_2/experiment.yaml",
        baseline_config_path=ROOT / "configs/baseline.yaml",
        candidate_id="E12_BLEND_E09_E04_80_20",
        component_prediction_paths={"E09_PROCESS_CHANGE_E02": e09, "E04": e04},
        data_config_path=data_config,
        stage="test_a",
        output=tmp_path / "derived",
    )

    result = pd.read_csv(output / "result.csv", dtype={"sample_id": "string"})
    assert result["sample_id"].tolist() == ["001", "002"]
    assert result["pred_tap_iron"].tolist() == pytest.approx([12.0, 24.0])
    assert result["pred_tap_time_len"].tolist() == pytest.approx([34.0, 46.0])
    manifest = json.loads((output / "prediction_manifest.json").read_text())
    assert manifest["status"] == "PASS"
    assert set(manifest["components"]) == {"E09_PROCESS_CHANGE_E02", "E04"}
    assert json.loads((output / "final_status.json").read_text())["status"] == "PASS"


def test_registered_targetwise_blend_accepts_prior_derived_components(tmp_path):
    samples = tmp_path / "samples.csv"
    pd.DataFrame({"sample_id": ["001", "002"]}).to_csv(samples, index=False)
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "paths": {"test_a_samples": str(samples)}}
        ),
        encoding="utf-8",
    )
    e12 = tmp_path / "e12"
    e14 = tmp_path / "e14"
    _write_component(e12, "E12_BLEND_E09_E04_80_20", [10.0, 20.0], [30.0, 40.0])
    _write_component(e14, "E14_TIMECAL_E09", [11.0, 21.0], [27.0, 37.0])

    output = run_derived_prediction(
        experiment_config_path=ROOT / "configs/optimization_v0_2/experiment.yaml",
        baseline_config_path=ROOT / "configs/baseline.yaml",
        candidate_id="E15_TARGETWISE_E12_E14",
        component_prediction_paths={
            "E12_BLEND_E09_E04_80_20": e12,
            "E14_TIMECAL_E09": e14,
        },
        data_config_path=data_config,
        stage="test_a",
        output=tmp_path / "targetwise",
    )

    result = pd.read_csv(output / "result.csv", dtype={"sample_id": "string"})
    assert result["pred_tap_iron"].tolist() == pytest.approx([10.0, 20.0])
    assert result["pred_tap_time_len"].tolist() == pytest.approx([27.0, 37.0])


def test_derived_prediction_rejects_incomplete_component_mapping(tmp_path):
    samples = tmp_path / "samples.csv"
    pd.DataFrame({"sample_id": ["001"]}).to_csv(samples, index=False)
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "paths": {"test_a_samples": str(samples)}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="do not match"):
        run_derived_prediction(
            experiment_config_path=ROOT / "configs/optimization_v0_2/experiment.yaml",
            baseline_config_path=ROOT / "configs/baseline.yaml",
            candidate_id="E12_BLEND_E09_E04_80_20",
            component_prediction_paths={},
            data_config_path=data_config,
            stage="test_a",
            output=tmp_path / "failed",
        )
    assert json.loads((tmp_path / "failed/final_status.json").read_text())["status"] == "FAILED"
