import json
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[2]
CONFIG_ROOT = ROOT / "configs" / "optimization" / "v0.2"


def _runner():
    module = find_spec("bf_tap.models.optimization_v02.m1_blend.runner")
    assert module is not None, "M1 selection runner must exist"
    return import_module("bf_tap.models.optimization_v02.m1_blend.runner")


def _write_prediction_pair(root, fold_id, sample_id):
    fold = root / fold_id
    fold.mkdir(parents=True)
    pd.DataFrame(
        {
            "sample_id": [sample_id],
            "pred_tap_iron": [100.0],
            "pred_tap_time_len": [60.0],
        }
    ).to_csv(fold / "predictions.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": [sample_id],
            "pred_tap_iron": [120.0],
            "pred_tap_time_len": [50.0],
        }
    ).to_csv(fold / "predictions_B1.csv", index=False)


def test_m1_runner_selects_weights_without_parsing_protected_targets(tmp_path, monkeypatch):
    runner = _runner()
    labels = tmp_path / "train_samples.csv"
    labels.write_text(
        "sample_id,tap_no,spout_no,reference_time,tap_iron,tap_time_len\n"
        "s1,1,1,2024-06-15 00:00:00,100,50\n"
        "s2,2,1,2024-07-15 00:00:00,100,50\n"
        "s3,3,1,2024-08-15 00:00:00,100,50\n"
        "protected,4,1,2024-11-15 00:00:00,PROTECTED_SENTINEL,PROTECTED_SENTINEL\n",
        encoding="utf-8",
    )
    baseline_oof = tmp_path / "baseline_oof"
    for fold_id, sample_id in (
        ("ROLL_2024_06", "s1"),
        ("ROLL_2024_07", "s2"),
        ("ROLL_2024_08", "s3"),
    ):
        _write_prediction_pair(baseline_oof, fold_id, sample_id)

    monkeypatch.chdir(ROOT)
    destination = runner.run_m1_selection(
        labels_path=labels,
        baseline_oof_dir=baseline_oof,
        common_config_path=CONFIG_ROOT / "common.yaml",
        candidate_config_path=CONFIG_ROOT / "m1_blend.yaml",
        protection_policy_path=ROOT / "configs" / "protection.yaml",
        output_root=tmp_path / "runs",
        run_id="m1-test",
    )

    selection = json.loads((destination / "selection.json").read_text())
    final_status = json.loads((destination / "final_status.json").read_text())
    predictions = pd.read_csv(destination / "oof_predictions.csv")

    assert selection["weights"] == {"tap_iron": 1.0, "tap_time_len": 0.0}
    assert selection["metrics"]["pooled"]["loss"] == 0.0
    assert predictions["sample_id"].tolist() == ["s1", "s2", "s3"]
    assert "protected" not in set(predictions["sample_id"])
    assert final_status == {
        "candidate_id": "M1_BLEND",
        "safe_label_end_exclusive": "2024-09-01 00:00:00+08:00",
        "status": "PASS",
    }
    state = json.loads((destination / "run_state.json").read_text())
    assert set(state["data_manifest"]) == {
        "ROLL_2024_06_B1",
        "ROLL_2024_06_catboost",
        "ROLL_2024_07_B1",
        "ROLL_2024_07_catboost",
        "ROLL_2024_08_B1",
        "ROLL_2024_08_catboost",
        "train_samples",
    }
