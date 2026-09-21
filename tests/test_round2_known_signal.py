"""Exactly two synthetic regression fits; never modify competition labels."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("sklearn")
from bf_tap_r2.audit import digest, write_json
from bf_tap_r2.data import FEATURES
from bf_tap_r2.models import SnapshotRegressor


@pytest.mark.parametrize("target,feature,offset,slope", [
    ("tap_iron", "air_volume", 500., 50.), ("tap_time_len", "hot_air_temp", 125., 15.)])
def test_known_signal_learned(target, feature, offset, slope):
    rng = np.random.default_rng(20260922)
    frame = pd.DataFrame(rng.normal(size=(512, 21)), columns=FEATURES)
    frame["spout_no"] = np.arange(len(frame)) % 2 + 1
    frame["sample_id"] = [f"SYNTHETIC_{i}" for i in range(len(frame))]
    training, validation = frame.iloc[:384], frame.iloc[384:]
    mean, scale = training[feature].mean(), training[feature].std(ddof=0)
    y = offset + slope * (frame[feature] - mean) / scale
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/round2_v0_1/models.yaml").read_text())
    directory = os.environ.get("R2_KNOWN_SIGNAL_EVIDENCE")
    if directory:
        output = Path(directory)
        if not output.resolve().is_relative_to(root / "local/runs/round2-v0.2"):
            raise ValueError("Synthetic evidence must stay private")
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / f"{target}-started.json", {"regressor_fits": 1, "test_source_sha256": digest(Path(__file__))})
    model = SnapshotRegressor("L1", config).fit(training, y.iloc[:384])
    mae = float(np.mean(np.abs(y.iloc[384:] - model.predict(validation))))
    base = float(np.mean(np.abs(y.iloc[384:] - np.median(y.iloc[:384]))))
    report = {"target": target, "mae": mae, "baseline_mae": base, "ratio": mae / base,
              "threshold": .1, "pass": bool(mae <= .1 * base), "regressor_fits": 1,
              "data": "synthetic_only", "training_rows": 384, "validation_rows": 128,
              "scaling_scope": "synthetic_training_partition_only", "test_source_sha256": digest(Path(__file__))}
    if directory:
        write_json(output / f"{target}.json", report)
    assert report["pass"], report
