from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.normalized_models import JointSnapshotRegressor, NormalizedSnapshotRegressor
from bf_tap_r2.v2_robust_joint import check_packages

ROOT = Path(__file__).resolve().parents[1]


def synthetic():
    rng = np.random.default_rng(223)
    frame = pd.DataFrame({f: rng.normal(size=60) for f in FEATURES})
    frame["spout_no"] = np.tile([1, 2], 30)
    frame["sample_id"] = [f"test-{i}" for i in range(60)]
    frame["tap_iron"] = 500 + 30*frame.air_volume
    frame["tap_time_len"] = 120 + 12*frame.air_volume + 4*frame.oxygen
    return frame


@pytest.mark.parametrize("route", ["N1", "H1", "J1"])
def test_fold_only_scale_roundtrip_output_shape_and_no_label_input(route, tmp_path):
    spec = yaml.safe_load((ROOT / "configs/round2_v2_3/experiment.yaml").read_text())
    parameters = dict(spec["common"], iterations=8, depth=2, thread_count=1, loss_function=spec["losses"][route])
    frame = synthetic()
    training, valid = frame.iloc[:40], frame.iloc[40:]
    fields = list(TARGETS) if route == "J1" else [TARGETS[0]]
    model = JointSnapshotRegressor(parameters) if route == "J1" else NormalizedSnapshotRegressor(parameters, fields[0])
    model.fit(training, training[fields] if route == "J1" else training[fields[0]])
    np.testing.assert_allclose(model.target_scales_, training[fields].mean())
    assert model.actual_parameters_["iterations"] == 8
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    x = valid.drop(columns=list(TARGETS))
    predicted = restored.predict(x)
    assert predicted.shape == ((20, 2) if route == "J1" else (20,))
    assert restored.predict(x.iloc[:1]).shape == ((1, 2) if route == "J1" else (1,))
    np.testing.assert_allclose(predicted, model.predict(valid), rtol=1e-13)
    poisoned = valid.copy()
    poisoned[list(TARGETS)] = -1e10
    np.testing.assert_array_equal(predicted, restored.predict(poisoned))
    np.testing.assert_array_equal(predicted, restored.predict(x.iloc[::-1])[::-1])
    np.testing.assert_array_equal(predicted, np.concatenate([restored.predict(x.iloc[:7]), restored.predict(x.iloc[7:])]))


def test_joint_rejects_reversed_target_order_and_nonpositive_scale():
    frame = synthetic()
    model = JointSnapshotRegressor({"loss_function": "MultiRMSE"})
    with pytest.raises(ValueError, match="order"):
        model.fit(frame, frame[list(TARGETS)[::-1]])
    labels = frame[list(TARGETS)].copy()
    labels["tap_time_len"] = 0
    with pytest.raises(ValueError, match="positive"):
        model.fit(frame, labels)


def test_transform_is_mean_centered_and_restored_in_original_units(monkeypatch):
    frame = synthetic()
    observed = {}
    class FakeEstimator:
        def __init__(self, **kwargs):
            pass
        def fit(self, x, y):
            observed["features"] = list(x.columns)
            observed["labels"] = y
        def get_all_params(self):
            return {}
        def predict(self, x):
            return np.tile([.1, -.1], (len(x), 1))
    monkeypatch.setattr("bf_tap_r2.normalized_models.CatBoostRegressor", FakeEstimator)
    model = JointSnapshotRegressor({"loss_function": "MultiRMSE"}).fit(frame, frame[list(TARGETS)])
    np.testing.assert_allclose(observed["labels"].mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(observed["labels"], (frame[list(TARGETS)]-frame[list(TARGETS)].mean())/frame[list(TARGETS)].mean())
    assert observed["features"] == [*FEATURES, "spout_no"]
    np.testing.assert_allclose(model.predict(frame.iloc[:1]), [model.target_scales_*[1.1, .9]])


def test_reject_invalid_independent_label_shape():
    frame = synthetic()
    model = NormalizedSnapshotRegressor({"loss_function": "RMSE"}, "tap_iron")
    with pytest.raises(ValueError, match="one-dimensional"):
        model.fit(frame, frame[list(TARGETS)])


def test_frozen_scope():
    spec = yaml.safe_load((ROOT / "configs/round2_v2_3/experiment.yaml").read_text())
    assert spec["budget"] == {"cv_fits": 50, "full_fits": 0}
    assert spec["losses"] == {"N1": "RMSE", "H1": "Huber:delta=0.05", "J1": "MultiRMSE"}
    assert spec["platform"]["remaining_quota_user_reported"] == 0
    for target in TARGETS:
        assert spec["candidates"][target] == ["N1", "H1", "J1", "AH", "AJ"]


def test_report_replay_and_both_reference_layers(tmp_path, monkeypatch):
    import json
    from copy import deepcopy
    from bf_tap_r2.cv import target_metrics
    from bf_tap_r2.v2_robust_joint import finish

    spec = yaml.safe_load((ROOT / "configs/round2_v2_3/experiment.yaml").read_text())
    spec["bootstrap"]["repetitions"] = 20  # Synthetic report test only.
    policy = yaml.safe_load((ROOT / spec["policy"]).read_text())
    frame = synthetic()
    folds = np.arange(len(frame)) % 5
    old_metrics, old_predictions, predictions = {}, {}, {}
    for target in TARGETS:
        y = frame[target].to_numpy()
        old_predictions[target] = {}
        old_metrics[target] = {}
        for route, factor in [(spec["reference_by_target"][target], 1.05), (spec["v22_by_target"][target], 1.04)]:
            old_predictions[target][route] = {s: y*factor for s in spec["split_seeds"]}
            old_metrics[target][route] = {str(s): target_metrics(frame, target, y*factor, folds) for s in spec["split_seeds"]}
        predictions[target] = {r: {s: y*factor for s in spec["split_seeds"]} for r, factor in [("N1", 1.03), ("H1", 1.02), ("J1", 1.01)]}
        predictions[target].update(AH={}, AJ={})
    for folder in ("selection", "oof", "diagnostics"):
        (tmp_path / folder).mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"policy": policy}))
    monkeypatch.setattr("bf_tap_r2.v2_robust_joint.fold_vector", lambda *args: folds)
    monkeypatch.setattr("bf_tap_r2.v2_robust_joint.recover_oof", lambda *args, **kwargs: (frame, old_metrics, old_predictions))
    result = finish(ROOT, tmp_path, spec, frame, deepcopy(predictions))
    finish(ROOT, tmp_path, spec, frame, deepcopy(predictions), replay=True)
    assert all(r["delta_vs_current"] < 0 for r in result["comparisons"])
    assert result["platform_queue_first"] == "V22_I_ONLY"
    assert result["new_packages"] == 0
    assert all(r["candidate"] == "J1" for r in result["tiers"]["formal_selected"])


def test_ledger_timestamp_is_transport_metadata_but_model_fields_are_checked():
    from bf_tap_r2.v2_robust_joint import verify_ledger_record
    record = {"event": "complete", "model": "fold.joblib", "model_sha256": "abc"}
    verify_ledger_record(dict(record, time="2026-09-22T00:00:00Z"), record)
    with pytest.raises(ValueError, match="Ledger"):
        verify_ledger_record(dict(record, time="now", model_sha256="changed"), record)
    with pytest.raises(ValueError, match="Ledger"):
        verify_ledger_record(record, record)


def test_recovery_rejects_existing_or_nonprivate_destinations(tmp_path):
    from bf_tap_r2.v2_robust_joint import recover
    source = tmp_path / "local/runs/round2-v2.3/failed"
    source.mkdir(parents=True)
    with pytest.raises(ValueError, match="fresh private"):
        recover(tmp_path, source, source)
    with pytest.raises(ValueError, match="fresh private"):
        recover(tmp_path, source, tmp_path / "public")
    with pytest.raises(ValueError, match="retained failed"):
        recover(tmp_path, source, source.parent / "replay")
