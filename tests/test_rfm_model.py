import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES
from bf_tap_r2.rfm_model import RFMRegressor, SavedRFM, fit_partition
from bf_tap_r2.rfm_protocol import ReservationLedger
from bf_tap_r2.v3_4_bags import group_safe_inner_folds
from bf_tap_r2.v3_6_networks import NumericPreprocessor


def sample(rows=40):
    rng = np.random.default_rng(56002)
    frame = pd.DataFrame(rng.normal(size=(rows, len(FEATURES))), columns=FEATURES)
    frame["sample_id"] = [f"synthetic-{i:03}" for i in range(rows)]
    frame["spout_no"] = np.arange(rows) % 2 + 1
    frame["tap_iron"] = 500 + 10 * frame.air_volume + 3 * frame.oxygen
    frame["tap_time_len"] = 100 + 2 * frame.air_volume + frame.humidity
    return frame


def recorder(tmp_path, name="events"):
    return ReservationLedger.create(tmp_path / name, {"procedure": 4, "solve": 16, "update": 12})


def test_full_state_zero_matches_fixed_and_has_exact_native_transform(tmp_path):
    frame = sample()
    fixed = RFMRegressor("FIXED_KRR").fit_path(frame, frame.tap_iron.to_numpy(), 0,
        recorder(tmp_path, "fixed").scoped(("fixed",)))
    full = RFMRegressor("FULL_RFM").fit_path(frame, frame.tap_iron.to_numpy(), 3,
        recorder(tmp_path, "full").scoped(("full",)))
    query = sample(12)
    np.testing.assert_array_equal(fixed[0].predict(query), full[0].predict(query))
    np.testing.assert_array_equal(full[0].metric, np.eye(21))
    transform = NumericPreprocessor(structure="raw_tabm").fit(frame)
    numeric, category = transform.transform_mlp(frame)
    np.testing.assert_array_equal(full[0].centers, numeric.astype(np.float64))
    np.testing.assert_array_equal(full[0].center_categories, category.astype(np.float64))
    assert len(full) == 4 and full[-1].state == 3
    for model in full:
        assert model.bandwidth == full[0].bandwidth
        assert model.target_mean == frame.tap_iron.to_numpy().mean()
        assert model.target_std == frame.tap_iron.to_numpy().std()
        assert model.solver_residual < 1e-8


def test_inner_isolation_fresh_refit_and_artifacts(tmp_path, monkeypatch):
    import bf_tap_r2.rfm_model as module
    frame = sample()
    calls = []
    native = module.RFMRegressor.fit_path
    def capture(self, data, y, updates, reserve, **kwargs):
        path = native(self, data, y, updates, reserve, **kwargs)
        calls.append((data.copy(), np.array(y), path))
        return path
    monkeypatch.setattr(module.RFMRegressor, "fit_path", capture)
    ledger = recorder(tmp_path)
    model, metadata = fit_partition(frame, "tap_time_len", "FULL_RFM", ledger.scoped(("unit",)),
                                    directory=tmp_path / "models")
    mask = group_safe_inner_folds(frame, seed=42)["fold"] != 0
    assert calls[0][0].sample_id.tolist() == frame.loc[mask, "sample_id"].tolist()
    assert calls[1][0].sample_id.tolist() == frame.sample_id.tolist()
    assert len(calls[1][2]) == metadata["selected_state"] + 1
    np.testing.assert_array_equal(calls[1][2][0].metric, np.eye(21))
    # Independent reduction can use a different array layout from native fit.
    np.testing.assert_allclose(calls[0][2][0].preprocessor.means_, frame.loc[mask, FEATURES].to_numpy().mean(0), rtol=0, atol=1e-14)
    np.testing.assert_allclose(model.preprocessor.means_, frame.loc[:, FEATURES].to_numpy().mean(0), rtol=0, atol=1e-14)
    assert calls[0][2][0].bandwidth != calls[1][2][0].bandwidth
    assert metadata["calibration_ids"] == frame.loc[~mask, "sample_id"].tolist()
    assert model.fit_ids == frame.sample_id.tolist()
    report = ledger.inspect()
    assert report["completed"]["procedure"] == 2
    assert report["completed"]["solve"] == 5 + metadata["selected_state"]
    assert report["completed"]["update"] == 3 + metadata["selected_state"]
    assert (tmp_path / "models" / "selection.json").is_file()
    assert (tmp_path / "models" / "inner" / "state-3" / "complete.json").is_file()


def test_unknown_category_and_query_target_id_are_not_predictors(tmp_path):
    frame = sample()
    frame["spout_no"] = 2
    model = RFMRegressor("FIXED_KRR").fit_path(frame, frame.tap_iron.to_numpy(), 0,
        recorder(tmp_path).scoped(("unit",)))[0]
    query = sample(8)
    query["spout_no"] = 999
    _, categories = model.inputs(query)
    np.testing.assert_array_equal(categories[:, 0], np.ones(8))
    assert categories.shape == (8, 2)
    base = model.predict(query)
    altered = query.assign(sample_id="irrelevant", tap_iron=np.nan, tap_time_len=-1e12)
    np.testing.assert_array_equal(base, model.predict(altered))
    np.testing.assert_array_equal(base, model.predict(query.drop(columns=["sample_id", "tap_iron", "tap_time_len"])))


def test_saved_cold_roundtrip_and_tamper(tmp_path):
    frame = sample()
    model = RFMRegressor("FULL_RFM").fit_path(frame, frame.tap_iron.to_numpy(), 1,
        recorder(tmp_path).scoped(("unit",)))[-1]
    directory = tmp_path / "saved"
    identity = model.save(directory)
    cold = SavedRFM.load(directory, expected_sha256=identity["complete_sha256"])
    np.testing.assert_array_equal(model.predict(frame), cold.predict(frame))
    np.testing.assert_allclose(model.predict(frame)[::-1], cold.predict(frame.iloc[::-1]), rtol=0, atol=1e-8)
    np.testing.assert_allclose(model.predict(frame), np.concatenate([
        cold.predict(frame.iloc[i:i+3]) for i in range(0, len(frame), 3)]), rtol=0, atol=1e-8)
    with pytest.raises(FileExistsError):
        model.save(directory)
    with pytest.raises(ValueError, match="identity"):
        SavedRFM.load(directory, expected_sha256="0" * 64)
    with np.load(directory / "arrays.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    arrays["centers"][0, 0] += 1
    np.savez(directory / "arrays.npz", **arrays)
    with pytest.raises(ValueError, match="hash"):
        SavedRFM.load(directory, expected_sha256=identity["complete_sha256"])


def test_cold_fresh_process_and_no_fitting_at_prediction(tmp_path, monkeypatch):
    frame = sample()
    model = RFMRegressor("FIXED_KRR").fit_path(frame, frame.tap_iron.to_numpy(), 0,
        recorder(tmp_path).scoped(("unit",)))[0]
    identity = model.save(tmp_path / "model")
    frame.to_json(tmp_path / "query.json", orient="table", double_precision=15)
    code = """
import json, pathlib, sys, numpy as np, pandas as pd
from bf_tap_r2 import rfm_model as m
def forbidden(*args, **kwargs): raise RuntimeError('cold attempted fitting')
m.solve_kernel = forbidden
m.NumericPreprocessor.fit = forbidden
p = pathlib.Path(sys.argv[1])
model = m.SavedRFM.load(p/'model', expected_sha256=sys.argv[2])
np.save(p/'cold.npy', model.predict(pd.read_json(p/'query.json', orient='table')))
"""
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    subprocess.run([sys.executable, "-c", code, str(tmp_path), identity["complete_sha256"]],
                   check=True, env=env, capture_output=True, text=True)
    np.testing.assert_allclose(np.load(tmp_path / "cold.npy"), model.predict(frame), rtol=0, atol=1e-8)


def test_same_selection_error_prefers_earliest_state(tmp_path, monkeypatch):
    monkeypatch.setattr(SavedRFM, "predict", lambda self, query: np.ones(len(query)))
    _, meta = fit_partition(sample(), "tap_iron", "FULL_RFM", recorder(tmp_path).scoped(("unit",)))
    assert meta["selected_state"] == 0


def test_calibration_poison_cannot_change_inner_fit_path(tmp_path, monkeypatch):
    import bf_tap_r2.rfm_model as module
    frame = sample()
    mask = group_safe_inner_folds(frame, seed=42)["fold"] == 0
    paths = []
    original = module.RFMRegressor.fit_path
    def capture(self, data, y, updates, reserve, **kwargs):
        models = original(self, data, y, updates, reserve, **kwargs)
        paths.append(models)
        return models
    monkeypatch.setattr(module.RFMRegressor, "fit_path", capture)
    fit_partition(frame, "tap_iron", "FULL_RFM", recorder(tmp_path, "first").scoped(("first",)))
    changed = frame.copy()
    changed.loc[mask, list(FEATURES)] += 100
    changed.loc[mask, "tap_iron"] += 1000
    changed["tap_time_len"] = np.nan  # nonselected target is never read
    fit_partition(changed, "tap_iron", "FULL_RFM", recorder(tmp_path, "second").scoped(("second",)))
    assert paths[0][0].fit_ids == paths[2][0].fit_ids
    for before, after in zip(paths[0], paths[2]):
        np.testing.assert_array_equal(before.alpha, after.alpha)
        np.testing.assert_array_equal(before.metric, after.metric)
        np.testing.assert_array_equal(before.centers, after.centers)
        assert before.bandwidth == after.bandwidth
        assert before.target_mean == after.target_mean


def test_artifact_metadata_tamper_rejected_by_external_identity(tmp_path):
    frame = sample()
    model = RFMRegressor("FIXED_KRR").fit_path(frame, frame.tap_iron.to_numpy(), 0,
        recorder(tmp_path).scoped(("unit",)))[0]
    identity = model.save(tmp_path / "model")
    metadata = tmp_path / "model" / "metadata.json"
    record = json.loads(metadata.read_text())
    record["target_mean"] += 10
    metadata.write_text(json.dumps(record))
    from bf_tap_r2.rfm_protocol import file_hash
    complete = tmp_path / "model" / "complete.json"
    envelope = json.loads(complete.read_text())
    envelope["hashes"]["metadata.json"] = file_hash(metadata)
    complete.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="identity"):
        SavedRFM.load(tmp_path / "model", expected_sha256=identity["complete_sha256"])


@pytest.mark.parametrize("defect", ["constant_feature", "constant_target", "nan_target", "duplicate_id", "nan_feature"])
def test_bad_training_input_is_retained_failure_before_solve(tmp_path, defect):
    frame = sample()
    y = frame.tap_iron.to_numpy(copy=True)
    if defect == "constant_feature": frame[FEATURES[0]] = 1
    if defect == "constant_target": y[:] = 1
    if defect == "nan_target": y[0] = np.nan
    if defect == "duplicate_id": frame.loc[1, "sample_id"] = frame.loc[0, "sample_id"]
    if defect == "nan_feature": frame.loc[0, FEATURES[0]] = np.nan
    ledger = recorder(tmp_path)
    with pytest.raises(ValueError):
        RFMRegressor("FULL_RFM").fit_path(frame, y, 3, ledger.scoped(("unit",)))
    report = ledger.inspect()
    assert report["failed"]["procedure"] == 1
    assert report["started"]["solve"] == 0


def test_solver_failure_keeps_prior_state_and_prevents_same_key_retry(tmp_path, monkeypatch):
    from bf_tap_r2 import rfm_model as m
    original = m.solve_kernel
    count = 0
    def fail_second(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2: raise RuntimeError("deliberate second-state failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(m, "solve_kernel", fail_second)
    frame = sample()
    ledger = recorder(tmp_path)
    with pytest.raises(RuntimeError):
        RFMRegressor("FULL_RFM").fit_path(frame, frame.tap_iron.to_numpy(), 3,
            ledger.scoped(("unit",)), directory=tmp_path / "models")
    assert (tmp_path / "models" / "state-0" / "complete.json").exists()
    assert ledger.inspect()["failed"]["solve"] == 1
    with pytest.raises(FileExistsError):
        RFMRegressor("FULL_RFM").fit_path(frame, frame.tap_iron.to_numpy(), 3, ledger.scoped(("unit",)))
