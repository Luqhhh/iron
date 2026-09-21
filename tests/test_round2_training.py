from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("sklearn")

from bf_tap_r2.cv import baseline_predictions, combined_metrics
from bf_tap_r2.data import FEATURES
from bf_tap_r2.models import SnapshotRegressor, inputs
from bf_tap_r2.splits import make_folds


class FakeRegressor:
    """No fitting: exercise orchestration and enforce train/validation separation."""
    def __init__(self, route, config):
        self.target_scale_ = 1.0

    def fit(self, frame, target):
        self.training_ids = set(frame.sample_id)
        self.value = float(np.median(target))
        return self

    def predict(self, frame):
        assert not (self.training_ids & set(frame.sample_id))
        return np.repeat(self.value, len(frame))


def sample_frame(n=40):
    rng = np.random.default_rng(1)
    return pd.DataFrame({"sample_id": [f"R2S_TRAIN_{i:03d}" for i in range(n)],
                         "spout_no": np.arange(n) % 2 + 1,
                         **{name: rng.uniform(0, 10, n) for name in FEATURES},
                         "tap_iron": rng.uniform(10, 20, n), "tap_time_len": rng.uniform(1, 2, n)})


def config():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/round2_v0_1/models.yaml").read_text())


def test_folds_are_stable_id_aligned_complete_and_shared():
    frame = sample_frame()
    a = make_folds(frame, 42)
    pd.testing.assert_frame_equal(a, make_folds(frame.sample(frac=1, random_state=9), 42))
    assert a.sample_id.nunique() == len(frame)
    assert set(a.fold) == set(range(5))
    assert not np.array_equal(a.fold, make_folds(frame, 3407).fold)
    counts = a.groupby(["fold", "spout_no"]).size()
    assert counts.max() - counts.min() <= 1


def test_groups_never_cross_folds():
    frame = sample_frame(100)
    groups = pd.Series(np.arange(100) // 4)
    result = make_folds(frame, 42, groups=groups)
    assert result.groupby("group_id").fold.nunique().max() == 1


@pytest.mark.parametrize("route", ["L1", "L2", "S1"])
def test_preprocessing_only_fits_training_fold_and_predictions_align(route):
    frame = sample_frame()
    train = frame.iloc[:30].copy()
    valid = frame.iloc[30:].copy()
    valid.loc[:, FEATURES[0]] = 1e6
    valid.loc[:, "spout_no"] = 99
    model = SnapshotRegressor(route, config()).fit(train, train.tap_iron)
    pre = model.estimator_.named_steps["preprocess"]
    encoder = pre.named_transformers_["spout"]
    assert 99 not in encoder.categories_[0]
    numeric = pre.named_transformers_["numeric"]
    if route in ("L1", "L2"):
        np.testing.assert_allclose(numeric.mean_, train[list(FEATURES)].mean())
        assert numeric.n_samples_seen_ == len(train)
    else:
        assert numeric.bsplines_[0].t.max() < 100
    assert model.target_scale_ == (train.tap_iron.median() if route == "L2" else 1)
    expected = model.predict(valid)
    np.testing.assert_allclose(expected, model.predict(valid.iloc[::-1])[::-1], rtol=1e-12)
    shuffled_ids = valid.copy()
    shuffled_ids["sample_id"] = "unused_id"
    np.testing.assert_array_equal(expected, model.predict(shuffled_ids))
    if route in ("L1", "L2"):
        assert numeric.n_samples_seen_ == len(train)


def test_inputs_exclude_targets_and_ids_and_reject_missing():
    frame = sample_frame()
    assert inputs(frame).columns.tolist() == [*FEATURES, "spout_no"]
    frame.loc[0, FEATURES[0]] = np.nan
    with pytest.raises(ValueError):
        inputs(frame)


def test_median_baselines_use_only_training_fold():
    frame = sample_frame(10)
    folds = np.array([0] * 5 + [1] * 5)
    frame.loc[:4, "tap_iron"] = 1000000
    pred = baseline_predictions(frame, "tap_iron", folds, False)
    np.testing.assert_array_equal(pred[:5], np.repeat(frame.tap_iron.iloc[5:].median(), 5))
    frame.loc[:4, "spout_no"] = 99
    by_spout = baseline_predictions(frame, "tap_iron", folds, True)
    np.testing.assert_array_equal(by_spout[:5], pred[:5])


def test_combined_metrics_are_target_equal_weight():
    score = {"tap_iron": {"wmape": .1, "by_fold": {"0": .2}, "by_spout": {"1": .3}},
             "tap_time_len": {"wmape": .3, "by_fold": {"0": .4}, "by_spout": {"1": .5}}}
    result = combined_metrics(score)
    assert result["J"] == pytest.approx(.2)
    assert result["J_by_fold"]["0"] == pytest.approx(.3)
    assert result["J_by_spout"]["1"] == pytest.approx(.4)


def test_cv_orchestration_budget_and_oof_coverage(tmp_path, monkeypatch):
    import json
    import shutil
    from bf_tap_r2 import cv

    repo = Path(__file__).resolve().parents[1]
    shutil.copytree(repo / "configs/round2_v0_1", tmp_path / "configs/round2_v0_1")
    (tmp_path / "uv.lock").write_text("synthetic lock")
    frame = sample_frame()
    monkeypatch.setattr(cv, "verify_audit", lambda *args: {})
    monkeypatch.setattr(cv, "load_snapshot", lambda *args: (frame, {}))
    monkeypatch.setattr(cv.subprocess, "check_output", lambda *args, **kwargs: "synthetic-code-sha")
    monkeypatch.setattr(cv, "SnapshotRegressor", FakeRegressor)
    output = tmp_path / "local/runs/round2-v0.1/test"
    result = cv.run(tmp_path, output, tmp_path / "unused")
    assert result["regressor_fits"] == 70
    assert result["baseline_fold_fits"] == 40
    assert result["recheck_selection"] == {"tap_iron": ["C1", "C2"], "tap_time_len": ["C1", "C2"]}
    assert len(pd.read_csv(output / "fold_assignments.csv")) == 2 * len(frame)
    for route in ("L1", "L2", "S1", "C1", "C2"):
        oof = pd.read_csv(output / "oof" / f"42-{route}-tap_iron.csv")
        assert oof.sample_id.nunique() == len(frame)
        assert oof.pred_tap_iron.notna().all()
    events = [json.loads(line) for line in (output / "fit_ledger.jsonl").read_text().splitlines()]
    assert sum(e["kind"] == "regressor_fit_complete" for e in events) == 70
    with pytest.raises(FileExistsError):
        cv.run(tmp_path, output, tmp_path / "unused")
