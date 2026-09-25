from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bf_tap_r2.data import FEATURES
from bf_tap_r2.v4_1_residual import (
    ShrinkBinnedResidualCorrector,
    candidate_specs,
    cross_fitted_predictions,
    load_v41_config,
    make_corrector,
    public_anchor_trials,
    residual_feature_frame,
)
from bf_tap_r2.v4_1_run import _summarize
from bf_tap_r2.v4_1_report import _complete_events


class Memorizer:
    def fit(self, frame, target):
        self.rows_ = frame.loc[:, [*FEATURES, "spout_no"]].to_numpy(dtype=float)
        self.target_ = np.asarray(target, dtype=float)
        return self

    def predict(self, frame):
        rows = frame.loc[:, [*FEATURES, "spout_no"]].to_numpy(dtype=float)
        out = np.zeros(len(frame), dtype=float)
        for index, row in enumerate(rows):
            matches = np.all(np.isclose(self.rows_, row, rtol=0.0, atol=1e-12), axis=1)
            if matches.any():
                out[index] = self.target_[np.argmax(matches)]
        return out


def _frame(n=120, seed=4):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    frame.insert(0, "sample_id", [f"R2S_TRAIN_{i:05d}" for i in range(n)])
    frame.insert(1, "spout_no", np.where(np.arange(n) % 2, 2, 1))
    frame["tap_iron"] = 100.0 + frame["pig"] * 3.0 + rng.normal(scale=0.3, size=n)
    frame["tap_time_len"] = 80.0 + frame["all_quality"] * 2.0 + rng.normal(scale=0.4, size=n)
    return frame


def test_v41_plan_and_candidate_budget_are_frozen():
    config = load_v41_config(Path.cwd())
    assert config["status"] in {
        "PLAN_FROZEN_NOT_EXECUTED",
        "COMPLETE_DEVELOPMENT_FORMAL_TIME_CANDIDATE_NO_RELEASE",
    }
    assert config["protocol"]["split_seeds"] == [42, 3407]
    assert config["protocol"]["coarse_folds"] == [0, 1]
    assert config["continuation_gate"]["require_both_split_seeds_positive"] is True
    assert config["outputs"]["automatic_package"] is False
    for target in ("tap_iron", "tap_time_len"):
        specs = candidate_specs(config, target)
        assert len(specs) == 12
        assert {row["family"] for row in specs} == {
            "shrink_bins", "lightgbm_residual", "histgb_residual"
        }


def test_public_anchor_trials_are_rebuilt_without_private_ledgers():
    iron, iron_weights = public_anchor_trials(Path.cwd(), "tap_iron")
    time, time_weights = public_anchor_trials(Path.cwd(), "tap_time_len")
    assert [row["trial_id"] for row in iron] == [
        "v34-s1-ebm_boundary-0016", "v34-s1-ebm_boundary-0020"
    ]
    assert [row["trial_id"] for row in time] == ["v34-s1-ebm_boundary-0089"]
    assert np.isclose(iron_weights.sum(), 1.0)
    assert np.isclose(time_weights.sum(), 1.0)
    assert all(row["target_transform"] == "log1p" for row in [*iron, *time])


def test_cross_fit_excludes_the_row_from_a_memorizing_base():
    frame = _frame(60)
    target = np.arange(len(frame), dtype=float) + 1.0
    folds = np.arange(len(frame)) % 3
    oof, full = cross_fitted_predictions(frame, target, folds, Memorizer)
    assert np.allclose(oof, 0.0, rtol=0.0, atol=1e-12)
    assert np.allclose(full.predict(frame), target, rtol=0.0, atol=1e-12)


def test_residual_feature_frame_and_shrink_corrector_are_finite():
    frame = _frame(100)
    base = frame["tap_iron"].to_numpy(dtype=float) - np.where(frame.spout_no == 1, 1.5, -0.5)
    features = residual_feature_frame(frame, base)
    assert features.columns[0] == "anchor_prediction"
    assert features.shape == (len(frame), len(FEATURES) + 2)
    model = ShrinkBinnedResidualCorrector(
        bins=6, shrinkage=20.0, statistic="mean", alpha=0.5, clip_quantile=0.95
    ).fit(frame, base, frame["tap_iron"].to_numpy(dtype=float))
    correction = model.predict_correction(frame, base)
    assert correction.shape == (len(frame),)
    assert np.isfinite(correction).all()
    assert np.mean(correction[frame.spout_no.to_numpy() == 1]) > 0.0


def test_tree_correctors_fit_predict_and_keep_cross_fitted_source_label():
    frame = _frame(120)
    base = frame["tap_time_len"].to_numpy(dtype=float) + np.sin(frame["pig"].to_numpy())
    config = load_v41_config(Path.cwd())
    for spec in candidate_specs(config, "tap_time_len"):
        if spec["family"] == "shrink_bins":
            continue
        model = make_corrector(spec).fit(frame, base, frame["tap_time_len"].to_numpy(dtype=float))
        correction = model.predict_correction(frame.iloc[:13], base[:13])
        assert correction.shape == (13,)
        assert np.isfinite(correction).all()
        assert model.fit_meta_["residual_source"] == "cross_fitted_anchor_only"


def test_coarse_summary_requires_both_seeds_and_frozen_delta_gate():
    config = load_v41_config(Path.cwd())
    tasks = []
    for seed, good_delta, unstable_delta in ((42, 0.006, 0.02), (3407, 0.008, -0.001)):
        records = []
        for name, delta in (("GOOD", good_delta), ("UNSTABLE", unstable_delta)):
            anchor = 0.04
            records.append({
                "candidate_id": f"v41-tap_iron-{name}",
                "name": name,
                "family": "synthetic",
                "target": "tap_iron",
                "seed": seed,
                "anchor_wmape": anchor,
                "candidate_wmape": anchor - delta / 50.0,
                "package_delta_single": delta,
            })
        tasks.append({"records": records})
    summary = _summarize(tasks, config)
    by_name = {row["name"]: row for row in summary["candidates"]}
    assert by_name["GOOD"]["continuation_gate_passed"] is True
    assert by_name["UNSTABLE"]["continuation_gate_passed"] is False
    assert summary["qualifying_count"] == 1


def test_complete_event_reader_requires_both_time_seeds(tmp_path):
    ledger = tmp_path / "fit_ledger.jsonl"
    ledger.write_text(
        "\n".join(json.dumps(row) for row in [
            {"event": "complete", "target": "tap_time_len", "seed": 42},
            {"event": "complete", "target": "tap_time_len", "seed": 3407},
            {"event": "complete", "target": "tap_iron", "seed": 42},
        ]) + "\n",
        encoding="utf-8",
    )
    assert set(_complete_events(ledger)) == {42, 3407}
