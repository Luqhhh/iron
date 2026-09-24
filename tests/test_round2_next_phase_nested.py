"""Nesting-contract tests for the next-phase outer evaluation harness.

The harness exists because V3.4's final-outer driver was never checked in.
Its whole reason for existing is that nothing outside the current training
part may inform a fold's weights or its member fits, so most of these tests
attack that boundary directly with members that would expose a violation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.metrics import wmape
from bf_tap_r2.next_phase_nested import evaluate_outer, fit_weights, inner_oof
from bf_tap_r2.splits import make_folds


def frame(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {name: rng.normal(size=rows) for name in FEATURES}
    data["tap_iron"] = rng.normal(size=rows) * 100.0 + 500.0
    data["tap_time_len"] = rng.normal(size=rows) * 20.0 + 120.0
    data["sample_id"] = [f"R2S2_TRAIN_{i:012X}" for i in range(rows)]
    data["spout_no"] = [1 + (i % 2) for i in range(rows)]
    return pd.DataFrame(data)


class Memoriser:
    """Returns the target it saw in training, else zero.

    A leaked harness lets a row be predicted by a fit that already saw it.
    A correctly nested harness must return zero for every row.
    """

    name = "memoriser"

    def fit_predict(self, train, valid, target):
        known = dict(zip(train["sample_id"], train[target].to_numpy(dtype=float)))
        return np.array([known.get(sid, 0.0) for sid in valid["sample_id"]], dtype=float)


class MeanPredictor:
    """Predicts its own training-part mean; carries no signal."""

    def __init__(self, name: str):
        self.name = name

    def fit_predict(self, train, valid, target):
        return np.full(len(valid), float(train[target].mean()))


def test_inner_oof_shape_and_coverage():
    train = frame(60, 1)
    members = [Memoriser(), MeanPredictor("mean")]
    predictions = inner_oof(train, "tap_iron", members, inner_seed=7, inner_folds=5)
    assert predictions.shape == (60, 2)
    assert np.isfinite(predictions).all()


def test_inner_oof_never_predicts_a_row_with_a_fit_that_saw_it():
    train = frame(60, 2)
    predictions = inner_oof(train, "tap_iron", [Memoriser()], inner_seed=7, inner_folds=5)
    assert np.allclose(predictions, 0.0)


def test_inner_oof_rejects_members_that_return_the_wrong_length():
    class Short:
        name = "short"

        def fit_predict(self, train, valid, target):
            return np.zeros(len(valid) - 1)

    with pytest.raises(ValueError):
        inner_oof(frame(40, 3), "tap_iron", [Short()], inner_seed=7, inner_folds=5)


def test_inner_oof_rejects_non_finite_predictions():
    class NotFinite:
        name = "nan"

        def fit_predict(self, train, valid, target):
            return np.full(len(valid), np.nan)

    with pytest.raises(ValueError):
        inner_oof(frame(40, 4), "tap_iron", [NotFinite()], inner_seed=7, inner_folds=5)


def test_fit_weights_are_a_simplex_and_ignore_a_useless_member():
    train = frame(120, 5)
    y = train["tap_iron"].to_numpy(dtype=float)
    informative = y + np.random.default_rng(6).normal(scale=1.0, size=len(y))
    useless = np.full(len(y), float(y.mean()))
    result = fit_weights(np.column_stack([useless, informative]), y, ["useless", "informative"])
    weights = result["weights"]
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()
    assert weights[1] > weights[0]
    assert result["objective"] < 0.01


def test_fit_weights_reports_a_perfect_member_as_zero_error():
    train = frame(40, 7)
    y = train["tap_iron"].to_numpy(dtype=float)
    result = fit_weights(y[:, None], y, ["exact"])
    assert np.isclose(result["weights"][0], 1.0)
    assert result["objective"] == pytest.approx(0.0, abs=1e-12)


def test_evaluate_outer_pools_over_the_whole_coverage_set():
    train = frame(120, 8)
    report = evaluate_outer(
        train,
        [MeanPredictor("mean")],
        outer_seed=42,
        inner_seed=7771,
        outer_folds=5,
        inner_folds=5,
    )
    assert set(report["targets"]) == set(TARGETS)
    folds = (
        make_folds(train, 42).set_index("sample_id").loc[train.sample_id, "fold"].to_numpy()
    )
    for target in TARGETS:
        y = train[target].to_numpy(dtype=float)
        expected = np.zeros(len(train))
        for fold in range(5):
            # MeanPredictor returns its training-part mean, so the pooled value
            # is reproducible without trusting the harness's own arithmetic.
            expected[folds == fold] = y[folds != fold].mean()
        entry = report["targets"][target]
        assert len(entry["fold_wmape"]) == 5
        assert entry["pooled_wmape"] == pytest.approx(wmape(y, expected))
    assert report["package_score"] == pytest.approx(
        100.0
        - 50.0
        * (
            report["targets"]["tap_iron"]["pooled_wmape"]
            + report["targets"]["tap_time_len"]["pooled_wmape"]
        )
    )


def test_package_score_matches_the_historical_frozen_formula():
    from bf_tap_r2.v3_local_search import package_local_score

    train = frame(120, 11)
    report = evaluate_outer(train, [MeanPredictor("mean")], outer_seed=42, inner_seed=7771)
    assert report["package_score"] == pytest.approx(
        package_local_score(
            report["targets"]["tap_iron"]["pooled_wmape"],
            report["targets"]["tap_time_len"]["pooled_wmape"],
        )
    )


def test_catboost_member_uses_the_frozen_recipe_and_fits():
    from bf_tap_r2.next_phase_nested import C2_PARAMS, CatBoostMember

    assert CatBoostMember("c2_raw").params["iterations"] == C2_PARAMS["iterations"] == 1500
    fast = CatBoostMember("c2_fast", params={**C2_PARAMS, "iterations": 20, "thread_count": 1})
    train = frame(80, 10)
    values = fast.fit_predict(train, train, "tap_iron")
    assert values.shape == (80,)
    assert np.isfinite(values).all()


def test_r0_is_the_single_reproducible_c2_anchor():
    from bf_tap_r2.next_phase_nested import r0_members

    members = r0_members()
    assert [member.name for member in members] == ["c2_raw"]


def test_evaluate_outer_only_pays_for_the_targets_it_is_asked_for():
    train = frame(120, 9)
    calls: list[str] = []

    class Recorder(MeanPredictor):
        def fit_predict(self, train_part, valid, target):
            calls.append(target)
            return super().fit_predict(train_part, valid, target)

    evaluate_outer(
        train,
        [Recorder("recorder")],
        outer_seed=42,
        inner_seed=7771,
        outer_folds=5,
        inner_folds=5,
        targets=("tap_time_len",),
    )
    assert set(calls) == {"tap_time_len"}
