from copy import deepcopy
import inspect

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES
from bf_tap_r2.row_nested_blend import select_and_predict, score_prediction, synthetic_probe


class Constant:
    def __init__(self, value=10.0, calls=None):
        self.value, self.calls = value, calls

    def fit(self, frame, labels):
        assert set(frame.columns) == {"sample_id", "spout_no", *FEATURES}
        assert (labels < 1000).all()  # Sentinel E labels may never enter fitting.
        if self.calls is not None:
            self.calls.append((tuple(frame.sample_id), tuple(labels)))
        return self

    def predict(self, frame):
        return np.full(len(frame), self.value)


@pytest.fixture
def inputs():
    values = np.arange(16, dtype=float)
    frame = pd.DataFrame({name: values + j for j, name in enumerate(FEATURES)})
    frame["sample_id"] = [f"row-{i}" for i in range(16)]
    frame["spout_no"] = 1 + np.arange(16) % 2
    return (frame.iloc[:12].copy(), pd.Series(12.0, index=frame.sample_id.iloc[:12]),
            frame.iloc[12:].copy())


def arguments(calls=None):
    return dict(inner_folds=np.arange(12) % 3,
                reference_factory=lambda: Constant(10.0, calls),
                candidate_factories={"better": lambda: Constant(12.0, calls)},
                alpha_grid=[1.0, 0.5, 0.0], max_fit_calls=8)


def test_engine_builds_inner_oof_and_fresh_outer_refits(inputs):
    train, labels, evaluation = inputs
    calls = []
    result = select_and_predict(train, labels, evaluation, **arguments(calls))
    assert result.candidate == "better" and result.alpha == 1.0
    np.testing.assert_array_equal(result.prediction, np.full(4, 12.0))
    assert len(result.fit_ledger) == len(calls) == 8
    train_ids, eval_ids = set(train.sample_id), set(evaluation.sample_id)
    for event in result.fit_ledger:
        fitted, predicted = set(event["training_ids"]), set(event["prediction_ids"])
        assert fitted <= train_ids and not fitted & eval_ids
        assert not fitted & predicted
        if event["phase"] == "outer_refit":
            assert fitted == train_ids and predicted == eval_ids
    for model in ["reference", "better"]:
        ids = [i for event in result.fit_ledger if event["phase"] == "inner_oof"
               and event["model"] == model for i in event["prediction_ids"]]
        assert len(ids) == len(set(ids)) == len(train)
        assert set(ids) == train_ids


def test_evaluation_labels_are_only_accepted_by_separate_scoring(inputs):
    assert "evaluation_labels" not in inspect.signature(select_and_predict).parameters
    train, labels, evaluation = inputs
    first = select_and_predict(train, labels, evaluation, **arguments())
    old_score = score_prediction(pd.Series(12.0, index=evaluation.sample_id), first)
    changed_score = score_prediction(pd.Series(1200.0, index=evaluation.sample_id), first)
    second = select_and_predict(train, labels, evaluation, **arguments())
    assert old_score != changed_score
    assert (first.candidate, first.alpha, first.fit_ledger) == (second.candidate, second.alpha, second.fit_ledger)
    np.testing.assert_array_equal(first.prediction, second.prediction)


@pytest.mark.parametrize("defect", ["id_overlap", "id_duplicate", "outer_group", "inner_group",
                                    "hidden_label", "hidden_feature", "duplicate_column",
                                    "one_fold", "bad_fold", "nonfinite_label", "bad_budget",
                                    "zero_missing", "duplicate_alpha", "nonfinite_alpha",
                                    "global_oof"])
def test_bad_boundaries_and_protocols_fail_before_any_fit(inputs, defect):
    train, labels, evaluation = deepcopy(inputs)
    calls = []
    params = arguments(calls)
    if defect == "id_overlap":
        evaluation.loc[evaluation.index[0], "sample_id"] = train.sample_id.iloc[0]
    elif defect == "id_duplicate":
        train.loc[train.index[1], "sample_id"] = train.sample_id.iloc[0]
    elif defect == "outer_group":
        evaluation.loc[evaluation.index[0], list(FEATURES)] = train[list(FEATURES)].iloc[0].to_numpy()
    elif defect == "inner_group":
        train.loc[train.index[1], list(FEATURES)] = train[list(FEATURES)].iloc[0].to_numpy()
    elif defect == "hidden_label":
        evaluation["tap_iron"] = 1200.0
    elif defect == "hidden_feature":
        train["global_prediction"] = labels
    elif defect == "duplicate_column":
        train = pd.concat([train, train[[FEATURES[0]]]], axis=1)
    elif defect == "one_fold":
        params["inner_folds"] = np.zeros(12, dtype=int)
    elif defect == "bad_fold":
        params["inner_folds"] = np.full(12, -1, dtype=int)
    elif defect == "nonfinite_label":
        labels.iloc[0] = np.nan
    elif defect == "bad_budget":
        params["max_fit_calls"] = 7
    elif defect == "zero_missing":
        params["alpha_grid"] = [0.5, 1.0]
    elif defect == "duplicate_alpha":
        params["alpha_grid"] = [0.0, 0.0]
    elif defect == "nonfinite_alpha":
        params["alpha_grid"] = [0.0, np.nan]
    elif defect == "global_oof":
        params["candidate_factories"] = {"old_oof": np.ones(12)}
    with pytest.raises(ValueError):
        select_and_predict(train, labels, evaluation, **params)
    assert calls == []


def test_duplicate_groups_are_kept_together_even_across_spouts(inputs):
    train, labels, evaluation = inputs
    train.loc[1, list(FEATURES)] = train[list(FEATURES)].iloc[0].to_numpy()
    params = arguments()
    params["inner_folds"][1] = params["inner_folds"][0]
    assert select_and_predict(train, labels, evaluation, **params).alpha == 1.0


def test_zero_gain_prefers_reference_and_spends_no_candidate_outer_fit(inputs):
    train, labels, evaluation = inputs
    params = arguments()
    params["candidate_factories"] = {"tie": lambda: Constant(10.0)}
    result = select_and_predict(train, labels, evaluation, **params)
    assert result.candidate is None and result.alpha == 0.0
    assert len(result.fit_ledger) == 7


def test_frozen_candidate_tie_order_and_evaluation_row_order(inputs):
    train, labels, evaluation = inputs
    params = arguments()
    params["candidate_factories"] = {"first": lambda: Constant(12.0), "second": lambda: Constant(12.0)}
    params["max_fit_calls"] = 11
    result = select_and_predict(train, labels, evaluation.iloc[::-1], **params)
    assert result.candidate == "first"
    assert result.evaluation_ids == tuple(evaluation.sample_id.iloc[::-1])


def test_reusing_an_estimator_across_inner_folds_is_refused(inputs):
    singleton = Constant()
    params = arguments()
    params["reference_factory"] = lambda: singleton
    with pytest.raises(ValueError, match="fresh estimators"):
        select_and_predict(*inputs, **params)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_predictions_fail_closed(inputs, value):
    params = arguments()
    params["reference_factory"] = lambda: Constant(value)
    with pytest.raises(ValueError, match="finite values"):
        select_and_predict(*inputs, **params)


def test_synthetic_probe_is_exactly_reproducible_without_official_targets():
    first, second = synthetic_probe(), synthetic_probe()
    assert first == second
    assert first["official_target_fits"] == 0
    assert first["fit_calls_per_run"] == 8
    assert first["prediction_equal_after_evaluation_label_change"]
    assert first["choice_equal_after_evaluation_label_change"] and first["fit_ledger_equal"]


def test_labels_are_id_aligned_under_shuffling(inputs):
    train, labels, evaluation = inputs
    first = select_and_predict(train, labels, evaluation, **arguments())
    shuffled = select_and_predict(train, labels.iloc[::-1], evaluation, **arguments())
    assert first.fit_ledger == shuffled.fit_ledger
    np.testing.assert_array_equal(first.prediction, shuffled.prediction)
    actual = pd.Series([12.0, 13.0, 14.0, 15.0], index=evaluation.sample_id)
    assert score_prediction(actual, first) == score_prediction(actual.iloc[::-1], first)


@pytest.mark.parametrize("defect", ["missing", "extra", "duplicate", "anonymous"])
def test_bad_score_label_identity_is_refused(inputs, defect):
    train, labels, evaluation = inputs
    result = select_and_predict(train, labels, evaluation, **arguments())
    actual = pd.Series(12.0, index=evaluation.sample_id)
    if defect == "missing":
        actual = actual.iloc[:-1]
    elif defect == "extra":
        actual.loc["unknown"] = 12.0
    elif defect == "duplicate":
        actual = pd.concat([actual, actual.iloc[:1]])
    elif defect == "anonymous":
        actual = actual.to_numpy()
    with pytest.raises(ValueError, match="sample-ID index"):
        score_prediction(actual, result)
