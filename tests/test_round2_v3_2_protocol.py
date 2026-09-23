from types import SimpleNamespace

import pytest

from bf_tap_r2.v3_1_models import iteration_fields


def test_catboost_zero_index_is_one_round_not_configured_fallback():
    estimator = SimpleNamespace(get_best_iteration=lambda: 0, tree_count_=1)
    fields = iteration_fields("catboost", estimator, 20)
    assert fields == {
        "best_iteration_index": 0,
        "selected_num_boost_round": 1,
        "actual_num_boost_round": 1,
        "configured_num_boost_round": 20,
    }


def test_catboost_last_index_adds_one_round():
    estimator = SimpleNamespace(get_best_iteration=lambda: 19, tree_count_=20)
    fields = iteration_fields("catboost", estimator, 20)
    assert fields["best_iteration_index"] == 19
    assert fields["selected_num_boost_round"] == 20
    assert fields["actual_num_boost_round"] == 20


def test_catboost_no_valid_best_uses_configured_rounds_without_silent_zero():
    estimator = SimpleNamespace(get_best_iteration=lambda: -1, tree_count_=20)
    fields = iteration_fields("catboost", estimator, 20)
    assert fields["best_iteration_index"] is None
    assert fields["selected_num_boost_round"] == 20
    assert fields["actual_num_boost_round"] == 20


def test_lightgbm_uses_retained_iterations_directly():
    estimator = SimpleNamespace(best_iteration_=20)
    fields = iteration_fields("lightgbm", estimator, 100)
    assert fields["best_iteration_index"] == 19
    assert fields["selected_num_boost_round"] == 20
    assert fields["actual_num_boost_round"] == 20


def test_lightgbm_no_best_uses_configured_rounds():
    estimator = SimpleNamespace(best_iteration_=0)
    fields = iteration_fields("lightgbm", estimator, 100)
    assert fields["best_iteration_index"] is None
    assert fields["selected_num_boost_round"] == 100


def test_xgboost_zero_index_is_one_round_and_actual_rounds_are_read():
    class Booster:
        def num_boosted_rounds(self):
            return 1

    estimator = SimpleNamespace(best_iteration=0, get_booster=lambda: Booster())
    fields = iteration_fields("xgboost", estimator, 50)
    assert fields["best_iteration_index"] == 0
    assert fields["selected_num_boost_round"] == 1
    assert fields["actual_num_boost_round"] == 1


def test_xgboost_missing_or_negative_index_uses_configured_rounds():
    estimator = SimpleNamespace(best_iteration=-1)
    fields = iteration_fields("xgboost", estimator, 50)
    assert fields["best_iteration_index"] is None
    assert fields["selected_num_boost_round"] == 50


def test_iteration_fields_rejects_unknown_family_configuration():
    fields = iteration_fields("spline", SimpleNamespace(), 0)
    assert fields == {
        "best_iteration_index": None,
        "selected_num_boost_round": 1,
        "actual_num_boost_round": 1,
        "configured_num_boost_round": 1,
    }
