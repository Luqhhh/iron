from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.preprocessing import SplineTransformer

from bf_tap_r2.v4_cross_target import (
    CrossTargetChainRegressor,
    SharedOnlyCrossTargetRegressor,
    SharedPrivateCrossTargetRegressor,
)
from bf_tap_r2.v4_leaf_estimators import (
    HonestLeafForest,
    LeafPartitionForest,
    assert_lightgbm_linear_tree_supported,
    fit_lightgbm_linear_tree,
)
from bf_tap_r2.v4_projection_models import (
    CrossFittedResidualProjectionRegressor,
    FixedLinearCoordinateRegressor,
    ProjectionPursuitRegressor,
)
from bf_tap_r2.v4_run import unit_specs as v4_unit_specs
from bf_tap_r2.v4_smooth_models import (
    LinearControlRegressor,
    PairwiseTensorProductRegressor,
    SmoothAdditiveRegressor,
    TripleTensorProductRegressor,
    assert_ebm_has_no_terms_above,
    assert_ebm_term_arity,
    assert_spline_transformer_univariate,
    fit_ebm_with_explicit_triple,
    select_pair_terms_internal_validation,
)


def _data(n: int = 160, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6))
    # Heteroscedastic and nonlinear signal keeps leaf mean/median/mixture
    # estimands distinct without importing real competition data.
    y = (
        2.0 * x[:, 0]
        + np.sin(1.7 * x[:, 1])
        + 0.5 * x[:, 2] * x[:, 3]
        + rng.normal(scale=0.15 + 0.8 * np.abs(x[:, 1]), size=n)
    )
    return x, y


class Memorizer:
    """Leakage sentinel: return stored labels only for exact training rows."""

    def fit(self, x, y):
        self.x_ = np.asarray(x, dtype=float).copy()
        self.y_ = np.asarray(y, dtype=float).copy()
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        out = np.zeros(len(x), dtype=float)
        for row_index, row in enumerate(x):
            matches = np.all(np.isclose(self.x_, row, rtol=0.0, atol=1e-12), axis=1)
            if matches.any():
                out[row_index] = float(self.y_[np.argmax(matches)])
        return out


class ZeroResidualModel:
    def fit(self, x, y):
        self.n_features_in_ = np.asarray(x).shape[1]
        return self

    def predict(self, x):
        return np.zeros(len(x), dtype=float)


def test_v4_config_is_plan_only_and_budget_is_frozen():
    cfg_path = Path("configs/round2_v4/mechanism_search.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["status"] in {"PLAN_ONLY_NOT_EXECUTED", "COARSE_SCREEN_COMPLETED_NO_EXTENSION"}
    assert cfg["objective"]["platform_target_strictly_greater_than"] == 96.3
    assert cfg["search_budget"]["first_round_units_max"] == 40
    assert cfg["search_budget"]["extension_units_max"] == 24
    assert cfg["search_budget"]["extension_numeric_neighborhood_max"] == 8
    assert {key: cfg["families"][key]["units_max"] for key in ("F", "S", "P", "J")} == {
        "F": 12,
        "S": 12,
        "P": 8,
        "J": 8,
    }
    assert cfg["evaluation"]["complete_development"]["split_seeds"] == [42, 3407]
    assert cfg["evaluation"]["residuals_and_feature_region_diagnostics"]["final_outer_seed"] == 23003
    if cfg["status"] == "COARSE_SCREEN_COMPLETED_NO_EXTENSION":
        assert cfg["evaluation"]["residuals_and_feature_region_diagnostics"]["final_outer_status"].startswith("CONSUMED")
        assert cfg["execution"]["extension_triggered"] is False
        assert cfg["execution"]["extension_units_used"] == 0
    else:
        assert cfg["evaluation"]["residuals_and_feature_region_diagnostics"]["final_outer_status"] == "UNCONSUMED"
    assert cfg["promotion"]["agent_uploads"] == 0
    assert cfg["promotion"]["automatic_packaging"] is False


def test_leaf_partition_forest_f1_f2_f3_and_serialization():
    x, y = _data()
    forest = LeafPartitionForest(
        n_estimators=16, max_depth=3, min_samples_leaf=4, criterion="squared_error", random_state=0
    ).fit(x, y)
    mean = forest.predict(x, method="mean")
    mixture = forest.predict(x, method="distribution_median")
    median_of_medians = forest.predict(x, method="median_of_medians")
    for prediction in (mean, mixture, median_of_medians):
        assert prediction.shape == (len(x),)
        assert np.isfinite(prediction).all()
    assert not np.allclose(mixture, median_of_medians, rtol=0.0, atol=1e-12)
    restored = pickle.loads(pickle.dumps(forest))
    assert np.allclose(mixture, restored.predict(x, method="distribution_median"), rtol=0.0, atol=1e-12)
    assert forest.fit_meta_["response_methods"] == [
        "distribution_median",
        "mean",
        "median_of_medians",
    ]


def test_leaf_partition_forest_f4_changes_partition_not_response_rule():
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(size=(80, 2)), rng.normal(scale=5.0, size=(20, 2))]
    y = x[:, 0] + np.where(x[:, 1] > 0, 1.0, -1.0) + rng.normal(scale=0.5, size=len(x))
    squared = LeafPartitionForest(
        n_estimators=8, max_depth=3, min_samples_leaf=2, criterion="squared_error", random_state=7
    ).fit(x, y)
    absolute = LeafPartitionForest(
        n_estimators=8, max_depth=3, min_samples_leaf=2, criterion="absolute_error", random_state=7
    ).fit(x, y)
    assert squared.fit_meta_["criterion"] == "squared_error"
    assert absolute.fit_meta_["criterion"] == "absolute_error"
    squared_prediction = squared.predict(x, method="distribution_median")
    absolute_prediction = absolute.predict(x, method="distribution_median")
    assert not np.allclose(squared_prediction, absolute_prediction, rtol=0.0, atol=1e-12)


def test_honest_leaf_forest_keeps_structure_and_response_groups_disjoint():
    x, y = _data(90)
    groups = np.repeat(np.arange(18), 5)
    forest = HonestLeafForest(
        n_estimators=6, n_repeats=3, max_depth=3, min_samples_leaf=2, random_state=4
    ).fit(x, y, groups)
    prediction = forest.predict(x)
    assert prediction.shape == (len(x),)
    assert np.isfinite(prediction).all()
    assert forest.fit_meta_["group_isolation_checked"] is True
    for structure, response in zip(forest.structure_groups_, forest.response_groups_):
        assert not set(structure).intersection(response)
    assert forest.empty_leaf_fallback_count_ >= 0
    restored = pickle.loads(pickle.dumps(forest))
    assert np.allclose(prediction, restored.predict(x), rtol=0.0, atol=1e-12)


def test_lightgbm_linear_tree_capability_and_l1_rejection():
    with pytest.raises(ValueError):
        assert_lightgbm_linear_tree_supported("regression_l1")
    with pytest.raises(ValueError):
        assert_lightgbm_linear_tree_supported("mae")
    x, y = _data(100)
    with pytest.raises(ValueError):
        fit_lightgbm_linear_tree(x, y, objective="regression_l1")
    model = fit_lightgbm_linear_tree(
        x,
        y,
        n_estimators=6,
        num_leaves=3,
        min_child_samples=3,
        random_state=0,
        verbose=-1,
    )
    assert model.linear_tree_effective_["requested"] is True
    assert model.linear_tree_effective_["actual"] is True
    assert model.linear_tree_effective_["objective"] == "regression"
    prediction = model.predict(x)
    assert prediction.shape == (len(x),)
    assert np.isfinite(prediction).all()


def test_smooth_models_univariate_spline_and_tensor_terms():
    x, y = _data(120)
    linear_control = LinearControlRegressor(alpha=1.0).fit(x, y)
    additive = SmoothAdditiveRegressor(n_knots=3, degree=2, alpha=1.0).fit(x, y)
    pair = PairwiseTensorProductRegressor(pairs=[(0, 1), (2, 3)], n_knots=3, degree=2, alpha=1.0).fit(x, y)
    triple = TripleTensorProductRegressor(
        triples=[(0, 1, 2)], pairs=[(0, 1)], n_knots=3, degree=2, alpha=1.0
    ).fit(x, y)
    for model in (linear_control, additive, pair, triple):
        prediction = model.predict(x[:10])
        assert prediction.shape == (10,)
        assert np.isfinite(prediction).all()
        assert model.fit_meta_["selection_data"] == "training_rows_only"
    assert additive.fit_meta_["pairs"] == []
    assert pair.fit_meta_["pairs"] == [(0, 1), (2, 3)]
    assert triple.fit_meta_["triples"] == [(0, 1, 2)]
    assert additive.fit_meta_["spline_degree_univariate"] is True
    assert linear_control.fit_meta_["mechanism"] == "v4_linear_control"

    transformer = SplineTransformer(n_knots=4, degree=3, include_bias=False).fit(x)
    assert_spline_transformer_univariate(transformer, x.shape[1])
    with pytest.raises(AssertionError):
        assert_spline_transformer_univariate(transformer, x.shape[1] - 1)

    selected = select_pair_terms_internal_validation(
        x, y, candidates=[(0, 1), (2, 3)], n_splits=3, n_knots=3, degree=2, alpha=1.0
    )
    assert set(selected) == {(0, 1), (2, 3)}


def test_ebm_explicit_triple_arity_is_asserted_and_automatic_pairs_are_pairwise_only():
    x, y = _data(100)
    feature_frame = pd.DataFrame(x, columns=list("abcdef"))
    triple_model = fit_ebm_with_explicit_triple(
        feature_frame,
        y,
        [0, 1, 2],
        parameters={
            "max_rounds": 15,
            "max_bins": 16,
            "max_interaction_bins": 8,
            "outer_bags": 1,
            "inner_bags": 0,
            "random_state": 0,
            "n_jobs": 1,
        },
    )
    assert triple_model.explicit_triple_term_ == (0, 1, 2)
    assert_ebm_term_arity(triple_model, 3)

    from interpret.glassbox import ExplainableBoostingRegressor

    pairwise_model = ExplainableBoostingRegressor(
        interactions=2,
        max_rounds=10,
        max_bins=16,
        max_interaction_bins=8,
        outer_bags=1,
        inner_bags=0,
        random_state=0,
        n_jobs=1,
    ).fit(feature_frame, y)
    assert_ebm_has_no_terms_above(pairwise_model, max_arity=2)


def test_projection_pursuit_and_cross_fitted_residual_no_leakage():
    x, y = _data(120)
    fixed = FixedLinearCoordinateRegressor(alpha=1.0).fit(x, y)
    assert fixed.predict(x[:5]).shape == (5,)
    model = ProjectionPursuitRegressor(n_components=2, n_knots=3, degree=2, alpha=1.0).fit(x, y)
    prediction = model.predict(x[:8])
    assert prediction.shape == (8,)
    assert np.isfinite(prediction).all()
    assert model.fit_meta_["n_projection_scores"] == 2
    restored = pickle.loads(pickle.dumps(model))
    assert np.allclose(prediction, restored.predict(x[:8]), rtol=0.0, atol=1e-12)

    # A malicious base that returns a label only if it saw the exact row is
    # forced to its fallback in every inner fold.  The outer code must not
    # accidentally use true labels for the residual target.
    nested = CrossFittedResidualProjectionRegressor(Memorizer(), n_splits=5, random_state=0).fit(x, y)
    assert np.allclose(nested.inner_oof_, 0.0, rtol=0.0, atol=1e-12)
    assert not np.allclose(nested.inner_oof_, y, rtol=0.0, atol=1e-12)

    # Zero-correction fallback: a zero residual model may not double-count the
    # base prediction.
    fallback = CrossFittedResidualProjectionRegressor(
        Memorizer(), n_splits=5, random_state=0, projection_factory=ZeroResidualModel
    ).fit(x, y)
    base_prediction = np.asarray(fallback.base_model_.predict(x), dtype=float)
    assert np.allclose(fallback.predict(x), base_prediction, rtol=0.0, atol=1e-12)


def test_cross_target_chain_second_stage_receives_inner_oof_only():
    x, y = _data(120)
    chain = CrossTargetChainRegressor(
        Memorizer(), Memorizer(), direction="time_to_iron", n_splits=5, random_state=0
    ).fit(x, y, y * 0.5)
    assert chain.fit_meta_["true_source_target_used_as_feature"] is False
    assert np.allclose(chain.inner_source_oof_, 0.0, rtol=0.0, atol=1e-12)
    # The target-stage sentinel stores the exact design matrix.  Its last
    # column must be the inner OOF source prediction, not the true source label.
    assert np.allclose(chain.target_model_.x_[:, -1], chain.inner_source_oof_, rtol=0.0, atol=1e-12)
    prediction = chain.predict(x[:8])
    assert prediction.shape == (8,)
    assert np.isfinite(prediction).all()

    shared = SharedPrivateCrossTargetRegressor(n_components=2, n_knots=3, degree=2).fit(x, y, y * 0.5)
    outputs = shared.predict(x[:8])
    assert set(outputs) == {"tap_iron", "tap_time_len"}
    assert all(values.shape == (8,) for values in outputs.values())
    assert all(np.isfinite(values).all() for values in outputs.values())


def test_v4_run_schedule_has_40_unique_planned_slots():
    specs = v4_unit_specs()
    assert len(specs) == 40
    assert [spec["family"] for spec in specs].count("F") == 12
    assert [spec["family"] for spec in specs].count("S") == 12
    assert [spec["family"] for spec in specs].count("P") == 8
    assert [spec["family"] for spec in specs].count("J") == 8
    blocked = [spec for spec in specs if spec["blocked"]]
    assert len(blocked) == 2
    assert {spec["target"] for spec in blocked} == {"tap_iron", "tap_time_len"}
    assert all("B_star" in spec["mechanism"] for spec in blocked)
    keys = [(spec["family"], spec["target"], spec["mechanism"]) for spec in specs]
    assert len(keys) == len(set(keys))
    with pytest.raises(RuntimeError):
        SharedOnlyCrossTargetRegressor(n_components=2).predict([[1.0, 2.0]])
