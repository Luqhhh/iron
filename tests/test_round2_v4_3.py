"""V4.3 strong-base time-residual tests.

The tests here are deliberately light: they pin the frozen configuration, the
composition arithmetic, the residual learners, the export clipping rule, and the
append-only private-output guard.  They never fit a parent member, so the suite
stays fast; the real nested refits are exercised by the private runner.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES
from bf_tap_r2.v4_3_parent import (
    compose_parent,
    member_specs,
    parent_time_spec,
    recorded_target_composition,
)
from bf_tap_r2.v4_3_residual import (
    CONFIG_VERSION,
    FAMILIES,
    TARGET,
    ShrinkBinnedResidualCorrector,
    TreeResidualCorrector,
    candidate_specs,
    evaluate_candidate_corrections,
    load_v43_config,
    make_corrector,
    residual_feature_frame,
)
from bf_tap_r2.v4_3_run import PRIVATE_ROOT, _private_output

ROOT = Path(__file__).resolve().parents[1]


def _frame(n: int = 120, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(index=range(n))
    frame["sample_id"] = [f"S{i:06d}" for i in range(n)]
    frame["spout_no"] = np.asarray([(i % 4) + 1 for i in range(n)], dtype=int)
    for j, name in enumerate(FEATURES):
        frame[name] = 1.0 + rng.normal(0.0, 0.1, n) + 0.01 * j
    frame["tap_iron"] = 100.0 + 2.0 * rng.normal(size=n)
    frame["tap_time_len"] = 30.0 + 1.0 * rng.normal(size=n)
    return frame


def _flat_config() -> dict:
    """Minimal in-memory config mirroring the frozen composition arithmetic."""
    return {
        "version": CONFIG_VERSION,
        "parent": {
            "target_in_scope": TARGET,
            "iron_side": "unchanged_and_out_of_scope",
            "time": {
                "weights": [0.5, 0.3, 0.2],
                "experts": ["e1", "e2"],
                "a_dev_weights": [0.4, 0.5, 0.1],
                "a_dev_experts": ["a1", "a2"],
                "l1_pool": ["p1", "p2"],
            },
        },
        "candidate_budget": {"per_target": 12},
        "candidates": {
            "shrink_bins": [{"name": "SB06_M50_A050", "bins": 6, "shrinkage": 50.0,
                             "statistic": "mean", "alpha": 0.50}],
            "lightgbm_residual": [{"name": "LGB_L1_L7_A025", "objective": "regression_l1",
                                   "num_leaves": 7, "alpha": 0.25}],
            "histgb_residual": [{"name": "HGB_L2_L7_A025", "loss": "squared_error",
                                 "max_leaf_nodes": 7, "alpha": 0.25}],
        },
    }


def test_config_is_frozen() -> None:
    config = load_v43_config(ROOT)
    assert config["version"] == CONFIG_VERSION
    assert config["parent"]["target_in_scope"] == TARGET
    assert config["parent"]["iron_side"] == "unchanged_and_out_of_scope"
    assert int(config["candidate_budget"]["per_target"]) == 12
    assert config["protocol"]["split_seeds"] == [42, 3407]
    assert config["protocol"]["coarse_folds"] == [0, 1]
    assert config["protocol"]["complete_folds"] == [0, 1, 2, 3, 4]
    assert int(config["protocol"]["inner_folds"]) == 3
    assert float(config["continuation_gate"]["mean_package_delta_single_min"]) == 0.005
    assert float(config["continuation_gate"]["local_working_gate"]) == 96.25
    assert config["outputs"]["agent_uploads"] == 0
    assert config["outputs"]["automatic_package"] is False


def test_candidate_pool_is_the_frozen_twelve() -> None:
    config = load_v43_config(ROOT)
    specs = candidate_specs(config)
    assert len(specs) == 12
    assert {spec["family"] for spec in specs} == set(FAMILIES)
    assert len({spec["candidate_id"] for spec in specs}) == 12
    assert all(spec["candidate_id"].startswith("v43-tap_time_len-") for spec in specs)
    spec = next(spec for spec in specs if spec["family"] == "lightgbm_residual")
    assert spec["defaults"]["reg_lambda"] == 30.0


def test_parent_time_spec_matches_recorded_composition() -> None:
    config = load_v43_config(ROOT)
    spec = parent_time_spec(ROOT, config)
    weights, experts = recorded_target_composition(ROOT, config, TARGET)
    assert spec["package_weights"] == weights
    assert spec["package_experts"] == experts
    assert len(experts) == 2
    assert len(spec["l1_pool"]) == 5
    members = member_specs(config)
    assert {member.group for member in members} == {"l1_pool", "a_expert", "parent_expert"}
    l1_weight = [member.weight for member in members if member.group == "l1_pool"][0]
    non_pool = sum(member.weight for member in members if member.group != "l1_pool")
    assert abs(l1_weight + non_pool - 1.0) < 1e-12


def test_compose_parent_reproduces_hand_blend() -> None:
    config = _flat_config()
    n = 5
    pool = {"p1": np.full(n, 2.0), "p2": np.full(n, 4.0)}
    l1 = 0.25 * 2.0 + 0.75 * 4.0
    a_dev = 0.4 * l1 + 0.5 * 10.0 + 0.1 * 20.0
    expected = 0.5 * a_dev + 0.3 * 100.0 + 0.2 * 200.0
    parent = compose_parent(
        config,
        l1_pool_predictions=pool,
        l1_weights=[0.25, 0.75],
        a_expert_predictions=[np.full(n, 10.0), np.full(n, 20.0)],
        parent_expert_predictions=[np.full(n, 100.0), np.full(n, 200.0)],
    )
    assert np.allclose(parent, expected)
    with pytest.raises(ValueError):
        compose_parent(
            config,
            l1_pool_predictions=pool,
            l1_weights=[0.25, 0.5],
            a_expert_predictions=[np.full(n, 10.0), np.full(n, 20.0)],
            parent_expert_predictions=[np.full(n, 100.0), np.full(n, 200.0)],
        )


def test_residual_feature_frame_layout() -> None:
    frame = _frame(40)
    base = frame["tap_time_len"].to_numpy(dtype=float) - 1.0
    matrix = residual_feature_frame(frame, base)
    assert list(matrix.columns) == ["base_prediction", *FEATURES, "spout_no"]
    assert np.allclose(matrix["base_prediction"].to_numpy(dtype=float), base)
    with pytest.raises(ValueError):
        residual_feature_frame(frame, np.full(len(frame) - 1, 1.0))


def test_shrink_corrector_recovers_constant_residual() -> None:
    frame = _frame(200)
    base = frame["tap_time_len"].to_numpy(dtype=float) - 3.0
    corrector = ShrinkBinnedResidualCorrector(
        bins=6, shrinkage=10.0, statistic="mean", alpha=0.5, clip_quantile=1.0
    ).fit(frame, base, frame["tap_time_len"].to_numpy(dtype=float))
    correction = corrector.predict_correction(frame, base)
    assert np.allclose(correction, 1.5, atol=1e-9)
    assert corrector.fit_meta_["residual_source"] == "nested_cross_fitted_parent_only"


def test_tree_correctors_are_finite_and_shrunk() -> None:
    frame = _frame(240)
    y = frame["tap_time_len"].to_numpy(dtype=float)
    base = y - 2.0
    for spec in (
        {"family": "lightgbm_residual", "objective": "regression_l1", "num_leaves": 7,
         "alpha": 0.25, "clip_quantile": 0.95,
         "defaults": {"n_estimators": 20, "learning_rate": 0.05, "min_child_samples": 40,
                      "reg_lambda": 10.0, "max_depth": -1, "verbosity": -1, "n_jobs": 1,
                      "random_state": 1}},
        {"family": "histgb_residual", "loss": "squared_error", "max_leaf_nodes": 7,
         "alpha": 0.5, "clip_quantile": 0.95,
         "defaults": {"max_iter": 30, "learning_rate": 0.05, "min_samples_leaf": 40,
                      "l2_regularization": 10.0, "random_state": 1}},
    ):
        corrector = make_corrector(spec).fit(frame, base, y)
        correction = corrector.predict_correction(frame, base)
        assert correction.shape == y.shape
        assert np.isfinite(correction).all()
        assert np.abs(correction).max() <= corrector.clip_ + 1e-12
        assert np.abs(correction).max() <= spec["alpha"] * 2.0 + 1e-9


def test_export_clipping_and_shape_guards() -> None:
    y = np.asarray([1.0, 2.0, 3.0])
    parent = np.asarray([1.0, 2.0, 3.0])
    clipped = evaluate_candidate_corrections(None, y, parent, np.asarray([-5.0, 0.5, -0.25]))
    assert np.allclose(clipped, [0.0, 2.5, 2.75])
    raw = evaluate_candidate_corrections(None, y, parent, np.asarray([-5.0, 0.5, -0.25]),
                                         clip_at_zero=False)
    assert np.allclose(raw, [-4.0, 2.5, 2.75])
    with pytest.raises(ValueError):
        evaluate_candidate_corrections(None, y, parent, np.asarray([0.0, 0.0]))


def test_private_output_guard() -> None:
    inside = ROOT / PRIVATE_ROOT / "coarse-r1"
    assert _private_output(ROOT, inside) == inside.resolve()
    with pytest.raises(ValueError):
        _private_output(ROOT, ROOT / "local/runs/round2-v4.3-other")


def test_candidate_specs_reject_out_of_scope_target() -> None:
    with pytest.raises(ValueError):
        candidate_specs(load_v43_config(ROOT), target="tap_iron")


def test_recorded_parent_identity_when_private_caches_exist() -> None:
    """Recompute the frozen parent development score from the private caches.

    The private V3.4/V3.6 development caches are absent on CI and on any machine
    that did not run the V3.4/V3.6 phases, so the check is skipped there.  When
    the caches exist this pins the exact numbers the V4.3 residual targets are
    measured against.
    """
    from bf_tap_r2.v3_run import load_training_frame
    from bf_tap_r2.v4_3_parent import PARENT_DEVELOPMENT_SCORE, parent_development_score

    cache = ROOT / "local/runs/round2-v3.4-ebm-and-constrained-composition/l1-oof-r1/seed-42"
    if not cache.exists():
        pytest.skip("private V3.4 development OOF cache is not present")
    train = load_training_frame(ROOT)
    result = parent_development_score(ROOT, train, load_v43_config(ROOT))
    assert result["mean_package_score"] == pytest.approx(PARENT_DEVELOPMENT_SCORE, abs=1e-9)
    assert result["per_seed_package_score"]["42"] == pytest.approx(96.20130931837855, abs=1e-9)
    assert result["per_seed_package_score"]["3407"] == pytest.approx(96.20621581960641, abs=1e-9)
    assert result["per_seed_target_wmape"]["tap_time_len"]["42"] == pytest.approx(
        0.03847465422923957, abs=1e-12
    )
