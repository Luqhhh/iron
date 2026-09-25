from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES
from bf_tap_r2.v4_1_nested import build_or_load_nested_cache
from bf_tap_r2.v4_1_paired_terms import (
    _categorical_feature_types,
    _pair_terms,
    build_c_units,
    select_triples_from_training_importances,
)
from bf_tap_r2.v4_1_residuals import (
    ConstantMedianResidual,
    LocalWeightedMedianResidual,
    exact_l1_alpha,
    weighted_median,
)
from bf_tap_r2.v4_1_run import _private_output


def test_exact_l1_alpha_handles_zero_direction_and_zero_fallback() -> None:
    y = np.array([1.0, 2.0, 3.0])
    baseline = np.array([0.0, 0.0, 0.0])
    assert exact_l1_alpha(y, baseline, np.zeros(3)) == 0.0
    # A direction that improves every row gives alpha 1; a worsening direction
    # gives the zero boundary.
    assert exact_l1_alpha(y, baseline, np.array([1.0, 1.0, 1.0])) == 1.0
    assert exact_l1_alpha(y, baseline, np.array([-1.0, -1.0, -1.0])) == 0.0


def test_weighted_median_is_scale_free_and_deterministic() -> None:
    values = np.array([-2.0, 1.0, 4.0])
    weights = np.array([1.0, 2.0, 1.0])
    assert weighted_median(values, weights) == 1.0
    assert weighted_median(values * 100.0, weights) == 100.0


def test_constant_median_does_not_clip_negative_residual() -> None:
    model = ConstantMedianResidual().fit(None, np.array([-5.0, -1.0, -3.0]))
    prediction = model.predict(np.zeros((2, 2)))
    assert np.all(prediction == -3.0)


def test_e3_shrinks_to_zero_when_bank_is_below_support() -> None:
    state = np.arange(20, dtype=float).reshape(10, 2)
    residual = np.linspace(-1.0, 1.0, 10)
    model = LocalWeightedMedianResidual(min_bank=100).fit(state, residual)
    assert np.all(model.predict(state[:3]) == 0.0)


def test_e3_nearest_bank_row_returns_its_residual_without_clipping() -> None:
    state = np.array([[0.0], [1.0], [2.0]])
    residual = np.array([-2.0, 0.5, 3.0])
    model = LocalWeightedMedianResidual(min_bank=1).fit(state, residual)
    prediction = model.predict(np.array([[2.0]]))
    assert prediction.shape == (1,)
    assert prediction[0] > 0.0


def test_c_units_and_category_contract() -> None:
    units = build_c_units()
    assert len(units) == 10
    assert [unit["method_id"] for unit in units].count("C0") == 2
    feature_types = _categorical_feature_types()
    assert len(feature_types) == len(FEATURES) + 1
    assert feature_types[-1] == "nominal"
    assert feature_types[:-1] == ["continuous"] * len(FEATURES)


class _DummyEstimator:
    def __init__(self, terms, importances):
        self.term_features_ = terms
        self._importances = np.asarray(importances, dtype=float)

    def term_importances(self):
        return self._importances


class _DummyV34:
    def __init__(self, terms, importances):
        self.impl = type("Impl", (), {"estimator_": _DummyEstimator(terms, importances)})()


def test_triple_selection_uses_training_term_importances_only() -> None:
    model = _DummyV34(
        [(0,), (1,), (2,), (0, 1), (1, 2)],
        [0.1, 0.9, 0.2, 0.8, 0.1],
    )
    selected = select_triples_from_training_importances(model, max_triples=2)
    assert len(selected) == 2
    assert all(len(set(triple)) == 3 for triple in selected)
    assert all(max(triple) < len(FEATURES) + 1 for triple in selected)


def test_pair_terms_extract_and_validate_pair_set() -> None:
    model = _DummyV34([(0,), (0, 1), (1, 2)], [1.0, 1.0, 1.0])
    assert _pair_terms(model) == [(0, 1), (1, 2)]


def test_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    root = tmp_path
    output = root / "local/runs/example"
    _private_output(root, output)
    assert output.is_dir()
    with pytest.raises(FileExistsError):
        _private_output(root, output)


class _RecordingFactory:
    def __init__(self, n_features: int) -> None:
        self.calls: list[tuple[set[str], set[str]]] = []
        self.n_features = n_features

    def fit_predict(self, train: pd.DataFrame, query: pd.DataFrame) -> dict:
        train_ids, query_ids = set(train["sample_id"]), set(query["sample_id"])
        assert not train_ids & query_ids
        self.calls.append((train_ids, query_ids))
        n_query = len(query)
        state = np.column_stack([
            np.full(n_query, 0.1), np.full(n_query, 0.2),
            np.full(n_query, 0.3), np.full(n_query, 0.4),
        ])
        return {
            "predictions": {"tap_iron": np.full(n_query, 1.0), "tap_time_len": np.full(n_query, 2.0)},
            "disagreement": {"tap_iron": np.full(n_query, 0.3), "tap_time_len": np.full(n_query, 0.4)},
            "members": {},
            "meta": {},
        }


def test_nested_cache_keeps_fit_and_query_groups_disjoint(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    n = 60
    frame = pd.DataFrame({
        "sample_id": [f"R2S2_TRAIN_{i:012X}" for i in range(n)],
        "spout_no": rng.integers(1, 5, size=n),
        "fold": np.zeros(n, dtype=int),
    })
    for name in FEATURES:
        frame[name] = rng.normal(size=n)
    folds = np.arange(n, dtype=int) % 2
    cache_path = tmp_path / "cache.npz"
    factory = _RecordingFactory(len(FEATURES))
    cache = build_or_load_nested_cache(
        tmp_path, frame, folds, seed=42, outer_fold=0, factory=factory,
        cache_path=cache_path, n_meta=2, n_inner=2,
    )
    assert "final_pred_tap_iron" in cache
    assert "base_oof_T_pred_tap_iron" in cache
    assert len(factory.calls) >= 7
    for train_ids, query_ids in factory.calls:
        assert not train_ids & query_ids


# ---------------------------------------------------------------------------
# 2026-09-25 strong-increment implementation tests
# ---------------------------------------------------------------------------

from bf_tap_r2.v4_1_c234 import apply_slot_replacement  # noqa: E402
from bf_tap_r2.v4_1_models import (  # noqa: E402
    V41EBMRegressor,
    normalise_interactions,
    resolve_interactions,
    select_shared_edge_triples,
    structure_audit,
)


def test_slot_replacement_uses_original_units_and_exact_weight() -> None:
    baseline = np.array([100.0, 200.0])
    parent = np.array([90.0, 150.0])
    replacement = np.array([110.0, 130.0])
    result = apply_slot_replacement(baseline, parent, replacement, 0.25)
    assert np.allclose(result, baseline + 0.25 * (replacement - parent))
    clipped = apply_slot_replacement(np.array([1.0]), np.array([0.0]), np.array([-10.0]), 1.0,
                                     clip_nonnegative=True)
    assert clipped[0] == 0.0


def test_normalise_interactions_supports_pairs_and_triples() -> None:
    assert normalise_interactions(3) == 3
    assert normalise_interactions([(2, 1), (1, 2), (0, 1, 2)]) == [(1, 2), (0, 1, 2)]
    with pytest.raises(ValueError):
        normalise_interactions([(0, 0)])
    with pytest.raises(ValueError):
        normalise_interactions([(0, 1), ()])
    with pytest.raises(ValueError):
        normalise_interactions([])


def test_resolve_interactions_maps_names_and_indices_to_actual_frame() -> None:
    names = ["a", "b", "c", "d"]
    assert resolve_interactions([("b", "a"), ("d", "c", "a")], names) == [(0, 1), (0, 2, 3)]
    assert resolve_interactions([(3, 0), (1, 2, 0)], names) == [(0, 3), (0, 1, 2)]
    with pytest.raises(ValueError):
        resolve_interactions([("missing", "a")], names)


def test_shared_edge_triple_candidates_are_canonical_and_ranked() -> None:
    candidates = select_shared_edge_triples(
        [(0, 1), (1, 2), (2, 3)],
        [1.0, 5.0, 2.0],
    )
    assert candidates[:2] == [(1, 2, 3), (0, 1, 2)]
    assert select_shared_edge_triples([(0, 1), (2, 3)], [1.0, 1.0]) == []


def test_v41_ebm_preserves_pairs_and_adds_triples_native() -> None:
    pytest.importorskip("interpret")
    rng = np.random.default_rng(20260925)
    n = 320
    frame = pd.DataFrame({name: rng.normal(size=n) for name in FEATURES})
    frame["spout_no"] = rng.integers(1, 5, size=n)
    frame["sample_id"] = [f"R2S_TRAIN_{i:012X}" for i in range(n)]
    y = (
        5.0
        + 0.4 * frame["air_volume"].to_numpy()
        + 0.2 * frame["oxygen"].to_numpy()
        + 0.1 * frame["spout_no"].to_numpy()
        + rng.normal(scale=0.05, size=n)
    )
    trial = {
        "trial_id": "synthetic_v41",
        "kind": "ebm_training",
        "target": "tap_iron",
        "target_transform": "log1p",
        "feature_set": "raw",
        "parameters": {
            "max_rounds": 30,
            "early_stopping_rounds": 10,
            "max_bins": 32,
            "max_interaction_bins": 16,
            "learning_rate": 0.05,
            "outer_bags": 4,
            "inner_bags": 0,
            "random_state": 42,
            "n_jobs": 1,
            "max_leaves": 2,
            "min_samples_leaf": 20,
            "objective": "rmse",
            "interactions": [(0, 1), (0, 2, 3)],
        },
        "protocol": {"inner_splits": 5, "bag_seed": 42},
    }
    model = V41EBMRegressor(trial).fit(frame, y)
    predicted = model.predict(frame)
    assert predicted.shape == (n,)
    audit = structure_audit(model, expected_pairs=[(0, 1)], expected_triples=[(0, 2, 3)])
    assert audit["pair_set_preserved"] is True
    assert audit["triple_set_preserved"] is True
    assert audit["n_triples"] == 1
    # Row order, single-row and block predictions must agree.
    assert np.allclose(model.predict(frame.iloc[::-1])[::-1], predicted)
    assert np.allclose(model.predict(frame.iloc[[7]]), predicted[[7]])
    block = np.concatenate([model.predict(frame.iloc[i:i + 17]) for i in range(0, n, 17)])
    assert np.allclose(block, predicted)
    # The requested interaction parameter must be visible on the estimator.
    actual = model.estimator_.get_params(deep=False)["interactions"]
    assert [[int(v) for v in term] for term in actual] == [[0, 1], [0, 2, 3]]
    # Serialization readback must preserve predictions.
    import pickle
    restored = pickle.loads(pickle.dumps(model))
    assert np.allclose(restored.predict(frame), predicted)
