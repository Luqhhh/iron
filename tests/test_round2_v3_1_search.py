from pathlib import Path

import numpy as np
import pytest

from bf_tap_r2.v3_1_fusion import (
    V31_ALPHAS,
    greedy_forward_select_lp,
    lp_simplex_weights,
    pooled_wmape,
    simple_mix,
    triage_delta_v31,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v31_triage_is_monotonic_at_all_bands():
    assert triage_delta_v31(-0.01) == "not_independent"
    assert triage_delta_v31(0.0199) == "not_independent"
    assert triage_delta_v31(0.02) == "fallback"
    assert triage_delta_v31(0.0499) == "fallback"
    assert triage_delta_v31(0.05) == "candidate_pool"
    assert triage_delta_v31(0.1499) == "candidate_pool"
    assert triage_delta_v31(0.15) == "primary"
    assert triage_delta_v31(0.30) == "primary"


def test_v31_pooled_wmape_matches_manual_abs_error_ratio():
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([11.0, 18.0, 33.0])
    assert pooled_wmape(y, p) == pytest.approx((1 + 2 + 3) / 60.0)


def test_v31_simple_mix_uses_original_unit_convex_combination():
    ref = np.array([100.0, 100.0])
    cand = np.array([80.0, 120.0])
    assert np.allclose(simple_mix(ref, cand, 0.25), [95.0, 105.0])
    for alpha in V31_ALPHAS:
        np.testing.assert_allclose(simple_mix(ref, cand, alpha), (1 - alpha) * ref + alpha * cand)


def test_v31_lp_recovers_cancelling_experts_and_checks_objective():
    rng = np.random.default_rng(31)
    y = 100.0 + rng.normal(0, 5, size=80)
    error = rng.normal(0, 2, size=80)
    p = np.stack([y + error, y - error, rng.normal(100, 5, size=80)], axis=1)
    result = lp_simplex_weights({"42": y, "3407": y + 0.0 * error}, {"42": p, "3407": p})
    weights = result["weights"]
    assert weights.shape == (3,)
    assert weights.min() >= -1e-12
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert weights[0] == pytest.approx(0.5, abs=1e-6)
    assert weights[1] == pytest.approx(0.5, abs=1e-6)
    assert result["objective"] < 1e-9
    recomputed = np.mean([
        pooled_wmape(np.array([y, y]), np.stack([p @ weights, p @ weights]))
    ]) if False else None
    # explicit original-unit recomputation for both seed blocks
    rec = (pooled_wmape(y, p @ weights) + pooled_wmape(y, p @ weights)) / 2
    assert rec == pytest.approx(result["objective"], abs=1e-10)


def test_v31_lp_rejects_bad_shapes_and_returns_feasible_weights():
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        lp_simplex_weights({"42": y}, {"42": np.ones((4, 2))})
    p = np.stack([y + 0.1, y - 0.1, y + 0.2], axis=1)
    result = lp_simplex_weights({"42": y}, {"42": p})
    assert result["objective"] == pytest.approx(pooled_wmape(y, p @ result["weights"]), abs=1e-10)
    assert result["weights"].sum() == pytest.approx(1.0)


def test_v31_greedy_forward_select_uses_lp_and_improves():
    rng = np.random.default_rng(32)
    y = {str(s): 100.0 + rng.normal(0, 5, size=60) for s in (42, 3407)}
    err = {str(s): rng.normal(0, 2, size=60) for s in (42, 3407)}
    p = {s: np.stack([y[s] + err[s], y[s] - err[s], rng.normal(100, 5, size=60)], axis=1) for s in y}
    fit = greedy_forward_select_lp(y, p, ["p1", "p2", "noise"], max_members=3)
    assert set(fit["members"]) == {"p1", "p2"}
    assert fit["objective"] < 1e-9
    assert abs(fit["weights"]["p1"] - 0.5) < 1e-6
    assert abs(fit["weights"]["p2"] - 0.5) < 1e-6


def test_v31_s1_budget_contract():
    from bf_tap_r2.v3_1_search import load_config, validate_s1_budget
    spec = load_config(ROOT)
    validate_s1_budget(spec["s1_budget"])
    bad = dict(spec["s1_budget"])
    bad["total"] = 319
    with pytest.raises(ValueError, match="320"):
        validate_s1_budget(bad)


def test_v31_trial_identity_is_canonical_and_rejects_mismatch():
    from bf_tap_r2.v3_1_search import canonical_trial_hash, identity_matches, trial_identity
    trial_a = {"family": "catboost", "target": "tap_iron", "parameters": {"depth": 4, "learning_rate": 0.03}}
    trial_b = {"parameters": {"learning_rate": 0.03, "depth": 4}, "target": "tap_iron", "family": "catboost"}
    assert canonical_trial_hash(trial_a) == canonical_trial_hash(trial_b)
    desired = trial_identity(trial_a, batch_id="v31-s1", data_hash="data", fold_hash="fold",
                             fold_ids=[0, 1], model_seed=42, stage="coarse", code_version="abc")
    stored = dict(desired)
    assert identity_matches(stored, desired)
    stored["batch_id"] = "other"
    assert not identity_matches(stored, desired)
    with pytest.raises(ValueError):
        trial_identity({**trial_a, "target": "bad"}, batch_id="v31-s1", data_hash="data",
                       fold_hash="fold", fold_ids=[0, 1], model_seed=42, stage="coarse",
                       code_version="abc")
