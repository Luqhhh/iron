from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import TARGETS
from bf_tap_r2.v3_local_search import (
    FAMILY_KEYS,
    TOTAL_BUDGET,
    TrialRegressor,
    apply_target_transform,
    build_run_manifest,
    expression_frame,
    fit_target_transform,
    inverse_target_transform,
    load_spec,
    nested_fusion,
    sample_trials,
    triage_delta,
    validate_budget,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v3_schedule_is_deterministic_balanced_and_legal():
    spec = load_spec(ROOT)
    first = sample_trials(spec)
    second = sample_trials(spec)
    assert first == second
    assert len(first) == TOTAL_BUDGET
    assert len({t["trial_id"] for t in first}) == TOTAL_BUDGET
    counts = pd.Series([t["family"] for t in first]).value_counts().to_dict()
    assert counts == {k: int(spec["search_budget"][k]) for k in FAMILY_KEYS}
    for target in TARGETS:
        assert sum(t["target"] == target for t in first) == spec["search_budget"]["per_target"]
    for trial in first:
        assert trial["feature_set"] in {"raw", "four"}
        assert trial["target_transform"] in {"identity", "mean", "mean_std", "log1p"}
        params = trial["parameters"]
        if trial["family"] == "catboost":
            grow = params["grow_policy"]
            if grow == "SymmetricTree":
                assert 3 <= params["depth"] <= 8
                assert "min_data_in_leaf" not in params
            else:
                assert "min_data_in_leaf" in params
                assert params.get("boosting_type") != "Ordered"
            if params["bootstrap_type"] in {"MVS", "Bernoulli"}:
                assert "subsample" in params
                assert 0.0 <= params["subsample"] <= 1.0
            elif params["bootstrap_type"] == "Bayesian":
                assert "bagging_temperature" in params
            else:
                assert params["bootstrap_type"] == "No"


def test_v3_budget_contract_rejects_wrong_total_or_family_count():
    with pytest.raises(ValueError, match="total"):
        validate_budget({"total": 399, "catboost": 239, "lightgbm": 40, "xgboost": 40,
                         "mlp": 20, "kernel": 20, "expression": 40, "per_target": 200})
    bad = {"total": 400, "catboost": 240, "lightgbm": 40, "xgboost": 40,
           "mlp": 20, "kernel": 20, "expression": 40, "per_target": 200}
    bad["expression"] = 39
    with pytest.raises(ValueError):
        validate_budget(bad)


@pytest.mark.parametrize("kind", ["identity", "mean", "mean_std", "log1p"])
def test_v3_target_transform_roundtrip_is_fold_only(kind):
    y = np.array([100.0, 101.0, 99.0, 102.0, 98.0])
    z, state = fit_target_transform(y, kind)
    assert z.shape == y.shape
    np.testing.assert_allclose(inverse_target_transform(z, kind, state), y, rtol=1e-13, atol=1e-12)
    if kind in {"mean", "mean_std", "log1p"}:
        with pytest.raises(ValueError):
            fit_target_transform(y - 200.0, kind)
    if kind == "mean":
        np.testing.assert_allclose(state["mean"], y.mean())
    if kind == "mean_std":
        np.testing.assert_allclose(state["std"], y.std())
    extrapolated = np.array([200.0, 201.0, 199.0, 202.0, 198.0])
    np.testing.assert_allclose(inverse_target_transform(apply_target_transform(extrapolated, kind, state), kind, state),
                               extrapolated, rtol=1e-12, atol=1e-10)


def test_v3_feature_expression_keeps_declared_columns_and_rejects_zero_denominators():
    frame = pd.DataFrame({
        "air_volume": [1.0, 2.0],
        "cold_air_press": [1.0, 1.0],
        "hot_air_press": [1.0, 1.0],
        "oxygen": [30.0, 40.0],
        "hot_air_temp": [1100.0, 1150.0],
        "coal_rate": [1.0, 1.0],
        "humidity": [1.0, 1.0],
        "gas_rate": [1.0, 1.0],
        "furnace_top_press": [1.0, 1.0],
        "upper_press_diff": [1.0, 1.0],
        "lower_press_diff": [1.0, 1.0],
        "total_press_diff": [10.0, 10.0],
        "air_press_ratio": [1.0, 1.0],
        "furnace_top_temp_avg": [1.0, 1.0],
        "air_speed": [1.0, 1.0],
        "furnace_throat_temp": [1000.0, 1000.0],
        "pig": [1.0, 1.0],
        "all_quality": [1.0, 1.0],
        "consumption": [1.0, 1.0],
        "fuel_rate": [1.0, 1.0],
        "coke_rate": [1.0, 1.0],
        "spout_no": [1, 2],
    })
    raw = expression_frame(frame, "raw")
    assert list(raw.columns) == [*frame.columns[:-1], "spout_no"]
    four = expression_frame(frame, "four")
    assert list(four.columns) == [*raw.columns, "oxygen_per_air_volume", "pressure_per_air_volume",
                                  "thermal_difference", "upper_pressure_fraction"]
    broken = frame.copy()
    broken.loc[0, "total_press_diff"] = 0.0
    with pytest.raises(ValueError, match="positive denominators"):
        expression_frame(broken, "four")


def test_v3_nested_fusion_recovers_cancelling_experts_without_touching_heldout_labels():
    rng = np.random.default_rng(17)
    labels = {s: 100.0 + rng.normal(0, 5, size=120) for s in ("42", "3407")}
    library = {}
    for seed, y in labels.items():
        error = rng.normal(0, 2, size=len(y))
        library[f"pos:{seed}"] = {seed: y + error}  # placeholder, replaced below
    # Rebuild with two seeds per candidate because nested_fusion needs all seeds.
    error_by_seed = {s: rng.normal(0, 2, size=120) for s in labels}
    library = {}
    for candidate in ("p1", "p2", "noise"):
        library[candidate] = {}
        for seed, y in labels.items():
            if candidate == "p1":
                library[candidate][seed] = y + error_by_seed[seed]
            elif candidate == "p2":
                library[candidate][seed] = y - error_by_seed[seed]
            else:
                library[candidate][seed] = rng.normal(100, 5, size=len(y))
    result = nested_fusion(library, "tap_iron", labels, max_pool=3, max_members=3)
    assert result["pool_size"] == 3
    assert result["mean_heldout_wmape"] < 1e-8
    for record in result["fold"].values():
        assert set(record["selected"]) == {"p1", "p2"}
        assert abs(record["weights"]["p1"] - 0.5) < 1e-6
        assert abs(record["weights"]["p2"] - 0.5) < 1e-6


def test_v3_triage_thresholds_are_explicit():
    thresholds = load_spec(ROOT)["submission_thresholds"]
    assert triage_delta(0.0, thresholds) == "not_independent"
    assert triage_delta(0.03, thresholds) == "fallback"
    assert triage_delta(0.05, thresholds) == "candidate_pool"
    assert triage_delta(0.15, thresholds) == "primary"
    assert triage_delta(0.20, thresholds) == "primary"


def test_v3_run_manifest_has_no_labels_or_predictions():
    spec = load_spec(ROOT)
    manifest = build_run_manifest(spec, ROOT)
    assert manifest["trials"] == 400
    assert manifest["agent_uploads"] == 0
    assert "WMAPE" not in str(manifest)
    assert "heldout" not in str(manifest)
