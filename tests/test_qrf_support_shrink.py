import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.optimization.qrf_support_shrink import (
    CANDIDATE,
    REFERENCE,
    REPLAY_CONTROL,
    LambdaBudget,
    acceptance,
    apply_shrink,
    build_direction_bank,
    build_prediction_directions,
    fit_lambda,
    fixed_cutoff_median,
    fixed_cutoff_medians,
    normalized_support,
    replay_v21,
    select_outer_oof,
    serialize_six_decimals,
    verify_h2_bank,
    verify_lambda_certificate,
    verify_prediction_directions,
    verify_median_certificate,
)


TZ = "Asia/Shanghai"


def history(n=120, start="2024-01-01", spout=None):
    reference = pd.date_range(start, periods=n, freq="12h", tz=TZ)
    if spout is None:
        spout = ["1" if i % 2 == 0 else "2" for i in range(n)]
    return pd.DataFrame({
        "sample_id": [f"h{i:03d}" for i in range(n)],
        "spout_no": spout,
        "reference_time": reference,
        "label_available_at": reference + pd.Timedelta(hours=2),
        "tap_time_len": np.arange(n, dtype=float) + 80,
    })


def queries(n=6, start="2024-05-01 01:00"):
    reference = pd.date_range(start, periods=n, freq="3h", tz=TZ)
    return pd.DataFrame({
        "sample_id": [f"q{i:03d}" for i in range(n)],
        "spout_no": ["1" if i % 2 == 0 else "2" for i in range(n)],
        "reference_time": reference,
        "label_available_at": reference + pd.Timedelta(hours=3),
    })


def bank(n=120):
    q = queries(n, "2024-05-01 01:00")
    h = history(180, "2024-01-01")
    medians = fixed_cutoff_medians(h, pd.Timestamp("2024-04-01", tz=TZ))
    return build_direction_bank(
        q,
        np.linspace(100, 130, n),
        np.linspace(1, 297, n),
        297,
        medians,
        model_cutoff=pd.Timestamp("2024-04-01", tz=TZ),
        training_identity_sha256="training",
        model_bundle_sha256="bundle",
    )


def labels_for(value):
    return pd.DataFrame({
        "sample_id": value.sample_id,
        "spout_no": value.spout_no,
        "reference_time": value.reference_time,
        "label_available_at": value.label_available_at,
        "tap_time_len": value.Q + 0.5 * value.d,
    })


def test_support_denominator_is_model_training_count_not_batch_size():
    full_u, _ = normalized_support([10.0, 50.0, 100.0], 100)
    subset_u, _ = normalized_support([50.0], 100)
    assert subset_u[0] == full_u[1] == 0.5


def test_equal_and_concentrated_support_endpoints():
    u, a = normalized_support([100.0, 1.0], 100)
    assert np.array_equal(u, [1.0, 0.01])
    assert np.array_equal(a, [0.0, 0.99])


def test_support_only_snaps_numerical_roundoff_and_rejects_real_overflow():
    u, _ = normalized_support([1 - 1e-10, 100 + 1e-10], 100)
    assert np.array_equal(u, [0.01, 1.0])
    with pytest.raises(ContractError, match="outside"):
        normalized_support([0.9], 100)
    with pytest.raises(ContractError, match="outside"):
        normalized_support([100.1], 100)


def test_fixed_cutoff_median_is_anchored_and_uses_pandas_even_median():
    h = pd.DataFrame({
        "sample_id": ["a", "b", "future-query-window"],
        "spout_no": ["1", "1", "1"],
        "reference_time": pd.to_datetime([
            "2024-03-01 00:00+08:00", "2024-03-02 00:00+08:00", "2024-04-20 00:00+08:00"
        ], utc=True).tz_convert(TZ),
        "label_available_at": pd.to_datetime([
            "2024-03-01 01:00+08:00", "2024-03-02 01:00+08:00", "2024-04-20 01:00+08:00"
        ], utc=True).tz_convert(TZ),
        "tap_time_len": [100.0, 120.0, 10000.0],
    })
    cert = fixed_cutoff_median(h, pd.Timestamp("2024-04-01", tz=TZ), "1")
    assert cert["median"] == 110.0 and cert["source"] == "recent60" and cert["selected_rows"] == 2
    assert verify_median_certificate(h, cert)


def test_certified_history_available_at_alias_is_accepted():
    value = history(4).rename(columns={"label_available_at": "available_at"})
    cert = fixed_cutoff_median(value, pd.Timestamp("2024-05-01", tz=TZ), "1")
    assert cert["selected_rows"] == 2


def test_empty_recent_window_falls_back_and_missing_spout_does_not_map():
    h = history(2, "2024-01-01", ["1", "1"])
    cutoff = pd.Timestamp("2024-04-01", tz=TZ)
    one = fixed_cutoff_median(h, cutoff, "1")
    two = fixed_cutoff_median(h, cutoff, "2")
    assert one["source"] == "all_legal_same_spout_fallback" and one["fallback"]
    assert two["median"] is None and two["source"] == "no_legal_same_spout_history"
    q = queries(1)
    q["spout_no"] = "unknown"
    b = build_direction_bank(q, [100.0], [1.0], 10, {"1": one, "2": two},
        model_cutoff=cutoff, training_identity_sha256="t", model_bundle_sha256="m")
    assert np.isnan(b.loc[0, "M"]) and b.loc[0, "d"] == 0


def test_genuine_h2_bank_rejects_h1_relabeling():
    value = bank(4)
    assert verify_h2_bank(value, {4: 297})
    forged = value.copy()
    forged["reference_time"] = forged.reference_time - pd.DateOffset(months=1)
    forged["label_available_at"] = forged.reference_time + pd.Timedelta(hours=2)
    with pytest.raises(ContractError, match="genuine calendar H2"):
        verify_h2_bank(forged)


def test_late_and_current_outer_labels_are_excluded_row_by_row():
    value = bank(120)
    labels = labels_for(value)
    labels.loc[0, "label_available_at"] = pd.Timestamp("2024-06-02", tz=TZ)
    selected, status = select_outer_oof(value, labels, pd.Timestamp("2024-06-01", tz=TZ), "1")
    assert value.loc[0, "sample_id"] not in set(selected.sample_id)
    assert (selected.reference_time < pd.Timestamp("2024-06-01", tz=TZ)).all()
    assert (selected.model_cutoff < pd.Timestamp("2024-06-01", tz=TZ)).all()
    assert not status["fit_eligible"]  # 60 rows for this spout, minus the late row.


def test_insufficient_per_spout_oof_returns_exact_zero_without_fit(tmp_path):
    value = bank(120)
    labels = labels_for(value)
    selected, _ = select_outer_oof(value, labels, pd.Timestamp("2024-06-01", tz=TZ), "1")
    result = fit_lambda(selected)
    assert result["lambda"] == 0 and not result["fit_performed"]
    budget = LambdaBudget(tmp_path)
    saved = budget.derive(6, "1", selected)
    assert saved["lambda"] == 0
    assert budget.counts() == {"parameter_slots_completed": 1, "fit_attempts": 0, "zero_fallbacks": 1}


def test_lad_recovers_smallest_weighted_median_and_zero_direction():
    selected = pd.DataFrame({
        "sample_id": [f"x{i}" for i in range(100)],
        "Q": np.zeros(100),
        "d": np.ones(100),
        "tap_time_len": np.r_[np.full(50, 0.2), np.full(50, 0.8)],
    })
    assert fit_lambda(selected)["lambda"] == 0.2
    selected["d"] = 0.0
    zero = fit_lambda(selected)
    assert zero["lambda"] == 0.0 and not zero["fit_performed"] and zero["fallback"] == "ZERO_DIRECTION"


def test_saved_lambda_certificate_verifies_under_zero_fit_guard():
    from bf_tap.optimization.v13_common import zero_fit
    selected = pd.DataFrame({
        "sample_id": [f"x{i}" for i in range(100)], "Q": 1.0, "d": 1.0,
        "tap_time_len": [1.25] * 60 + [1.75] * 40,
    })
    certificate = fit_lambda(selected)
    with zero_fit() as counts:
        assert verify_lambda_certificate(selected, certificate)
    assert counts["attempted_calibration_fits"] == 0


def test_lambda_zero_identity_lambda_one_convexity_and_spout_isolation():
    value = bank(6)
    v10 = pd.DataFrame({
        "sample_id": value.sample_id,
        "pred_tap_iron": np.arange(6, dtype=float) + 200,
        "pred_tap_time_len": value.Q,
    })
    identity = apply_shrink(v10, value, {"1": 0.0, "2": 0.0})
    assert identity.equals(v10)
    changed = apply_shrink(v10, value, {"1": 1.0, "2": 0.0})
    s1 = value.spout_no == "1"
    s2 = ~s1
    expected_s1 = value.loc[s1, "Q"] + value.loc[s1, "d"]
    assert np.array_equal(changed.loc[s1, "pred_tap_time_len"], expected_s1)
    assert np.array_equal(changed.loc[s2, "pred_tap_time_len"], v10.loc[s2, "pred_tap_time_len"])
    lo = np.minimum(value.loc[s1, "Q"], value.loc[s1, "M"])
    hi = np.maximum(value.loc[s1, "Q"], value.loc[s1, "M"])
    assert ((changed.loc[s1, "pred_tap_time_len"] >= lo) & (changed.loc[s1, "pred_tap_time_len"] <= hi)).all()
    assert np.array_equal(changed.pred_tap_iron, v10.pred_tap_iron)


def test_stage_prediction_directions_need_no_fake_label_metadata():
    q = queries(4).drop(columns="label_available_at")
    h = history(180)
    medians = fixed_cutoff_medians(h, pd.Timestamp("2024-04-01", tz=TZ))
    directions = build_prediction_directions(
        q, [100.0, 101.0, 102.0, 103.0], [10.0, 20.0, 30.0, 40.0], 100, medians,
        model_cutoff=pd.Timestamp("2024-04-01", tz=TZ),
        training_identity_sha256="training", model_bundle_sha256="bundle",
    )
    assert "label_available_at" not in directions and verify_prediction_directions(directions)
    v10 = pd.DataFrame({
        "sample_id": q.sample_id, "pred_tap_iron": 200.0,
        "pred_tap_time_len": directions.Q,
    })
    assert len(apply_shrink(v10, directions, {"1": 0.5, "2": 0.5})) == len(q)


def test_v21_replay_uses_v10_six_decimal_input_then_serializes_again():
    v10 = pd.DataFrame({"sample_id": ["q"], "pred_tap_iron": [200.123456], "pred_tap_time_len": [100.123457]})
    q = pd.DataFrame({"sample_id": ["q"], "spout_no": ["1"], "reference_time": [pd.Timestamp("2024-12-10", tz=TZ)]})
    h = pd.DataFrame({
        "spout_no": ["1", "1"],
        "reference_time": [pd.Timestamp("2024-11-01", tz=TZ), pd.Timestamp("2024-11-02", tz=TZ)],
        "tap_end_time": [pd.Timestamp("2024-11-01 02:00", tz=TZ), pd.Timestamp("2024-11-02 02:00", tz=TZ)],
        "tap_time_len": [120.0, 140.0],
    })
    replay, diagnostic = replay_v21(v10, q, h, [499.0], cutoff=pd.Timestamp("2024-12-01 01:44", tz=TZ))
    expected = 0.75 * 100.123457 + 0.25 * 130.0
    assert replay.loc[0, "pred_tap_time_len"] == expected
    assert serialize_six_decimals(replay).decode().splitlines()[1] == f"q,200.123456,{expected:.6f}"
    assert diagnostic["changed_rows"] == 1 and diagnostic["fallback_rows"] == 0


def metric(loss):
    return {"loss": loss}


def full_grid(delta=-0.001, replay_delta=-0.0005):
    result = {}
    for month, horizons in {6: 4, 7: 4, 8: 4, 9: 3, 10: 2, 11: 1}.items():
        for horizon in range(1, horizons + 1):
            value = 0.2 + (delta if horizon == 2 else 0.0)
            result[f"O2024{month:02d}_H{horizon}"] = {
                "horizon": horizon,
                "candidates": {
                    REFERENCE: {"overall": metric(0.2)},
                    CANDIDATE: {"overall": metric(value)},
                    REPLAY_CONTROL: {"overall": metric(0.2 + (replay_delta if horizon == 2 else 0.0))},
                },
            }
    for name in ("DEV_LONG", "DEV_SHORT"):
        result[name] = {"horizon": None, "candidates": {
            REFERENCE: {"overall": metric(0.2)}, CANDIDATE: {"overall": metric(0.2)},
            REPLAY_CONTROL: {"overall": metric(0.2)},
        }}
    return result


def test_acceptance_uses_v10_delta_and_v21_h2_control():
    metrics = full_grid()
    result = acceptance(metrics, {REFERENCE: {"J": 0.2}, CANDIDATE: {"J": 0.19975}}, engineering=True, iron_exact=True)
    assert result["status"] == "DEV_ACCEPTED_PENDING_OFFICIAL_IDENTITY"
    metrics = full_grid(delta=-0.0004, replay_delta=-0.0005)
    failed = acceptance(metrics, {REFERENCE: {"J": 0.2}, CANDIDATE: {"J": 0.2}}, engineering=True, iron_exact=True)
    assert not failed["gates"]["H2_mean_E"] and not failed["gates"]["H2_not_worse_than_V21_replay"]


def test_parameter_outputs_are_exclusive_and_cannot_be_overwritten(tmp_path):
    selected = pd.DataFrame({
        "sample_id": [f"x{i}" for i in range(100)], "Q": 1.0, "d": 1.0, "tap_time_len": 1.5,
    })
    budget = LambdaBudget(tmp_path)
    first = budget.derive(6, "1", selected)
    assert budget.derive(6, "1", selected) == first
    changed = selected.copy()
    changed.loc[0, "tap_time_len"] = 2.0
    with pytest.raises(ContractError, match="different OOF"):
        budget.derive(6, "1", changed)


def test_lambda_budget_counts_intents_separately_from_twelve_results(tmp_path):
    selected = pd.DataFrame({
        "sample_id": [f"x{i}" for i in range(100)], "Q": 1.0, "d": 1.0, "tap_time_len": 1.5,
    })
    budget = LambdaBudget(tmp_path)
    for month in range(6, 12):
        for spout in ("1", "2"):
            budget.derive(month, spout, selected)
    assert budget.counts() == {"parameter_slots_completed": 12, "fit_attempts": 12, "zero_fallbacks": 0}


def test_registration_records_v16_state_analog_not_burden_plan():
    config = Path("configs/optimization_v0_16/experiment.yaml").read_text(encoding="utf-8")
    v22 = Path("configs/optimization_v0_22/experiment.yaml").read_text(encoding="utf-8")
    assert "V9_STATE_ANALOG_STRUCTURAL" in config
    assert "V22_CAUSAL_H2_QRF_SHRINK" in v22
    assert "burden_event_segmentation" not in v22
