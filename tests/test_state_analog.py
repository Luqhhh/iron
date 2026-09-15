import numpy as np
import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.optimization.state_analog import StateAlphaCalibrator, StateAnalogCalibrator


def bank(rows=160):
    state = np.r_[np.linspace(-1.2, -0.3, rows // 2), np.linspace(0.3, 1.2, rows // 2)]
    ratio = (state > 0).astype(float)
    cutoff = pd.Timestamp("2024-06-01", tz="Asia/Shanghai")
    reference = pd.date_range("2024-04-01", periods=rows, freq="6h", tz="Asia/Shanghai")
    base_iron = np.full(rows, 100.0)
    base_time = np.full(rows, 20.0)
    direction_iron = np.full(rows, 20.0)
    direction_time = np.full(rows, 5.0)
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(rows)],
            "reference_time": reference,
            "label_available_at": reference + pd.Timedelta(hours=2),
            "fold_cutoff": pd.Timestamp("2024-04-01", tz="Asia/Shanghai"),
            "train_reference_max": pd.Timestamp("2024-03-31", tz="Asia/Shanghai"),
            "train_available_max": pd.Timestamp("2024-04-01", tz="Asia/Shanghai"),
            "history_available_max": pd.Timestamp("2024-04-01", tz="Asia/Shanghai"),
            "pred_tap_iron": base_iron,
            "pred_tap_time_len": base_time,
            "pred_rate": np.full(rows, 6.0),
            "tap_iron": base_iron + ratio * direction_iron,
            # base_iron / rate - base_time = -3.333..., so use the actual
            # structural direction rather than the illustrative value above.
            "tap_time_len": base_time + ratio * (base_iron / 6.0 - base_time),
            "state": state,
        }
    )


def test_state_local_coefficients_separate_regimes_and_stay_bounded():
    calibration = StateAnalogCalibrator(("state",), neighbors=32, prior_strength=4).fit(
        bank(), pd.Timestamp("2024-06-01", tz="Asia/Shanghai")
    )
    query = pd.DataFrame({"state": [-0.8, 0.8]})
    got = calibration.coefficients(query)
    assert got.alpha.shape == (2, 2)
    assert (got.alpha >= 0).all() and (got.alpha <= 1).all()
    assert got.alpha[0, 0] < 0.25
    assert got.alpha[1, 0] > 0.75
    assert got.alpha[0, 1] < 0.25
    assert got.alpha[1, 1] > 0.75


def test_unknown_state_reverts_toward_global_and_prediction_is_reorder_stable():
    calibration = StateAnalogCalibrator(("state",), neighbors=32, prior_strength=8).fit(
        bank(), pd.Timestamp("2024-06-01", tz="Asia/Shanghai")
    )
    states = pd.DataFrame({"state": [-0.8, 50.0, 0.8]})
    diagnostics = calibration.coefficients(states)
    assert diagnostics.reliability[1] < diagnostics.reliability[[0, 2]].min()
    predictions = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "pred_tap_iron": [100.0] * 3,
            "pred_tap_time_len": [20.0] * 3,
            "pred_rate": [6.0] * 3,
        }
    )
    direct, _ = calibration.predict(predictions, states)
    order = [2, 0, 1]
    shuffled, _ = calibration.predict(
        predictions.iloc[order].reset_index(drop=True), states.iloc[order].reset_index(drop=True)
    )
    restored = shuffled.set_index("sample_id").loc[direct.sample_id].reset_index()
    np.testing.assert_allclose(direct[["pred_tap_iron", "pred_tap_time_len"]], restored[["pred_tap_iron", "pred_tap_time_len"]])


def test_fit_rejects_future_oof_evidence():
    frame = bank()
    frame.loc[0, "train_available_max"] = pd.Timestamp(
        "2024-04-01 00:00:01", tz="Asia/Shanghai"
    )
    with pytest.raises(ContractError, match="temporal leakage"):
        StateAnalogCalibrator(("state",), neighbors=32).fit(
            frame, pd.Timestamp("2024-06-01", tz="Asia/Shanghai")
        )


def test_state_alpha_uses_causal_gate_and_learns_varying_coefficient():
    calibration = StateAlphaCalibrator(
        ("state",),
        parameters={
            **StateAlphaCalibrator.DEFAULT_PARAMETERS,
            "iterations": 80,
            "depth": 2,
        },
    ).fit(bank(480), pd.Timestamp("2024-08-01", tz="Asia/Shanghai"))
    assert calibration.gate_validation_["validation_rows"] > 0
    assert (calibration.beta_ >= 0).all() and (calibration.beta_ <= 1).all()
    query = pd.DataFrame({"state": [-0.8, 0.8]})
    got = calibration.coefficients(query)
    assert got.alpha.shape == (2, 2)
    assert (got.alpha >= 0).all() and (got.alpha <= 1).all()
    assert got.alpha[0, 0] < got.alpha[1, 0]
