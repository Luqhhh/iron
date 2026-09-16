import numpy as np
import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.optimization.history_centered_v25 import (
    CANDIDATE_A,
    CANDIDATE_B,
    baseline_audit,
    compose_iron,
    compose_time,
    history_baseline,
    isolate,
    reconstruct,
)


def frame():
    return pd.DataFrame({
        "history__spout__tap_iron__last100_median": [10.0, np.nan, 0.0, np.nan],
        "history__spout__tap_iron__last100_count": [2.0, 0.0, 1.0, 0.0],
        "history__all__tap_iron__last100_median": [20.0, 30.0, 40.0, np.nan],
        "history__all__tap_iron__last100_count": [3.0, 4.0, 5.0, 0.0],
    })


def test_baseline_priority_real_zero_and_cold_start():
    baseline, source = history_baseline(frame(), "tap_iron")
    assert baseline.tolist() == [10.0, 30.0, 0.0, 0.0]
    assert source.tolist() == [2, 1, 2, 0]


def test_baseline_rejects_invalid_counts_and_missing_positive_median():
    value = frame(); value.loc[0, "history__spout__tap_iron__last100_count"] = .5
    with pytest.raises(ContractError, match="counts"): history_baseline(value, "tap_iron")
    value = frame(); value.loc[0, "history__spout__tap_iron__last100_median"] = np.nan
    with pytest.raises(ContractError, match="median"): history_baseline(value, "tap_iron")


def test_signed_reconstruction_clips_after_addition_not_before():
    got = reconstruct([100.0, 5.0], [-20.0, -10.0])
    assert got.tolist() == [80.0, 0.0]
    assert got[0] != 100.0 + max(0.0, -20.0)


def test_baseline_audit_preserves_signed_response_and_sources():
    audit = baseline_audit([10.0, 30.0, 0.0], [2, 1, 0], [8.0, 35.0, 1.0])
    assert audit["sources"] == {"zero": 1, "all": 1, "spout": 1}
    assert audit["negative_signed_rows"] == 1
    assert audit["reconstruction_max_abs"] == 0.0


def test_A_rate_fallback_uses_new_base():
    got, audit = compose_iron([10, 10], [20, 20], [0, 2], [100, 100], .25)
    assert got.tolist() == [12.0, 59.0]
    assert audit["rate_unusable"] == 1


def test_B_uses_fixed_gate_and_six_decimal_roundtrip():
    got = compose_time([10.1234566, 20], np.array([True, False]), [14, 99])
    assert got[0] == .75 * 10.123457 + .25 * 14
    assert got[1] == 20


def test_target_isolation():
    parent = pd.DataFrame({"sample_id": ["x"], "pred_tap_iron": [1.0], "pred_tap_time_len": [2.0]})
    a = isolate(parent, iron=[3.0], candidate=CANDIDATE_A)
    b = isolate(parent, time=[4.0], candidate=CANDIDATE_B)
    assert a.pred_tap_time_len.tolist() == [2.0]
    assert b.pred_tap_iron.tolist() == [1.0]


def test_unknown_target_and_candidate_rejected():
    with pytest.raises(ContractError): history_baseline(frame(), "wrong")
    parent = pd.DataFrame({"sample_id": ["x"], "pred_tap_iron": [1.0], "pred_tap_time_len": [2.0]})
    with pytest.raises(ContractError): isolate(parent, iron=[3.0], candidate="wrong")

