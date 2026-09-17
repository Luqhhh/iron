import numpy as np
import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.optimization.fixed_blend_v28 import (
    CANDIDATE_A,
    CANDIDATE_B,
    cancellation_diagnostic,
    canonical_six,
    compose_candidate,
    equal_blend_six,
    submission_bytes,
)


def endpoint(iron=("1.000000", "2.000000"), time=("3.000000", "4.000000")):
    return pd.DataFrame({"sample_id": ["a", "b"], "pred_tap_iron": iron, "pred_tap_time_len": time})


def test_candidate_ids_are_frozen():
    assert CANDIDATE_A == "V28I_CB_QRF_EQUAL_BLEND"
    assert CANDIDATE_B == "V28T_L1_L2_QRF_EQUAL_BLEND"


def test_integer_micro_unit_average_uses_ties_to_even():
    assert equal_blend_six("1.000000", "1.000001") == "1.000000"
    assert equal_blend_six("1.000001", "1.000002") == "1.000002"
    assert equal_blend_six("0.000000", "2.000000") == "1.000000"


def test_canonical_six_accepts_short_exact_text_but_rejects_extra_precision():
    assert canonical_six("1.2") == "1.200000"
    with pytest.raises(ContractError, match="six-decimal"):
        canonical_six("1.0000004")
    with pytest.raises(ContractError, match="six-decimal"):
        canonical_six("-1")


def test_iron_blend_aligns_by_id_and_preserves_parent_time():
    donor = endpoint(iron=("4.000000", "6.000000"), time=("3.000000", "4.000000")).iloc[::-1]
    got = compose_candidate(endpoint(), donor, changed_target="tap_iron")
    assert got.sample_id.tolist() == ["a", "b"]
    assert got.pred_tap_iron.tolist() == ["2.500000", "4.000000"]
    assert got.pred_tap_time_len.tolist() == ["3.000000", "4.000000"]


def test_time_blend_preserves_parent_iron():
    donor = endpoint(time=("5.000000", "8.000000"))
    got = compose_candidate(endpoint(), donor, changed_target="tap_time_len")
    assert got.pred_tap_iron.tolist() == ["1.000000", "2.000000"]
    assert got.pred_tap_time_len.tolist() == ["4.000000", "6.000000"]


def test_reverse_composition_preserves_reverse_order():
    parent = endpoint().iloc[::-1]
    donor = endpoint(time=("5.000000", "8.000000")).iloc[::-1]
    got = compose_candidate(parent, donor, changed_target="tap_time_len")
    assert got.sample_id.tolist() == ["b", "a"]
    assert got.pred_tap_time_len.tolist() == ["6.000000", "4.000000"]


def test_wrong_isolated_target_or_id_set_is_rejected():
    with pytest.raises(ContractError, match="isolated"):
        compose_candidate(endpoint(), endpoint(time=("9.000000", "4.000000")), changed_target="tap_iron")
    donor = endpoint().copy()
    donor.loc[0, "sample_id"] = "x"
    with pytest.raises(ContractError, match="ID sets"):
        compose_candidate(endpoint(), donor, changed_target="tap_time_len")


def test_payload_is_deterministic_and_six_decimal():
    value = compose_candidate(endpoint(), endpoint(time=("5.000000", "8.000000")), changed_target="tap_time_len")
    expected = b"sample_id,pred_tap_iron,pred_tap_time_len\na,1.000000,4.000000\nb,2.000000,6.000000\n"
    assert submission_bytes(value) == expected
    assert submission_bytes(value) == submission_bytes(value)


def test_cancellation_identity_and_rounding_bound():
    actual = np.array([10.0, 10.0, 10.0])
    left = np.array([12.0, 12.0, 9.0])
    right = np.array([6.0, 13.0, 11.0])
    exact = (left + right) / 2
    got = cancellation_diagnostic(actual, left, right, exact)
    assert got["opposite_sign_count"] == 2
    assert got["identity_residual"] <= 1e-12
    assert got["cancellation_sum"] == pytest.approx(3.0)


def test_invalid_diagnostic_alignment_is_rejected():
    with pytest.raises(ContractError, match="aligned"):
        cancellation_diagnostic(np.array([1.0]), np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]))
