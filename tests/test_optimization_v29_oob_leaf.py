import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.optimization.oob_leaf_v29 import (
    CANDIDATE_A,
    CANDIDATE_B,
    compose_iron_candidate,
    compose_time_candidate,
)


def endpoint(iron=("10.000000", "20.000000"), time=("100.000000", "200.000000")):
    return pd.DataFrame({"sample_id": ["a", "b"], "pred_tap_iron": iron, "pred_tap_time_len": time})


def test_ids_frozen():
    assert CANDIDATE_A == "V29I_OOB_LEAF_QRF_BLEND"
    assert CANDIDATE_B == "V29T_OOB_LEAF_QRF_TIME"


def test_iron_uses_catboost_endpoint_not_parent_blend():
    parent = endpoint(iron=("12.000000", "22.000000"))
    result = compose_iron_candidate(parent, endpoint(), [14.0, 24.0])
    assert result.pred_tap_iron.tolist() == ["12.000000", "22.000000"]
    assert result.pred_tap_time_len.tolist() == parent.pred_tap_time_len.tolist()


def test_iron_micro_tie_uses_v28_half_even():
    parent = endpoint()
    catboost = endpoint(iron=("1.000000", "1.000001"))
    result = compose_iron_candidate(parent, catboost, [1.000001, 1.000002])
    assert result.pred_tap_iron.tolist() == ["1.000000", "1.000002"]


def test_iron_aligns_complete_endpoint_by_id():
    parent = endpoint()
    result = compose_iron_candidate(parent, endpoint().iloc[::-1], [14.0, 24.0])
    assert result.sample_id.tolist() == ["a", "b"]


def test_iron_requires_same_parent_time():
    with pytest.raises(ContractError, match="time endpoint"):
        compose_iron_candidate(endpoint(), endpoint(time=("101.000000", "200.000000")), [1, 2])


def test_iron_rejects_wrong_length():
    with pytest.raises(ContractError, match="row count"):
        compose_iron_candidate(endpoint(), endpoint(), [1])


def test_time_copies_v28i_iron_strings():
    parent = endpoint(iron=("12.123456", "22.654321"))
    result = compose_time_candidate(parent, [111.25, 222.5])
    assert result.pred_tap_iron.tolist() == parent.pred_tap_iron.tolist()
    assert result.pred_tap_time_len.tolist() == ["111.250000", "222.500000"]


def test_time_aligns_length():
    with pytest.raises(ContractError, match="row count"):
        compose_time_candidate(endpoint(), [1])


def test_nonfinite_oob_endpoint_rejected():
    with pytest.raises(ContractError, match="finite"):
        compose_time_candidate(endpoint(), [1, float("nan")])


def test_negative_oob_endpoint_rejected():
    with pytest.raises(ContractError, match="nonnegative"):
        compose_iron_candidate(endpoint(), endpoint(), [1, -1])

