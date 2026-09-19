from __future__ import annotations

import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.optimization.qrf_sampling_v33 import CANDIDATE_A, CANDIDATE_B, compose_time_candidate


def parent():
    return pd.DataFrame({
        "sample_id": ["a", "b"],
        "pred_tap_iron": ["10.000001", "20.000002"],
        "pred_tap_time_len": ["30.000003", "40.000004"],
    })


@pytest.mark.parametrize("label", [CANDIDATE_A, CANDIDATE_B])
def test_compose_time_preserves_parent_iron_strings(label):
    result = compose_time_candidate(parent(), [1.2345674, 2.3456786], label=label)
    assert result.pred_tap_iron.tolist() == parent().pred_tap_iron.tolist()
    assert result.pred_tap_time_len.tolist() == ["1.234567", "2.345679"]


def test_compose_rejects_unregistered_label_and_shape():
    with pytest.raises(ContractError):
        compose_time_candidate(parent(), [1, 2], label="other")
    with pytest.raises(ContractError):
        compose_time_candidate(parent(), [1], label=CANDIDATE_A)
