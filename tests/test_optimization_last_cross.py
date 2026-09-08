import pytest

from bf_tap.exceptions import ContractError
from bf_tap.optimization.v3_followup import fit_cross_target
from bf_tap.optimization.v3_last_cross import registration


def test_last_cross_is_bounded_and_closes_search():
    cfg = registration()
    assert cfg["cumulative_feature_crosses"] == 2
    assert cfg["stop_after_this_cross"] is True
    assert cfg["additional_parameter_configurations"] == cfg["additional_target_combinations"] == 0
    assert cfg["target_configs"] == {"tap_iron": "CB08", "tap_time_len": "CB02"}


def test_cross_feature_candidate_mismatch_refused_before_data_access():
    with pytest.raises(ContractError, match="identity"):
        fit_cross_target(None, None, None, None, None, increment="F-C", candidate="CB-FB-raw")
