from pathlib import Path

import pandas as pd
import pytest

from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.optimization.time_leaf_v35 import (
    CANDIDATE_A, CANDIDATE_B, PARENT, PROTOCOL, compose_time_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


def endpoint(iron=("10.000000", "20.000000"), time=("100.000000", "200.000000")):
    return pd.DataFrame({"sample_id": ["a", "b"], "pred_tap_iron": list(iron), "pred_tap_time_len": list(time)})


def test_ids():
    assert CANDIDATE_A == "V35A_OOB_MIDPOINT_LEAF_MEAN_TIME"
    assert CANDIDATE_B == "V35B_OOB_LOWER_MEDIAN_OF_LEAF_POINTS_TIME"
    assert PARENT == "V34T_OOB_LEAF_MEDIAN_BAGGING_TIME"
    assert PROTOCOL == "FROZEN_OOB_TIME_LEAF_LOCATION_AND_VOTE_v035"


@pytest.mark.parametrize("label", [CANDIDATE_A, CANDIDATE_B])
def test_compose_changes_only_time(label):
    parent = endpoint(iron=("12.345678", "23.456789"))
    result = compose_time_candidate(parent, [1.2345674, 2.3456789], label=label)
    assert result.pred_tap_iron.tolist() == parent.pred_tap_iron.tolist()
    assert result.pred_tap_time_len.tolist() == ["1.234567", "2.345679"]


def test_compose_rejects_invalid_values_and_label():
    with pytest.raises(ContractError): compose_time_candidate(endpoint(), [1.0], label=CANDIDATE_A)
    with pytest.raises(ContractError): compose_time_candidate(endpoint(), [1.0, -2.0], label=CANDIDATE_B)
    with pytest.raises(ContractError): compose_time_candidate(endpoint(), [1.0, 2.0], label="other")


def test_registration_config():
    config = load_yaml(ROOT / "configs/optimization_v0_35/experiment.yaml")
    assert config["branch"] == "optimization-v0.35-time-leaf-location-and-vote"
    assert config["base_commit"] == "6eee2e77e0d1393ea1e2f148eb230fa93a54ae4b"
    assert config["candidates"] == {"A": CANDIDATE_A, "B": CANDIDATE_B}
    assert config["budget"]["new_model_or_tree_fits"] == 0
    assert config["budget"]["new_midpoint_tables_A_total"] == 7
    assert config["budget"]["new_lower_tables_B_total"] == 0


def test_public_files_exist():
    paths = ["configs/optimization_v0_35/experiment.yaml", "configs/optimization_v0_35/access_scope.yaml",
             "src/bf_tap/optimization/time_leaf_v35.py", "src/bf_tap/optimization/time_leaf_v35_run.py",
             "scripts/optimization_v35_time_leaf.py", "scripts/optimization_v35_cold_check.py",
             "workers/qrf_v035/time_leaf_location_vote.py", "workers/qrf_v035/worker.py",
             "workers/qrf_v035/test_time_leaf_location_vote.py", "docs/optimization_v0_35/PLAN.md"]
    assert all((ROOT / path).is_file() for path in paths)
