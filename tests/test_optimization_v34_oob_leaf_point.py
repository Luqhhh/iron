from pathlib import Path

import pandas as pd
import pytest

from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.optimization.oob_leaf_point_v34 import (
    ATTACHMENT_PROTOCOL,
    CANDIDATE_A,
    CANDIDATE_B,
    LEAF_POINT_PROTOCOL,
    PARENT,
    compose_iron_candidate,
    compose_time_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


def endpoint(iron=("10.000000", "20.000000"), time=("100.000000", "200.000000"), ids=("a", "b")):
    return pd.DataFrame({"sample_id": list(ids), "pred_tap_iron": list(iron), "pred_tap_time_len": list(time)})


def test_frozen_ids_and_protocols():
    assert CANDIDATE_A == "V34I_OOB_LEAF_MEDIAN_BAGGING_BLEND"
    assert CANDIDATE_B == "V34T_OOB_LEAF_MEDIAN_BAGGING_TIME"
    assert PARENT == "V30A_OOB_BOTH_TARGETS"
    assert LEAF_POINT_PROTOCOL == "FROZEN_OOB_LEAF_LOWER_MEDIAN_BAGGING_v034"
    assert ATTACHMENT_PROTOCOL == "QRF_FROZEN_FOREST_OOB_LEAF_RESPONSE_v029"


def test_iron_rebuilds_complete_endpoint_and_copies_time():
    parent = endpoint(iron=("999.000000", "999.000000"), time=("111.111111", "222.222222"))
    complete = endpoint(iron=("10.000000", "20.000000"))
    result = compose_iron_candidate(parent, complete, [12.0, 22.0])
    assert result.pred_tap_iron.tolist() == ["11.000000", "21.000000"]
    assert result.pred_tap_time_len.tolist() == parent.pred_tap_time_len.tolist()


def test_iron_does_not_average_parent_again():
    result = compose_iron_candidate(endpoint(iron=("0.000000", "0.000000")), endpoint(iron=("100.000000", "200.000000")), [100.0, 200.0])
    assert result.pred_tap_iron.tolist() == ["100.000000", "200.000000"]


def test_iron_aligns_complete_endpoint_by_id():
    result = compose_iron_candidate(endpoint(), endpoint().iloc[::-1], [12.0, 22.0])
    assert result.sample_id.tolist() == ["a", "b"]
    with pytest.raises(ContractError, match="sample ID sets"):
        compose_iron_candidate(endpoint(), endpoint(ids=("a", "c")), [12.0, 22.0])


def test_time_replaces_only_time_and_roundtrips():
    parent = endpoint(iron=("12.345678", "23.456789"))
    result = compose_time_candidate(parent, [1.2345674, 2.3456789])
    assert result.pred_tap_iron.tolist() == parent.pred_tap_iron.tolist()
    assert result.pred_tap_time_len.tolist() == ["1.234567", "2.345679"]


@pytest.mark.parametrize("function,args", [
    (compose_iron_candidate, (endpoint(), endpoint(), [1.0])),
    (compose_time_candidate, (endpoint(), [1.0])),
    (compose_time_candidate, (endpoint(), [1.0, -2.0])),
])
def test_invalid_endpoint_values_are_rejected(function, args):
    with pytest.raises(ContractError):
        function(*args)


def test_registration_config_is_frozen():
    config = load_yaml(ROOT / "configs/optimization_v0_34/experiment.yaml")
    assert config["branch"] == "optimization-v0.34-oob-leaf-median-bagging"
    assert config["base_commit"] == "e8dda75a93f159620d9f70ae0cfeed3e2cbf68d0"
    assert config["candidates"] == {"A": CANDIDATE_A, "B": CANDIDATE_B}
    assert config["protocol"] == LEAF_POINT_PROTOCOL
    assert config["budget"]["new_model_or_tree_fits"] == 0
    assert config["budget"]["leaf_point_tables_total"] == 14
    assert config["budget"]["automatic_uploads"] == 0
    assert config["budget"]["desktop_writes"] == 0


def test_expected_public_files_are_present():
    paths = [
        "configs/optimization_v0_34/experiment.yaml", "configs/optimization_v0_34/access_scope.yaml",
        "src/bf_tap/optimization/oob_leaf_point_v34.py", "src/bf_tap/optimization/oob_leaf_point_v34_run.py",
        "scripts/optimization_v34_oob_leaf_point.py", "scripts/optimization_v34_cold_check.py",
        "workers/qrf_v034/leaf_point_bagging.py", "workers/qrf_v034/worker.py",
        "workers/qrf_v034/test_leaf_point_bagging.py", "docs/optimization_v0_34/PLAN.md",
    ]
    assert all((ROOT / path).is_file() for path in paths)
