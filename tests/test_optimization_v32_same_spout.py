from pathlib import Path

import pandas as pd
import pytest

from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.optimization.same_spout_v32 import (
    ATTACHMENT_PROTOCOL,
    CANDIDATE_A,
    CANDIDATE_B,
    PARENT,
    SAME_SPOUT_PROTOCOL,
    compose_iron_candidate,
    compose_time_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


def endpoint(iron=("10.000000", "20.000000"), time=("100.000000", "200.000000"), ids=("a", "b")):
    return pd.DataFrame({"sample_id": list(ids), "pred_tap_iron": list(iron), "pred_tap_time_len": list(time)})


def test_frozen_candidate_and_protocol_ids():
    assert CANDIDATE_A == "V32I_SAME_SPOUT_OOB_BLEND"
    assert CANDIDATE_B == "V32T_SAME_SPOUT_OOB_TIME"
    assert PARENT == "V30A_OOB_BOTH_TARGETS"
    assert SAME_SPOUT_PROTOCOL == "QRF_FROZEN_FOREST_SAME_SPOUT_OOB_RESPONSE_v032"
    assert ATTACHMENT_PROTOCOL == "QRF_FROZEN_FOREST_OOB_LEAF_RESPONSE_v029"


def test_a_rebuilds_original_50_50_and_copies_parent_time():
    parent = endpoint(iron=("999.000000", "999.000000"), time=("111.111111", "222.222222"))
    complete = endpoint(iron=("10.000000", "20.000000"))
    result = compose_iron_candidate(parent, complete, [12.0, 22.0])
    assert result.pred_tap_iron.tolist() == ["11.000000", "21.000000"]
    assert result.pred_tap_time_len.tolist() == parent.pred_tap_time_len.tolist()


def test_a_does_not_average_parent_iron_again():
    parent = endpoint(iron=("0.000000", "0.000000"))
    complete = endpoint(iron=("100.000000", "200.000000"))
    result = compose_iron_candidate(parent, complete, [100.0, 200.0])
    assert result.pred_tap_iron.tolist() == ["100.000000", "200.000000"]
    assert result.pred_tap_iron.tolist() != ["50.000000", "100.000000"]


def test_a_aligns_complete_endpoint_by_id():
    result = compose_iron_candidate(endpoint(), endpoint().iloc[::-1], [12.0, 22.0])
    assert result.sample_id.tolist() == ["a", "b"]
    with pytest.raises(ContractError, match="sample ID sets"):
        compose_iron_candidate(endpoint(), endpoint(ids=("a", "c")), [12.0, 22.0])


def test_b_replaces_only_time_and_roundtrips_six():
    parent = endpoint(iron=("12.345678", "23.456789"))
    result = compose_time_candidate(parent, [1.2345674, 2.3456789])
    assert result.pred_tap_iron.tolist() == parent.pred_tap_iron.tolist()
    assert result.pred_tap_time_len.tolist() == ["1.234567", "2.345679"]


@pytest.mark.parametrize("function,arguments", [
    (compose_iron_candidate, (endpoint(), endpoint(), [1.0])),
    (compose_time_candidate, (endpoint(), [1.0])),
])
def test_composition_rejects_wrong_row_count(function, arguments):
    with pytest.raises(ContractError, match="row count"):
        function(*arguments)


def test_registration_contract_and_budget():
    value = load_yaml("configs/optimization_v0_32/experiment.yaml")
    assert value["branch"] == "optimization-v0.32-same-spout-oob-responses"
    assert value["base_commit"] == "656be40fc19283972f6c757917d8caf460d75cd4"
    assert value["candidates"] == {"A": CANDIDATE_A, "B": CANDIDATE_B}
    assert value["protocol"] == SAME_SPOUT_PROTOCOL
    assert value["oob_attachment_protocol"] == ATTACHMENT_PROTOCOL
    assert value["member_rule"]["known_query_empty_intersection"] == "use_original_selected_set_exact"
    assert value["member_rule"]["unknown_query"] == "bypass_conditioning"
    assert value["aggregation_rule"]["tree_mass"] == "equal"
    assert value["aggregation_rule"]["occurrence_pooling_v031"] == "forbidden"
    budget = value["budget"]
    assert budget["new_model_or_tree_fits"] == 0
    assert budget["conditional_read_protocol_A_total"] == 7
    assert budget["conditional_read_protocol_B_total"] == 7
    assert budget["certified_oob_attachment_reuse_total"] == 14
    assert budget["new_candidate_platform_tests"] == 2
    assert budget["third_candidate"] == 0


def test_parent_source_identities_frozen():
    value = load_yaml("configs/optimization_v0_32/experiment.yaml")
    assert value["source_identities"]["v30a_result_sha256"] == "97b6c3e0c648a0c6454cc9b36f518625487c03d1efd7621746d049457c4aed9a"
    assert value["source_identities"]["v30a_zip_sha256"] == "fffcf23b04b3069cd71027682047eea764a2747c2dea48cca91320d67e113986"


def test_required_v32_sources_exist():
    for relative in (
        "workers/qrf_v032/same_spout.py", "workers/qrf_v032/worker.py", "workers/qrf_v032/test_same_spout.py",
        "src/bf_tap/optimization/same_spout_v32.py", "src/bf_tap/optimization/same_spout_v32_run.py",
        "scripts/optimization_v32_same_spout.py", "scripts/optimization_v32_cold_check.py",
        "configs/optimization_v0_32/experiment.yaml", "configs/optimization_v0_32/access_scope.yaml",
        "docs/optimization_v0_32/PLAN.md",
    ):
        assert (ROOT / relative).is_file(), relative
