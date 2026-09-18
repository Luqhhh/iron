import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.optimization.oob_pool_v31 import (
    ATTACHMENT_PROTOCOL,
    CANDIDATE_A,
    CANDIDATE_B,
    PARENT,
    POOLED_PROTOCOL,
    V21_REPLAY,
    V26A_PARENT,
    V28I,
    V29I,
    V29T,
    assert_legacy_reproduces_parent,
    compose_iron_candidate,
    compose_time_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


def endpoint(iron=("10.000000", "20.000000"), time=("100.000000", "200.000000"), ids=("a", "b")):
    return pd.DataFrame({"sample_id": list(ids), "pred_tap_iron": list(iron), "pred_tap_time_len": list(time)})


def test_candidate_ids_and_protocols_frozen():
    assert CANDIDATE_A == "V31I_OOB_OCCURRENCE_POOL_BLEND"
    assert CANDIDATE_B == "V31T_OOB_OCCURRENCE_POOL_TIME"
    assert PARENT == "V30A_OOB_BOTH_TARGETS"
    assert V29I == "V29I_OOB_LEAF_QRF_BLEND"
    assert V29T == "V29T_OOB_LEAF_QRF_TIME"
    assert POOLED_PROTOCOL == "QRF_FROZEN_FOREST_OOB_OCCURRENCE_POOLING_v031"
    assert ATTACHMENT_PROTOCOL == "QRF_FROZEN_FOREST_OOB_LEAF_RESPONSE_v029"


def test_a_blends_original_complete_iron_not_parent_iron_and_copies_parent_time():
    parent = endpoint(iron=("999.000000", "999.000000"), time=("111.111111", "222.222222"))
    complete = endpoint(iron=("10.000000", "20.000000"), time=("1.000000", "2.000000"))
    result = compose_iron_candidate(parent, complete, [12.0, 22.0])
    assert result.pred_tap_iron.tolist() == ["11.000000", "21.000000"]
    assert result.pred_tap_time_len.tolist() == ["111.111111", "222.222222"]
    assert result.pred_tap_iron.tolist() != parent.pred_tap_iron.tolist()


def test_a_aligns_by_sample_id_and_rejects_mismatch():
    result = compose_iron_candidate(endpoint(), endpoint().iloc[::-1], [1.0, 2.0])
    assert result.sample_id.tolist() == ["a", "b"]
    with pytest.raises(ContractError, match="sample ID sets"):
        compose_iron_candidate(endpoint(), endpoint(ids=("a", "c")), [1.0, 2.0])


def test_a_uses_original_complete_iron_not_parent_iron():
    parent = endpoint(iron=("0.000000", "0.000000"))
    complete = endpoint(iron=("100.000000", "200.000000"))
    result = compose_iron_candidate(parent, complete, [100.0, 200.0])
    assert result.pred_tap_iron.tolist() == ["100.000000", "200.000000"]
    # The prohibited rowwise average of the V30A parent and the new QRF would
    # have produced 50/100 here, so this also rejects that composition.
    assert result.pred_tap_iron.tolist() != ["50.000000", "100.000000"]


def test_b_replaces_only_time_and_keeps_parent_iron():
    parent = endpoint(iron=("12.345678", "23.456789"), time=("1.000000", "2.000000"))
    result = compose_time_candidate(parent, [55.5, 66.25])
    assert result.pred_tap_iron.tolist() == parent.pred_tap_iron.tolist()
    assert result.pred_tap_time_len.tolist() == ["55.500000", "66.250000"]


def test_b_roundtrips_six_decimals_before_serialization():
    parent = endpoint()
    result = compose_time_candidate(parent, [1.2345674, 2.3456789])
    assert result.pred_tap_time_len.tolist() == ["1.234567", "2.345679"]
    with pytest.raises(ContractError, match="row count"):
        compose_time_candidate(parent, [1.0])


def test_legacy_switch_back_reproduces_parent():
    parent = endpoint(iron=("11.000000", "21.000000"), time=("55.500000", "66.250000"))
    complete = endpoint(iron=("10.000000", "20.000000"))
    receipt = assert_legacy_reproduces_parent(parent, complete, [12.0, 22.0], [55.5, 66.25])
    assert receipt["legacy_iron_switch_back_exact"] is True
    assert receipt["legacy_time_switch_back_exact"] is True


def test_legacy_switch_back_rejects_nonparent():
    parent = endpoint(iron=("11.000000", "21.000000"), time=("55.500000", "66.250000"))
    complete = endpoint(iron=("10.000000", "20.000000"))
    with pytest.raises(ContractError, match="legacy iron"):
        assert_legacy_reproduces_parent(parent, complete, [13.0, 23.0], [55.5, 66.25])


def registration():
    return load_yaml("configs/optimization_v0_31/experiment.yaml")


def test_registration_contract():
    value = registration()
    assert value["branch"] == "optimization-v0.31-oob-occurrence-pooling"
    assert value["base_commit"] == "383610d615d9a4876a1d1dd44c3fb1ce238a5944"
    assert value["candidates"] == {"A": CANDIDATE_A, "B": CANDIDATE_B}
    assert value["protocol"] == POOLED_PROTOCOL
    assert value["pooled_reference_protocol"] == POOLED_PROTOCOL
    assert value["oob_attachment_protocol"] == ATTACHMENT_PROTOCOL
    budget = value["budget"]
    assert budget["new_model_or_tree_fits"] == 0
    assert budget["new_tree_fits"] == 0
    assert budget["new_preprocessing_lad_beta_lambda_bias_fits"] == 0
    assert budget["pooled_read_protocol_A_total"] == 7
    assert budget["pooled_read_protocol_B_total"] == 7
    assert budget["certified_oob_attachment_reuse_total"] == 14
    assert budget["new_candidate_packages"] == 2
    assert budget["new_candidate_platform_tests"] == 2
    assert budget["third_candidate"] == 0
    assert value["platform_order"] == ["A", "B"]
    assert value["platform_feedback_may_change_second_candidate"] is False
    assert value["pooled_rule"]["weight"] == "c_i_over_S"


def test_reference_identities_registered():
    value = registration()
    assert value["source_identities"]["v30a_result_sha256"] == "97b6c3e0c648a0c6454cc9b36f518625487c03d1efd7621746d049457c4aed9a"
    assert value["source_identities"]["v30a_zip_sha256"] == "fffcf23b04b3069cd71027682047eea764a2747c2dea48cca91320d67e113986"
    assert value["source_identities"]["v29i_result_sha256"] == "2b5bb1284c25ba9548d7d0c91d167adee0cb34b1d658b10e778bedd35063cf6c"
    assert value["source_identities"]["v29t_result_sha256"] == "53bf201d6b524c10261364c530bd46a32dc794659b68cdb98a68764cc3ab3dcd"
    assert value["source_identities"]["v26a_result_sha256"] == "9b42d41778fa23dfb07e825ae2ad44461e4d9886c11ea08e23a2dc553428cbad"
    assert value["source_identities"]["v21_result_sha256"] == "d5a2e118655107af39a76b99f3f1d053cc9469885d859113ba6f18ef20b9cece"


def test_source_files_registered_and_planned():
    for relative in (
        "workers/qrf_v031/pooled_response.py",
        "workers/qrf_v031/pooled_oob_reference.py",
        "workers/qrf_v031/worker.py",
        "workers/qrf_v031/test_pooled_response.py",
        "src/bf_tap/optimization/oob_pool_v31.py",
        "src/bf_tap/optimization/oob_pool_v31_run.py",
        "scripts/optimization_v31_oob_pool.py",
        "scripts/optimization_v31_cold_check.py",
        "configs/optimization_v0_31/experiment.yaml",
        "configs/optimization_v0_31/access_scope.yaml",
        "docs/optimization_v0_31/PLAN.md",
    ):
        assert (ROOT / relative).is_file(), relative


def _load_module(name, relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_occurrence_rule_integer_lower_median_and_equal_examples():
    reference = _load_module("v31_pooled_reference_test", "workers/qrf_v031/pooled_oob_reference.py")
    y = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    selected = [np.asarray([0]), np.asarray(list(range(1, 10)))]
    value, counts = reference.pooled_lower_median(y, selected)
    assert int(counts.sum()) == 10
    assert int(counts[0]) == 1
    assert value == 50.0
    even_value, even_counts = reference.pooled_lower_median(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        [np.asarray([0, 1]), np.asarray([2, 3])],
    )
    assert int(even_counts.sum()) == 4
    assert even_value == 2.0


def test_occurrence_rule_counts_cross_tree_repeats_but_rejects_duplicate_within_leaf():
    reference = _load_module("v31_pooled_reference_test_2", "workers/qrf_v031/pooled_oob_reference.py")
    y = np.asarray([1.0, 2.0, 3.0, 4.0])
    value, counts = reference.pooled_lower_median(y, [np.asarray([0, 1]), np.asarray([0, 1])])
    assert value == 1.0
    assert int(counts[0]) == 2 and int(counts.sum()) == 4
    with pytest.raises(ValueError, match="repeated training row"):
        reference.occurrence_counts([np.asarray([0, 0])], len(y))
