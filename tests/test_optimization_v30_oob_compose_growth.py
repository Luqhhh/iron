import ast
from pathlib import Path

import pandas as pd
import pytest

from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.optimization.oob_compose_v30 import (
    APPENDED_FOREST_PROTOCOL,
    CANDIDATE_A,
    CANDIDATE_B,
    IRON_SOURCE,
    OOB_ATTACHMENT_PROTOCOL,
    PARENT,
    PRIMARY_AGGREGATION,
    TIME_SOURCE,
    align_sources,
    compose_1024_time,
    compose_both_targets,
    verify_composition_identities,
    verify_isolated_deltas,
)


ROOT = Path(__file__).resolve().parents[1]


def endpoint(iron=("10.000000", "20.000000"), time=("100.000000", "200.000000"), ids=("a", "b")):
    return pd.DataFrame({"sample_id": list(ids), "pred_tap_iron": list(iron), "pred_tap_time_len": list(time)})


def test_candidate_ids_frozen():
    assert CANDIDATE_A == "V30A_OOB_BOTH_TARGETS"
    assert CANDIDATE_B == "V30B_OOB_TIME_1024"
    assert IRON_SOURCE == "V29I_OOB_LEAF_QRF_BLEND"
    assert TIME_SOURCE == "V29T_OOB_LEAF_QRF_TIME"


def test_a_copies_complete_columns_without_averaging():
    v29i = endpoint(iron=("11.111111", "22.222222"))
    v29t = endpoint(time=("111.000000", "222.500000"))
    result = compose_both_targets(v29i, v29t)
    assert result.pred_tap_iron.tolist() == ["11.111111", "22.222222"]
    assert result.pred_tap_time_len.tolist() == ["111.000000", "222.500000"]


def test_a_aligns_by_sample_id_and_rejects_mismatch():
    v29i = endpoint()
    v29t = endpoint().iloc[::-1]
    aligned_i, aligned_t = align_sources(v29i, v29t)
    assert aligned_i.sample_id.tolist() == aligned_t.sample_id.tolist() == ["a", "b"]
    with pytest.raises(ContractError, match="sample ID sets"):
        compose_both_targets(v29i, endpoint(ids=("a", "c")))


def test_b_replaces_only_time_and_keeps_iron_strings():
    a = compose_both_targets(endpoint(iron=("12.345678", "23.456789")), endpoint())
    result = compose_1024_time(a, [55.5, 66.25])
    assert result.pred_tap_iron.tolist() == a.pred_tap_iron.tolist()
    assert result.pred_tap_time_len.tolist() == ["55.500000", "66.250000"]
    with pytest.raises(ContractError, match="row count"):
        compose_1024_time(a, [1.0])
    with pytest.raises(ContractError, match="finite"):
        compose_1024_time(a, [1.0, float("nan")])


def registration():
    return load_yaml("configs/optimization_v0_30/experiment.yaml")


def test_registration_contract():
    value = registration()
    assert value["branch"] == "optimization-v0.30-oob-compose-and-time-growth"
    assert value["base_commit"] == "5e00f6340e5af3f4abedfc0f6365b2cb8a84a8d3"
    assert value["candidates"] == {"A": CANDIDATE_A, "B": CANDIDATE_B}
    assert value["protocol"] == "QRF_OOB_COMPOSE_AND_TIME_GROWTH_v030"
    assert value["appended_forest_protocol"] == APPENDED_FOREST_PROTOCOL
    assert value["oob_attachment_protocol"] == OOB_ATTACHMENT_PROTOCOL
    assert value["oob_core_protocol"] == "QRF_FROZEN_FOREST_OOB_LEAF_RESPONSE_v029"
    budget = value["budget"]
    assert budget["B_forest_append_fits_total"] == 7
    assert budget["B_new_trees_total"] == 5376
    assert budget["B_reused_parent_trees_total"] == 1792
    assert budget["B_models_held_trees_total"] == 7168
    assert budget["new_1024_oob_attachments_total"] == 7
    assert budget["new_candidate_platform_tests"] == 2
    assert budget["third_candidate"] == 0
    assert value["platform_order"] == ["A", "B"]
    assert value["platform_feedback_may_change_second_candidate"] is False
    identities = value["source_identities"]
    assert identities["v29i_result_sha256"] == "2b5bb1284c25ba9548d7d0c91d167adee0cb34b1d658b10e778bedd35063cf6c"
    assert identities["v29t_result_sha256"] == "53bf201d6b524c10261364c530bd46a32dc794659b68cdb98a68764cc3ab3dcd"
    assert identities["v29i_zip_sha256"] == "68320081ec34e8e1296ddff3ac75fdb5dfedffac5d07fff7be95bc74e1b3a581"
    assert identities["v29t_zip_sha256"] == "650765ff7b749eda2ff41c7f8c0e99a33e4171958ee79918a6e4f14e623f244b"


def test_source_files_registered_and_planned():
    for relative in (
        "workers/qrf_v030/append_forest.py",
        "workers/qrf_v030/oob_append.py",
        "workers/qrf_v030/worker.py",
        "workers/qrf_v030/test_append_forest.py",
        "workers/qrf_v030/test_oob_append.py",
        "scripts/optimization_v30_oob_compose_growth.py",
        "scripts/optimization_v30_cold_check.py",
        "docs/optimization_v0_30/PLAN.md",
    ):
        assert (ROOT / relative).is_file(), relative


def _module_constants(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Constant):
                result[node.targets[0].id] = node.value.value
    return result


def test_worker_constants_match_registration():
    config = registration()
    constants = _module_constants("workers/qrf_v030/append_forest.py")
    assert constants["PROTOCOL"] == config["appended_forest_protocol"]
    assert constants["PARENT_PROTOCOL"] == "QRF_FULLTRAIN_PARTITION_v026"
    assert constants["PARENT_CANDIDATE_ID"] == "V26A_QRF_ABSOLUTE_SPLIT_TIME"
    assert constants["PARENT_TREES"] == 256
    assert constants["TOTAL_TREES"] == 1024
    assert constants["TOTAL_TREES"] - constants["PARENT_TREES"] == 768
    assert constants["WARM_START"] is True
    assert constants["TARGET"] == "tap_time_len"
    assert constants["UNIT"] == "minutes"
    attach = _module_constants("workers/qrf_v030/oob_append.py")
    assert attach["ATTACHMENT_PROTOCOL"] == config["oob_attachment_protocol"]
    assert attach["CORE_PROTOCOL"] == config["oob_core_protocol"]
    assert attach["REGRESSION_TREES"] == 256


def scenario():
    units = ("O202406_H1", "DEV_LONG")
    algorithms = {
        PARENT: (0.200, 0.300),
        IRON_SOURCE: (0.190, 0.300),
        TIME_SOURCE: (0.200, 0.280),
        CANDIDATE_A: (0.190, 0.280),
        CANDIDATE_B: (0.190, 0.270),
    }
    rows = []
    for unit in units:
        for algorithm, (iron, time) in algorithms.items():
            loss = (iron + time) / 2
            for target, wmape in (("tap_iron", iron), ("tap_time_len", time)):
                rows.append({
                    "algorithm": algorithm, "unit": unit, "target": target, "n": 10,
                    "cutoff": unit, "horizon": 1, "target_sum": 1000.0,
                    "absolute_error_sum": wmape * 1000.0, "wmape": wmape, "E": loss,
                })
    scorecard = pd.DataFrame(rows)
    summary = []
    for algorithm, (iron, time) in algorithms.items():
        for scope in (*units, "H1", "J"):
            summary.append({
                "algorithm": algorithm, "scope": scope, "aggregation": PRIMARY_AGGREGATION,
                "wmape_iron": iron, "wmape_time": time, "E": (iron + time) / 2,
            })
    return scorecard, pd.DataFrame(summary)


def test_composition_identities_pass_and_isolate_targets():
    scorecard, summary = scenario()
    receipt = verify_composition_identities(scorecard, summary)
    assert receipt["checked_cells"] == 2
    assert receipt["checked_macro_summaries"] == 4
    assert receipt["maximum_absolute_delta_identity_residual"] <= receipt["tolerance"]
    assert verify_isolated_deltas(scorecard, summary, {CANDIDATE_A: "tap_iron"}, parent=TIME_SOURCE)["checked_cells"] == 2
    assert verify_isolated_deltas(scorecard, summary, {CANDIDATE_B: "tap_time_len"}, parent=CANDIDATE_A)["checked_cells"] == 2


def test_composition_identities_reject_averaged_candidate():
    scorecard, summary = scenario()
    broken = scorecard.copy()
    mask = broken.algorithm == CANDIDATE_A
    mask &= broken.unit == "DEV_LONG"
    broken.loc[mask & (broken.target == "tap_iron"), "wmape"] = 0.195
    broken.loc[mask, "E"] = 0.2375
    with pytest.raises(ContractError, match="A_iron_exact_copy"):
        verify_composition_identities(broken, summary)
