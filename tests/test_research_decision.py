from copy import deepcopy
import json
from pathlib import Path

import pytest

from bf_tap_r2.research_decision import validate_record


ROOT = Path("docs/research_decision_20260930")


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["DE3_EXAMPLE.json", "MODERNNCA_FULL_RECIPE.json"])
def test_public_decision_records_are_structurally_valid(name):
    assert validate_record(read(name)) == []


def test_local_gain_and_platform_report_are_not_conflated():
    record = read("DE3_EXAMPLE.json")
    assert [item["kind"] for item in record["evidence"]] == ["local", "platform"]
    assert record["evidence"][1]["verification"] == "user_reported"
    assert record["evidence"][0]["effect"] != record["evidence"][1]["effect"]


@pytest.mark.parametrize("mutation,expected", [
    (lambda r: r["context"].pop("data_boundary"), "context.data_boundary"),
    (lambda r: r.update(phase="attribution", changed_factors=["a", "b"]), "attribution_controls"),
    (lambda r: r["decision"].update(action="stop_exact_recipe", scope="tested_subfamily"), "exact_recipe scope"),
    (lambda r: r["decision"].update(action="exclude_proven_family", scope="proven_fixed_family"), "proof_source"),
    (lambda r: r["evidence"][1].update(verification="independent_receipt"), "receipt_reference"),
    (lambda r: r.update(time_budget=7200), "prospective wall-clock gate"),
])
def test_unsafe_or_unsupported_decision_fails(mutation, expected):
    record = deepcopy(read("DE3_EXAMPLE.json"))
    mutation(record)
    assert expected in "\n".join(validate_record(record))


def test_unobserved_plan_cannot_claim_results_or_promotion():
    record = read("MODERNNCA_FULL_RECIPE.json")
    record["evidence"] = deepcopy(read("DE3_EXAMPLE.json")["evidence"])
    record["decision"]["action"] = "continue"
    errors = "\n".join(validate_record(record))
    assert "planned decisions cannot contain observed evidence" in errors
    assert "planned decision must remain pending" in errors
