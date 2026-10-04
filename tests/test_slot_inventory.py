from pathlib import Path

from bf_tap_r2.slot_screen import read_json


def test_inventory_spec_entries_are_well_formed():
    spec = read_json(Path("configs/slot_inventory/SPEC.json"))
    ids = [item["id"] for item in spec["inventory"]]
    assert len(ids) == len(set(ids)) == 8
    for item in spec["inventory"]:
        assert item["dir"].startswith("local/runs/")
        assert item["parent_dir"].startswith("local/runs/")
        assert isinstance(item["local_evidence"], str) and item["local_evidence"]
    assert spec["new_fits"] == 0 and spec["new_packages"] == 0
    assert spec["agent_uploads"] == 0 and spec["desktop_writes"] == 0
    assert spec["rule_source"] == "configs/slot_screen/SPEC.json"


def test_inventory_measured_set_covers_every_declared_parent():
    spec = read_json(Path("configs/slot_inventory/SPEC.json"))
    measured = set(spec["measured_packages"])
    for item in spec["inventory"]:
        assert item["parent_dir"] in measured, item["id"]


def test_every_inventory_entry_documents_its_evidence_limit():
    spec = read_json(Path("configs/slot_inventory/SPEC.json"))
    for item in spec["inventory"]:
        assert isinstance(item["local_evidence_limit"], str) and len(item["local_evidence_limit"]) > 20
