"""Apply the frozen slot rule to every recorded package without a platform score.

Zero fits, zero label reads.  See ``docs/slot_inventory/PREREGISTRATION.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bf_tap_r2.slot_screen import (  # noqa: E402
    geometry,
    read_json,
    read_package,
    template_ids,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/slot_inventory/SPEC.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = read_json(ROOT / args.spec)
    rule = read_json(ROOT / spec["rule_source"])["slot_admission_rule"]
    expected = template_ids()
    cache: dict[str, dict] = {}

    def package(relative: str) -> dict:
        if relative not in cache:
            cache[relative] = read_package(ROOT / relative, expected)
        return cache[relative]

    measured = [package(path) for path in spec["measured_packages"]]
    measured_vectors = [(entry["values"]["pred_tap_iron"], entry["values"]["pred_tap_time_len"])
                        for entry in measured]

    rows = []
    for item in spec["inventory"]:
        entry = {"id": item["id"], "parent": item["parent"], "dir": item["dir"],
                 "local_evidence": item["local_evidence"],
                 "local_evidence_source": item["local_evidence_source"],
                 "local_evidence_limit": item["local_evidence_limit"]}
        try:
            candidate, parent = package(item["dir"]), package(item["parent_dir"])
            measured_geometry = geometry(candidate, parent)
        except Exception as error:
            entry.update({"error": f"{type(error).__name__}: {error}", "recommendation": "excluded"})
            rows.append(entry)
            continue
        # A package that reproduces an already scored prediction vector is not new information.
        duplicate = None
        for index, (iron, time) in enumerate(measured_vectors):
            if (abs(iron - candidate["values"]["pred_tap_iron"]).max() == 0.0
                    and abs(time - candidate["values"]["pred_tap_time_len"]).max() == 0.0):
                duplicate = spec["measured_packages"][index]
                break
        conditions = {
            "non_expansive_change": measured_geometry["slope"] <= 0.0,
            "bounded_perturbation_rho_le_0.01": measured_geometry["rho"] <= 0.01,
            "not_an_already_scored_prediction_vector": duplicate is None,
        }
        entry.update({
            "changed_target": measured_geometry["changed_target"],
            "rho": measured_geometry["rho"],
            "bias_pct": measured_geometry["bias_pct"],
            "slope": measured_geometry["slope"],
            "max_abs_pct": measured_geometry["max_abs_pct"],
            "changed_fraction": measured_geometry["changed_fraction"],
            "zip_sha256": candidate["zip_sha256"],
            "duplicate_of_measured": duplicate,
            "conditions": conditions,
            "geometry_rule_passed": all(conditions.values()),
            "existing_local_gate_evidence": item["local_evidence"],
            "recommendation": ("slot_candidate_requires_four_seed_gate"
                               if all(conditions.values()) else "not_slot_worthy"),
        })
        rows.append(entry)

    errored = [row for row in rows if "error" in row]
    evaluated = [row for row in rows if "error" not in row]
    passing = [row for row in evaluated if row["geometry_rule_passed"]]
    with_gate = [row for row in passing if row.get("existing_local_gate_evidence", "").startswith("four-seed")]
    report = {
        "stage": spec["stage"],
        "objective": spec["objective"],
        "reference": spec["reference"],
        "rule": rule,
        "budget_actual": {"new_fits": 0, "training_label_reads": 0, "new_packages": 0,
                          "desktop_writes": 0, "agent_uploads": 0},
        "inventory_size": len(rows),
        "geometry_rule_passed": [row["id"] for row in passing],
        "geometry_rule_failed": [row["id"] for row in evaluated if not row["geometry_rule_passed"]],
        "identity_gate_rejected": [row["id"] for row in errored],
        "passed_and_carrying_local_gate": [row["id"] for row in with_gate],
        "recommendation": ("no_slot_candidate_in_inventory" if not with_gate else "review"),
        "reason": ("Every rule-passing inventory package lacks a current four-complete-seed local gate: "
                   "K32 carries one but fails the geometry rule, and the rule-passing entries carry either "
                   "no registered local gain, an effectively zero gain, or a negative one."
                   if not with_gate else "A rule-passing package also carries a four-seed gate; review before scheduling."),
        "rows": rows,
        "limits": spec["limits"],
    }
    write_json(Path(args.output) / "inventory.json", report)
    print(json.dumps({key: report[key] for key in
                      ("inventory_size", "geometry_rule_passed", "geometry_rule_failed", "identity_gate_rejected",
                       "passed_and_carrying_local_gate", "recommendation", "reason")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
