#!/usr/bin/env python3
"""Append-only receipt exporter for an existing completed v0.31 run.

It does not rerun derivation and does not overwrite frozen files.  The script
materializes the suggested diagnostic views (``source_inventory.json``,
``pool_diagnostics.json`` and an awaiting ``platform_feedback.json``) from the
executed run's manifest and frozen worker outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from bf_tap.artifacts import atomic_write_json, file_sha256


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _frames_changed(left: pd.DataFrame, right: pd.DataFrame, columns: tuple[str, ...]):
    if left.sample_id.tolist() != right.sample_id.tolist():
        right = right.set_index("sample_id").loc[left.sample_id].reset_index()
    return {column: int((left[column] != right[column]).sum()) for column in columns}


def export(run: Path):
    manifest = read_json(run / "manifest.json")
    packages = read_json(run / "packages_frozen_before_feedback.json")
    if not (run / "completion.json").is_file():
        raise ValueError("completed v0.31 run required")

    attachment_names = sorted(manifest["evidence"])
    source_inventory = {
        "status": "FROZEN_SOURCE_INVENTORY_FROM_EXECUTED_MANIFEST",
        "manifest_sha256": file_sha256(run / "manifest.json"),
        "code_commit": manifest["code_commit"],
        "source_runs": manifest["source_runs"],
        "source_files": manifest["sources"],
        "evidence": manifest["evidence"],
        "candidate_targets": manifest["candidate_targets"],
        "worker_registration": manifest["worker_registration"],
        "certified_v029_attachment_evidence": {
            name: manifest["evidence"][name] for name in attachment_names
            if "v29_attachment_" in name and not name.endswith("_metadata")
        },
        "new_model_or_tree_fits": 0,
        "new_preprocessor_or_calibration_fits": 0,
    }
    if not (run / "source_inventory.json").exists():
        atomic_write_json(run / "source_inventory.json", source_inventory)

    parent_final = pd.read_csv(
        Path(manifest["source_runs"]["v30"]) / "submissions" / "V30A_OOB_BOTH_TARGETS" / "result.csv",
        dtype=str, keep_default_na=False,
    )
    entries = []
    for candidate in ("A", "B"):
        for slot in range(6, 13):
            view = read_json(run / "pool_views" / f"{candidate}_{slot}.json")
            receipt = read_json(run / "worker_predictions" / candidate / f"{slot}.npz.json")
            with np.load(run / "worker_predictions" / candidate / f"{slot}.npz", allow_pickle=False) as source:
                median = source["median"]
                legacy = source["legacy_median"]
            prediction = pd.read_csv(run / "predictions" / f"{slot}_{view['candidate']}.csv", dtype=str, keep_default_na=False)
            if slot == 12:
                changed_vs_parent = _frames_changed(
                    prediction, parent_final, ("pred_tap_iron", "pred_tap_time_len"),
                )
            else:
                changed_vs_parent = None
            entry = {
                "candidate": view["candidate"],
                "candidate_key": candidate,
                "slot": slot,
                "target": view["target"],
                "unit": view["unit"],
                "attachment_sha256": view["attachment_sha256"],
                "legacy_equal_tree_switch_back_exact": view["legacy_equal_tree_switch_back_exact"],
                "changed_from_legacy_count": view["changed_from_legacy_count"],
                "lower_boundary": receipt["lower_boundary"],
                "upper_boundary": receipt["upper_boundary"],
                "nonfinite_or_negative": receipt["nonfinite_or_negative_pooled"],
                "diagnostics": view["diagnostics"],
            }
            if slot == 12:
                entry["full_final_prediction_changed_rows_vs_parent"] = changed_vs_parent
            entries.append(entry)
    aggregate = {}
    for candidate in ("A", "B"):
        part = [entry for entry in entries if entry["candidate_key"] == candidate and entry["slot"] != 12]
        aggregate[candidate] = {
            "development_changed_from_legacy_min": int(min(entry["changed_from_legacy_count"] for entry in part)),
            "development_changed_from_legacy_max": int(max(entry["changed_from_legacy_count"] for entry in part)),
            "fallback_tree_fraction_mean_max": float(max(entry["diagnostics"]["fallback_tree_count_mean"] for entry in part) / 256.0),
            "fallback_mass_fraction_mean_max": float(max(entry["diagnostics"]["fallback_mass_fraction_mean"] for entry in part)),
            "S_occurrence_total_min": int(min(entry["diagnostics"]["S_occurrence_total_min"] for entry in part)),
            "S_occurrence_total_max": int(max(entry["diagnostics"]["S_occurrence_total_max"] for entry in part)),
            "distinct_selected_rows_min": int(min(entry["diagnostics"]["distinct_selected_rows_min"] for entry in part)),
            "distinct_selected_rows_max": int(max(entry["diagnostics"]["distinct_selected_rows_max"] for entry in part)),
        }
    pool_diagnostics = {
        "status": "POOL_DIAGNOSTICS_FROM_FROZEN_WORKER_OUTPUTS",
        "manifest_sha256": file_sha256(run / "manifest.json"),
        "S_definition": "sum_b_n_b_occurrences_not_new_samples",
        "entries": entries,
        "aggregate": aggregate,
        "packages": packages["receipts"],
        "final_parent_identity_note": "slot 12 changed-vs-parent values are present when a frozen parent_12.csv view is available",
    }
    if not (run / "pool_diagnostics.json").exists():
        atomic_write_json(run / "pool_diagnostics.json", pool_diagnostics)

    feedback_path = run / "platform_feedback.json"
    if not feedback_path.exists():
        atomic_write_json(feedback_path, {
            "status": "AWAITING_USER_REPORTED_PLATFORM_FEEDBACK",
            "candidate_order": ["A", "B"],
            "platform_upload_budget": 2,
            "agent_platform_uploads": 0,
            "user_reported_feedback": [],
            "source": "independent_append_only_file",
            "note": "The two frozen packages are prepared for one explicit A->B platform test each; no agent upload is authorized or attempted.",
        })
    return source_inventory, pool_diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    arguments = parser.parse_args()
    export(arguments.run.resolve())


if __name__ == "__main__":
    main()
