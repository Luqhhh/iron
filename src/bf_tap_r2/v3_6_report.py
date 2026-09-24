"""Generate the V3.6 private summary and the public-facing result numbers.

The function is intentionally read-only with respect to model runs.  It reads
the frozen fixed ledger and complete-development predictions, then reports the
A-relative development composition.  It never fits a model or writes a package.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .v3_6_reference import build_a_development_replay
from .v3_6_selection import (
    evaluate_global_composition,
    read_complete_records,
    select_shortlist,
)

RUN_ROOT = Path("local/runs/round2-v3.6-loss-training-and-numeric-encoding")


def _line_target_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for record in records:
        line = str(record.get("line"))
        target = str(record.get("target"))
        out.setdefault(line, {})
        out[line][target] = out[line].get(target, 0) + 1
    return out


def build_v36_summary(root: Path | str, *, fixed_dir: Path, complete_dir: Path,
                      max_per_line_per_target: int = 4) -> dict[str, Any]:
    root = Path(root)
    fixed_ledger = Path(fixed_dir) / "fit_ledger.jsonl"
    fixed_records = read_complete_records(fixed_ledger)
    fixed_counts = _line_target_counts(fixed_records)
    expected_counts = {
        "O": {"tap_iron": 32, "tap_time_len": 32},
        "D": {"tap_iron": 32, "tap_time_len": 64},
        "N": {"tap_iron": 16, "tap_time_len": 48},
    }
    if fixed_counts != expected_counts:
        raise ValueError(f"V3.6 fixed ledger counts differ: {fixed_counts}")
    shortlist = select_shortlist(fixed_ledger, max_per_line_per_target=max_per_line_per_target)
    # N-time complete development was stopped under the pre-registered limited
    # budget rule after the fixed batch showed no strong single model.  N-iron
    # was refined because fixed PLE-MLP iron results were fast and promising
    # relative to the rest of the N line, not because they were preselected.
    composition_shortlist = [
        trial for trial in shortlist
        if not (str(trial.get("line")) == "N" and str(trial.get("target")) == "tap_time_len")
    ]
    composition = evaluate_global_composition(
        root,
        Path(complete_dir),
        shortlist=composition_shortlist,
        max_new_experts=2,
        max_new_weight=0.5,
    )
    a_replay = build_a_development_replay(root)
    n_single = {}
    for target in ("tap_iron", "tap_time_len"):
        rows = [record for record in fixed_records if str(record.get("line")) == "N" and str(record.get("target")) == target]
        rows.sort(key=lambda record: (float(record.get("pooled_wmape", float("inf"))), str(record.get("trial_id"))))
        n_single[target] = {
            "trial_id": str(rows[0]["trial_id"]),
            "pooled_wmape": float(rows[0]["pooled_wmape"]),
            "structure": str(rows[0]["trial"].get("structure")),
            "capacity": str(rows[0]["trial"].get("capacity_name")),
            "training_setting": str(rows[0]["trial"].get("training_setting")),
        } if rows else None
    delta = float(composition["delta_A_dev"])
    trigger = bool(
        composition["both_complete_splits_positive"]
        and delta >= 0.02
    )
    summary: dict[str, Any] = {
        "version": "round2-v3.6-loss-training-and-numeric-encoding",
        "fixed_batch": {
            "total": 224,
            "counts": fixed_counts,
            "ledger": str(fixed_ledger),
            "complete": fixed_counts == expected_counts,
        },
        "a_development_replay": {
            "a_dev_package_score": float(a_replay["a_dev_package_score"]),
            "v35_dev_package_score": float(a_replay["v35_dev_package_score"]),
            "delta_v35_vs_a_dev": float(a_replay["delta_v35_vs_a_dev"]),
            "objects": a_replay["objects"],
        },
        "complete_development_composition": composition,
        "conditional_extension": {
            "triggered": trigger,
            "mean_complete_split_delta_A_required": 0.02,
            "observed_mean_complete_split_delta_A": delta,
            "both_complete_splits_positive": bool(composition["both_complete_splits_positive"]),
            "reason": (
                "mean complete-split gain below 0.02 and local working gate 96.25 not reached"
                if not trigger
                else "pre-registered trigger satisfied"
            ),
        },
        "local_working_gate": {
            "threshold": 96.25,
            "candidate_dev_package": float(composition["candidate_package_score"]),
            "passed": bool(composition["candidate_package_score"] >= 96.25),
        },
        "final_outer": {
            "outer_seed": 23003,
            "consumed": False,
            "reason": "development candidate did not pass local working gate or conditional-extension trigger",
        },
        "n_line": {
            "status": "STOPPED_AFTER_FIXED_BATCH_NO_STRONG_SINGLE_MODEL",
            "best_fixed_coarse_single": n_single,
            "complete_development_time_refined": False,
            "reason": "N time single models were materially weaker than A_dev and O/D candidates; limited-budget stop rule applied",
        },
        "ple_mlp_correction": {
            "description": "PLE bins are fitted on the training part and applied to the same raw numerical representation for PLE-MLP trials.",
            "superseded_records_preserved": True,
            "n_ple_fixed_rerun_complete": True,
        },
        "agent_uploads": 0,
        "new_platform_packages": 0,
    }
    return summary


def write_v36_summary(root: Path | str, output: Path, **kwargs: Any) -> dict[str, Any]:
    summary = build_v36_summary(root, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


__all__ = ["RUN_ROOT", "build_v36_summary", "write_v36_summary"]
