"""Structural checks for prospective research decisions, not model promotion.

The record deliberately separates discovery from attribution and local evidence
from platform feedback. It cannot validate score provenance or release a model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SCHEMA = "research-decision-v1"
PHASES = {"discovery", "attribution"}
STATUSES = {"planned", "observed"}
SCOPES = {"exact_recipe", "tested_subfamily", "proven_fixed_family", "unresolved"}
ACTIONS = {"pending", "continue", "hold_for_feedback", "stop_exact_recipe", "exclude_proven_family"}
FORBIDDEN_TIME_KEYS = {"time_budget", "time_limit_seconds", "wall_clock_cap", "max_runtime_seconds", "projected_time_gate"}


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _strings(value):
    return isinstance(value, list) and bool(value) and all(_nonempty(item) for item in value)


def validate_record(record):
    """Return human-readable structural errors; make no statistical claims."""
    errors = []
    if not isinstance(record, dict):
        return ["record must be an object"]

    def check_time_keys(value, path="record"):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in FORBIDDEN_TIME_KEYS:
                    errors.append(f"{path}.{key}: prospective wall-clock gate is not allowed")
                check_time_keys(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                check_time_keys(item, f"{path}[{index}]")

    check_time_keys(record)
    if record.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    for key in ("record_id", "hypothesis"):
        if not _nonempty(record.get(key)):
            errors.append(f"{key} must be nonempty text")
    phase, status = record.get("phase"), record.get("status")
    if phase not in PHASES:
        errors.append(f"phase must be one of {sorted(PHASES)}")
    if status not in STATUSES:
        errors.append(f"status must be one of {sorted(STATUSES)}")
    if record.get("time_policy") != "no_wall_clock_rejection":
        errors.append("time_policy must be no_wall_clock_rejection")

    context = record.get("context")
    if not isinstance(context, dict):
        errors.append("context must be an object")
        context = {}
    for key in ("dataset", "split", "target", "metric", "reference", "source_revision", "data_boundary"):
        if not _nonempty(context.get(key)):
            errors.append(f"context.{key} must be nonempty text")
    changed = record.get("changed_factors")
    if not _strings(changed):
        errors.append("changed_factors must be a nonempty list of text")
        changed = []
    if not _strings(record.get("held_constant")):
        errors.append("held_constant must be a nonempty list of text")
    if phase == "attribution" and len(changed) > 1 and not _strings(record.get("attribution_controls")):
        errors.append("multi-factor attribution requires explicit attribution_controls")

    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        errors.append("evidence must be a list")
        evidence = []
    if status == "observed" and not evidence:
        errors.append("observed decisions require evidence")
    if status == "planned" and evidence:
        errors.append("planned decisions cannot contain observed evidence")
    for index, item in enumerate(evidence):
        prefix = f"evidence[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        kind = item.get("kind")
        if kind not in {"local", "platform", "analytic_bound"}:
            errors.append(f"{prefix}.kind is invalid")
        for key in ("source", "effect_metric", "limitations"):
            if not _nonempty(item.get(key)):
                errors.append(f"{prefix}.{key} must be nonempty text")
        effect = item.get("effect")
        if isinstance(effect, bool) or not isinstance(effect, (int, float)) or not math.isfinite(effect):
            errors.append(f"{prefix}.effect must be finite numeric")
        if kind == "platform":
            if item.get("verification") not in {"user_reported", "independent_receipt"}:
                errors.append(f"{prefix}.verification must distinguish report from receipt")
            if item.get("verification") == "independent_receipt" and not _nonempty(item.get("receipt_reference")):
                errors.append(f"{prefix}.receipt_reference required for independent receipt")
        elif "verification" in item:
            errors.append(f"{prefix}.verification is reserved for platform evidence")

    decision = record.get("decision")
    if not isinstance(decision, dict):
        errors.append("decision must be an object")
        decision = {}
    action, scope = decision.get("action"), decision.get("scope")
    if action not in ACTIONS:
        errors.append(f"decision.action must be one of {sorted(ACTIONS)}")
    if scope not in SCOPES:
        errors.append(f"decision.scope must be one of {sorted(SCOPES)}")
    for key in ("rationale", "reopen_condition", "next_check"):
        if not _nonempty(decision.get(key)):
            errors.append(f"decision.{key} must be nonempty text")
    if status == "planned" and action != "pending":
        errors.append("planned decision must remain pending")
    if action == "stop_exact_recipe" and scope != "exact_recipe":
        errors.append("stop_exact_recipe can only claim exact_recipe scope")
    if action == "exclude_proven_family":
        if scope != "proven_fixed_family" or not _nonempty(decision.get("proof_source")):
            errors.append("family exclusion requires proven_fixed_family scope and proof_source")
        if not any(isinstance(item, dict) and item.get("kind") == "analytic_bound" for item in evidence):
            errors.append("family exclusion requires analytic_bound evidence")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="JSON decision record to check without executing experiments")
    args = parser.parse_args()
    try:
        record = json.loads(args.record.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    errors = validate_record(record)
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")
    print(f"structure valid: {args.record} (not an evidence or release audit)")


if __name__ == "__main__":
    main()
