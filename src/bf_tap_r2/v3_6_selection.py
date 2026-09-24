"""V3.6 fixed-batch shortlist, complete-development refinement, and A-relative
development composition diagnostics.

This module deliberately sits outside the frozen runner source-hash set: it
selects work and analyses completed predictions, but it never changes model
fitting or trial identity.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import wmape
from .v3_4_composition import constrained_forward_select
from .v3_6_reference import DEVELOPMENT_SEEDS, TARGETS, build_reference_vectors
from .v3_run import load_training_frame

RUN_ROOT = Path("local/runs/round2-v3.6-loss-training-and-numeric-encoding")


def read_complete_records(ledger: Path) -> list[dict]:
    out: list[dict] = []
    if not ledger.exists():
        return out
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "complete":
            out.append(event)
    return out


def select_shortlist(coarse_ledger: Path, *, max_per_line_per_target: int = 4) -> list[dict]:
    """Deterministically select the best coarse trial per line and target."""
    records = read_complete_records(coarse_ledger)
    selected: list[dict] = []
    for line in ("O", "D", "N"):
        for target in TARGETS:
            rows = [
                record for record in records
                if str(record.get("line")) == line and str(record.get("target")) == target
                and isinstance(record.get("trial"), Mapping)
            ]
            rows.sort(key=lambda row: (float(row.get("pooled_wmape", float("inf"))), str(row.get("trial_id"))))
            for record in rows[: max(0, int(max_per_line_per_target))]:
                trial = deepcopy(dict(record["trial"]))
                trial["_coarse_pooled_wmape"] = float(record["pooled_wmape"])
                selected.append(trial)
    return selected


def _trial_ids_by_target(trials: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    out = {target: [] for target in TARGETS}
    for trial in trials:
        target = str(trial["target"])
        if target not in out:
            raise ValueError(f"Unexpected target in shortlist: {target}")
        out[target].append(str(trial["trial_id"]))
    return out


def _load_development_predictions(root: Path, complete_dir: Path, target: str,
                                  seed: str, trial_id: str) -> np.ndarray:
    path = complete_dir / f"seed-{seed}" / f"pred-{trial_id}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    values = np.load(path)
    train = load_training_frame(root)
    if values.shape != (len(train),):
        raise ValueError(f"Complete-development prediction shape mismatch: {path}")
    return values.astype(float, copy=False)


def evaluate_composition_for_target(root: Path, complete_dir: Path, *,
                                    candidate_ids: Sequence[str], target: str,
                                    reference: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
                                    max_new_experts: int = 2,
                                    max_new_weight: float = 0.5) -> dict[str, Any]:
    train = load_training_frame(root)
    y_by_seed = {seed: train[target].to_numpy(dtype=float) for seed in DEVELOPMENT_SEEDS}
    base_by_seed = {seed: np.asarray(reference["A"][target][seed], dtype=float) for seed in DEVELOPMENT_SEEDS}
    candidate_pred = {
        str(trial_id): {
            seed: _load_development_predictions(root, complete_dir, target, seed, str(trial_id))
            for seed in DEVELOPMENT_SEEDS
        }
        for trial_id in candidate_ids
    }
    # Rank candidates by their own mean development WMAPE before the constrained
    # combination; the composition order itself is solved by the LP below.
    candidate_scores = {
        name: float(np.mean([wmape(y_by_seed[seed], values[seed]) for seed in DEVELOPMENT_SEEDS]))
        for name, values in candidate_pred.items()
    }
    ordered_ids = sorted(candidate_pred, key=lambda name: (candidate_scores[name], name))
    names = ["A_dev", *ordered_ids]
    p_by_seed = {
        seed: np.column_stack([base_by_seed[seed], *[candidate_pred[name][seed] for name in ordered_ids]])
        for seed in DEVELOPMENT_SEEDS
    }
    selection = constrained_forward_select(
        y_by_seed,
        p_by_seed,
        names=names,
        l1_index=0,
        candidate_indices=list(range(1, len(names))),
        max_new_experts=int(max_new_experts),
        max_new_weight=float(max_new_weight),
    )
    weights = np.asarray(selection["weights"], dtype=float)
    selected_experts = [str(name) for name in selection["new_experts"]]
    selected_columns = ["A_dev", *selected_experts]
    per_seed_wmape: dict[str, float] = {}
    per_seed_package: dict[str, float] = {}
    for seed in DEVELOPMENT_SEEDS:
        columns = [base_by_seed[seed]] + [candidate_pred[name][seed] for name in selected_experts]
        pred = np.column_stack(columns) @ weights
        per_seed_wmape[seed] = float(wmape(y_by_seed[seed], pred))
        # Target WMAPE on the unchanged side is required for a package score.
    a_per_seed_target = {
        seed: float(wmape(y_by_seed[seed], base_by_seed[seed]))
        for seed in DEVELOPMENT_SEEDS
    }
    # Package score needs both targets.  The caller evaluates one target at a
    # time; this function returns the target component and keeps the A-relative
    # diagnostic explicit.
    mean_wmape = float(np.mean(list(per_seed_wmape.values())))
    a_mean_wmape = float(np.mean(list(a_per_seed_target.values())))
    return {
        "target": target,
        "candidate_ids": list(candidate_ids),
        "single_wmape": candidate_scores,
        "selected_experts": selected_experts,
        "weights": [float(v) for v in weights],
        "new_weight_total": float(selection["new_weight_total"]),
        "objective_mean_wmape": mean_wmape,
        "a_dev_mean_wmape": a_mean_wmape,
        "delta_wmape_vs_a_dev": a_mean_wmape - mean_wmape,
        "per_seed_wmape": per_seed_wmape,
        "a_dev_per_seed_wmape": a_per_seed_target,
        "per_seed_delta_wmape": {
            seed: a_per_seed_target[seed] - per_seed_wmape[seed] for seed in DEVELOPMENT_SEEDS
        },
        "history": selection["history"],
    }


def evaluate_global_composition(root: Path, complete_dir: Path, *,
                                shortlist: Sequence[Mapping[str, Any]],
                                max_new_experts: int = 2,
                                max_new_weight: float = 0.5) -> dict[str, Any]:
    """Compose both targets independently against the recorded A development replay."""
    train = load_training_frame(root)
    reference = build_reference_vectors(root, train)
    ids_by_target = _trial_ids_by_target(shortlist)
    targets: dict[str, Any] = {}
    for target in TARGETS:
        if not ids_by_target[target]:
            raise ValueError(f"No V3.6 shortlist candidates for target {target}")
        targets[target] = evaluate_composition_for_target(
            root,
            complete_dir,
            candidate_ids=ids_by_target[target],
            target=target,
            reference=reference,
            max_new_experts=int(max_new_experts),
            max_new_weight=float(max_new_weight),
        )
    target_wmape = {target: targets[target]["objective_mean_wmape"] for target in TARGETS}
    a_target_wmape = {target: targets[target]["a_dev_mean_wmape"] for target in TARGETS}
    package_score = 100.0 - 100.0 * float(np.mean(list(target_wmape.values())))
    a_package_score = 100.0 - 100.0 * float(np.mean(list(a_target_wmape.values())))
    per_seed_package: dict[str, float] = {}
    a_per_seed_package: dict[str, float] = {}
    for seed in DEVELOPMENT_SEEDS:
        per_seed_package[seed] = 100.0 - 100.0 * float(
            np.mean([targets[target]["per_seed_wmape"][seed] for target in TARGETS])
        )
        a_per_seed_package[seed] = 100.0 - 100.0 * float(
            np.mean([targets[target]["a_dev_per_seed_wmape"][seed] for target in TARGETS])
        )
    both_positive = all(
        per_seed_package[seed] - a_per_seed_package[seed] > 0.0 for seed in DEVELOPMENT_SEEDS
    )
    return {
        "reference": "A_development_replay",
        "a_dev_package_score": float(a_package_score),
        "candidate_package_score": float(package_score),
        "delta_A_dev": float(package_score - a_package_score),
        "both_complete_splits_positive": bool(both_positive),
        "per_seed_candidate_package_score": per_seed_package,
        "per_seed_a_dev_package_score": a_per_seed_package,
        "per_seed_delta_A_dev": {
            seed: float(per_seed_package[seed] - a_per_seed_package[seed])
            for seed in DEVELOPMENT_SEEDS
        },
        "targets": targets,
        "agent_uploads": 0,
    }


__all__ = [
    "RUN_ROOT",
    "evaluate_composition_for_target",
    "evaluate_global_composition",
    "read_complete_records",
    "select_shortlist",
]
