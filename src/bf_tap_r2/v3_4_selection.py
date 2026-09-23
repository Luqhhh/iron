"""V3.4 complete diversity refinement selection.

The task book limits full-development refinement to ten items per target:
five strong single models, three candidates that actually improve a simple L1
mix, and two structurally different candidates.  This module implements that
selection as a deterministic pure function so it can be replayed before any
outer-validation score is read.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import wmape
from .v3_4_composition import simple_mix_alphas


def canonical_structure(candidate: Mapping[str, Any]) -> tuple:
    """Return a label-free structural key for diversity accounting."""
    trial = candidate.get("trial", candidate)
    kind = str(trial.get("kind", trial.get("family", "")))
    params = dict(trial.get("parameters", {}))
    if kind in {"ebm", "ebm_boundary", "ebm_base"}:
        return (
            "ebm_boundary",
            str(trial.get("target_transform")),
            int(params.get("max_bins", -1)),
            int(params.get("min_samples_leaf", -1)),
            int(params.get("interactions", -1)),
            int(params.get("max_interaction_bins", -1)),
        )
    if kind == "ebm_residual":
        base = dict(params.get("base_trial", {}))
        corrector = dict(params.get("corrector", {}))
        return (
            "ebm_residual",
            str(base.get("trial_id", "")),
            str(params.get("coordinate", "")),
            str(corrector.get("kind", "")),
            str(corrector.get("name", "")),
        )
    if kind == "global_spout_shrink":
        parent = dict(params.get("parent_trial", {}))
        return (
            "global_spout_shrink",
            str(parent.get("trial_id", "")),
            float(params.get("local_l2_multiplier", -1.0)),
            float(params.get("beta", -1.0)),
        )
    return (kind, str(trial.get("target")), str(trial.get("target_transform")), str(params))


def _mean_wmape(y_by_seed: Mapping[str, np.ndarray],
                prediction_by_seed: Mapping[str, np.ndarray]) -> float:
    seeds = sorted(y_by_seed)
    if sorted(prediction_by_seed) != seeds:
        raise ValueError("Candidate does not cover every development seed")
    return float(np.mean([
        wmape(np.asarray(y_by_seed[seed], dtype=float), np.asarray(prediction_by_seed[seed], dtype=float))
        for seed in seeds
    ]))


def _mix_improvement(y_by_seed: Mapping[str, np.ndarray],
                     l1_by_seed: Mapping[str, np.ndarray],
                     candidate_by_seed: Mapping[str, np.ndarray],
                     l1_mean: float,
                     alphas: Sequence[float]) -> dict[str, Any]:
    scan = simple_mix_alphas(y_by_seed, l1_by_seed, candidate_by_seed, alphas)
    best = min(scan, key=lambda row: (float(row["mean_wmape"]), float(row["alpha"])))
    return {
        "best_alpha": float(best["alpha"]),
        "best_mix_wmape": float(best["mean_wmape"]),
        "improvement_vs_l1": float(l1_mean - best["mean_wmape"]),
    }


def select_diverse_refinement(candidates: Sequence[Mapping[str, Any]],
                              y_by_seed: Mapping[str, np.ndarray],
                              l1_by_seed: Mapping[str, np.ndarray],
                              *,
                              target: str,
                              max_per_target: int = 10,
                              singles_count: int = 5,
                              combo_count: int = 3,
                              structure_count: int = 2,
                              alphas: Sequence[float] = (0.10, 0.25, 0.50)) -> dict[str, Any]:
    """Select up to ten refined items for one target.

    Each candidate mapping must contain ``trial_id``, ``trial`` and
    ``prediction_by_seed``.  The selection uses only development labels and
    predictions supplied by the caller.
    """
    seeds = sorted(y_by_seed)
    if sorted(l1_by_seed) != seeds:
        raise ValueError("L1 development predictions do not cover every seed")
    pool = [dict(candidate) for candidate in candidates if candidate.get("target") == target]
    if not pool:
        raise ValueError(f"No candidates for {target}")
    for candidate in pool:
        if sorted(candidate["prediction_by_seed"]) != seeds:
            raise ValueError(f"Candidate {candidate['trial_id']} lacks a development seed")
    scores = {str(candidate["trial_id"]): _mean_wmape(y_by_seed, candidate["prediction_by_seed"])
              for candidate in pool}
    by_id = {str(candidate["trial_id"]): candidate for candidate in pool}
    l1_mean = float(np.mean([
        wmape(np.asarray(y_by_seed[seed], dtype=float), np.asarray(l1_by_seed[seed], dtype=float))
        for seed in seeds
    ]))

    rank = sorted(by_id, key=lambda tid: (scores[tid], tid))
    selected: list[str] = []
    roles: dict[str, str] = {}
    reasons: dict[str, dict[str, Any]] = {}

    def take(tid: str, role: str, meta: dict[str, Any]) -> None:
        if tid in roles or len(selected) >= int(max_per_target):
            return
        selected.append(tid)
        roles[tid] = role
        reasons[tid] = dict(meta)

    for tid in rank[: max(0, int(singles_count))]:
        take(tid, "single_strong", {"mean_wmape": scores[tid]})

    combo_rows = []
    for tid in rank:
        if tid in roles:
            continue
        improvement = _mix_improvement(y_by_seed, l1_by_seed,
                                       by_id[tid]["prediction_by_seed"], l1_mean, alphas)
        if improvement["improvement_vs_l1"] > 1e-12:
            combo_rows.append((improvement["improvement_vs_l1"], scores[tid], tid, improvement))
    combo_rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    for _, _, tid, improvement in combo_rows[: max(0, int(combo_count))]:
        take(tid, "simple_mix_effective", {"mean_wmape": scores[tid], **improvement})

    selected_lines = {str(by_id[tid]["trial"]["line"]) for tid in selected}
    selected_structures = {canonical_structure(by_id[tid]) for tid in selected}
    # First prefer a line/kind not yet represented, then a genuinely new
    # structural key.  This is the "different structure" part.
    structure_taken = 0
    for requirement in ("new_line", "new_structure"):
        for tid in rank:
            if structure_taken >= int(structure_count) or tid in roles:
                continue
            candidate = by_id[tid]
            line = str(candidate["trial"]["line"])
            structure = canonical_structure(candidate)
            if requirement == "new_line":
                if line in selected_lines:
                    continue
            else:
                if structure in selected_structures:
                    continue
            take(tid, "different_structure",
                 {"mean_wmape": scores[tid], "line": line, "structure": structure,
                  "reason": requirement})
            selected_lines.add(line)
            selected_structures.add(structure)
            structure_taken += 1
            if structure_taken >= int(structure_count):
                break
        if structure_taken >= int(structure_count):
            break
    # Keep the schedule bounded and deterministic; do not fill empty slots
    # merely to reach ten.
    selected = selected[: int(max_per_target)]
    return {
        "target": target,
        "selected": [
            {
                "trial_id": tid,
                "role": roles[tid],
                "mean_wmape": scores[tid],
                "line": str(by_id[tid]["trial"]["line"]),
                "structure": canonical_structure(by_id[tid]),
                **reasons[tid],
            }
            for tid in selected
        ],
        "selected_trial_ids": selected,
        "l1_mean_wmape": l1_mean,
        "candidate_count": len(pool),
        "role_counts": dict(Counter(roles[tid] for tid in selected)),
    }
