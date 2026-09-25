"""Round2 V5 zero-fit diagnostics: error covariance, composition shape, backtest.

Three questions, all answered from recorded out-of-fold predictions:

1. **Covariance / admissibility** — which library members are decorrelated from
   the current column *and* bounded in accuracy loss?  Low correlation alone
   selects noise (the least correlated coarse candidates are about twice as
   bad as the base and never improve under a nested two-fold alpha test).
2. **Composition shape** — what does each released member actually contribute,
   and what happens when a near-zero member is dropped and replaced?
3. **Backtest** — would the V5 admission rule have refused the candidates that
   already failed on the platform?

Nothing here fits a model, promotes a candidate, packages or uploads.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .v5_library import CandidateLibrary, ColumnReference, LibraryEntry
from .v5_resolution import (
    admit,
    blend_record,
    paired_summary,
    seed_level_summary,
    wmape,
)
from .v5_spec import V5Spec

__all__ = [
    "BACKTEST_SOURCES",
    "AdmissibilityRecord",
    "admissibility_table",
    "best_admissible",
    "leave_one_out",
    "replacement_sweep",
    "backtest_gains",
    "backtest_discrimination",
]

#: Provenance of the recorded per-cell gains used by the admission backtest.
#: Each entry names the historical unit, the artifact that stores it, and the
#: matching trial id.  Gates are never re-derived from these numbers; they only
#: replay the V5 decision function on already-published evidence.
BACKTEST_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "candidate": "V42_iron_N2_quarter",
        "role": "known_bad",
        "path": "local/runs/round2-v4.2-structure-search/n2-seed-init-v2/coarse_stage_summary.json",
        "trial_id": "N-N2-tap_iron-SEED_INIT_V2",
        "folds": [0, 1],
        "platform_note": "user-reported platform 96.2533, -0.0201 against V36",
    },
    {
        "candidate": "V42_iron_N2_quarter_full_coverage",
        "role": "replication_control",
        "path": "local/runs/round2-v4.2-structure-search/n2-seed-init-v2/coarse_summary.json",
        "trial_id": "N-N2-tap_iron-SEED_INIT_V2",
        "folds": [0, 1, 2, 3, 4],
        "platform_note": "same candidate at complete ten-cell coverage",
    },
    {
        "candidate": "NODE_per_depth_N4",
        "role": "known_bad",
        "path": "local/runs/round2-v4.4-mechanism-completion/N4-tap_iron/coarse_stage_summary.json",
        "trial_id": "N-N4-tap_iron-V4_4",
        "folds": [0, 1],
    },
    {
        "candidate": "ODST_core_N5",
        "role": "known_bad",
        "path": "local/runs/round2-v4.4-mechanism-completion/N5-tap_iron/coarse_stage_summary.json",
        "trial_id": "N-N5-tap_iron-V4_4",
        "folds": [0, 1],
    },
    {
        "candidate": "TabR_full_fusion_R4",
        "role": "known_bad",
        "path": "local/runs/round2-v4.4-mechanism-completion/r-coarse-r2/coarse_summary.json",
        "trial_id": "R-R4-tap_iron",
        "folds": [0, 1],
    },
)


@dataclass
class AdmissibilityRecord:
    key: str
    name: str
    target: str
    family: str
    source: str
    packagability: str
    convention: str
    residual_correlation: float
    single_wmape: float
    base_wmape: float
    accuracy_ratio: float
    admissible: bool
    reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "target": self.target,
            "family": self.family,
            "source": self.source,
            "packagability": self.packagability,
            "convention": self.convention,
            "residual_correlation": float(self.residual_correlation),
            "single_wmape": float(self.single_wmape),
            "base_wmape": float(self.base_wmape),
            "accuracy_ratio": float(self.accuracy_ratio),
            "admissible": bool(self.admissible),
            "reasons": list(self.reasons),
        }


def _entry_vectors(entry: LibraryEntry, target: str) -> dict[int, np.ndarray]:
    return {int(seed): np.asarray(values, dtype=float) for seed, values in entry.vectors.items()}


def admissibility_table(y: Sequence[float], base_by_seed: Mapping[int, np.ndarray],
                        entries: Iterable[LibraryEntry], spec: V5Spec,
                        skip_keys: Iterable[str] = ()) -> list[AdmissibilityRecord]:
    """Correlation and bounded-accuracy-loss screen for every candidate column."""
    actual = np.asarray(y, dtype=float)
    correlation_max = float(spec.admissibility["residual_correlation_max"])
    ratio_max = float(spec.admissibility["single_wmape_ratio_max"])
    skip = set(skip_keys)
    records: list[AdmissibilityRecord] = []
    seeds = sorted(int(s) for s in base_by_seed)
    base_mean = float(np.mean([wmape(actual, np.asarray(base_by_seed[s], dtype=float)) for s in seeds]))
    for entry in entries:
        if entry.key in skip or not set(seeds).issubset(entry.vectors):
            continue
        vectors = {s: np.asarray(entry.vectors[s], dtype=float) for s in seeds}
        if any(not np.isfinite(v).all() for v in vectors.values()):
            continue
        correlations = [
            float(np.corrcoef(actual - np.asarray(base_by_seed[s], dtype=float), actual - vectors[s])[0, 1])
            for s in seeds
        ]
        single = float(np.mean([wmape(actual, vectors[s]) for s in seeds]))
        correlation = float(np.mean(correlations))
        ratio = float(single / base_mean) if base_mean > 0 else float("inf")
        reasons: list[str] = []
        if not np.isfinite(correlation) or correlation > correlation_max:
            reasons.append("correlation_above_max")
        if ratio > ratio_max:
            reasons.append("accuracy_loss_above_max")
        records.append(AdmissibilityRecord(
            key=entry.key,
            name=entry.name,
            target=entry.target,
            family=entry.family,
            source=entry.source,
            packagability=entry.packagability,
            convention=entry.convention,
            residual_correlation=correlation,
            single_wmape=single,
            base_wmape=base_mean,
            accuracy_ratio=ratio,
            admissible=not reasons,
            reasons=reasons,
        ))
    records.sort(key=lambda r: (not r.admissible, r.residual_correlation, r.key))
    return records


def best_admissible(records: Sequence[AdmissibilityRecord], limit: int | None = None) -> list[AdmissibilityRecord]:
    admissible = [r for r in records if r.admissible]
    return admissible if limit is None else admissible[: int(limit)]


def leave_one_out(reference: ColumnReference, y: Sequence[float], target: str,
                  folds: Mapping[int, np.ndarray]) -> dict[str, Any]:
    """Drop each released member in turn and renormalise the remaining weights."""
    actual = np.asarray(y, dtype=float)
    members = ["A_dev", *reference.members[target]]
    weights = [float(v) for v in reference.weights[target]]
    seeds = sorted(int(s) for s in reference.base)
    # The other target's column is untouched, so score = 100 - 50*(W_this + const)
    # and a WMAPE change in this column moves the score by -50 * delta.
    base_wmape = {s: wmape(actual, reference.base_for(target, s)) for s in seeds}
    drops: list[dict[str, Any]] = []
    for index, name in enumerate(members):
        keep = [i for i in range(len(members)) if i != index]
        keep_weights = np.asarray([weights[i] for i in keep], dtype=float)
        keep_weights = keep_weights / keep_weights.sum()
        seed_wmape: dict[str, float] = {}
        seed_delta: dict[str, float] = {}
        for seed in seeds:
            blended = np.zeros(len(actual), dtype=float)
            for weight, member_index in zip(keep_weights, keep):
                blended = blended + float(weight) * reference.member_vector(target, seed, members[member_index])
            blended = np.maximum(blended, 0.0)
            seed_wmape[str(seed)] = wmape(actual, blended)
            seed_delta[str(seed)] = float(50.0 * (base_wmape[seed] - seed_wmape[str(seed)]))
        drops.append({
            "member": name,
            "weight": float(weights[index]),
            "per_seed_wmape": seed_wmape,
            "per_seed_score_delta": seed_delta,
            "mean_score_delta": float(np.mean(list(seed_delta.values()))),
        })
    drops.sort(key=lambda r: r["mean_score_delta"])
    return {
        "target": target,
        "members": members,
        "weights": weights,
        "base_per_seed_wmape": {str(s): base_wmape[s] for s in seeds},
        "drops": drops,
    }


def replacement_sweep(y: Sequence[float], folds: Mapping[int, np.ndarray],
                      reference: ColumnReference, library: CandidateLibrary,
                      target: str, spec: V5Spec,
                      candidates: Sequence[str], drop_member: str | None = None,
                      limit: int = 25) -> dict[str, Any]:
    """Replace (never re-weight) a member and re-evaluate under the nested protocol.

    The base column for the sweep is the released column with ``drop_member``
    removed and the remaining weights renormalised.  Each candidate is then
    blended on top of that base with a weight selected on the fit seed only.
    """
    actual = np.asarray(y, dtype=float)
    members = ["A_dev", *reference.members[target]]
    weights = [float(v) for v in reference.weights[target]]
    drop_name = drop_member or members[-1]
    if drop_name not in members:
        raise KeyError(f"V5 replacement sweep: unknown member {drop_name!r}")
    keep = [i for i in range(len(members)) if members[i] != drop_name]
    keep_weights = np.asarray([weights[i] for i in keep], dtype=float)
    keep_weights = keep_weights / keep_weights.sum()
    seeds = sorted(int(s) for s in reference.base)
    base_by_seed: dict[int, np.ndarray] = {}
    for seed in seeds:
        blended = np.zeros(len(actual), dtype=float)
        for weight, index in zip(keep_weights, keep):
            blended = blended + float(weight) * reference.member_vector(target, seed, members[index])
        base_by_seed[seed] = np.maximum(blended, 0.0)

    entries = library.complete_entries(target)
    rows: list[dict[str, Any]] = []
    grid = spec.alpha_grid
    for key in candidates[: int(limit)]:
        entry = entries.get(key)
        if entry is None:
            continue
        candidate_by_seed = {s: np.asarray(entry.vectors[s], dtype=float) for s in seeds}
        record = blend_record(actual, {s: np.asarray(folds[s]) for s in seeds},
                              base_by_seed, candidate_by_seed, grid)
        decision = admit(
            record, min_seeds=int(spec.raw["resolution"]["seed_level"]["min_seeds_for_promotion"]),
            positive_cells_min=int(spec.raw["resolution"]["fold_level"]["positive_cells_min"]),
            positive_cells_total=int(spec.raw["resolution"]["fold_level"]["positive_cells_total"]),
        )
        rows.append({
            "key": key,
            "name": entry.name,
            "family": entry.family,
            "packagability": entry.packagability,
            "residual_correlation": float(np.mean([
                float(np.corrcoef(actual - base_by_seed[s], actual - candidate_by_seed[s])[0, 1]) for s in seeds
            ])),
            "nested": record,
            "decision": decision,
        })
    rows.sort(key=lambda r: -float(r["nested"]["fold_summary"]["mean"]))
    return {
        "target": target,
        "dropped_member": drop_name,
        "base_per_seed_wmape": {str(s): wmape(actual, base_by_seed[s]) for s in seeds},
        "dropped_column_vs_released": {
            "per_seed_score_delta": {
                str(s): float(50.0 * (wmape(actual, reference.base_for(target, s)) - wmape(actual, base_by_seed[s])))
                for s in seeds
            }
        },
        "candidates": rows,
    }


def _fixed_quarter_gains(path: Path, trial_id: str) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else [payload]
    for row in rows:
        if str(row.get("trial_id")) != str(trial_id):
            continue
        out: dict[str, dict[str, float]] = {}
        for seed, block in sorted(row["per_seed"].items()):
            fold_level = block.get("fold_level") or {}
            out[str(seed)] = {str(fold): float(block_data["fixed_quarter_gain"])
                              for fold, block_data in sorted(fold_level.items())}
        return out
    raise KeyError(f"V5 backtest: trial {trial_id!r} not found in {path}")


def backtest_gains(root: Path | str, sources: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Read the recorded per-cell gains of the historical candidates."""
    root = Path(root)
    out: list[dict[str, Any]] = []
    for source in (sources or BACKTEST_SOURCES):
        path = root / str(source["path"])
        if not path.is_file():
            raise FileNotFoundError(f"V5 backtest artifact missing: {path}")
        gains = _fixed_quarter_gains(path, str(source["trial_id"]))
        out.append({**{k: v for k, v in source.items() if k != "path"},
                    "path": str(source["path"]),
                    "gains": gains})
    return out


def backtest_discrimination(root: Path | str, spec: V5Spec,
                            sources: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Replay the old gate and the V5 rule on already-published candidates."""
    records = backtest_gains(root, sources)
    rows: list[dict[str, Any]] = []
    for record in records:
        gains: dict[int, float] = {}
        cells: list[float] = []
        for seed, per_fold in sorted(record["gains"].items()):
            values = [float(v) for _, v in sorted(per_fold.items())]
            cells.extend(values)
            gains[int(seed)] = float(np.mean(values))
        fold_summary = paired_summary(cells)
        record_decision = admit(
            {"fold_summary": fold_summary, "seed_gains": gains},
            min_seeds=int(spec.raw["resolution"]["seed_level"]["min_seeds_for_promotion"]),
            positive_cells_min=int(spec.raw["resolution"]["fold_level"]["positive_cells_min"]),
            positive_cells_total=int(spec.raw["resolution"]["fold_level"]["positive_cells_total"]),
        )
        old_gate = bool(fold_summary["mean"] >= 0.005 and all(v > 0 for v in gains.values()))
        rows.append({
            "candidate": record["candidate"],
            "role": record.get("role"),
            "platform_note": record.get("platform_note"),
            "fold_summary": fold_summary,
            "seed_gains": gains,
            "seed_summary": seed_level_summary(gains),
            "old_gate_would_admit": old_gate,
            "v5_admitted": bool(record_decision["admitted"]),
            "v5_reasons": record_decision["reasons"],
            "v5_fold_only_admitted": not [r for r in record_decision["reasons"]
                                          if r in {"positive_cells_below_minimum", "fold_cells_incomplete",
                                                   "initial_split_seed_not_positive"}],
        })
    known_bad = [r for r in rows if str(r["role"]) == "known_bad"]
    replication = [r for r in rows if str(r["role"]) != "known_bad"]
    rejected = [r for r in known_bad if not r["v5_admitted"]]
    return {
        "rows": rows,
        "known_bad_total": len(known_bad),
        "known_bad_rejected": len(rejected),
        "known_bad_admitted": [r["candidate"] for r in known_bad if r["v5_admitted"]],
        "replication_controls": [r["candidate"] for r in replication],
        "old_gate_admitted_known_bad": [r["candidate"] for r in known_bad if r["old_gate_would_admit"]],
        "v5_fold_only_admitted_known_bad": [r["candidate"] for r in known_bad if r["v5_fold_only_admitted"]],
        "requirement_met": len(rejected) >= int(spec.backtest["known_bad_min_rejected"]),
    }
