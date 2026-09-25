"""Round2 V5 Stage 1: the zero-fit error-covariance screen.

Stage 1 answers one question with recorded evidence only: does the existing
two-seed out-of-fold pool contain a member worth spending new fits on?

Every output is written under ``local/runs/round2-v5-error-covariance/``.  The
runner never fits a model, never packages, never uploads.

Two decisions are reported side by side for every candidate:

* ``provisional`` — the fold-level criteria plus the fold-level lower bound on
  the two available split seeds.  This is screening evidence, explicitly not a
  promotion: the same ten cells were used to select the released column.
* ``promotion`` — the full V5 rule, which requires at least four split seeds and
  therefore returns ``insufficient_split_seeds`` for every two-seed candidate by
  construction.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data import TARGETS
from .v5_diagnostics import (
    admissibility_table,
    backtest_discrimination,
    leave_one_out,
    replacement_sweep,
)
from .v5_library import (
    build_candidate_library,
    fold_vector,
    load_column_reference,
    load_v5_training_frame,
)
from .v5_resolution import (
    admit,
    blend_record,
    fold_criteria_met,
    wmape,
)
from .v5_spec import V5Spec, load_v5_spec

__all__ = ["run_stage1", "main"]

DEFAULT_OUTPUT = "local/runs/round2-v5-error-covariance/stage1-r1"
EVIDENCE_SEEDS = (42, 3407)


def _private_output(root: Path, output: Path) -> Path:
    allowed = (root / "local/runs/round2-v5-error-covariance").resolve()
    resolved = (output if output.is_absolute() else root / output).resolve()
    if not resolved.is_relative_to(allowed):
        raise ValueError(f"V5 Stage 1 output must stay private under {allowed}")
    return resolved


def _provisional_decision(record: Mapping[str, Any], spec: V5Spec) -> dict[str, Any]:
    """Fold-level screening decision on the two available split seeds."""
    fold_summary = dict(record["fold_summary"])
    gains = {int(k): float(v) for k, v in record["seed_gains"].items()}
    ok, reasons = fold_criteria_met(
        fold_summary,
        int(spec.raw["resolution"]["fold_level"]["positive_cells_min"]),
        int(spec.raw["resolution"]["fold_level"]["positive_cells_total"]),
        both_initial_seeds_positive=all(v > 0 for v in gains.values()) if gains else False,
    )
    if fold_summary.get("lcb95") is None or float(fold_summary["lcb95"]) <= 0.0:
        reasons = [*reasons, "fold_level_lcb_not_positive"]
        ok = False
    return {
        "provisional": bool(ok),
        "label": "PROVISIONAL_SCREENING_NOT_PROMOTION",
        "reasons": reasons,
        "fold_summary": fold_summary,
        "seed_gains": gains,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def run_stage1(root: Path | str, spec: V5Spec | None = None,
               output: Path | str = DEFAULT_OUTPUT,
               candidate_limit: int = 30,
               replacement_limit: int = 25) -> dict[str, Any]:
    """Execute the Stage 1 screen and write its private evidence."""
    root = Path(root).resolve()
    spec = spec or load_v5_spec(root)
    started = time.time()
    out = _private_output(root, Path(output))
    train = load_v5_training_frame(root)
    folds = {int(seed): fold_vector(root, train, int(seed), spec) for seed in EVIDENCE_SEEDS}
    library = build_candidate_library(root, spec, train)
    reference = load_column_reference(root, train, spec)

    payload: dict[str, Any] = {
        "version": spec.version,
        "stage": "stage1_zero_fit_screen",
        "status": "RUNNING",
        "library_audit": library.audit,
        "column_reference": {
            "summary": str(reference.summary_path.relative_to(root)),
            "summary_sha256": reference.summary_sha256,
            "weights": {t: [float(v) for v in reference.weights[t]] for t in TARGETS},
            "selected_experts": {t: list(reference.members[t]) for t in TARGETS},
        },
        "evidence_seeds": list(EVIDENCE_SEEDS),
        "targets": {},
        "agent_uploads": 0,
    }
    promotion_rule = {
        "min_seeds": int(spec.raw["resolution"]["seed_level"]["min_seeds_for_promotion"]),
        "positive_cells_min": int(spec.raw["resolution"]["fold_level"]["positive_cells_min"]),
        "positive_cells_total": int(spec.raw["resolution"]["fold_level"]["positive_cells_total"]),
    }
    sweep: dict[str, Any] = {}
    admissible_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        actual = train[target].to_numpy(dtype=float)
        base_by_seed = {seed: reference.base_for(target, seed) for seed in EVIDENCE_SEEDS}
        released = set(reference.members[target])
        records = admissibility_table(actual, base_by_seed, library.entries(target).values(), spec)
        table = [record.as_dict() for record in records]
        for row in table:
            row["in_released_composition"] = row["name"] in released
            admissible_rows.append({**row, "evaluated_target": target})
        admissible = [r for r in records if r.admissible and r.name not in released]
        ranked = sorted(admissible, key=lambda r: r.residual_correlation)[: int(candidate_limit)]

        candidates: list[dict[str, Any]] = []
        for record in ranked:
            entry = library.entries(target)[record.key]
            candidate_by_seed = {seed: np.asarray(entry.vectors[seed], dtype=float) for seed in EVIDENCE_SEEDS}
            nested = blend_record(actual, folds, base_by_seed, candidate_by_seed, spec.alpha_grid)
            decision = admit(nested, **promotion_rule)
            candidates.append({
                "key": record.key,
                "name": record.name,
                "family": record.family,
                "source": record.source,
                "packagability": record.packagability,
                "convention": record.convention,
                "residual_correlation": record.residual_correlation,
                "accuracy_ratio": record.accuracy_ratio,
                "nested": nested,
                "provisional": _provisional_decision(nested, spec),
                "promotion": decision,
            })
        candidates.sort(key=lambda r: -float(r["nested"]["fold_summary"]["mean"]))

        loo = leave_one_out(reference, actual, target, folds)
        # Replace the near-zero non-base members — never the base endpoint, and no
        # weight re-tuning of the released members.  Both non-base members are
        # swept so the choice is evidence-driven rather than arbitrary.
        weak_members = sorted(
            (row for row in loo["drops"] if row["member"] != "A_dev"),
            key=lambda row: abs(float(row["mean_score_delta"])),
        )
        top_keys = [r["key"] for r in candidates[: int(replacement_limit)]]
        replacement: dict[str, Any] = {}
        for drop in weak_members:
            if not top_keys:
                break
            replacement[drop["member"]] = replacement_sweep(
                actual, folds, reference, library, target, spec,
                candidates=top_keys, drop_member=drop["member"], limit=int(replacement_limit),
            )
            for row in replacement[drop["member"]]["candidates"]:
                row["provisional"] = _provisional_decision(row["nested"], spec)

        best_provisional = []
        for drop, sweep_rows in replacement.items():
            for row in sweep_rows["candidates"]:
                if row["provisional"]["provisional"]:
                    best_provisional.append({**row, "dropped_member": drop})
        for row in candidates:
            if row["provisional"]["provisional"]:
                best_provisional.append({**row, "dropped_member": None})
        best_provisional.sort(key=lambda r: -float(r["nested"]["fold_summary"]["mean"]))
        payload["targets"][target] = {
            "base_per_seed_wmape": {str(s): wmape(actual, base_by_seed[s]) for s in EVIDENCE_SEEDS},
            "admissible_count": int(sum(1 for r in records if r.admissible)),
            "admissible_excluding_released_total": int(len(admissible)),
            "admissible_screened_count": len(ranked),
            "screened_candidates": candidates,
            "leave_one_out": loo,
            "replacement_sweep": replacement,
            "best_provisional_candidate": (best_provisional[0]["name"] if best_provisional else None),
            "best_provisional_gain": (
                float(best_provisional[0]["nested"]["fold_summary"]["mean"]) if best_provisional else None
            ),
            "promoted": [r["name"] for r in candidates if r["promotion"]["admitted"]],
        }
        sweep[target] = {
            "screened_candidates": candidates,
            "leave_one_out": loo,
            "replacement_sweep": replacement,
        }

    backtest = backtest_discrimination(root, spec)
    payload["backtest"] = backtest

    best_overall = None
    for target in TARGETS:
        value = payload["targets"][target]["best_provisional_gain"]
        if value is None:
            continue
        if best_overall is None or value > best_overall["gain"]:
            best_overall = {"target": target, "gain": float(value),
                            "candidate": payload["targets"][target]["best_provisional_candidate"]}
    payload["best_provisional_overall"] = best_overall
    payload["existing_pool_exhausted"] = bool(
        not backtest["requirement_met"] or best_overall is None or best_overall["gain"] < 0.01
    )
    payload["stage1_stop_rule"] = {
        "threshold_score_points": 0.01,
        "decision": ("existing_two_seed_pool_exhausted" if payload["existing_pool_exhausted"]
                     else "candidate_worth_stage2_fits"),
    }
    payload["elapsed_seconds"] = float(time.time() - started)
    payload["status"] = "COMPLETE_NO_PROMOTION"

    out.mkdir(parents=True, exist_ok=True)
    (out / "stage1_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                                             encoding="utf-8")
    _write_csv(out / "admissibility.csv", admissible_rows,
               ["evaluated_target", "key", "name", "target", "family", "source", "packagability",
                "convention", "residual_correlation", "single_wmape", "base_wmape", "accuracy_ratio",
                "admissible", "in_released_composition", "reasons"])
    screen_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for row in payload["targets"][target]["screened_candidates"]:
            screen_rows.append({
                "target": target,
                "key": row["key"],
                "name": row["name"],
                "family": row["family"],
                "packagability": row["packagability"],
                "residual_correlation": row["residual_correlation"],
                "accuracy_ratio": row["accuracy_ratio"],
                "fold_mean_score": row["nested"]["fold_summary"]["mean"],
                "fold_lcb95": row["nested"]["fold_summary"]["lcb95"],
                "positive_cells": row["nested"]["fold_summary"]["positive"],
                "seed_42": row["nested"]["seed_gains"].get("42", row["nested"]["seed_gains"].get(42)),
                "seed_3407": row["nested"]["seed_gains"].get("3407", row["nested"]["seed_gains"].get(3407)),
                "provisional": row["provisional"]["provisional"],
                "promotion_admitted": row["promotion"]["admitted"],
                "promotion_reasons": ";".join(row["promotion"]["reasons"]),
            })
    _write_csv(out / "screened_candidates.csv", screen_rows,
               ["target", "key", "name", "family", "packagability", "residual_correlation",
                "accuracy_ratio", "fold_mean_score", "fold_lcb95", "positive_cells", "seed_42",
                "seed_3407", "provisional", "promotion_admitted", "promotion_reasons"])
    (out / "backtest.json").write_text(json.dumps(backtest, ensure_ascii=False, indent=2, default=float),
                                       encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--candidate-limit", type=int, default=30)
    parser.add_argument("--replacement-limit", type=int, default=25)
    args = parser.parse_args(argv)
    payload = run_stage1(args.root, output=args.output,
                         candidate_limit=args.candidate_limit,
                         replacement_limit=args.replacement_limit)
    summary = {
        "status": payload["status"],
        "library": {k: v for k, v in payload["library_audit"].items()
                    if k in {"entries_per_target", "two_seed_full_per_target"}},
        "backtest_requirement_met": payload["backtest"]["requirement_met"],
        "backtest_known_bad_rejected": f"{payload['backtest']['known_bad_rejected']}/{payload['backtest']['known_bad_total']}",
        "best_provisional_overall": payload["best_provisional_overall"],
        "existing_pool_exhausted": payload["existing_pool_exhausted"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
