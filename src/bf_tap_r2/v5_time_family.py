"""Round2 V5 Stage 2a: complete the time column's N family at full coverage.

The V3.6 round stopped the N (PLE-MLP / TabM) line for the time target because
its single models were weaker than the A_dev/O/D candidates.  The iron side was
equally weak singly yet produced the released composition's entire gain, because
it is the only family that is genuinely decorrelated from the base
(``rho = 0.9095`` against a pool whose next-best member is ``0.985``).

At folds 0/1 the time-side N candidates include members with ``rho`` near 0.78
and bounded accuracy loss, so the family was stopped on the wrong criterion.
This module re-screens them with the V5 admissibility rule and, when they pass,
refines the selected trials to complete five-fold coverage using the frozen
V3.6 evaluator (``evaluate_v36_outer_folds``), which never sees validation
labels while fitting.

Nothing here packages, promotes or uploads.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .v5_library import (
    CandidateLibrary,
    ColumnReference,
    build_candidate_library,
    fold_vector,
    load_column_reference,
    load_v5_training_frame,
)
from .v5_resolution import select_alpha, wmape
from .v5_spec import V5Spec, load_v5_spec

__all__ = ["screen_time_n_candidates", "refine_time_n_candidates", "run_time_family", "main"]

DEFAULT_OUTPUT = "local/runs/round2-v5-error-covariance/time-n-family-r1"
SCREEN_TARGET = "tap_time_len"
SCREEN_FAMILY = "N"
SCREEN_SEED = 42
SCREEN_FOLDS = (0, 1)


def _private_output(root: Path, output: Path) -> Path:
    allowed = (root / "local/runs/round2-v5-error-covariance").resolve()
    resolved = (output if output.is_absolute() else root / output).resolve()
    if not resolved.is_relative_to(allowed):
        raise ValueError(f"V5 Stage 2a output must stay private under {allowed}")
    return resolved


def screen_time_n_candidates(root: Path | str, spec: V5Spec, train,
                             library: CandidateLibrary, reference: ColumnReference) -> list[dict[str, Any]]:
    """Rank the time-side N coarse trials with the V5 admissibility rule.

    Correlation and accuracy loss are measured on the same covered rows as the
    coarse prediction (folds 0/1 of split seed 42).  The projected gain uses a
    two-way nested fold protocol: the blend weight is chosen on one fold and
    scored on the other, in both directions.
    """
    root = Path(root).resolve()
    actual = train[SCREEN_TARGET].to_numpy(dtype=float)
    folds = fold_vector(root, train, SCREEN_SEED, spec)
    mask = np.isin(folds, list(SCREEN_FOLDS))
    base = reference.base_for(SCREEN_TARGET, SCREEN_SEED)
    base_wmape = wmape(actual[mask], base[mask])
    correlation_max = float(spec.admissibility["residual_correlation_max"])
    ratio_max = float(spec.admissibility["single_wmape_ratio_max"])
    grid = spec.alpha_grid

    rows: list[dict[str, Any]] = []
    for entry in library.entries(SCREEN_TARGET).values():
        if entry.source != "v36_coarse_diagnostic" or entry.family != SCREEN_FAMILY:
            continue
        vector = np.asarray(entry.vectors[SCREEN_SEED], dtype=float)
        if not np.isfinite(vector[mask]).all():
            continue
        correlation = float(np.corrcoef((actual - base)[mask], (actual - vector)[mask])[0, 1])
        single = wmape(actual[mask], vector[mask])
        ratio = single / base_wmape
        admissible = bool(correlation <= correlation_max and ratio <= ratio_max)

        gains: list[float] = []
        alphas: list[float] = []
        for fit_fold, eval_fold in ((0, 1), (1, 0)):
            fit_mask = folds == fit_fold
            eval_mask = folds == eval_fold
            alpha, _ = select_alpha(actual[fit_mask], base[fit_mask], vector[fit_mask], grid)
            delta = wmape(actual[eval_mask], base[eval_mask]) - wmape(
                actual[eval_mask],
                (1.0 - float(alpha)) * base[eval_mask] + float(alpha) * vector[eval_mask],
            )
            gains.append(float(50.0 * delta))
            alphas.append(float(alpha))
        rows.append({
            "trial_id": entry.name,
            "key": entry.key,
            "admissible": admissible,
            "residual_correlation": correlation,
            "single_wmape": single,
            "base_wmape_folds01": base_wmape,
            "accuracy_ratio": ratio,
            "projected_gain_score": float(np.mean(gains)),
            "projected_fold_gains": gains,
            "projected_alphas": alphas,
            "projected_gain_positive": bool(np.mean(gains) > 0),
        })
    rows.sort(key=lambda r: (-float(r["projected_gain_score"]), float(r["residual_correlation"]), r["trial_id"]))
    return rows


_WORKER_TRAIN = None
_WORKER_FOLDS = None


def _init_worker(train, folds) -> None:
    global _WORKER_TRAIN, _WORKER_FOLDS
    _WORKER_TRAIN, _WORKER_FOLDS = train, folds


def _worker_run(payload: Mapping[str, Any]) -> dict[str, Any]:
    from .v3_6_models import evaluate_v36_outer_folds

    if _WORKER_TRAIN is None or _WORKER_FOLDS is None:
        raise RuntimeError("V5 time-family worker was not initialised")
    return evaluate_v36_outer_folds(
        _WORKER_TRAIN, _WORKER_FOLDS, payload["trial"], fold_ids=tuple(payload["fold_ids"])
    )


def _read_completed(ledger: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not ledger.is_file():
        return completed
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "complete":
            completed[str(event["trial_id"])] = event
    return completed


def refine_time_n_candidates(root: Path | str, output: Path | str, trial_ids: Sequence[str],
                             seeds: Sequence[int], folds: Sequence[int],
                             workers: int = 16) -> dict[str, Any]:
    """Refit the selected trials to complete coverage with the frozen V3.6 evaluator.

    The evaluator is ``bf_tap_r2.v3_6_models.evaluate_v36_outer_folds``: it fits on
    the outer training part only and never sees validation labels.  Outputs stay
    inside the V5 private run directory; the V3.6 runner refuses foreign output
    paths, so the batch plumbing is repeated here rather than relaxed there.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from .v3_6_run import _bag_hash_for_trial, _code_version, _hash_folds, _hash_frame
    from .v3_6_sampler import sample_v36

    root = Path(root).resolve()
    out = _private_output(root, Path(output))
    train = load_v5_training_frame(root)
    trials = {str(t["trial_id"]): t for t in sample_v36(root)}
    missing = [tid for tid in trial_ids if tid not in trials]
    if missing:
        raise KeyError(f"V5 time-family trials are not in the V3.6 sampler: {missing}")
    data_hash = _hash_frame(train)
    code_version = _code_version(root)
    batches: dict[str, Any] = {}
    for seed in seeds:
        seed = int(seed)
        seed_dir = out / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        fold_vector_local = fold_vector(root, train, seed, load_v5_spec(root))
        fold_hash = _hash_folds(fold_vector_local, tuple(int(v) for v in folds))
        ledger = seed_dir / "fit_ledger.jsonl"
        completed = _read_completed(ledger)
        pending: list[dict[str, Any]] = []
        identities: dict[str, dict[str, Any]] = {}
        for trial_id in trial_ids:
            trial = trials[trial_id]
            identity = {
                "trial_id": trial_id,
                "stage": "v5-time-n-family",
                "fold_seed": seed,
                "fold_ids": [int(v) for v in folds],
                "data_hash": data_hash,
                "fold_hash": fold_hash,
                "code_version": code_version,
                "bag_hash": _bag_hash_for_trial(trial, train),
                "numeric-encoding": "numeric-network-train-fold-only-v1",
            }
            identities[trial_id] = identity
            old = completed.get(trial_id)
            if old is not None:
                if old.get("identity", {}) != identity:
                    raise ValueError(f"V5 time-family cache identity mismatch: {trial_id}")
                continue
            pending.append({"trial": trial, "fold_ids": [int(v) for v in folds]})
        (seed_dir / "batch.json").write_text(json.dumps({
            "version": "round2-v5-error-covariance-resolution",
            "stage": "v5-time-n-family",
            "fold_seed": seed,
            "fold_ids": [int(v) for v in folds],
            "data_hash": data_hash,
            "fold_hash": fold_hash,
            "code_version": code_version,
            "slots_selected": len(trial_ids),
            "pending_at_start": len(pending),
            "agent_uploads": 0,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        results: list[dict[str, Any]] = []
        if pending:
            with ProcessPoolExecutor(max_workers=max(1, int(workers)), initializer=_init_worker,
                                     initargs=(train, fold_vector_local)) as executor:
                futures = {executor.submit(_worker_run, payload): payload for payload in pending}
                for future in as_completed(futures):
                    payload = futures[future]
                    trial = payload["trial"]
                    trial_id = str(trial["trial_id"])
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001 - failure evidence is kept
                        with ledger.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps({
                                "event": "failed",
                                "trial_id": trial_id,
                                "identity": identities[trial_id],
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            }, ensure_ascii=False) + "\n")
                        continue
                    prediction = np.asarray(result.pop("predictions"), dtype=float)
                    np.save(seed_dir / f"pred-{trial_id}.npy", prediction)
                    record = {**{k: v for k, v in result.items() if k != "fit_meta"},
                              "fit_meta": result.get("fit_meta", []),
                              "identity": identities[trial_id]}
                    with ledger.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"event": "complete", **record},
                                                ensure_ascii=False, default=float) + "\n")
                    results.append(record)
        batches[str(seed)] = {
            "seed": seed,
            "selected_trials": len(trial_ids),
            "completed_this_run": len(results),
            "already_completed": len(completed),
            "directory": str(seed_dir.relative_to(root)),
        }
    return batches


def run_time_family(root: Path | str, spec: V5Spec | None = None,
                    output: Path | str = DEFAULT_OUTPUT,
                    max_trials: int | None = None,
                    seeds: Sequence[int] = (42, 3407),
                    extra_seed_top_k: int = 0,
                    workers: int = 16,
                    dry_run: bool = False,
                    trial_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Screen the time-side N family and refine the selected trials.

    ``trial_ids`` lets a caller refine an explicitly named subset of the
    pre-registered selection (for example to complete one split seed for the
    cheaper candidates while the expensive ones are still running).  The
    selection itself is never widened by this option.
    """
    root = Path(root).resolve()
    spec = spec or load_v5_spec(root)
    stage2a = spec.stage2a
    started = time.time()
    out = _private_output(root, Path(output))
    train = load_v5_training_frame(root)
    library = build_candidate_library(root, spec, train)
    reference = load_column_reference(root, train, spec)
    rows = screen_time_n_candidates(root, spec, train, library, reference)
    limit = int(max_trials if max_trials is not None else stage2a["max_selected_trials"])
    pre_registered = [r for r in rows if r["admissible"] and r["projected_gain_positive"]][:limit]
    allowed = {r["trial_id"] for r in pre_registered}
    if trial_ids is None:
        selected = pre_registered
    else:
        unknown = sorted(set(str(v) for v in trial_ids) - allowed)
        if unknown:
            raise ValueError(f"V5 Stage 2a requested trials outside the pre-registered selection: {unknown}")
        selected = [r for r in pre_registered if r["trial_id"] in set(str(v) for v in trial_ids)]

    payload: dict[str, Any] = {
        "version": spec.version,
        "stage": "stage2a_time_n_family_completion",
        "target": SCREEN_TARGET,
        "family": SCREEN_FAMILY,
        "screen_seed": SCREEN_SEED,
        "screen_folds": list(SCREEN_FOLDS),
        "candidate_count": len(rows),
        "admissible_count": int(sum(1 for r in rows if r["admissible"])),
        "pre_registered_selection": [r["trial_id"] for r in pre_registered],
        "selected": [r["trial_id"] for r in selected],
        "screening": rows,
        "dry_run": bool(dry_run),
        "agent_uploads": 0,
    }
    out.mkdir(parents=True, exist_ok=True)
    # ``selection.json`` always records the *pre-registered* screening selection; a
    # partial invocation (for example one split seed of the cheaper candidates)
    # must not shrink the artefact the evaluator reads.
    selection_record = {
        **payload,
        "selected": [r["trial_id"] for r in pre_registered],
        "requested_subset": [r["trial_id"] for r in selected],
    }
    (out / "selection.json").write_text(
        json.dumps(selection_record, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    with (out / "selection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "trial_id", "key", "admissible", "residual_correlation", "single_wmape",
            "base_wmape_folds01", "accuracy_ratio", "projected_gain_score",
            "projected_gain_positive", "projected_alphas"], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if dry_run or not selected:
        payload["status"] = "SCREENED_ONLY"
        payload["elapsed_seconds"] = float(time.time() - started)
        (out / "stage2a_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        return payload

    trial_ids = [r["trial_id"] for r in selected]
    payload["refine_batches"] = refine_time_n_candidates(root, out, trial_ids, seeds,
                                                         stage2a["folds"], workers=workers)
    if int(extra_seed_top_k) > 0:
        extra = int(stage2a["extra_seed"])
        top = [r["trial_id"] for r in selected[: int(extra_seed_top_k)]]
        payload["extra_seed_batch"] = refine_time_n_candidates(
            root, out, top, [extra], stage2a["folds"], workers=workers)
    payload["status"] = "COMPLETE"
    payload["elapsed_seconds"] = float(time.time() - started)
    (out / "stage2a_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return payload


def evaluate_refined_candidates(root: Path | str, spec: V5Spec | None = None,
                                output: Path | str = DEFAULT_OUTPUT,
                                seeds: Sequence[int] = (42, 3407)) -> dict[str, Any]:
    """Re-score the newly complete N-family candidates against the released column.

    The comparison uses the same nested protocol as Stage 1: the blend weight is
    selected on one split seed and the paired cells are scored on the other, in
    both directions.  Promotion still requires four split seeds, so every result
    here is provisional screening evidence.
    """
    from .v5_resolution import admit, blend_record

    root = Path(root).resolve()
    spec = spec or load_v5_spec(root)
    out = _private_output(root, Path(output))
    train = load_v5_training_frame(root)
    reference = load_column_reference(root, train, spec)
    actual = train[SCREEN_TARGET].to_numpy(dtype=float)
    seeds = [int(s) for s in seeds]
    folds = {seed: fold_vector(root, train, seed, spec) for seed in seeds}
    base_by_seed = {seed: reference.base_for(SCREEN_TARGET, seed) for seed in seeds}

    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    selected = [str(v) for v in selection.get("selected", [])]
    rows: list[dict[str, Any]] = []
    for trial_id in selected:
        vectors: dict[int, np.ndarray] = {}
        complete = True
        for seed in seeds:
            path = out / f"seed-{seed}" / f"pred-{trial_id}.npy"
            if not path.is_file():
                complete = False
                break
            values = np.load(path).astype(float, copy=False)
            if values.shape != (len(train),) or not np.isfinite(values).all():
                raise ValueError(f"V5 refined candidate is invalid: {path}")
            vectors[seed] = values
        if not complete:
            continue
        correlations = [
            float(np.corrcoef(actual - base_by_seed[seed], actual - vectors[seed])[0, 1])
            for seed in seeds
        ]
        single = float(np.mean([wmape(actual, vectors[seed]) for seed in seeds]))
        base_mean = float(np.mean([wmape(actual, base_by_seed[seed]) for seed in seeds]))
        nested = blend_record(actual, folds, base_by_seed, vectors, spec.alpha_grid)
        decision = admit(
            nested,
            min_seeds=int(spec.raw["resolution"]["seed_level"]["min_seeds_for_promotion"]),
            positive_cells_min=int(spec.raw["resolution"]["fold_level"]["positive_cells_min"]),
            positive_cells_total=int(spec.raw["resolution"]["fold_level"]["positive_cells_total"]),
        )
        rows.append({
            "trial_id": trial_id,
            "family": SCREEN_FAMILY,
            "residual_correlation": float(np.mean(correlations)),
            "accuracy_ratio": float(single / base_mean) if base_mean > 0 else float("inf"),
            "single_wmape": single,
            "base_wmape": base_mean,
            "nested": nested,
            "promotion": decision,
        })
    rows.sort(key=lambda r: -float(r["nested"]["fold_summary"]["mean"]))
    payload = {
        "version": spec.version,
        "stage": "stage2a_refined_evaluation",
        "target": SCREEN_TARGET,
        "seeds": seeds,
        "candidates": rows,
        "elapsed_selection": selection.get("elapsed_seconds"),
        "agent_uploads": 0,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "refined_evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    with (out / "refined_evaluation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "trial_id", "residual_correlation", "accuracy_ratio", "single_wmape", "base_wmape",
            "fold_mean_score", "fold_lcb95", "positive_cells", "seed_42", "seed_3407",
            "promotion_admitted", "promotion_reasons"], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "trial_id": row["trial_id"],
                "residual_correlation": row["residual_correlation"],
                "accuracy_ratio": row["accuracy_ratio"],
                "single_wmape": row["single_wmape"],
                "base_wmape": row["base_wmape"],
                "fold_mean_score": row["nested"]["fold_summary"]["mean"],
                "fold_lcb95": row["nested"]["fold_summary"]["lcb95"],
                "positive_cells": row["nested"]["fold_summary"]["positive"],
                "seed_42": row["nested"]["seed_gains"].get(42),
                "seed_3407": row["nested"]["seed_gains"].get(3407),
                "promotion_admitted": row["promotion"]["admitted"],
                "promotion_reasons": ";".join(row["promotion"]["reasons"]),
            })
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 3407])
    parser.add_argument("--extra-seed-top-k", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--trial-ids", nargs="+", default=None)
    args = parser.parse_args(argv)
    if args.evaluate_only:
        payload = evaluate_refined_candidates(args.root, output=args.output)
        print(json.dumps([{k: row[k] for k in ("trial_id", "residual_correlation", "accuracy_ratio")}
                          | {"fold_mean_score": row["nested"]["fold_summary"]["mean"],
                             "positive_cells": row["nested"]["fold_summary"]["positive"],
                             "promotion_reasons": row["promotion"]["reasons"]}
                          for row in payload["candidates"]], ensure_ascii=False, indent=2, default=float))
        return 0
    payload = run_time_family(args.root, output=args.output, max_trials=args.max_trials,
                              seeds=tuple(args.seeds), extra_seed_top_k=args.extra_seed_top_k,
                              workers=args.workers, dry_run=args.dry_run,
                              trial_ids=args.trial_ids)
    print(json.dumps({
        "status": payload["status"],
        "candidate_count": payload["candidate_count"],
        "admissible_count": payload["admissible_count"],
        "selected": payload["selected"],
        "dry_run": payload["dry_run"],
        "elapsed_seconds": payload.get("elapsed_seconds"),
    }, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
