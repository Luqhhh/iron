"""Round2 V5 Stage 2b: independent split-seed replication of a frozen finalist.

The pre-registration fixes the split seed, not the fold, as the promotion unit.
The two recorded development seeds (42/3407) were used to select the released
column and the candidates, so they cannot confirm anything on their own.  This
module extends a candidate's evidence to derived split seeds with the frozen
V3.6 fixed-recipe baseline refit inside every outer fold:

* the baseline for each fold is ``V36FixedRecipeFactory.fit_predict`` on that
  fold's training part — the same fixed-recipe adapter the V4.1/V4.2 screens
  used, so the reproduced column is comparable to the released one;
* the candidate is refit on the same training part with the frozen V3.6 or V3
  trial specification;
* the held-out fold labels are used only for scoring.

Every fold is cached on disk and skipped when its identity already matches, so a
long run is resumable.  Nothing here packages or uploads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .data import TARGETS
from .v5_library import fold_vector, load_column_reference, load_v5_training_frame
from .v5_resolution import (
    admit,
    nested_blend,
    paired_cells,
    paired_summary,
    seed_gains,
    wmape,
)
from .v5_spec import V5Spec, load_v5_spec

__all__ = ["load_candidate", "replicate_candidate", "replication_status", "main"]

DEFAULT_OUTPUT = "local/runs/round2-v5-error-covariance/replication-r1"

CandidateFit = Callable[[Any, Any, str], np.ndarray]


def _private_output(root: Path, output: Path) -> Path:
    allowed = (root / "local/runs/round2-v5-error-covariance").resolve()
    resolved = (output if output.is_absolute() else root / output).resolve()
    if not resolved.is_relative_to(allowed):
        raise ValueError(f"V5 Stage 2b output must stay private under {allowed}")
    return resolved


def load_candidate(root: Path | str, spec: V5Spec, kind: str, trial_id: str) -> tuple[CandidateFit, dict[str, Any]]:
    """Return a ``(train, query, target) -> prediction`` fitter for one candidate."""
    root = Path(root).resolve()
    if kind == "v36":
        from .v3_6_models import V36Regressor
        from .v3_6_sampler import sample_v36

        trials = {str(t["trial_id"]): t for t in sample_v36(root)}
        if trial_id not in trials:
            raise KeyError(f"V5 replication: unknown V3.6 trial {trial_id!r}")
        trial = trials[trial_id]

        def fit(train, query, target):  # noqa: ANN001 - local closure
            model = V36Regressor(dict(trial))
            model.fit(train.reset_index(drop=True), train[target].to_numpy(dtype=float))
            return np.asarray(model.predict(query.reset_index(drop=True)), dtype=float)

        return fit, {"kind": kind, "trial_id": trial_id, "target": str(trial["target"]),
                     "line": str(trial.get("line"))}
    if kind == "v3":
        from .v3_local_search import TrialRegressor
        from .v3_run import read_complete_records, trial_details

        ledgers = sorted((root / "local/runs/round2-v3-local-search").glob("coarse-*-r1/fit_ledger.jsonl"))
        trials: dict[str, dict[str, Any]] = {}
        for ledger in ledgers:
            for record in read_complete_records(ledger):
                trials.setdefault(str(record["trial_id"]), trial_details(record))
        if trial_id not in trials:
            raise KeyError(f"V5 replication: unknown V3 trial {trial_id!r}")
        trial = trials[trial_id]

        def fit(train, query, target):  # noqa: ANN001 - local closure
            model = TrialRegressor(dict(trial))
            model.fit(train.reset_index(drop=True), train[target].to_numpy(dtype=float))
            return np.asarray(model.predict(query.reset_index(drop=True)), dtype=float)

        return fit, {"kind": kind, "trial_id": trial_id, "target": str(trial["target"]),
                     "family": str(trial.get("family"))}
    raise ValueError(f"V5 replication: unsupported candidate kind {kind!r}")


def _fold_cache_path(output: Path, seed: int, fold: int) -> Path:
    return output / f"seed-{int(seed)}" / f"fold-{int(fold)}.npz"


def _identity(train, folds, seed: int, fold: int, candidate: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"seed": int(seed), "fold": int(fold), **dict(candidate)},
                             sort_keys=True, default=str).encode("utf-8"))
    digest.update(np.asarray(folds, dtype=np.int64).tobytes())
    digest.update(",".join(str(v) for v in train.sample_id.astype(str)).encode("utf-8"))
    for target in TARGETS:
        digest.update(np.asarray(train[target], dtype=float).tobytes())
    return digest.hexdigest()


def _run_one_fold(root: Path, train, folds: np.ndarray, target: str, seed: int, fold: int,
                  fit: CandidateFit, candidate_meta: Mapping[str, Any],
                  base_workers: int, output: Path) -> dict[str, Any]:
    from .v4_1_reference import V36FixedRecipeFactory

    path = _fold_cache_path(output, seed, fold)
    identity = _identity(train, folds, seed, fold, candidate_meta)
    if path.is_file():
        cached = np.load(path)
        if str(cached["identity"]) == identity:
            return {"seed": int(seed), "fold": int(fold), "cached": True,
                    "base": np.asarray(cached["base"], dtype=float),
                    "candidate": np.asarray(cached["candidate"], dtype=float),
                    "mask": np.asarray(cached["mask"], dtype=bool)}
    fold = int(fold)
    training = train.loc[folds != fold].reset_index(drop=True)
    valid = train.loc[folds == fold].reset_index(drop=True)
    factory = V36FixedRecipeFactory(root, workers=int(base_workers))
    bundle = factory.fit_predict(training, valid)
    base = np.asarray(bundle["b36"][target], dtype=float)
    candidate = np.asarray(fit(training, valid, target), dtype=float)
    mask = folds == fold
    position = np.where(mask)[0]
    full_base = np.full(len(train), np.nan)
    full_candidate = np.full(len(train), np.nan)
    full_base[position] = base
    full_candidate[position] = candidate
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, base=full_base, candidate=full_candidate, mask=mask,
                        identity=np.asarray(identity))
    return {"seed": int(seed), "fold": fold, "cached": False, "base": full_base,
            "candidate": full_candidate, "mask": mask}


def _recorded_vector(root: Path, spec: V5Spec, kind: str, trial_id: str, seed: int) -> np.ndarray:
    """Locate one candidate's already-recorded complete OOF vector.

    V5-refined candidates live in the V5 run tree; frozen V3.6 experts live in the
    V3.6 development cache; V3 refined trials live in the V3 local-search tree.
    """
    probes: list[Path] = [
        root / f"local/runs/round2-v5-error-covariance/time-n-family-r1/seed-{int(seed)}"
        / f"pred-{trial_id}.npy",
    ]
    if kind == "v36":
        probes.append(root / str(spec.raw["reference"]["v36_development_cache"])
                      / f"seed-{int(seed)}" / f"pred-{trial_id}.npy")
    else:
        for candidate_dir in sorted((root / "local/runs/round2-v3-local-search").glob("refine-*-r1")):
            probes.append(candidate_dir / f"pred-{int(seed)}-{trial_id}.npy")
    for path in probes:
        if path.is_file():
            values = np.load(path).astype(float, copy=False)
            if values.shape != (2754,) or not np.isfinite(values).all():
                raise ValueError(f"V5 replication: recorded vector is invalid: {path}")
            return values
    raise FileNotFoundError(
        f"V5 replication needs the recorded {trial_id} vector at seed {seed}; "
        f"searched {[str(p) for p in probes]}")


def replicate_candidate(root: Path | str, spec: V5Spec, output: Path | str,
                        kind: str, trial_id: str, target: str,
                        seeds: Sequence[int], folds: Sequence[int] = (0, 1, 2, 3, 4),
                        base_workers: int = 12, recorded_seeds: Sequence[int] = (42, 3407),
                        max_new_folds: int | None = None) -> dict[str, Any]:
    """Extend one candidate's evidence to derived split seeds and decide."""
    root = Path(root).resolve()
    out = _private_output(root, Path(output))
    out.mkdir(parents=True, exist_ok=True)
    train = load_v5_training_frame(root)
    fit, meta = load_candidate(root, spec, kind, trial_id)
    if str(meta["target"]) != str(target):
        raise ValueError(f"V5 replication: candidate target {meta['target']!r} != {target!r}")

    results: list[dict[str, Any]] = []
    new_folds = 0
    derived_folds: dict[int, np.ndarray] = {}
    derived_base: dict[int, np.ndarray] = {}
    derived_candidate: dict[int, np.ndarray] = {}
    for seed in seeds:
        seed = int(seed)
        fold_vector_local = fold_vector(root, train, seed, spec)
        base_full = np.full(len(train), np.nan)
        candidate_full = np.full(len(train), np.nan)
        for fold in folds:
            fold = int(fold)
            if max_new_folds is not None and new_folds >= int(max_new_folds):
                break
            cached = _fold_cache_path(out, seed, fold).is_file()
            result = _run_one_fold(root, train, fold_vector_local, target, seed, fold,
                                   fit, meta, base_workers, out)
            if not result["cached"]:
                new_folds += 1
            position = fold_vector_local == fold
            base_full[position] = np.asarray(result["base"], dtype=float)[position]
            candidate_full[position] = np.asarray(result["candidate"], dtype=float)[position]
            results.append(result)
        derived_folds[seed] = fold_vector_local
        derived_base[seed] = base_full
        derived_candidate[seed] = candidate_full
        if max_new_folds is not None and new_folds >= int(max_new_folds):
            break

    actual = train[target].to_numpy(dtype=float)
    reference = load_column_reference(root, train, spec)
    folds_by_seed: dict[int, np.ndarray] = {
        int(seed): fold_vector(root, train, int(seed), spec) for seed in recorded_seeds
    }
    base_by_seed: dict[int, np.ndarray] = {
        int(seed): reference.base_for(target, int(seed)) for seed in recorded_seeds
    }
    candidate_by_seed: dict[int, np.ndarray] = {}
    for seed in recorded_seeds:
        candidate_by_seed[int(seed)] = _recorded_vector(root, spec, kind, trial_id, int(seed))

    for seed, fold_vector_local in derived_folds.items():
        if not np.isfinite(derived_base[seed]).all():
            raise ValueError(f"V5 replication: incomplete baseline coverage at seed {seed}")
        if not np.isfinite(derived_candidate[seed]).all():
            raise ValueError(f"V5 replication: incomplete candidate coverage at seed {seed}")
        folds_by_seed[seed] = fold_vector_local
        base_by_seed[seed] = derived_base[seed]
        candidate_by_seed[seed] = derived_candidate[seed]

    # The candidate is never used alone: the released column is the incumbent and a
    # small diversification weight is blended in.  Comparing the raw candidate
    # against the base would measure its standalone 7% accuracy deficit (about
    # -0.13 score points) rather than the blend's gain, so the blend weight is
    # selected on the other split seeds and scored on the held-out one.
    nested = nested_blend(actual, folds_by_seed, base_by_seed, candidate_by_seed, spec.alpha_grid)
    raw_cells = paired_cells(actual, folds_by_seed, base_by_seed, candidate_by_seed)
    raw_gains = seed_gains(raw_cells)
    decision = admit(
        nested,
        min_seeds=int(spec.raw["resolution"]["seed_level"]["min_seeds_for_promotion"]),
        positive_cells_min=int(spec.raw["resolution"]["fold_level"]["positive_cells_min"]),
        positive_cells_total=int(spec.raw["resolution"]["fold_level"]["positive_cells_total"]),
    )
    payload = {
        "version": spec.version,
        "stage": "stage2b_independent_seed_replication",
        "candidate": {"kind": kind, "trial_id": trial_id, **dict(meta)},
        "target": target,
        "derived_seeds": [int(s) for s in seeds],
        "recorded_seeds": [int(s) for s in recorded_seeds],
        "protocol": (
            "blend weight selected on the other two split seeds and scored on the held-out "
            "one; the released column is the blend endpoint, never replaced outright"
        ),
        "blend_alphas": {str(k): float(v) for k, v in nested["alphas"].items()},
        "base_per_seed_wmape": {str(s): wmape(actual, base_by_seed[s]) for s in sorted(base_by_seed)},
        "candidate_only_per_seed_wmape": {str(s): wmape(actual, candidate_by_seed[s])
                                          for s in sorted(candidate_by_seed)},
        "candidate_only": {
            "fold_summary": paired_summary([cell.delta_score for cell in raw_cells]),
            "seed_gains": raw_gains,
            "note": "standalone comparison, reported for context only; it is not the candidate",
        },
        "seed_coverage": {str(cell.seed): int(cell.rows)
                          for cell in paired_cells(actual, folds_by_seed, base_by_seed,
                                                   {s: np.asarray(base_by_seed[s]) for s in base_by_seed})},
        "nested": nested,
        "seed_gains": nested["seed_gains"],
        "seed_summary": nested["seed_summary"],
        "decision": decision,
        "new_folds_fitted": int(new_folds),
        "agent_uploads": 0,
    }
    (out / f"replication-{trial_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return payload


def replication_status(root: Path | str, output: Path | str, trial_id: str,
                       seeds: Sequence[int] = (7777, 12011),
                       folds: Sequence[int] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    """Write a self-describing status record for a (possibly unfinished) replication.

    An interrupted replication must remain legible as
    ``NOT_EVALUATED_INCOMPLETE_COVERAGE`` rather than looking like a failure or a
    silent absence.  Only cached folds are listed; no result is claimed.
    """
    root = Path(root).resolve()
    out = _private_output(root, Path(output))
    cached = [
        {"seed": int(seed), "fold": int(fold)}
        for seed in seeds for fold in folds
        if _fold_cache_path(out, int(seed), int(fold)).is_file()
    ]
    expected = [(int(s), int(f)) for s in seeds for f in folds]
    complete = len(cached) == len(expected)
    payload = {
        "candidate": {"kind": "v36", "trial_id": str(trial_id)},
        "derived_seeds": [int(s) for s in seeds],
        "folds_per_seed": len(list(folds)),
        "cached_folds": cached,
        "cached_count": len(cached),
        "expected_count": len(expected),
        "status": "COMPLETE" if complete else "NOT_EVALUATED_INCOMPLETE_COVERAGE",
        "agent_uploads": 0,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"replication-{trial_id}-status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--kind", required=True, choices=["v36", "v3"])
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--target", required=True, choices=list(TARGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7777, 12011])
    parser.add_argument("--base-workers", type=int, default=12)
    parser.add_argument("--max-new-folds", type=int, default=None)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    spec = load_v5_spec(root)
    started = time.time()
    payload = replicate_candidate(root, spec, args.output, args.kind, args.trial_id, args.target,
                                  seeds=tuple(args.seeds), base_workers=args.base_workers,
                                  max_new_folds=args.max_new_folds)
    print(json.dumps({
        "candidate": payload["candidate"],
        "seed_gains": payload["seed_gains"],
        "seed_summary": payload["seed_summary"],
        "admitted": payload["decision"]["admitted"],
        "reasons": payload["decision"]["reasons"],
        "seconds": float(time.time() - started),
    }, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
