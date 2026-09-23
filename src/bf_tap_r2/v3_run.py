"""Execution helpers for the Round2 V3 local-search batches.

The runner is append-only and private-run-only: it never writes a submission,
never uploads, and refuses to overwrite an existing trial record.  Full
platform release checks remain in the V2 release path.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from .data import TARGETS
from .splits import make_folds
from .v3_local_search import (
    XGBRegressor,
    evaluate_trial_folds,
    load_spec,
    sample_trials,
)

_WORKER_TRAIN: pd.DataFrame | None = None
_WORKER_FOLDS: np.ndarray | None = None
_WORKER_EARLY = True
_WORKER_FOLD_IDS: tuple[int, ...] = (0, 1)


def _init_worker(train: pd.DataFrame, folds: np.ndarray, early_stopping: bool,
                 fold_ids: tuple[int, ...]) -> None:
    global _WORKER_TRAIN, _WORKER_FOLDS, _WORKER_EARLY, _WORKER_FOLD_IDS
    _WORKER_TRAIN = train
    _WORKER_FOLDS = folds
    _WORKER_EARLY = bool(early_stopping)
    _WORKER_FOLD_IDS = tuple(int(v) for v in fold_ids)


def _worker_run(trial: dict) -> dict:
    if _WORKER_TRAIN is None or _WORKER_FOLDS is None:
        raise RuntimeError("V3 worker was not initialised")
    return evaluate_trial_folds(_WORKER_TRAIN, _WORKER_FOLDS, trial,
                                fold_ids=_WORKER_FOLD_IDS,
                                early_stopping=_WORKER_EARLY)


def load_training_frame(root: Path) -> pd.DataFrame:
    from .v2_release import load_v2
    return load_v2(root / "复赛_train", "train", 2754)


def load_fold_vector(root: Path, train: pd.DataFrame, seed: int) -> np.ndarray:
    frozen = root / "local/runs/round2-v2/comparison-r1" / f"folds-{seed}.csv"
    if frozen.exists():
        assignment = pd.read_csv(frozen, dtype={"group_id": str})
        if not assignment.sample_id.is_unique or set(assignment.sample_id) != set(train.sample_id):
            raise ValueError("V3 frozen fold identity mismatch")
        a = assignment.set_index("sample_id").loc[train.sample_id]
        if set(a.fold) != set(range(5)) or not (a.seed == seed).all():
            raise ValueError("V3 frozen fold vector is invalid")
        return a.fold.to_numpy()
    # New V3 confirmation seeds are derived with the same deterministic
    # stratified-group routine used for the original 42/3407 folds.
    assignment = make_folds(train, seed).set_index("sample_id").loc[train.sample_id]
    if set(assignment.fold) != set(range(5)) or not (assignment.seed == seed).all():
        raise ValueError("V3 derived fold vector is invalid")
    return assignment.fold.to_numpy()



def read_complete_records(ledger: Path) -> list[dict]:
    if not ledger.exists():
        return []
    records = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "complete":
            records.append(event)
    return records


def select_coarse_trials(ledger: Path, keep_per_target: int = 12) -> list[dict]:
    """Select top coarse trials per target, preserving their frozen specs."""
    records = read_complete_records(ledger)
    selected: list[dict] = []
    for target in TARGETS:
        rows = [r for r in records if r.get("target") == target]
        rows.sort(key=lambda r: (float(r["mean_wmape"]), str(r["trial_id"])))
        selected.extend(rows[: max(0, int(keep_per_target))])
    return selected


def trial_details(record: dict) -> dict:
    trial = record.get("trial")
    if not isinstance(trial, dict) or "trial_id" not in trial:
        raise ValueError(f"Coarse record lacks full trial spec: {record.get('trial_id')}")
    return dict(trial)


def _check_private_output(root: Path, output: Path) -> None:
    if not output.resolve().is_relative_to((root / "local/runs/round2-v3-local-search").resolve()):
        raise ValueError("V3 execution output must remain private")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _already_completed(ledger: Path) -> set[str]:
    if not ledger.exists():
        return set()
    done = set()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "complete":
            done.add(event["trial_id"])
    return done


def run_batch(root: Path, output: Path, family: str | None, limit: int | None,
              seed: int, fold_ids: tuple[int, ...], early_stopping: bool,
              workers: int, fold_seed: int = 42) -> dict:
    _check_private_output(root, output)
    output.mkdir(parents=True, exist_ok=True)
    spec = load_spec(root)
    trials = sample_trials(spec, seed=seed)
    if family is not None:
        trials = [t for t in trials if t["family"] == family]
    if limit is not None:
        trials = trials[: max(0, int(limit))]
    if not trials:
        raise ValueError("No V3 trials selected")
    ledger = output / "fit_ledger.jsonl"
    completed = _already_completed(ledger)
    pending = [t for t in trials if t["trial_id"] not in completed]
    skipped_xgb = [t for t in pending if t["family"] == "xgboost" and XGBRegressor is None]
    if skipped_xgb:
        with ledger.open("a", encoding="utf-8") as handle:
            for trial in skipped_xgb:
                handle.write(json.dumps({
                    "event": "blocked",
                    "trial_id": trial["trial_id"],
                    "family": trial["family"],
                    "reason": "xgboost is not installed in the locked V3 environment",
                    "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }, ensure_ascii=False) + "\n")
        pending = [t for t in pending if t not in skipped_xgb]

    train = load_training_frame(root)
    # A single frozen fold seed is used for one batch; refine runs are launched
    # separately for additional seeds.
    folds = load_fold_vector(root, train, int(fold_seed))

    results = []
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, int(workers)),
                                 initializer=_init_worker,
                                 initargs=(train, folds, early_stopping, tuple(fold_ids))) as executor:
            futures = {executor.submit(_worker_run, trial): trial for trial in pending}
            for future in as_completed(futures):
                trial = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - failure evidence is preserved
                    with ledger.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "event": "failed", "trial_id": trial["trial_id"],
                            "family": trial["family"], "target": trial["target"],
                            "error_type": type(exc).__name__, "error": str(exc),
                            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }, ensure_ascii=False) + "\n")
                    continue
                prediction = np.asarray(result.pop("predictions"), dtype=float)
                np.save(output / f"pred-{trial['trial_id']}.npy", prediction)
                record = {**_jsonable(result), "trial": _jsonable(trial)}
                with ledger.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"event": "complete", **record}, ensure_ascii=False,
                                            allow_nan=False) + "\n")
                (output / f"trial-{trial['trial_id']}.json").write_text(
                    json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
                results.append(record)
                print(json.dumps({"trial_id": trial["trial_id"], "family": trial["family"],
                                  "target": trial["target"], "mean_wmape": record["mean_wmape"]},
                                 ensure_ascii=False), flush=True)
    return {
        "status": "complete",
        "selected_trials": len(trials),
        "pending_at_start": len(pending),
        "completed_this_run": len(results),
        "already_completed": len(completed),
        "blocked_xgboost": len(skipped_xgb),
        "fold_seed": int(fold_seed),
        "fold_ids": list(fold_ids),
        "early_stopping": bool(early_stopping),
        "workers": int(workers),
        "platform_uploads": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", default=None,
                        choices=["catboost", "lightgbm", "xgboost", "mlp", "kernel", "expression"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    summary = run_batch(Path(args.root), Path(args.output), args.family, args.limit,
                        args.seed, tuple(args.folds), not args.no_early_stopping, args.workers,
                        fold_seed=args.fold_seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
