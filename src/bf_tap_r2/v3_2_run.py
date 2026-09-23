"""Round2 V3.2 coarse runner: 256 directed configurations, seed-42 folds 0/1."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from .data import FEATURES, TARGETS
from .v3_1_models import evaluate_v31_outer_folds
from .v3_1_search import load_config, trial_identity
from .v3_2_sampler import sample_v32
from .v3_run import load_fold_vector, load_training_frame

_WORKER_TRAIN: pd.DataFrame | None = None
_WORKER_FOLDS: np.ndarray | None = None


def _init_worker(train: pd.DataFrame, folds: np.ndarray) -> None:
    global _WORKER_TRAIN, _WORKER_FOLDS
    _WORKER_TRAIN = train
    _WORKER_FOLDS = folds


def _worker_run(payload: dict) -> dict:
    if _WORKER_TRAIN is None or _WORKER_FOLDS is None:
        raise RuntimeError("V3.2 worker not initialised")
    trial = payload["trial"]
    result = evaluate_v31_outer_folds(_WORKER_TRAIN, _WORKER_FOLDS, trial,
                                      fold_ids=tuple(payload["fold_ids"]),
                                      inner_seed=int(payload["inner_seed"]))
    return {"trial": trial, "identity": payload["identity"], **result}


def _hash_frame(frame: pd.DataFrame) -> str:
    values = np.ascontiguousarray(frame[[*FEATURES, *TARGETS, "spout_no"]].to_numpy(dtype=float))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _hash_folds(folds: np.ndarray, fold_ids: tuple[int, ...]) -> str:
    payload = np.asarray([int(v) for v in fold_ids], dtype=np.int64).tobytes() + np.ascontiguousarray(folds).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _code_version(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        files = [root / "src/bf_tap_r2/v3_1_models.py", root / "src/bf_tap_r2/v3_2_sampler.py",
                 root / "src/bf_tap_r2/v3_2_run.py"]
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.read_bytes())
        return digest.hexdigest()


def _already_complete(ledger: Path) -> dict[str, dict]:
    found: dict[str, dict] = {}
    if not ledger.exists():
        return found
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "complete":
            found[event["trial_id"]] = event
    return found


def run_coarse(root: Path, output: Path, *, sample_seed: int = 20260924,
               fold_seed: int = 42, fold_ids: tuple[int, ...] = (0, 1),
               workers: int = 16, limit: int | None = None) -> dict:
    private_root = (root / "local/runs/round2-v3.2-ensemble-and-target-search").resolve()
    if not output.resolve().is_relative_to(private_root):
        raise ValueError("V3.2 output must remain under the private V3.2 run root")
    output.mkdir(parents=True, exist_ok=True)
    trials = sample_v32(root, seed=sample_seed)
    if limit is not None:
        trials = trials[: max(0, int(limit))]
    train = load_training_frame(root)
    folds = load_fold_vector(root, train, fold_seed)
    data_hash = _hash_frame(train)
    fold_hash = _hash_folds(folds, tuple(fold_ids))
    code_version = _code_version(root)
    batch_id = f"v32-s1-coarse-{sample_seed}"
    identities = {t["trial_id"]: trial_identity(
        t, batch_id=batch_id, data_hash=data_hash, fold_hash=fold_hash,
        fold_ids=fold_ids, model_seed=42, stage="coarse", code_version=code_version
    ) for t in trials}
    (output / "batch.json").write_text(json.dumps({
        "batch_id": batch_id, "sample_seed": sample_seed, "fold_seed": fold_seed,
        "fold_ids": list(fold_ids), "data_hash": data_hash, "fold_hash": fold_hash,
        "code_version": code_version, "trials": len(trials), "workers": workers,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    trial_path = output / "trials.jsonl"
    if not trial_path.exists():
        with trial_path.open("x", encoding="utf-8") as handle:
            for trial in trials:
                handle.write(json.dumps(trial, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    ledger = output / "fit_ledger.jsonl"
    completed = _already_complete(ledger)
    pending = []
    for trial in trials:
        tid = trial["trial_id"]
        old = completed.get(tid)
        if old is not None:
            if json.dumps(old.get("identity"), sort_keys=True, default=str) != json.dumps(identities[tid], sort_keys=True, default=str):
                raise ValueError(f"V3.2 cache identity mismatch for {tid}")
            continue
        pending.append(trial)
    results = []
    if pending:
        payloads = [{"trial": t, "identity": identities[t["trial_id"]],
                     "fold_ids": list(fold_ids), "inner_seed": 7000 + i}
                    for i, t in enumerate(pending)]
        with ProcessPoolExecutor(max_workers=max(1, int(workers)), initializer=_init_worker,
                                 initargs=(train, folds)) as executor:
            futures = {executor.submit(_worker_run, p): p for p in payloads}
            for future in as_completed(futures):
                payload = futures[future]
                trial = payload["trial"]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    with ledger.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "event": "failed", "trial_id": trial["trial_id"],
                            "identity": payload["identity"], "error_type": type(exc).__name__,
                            "error": str(exc), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }, ensure_ascii=False) + "\n")
                    continue
                predictions = np.asarray(result.pop("predictions"), dtype=float)
                np.save(output / f"pred-{trial['trial_id']}.npy", predictions)
                record = {
                    "event": "complete", "trial_id": trial["trial_id"],
                    "identity": result["identity"], "target": trial["target"], "line": trial.get("line"),
                    "family": trial["family"], "pooled_wmape": result["pooled_wmape"],
                    "mean_wmape": result["mean_wmape"], "fold_scores": result["fold_scores"],
                    "fit_meta": result["fit_meta"], "trial": trial,
                }
                with ledger.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                (output / f"trial-{trial['trial_id']}.json").write_text(
                    json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
                results.append(record)
                print(json.dumps({"trial_id": trial["trial_id"], "target": trial["target"],
                                  "line": trial.get("line"), "pooled_wmape": result["pooled_wmape"]},
                                 ensure_ascii=False), flush=True)
    return {"batch_id": batch_id, "trials": len(trials), "pending_at_start": len(pending),
            "completed_this_run": len(results), "already_complete": len(completed),
            "agent_uploads": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-seed", type=int, default=20260924)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(run_coarse(args.root, args.output, sample_seed=args.sample_seed,
                                fold_seed=args.fold_seed, fold_ids=tuple(args.folds),
                                workers=args.workers, limit=args.limit), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
