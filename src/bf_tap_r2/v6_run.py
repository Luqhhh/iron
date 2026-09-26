"""Round2 V6 runner: fit V6 trials on outer folds with the frozen V3.6 evaluator.

The V6 trial specifications are produced by :mod:`bf_tap_r2.v6_sampler`; their
``kind``, ``structure``, ``capacity_name`` and ``parameters`` follow the frozen
V3.6 convention, so the frozen evaluator
(:func:`bf_tap_r2.v3_6_models.evaluate_v36_outer_folds`) fits and scores them
without any new training code.  The evaluator fits on the outer training part
only and never sees validation labels.

Writes ``pred-<trial>.npy`` plus an append-only ``fit_ledger.jsonl`` per split
seed under ``local/runs/round2-v6-iron-capacity-networks/``.  Resumable by
identity; nothing here packages or uploads.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .v5_library import fold_vector, load_v5_training_frame
from .v6_sampler import sample_v6_iron_capacity_trials
from .v6_spec import V6Spec, load_v6_spec

__all__ = ["read_completed", "run_v6_batch", "main"]

DEFAULT_OUTPUT = "local/runs/round2-v6-iron-capacity-networks/probe-r1"

_WORKER_TRAIN = None
_WORKER_FOLDS = None


def _init_worker(train, folds) -> None:
    global _WORKER_TRAIN, _WORKER_FOLDS
    _WORKER_TRAIN, _WORKER_FOLDS = train, folds


def _worker_run(payload: Mapping[str, Any]) -> dict[str, Any]:
    from .v3_6_models import evaluate_v36_outer_folds

    if _WORKER_TRAIN is None or _WORKER_FOLDS is None:
        raise RuntimeError("V6 worker was not initialised")
    return evaluate_v36_outer_folds(_WORKER_TRAIN, _WORKER_FOLDS, payload["trial"],
                                    fold_ids=tuple(payload["fold_ids"]))


def _private_output(root: Path, output: Path) -> Path:
    allowed = (root / "local/runs/round2-v6-iron-capacity-networks").resolve()
    resolved = (output if output.is_absolute() else root / output).resolve()
    if not resolved.is_relative_to(allowed):
        raise ValueError(f"V6 output must stay private under {allowed}")
    return resolved


def read_completed(ledger: Path) -> dict[str, dict[str, Any]]:
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


def run_v6_batch(root: Path | str, output: Path | str, *, seed: int, folds: Sequence[int],
                 trial_ids: Sequence[str] | None = None, workers: int = 8,
                 stage: str = "v6-probe") -> dict[str, Any]:
    """Fit the selected V6 trials on ``folds`` of ``seed`` and cache the predictions."""
    from .v3_6_run import _bag_hash_for_trial, _code_version, _hash_folds, _hash_frame

    root = Path(root).resolve()
    out = _private_output(root, Path(output))
    spec: V6Spec = load_v6_spec(root)
    train = load_v5_training_frame(root)
    trials = {str(t["trial_id"]): t for t in sample_v6_iron_capacity_trials(root)}
    probe = [str(v) for v in spec.budget["stage_a2_iron_capacity_screen"]["probe"]["trial_ids"]]
    # the probe stage defaults to exactly the authorised probe trials
    requested = [str(v) for v in (trial_ids if trial_ids is not None
                                 else (probe if stage == "v6-probe" else sorted(trials)))]
    unknown = sorted(set(requested) - set(trials))
    if unknown:
        raise KeyError(f"V6 batch requested unknown trials: {unknown}")
    if stage == "v6-probe":
        unauthorised = sorted(set(requested) - set(probe))
        if unauthorised:
            raise ValueError(
                f"the probe stage is limited to {probe}; unauthorised trials: {unauthorised}")

    fold_vector_local = fold_vector(root, train, int(seed), None)
    data_hash = _hash_frame(train)
    fold_hash = _hash_folds(fold_vector_local, tuple(int(v) for v in folds))
    code_version = _code_version(root)
    ledger = out / "fit_ledger.jsonl"
    completed = read_completed(ledger)
    pending: list[dict[str, Any]] = []
    identities: dict[str, dict[str, Any]] = {}
    for trial_id in requested:
        trial = trials[trial_id]
        identity = {
            "trial_id": trial_id,
            "stage": str(stage),
            "fold_seed": int(seed),
            "fold_ids": [int(v) for v in folds],
            "data_hash": data_hash,
            "fold_hash": fold_hash,
            "code_version": code_version,
            "bag_hash": _bag_hash_for_trial(trial, train),
            "version": str(trial.get("version")),
        }
        identities[trial_id] = identity
        old = completed.get(trial_id)
        if old is not None:
            if old.get("identity", {}) != identity:
                raise ValueError(f"V6 cache identity mismatch: {trial_id}")
            continue
        pending.append({"trial": trial, "fold_ids": [int(v) for v in folds]})

    out.mkdir(parents=True, exist_ok=True)
    (out / "batch.json").write_text(json.dumps({
        "version": spec.version,
        "stage": str(stage),
        "fold_seed": int(seed),
        "fold_ids": [int(v) for v in folds],
        "data_hash": data_hash,
        "fold_hash": fold_hash,
        "code_version": code_version,
        "slots_selected": len(requested),
        "pending_at_start": len(pending),
        "agent_uploads": 0,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    results: list[dict[str, Any]] = []
    started = time.time()
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
                np.save(out / f"pred-{trial_id}.npy", prediction)
                record = {**result, "identity": identities[trial_id],
                          "family": f"{trial['structure']}|{trial['capacity_name']}",
                          "structure": str(trial["structure"]),
                          "capacity_name": str(trial["capacity_name"]),
                          "training_setting": str(trial["training_setting"])}
                with ledger.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"event": "complete", **record},
                                            ensure_ascii=False, default=float) + "\n")
                results.append(record)
                print(json.dumps({"trial_id": trial_id, "family": record["family"],
                                  "mean_wmape": record.get("mean_wmape")},
                                 ensure_ascii=False), flush=True)
    return {
        "status": "complete",
        "stage": str(stage),
        "fold_seed": int(seed),
        "fold_ids": [int(v) for v in folds],
        "selected": len(requested),
        "pending_at_start": len(pending),
        "completed_this_run": len(results),
        "already_completed": len(completed),
        "seconds": float(time.time() - started),
        "directory": str(out.relative_to(root)),
        "agent_uploads": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--trial-ids", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stage", default="v6-probe")
    args = parser.parse_args(argv)
    summary = run_v6_batch(args.root, args.output, seed=args.seed, folds=tuple(args.folds),
                           trial_ids=args.trial_ids, workers=args.workers, stage=args.stage)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
