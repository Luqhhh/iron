"""Round2 V3.5 append-only runner.

The runner writes only under the private V3.5 run directory.  It never creates a
submission, uploads a package, or overwrites existing evidence.  Trial identity
includes recipe hash, data/fold hashes, source digest and EBM bag protocol hash
so dirty caches are rejected by construction.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import FEATURES, TARGETS
from .v3_1_search import canonical_trial_hash, trial_identity
from .v3_4_bags import protocol_bag_hash
from .v3_5_models import evaluate_v35_outer_folds
from .v3_5_sampler import load_v35_config, sample_v35
from .v3_run import load_fold_vector, load_training_frame

_WORKER_TRAIN: pd.DataFrame | None = None
_WORKER_FOLDS: np.ndarray | None = None


def _init_worker(train: pd.DataFrame, folds: np.ndarray) -> None:
    global _WORKER_TRAIN, _WORKER_FOLDS
    _WORKER_TRAIN, _WORKER_FOLDS = train, folds


def _worker_run(payload: dict) -> dict:
    if _WORKER_TRAIN is None or _WORKER_FOLDS is None:
        raise RuntimeError("V3.5 worker was not initialised")
    result = evaluate_v35_outer_folds(
        _WORKER_TRAIN, _WORKER_FOLDS, payload["trial"], fold_ids=tuple(payload["fold_ids"])
    )
    return {"trial": payload["trial"], "identity": payload["identity"], **result}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_frame(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for value in frame["sample_id"].astype(str).tolist():
        digest.update(value.encode("utf-8"))
        digest.update(b"\x00")
    values = np.ascontiguousarray(frame[[*FEATURES, *TARGETS, "spout_no"]].to_numpy(dtype=float))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _hash_folds(folds: np.ndarray, fold_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(folds, dtype=np.int64).tobytes())
    digest.update(",".join(str(int(v)) for v in fold_ids).encode("ascii"))
    return digest.hexdigest()


def _source_files(root: Path) -> list[Path]:
    files = [
        root / "configs/round2_v3_5/search.yaml",
        root / "src/bf_tap_r2/v3_5_sampler.py",
        root / "src/bf_tap_r2/v3_5_models.py",
        root / "src/bf_tap_r2/v3_5_composition.py",
        root / "src/bf_tap_r2/v3_5_run.py",
    ]
    return [path for path in files if path.exists()]


def _code_version(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(_source_files(root), key=lambda p: str(p)):
        digest.update(path.read_bytes())
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        head = "no-git-head"
    return f"{head}:{digest.hexdigest()}"


def _known_events(ledger: Path) -> set[tuple[str, str]]:
    if not ledger.exists():
        return set()
    known: set[tuple[str, str]] = set()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        known.add((str(event.get("event")), str(event.get("trial_id"))))
    return known


def _completed(ledger: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not ledger.exists():
        return out
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "complete":
            out[str(event["trial_id"])] = event
    return out


def _bag_hash_for_trial(trial: Mapping[str, Any], train: pd.DataFrame) -> str | None:
    kind = str(trial.get("kind"))
    if kind in {"ebm", "ebm_boundary", "ebm_base", "ebm_regularized", "ebm_expression"}:
        params = dict(trial.get("parameters", {}))
        protocol = dict(trial.get("protocol", {}))
        return protocol_bag_hash(
            train,
            n_outer_bags=int(params.get("outer_bags", 4)),
            n_inner_splits=int(protocol.get("inner_splits", 5)),
            seed=int(protocol.get("bag_seed", params.get("random_state", 42))),
        )
    if kind == "global_spout_ebm":
        params = dict(trial.get("parameters", {}))
        parent = dict(params.get("parent_trial", {}))
        parent_params = dict(parent.get("parameters", {}))
        protocol = dict(parent.get("protocol", trial.get("protocol", {})))
        return protocol_bag_hash(
            train,
            n_outer_bags=int(parent_params.get("outer_bags", 4)),
            n_inner_splits=int(protocol.get("inner_splits", 5)),
            seed=int(protocol.get("bag_seed", parent_params.get("random_state", 42))),
        )
    return None


def v35_trial_identity(trial: Mapping[str, Any], *, batch_id: str, data_hash: str,
                       fold_hash: str, fold_ids: Sequence[int], model_seed: int,
                       stage: str, code_version: str, source_hash: str,
                       bag_hash: str | None = None) -> dict:
    base = trial_identity(
        trial,
        batch_id=batch_id,
        data_hash=data_hash,
        fold_hash=fold_hash,
        fold_ids=fold_ids,
        model_seed=model_seed,
        stage=stage,
        code_version=code_version,
    )
    base.update({
        "line": str(trial.get("line")),
        "kind": str(trial.get("kind")),
        "source_hash": str(source_hash),
        "bag_hash": None if bag_hash is None else str(bag_hash),
        "training_protocol": str(trial.get("protocol", {}).get("bags", "ebm-group-safe-bags-v1")),
        "component_key": (
            None if not trial.get("parameters", {}).get("component_key")
            else str(trial["parameters"]["component_key"])
        ),
        "repair_of": str(trial.get("repair_of", "")) or None,
    })
    return base


def _identity_matches(stored: Mapping[str, Any], desired: Mapping[str, Any]) -> bool:
    return json.dumps(stored, sort_keys=True, default=str) == json.dumps(desired, sort_keys=True, default=str)


def _private_output(root: Path, output: Path) -> None:
    private = (root / "local/runs/round2-v3.5-regularized-ebm-and-composition").resolve()
    if not output.resolve().is_relative_to(private):
        raise ValueError(
            "V3.5 output must remain private under "
            "local/runs/round2-v3.5-regularized-ebm-and-composition"
        )


def run_batch(root: Path, output: Path, *, stage: str, fold_seed: int,
              fold_ids: Sequence[int], workers: int = 16,
              limit: int | None = None, trial_ids: Sequence[str] | None = None) -> dict:
    _private_output(root, output)
    output.mkdir(parents=True, exist_ok=True)
    config = load_v35_config(root)
    trials = sample_v35(root)
    if trial_ids is not None:
        wanted = {str(value) for value in trial_ids}
        trials = [trial for trial in trials if trial["trial_id"] in wanted]
    if limit is not None:
        trials = trials[: max(0, int(limit))]
    if not trials:
        raise ValueError("No V3.5 trials selected")
    train = load_training_frame(root)
    folds = load_fold_vector(root, train, int(fold_seed))
    data_hash = _hash_frame(train)
    fold_hash = _hash_folds(folds, tuple(fold_ids))
    code_version = _code_version(root)
    source_hash = _sha256_bytes(code_version.encode("utf-8"))
    ledger = output / "fit_ledger.jsonl"
    completed = _completed(ledger)
    known = _known_events(ledger)
    pending: list[dict] = []
    scheduled: list[dict] = []
    identities: dict[str, dict] = {}
    for trial in trials:
        status = trial.get("status", "available")
        if status == "not_applicable":
            if ("not_applicable", trial["trial_id"]) not in known:
                with ledger.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "event": "not_applicable",
                        "trial_id": trial["trial_id"],
                        "line": trial["line"],
                        "target": trial.get("target"),
                        "reason": trial.get("reason", ""),
                    }, ensure_ascii=False) + "\n")
            continue
        if status == "duplicate":
            if ("duplicate", trial["trial_id"]) not in known:
                with ledger.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "event": "duplicate",
                        "trial_id": trial["trial_id"],
                        "duplicate_of": trial.get("duplicate_of"),
                    }, ensure_ascii=False) + "\n")
            continue
        bag_hash = _bag_hash_for_trial(trial, train)
        identity = v35_trial_identity(
            trial,
            batch_id=f"v35-{stage}-seed{fold_seed}",
            data_hash=data_hash,
            fold_hash=fold_hash,
            fold_ids=tuple(fold_ids),
            model_seed=int(fold_seed),
            stage=stage,
            code_version=code_version,
            source_hash=source_hash,
            bag_hash=bag_hash,
        )
        identities[trial["trial_id"]] = identity
        old = completed.get(trial["trial_id"])
        if old is not None:
            if not _identity_matches(old.get("identity", {}), identity):
                raise ValueError(f"V3.5 cache identity mismatch: {trial['trial_id']}")
            continue
        pending.append(trial)
        scheduled.append(trial)
    batch_manifest = {
        "version": config["version"],
        "stage": stage,
        "fold_seed": int(fold_seed),
        "fold_ids": [int(v) for v in fold_ids],
        "data_hash": data_hash,
        "fold_hash": fold_hash,
        "code_version": code_version,
        "source_hash": source_hash,
        "slots_selected": len(trials),
        "pending_at_start": len(pending),
        "agent_uploads": 0,
    }
    (output / "batch.json").write_text(
        json.dumps(batch_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results: list[dict] = []
    if pending:
        payloads = [
            {"trial": trial, "identity": identities[trial["trial_id"]],
             "fold_ids": [int(v) for v in fold_ids]}
            for trial in pending
        ]
        with ProcessPoolExecutor(
            max_workers=max(1, int(workers)), initializer=_init_worker, initargs=(train, folds)
        ) as executor:
            futures = {executor.submit(_worker_run, payload): payload for payload in payloads}
            for future in as_completed(futures):
                payload = futures[future]
                trial = payload["trial"]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - preserve failure evidence
                    with ledger.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "event": "failed",
                            "trial_id": trial["trial_id"],
                            "identity": payload["identity"],
                            "line": trial.get("line"),
                            "target": trial.get("target"),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }, ensure_ascii=False) + "\n")
                    continue
                prediction = np.asarray(result.pop("predictions"), dtype=float)
                np.save(output / f"pred-{trial['trial_id']}.npy", prediction)
                record = {
                    "event": "complete",
                    "trial_id": trial["trial_id"],
                    "identity": result["identity"],
                    "target": trial["target"],
                    "line": trial["line"],
                    "kind": trial["kind"],
                    "status": trial.get("status", "available"),
                    "pooled_wmape": result["pooled_wmape"],
                    "mean_wmape": result["mean_wmape"],
                    "fold_scores": result["fold_scores"],
                    "fit_meta": result["fit_meta"],
                    "trial": trial,
                    "agent_uploads": 0,
                }
                with ledger.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                results.append(record)
                print(json.dumps({
                    "trial_id": trial["trial_id"],
                    "line": trial["line"],
                    "target": trial["target"],
                    "pooled_wmape": result["pooled_wmape"],
                }, ensure_ascii=False), flush=True)
    return {
        "status": "complete",
        "stage": stage,
        "slots_selected": len(trials),
        "pending_at_start": len(pending),
        "completed_this_run": len(results),
        "already_complete": sum(1 for trial in scheduled if trial["trial_id"] in completed),
        "fold_seed": int(fold_seed),
        "fold_ids": [int(v) for v in fold_ids],
        "workers": int(workers),
        "agent_uploads": 0,
    }


def _read_records(ledger: Path) -> list[dict]:
    if not ledger.exists():
        return []
    out: list[dict] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "complete":
            out.append(event)
    return out


def select_refine_trials(coarse_records: Sequence[dict], max_per_target: int = 12) -> list[dict]:
    """Deterministic coarse shortlist.

    The full V3.5 refinement-selection policy is available in the private run
    driver; this public helper provides the mechanical guarded fallback used when
    the optional L1 development predictions are unavailable.  It never
    fabricates diversity: absent qualified structures simply shorten the list.
    """
    selected: list[dict] = []
    for target in TARGETS:
        rows = [r for r in coarse_records if r.get("target") == target]
        rows.sort(key=lambda r: (float(r.get("pooled_wmape", float("inf"))), str(r.get("trial_id"))))
        seen: set[str] = set()
        for row in rows:
            trial = row.get("trial")
            if not isinstance(trial, Mapping):
                continue
            key = canonical_trial_hash(trial)
            if key in seen:
                continue
            seen.add(key)
            selected.append(dict(trial))
            if sum(1 for item in selected if item.get("target") == target) >= int(max_per_target):
                break
    return selected


def run_refine(root: Path, coarse_ledger: Path, output: Path, *,
               seeds: Sequence[int] = (42, 3407), workers: int = 16,
               max_per_target: int = 12) -> dict:
    _private_output(root, output)
    output.mkdir(parents=True, exist_ok=True)
    coarse = _read_records(coarse_ledger)
    selected = select_refine_trials(coarse, max_per_target=max_per_target)
    if not selected:
        raise ValueError("No coarse records available for V3.5 refine")
    summary = {
        "refine_selected": len(selected),
        "seeds": [int(s) for s in seeds],
        "completed_per_seed": {},
        "agent_uploads": 0,
    }
    for seed in seeds:
        seed_output = output / f"seed-{int(seed)}"
        seed_records = run_batch(
            root, seed_output, stage=f"refine-seed{int(seed)}", fold_seed=int(seed),
            fold_ids=(0, 1, 2, 3, 4), workers=workers,
            trial_ids=[str(record["trial_id"]) for record in selected],
        )
        summary["completed_per_seed"][str(int(seed))] = seed_records
    (output / "refine_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    coarse = sub.add_parser("coarse")
    coarse.add_argument("--output", type=Path, required=True)
    coarse.add_argument("--workers", type=int, default=32)
    coarse.add_argument("--limit", type=int, default=None)
    coarse.add_argument("--fold-seed", type=int, default=42)
    refine = sub.add_parser("refine")
    refine.add_argument("--coarse-ledger", type=Path, required=True)
    refine.add_argument("--output", type=Path, required=True)
    refine.add_argument("--workers", type=int, default=32)
    refine.add_argument("--max-per-target", type=int, default=12)
    args = parser.parse_args(argv)
    root = Path.cwd()
    if args.command == "coarse":
        summary = run_batch(
            root, args.output, stage="coarse", fold_seed=int(args.fold_seed),
            fold_ids=(0, 1), workers=int(args.workers), limit=args.limit,
        )
    elif args.command == "refine":
        summary = run_refine(
            root, args.coarse_ledger, args.output, workers=int(args.workers),
            max_per_target=int(args.max_per_target),
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
