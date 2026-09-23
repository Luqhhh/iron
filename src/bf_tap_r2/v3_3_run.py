"""V3.3 structural-search coarse runner."""
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
from .v3_1_search import trial_identity
from .v3_3_models import evaluate_v33_outer_folds
from .v3_3_sampler import sample_v33
from .v3_run import load_fold_vector, load_training_frame

_WORKER_TRAIN: pd.DataFrame | None = None
_WORKER_FOLDS: np.ndarray | None = None


def _init_worker(train: pd.DataFrame, folds: np.ndarray) -> None:
    global _WORKER_TRAIN, _WORKER_FOLDS
    _WORKER_TRAIN, _WORKER_FOLDS = train, folds


def _worker_run(payload: dict) -> dict:
    trial = payload["trial"]
    result = evaluate_v33_outer_folds(_WORKER_TRAIN, _WORKER_FOLDS, trial, fold_ids=tuple(payload["fold_ids"]))
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
        digest = hashlib.sha256()
        for name in ("v3_3_models.py", "v3_3_residual.py", "v3_3_sampler.py", "v3_3_run.py"):
            digest.update((root / "src/bf_tap_r2" / name).read_bytes())
        return digest.hexdigest()


def _completed(ledger: Path) -> dict[str, dict]:
    if not ledger.exists():
        return {}
    out = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "complete":
            out[event["trial_id"]] = event
    return out


def run_coarse(root: Path, output: Path, *, workers: int = 16) -> dict:
    private = (root / "local/runs/round2-v3.3-structure-search").resolve()
    if not output.resolve().is_relative_to(private):
        raise ValueError("V3.3 output must remain private")
    output.mkdir(parents=True, exist_ok=True)
    trials = sample_v33(root)
    train = load_training_frame(root); folds = load_fold_vector(root, train, 42)
    data_hash = _hash_frame(train); fold_hash = _hash_folds(folds, (0, 1)); code_version = _code_version(root)
    identities = {t["trial_id"]: trial_identity(t, batch_id="v33-s1-coarse", data_hash=data_hash,
                                               fold_hash=fold_hash, fold_ids=(0, 1), model_seed=42,
                                               stage="coarse", code_version=code_version) for t in trials}
    (output / "batch.json").write_text(json.dumps({"trials": len(trials), "data_hash": data_hash,
        "fold_hash": fold_hash, "code_version": code_version, "agent_uploads": 0}, ensure_ascii=False, indent=2))
    ledger = output / "fit_ledger.jsonl"; done = _completed(ledger); pending = []
    for trial in trials:
        old = done.get(trial["trial_id"])
        if old is not None:
            if json.dumps(old.get("identity"), sort_keys=True, default=str) != json.dumps(identities[trial["trial_id"]], sort_keys=True, default=str):
                raise ValueError(f"V3.3 cache identity mismatch: {trial['trial_id']}")
            continue
        pending.append(trial)
    results = []
    if pending:
        payloads = [{"trial": t, "identity": identities[t["trial_id"]], "fold_ids": [0, 1]} for t in pending]
        with ProcessPoolExecutor(max_workers=max(1, workers), initializer=_init_worker, initargs=(train, folds)) as ex:
            futs = {ex.submit(_worker_run, p): p for p in payloads}
            for f in as_completed(futs):
                p = futs[f]; trial = p["trial"]
                try:
                    result = f.result()
                except Exception as exc:  # noqa: BLE001
                    with ledger.open("a", encoding="utf-8") as h:
                        h.write(json.dumps({"event": "failed", "trial_id": trial["trial_id"],
                                            "identity": p["identity"], "error_type": type(exc).__name__,
                                            "error": str(exc)}, ensure_ascii=False) + "\n")
                    continue
                pred = np.asarray(result.pop("predictions"), dtype=float)
                np.save(output / f"pred-{trial['trial_id']}.npy", pred)
                record = {"event": "complete", "trial_id": trial["trial_id"], "identity": result["identity"],
                          "target": trial["target"], "line": trial["line"], "kind": trial["kind"],
                          "pooled_wmape": result["pooled_wmape"], "mean_wmape": result["mean_wmape"],
                          "fold_scores": result["fold_scores"], "trial": trial}
                with ledger.open("a", encoding="utf-8") as h:
                    h.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                results.append(record)
                print(json.dumps({"trial_id": trial["trial_id"], "line": trial["line"],
                                  "pooled_wmape": result["pooled_wmape"]}, ensure_ascii=False), flush=True)
    return {"trials": len(trials), "pending_at_start": len(pending), "completed_this_run": len(results),
            "already_complete": len(done), "agent_uploads": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    print(json.dumps(run_coarse(Path.cwd(), args.output, workers=args.workers), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
