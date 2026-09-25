"""Append-only V4.3 strong-base time-residual runner.

Each task cross-fits the frozen parent inside one outer training part, fits the
frozen V4.1 candidate pool on the resulting residual targets, and scores the
corrected prediction against the recorded parent prediction on the outer
validation fold.  No package is generated, nothing is uploaded, and every run
directory is append-only.
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

from .data import FEATURES
from .metrics import wmape
from .v3_run import load_fold_vector, load_training_frame
from .v4_3_parent import (
    PARENT_DEVELOPMENT_SCORE,
    load_recorded_parent_oof,
    nested_parent_oof,
)
from .v4_3_residual import (
    TARGET,
    candidate_specs,
    evaluate_candidate_corrections,
    load_v43_config,
    make_corrector,
)

PRIVATE_ROOT = "local/runs/round2-v4.3-strong-base-time-residual"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _data_hash(train: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for value in train["sample_id"].astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\x00")
    values = train.loc[:, [*FEATURES, "spout_no", "tap_iron", "tap_time_len"]].to_numpy(dtype=float)
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _source_hash(root: Path) -> str:
    paths = [
        root / "configs/round2_v4_3/experiment.yaml",
        root / "src/bf_tap_r2/v4_3_parent.py",
        root / "src/bf_tap_r2/v4_3_residual.py",
        root / "src/bf_tap_r2/v4_3_run.py",
        root / "src/bf_tap_r2/v3_6_reference.py",
        root / "src/bf_tap_r2/v3_4_models.py",
        root / "src/bf_tap_r2/v3_4_bags.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover - diagnostic only
        return "no-git-head"


def _private_output(root: Path, output: Path) -> Path:
    private = (root / PRIVATE_ROOT).resolve()
    resolved = output.resolve()
    if not resolved.is_relative_to(private):
        raise ValueError("V4.3 output must remain under its private local run root")
    return resolved


def _prepare_output(root: Path, output: Path) -> None:
    resolved = _private_output(root, output)
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"V4.3 output is append-only and already nonempty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)


def evaluate_fold_seed(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Cross-fit the parent, fit every candidate, and score one (seed, fold)."""
    root = Path(str(payload["root"]))
    seed = int(payload["seed"])
    fold_id = int(payload["fold_id"])
    config = load_v43_config(root)
    train = load_training_frame(root)
    outer_folds = load_fold_vector(root, train, seed)
    specs = candidate_specs(config)

    started = time.perf_counter()
    nested = nested_parent_oof(root, config, train, seed=seed, fold_id=fold_id)
    nested_seconds = time.perf_counter() - started
    training_mask = np.asarray(nested["training_mask"], dtype=bool)
    validation_mask = outer_folds == fold_id
    if training_mask.sum() + validation_mask.sum() != len(train):
        raise ValueError("V4.3 training/validation masks do not cover the frame")

    training = train.loc[training_mask].reset_index(drop=True)
    validation = train.loc[validation_mask].reset_index(drop=True)
    y_training = training[TARGET].to_numpy(dtype=float)
    y_validation = validation[TARGET].to_numpy(dtype=float)
    recorded, _members = load_recorded_parent_oof(root, train, config, seed)
    parent_validation = np.asarray(recorded[validation_mask], dtype=float)
    parent_oof = np.asarray(nested["parent_oof"], dtype=float)

    if not np.isfinite(parent_validation).all() or not np.isfinite(parent_oof).all():
        raise ValueError("V4.3 parent predictions are incomplete")
    parent_wmape = float(wmape(y_validation, parent_validation))

    corrections: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    fold_records: list[dict[str, Any]] = []
    for spec in specs:
        corrector = make_corrector(spec).fit(training, parent_oof, y_training)
        correction = corrector.predict_correction(validation, parent_validation)
        prediction = evaluate_candidate_corrections(validation, y_validation, parent_validation,
                                                    correction, clip_at_zero=True)
        name = str(spec["name"])
        corrections[name] = np.asarray(correction, dtype=float)
        predictions[name] = prediction
        score = float(wmape(y_validation, prediction))
        fold_records.append({
            "candidate_id": str(spec["candidate_id"]),
            "name": name,
            "family": str(spec["family"]),
            "wmape": score,
            "package_delta_single": float(50.0 * (parent_wmape - score)),
            "correction_abs_mean": float(np.mean(np.abs(correction))),
            "correction_clip": float(getattr(corrector, "clip_", float("nan"))),
            "fit_meta": _jsonable(corrector.fit_meta_),
        })

    arrays = {
        "sample_id": np.asarray(validation["sample_id"].astype(str).tolist(), dtype=np.str_),
        "target": y_validation,
        "parent": parent_validation,
        "fold": np.full(len(validation), fold_id, dtype=int),
    }
    for name, values in predictions.items():
        arrays[f"candidate__{name}"] = values
    for name, values in corrections.items():
        arrays[f"correction__{name}"] = values
    arrays["nested_parent_oof"] = parent_oof
    arrays["nested_sample_id"] = np.asarray(training["sample_id"].astype(str).tolist(), dtype=np.str_)

    return {
        "seed": seed,
        "fold_id": fold_id,
        "parent_wmape": parent_wmape,
        "training_rows": int(training_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "inner_seed": int(nested["inner_seed"]),
        "inner_fold_hash": str(nested["inner_fold_hash"]),
        "inner_group_hash": str(nested["inner_group_hash"]),
        "l1_weights": [float(value) for value in np.asarray(nested["l1_weights"])],
        "l1_objective": float(nested["l1_objective"]),
        "nested_fit_count": int(nested["fit_count"]),
        "nested_seconds": float(nested_seconds),
        "total_seconds": float(time.perf_counter() - started),
        "records": fold_records,
        "arrays": arrays,
    }


def _pooled_signature(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool evaluated folds within one seed and score parent vs candidate."""
    y = np.concatenate([np.asarray(row["target"], dtype=float) for row in rows])
    parent = np.concatenate([np.asarray(row["parent"], dtype=float) for row in rows])
    parent_wmape = float(wmape(y, parent))
    names = sorted({str(key) for row in rows for key in row["records_by_name"]})
    candidates: dict[str, Any] = {}
    for name in names:
        values = np.concatenate([np.asarray(row["records_by_name"][name], dtype=float) for row in rows])
        score = float(wmape(y, values))
        candidates[name] = {
            "wmape": score,
            "package_delta_single": float(50.0 * (parent_wmape - score)),
        }
    return {"parent_wmape": parent_wmape, "candidates": candidates, "rows": int(len(y))}


def _summarize(results: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
               fold_ids: Sequence[int]) -> dict[str, Any]:
    seeds = [int(value) for value in config["protocol"]["split_seeds"]]
    threshold = float(config["continuation_gate"]["mean_package_delta_single_min"])
    local_gate = float(config["continuation_gate"]["local_working_gate"])
    by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in seeds}
    for result in results:
        records_by_name = {
            str(record["name"]): result["arrays"][f"candidate__{record['name']}"]
            for record in result["records"]
        }
        by_seed[int(result["seed"])].append({
            "fold": int(result["fold_id"]),
            "target": result["arrays"]["target"],
            "parent": result["arrays"]["parent"],
            "records_by_name": records_by_name,
        })
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        rows = sorted(by_seed[seed], key=lambda row: int(row["fold"]))
        if [int(row["fold"]) for row in rows] != [int(value) for value in fold_ids]:
            raise ValueError(f"V4.3 seed {seed} fold coverage is incomplete")
        per_seed[str(seed)] = _pooled_signature(rows)

    names = sorted(per_seed[str(seeds[0])]["candidates"])
    candidates: list[dict[str, Any]] = []
    fold_positive: dict[str, int] = {name: 0 for name in names}
    fold_total = len(seeds) * len(fold_ids)
    for result in results:
        for record in result["records"]:
            if float(record["package_delta_single"]) > 0.0:
                fold_positive[str(record["name"])] += 1
    for name in names:
        deltas = [float(per_seed[str(seed)]["candidates"][name]["package_delta_single"]) for seed in seeds]
        row = {
            "name": name,
            "target": TARGET,
            "per_seed": {
                str(seed): {
                    "parent_wmape": per_seed[str(seed)]["parent_wmape"],
                    "candidate_wmape": per_seed[str(seed)]["candidates"][name]["wmape"],
                    "package_delta_single": per_seed[str(seed)]["candidates"][name]["package_delta_single"],
                }
                for seed in seeds
            },
            "mean_package_delta_single": float(np.mean(deltas)),
            "min_package_delta_single": float(np.min(deltas)),
            "both_split_seeds_positive": bool(all(value > 0.0 for value in deltas)),
            "positive_folds": int(fold_positive[name]),
            "evaluated_folds": int(fold_total),
        }
        row["continuation_gate_passed"] = bool(
            row["both_split_seeds_positive"]
            and row["mean_package_delta_single"] >= threshold
        )
        row["local_package_score_if_promoted"] = float(
            PARENT_DEVELOPMENT_SCORE + row["mean_package_delta_single"]
        )
        row["local_working_gate_passed"] = bool(row["local_package_score_if_promoted"] >= local_gate)
        candidates.append(row)
    candidates.sort(key=lambda row: (
        -float(row["mean_package_delta_single"]),
        -float(row["min_package_delta_single"]),
        str(row["name"]),
    ))
    qualifying = [row for row in candidates if row["continuation_gate_passed"]]
    return {
        "parent_name": str(config["parent"]["name"]),
        "parent_recorded_development_score": PARENT_DEVELOPMENT_SCORE,
        "claim_boundary": str(config["parent"]["claim_boundary"]),
        "continuation_threshold": threshold,
        "local_working_gate": local_gate,
        "fold_ids": [int(value) for value in fold_ids],
        "seeds": seeds,
        "per_seed_parent": {
            str(seed): {"parent_wmape": per_seed[str(seed)]["parent_wmape"], "rows": per_seed[str(seed)]["rows"]}
            for seed in seeds
        },
        "candidates": candidates,
        "qualifying": qualifying,
        "qualifying_count": len(qualifying),
        "next_step": "COMPLETE_FIVE_FOLDS_FOR_QUALIFIERS" if qualifying else "CLOSE_WITHOUT_EXTENSION",
    }


def run_search(root: Path, output: Path, *, fold_ids: Sequence[int], workers: int) -> dict[str, Any]:
    _prepare_output(root, output)
    config = load_v43_config(root)
    train = load_training_frame(root)
    seeds = [int(value) for value in config["protocol"]["split_seeds"]]
    manifest = {
        "version": config["version"],
        "status": "RUNNING",
        "head": _head(root),
        "source_hash": _source_hash(root),
        "data_hash": _data_hash(train),
        "parent_recorded_development_score": PARENT_DEVELOPMENT_SCORE,
        "fold_ids": [int(value) for value in fold_ids],
        "seeds": seeds,
        "target": TARGET,
        "workers": int(workers),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "automatic_package": False,
        "agent_uploads": 0,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ledger = output / "fit_ledger.jsonl"
    results: list[dict[str, Any]] = []
    payloads = [
        {"root": str(root), "seed": seed, "fold_id": int(fold_id)}
        for seed in seeds
        for fold_id in fold_ids
    ]
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(evaluate_fold_seed, payload): payload for payload in payloads}
        for future in as_completed(futures):
            payload = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - append failure evidence
                event = {
                    "event": "failed",
                    "seed": payload["seed"],
                    "fold_id": payload["fold_id"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                with ledger.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                raise
            arrays = result.pop("arrays")
            seed = int(result["seed"])
            fold_id = int(result["fold_id"])
            np.savez_compressed(output / f"pred-seed{seed}-fold{fold_id}.npz", **arrays)
            event = {"event": "complete", **_jsonable(result)}
            with ledger.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            result["arrays"] = arrays
            results.append(result)
            best = max(result["records"], key=lambda row: float(row["package_delta_single"]))
            print(json.dumps({
                "seed": seed,
                "fold": fold_id,
                "parent_wmape": result["parent_wmape"],
                "best_candidate": best["name"],
                "best_package_delta_single": best["package_delta_single"],
            }, ensure_ascii=False), flush=True)
    summary = _summarize(results, config, fold_ids)
    summary.update({
        "status": "COMPLETE",
        "fit_counts": {
            "nested_parent_member_fits": int(sum(int(result["nested_fit_count"]) for result in results)),
            "corrector_fits": int(sum(len(result["records"]) for result in results)),
        },
        "seconds": float(sum(float(result["total_seconds"]) for result in results)),
        "agent_uploads": 0,
        "platform_packages": 0,
    })
    (output / "summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["status"] = "COMPLETE"
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["summary_sha256"] = _sha256_bytes((output / "summary.json").read_bytes())
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_complete(root: Path, coarse_summary: Path, output: Path, *, workers: int) -> dict[str, Any]:
    coarse = json.loads(coarse_summary.read_text(encoding="utf-8"))
    qualifying = list(coarse.get("qualifying", []))
    if not qualifying:
        raise RuntimeError("V4.3 coarse screen has no continuation-gate qualifiers")
    config = load_v43_config(root)
    return run_search(
        root,
        output,
        fold_ids=tuple(int(value) for value in config["protocol"]["complete_folds"]),
        workers=int(workers),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--fold", type=int, default=0)
    smoke.add_argument("--workers", type=int, default=1)
    coarse = sub.add_parser("coarse")
    coarse.add_argument("--output", type=Path, required=True)
    coarse.add_argument("--workers", type=int, default=4)
    complete = sub.add_parser("complete")
    complete.add_argument("--coarse-summary", type=Path, required=True)
    complete.add_argument("--output", type=Path, required=True)
    complete.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    root = Path.cwd()
    if args.command == "smoke":
        result = run_search(root, args.output, fold_ids=(int(args.fold),), workers=int(args.workers))
    elif args.command == "coarse":
        config = load_v43_config(root)
        result = run_search(
            root,
            args.output,
            fold_ids=tuple(int(value) for value in config["protocol"]["coarse_folds"]),
            workers=int(args.workers),
        )
    else:
        result = run_complete(root, args.coarse_summary, args.output, workers=int(args.workers))
    print(json.dumps(_jsonable({
        "status": result.get("status"),
        "qualifying_count": result.get("qualifying_count"),
        "next_step": result.get("next_step"),
        "top": result.get("candidates", [])[:3],
    }), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
