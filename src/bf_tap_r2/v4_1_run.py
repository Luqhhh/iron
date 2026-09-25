"""Append-only V4.1 public-anchor residual search runner."""
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
from .v4_1_residual import (
    TARGETS,
    candidate_specs,
    fit_cross_fitted_anchor,
    load_v41_config,
    make_corrector,
    public_anchor_trials,
)


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
    values = train.loc[:, [*FEATURES, "spout_no", *TARGETS]].to_numpy(dtype=float)
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _source_hash(root: Path) -> str:
    paths = [
        root / "configs/round2_v4_1/experiment.yaml",
        root / "src/bf_tap_r2/v4_1_residual.py",
        root / "src/bf_tap_r2/v4_1_run.py",
        root / "src/bf_tap_r2/v3_4_models.py",
        root / "src/bf_tap_r2/v3_4_sampler.py",
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
    private = (root / "local/runs/round2-v4.1-orthogonal-residual-search").resolve()
    resolved = output.resolve()
    if not resolved.is_relative_to(private):
        raise ValueError("V4.1 output must remain under its private local run root")
    return resolved


def _prepare_output(root: Path, output: Path) -> None:
    resolved = _private_output(root, output)
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"V4.1 output is append-only and already nonempty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)


def _evaluate_target_seed(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(payload["root"]))
    target = str(payload["target"])
    seed = int(payload["seed"])
    fold_ids = tuple(int(value) for value in payload["fold_ids"])
    selected_names = None if payload.get("candidate_names") is None else {
        str(value) for value in payload["candidate_names"]
    }
    config = load_v41_config(root)
    train = load_training_frame(root)
    outer_folds = load_fold_vector(root, train, seed)
    specs = candidate_specs(config, target)
    if selected_names is not None:
        specs = [spec for spec in specs if str(spec["name"]) in selected_names]
    if not specs:
        raise ValueError(f"No V4.1 candidates selected for {target}")

    evaluated = np.isin(outer_folds, np.asarray(fold_ids, dtype=int))
    indices = np.flatnonzero(evaluated)
    anchor_prediction = np.full(len(train), np.nan, dtype=float)
    candidate_prediction = {
        str(spec["name"]): np.full(len(train), np.nan, dtype=float) for spec in specs
    }
    fold_rows: list[dict[str, Any]] = []
    fit_counts = {"anchor_ebm": 0, "corrector": 0}
    anchor_trials, _weights = public_anchor_trials(root, target)
    inner_splits = int(config["protocol"]["residual_inner_folds"])
    inner_seed_base = int(config["protocol"]["residual_inner_seed_base"])

    for fold_id in fold_ids:
        validation_mask = outer_folds == int(fold_id)
        training_mask = ~validation_mask
        training = train.loc[training_mask].reset_index(drop=True)
        validation = train.loc[validation_mask].reset_index(drop=True)
        inner_seed = inner_seed_base + seed * 10 + int(fold_id)
        started = time.perf_counter()
        anchor = fit_cross_fitted_anchor(
            root,
            training,
            target,
            n_splits=inner_splits,
            seed=inner_seed,
        )
        fit_counts["anchor_ebm"] += (inner_splits + 1) * len(anchor_trials)
        base_validation = anchor.fitted_model.predict(validation)
        anchor_prediction[validation_mask] = base_validation
        y_training = training[target].to_numpy(dtype=float)
        y_validation = validation[target].to_numpy(dtype=float)
        anchor_fold_wmape = float(wmape(y_validation, base_validation))
        fold_record = {
            "fold": int(fold_id),
            "training_rows": int(training_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
            "anchor_wmape": anchor_fold_wmape,
            "inner_seed": int(inner_seed),
            "inner_fold_hash": anchor.fold_hash,
            "inner_group_hash": anchor.group_hash,
            "candidates": {},
        }
        for spec in specs:
            corrector = make_corrector(spec).fit(
                training,
                anchor.oof_prediction,
                y_training,
            )
            fit_counts["corrector"] += 1
            correction = corrector.predict_correction(validation, base_validation)
            prediction = np.maximum(base_validation + correction, 0.0)
            name = str(spec["name"])
            candidate_prediction[name][validation_mask] = prediction
            candidate_wmape = float(wmape(y_validation, prediction))
            fold_record["candidates"][name] = {
                "candidate_id": str(spec["candidate_id"]),
                "family": str(spec["family"]),
                "wmape": candidate_wmape,
                "package_delta_single": float(50.0 * (anchor_fold_wmape - candidate_wmape)),
                "fit_meta": _jsonable(corrector.fit_meta_),
            }
        fold_record["seconds"] = float(time.perf_counter() - started)
        fold_rows.append(fold_record)

    y = train[target].to_numpy(dtype=float)[indices]
    anchor_values = anchor_prediction[indices]
    if not np.isfinite(anchor_values).all():
        raise ValueError("V4.1 anchor prediction is incomplete")
    anchor_wmape = float(wmape(y, anchor_values))
    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {
        "sample_id": np.asarray(train["sample_id"].astype(str).tolist(), dtype=np.str_)[indices],
        "target": y,
        "anchor": anchor_values,
        "fold": outer_folds[indices].astype(int),
    }
    by_name = {str(spec["name"]): spec for spec in specs}
    for name, full_prediction in candidate_prediction.items():
        values = full_prediction[indices]
        if not np.isfinite(values).all():
            raise ValueError(f"V4.1 candidate prediction is incomplete: {name}")
        score = float(wmape(y, values))
        spec = by_name[name]
        records.append({
            "candidate_id": str(spec["candidate_id"]),
            "name": name,
            "family": str(spec["family"]),
            "target": target,
            "seed": seed,
            "fold_ids": list(fold_ids),
            "anchor_wmape": anchor_wmape,
            "candidate_wmape": score,
            "wmape_improvement": float(anchor_wmape - score),
            "package_delta_single": float(50.0 * (anchor_wmape - score)),
            "spec": _jsonable(spec),
        })
        arrays[f"candidate__{name}"] = values
    return {
        "target": target,
        "seed": seed,
        "fold_ids": list(fold_ids),
        "anchor_wmape": anchor_wmape,
        "records": records,
        "fold_rows": fold_rows,
        "fit_counts": fit_counts,
        "arrays": arrays,
        "anchor_trial_ids": [str(row["trial_id"]) for row in anchor_trials],
    }


def _summarize(task_results: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    by_candidate: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in task_results:
        for record in result["records"]:
            key = (str(record["target"]), str(record["name"]))
            by_candidate.setdefault(key, []).append(dict(record))
    threshold = float(config["continuation_gate"]["mean_package_delta_single_min"])
    rows: list[dict[str, Any]] = []
    for (target, name), records in sorted(by_candidate.items()):
        records.sort(key=lambda row: int(row["seed"]))
        deltas = [float(row["package_delta_single"]) for row in records]
        if len(records) != len(config["protocol"]["split_seeds"]):
            raise ValueError(f"Incomplete V4.1 split coverage for {target}/{name}")
        row = {
            "target": target,
            "name": name,
            "candidate_id": records[0]["candidate_id"],
            "family": records[0]["family"],
            "per_seed": {
                str(record["seed"]): {
                    "anchor_wmape": record["anchor_wmape"],
                    "candidate_wmape": record["candidate_wmape"],
                    "package_delta_single": record["package_delta_single"],
                }
                for record in records
            },
            "mean_package_delta_single": float(np.mean(deltas)),
            "min_package_delta_single": float(np.min(deltas)),
            "both_split_seeds_positive": bool(all(delta > 0.0 for delta in deltas)),
        }
        row["continuation_gate_passed"] = bool(
            row["both_split_seeds_positive"]
            and row["mean_package_delta_single"] >= threshold
        )
        rows.append(row)
    rows.sort(key=lambda row: (
        -float(row["mean_package_delta_single"]),
        -float(row["min_package_delta_single"]),
        str(row["target"]),
        str(row["name"]),
    ))
    qualifying = [row for row in rows if row["continuation_gate_passed"]]
    return {
        "reference": str(config["reference"]["name"]),
        "claim_boundary": str(config["reference"]["claim_boundary"]),
        "continuation_threshold": threshold,
        "candidates": rows,
        "qualifying": qualifying,
        "qualifying_count": len(qualifying),
        "next_step": "COMPLETE_FIVE_FOLDS_FOR_QUALIFIERS" if qualifying else "CLOSE_WITHOUT_EXTENSION",
    }


def run_search(root: Path, output: Path, *, fold_ids: Sequence[int], workers: int,
               candidate_names_by_target: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    _prepare_output(root, output)
    config = load_v41_config(root)
    train = load_training_frame(root)
    seeds = [int(value) for value in config["protocol"]["split_seeds"]]
    manifest = {
        "version": config["version"],
        "status": "RUNNING",
        "head": _head(root),
        "source_hash": _source_hash(root),
        "data_hash": _data_hash(train),
        "fold_ids": [int(value) for value in fold_ids],
        "seeds": seeds,
        "targets": list(TARGETS),
        "workers": int(workers),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "automatic_package": False,
        "agent_uploads": 0,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payloads = []
    for target in TARGETS:
        names = None if candidate_names_by_target is None else list(candidate_names_by_target.get(target, []))
        if candidate_names_by_target is not None and not names:
            continue
        for seed in seeds:
            payloads.append({
                "root": str(root),
                "target": target,
                "seed": seed,
                "fold_ids": [int(value) for value in fold_ids],
                "candidate_names": names,
            })
    ledger = output / "fit_ledger.jsonl"
    task_results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(_evaluate_target_seed, payload): payload for payload in payloads}
        for future in as_completed(futures):
            payload = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - append failure evidence
                event = {
                    "event": "failed",
                    "target": payload["target"],
                    "seed": payload["seed"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                with ledger.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                raise
            arrays = result.pop("arrays")
            target = str(result["target"])
            seed = int(result["seed"])
            np.savez_compressed(output / f"pred-{target}-seed{seed}.npz", **arrays)
            event = {"event": "complete", **_jsonable(result)}
            with ledger.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            task_results.append(result)
            best = max(result["records"], key=lambda row: float(row["package_delta_single"]))
            print(json.dumps({
                "target": target,
                "seed": seed,
                "anchor_wmape": result["anchor_wmape"],
                "best": best["name"],
                "best_package_delta_single": best["package_delta_single"],
            }, ensure_ascii=False), flush=True)
    summary = _summarize(task_results, config)
    summary.update({
        "status": "COMPLETE",
        "fold_ids": [int(value) for value in fold_ids],
        "fit_counts": {
            key: int(sum(int(result["fit_counts"][key]) for result in task_results))
            for key in ("anchor_ebm", "corrector")
        },
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
        raise RuntimeError("V4.1 coarse screen has no continuation-gate qualifiers")
    names: dict[str, list[str]] = {target: [] for target in TARGETS}
    for row in qualifying:
        names[str(row["target"])].append(str(row["name"]))
    config = load_v41_config(root)
    return run_search(
        root,
        output,
        fold_ids=tuple(int(value) for value in config["protocol"]["complete_folds"]),
        workers=workers,
        candidate_names_by_target=names,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    coarse = sub.add_parser("coarse")
    coarse.add_argument("--output", type=Path, required=True)
    coarse.add_argument("--workers", type=int, default=2)
    complete = sub.add_parser("complete")
    complete.add_argument("--coarse-summary", type=Path, required=True)
    complete.add_argument("--output", type=Path, required=True)
    complete.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    root = Path.cwd()
    if args.command == "coarse":
        config = load_v41_config(root)
        result = run_search(
            root,
            args.output,
            fold_ids=tuple(int(value) for value in config["protocol"]["coarse_folds"]),
            workers=int(args.workers),
        )
    else:
        result = run_complete(
            root,
            args.coarse_summary,
            args.output,
            workers=int(args.workers),
        )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
