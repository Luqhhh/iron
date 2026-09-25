#!/usr/bin/env python3
"""Rebase the V4.1 C-line coarse evidence to the frozen V36 development replay.

The C models are still fitted exactly as in the original V4.1 C-line runner.
Only the package reference changes: for each target, the candidate replaces that
target in the complete V36 replay while the other target stays at V36.  This is
descriptive coarse evidence, not an independent validation or a submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from bf_tap_r2.data import TARGETS
from bf_tap_r2.metrics import wmape
from bf_tap_r2.v3_run import load_fold_vector, load_training_frame
from bf_tap_r2.v4_1_paired_terms import (
    fit_c1_category_ebm,
    fit_c2_c3_c4_on_training_part,
    v34_trial_map,
)
from bf_tap_r2.v4_run import COARSE_FOLDS, SEEDS

OTHER = {"tap_iron": "tap_time_len", "tap_time_len": "tap_iron"}
C0_SOURCE_METHOD = {"tap_iron": "v4-S-017", "tap_time_len": "v4-S-023"}
METHODS = ("C0", "C1", "C2", "C3", "C4")


def _private_output(root: Path, output: Path) -> None:
    if not output.resolve().is_relative_to((root / "local/runs").resolve()):
        raise ValueError("C-line outputs must stay beneath local/runs")


def _append(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        handle.flush()


def _load_c0(root: Path, seed: int, target: str, n: int) -> np.ndarray:
    method = C0_SOURCE_METHOD[target]
    path = root / "local/runs/round2-v4-mechanism-search/coarse-r2/oof" / f"{method}-seed{int(seed)}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.load(path, allow_pickle=False)
    if values.shape != (n,):
        raise ValueError(f"Unexpected C0 source shape: {values.shape}")
    return values


def _load_v36_replay(root: Path) -> dict[tuple[int, str], np.ndarray]:
    path = (
        root
        / "local/runs/round2-v4.1-strong-increment/diagnostic-v36-r1"
        / "v36_development_replay_NOT_TRAINING_DATA.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "Run diagnose_v4_against_v36.py first; missing V36 development replay: " + str(path)
        )
    with np.load(path, allow_pickle=False) as loaded:
        refs = {
            (int(seed), target): np.asarray(loaded[f"{seed}_{target}"], dtype=float)
            for seed in SEEDS
            for target in TARGETS
        }
    return refs


def _metrics(train: pd.DataFrame, folds: np.ndarray, seed: int, target: str,
             candidate: np.ndarray, v36: dict[tuple[int, str], np.ndarray]) -> dict[str, Any]:
    mask = np.isin(folds, list(COARSE_FOLDS))
    actual = train.loc[mask, target].to_numpy(dtype=float)
    other = OTHER[target]
    baseline_target = v36[(int(seed), target)][mask]
    baseline_other = v36[(int(seed), other)][mask]
    actual_other = train.loc[mask, other].to_numpy(dtype=float)
    candidate_values = candidate[mask]
    if not np.isfinite(candidate_values).all():
        raise ValueError(f"Incomplete candidate coverage for {target!r}")
    candidate_wmape = float(wmape(actual, candidate_values))
    baseline_target_wmape = float(wmape(actual, baseline_target))
    baseline_other_wmape = float(wmape(actual_other, baseline_other))
    baseline_pkg = 100.0 - 50.0 * (baseline_target_wmape + baseline_other_wmape)
    candidate_pkg = 100.0 - 50.0 * (candidate_wmape + baseline_other_wmape)
    blend = 0.5 * candidate_values + 0.5 * baseline_target
    blend_wmape = float(wmape(actual, blend))
    blend_pkg = 100.0 - 50.0 * (blend_wmape + baseline_other_wmape)
    return {
        "target": target,
        "candidate_wmape": candidate_wmape,
        "baseline_target_wmape": baseline_target_wmape,
        "baseline_other_wmape": baseline_other_wmape,
        "baseline_package_score": baseline_pkg,
        "candidate_package_score": candidate_pkg,
        "package_delta_single": candidate_pkg - baseline_pkg,
        "equal_blend_wmape": blend_wmape,
        "equal_blend_package_score": blend_pkg,
        "package_delta_equal_blend": blend_pkg - baseline_pkg,
    }


def run(root: Path, output: Path, *, seeds: Sequence[int], folds: Sequence[int]) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    _private_output(root, output)
    output.mkdir(parents=True, exist_ok=True)
    train = load_training_frame(root)
    trials = v34_trial_map(root)
    v36 = _load_v36_replay(root)
    ledger = output / "fit_ledger.jsonl"
    events: list[dict[str, Any]] = []
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                event = json.loads(line)
                if event.get("event") == "complete":
                    events.append(event)
    completed = {(str(e["method_id"]), int(e["seed"]), str(e["target"])) for e in events}

    for seed in map(int, seeds):
        fold_vector = load_fold_vector(root, train, seed)
        for target in TARGETS:
            missing = [name for name in METHODS if (name, seed, target) not in completed]
            if not missing:
                continue
            predictions = {name: np.full(len(train), np.nan, dtype=float) for name in missing}
            fit_meta: dict[str, list[dict[str, Any]]] = {name: [] for name in missing}
            if "C0" in missing:
                predictions["C0"] = _load_c0(root, seed, target, len(train))
            for fold in map(int, folds):
                train_mask = fold_vector != fold
                valid_mask = fold_vector == fold
                train_frame = train.loc[train_mask].reset_index(drop=True)
                valid_frame = train.loc[valid_mask].reset_index(drop=True)
                if "C1" in missing:
                    started = __import__("time").perf_counter()
                    c1_pred, c1_meta = fit_c1_category_ebm(train_frame, valid_frame, target)
                    predictions["C1"][valid_mask] = c1_pred
                    fit_meta["C1"].append({
                        "seed": seed, "fold": fold, "n_train": int(train_mask.sum()),
                        "n_valid": int(valid_mask.sum()),
                        "seconds": float(__import__("time").perf_counter() - started), **c1_meta,
                    })
                need234 = any(name in missing for name in ("C2", "C3", "C4"))
                if need234:
                    started = __import__("time").perf_counter()
                    c234 = fit_c2_c3_c4_on_training_part(train_frame, valid_frame, target, trials)
                    elapsed = float(__import__("time").perf_counter() - started)
                    for name in ("C2", "C3", "C4"):
                        if name in missing:
                            predictions[name][valid_mask] = c234[name]["prediction"]
                            fit_meta[name].append({
                                "seed": seed, "fold": fold, "n_train": int(train_mask.sum()),
                                "n_valid": int(valid_mask.sum()), "seconds_total": elapsed,
                                **c234[name]["meta"],
                            })
            pred_dir = output / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pred_dir / f"seed-{seed}-{target}.npz",
                sample_id=train.sample_id.astype(str).to_numpy(),
                fold=fold_vector,
                **{name: predictions[name] for name in missing},
            )
            for name in missing:
                metrics = _metrics(train, fold_vector, seed, target, predictions[name], v36)
                event = {
                    "event": "complete",
                    "status": "available",
                    "method_id": name,
                    "family": "C",
                    "target": target,
                    "seed": seed,
                    "folds": [int(v) for v in folds],
                    "reference": "V36_DEVELOPMENT_REPLAY_DESCRIPTIVE",
                    "n_fits_external": 0 if name == "C0" else len(folds),
                    "fit_meta": fit_meta[name],
                    "metrics": metrics,
                }
                _append(ledger, event)
                events.append(event)
                completed.add((name, seed, target))
                print(json.dumps({"done": {"method": name, "seed": seed, "target": target,
                                            "package_delta": metrics["package_delta_single"]}},
                                 ensure_ascii=False), flush=True)

    summary_rows = []
    for event in sorted(events, key=lambda e: (e["method_id"], e["seed"], e["target"])):
        m = event["metrics"]
        summary_rows.append({
            "method_id": event["method_id"], "target": event["target"], "seed": event["seed"],
            "candidate_wmape": m["candidate_wmape"],
            "v36_target_wmape": m["baseline_target_wmape"],
            "package_delta_single": m["package_delta_single"],
            "equal_blend_package_delta": m["package_delta_equal_blend"],
            "n_fits_external": event.get("n_fits_external", 0),
        })
    table = pd.DataFrame(summary_rows)
    table.to_csv(output / "c_line_v36_summary.csv", index=False)
    (output / "c_line_v36_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    manifest = {
        "status": "C_LINE_V36_REBASED_DESCRIPTIVE_NOT_SUBMISSION",
        "reference": "V36 frozen development replay (same-label descriptive)",
        "seeds": [int(v) for v in seeds],
        "folds": [int(v) for v in folds],
        "method_target_units": len(events),
        "platform_uploads": 0,
        "submission_packages": 0,
        "warning": "C-line candidate replaces one target in the V36 replay package; descriptive only.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(output), **manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--folds", type=int, nargs="+", default=list(COARSE_FOLDS))
    args = parser.parse_args(argv)
    result = run(args.root, args.output, seeds=args.seeds, folds=args.folds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
