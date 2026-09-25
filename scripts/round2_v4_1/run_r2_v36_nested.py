#!/usr/bin/env python3
"""Run the V4.1-r2 E0/E1-state/E3-state nested protocol on V36.

This runner is intentionally append-only/resumable at (seed, outer_fold)
granularity.  It writes only beneath ``local/runs`` and never creates a
submission or uploads anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.metrics import wmape
from bf_tap_r2.splits import make_folds
from bf_tap_r2.v3_run import load_training_frame
from bf_tap_r2.v4_1_r2_nested import (
    LocalMedianResidual,
    MedianResidual,
    NestedResidualBank,
    SplineMedianResidual,
)
from bf_tap_r2.v4_1_r2_e2 import StateLinearLeafResidual
from bf_tap_r2.v4_1_r2_v36 import FixedV36Backend

DEFAULT_SEEDS = (42, 3407)

DEFAULT_CORRECTIONS = ("E0_median", "E1_state_spline_median", "E3_state_local_median")


def _correction_factories(names: Sequence[str]):
    available = {
        "E0_median": MedianResidual,
        "E1_state_spline_median": SplineMedianResidual,
        "E3_state_local_median": LocalMedianResidual,
        "E2_state_linear_leaf": StateLinearLeafResidual,
    }
    selected = list(names) if names else list(DEFAULT_CORRECTIONS)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown correction recipes: {unknown}")
    if not selected:
        raise ValueError("At least one correction recipe is required")
    return {name: available[name] for name in selected}
DEFAULT_OUTER_FOLDS = (0, 1)


def _private_output(root: Path, output: Path) -> None:
    if not output.resolve().is_relative_to((root / "local/runs").resolve()):
        raise ValueError("V4.1-r2 outputs must remain beneath local/runs")


def _append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        handle.flush()


def _load_completed(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    done: dict[tuple[int, int], dict[str, Any]] = {}
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "complete":
            done[(int(event["seed"]), int(event["outer_fold"]))] = event
    return done


def _load_coarse_folds(root: Path, seed: int, sample_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    path = root / "local/runs/round2-v2/comparison-r1" / f"folds-{int(seed)}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"sample_id": str, "group_id": str})
    needed = {"sample_id", "group_id", "fold", "seed"}
    if not needed.issubset(frame.columns) or frame[list(needed)].isna().any().any():
        raise ValueError(f"Incomplete coarse fold assignment: {path}")
    if frame.sample_id.duplicated().any() or set(frame.sample_id) != set(sample_ids):
        raise ValueError(f"Coarse fold sample-ID mismatch: {path}")
    if not (frame.seed == int(seed)).all() or set(frame.fold) != set(range(5)):
        raise ValueError(f"Invalid coarse fold/seed values: {path}")
    if (frame.groupby("group_id").fold.nunique() > 1).any():
        raise ValueError(f"Duplicate group crosses coarse fold boundary: {path}")
    aligned = frame.set_index("sample_id").loc[list(sample_ids)]
    return aligned["fold"].to_numpy(dtype=int), aligned["group_id"].to_numpy(dtype=object)


def _project_split(x: pd.DataFrame, y: pd.DataFrame, groups: Any, seed: int):
    frame = pd.DataFrame({"sample_id": x.index.astype(str), "spout_no": x["spout_no"].to_numpy()})
    for name in FEATURES:
        frame[name] = x[name].to_numpy(dtype=float)
    assignment = make_folds(
        frame,
        seed=int(seed),
        n_splits=3,
        groups=pd.Series(np.asarray(groups), index=frame.index),
    )
    fold = assignment.set_index("sample_id").loc[frame["sample_id"], "fold"].to_numpy(dtype=int)
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(3)]


def _score_variants(y_valid: pd.DataFrame, variants: dict[str, pd.DataFrame]) -> dict[str, Any]:
    baseline = variants["baseline"]
    baseline_target = {
        target: float(wmape(y_valid[target].to_numpy(dtype=float), baseline[target].to_numpy(dtype=float)))
        for target in TARGETS
    }
    baseline_package = 100.0 - 100.0 * float(np.mean(list(baseline_target.values())))
    out: dict[str, Any] = {
        "baseline_target_wmape": baseline_target,
        "baseline_package_score": baseline_package,
        "variants": {},
    }
    for name, prediction in variants.items():
        target_wmape = {
            target: float(wmape(y_valid[target].to_numpy(dtype=float), prediction[target].to_numpy(dtype=float)))
            for target in TARGETS
        }
        package = 100.0 - 100.0 * float(np.mean(list(target_wmape.values())))
        out["variants"][name] = {
            "target_wmape": target_wmape,
            "package_score": package,
            "package_delta_vs_baseline": package - baseline_package,
            "target_delta_vs_baseline": {
                target: baseline_target[target] - target_wmape[target] for target in TARGETS
            },
        }
    return out


def run(root: Path, output: Path, *, seeds: Sequence[int], outer_folds: Sequence[int], workers: int,
        corrections: Sequence[str] | None = None) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    _private_output(root, output)
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / "fit_ledger.jsonl"
    completed = _load_completed(ledger)
    train = load_training_frame(root)
    if not train.sample_id.astype(str).str.startswith("R2S2_TRAIN_").all():
        raise ValueError("Unexpected training sample identity")

    for seed in map(int, seeds):
        folds, groups = _load_coarse_folds(root, seed, train.sample_id.astype(str).tolist())
        for outer_fold in map(int, outer_folds):
            key = (int(seed), int(outer_fold))
            if key in completed:
                print(json.dumps({"skip": {"seed": int(seed), "outer_fold": int(outer_fold)}}), flush=True)
                continue
            train_mask = folds != int(outer_fold)
            valid_mask = folds == int(outer_fold)
            outer_train = train.loc[train_mask].reset_index(drop=True)
            outer_valid = train.loc[valid_mask].reset_index(drop=True)
            train_ids = outer_train.sample_id.astype(str)
            valid_ids = outer_valid.sample_id.astype(str)
            x_train = outer_train.loc[:, [*FEATURES, "spout_no"]].copy()
            x_train.index = pd.Index(train_ids, name="sample_id")
            y_train = outer_train.loc[:, list(TARGETS)].copy()
            y_train.index = x_train.index
            x_valid = outer_valid.loc[:, [*FEATURES, "spout_no"]].copy()
            x_valid.index = pd.Index(valid_ids, name="sample_id")
            y_valid = outer_valid.loc[:, list(TARGETS)].copy()
            y_valid.index = x_valid.index
            groups_train = np.asarray(groups)[train_mask]
            started = time.perf_counter()
            print(json.dumps({
                "start": {"seed": int(seed), "outer_fold": int(outer_fold),
                          "n_train": int(len(outer_train)), "n_valid": int(len(outer_valid))}
            }), flush=True)
            bank = NestedResidualBank(
                baseline_factory=lambda: FixedV36Backend(root, workers=int(workers)),
                correction_factories=_correction_factories(corrections or DEFAULT_CORRECTIONS),
                n_splits=3,
                split_function=_project_split,
            )
            bank.fit(x_train, y_train, groups=groups_train)
            variants = bank.predict_variants(x_valid)
            score = _score_variants(y_valid=y_valid, variants=variants)
            elapsed = float(time.perf_counter() - started)
            prediction_dir = output / "predictions"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            arrays = {"sample_id": outer_valid.sample_id.astype(str).to_numpy()}
            for name, prediction in variants.items():
                for target in TARGETS:
                    arrays[f"{name}::{target}"] = prediction[target].to_numpy(dtype=float)
            np.savez_compressed(prediction_dir / f"seed-{seed}-fold-{outer_fold}.npz", **arrays)
            event = {
                "event": "complete",
                "seed": int(seed),
                "outer_fold": int(outer_fold),
                "n_train": int(len(outer_train)),
                "n_valid": int(len(outer_valid)),
                "seconds": elapsed,
                "factory": "fixed_v36_frozen_weights_v1",
                "alpha_selection": "inner training meta folds only, zero fallback",
                "audit": bank.audit_,
                "scores": score,
            }
            _append_event(ledger, event)
            completed[key] = event
            print(json.dumps({
                "done": {"seed": int(seed), "outer_fold": int(outer_fold), "seconds": round(elapsed, 3)},
                "package_delta": {name: payload["package_delta_vs_baseline"]
                                  for name, payload in score["variants"].items()},
            }, ensure_ascii=False), flush=True)

    events = list(_load_completed(ledger).values())
    summary: dict[str, Any] = {
        "status": "R2_V36_NESTED_PRIVATE_NOT_SUBMISSION",
        "factory": "fixed_v36_frozen_weights_v1",
        "warning": ("Fixed-recipe V36 adapter; V36 top-level weights were not re-selected. "
                    "Descriptive dev evidence only; not a platform forecast."),
        "events": len(events),
        "seeds": sorted({int(e["seed"]) for e in events}),
        "outer_folds": sorted({int(e["outer_fold"]) for e in events}),
        "methods": [],
        "method_summary": {},
        "platform_uploads": 0,
        "submission_packages": 0,
    }
    methods = sorted({name for e in events for name in e["scores"]["variants"] if name != "baseline"})
    summary["methods"] = methods
    for method in methods:
        per_target: dict[str, Any] = {}
        for target in TARGETS:
            vals = [float(e["scores"]["variants"][method]["target_delta_vs_baseline"][target]) for e in events]
            pkg = [float(e["scores"]["variants"][method]["package_delta_vs_baseline"]) for e in events]
            per_target[target] = {
                "mean_target_delta": float(np.mean(vals)) if vals else None,
                "min_target_delta": float(np.min(vals)) if vals else None,
                "max_target_delta": float(np.max(vals)) if vals else None,
            }
        summary["method_summary"][method] = {
            "mean_package_delta": float(np.mean(pkg)) if pkg else None,
            "per_target": per_target,
            "all_events": [
                {"seed": int(e["seed"]), "outer_fold": int(e["outer_fold"]),
                 "package_delta": float(e["scores"]["variants"][method]["package_delta_vs_baseline"]),
                 "target_delta": e["scores"]["variants"][method]["target_delta_vs_baseline"]}
                for e in events
            ],
        }
    summary["events_detail"] = [
        {
            "seed": int(e["seed"]),
            "outer_fold": int(e["outer_fold"]),
            "seconds": float(e["seconds"]),
            "audit": e["audit"],
            "scores": e["scores"],
        }
        for e in events
    ]
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    manifest = {
        "version": "v4.1-r2-v36-fixed-recipe-nested-v1",
        "status": summary["status"],
        "output": str(output),
        "events": len(events),
        "seeds": summary["seeds"],
        "outer_folds": summary["outer_folds"],
        "factory_declaration": ("A_frozen_deployment_weights_v1 plus frozen V36 selected experts; "
                                "released V36 top-level weights reused as fixed recipe, not reselected."),
        "model_training_fits": sum(int(e["audit"].get("base_fit_calls", 0)) for e in events),
        "corrector_fits": sum(int(e["audit"].get("corrector_fits", 0)) for e in events),
        "scalar_optimizations": sum(int(e["audit"].get("scalar_optimizations", 0)) for e in events),
        "platform_uploads": 0,
        "submission_packages": 0,
        "warning": summary["warning"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--outer-folds", type=int, nargs="+", default=list(DEFAULT_OUTER_FOLDS))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--corrections", type=str, nargs="+", default=list(DEFAULT_CORRECTIONS))
    args = parser.parse_args(argv)
    summary = run(
        args.root,
        args.output,
        seeds=args.seeds,
        outer_folds=args.outer_folds,
        workers=int(args.workers),
        corrections=args.corrections,
    )
    print(json.dumps({k: summary[k] for k in ("status", "events", "seeds", "outer_folds", "method_summary")},
                     ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
