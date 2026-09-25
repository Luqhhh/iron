"""Local, private-only execution helpers for the V4.1 taskbook.

This module exposes the earlier paired-terms runner and the 2026-09-25 C2/C3/C4
strong-increment screen.  It never writes a submission, never reads platform
labels, never uploads, and refuses to overwrite an existing evidence directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import FEATURES, TARGETS
from .v3_run import load_fold_vector, load_training_frame
from .v4_run import COARSE_FOLDS, SEEDS, _load_a_dev_reference, _metrics_for_unit
from .v4_1_c234 import run_c234_screen
from .v4_1_paired_terms import (
    V34_PARENT_TRIAL,
    fit_c1_category_ebm,
    fit_c2_c3_c4_on_training_part,
    v34_trial_map,
)

C0_SOURCE_METHOD = {"tap_iron": "v4-S-017", "tap_time_len": "v4-S-023"}


def _private_output(root: Path, output: Path) -> None:
    if not output.resolve().is_relative_to((root / "local/runs").resolve()):
        raise ValueError("V4.1 outputs must remain beneath local/runs")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing V4.1 output: {output}")
    output.mkdir(parents=True, exist_ok=False)


def _append_jsonl(path: Path, event: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        handle.flush()


def _read_complete(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "complete":
            events.append(event)
    return events


def _c0_reused_event(root: Path, source_relative: str, seed: int, target: str) -> dict[str, Any]:
    source = root / source_relative / "fit_ledger.jsonl"
    if not source.exists():
        raise FileNotFoundError(source)
    wanted = C0_SOURCE_METHOD[target]
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if (event.get("event") == "complete" and event.get("method_id") == wanted
                and int(event.get("seed", -1)) == int(seed)
                and event.get("target") == target and event.get("status") == "available"):
            return deepcopy(event)
    raise KeyError(f"C0 source event not found: {wanted} seed={seed} target={target}")


def run_paired_terms(root: Path | str = ".", output: Path | str | None = None,
                     *, seeds: Sequence[int] = SEEDS, folds: Sequence[int] = COARSE_FOLDS) -> dict[str, Any]:
    root = Path(root).resolve()
    output = Path(output).resolve() if output is not None else (
        root / "local/runs/round2-v4.1-strong-increment/paired-terms-r1"
    )
    if output.exists():
        if not (output / "fit_ledger.jsonl").is_file():
            raise FileExistsError(f"Refusing to reuse a non-ledger output directory: {output}")
    else:
        _private_output(root, output)
    train = load_training_frame(root)
    a_ref = _load_a_dev_reference(root, train)
    trials = v34_trial_map(root)
    ledger = output / "fit_ledger.jsonl"
    summaries: list[dict[str, Any]] = [
        event for event in (_read_complete(ledger) if ledger.exists() else [])
    ]
    completed = {(str(e["method_id"]), int(e["seed"]), str(e["target"])) for e in summaries}

    for seed in seeds:
        seed = int(seed)
        folds_vector = load_fold_vector(root, train, seed)
        for target in TARGETS:
            # C0 is the recorded V4 S5 numerical control; it is reusable and
            # costs no new external fit.
            c0_key = ("C0", seed, target)
            if c0_key in completed:
                c0_event = next(e for e in summaries if (e["method_id"], int(e["seed"]), e["target"]) == c0_key)
            else:
                c0_source = _c0_reused_event(
                    root, "local/runs/round2-v4-mechanism-search/coarse-r2", seed, target
                )
                c0_event = {
                    "event": "complete",
                    "status": "available",
                    "method_id": "C0",
                    "family": "C",
                    "target": target,
                    "mechanism": "C0_v4_s5_21_numeric_reused",
                    "seed": seed,
                    "folds": list(folds),
                    "reused_from": "local/runs/round2-v4-mechanism-search/coarse-r2",
                    "source_method_id": C0_SOURCE_METHOD[target],
                    "n_fits_external": 0,
                    "metrics": deepcopy(c0_source["metrics"]),
                    "fit_meta": {"reused": True, "source_event": C0_SOURCE_METHOD[target]},
                }
                _append_jsonl(ledger, c0_event)
                summaries.append(c0_event)
                completed.add(c0_key)

            if all((name, seed, target) in completed for name in ("C1", "C2", "C3", "C4")):
                continue
            predictions: dict[str, np.ndarray] = {
                name: np.full(len(train), np.nan, dtype=float) for name in ("C1", "C2", "C3", "C4")
            }
            fit_meta: dict[str, list[dict[str, Any]]] = {name: [] for name in predictions}
            for fold in folds:
                fold = int(fold)
                train_mask = folds_vector != fold
                valid_mask = folds_vector == fold
                train_frame = train.loc[train_mask].reset_index(drop=True)
                valid_frame = train.loc[valid_mask].reset_index(drop=True)
                y = np.asarray(train_frame[target], dtype=float)

                started = time.perf_counter()
                c1_pred, c1_meta = fit_c1_category_ebm(train_frame, valid_frame, target)
                predictions["C1"][valid_mask] = c1_pred
                fit_meta["C1"].append({
                    "seed": seed, "fold": fold, "n_train": int(train_mask.sum()),
                    "n_valid": int(valid_mask.sum()), "seconds": float(time.perf_counter() - started),
                    **c1_meta,
                })

                started = time.perf_counter()
                c234 = fit_c2_c3_c4_on_training_part(train_frame, valid_frame, target, trials)
                elapsed = float(time.perf_counter() - started)
                for name in ("C2", "C3", "C4"):
                    predictions[name][valid_mask] = c234[name]["prediction"]
                    fit_meta[name].append({
                        "seed": seed, "fold": fold, "n_train": int(train_mask.sum()),
                        "n_valid": int(valid_mask.sum()), "seconds_total": elapsed,
                        **c234[name]["meta"],
                    })
                print(json.dumps({"seed": seed, "target": target, "fold": fold,
                                  "seconds_c234": round(elapsed, 2)}, ensure_ascii=False), flush=True)

            for name in ("C1", "C2", "C3", "C4"):
                prediction = predictions[name]
                mask = np.isin(folds_vector, list(folds))
                if not np.isfinite(prediction[mask]).all():
                    raise ValueError(f"{name} has incomplete coarse-fold coverage")
                metrics = _metrics_for_unit(train, folds_vector, seed, target, prediction, a_ref)
                event = {
                    "event": "complete",
                    "status": "available",
                    "method_id": name,
                    "family": "C",
                    "target": target,
                    "mechanism": {
                        "C1": "C1_s5_plus_nominal_spout",
                        "C2": "C2_v34_strong_ebm_parent",
                        "C3": "C3_explicit_c2_pair_set",
                        "C4": "C4_c3_plus_two_training_triples",
                    }[name],
                    "seed": seed,
                    "folds": list(folds),
                    "data_hash": None,
                    "n_fits_external": len(folds),
                    "fit_meta": fit_meta[name],
                    "metrics": metrics,
                }
                key = (name, seed, target)
                if key in completed:
                    continue
                _append_jsonl(ledger, event)
                summaries.append(event)
                completed.add(key)

    table = pd.DataFrame([
        {
            "method_id": e["method_id"], "target": e["target"], "seed": e["seed"],
            "candidate_wmape": e["metrics"]["candidate_wmape"],
            "baseline_target_wmape": e["metrics"]["baseline_target_wmape"],
            "package_delta_single": e["metrics"]["package_delta_single"],
            "equal_blend_package_delta": e["metrics"]["package_delta_equal_blend"],
            "n_fits_external": e.get("n_fits_external", 0),
        }
        for e in summaries
    ])
    table.to_csv(output / "paired_terms_summary.csv", index=False)
    (output / "paired_terms_summary.json").write_text(
        json.dumps(list(summaries), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    manifest = {
        "status": "C_LINE_DESCRIPTIVE_PRIVATE_NOT_SUBMISSION",
        "seeds": [int(v) for v in seeds],
        "folds": [int(v) for v in folds],
        "method_target_units": 10,
        "new_units": 8,
        "reused_c0_units": 2,
        "external_fits_new": int(sum(e.get("n_fits_external", 0) for e in summaries if e["method_id"] != "C0")),
        "platform_uploads": 0,
        "submission_packages": 0,
        "warning": "C line is A-relative coarse-fold descriptive evidence, not a complete-development promotion claim.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(output), **manifest}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("paired_terms", "c234"), default="paired_terms",
                        help="paired_terms preserves the earlier C0-C4 runner; c234 runs the 2026-09-25 strong-increment screen.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--targets", nargs="+", choices=["tap_iron", "tap_time_len"], default=list(TARGETS))
    parser.add_argument("--methods", nargs="+", choices=["C2", "C3", "C4"], default=["C2", "C3", "C4"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--folds", type=int, nargs="+", default=list(COARSE_FOLDS))
    parser.add_argument("--n-inner", type=int, default=3, help="group-safe C4 internal folds")
    parser.add_argument("--inner-seed", type=int, default=41017, help="C4 internal fold seed")
    parser.add_argument("--no-clip", action="store_true", help="do not clip fixed-slot candidate predictions at zero")
    parser.add_argument("--allow-intractable-high-dim-c4", action="store_true",
                        help="attempt exact high-dimensional time C4 instead of recording it blocked")
    args = parser.parse_args(argv)
    if args.experiment == "c234":
        result = run_c234_screen(
            args.root, args.output, targets=args.targets, methods=args.methods,
            seeds=args.seeds, folds=args.folds,
            n_inner=args.n_inner, inner_seed=args.inner_seed,
            clip_nonnegative=not args.no_clip,
            allow_intractable_high_dim_c4=args.allow_intractable_high_dim_c4,
        )
    else:
        result = run_paired_terms(args.root, args.output, seeds=args.seeds, folds=args.folds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
