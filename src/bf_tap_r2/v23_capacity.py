#!/usr/bin/env python3
"""V23: capacity probe on the winning raw-TabM time family.

Two frozen probes (`P_WIDE`, `P_DEEP`) are fitted with the frozen V3.6 evaluator
on complete outer folds of the two development split seeds, then screened as an
incremental nested blend on top of the V21 time incumbent.  Never uploads.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .data import TARGETS
from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import wmape
from .v5_spec import load_v5_spec
from .v11_quantile import load_oof
from .v16_sequential_masks import load_references

ROOT_DEFAULT = Path("/home/lux1/iron")
SPEC_DEFAULT = Path("configs/round2_v23/SPEC.yaml")
OUTPUT_DEFAULT = Path("local/runs/round2-v23/dev-r1")
LEDGER = "local/runs/round2-v3.6-loss-training-and-numeric-encoding/fixed-r2-final/fit_ledger.jsonl"
TIME = "tap_time_len"
IRON = "tap_iron"

_WORKER_TRAIN = None
_WORKER_FOLDS = None


def base_trial(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen N-0048 recipe, read from the append-only V3.6 ledger."""
    base_id = str(spec["base_recipe"])
    for line in (root / LEDGER).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "complete" and str(event.get("trial_id")) == base_id:
            trial = dict(event["trial"])
            if str(trial["target"]) != TIME or str(trial["structure"]) != str(spec["structure"]):
                raise ValueError("Base recipe identity mismatch")
            return trial
    raise FileNotFoundError(f"V23 base recipe not found in the ledger: {base_id}")


def probe_trial(root: Path, spec: Mapping[str, Any], probe: str) -> dict[str, Any]:
    if probe not in spec["probes"]:
        raise KeyError(f"Unknown V23 probe: {probe}")
    trial = base_trial(root, spec)
    capacity = dict(spec["probes"][probe])
    trial["trial_id"] = f"v23-{probe}"
    trial["capacity_name"] = probe.lower()
    trial["parameters"].update({k: v for k, v in capacity.items()})
    return trial


def _init_worker(train, folds) -> None:
    global _WORKER_TRAIN, _WORKER_FOLDS
    _WORKER_TRAIN, _WORKER_FOLDS = train, folds


def _worker(payload: Mapping[str, Any]) -> np.ndarray:
    from .v3_6_models import evaluate_v36_outer_folds

    if _WORKER_TRAIN is None or _WORKER_FOLDS is None:
        raise RuntimeError("V23 worker was not initialised")
    result = evaluate_v36_outer_folds(_WORKER_TRAIN, _WORKER_FOLDS, payload["trial"],
                                      fold_ids=(int(payload["fold"]),))
    column = np.full(len(_WORKER_TRAIN), np.nan)
    fold = int(payload["fold"])
    column[_WORKER_FOLDS == fold] = result["predictions"][_WORKER_FOLDS == fold]
    return column


def fit(root: Path | str, spec_path: Path | str, output: Path | str, seed: int,
        workers: int = 3, probes: Sequence[str] | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    spec = yaml.safe_load((root / spec_path).read_text())
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"Set {name}=1")
    out = (root / output).resolve() / f"seed-{seed}"
    if not out.is_relative_to(root / "local/runs/round2-v23"):
        raise ValueError("V23 outputs must stay under local/runs/round2-v23")
    out.mkdir(parents=True, exist_ok=True)
    done = {path.name for path in out.glob("pred-*.npy")}
    frame = load_v5_training_frame(root)
    folds = fold_vector(root, frame, int(seed), load_v5_spec(root))
    jobs = []
    ledger = out / "fit_ledger.jsonl"
    for probe in (list(probes) if probes else list(spec["probes"])):
        trial = probe_trial(root, spec, probe)
        for fold in spec["folds"]:
            if f"pred-{probe}-f{int(fold)}.npy" not in done:
                jobs.append({"probe": probe, "trial": trial, "fold": int(fold)})
    completed = 0
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), initializer=_init_worker,
                             initargs=(frame, folds)) as pool:
        futures = {pool.submit(_worker, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            key = f"{job['probe']}-f{job['fold']}"
            try:
                column = future.result()
                path = out / f"pred-{key}.npy"
                with path.open("xb") as stream:
                    np.save(stream, column)
                event = {"event": "complete", "key": key, "seed": int(seed),
                         "probe": job["probe"], "rows": int(np.isfinite(column).sum()),
                         "wmape": float(wmape(frame.loc[folds == job["fold"], TIME],
                                              column[folds == job["fold"]]))}
            except Exception as exc:  # noqa: BLE001 - keep failure evidence
                event = {"event": "failed", "key": key, "seed": int(seed),
                         "probe": job["probe"], "error": repr(exc)}
            with ledger.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            completed += 1
            print(json.dumps(event), flush=True)
    return {"status": "complete", "seed": int(seed), "completed": completed,
            "already_completed": len(done), "directory": str(out.relative_to(root))}


def _incumbent(root: Path, spec: Mapping[str, Any], seeds: Sequence[int], frame):
    spec5 = load_v5_spec(root)
    folds = {seed: fold_vector(root, frame, seed, spec5) for seed in seeds}
    reference_spec = yaml.safe_load((root / spec["reference_source"]).read_text())
    _, _, a35, _, current, _, _ = load_references(root, frame, folds, reference_spec)
    v7, _ = load_oof(root, spec["v7_time_cache"], frame, folds, TIME, "tabm_plr001")
    n_member = {seed: np.load(root / spec["n_member_template"].format(seed=seed),
                              allow_pickle=False) for seed in seeds}
    v36 = {seed: (a35[seed][TIME] - 0.35 * n_member[seed]) / 0.65 for seed in seeds}
    pll = {}
    for seed in seeds:
        column = np.full(len(frame), np.nan)
        for fold in range(5):
            column[folds[seed] == fold] = np.load(
                root / spec["pll_member_template"].format(seed=seed, fold=fold),
                allow_pickle=False).ravel()
        pll[seed] = column
    weights = {str(k): float(v) for k, v in spec["incumbent_time_weights"].items()}
    endpoints = {"v36_time": v36, "n_time": n_member, "v7_member_time": v7, "pll_time": pll}
    time = {seed: sum(weights[name] * endpoints[name][seed] for name in weights) for seed in seeds}
    return {TIME: time, IRON: {seed: current[seed][IRON] for seed in seeds}}


def screen(root: Path | str, spec_path: Path | str, output: Path | str,
           probes: Sequence[str] | None = None, screen_name: str = "screen.json") -> dict[str, Any]:
    root = Path(root).resolve()
    spec = yaml.safe_load((root / spec_path).read_text())
    seeds = [int(seed) for seed in spec["split_seeds"]]
    frame = load_v5_training_frame(root)
    spec5 = load_v5_spec(root)
    folds = {seed: fold_vector(root, frame, seed, spec5) for seed in seeds}
    incumbent = _incumbent(root, spec, seeds, frame)
    grid = [float(value) for value in spec["blend_grid"]]
    y = frame[TIME].to_numpy()
    rows = {}
    for probe in (list(probes) if probes else list(spec["probes"])):
        member = {}
        for seed in seeds:
            column = np.full(len(frame), np.nan)
            for fold in spec["folds"]:
                mask = folds[seed] == fold
                path = (root / output) / f"seed-{seed}" / f"pred-{probe}-f{fold}.npy"
                values = np.load(path, allow_pickle=False)
                if values.shape == (len(frame),):
                    values = values[mask]
                if values.shape != (int(mask.sum()),) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid V23 prediction shape: {path}")
                column[mask] = values
            if not np.isfinite(column).all():
                raise ValueError(f"Incomplete V23 coverage: {probe} seed {seed}")
            member[seed] = column
        deltas, alphas = [], []
        for held in seeds:
            other = [seed for seed in seeds if seed != held][0]
            best_alpha, best = 0.0, float("inf")
            for alpha in grid:
                value = wmape(y, (1 - alpha) * incumbent[TIME][other] + alpha * member[other])
                if value < best:
                    best, best_alpha = value, float(alpha)
            blended = (1 - best_alpha) * incumbent[TIME][held] + best_alpha * member[held]
            deltas.append(50.0 * (wmape(y, incumbent[TIME][held]) - wmape(y, blended)))
            alphas.append(best_alpha)
        mean = float(np.mean(deltas))
        rows[probe] = {"seed_gains": {str(seed): round(gain, 6) for seed, gain in zip(seeds, deltas)},
                       "alphas": {str(seed): round(alpha, 3) for seed, alpha in zip(seeds, alphas)},
                       "mean": round(mean, 6),
                       "both_seeds_positive": bool(all(gain > 0 for gain in deltas)),
                       "promoted": bool(mean >= float(spec["promotion"]["min_development_mean_gain"])
                                        and all(gain > 0 for gain in deltas))}
    destination = Path(output) if Path(output).is_absolute() else root / output
    report_path = destination / screen_name
    if report_path.exists():
        raise FileExistsError("V23 screen already exists; refusing overwrite")
    report_path.write_text(json.dumps({"probes": rows, "new_fits": 0}, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fit", "screen"])
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--spec", type=Path, default=SPEC_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--probes", nargs="+", default=None)
    parser.add_argument("--screen-name", default="screen.json")
    args = parser.parse_args()
    if args.mode == "fit":
        fit(args.root, args.spec, args.output, args.seed, args.workers, args.probes)
    else:
        screen(args.root, args.spec, args.output, args.probes, args.screen_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
