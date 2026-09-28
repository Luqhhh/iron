#!/usr/bin/env python3
"""V26: target-representation probe on the frozen winning time recipe.

Fits the frozen ``N-0048`` large raw-TabM time recipe on a transformed target
(``log`` / ``sqrt``) and inverts the prediction, then screens the result as an
incremental nested blend on top of the V21 time incumbent.  Never uploads.

Renumbered from V24 to V26 on 2026-09-28 (public label only).  The closed
2026-09-28 private run directory keeps its historical name and is not moved; see
`docs/round2_round_numbering.md`.
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

from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import wmape
from .v5_spec import load_v5_spec
from .v11_quantile import load_oof
from .v16_sequential_masks import load_references
from .v25_capacity import base_trial

ROOT_DEFAULT = Path("/home/lux1/iron")
SPEC_DEFAULT = Path("configs/round2_v26/SPEC.yaml")
# Frozen 2026-09-28 private run directory.  The public round label moved to V26
# but the evidence directory is deliberately neither renamed nor rewritten.
OUTPUT_DEFAULT = Path("local/runs/round2-v24/dev-r1")
TIME = "tap_time_len"
IRON = "tap_iron"

_WORKER = None


def transform(y: np.ndarray, kind: str) -> np.ndarray:
    if kind == "log":
        return np.log(y)
    if kind == "sqrt":
        return np.sqrt(y)
    raise ValueError(f"Unknown V26 target transform: {kind}")


def inverse(prediction: np.ndarray, kind: str) -> np.ndarray:
    if kind in ("log", "exp"):
        return np.exp(prediction)
    if kind in ("sqrt", "square_clip"):
        return np.clip(prediction, 0.0, None) ** 2
    raise ValueError(f"Unknown V26 inverse transform: {kind}")


def _init_worker(frame, folds) -> None:
    global _WORKER
    _WORKER = (frame, folds)


def _worker(payload: Mapping[str, Any]) -> np.ndarray:
    from .v3_6_models import V36Regressor

    if _WORKER is None:
        raise RuntimeError("V26 worker was not initialised")
    frame, folds = _WORKER
    fold = int(payload["fold"])
    mask = folds == fold
    training = frame.loc[~mask].reset_index(drop=True)
    query = frame.loc[mask].drop(columns=[TIME, IRON]).reset_index(drop=True)
    target = transform(training[TIME].to_numpy(dtype=float), payload["fit"])
    model = V36Regressor(payload["trial"])
    model.fit(training, target)
    prediction = inverse(np.asarray(model.predict(query), dtype=float), payload["inverse"])
    if prediction.shape != (int(mask.sum()),) or not np.isfinite(prediction).all():
        raise ValueError("Invalid V26 prediction")
    return prediction


def fit(root: Path | str, spec_path: Path | str, output: Path | str, seed: int,
        workers: int = 3, probes: Sequence[str] | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    spec = yaml.safe_load((root / spec_path).read_text())
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"Set {name}=1")
    out = (root / output).resolve() / f"seed-{seed}"
    if not out.is_relative_to(root / "local/runs/round2-v24"):
        raise ValueError("V26 outputs must stay under local/runs/round2-v24")
    out.mkdir(parents=True, exist_ok=True)
    done = {path.name for path in out.glob("pred-*.npy")}
    base_spec = yaml.safe_load((root / spec["base_spec"]).read_text())
    frame = load_v5_training_frame(root)
    folds = fold_vector(root, frame, int(seed), load_v5_spec(root))
    trial = base_trial(root, base_spec)
    jobs = []
    for probe in (list(probes) if probes else list(spec["probes"])):
        recipe = spec["probes"][probe]
        for fold in spec["folds"]:
            if f"pred-{probe}-f{int(fold)}.npy" not in done:
                jobs.append({"probe": probe, "fold": int(fold), "trial": trial,
                             "fit": recipe["fit"], "inverse": recipe["inverse"]})
    ledger = out / "fit_ledger.jsonl"
    completed = 0
    if jobs:
        with ProcessPoolExecutor(max_workers=max(1, int(workers)), initializer=_init_worker,
                                 initargs=(frame, folds)) as pool:
            futures = {pool.submit(_worker, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                key = f"{job['probe']}-f{job['fold']}"
                try:
                    values = future.result()
                    path = out / f"pred-{key}.npy"
                    with path.open("xb") as stream:
                        np.save(stream, values)
                    mask = folds == job["fold"]
                    event = {"event": "complete", "key": key, "seed": int(seed),
                             "probe": job["probe"], "rows": int(mask.sum()),
                             "wmape": float(wmape(frame.loc[mask, TIME], values))}
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
    folds = {seed: fold_vector(root, frame, seed, load_v5_spec(root)) for seed in seeds}
    reference_spec = yaml.safe_load((root / spec["reference_source"]).read_text())
    _, _, a35, _, _, _, _ = load_references(root, frame, folds, reference_spec)
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
    return {seed: sum(weights[name] * endpoints[name][seed] for name in weights) for seed in seeds}


def screen(root: Path | str, spec_path: Path | str, output: Path | str,
           probes: Sequence[str] | None = None, screen_name: str = "screen.json") -> dict[str, Any]:
    root = Path(root).resolve()
    spec = yaml.safe_load((root / spec_path).read_text())
    destination = (root / output).resolve()
    report_path = destination / screen_name
    if not report_path.parent.is_relative_to(root / "local/runs/round2-v24"):
        raise ValueError("V26 evidence must stay private under local/runs/round2-v24")
    if report_path.exists():
        raise FileExistsError("V26 screen already exists; refusing overwrite")
    seeds = [int(seed) for seed in spec["split_seeds"]]
    frame = load_v5_training_frame(root)
    folds = {seed: fold_vector(root, frame, seed, load_v5_spec(root)) for seed in seeds}
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
                if values.shape != (int(mask.sum()),):
                    raise ValueError(f"Invalid V26 prediction shape: {path}")
                column[mask] = values
            if not np.isfinite(column).all():
                raise ValueError(f"Incomplete V26 coverage: {probe} seed {seed}")
            member[seed] = column
        deltas, alphas = [], []
        for held in seeds:
            other = [seed for seed in seeds if seed != held][0]
            best_alpha, best = 0.0, float("inf")
            for alpha in grid:
                value = wmape(y, (1 - alpha) * incumbent[other] + alpha * member[other])
                if value < best:
                    best, best_alpha = value, float(alpha)
            blended = (1 - best_alpha) * incumbent[held] + best_alpha * member[held]
            deltas.append(50.0 * (wmape(y, incumbent[held]) - wmape(y, blended)))
            alphas.append(best_alpha)
        mean = float(np.mean(deltas))
        rows[probe] = {
            "seed_gains": {str(seed): round(gain, 6) for seed, gain in zip(seeds, deltas)},
            "alphas": {str(seed): round(alpha, 3) for seed, alpha in zip(seeds, alphas)},
            "standalone_wmape": {str(seed): round(wmape(y, member[seed]), 6) for seed in seeds},
            "incumbent_wmape": {str(seed): round(wmape(y, incumbent[seed]), 6) for seed in seeds},
            "mean": round(mean, 6),
            "both_seeds_positive": bool(all(gain > 0 for gain in deltas)),
            "promoted": bool(mean >= float(spec["promotion"]["min_development_mean_gain"])
                             and all(gain > 0 for gain in deltas)),
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
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
