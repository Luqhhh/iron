#!/usr/bin/env python3
"""V19 complete-coverage screen of the iron medium/large capacity candidates.

Scores each frozen V6 trial as an incremental nested two-column blend on top of
the incumbent iron column (V12 iron) at complete five-fold coverage on split
seeds 42/3407.  Reads only private OOF caches; fits nothing itself.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import wmape
from .v5_spec import load_v5_spec
from .v16_sequential_masks import load_references

ROOT_DEFAULT = Path("/home/lux1/iron")
SPEC_DEFAULT = Path("configs/round2_v19/SPEC.yaml")
OUTPUT_DEFAULT = Path("local/runs/round2-v19/screen-r1.json")
IRON = "tap_iron"


def paired_lcb95(deltas: Sequence[float]) -> float:
    """Lower bound of the paired mean at 95% using the t quantile for n-1 df."""
    values = np.asarray(list(deltas), dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError("At least two paired observations are required")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    t = {2: 12.706, 3: 4.303, 4: 3.182}.get(n)
    if t is None:
        raise ValueError(f"No frozen t quantile for n={n}")
    return mean - t * sd / math.sqrt(n)


def incumbent_and_folds(root: Path, spec: Mapping[str, Any]):
    frame = load_v5_training_frame(root)
    seeds = [int(seed) for seed in spec["split_seeds"]]
    folds = {seed: fold_vector(root, frame, seed, load_v5_spec(root)) for seed in seeds}
    reference_spec = yaml.safe_load((root / "configs/round2_v16/SPEC.yaml").read_text())
    _, _, _, _, current, _, _ = load_references(root, frame, folds, reference_spec)
    incumbent = {seed: current[seed][IRON].copy() for seed in seeds}
    return frame, folds, incumbent


def load_candidate(root: Path, directory: Path, trial_id: str, seed: int, rows: int) -> np.ndarray:
    path = root / directory / f"seed-{seed}" / f"pred-{trial_id}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing complete-coverage prediction: {path}")
    values = np.load(path, allow_pickle=False)
    if values.shape != (rows,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid prediction matrix: {path} {values.shape}")
    if (values < 0).any():
        raise ValueError(f"Negative prediction in {path}")
    return values


def screen_candidate(frame, folds, incumbent, member, target, grid) -> dict[str, Any]:
    y = frame[target].to_numpy()
    seeds = sorted(folds)
    seeds_delta: dict[int, float] = {}
    alphas: dict[int, float] = {}
    cells: list[dict[str, Any]] = []
    for held in seeds:
        other = [seed for seed in seeds if seed != held][0]
        best_alpha, best = 0.0, math.inf
        for alpha in grid:
            value = wmape(y, (1 - alpha) * incumbent[other] + alpha * member[other])
            if value < best:
                best, best_alpha = value, float(alpha)
        blended = (1 - best_alpha) * incumbent[held] + best_alpha * member[held]
        incumbent_wmape = wmape(y, incumbent[held])
        blended_wmape = wmape(y, blended)
        seeds_delta[held] = 50.0 * (incumbent_wmape - blended_wmape)
        alphas[held] = best_alpha
        for fold in range(5):
            mask = folds[held] == fold
            cells.append({
                "seed": held, "fold": fold, "rows": int(mask.sum()),
                "delta_score": 50.0 * (wmape(y[mask], incumbent[held][mask])
                                        - wmape(y[mask], blended[mask])),
            })
    deltas = list(seeds_delta.values())
    return {
        "target": target,
        "seed_deltas": {str(seed): round(value, 6) for seed, value in sorted(seeds_delta.items())},
        "alphas": {str(seed): round(value, 4) for seed, value in sorted(alphas.items())},
        "mean_incremental_gain": round(float(np.mean(deltas)), 6),
        "all_seeds_positive": bool(all(value > 0 for value in deltas)),
        "paired_lcb95": round(paired_lcb95(deltas), 6),
        "cells": cells,
        "positive_cells": int(sum(1 for cell in cells if cell["delta_score"] > 0)),
        "cells_total": len(cells),
        "descriptive_only": "fold level is descriptive per the frozen gate",
    }


def run(root: Path | str = ROOT_DEFAULT, spec_path: Path | str = SPEC_DEFAULT,
        directory: Path | str = "local/runs/round2-v6-iron-capacity-networks/coverage-r1",
        output: Path | str = OUTPUT_DEFAULT) -> dict[str, Any]:
    root = Path(root).resolve()
    spec = yaml.safe_load((root / spec_path).read_text())
    frame, folds, incumbent = incumbent_and_folds(root, spec)
    grid = [float(value) for value in spec["blend_grid"]]
    members: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for trial_id in spec["trial_ids"]:
        member = {seed: load_candidate(root, Path(directory), trial_id, seed, len(frame))
                  for seed in folds}
        for seed in folds:
            reference = incumbent[seed]
            correlation = float(np.corrcoef(member[seed] - frame[IRON].to_numpy(),
                                            reference - frame[IRON].to_numpy())[0, 1])
            members.setdefault(trial_id, {})[str(seed)] = {
                "residual_correlation_with_incumbent": round(correlation, 4),
                "standalone_wmape": round(wmape(frame[IRON].to_numpy(), member[seed]), 6),
            }
        result = screen_candidate(frame, folds, incumbent, member, IRON, grid)
        result["trial_id"] = trial_id
        rows.append(result)
    rows.sort(key=lambda row: -row["mean_incremental_gain"])
    report = {
        "spec": str(spec_path),
        "directory": str(directory),
        "split_seeds": [int(seed) for seed in spec["split_seeds"]],
        "candidates": rows,
        "signatures": members,
        "promoted": [row["trial_id"] for row in rows if row["all_seeds_positive"]],
        "new_fits": 0,
        "agent_uploads": 0,
    }
    destination = (root / output).resolve()
    if not destination.is_relative_to(root / "local/runs/round2-v19"):
        raise ValueError("V19 screen evidence must stay private under local/")
    if destination.exists():
        raise FileExistsError("V19 screen output already exists; refusing overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "written", "output": str(destination.relative_to(root)),
                      "ranking": [(row["trial_id"], row["mean_incremental_gain"],
                                   row["all_seeds_positive"], row["positive_cells"])
                                  for row in rows]}, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--spec", type=Path, default=SPEC_DEFAULT)
    parser.add_argument("--directory", type=Path,
                        default=Path("local/runs/round2-v6-iron-capacity-networks/coverage-r1"))
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    run(args.root, args.spec, args.directory, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
