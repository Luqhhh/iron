#!/usr/bin/env python3
"""V22: incumbent-relative saturation screen of the whole reproducible library.

For every complete-coverage reproducible member of the frozen V5 candidate
library, measure the nested two-column blend gain on top of the strongest
incumbent (the V21 time column and the B0 iron column).  Zero fits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .v5_library import build_candidate_library, fold_vector, load_v5_training_frame
from .v5_resolution import wmape
from .v5_spec import load_v5_spec
from .v11_quantile import load_oof
from .v16_sequential_masks import load_references

ROOT_DEFAULT = Path("/home/lux1/iron")
OUTPUT_DEFAULT = Path("local/runs/round2-v22/saturation-r1.json")
TIME = "tap_time_len"
IRON = "tap_iron"


def incumbent_columns(root: Path, spec: Mapping[str, Any]) -> dict[str, dict[int, np.ndarray]]:
    """The V21 time column and the B0/V12 iron column at the two frozen seeds."""
    weights = {str(k): float(v) for k, v in spec["incumbent_time_weights"].items()}
    frame = load_v5_training_frame(root)
    spec5 = load_v5_spec(root)
    seeds = [int(seed) for seed in spec["split_seeds"]]
    folds = {seed: fold_vector(root, frame, seed, spec5) for seed in seeds}
    reference_spec = yaml.safe_load((root / spec["reference_source"]).read_text())
    _, _, a35, _, current, joint, _ = load_references(root, frame, folds, reference_spec)
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
        if not np.isfinite(column).all():
            raise ValueError(f"Incomplete P-LL member for seed {seed}")
        pll[seed] = column
    endpoints = {"v36_time": v36, "n_time": n_member, "v7_member_time": v7, "pll_time": pll,
                 "v12_member_iron": {seed: joint[seed][:, 0] for seed in seeds}}
    time = {seed: sum(weights[name] * endpoints[name][seed] for name in weights) for seed in seeds}
    iron = {seed: current[seed][IRON].copy() for seed in seeds}
    return {TIME: time, IRON: iron}


def nested_gain(target_values: Mapping[str, np.ndarray], incumbent: Mapping[int, np.ndarray],
                member: Mapping[int, np.ndarray], grid: Sequence[float]) -> tuple[float, list[float]]:
    seeds = sorted(incumbent)
    deltas, alphas = [], []
    for held in seeds:
        other = [seed for seed in seeds if seed != held][0]
        best_alpha, best = 0.0, float("inf")
        for alpha in grid:
            value = wmape(target_values, (1 - alpha) * incumbent[other] + alpha * member[other])
            if value < best:
                best, best_alpha = value, float(alpha)
        blended = (1 - best_alpha) * incumbent[held] + best_alpha * member[held]
        deltas.append(50.0 * (wmape(target_values, incumbent[held]) - wmape(target_values, blended)))
        alphas.append(best_alpha)
    return float(np.mean(deltas)), alphas


def run(root: Path | str = ROOT_DEFAULT,
        spec_path: Path | str = "configs/round2_v22/SPEC.yaml",
        output: Path | str = OUTPUT_DEFAULT) -> dict[str, Any]:
    root = Path(root).resolve()
    spec = yaml.safe_load((root / spec_path).read_text())
    destination = (root / output).resolve()
    if not destination.is_relative_to(root / "local/runs/round2-v22"):
        raise ValueError("V22 evidence must stay private under local/runs/round2-v22")
    if destination.exists():
        raise FileExistsError("V22 output already exists; refusing overwrite")
    frame = load_v5_training_frame(root)
    targets = {TIME: frame[TIME].to_numpy(), IRON: frame[IRON].to_numpy()}
    incumbent = incumbent_columns(root, spec)
    library = build_candidate_library(root, load_v5_spec(root))
    grid = [index / 40 for index in range(41)]
    report: dict[str, Any] = {"split_seeds": [int(s) for s in spec["split_seeds"]], "targets": {},
                              "new_fits": 0, "agent_uploads": 0}
    for target in (TIME, IRON):
        entries = library.complete_entries(target)
        rows = []
        for key, entry in entries.items():
            member = {seed: entry.vectors[seed] for seed in incumbent[target]}
            gain, alphas = nested_gain(targets[target], incumbent[target], member, grid)
            rows.append({"member": key, "gain": round(gain, 6),
                         "alphas": [round(alpha, 3) for alpha in alphas],
                         "packagability": entry.packagability})
        rows.sort(key=lambda row: -row["gain"])
        report["targets"][target] = {
            "members": len(rows),
            "above_0_0005": sum(1 for row in rows if row["gain"] > 0.0005),
            "best": rows[:10],
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "written", "output": str(destination.relative_to(root)),
                      "time_best": report["targets"][TIME]["best"][0],
                      "time_above_0_0005": report["targets"][TIME]["above_0_0005"],
                      "iron_best": report["targets"][IRON]["best"][0],
                      "iron_above_0_0005": report["targets"][IRON]["above_0_0005"]}, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--spec", type=Path, default=Path("configs/round2_v22/SPEC.yaml"))
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    run(args.root, args.spec, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
