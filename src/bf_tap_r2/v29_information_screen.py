#!/usr/bin/env python3
"""V29: information screen for the last derivable inputs and the covariate shift.

Two zero-fit diagnostics against the strongest incumbent:

* **derived-feature screen** — every pairwise product and ratio of the 21 frozen
  features (630 candidates) correlated with the incumbent residual. If none
  exceeds the null correlation scale, there is no pairwise interaction or ratio
  information left for a new model to exploit.
* **covariate shift** — standardised mean difference of every feature between the
  frozen training frame and the 322-row test set. If the two distributions agree,
  transductive normalisation has nothing to correct.

Never trains, never uploads.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .data import FEATURES
from .v2_release import load_v2
from .v5_library import fold_vector, load_v5_training_frame
from .v5_spec import load_v5_spec
from .v11_quantile import load_oof
from .v16_sequential_masks import load_references

ROOT_DEFAULT = Path("/home/lux1/iron")
SPEC_DEFAULT = Path("configs/round2_v29/SPEC.yaml")
OUTPUT_DEFAULT = Path("local/runs/round2-v29/information-screen-r1.json")
TIME = "tap_time_len"
IRON = "tap_iron"


def incumbent_columns(root: Path, spec: Mapping[str, Any]):
    frame = load_v5_training_frame(root)
    seeds = [int(seed) for seed in spec["split_seeds"]]
    folds = {seed: fold_vector(root, frame, seed, load_v5_spec(root)) for seed in seeds}
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
    iron = {seed: current[seed][IRON].copy() for seed in seeds}
    return frame, folds, {TIME: time, IRON: iron}


def residual_screen(frame, folds, incumbent, spec) -> dict[str, Any]:
    seeds = sorted(incumbent[TIME])
    targets = {TIME: frame[TIME].to_numpy(), IRON: frame[IRON].to_numpy()}
    residuals = {target: np.mean([targets[target] - incumbent[target][seed] for seed in seeds], axis=0)
                 for target in (TIME, IRON)}
    matrix = frame[list(FEATURES)].to_numpy(dtype=float)
    candidates: dict[str, np.ndarray] = {}
    for left, right in itertools.combinations(range(len(FEATURES)), 2):
        a_name, b_name = FEATURES[left], FEATURES[right]
        a, b = matrix[:, left], matrix[:, right]
        candidates[f"{a_name}*{b_name}"] = a * b
        if np.all(np.abs(b) > 1e-9):
            candidates[f"{a_name}/{b_name}"] = a / b
        if np.all(np.abs(a) > 1e-9):
            candidates[f"{b_name}/{a_name}"] = b / a
    rows = len(frame)
    null_scale = 3.0 / np.sqrt(rows)
    report: dict[str, Any] = {"candidates": len(candidates), "rows": rows,
                              "null_absolute_correlation_scale": round(float(null_scale), 6)}
    for target in (TIME, IRON):
        scored = []
        residual = residuals[target]
        residual_flat = float(residual.std()) == 0.0
        for name, values in candidates.items():
            if not np.isfinite(values).all() or float(values.std()) == 0.0:
                continue
            correlation = 0.0 if residual_flat else float(np.corrcoef(values, residual)[0, 1])
            scored.append({"name": name, "correlation": round(correlation, 6)})
        scored.sort(key=lambda row: -abs(row["correlation"]))
        report[target] = {
            "top": scored[:8],
            "maximum_absolute_correlation": abs(scored[0]["correlation"]),
            "candidates_above_null_scale": sum(1 for row in scored
                                               if abs(row["correlation"]) > null_scale),
        }
    return report


def covariate_shift_screen(root: Path, frame, rows: int) -> dict[str, Any]:
    test = load_v2(root / "复赛_test", "test", rows)
    shifts = []
    for feature in FEATURES:
        train = frame[feature].to_numpy(dtype=float)
        query = test[feature].to_numpy(dtype=float)
        pooled = float(np.sqrt((train.var(ddof=1) + query.var(ddof=1)) / 2.0))
        difference = float((query.mean() - train.mean()) / pooled) if pooled > 0 else 0.0
        shifts.append({"feature": feature, "standardised_mean_difference": round(difference, 4),
                       "train_mean": round(float(train.mean()), 4),
                       "test_mean": round(float(query.mean()), 4)})
    shifts.sort(key=lambda row: -abs(row["standardised_mean_difference"]))
    return {"test_rows": len(test),
            "maximum_absolute_standardised_mean_difference":
                abs(shifts[0]["standardised_mean_difference"]),
            "top": shifts[:5],
            "train_spout_counts": {str(k): int(v) for k, v in frame.spout_no.value_counts().items()},
            "test_spout_counts": {str(k): int(v) for k, v in test.spout_no.value_counts().items()}}


def run(root: Path | str = ROOT_DEFAULT, spec_path: Path | str = SPEC_DEFAULT,
        output: Path | str = OUTPUT_DEFAULT) -> dict[str, Any]:
    root = Path(root).resolve()
    spec = yaml.safe_load((root / spec_path).read_text())
    destination = (root / output).resolve()
    if not destination.is_relative_to(root / "local/runs/round2-v29"):
        raise ValueError("V29 evidence must stay private under local/runs/round2-v29")
    if destination.exists():
        raise FileExistsError("V29 output already exists; refusing overwrite")
    frame, folds, incumbent = incumbent_columns(root, spec)
    report = {"residual_screen": residual_screen(frame, folds, incumbent, spec),
              "covariate_shift": covariate_shift_screen(root, frame, int(spec["test_rows"])),
              "new_fits": 0, "agent_uploads": 0}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2)[:1600])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--spec", type=Path, default=SPEC_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    run(args.root, args.spec, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
