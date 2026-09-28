#!/usr/bin/env python3
"""V18 zero-fit evidence: platform concavity bounds and the local OOF surface.

Reads only private development OOF caches and the recorded user-reported
platform scores.  Fits nothing, trains nothing and uploads nothing.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .data import TARGETS
from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import package_score, wmape
from .v5_spec import load_v5_spec
from .v11_quantile import load_oof
from .v16_sequential_masks import load_references

ROOT_DEFAULT = Path("/home/lux1/iron")
OUTPUT_DEFAULT = Path("local/runs/round2-v18/surface-r1.json")

#: User-reported platform scores (not independently verified receipts).
PLATFORM_SCORES = {
    "V36": Fraction(962734, 10000),
    "A35": Fraction(963366, 10000),
    "A45": Fraction(963438, 10000),
    "A60": Fraction(963465, 10000),
    "A72": Fraction(963425, 10000),
    "V7": Fraction(963519, 10000),
    "V12": Fraction(963526, 10000),
}
ALLOWANCE = Fraction(1, 10000)


def concavity_bound(low: Fraction, high: Fraction, weight_low: Fraction, weight_high: Fraction,
                    target_weight: Fraction) -> Fraction:
    """Upper bound from the secant of a concave curve, extended past ``weight_high``."""
    if not weight_low < weight_high < target_weight:
        raise ValueError("Secant extension requires increasing weights")
    slope = (high - low) / (weight_high - weight_low)
    return high + slope * (target_weight - weight_high)


def platform_bounds() -> dict[str, Any]:
    a35 = PLATFORM_SCORES["A35"]
    v36 = PLATFORM_SCORES["V36"]
    iron_gain = PLATFORM_SCORES["V12"] - a35           # iron changed only
    time_gain = PLATFORM_SCORES["V7"] - a35            # time changed only
    iron_bound = concavity_bound(a35, PLATFORM_SCORES["V12"], Fraction(0), Fraction(1, 2), Fraction(1))
    time_bound = concavity_bound(a35, PLATFORM_SCORES["V7"], Fraction(0), Fraction(1, 2), Fraction(1))
    ceiling = v36 + (iron_bound - a35) + (time_bound - v36)
    return {
        "metric": "max(0, 100 - 50*WMAPE_iron - 50*WMAPE_time)",
        "score_error_allowance": str(ALLOWANCE),
        "iron_gain_at_half_weight": str(iron_gain),
        "time_gain_at_half_weight": str(time_gain),
        "conditional_combined_B0": str(a35 + iron_gain + time_gain),
        "iron_segment_upper_bound_at_full_weight_vs_A35": str(iron_bound - a35),
        "time_segment_upper_bound_at_full_weight_vs_V36": str(time_bound - v36),
        "two_line_joint_ceiling": str(ceiling),
        "two_line_joint_ceiling_below_96_4": str(Fraction(964, 10) - ceiling),
        "scope": "only the V36/V12m iron line and the A35/V7m time line; not a forecast",
    }


def _design_local(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    frame = load_v5_training_frame(root)
    seeds = [42, 3407]
    folds = {seed: fold_vector(root, frame, seed, load_v5_spec(root)) for seed in seeds}
    reference_spec = yaml.safe_load((root / spec["reference_source"]).read_text())
    _, a60, a35, _, current, joint, _ = load_references(root, frame, folds, reference_spec)
    v7_time, _ = load_oof(root, spec["v7_time_cache"], frame, folds, "tap_time_len", "tabm_plr001")
    n_time = {seed: np.load(root / "local/runs/round2-v5-error-covariance/time-n-family-r1" /
                            f"seed-{seed}/pred-v36-s1-N-0048.npy") for seed in seeds}
    v36_time = {seed: (a35[seed]["tap_time_len"] - 0.35 * n_time[seed]) / 0.65 for seed in seeds}
    endpoints = {
        "v36_iron": {seed: a35[seed]["tap_iron"].copy() for seed in seeds},
        "v12_member_iron": {seed: joint[seed][:, 0] for seed in seeds},
        "v36_time": v36_time,
        "n_time": n_time,
        "v7_member_time": {seed: v7_time[seed].copy() for seed in seeds},
    }
    y = {target: frame[target].to_numpy() for target in TARGETS}

    def column(name: str, weights: Mapping[str, float]) -> dict[int, np.ndarray]:
        return {seed: sum(weight * endpoints[key][seed] for key, weight in weights.items())
                for seed in seeds}

    designs = {
        "V18_B0_V12IRON_V7TIME": ({"v12_member_iron": 0.5, "v36_iron": 0.5},
                                  {"v36_time": 0.325, "n_time": 0.175, "v7_member_time": 0.5}),
        "V18_TIME_V75": ({"v12_member_iron": 0.5, "v36_iron": 0.5},
                         {"v36_time": 0.1625, "n_time": 0.0875, "v7_member_time": 0.75}),
        "V18_TIME_V100": ({"v12_member_iron": 0.5, "v36_iron": 0.5},
                          {"v7_member_time": 1.0}),
        "V18_IRON_W75": ({"v36_iron": 0.25, "v12_member_iron": 0.75},
                         {"v36_time": 0.325, "n_time": 0.175, "v7_member_time": 0.5}),
        "V18_TIME_A60V7_50": ({"v12_member_iron": 0.5, "v36_iron": 0.5},
                              {"v36_time": 0.2, "n_time": 0.3, "v7_member_time": 0.5}),
    }
    a35_score = {seed: package_score(wmape(y["tap_iron"], a35[seed]["tap_iron"]),
                                     wmape(y["tap_time_len"], a35[seed]["tap_time_len"]))
                 for seed in seeds}
    rows: dict[str, Any] = {}
    for name, (iron_weights, time_weights) in designs.items():
        iron = column("iron", iron_weights)
        time = column("time", time_weights)
        scores = {seed: package_score(wmape(y["tap_iron"], iron[seed]),
                                      wmape(y["tap_time_len"], time[seed]))
                  for seed in seeds}
        rows[name] = {
            "local_package_score": {str(seed): round(scores[seed], 6) for seed in seeds},
            "local_delta_vs_A35": {str(seed): round(scores[seed] - a35_score[seed], 6) for seed in seeds},
            "local_mean_delta_vs_A35": round(float(np.mean([scores[seed] - a35_score[seed]
                                                            for seed in seeds])), 6),
        }
    return {
        "split_seeds": seeds,
        "a35_local_package_score": {str(seed): round(a35_score[seed], 6) for seed in seeds},
        "designs": rows,
        "note": "single-seed-per-candidate complete five-fold OOF; descriptive only",
    }


def member_screen(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Incremental nested blend gain of every reproducible member on top of B0."""
    frame = load_v5_training_frame(root)
    seeds = [42, 3407]
    folds = {seed: fold_vector(root, frame, seed, load_v5_spec(root)) for seed in seeds}
    reference_spec = yaml.safe_load((root / spec["reference_source"]).read_text())
    _, _, a35, _, current, joint, _ = load_references(root, frame, folds, reference_spec)
    v7_time, _ = load_oof(root, spec["v7_time_cache"], frame, folds, "tap_time_len", "tabm_plr001")
    y = {target: frame[target].to_numpy() for target in TARGETS}
    b0 = {"tap_iron": {seed: current[seed]["tap_iron"] for seed in seeds},
          "tap_time_len": {seed: 0.5 * a35[seed]["tap_time_len"] + 0.5 * v7_time[seed] for seed in seeds}}

    def load_column(directory: str, key: str, target: str, joint_index: int | None = None):
        values = {}
        for seed in seeds:
            column = np.full(len(frame), np.nan)
            for fold in range(5):
                path = root / f"local/runs/{directory}/development-r1/{key}-s{seed}-f{fold}.npy"
                if not path.exists():
                    return None
                loaded = np.load(path, allow_pickle=False)
                column[folds[seed] == fold] = loaded.reshape(-1, 2)[:, joint_index].ravel() \
                    if joint_index is not None else loaded.ravel()
            if not np.isfinite(column).all():
                return None
            values[seed] = column
        return values

    catalogue: list[tuple[str, str, dict[int, np.ndarray]]] = []
    v7_dir = "round2-v7-periodic-networks"
    for target, short in (("tap_iron", "iron"), ("tap_time_len", "time")):
        for recipe in ("mlp_plr001", "mlp_plr01", "tabm_plr001", "tabm_plr01", "mlp_raw", "tabm_raw"):
            column = load_column(v7_dir, f"{target}-{recipe}", target)
            if column is not None:
                catalogue.append((f"v7/{target}-{recipe}", target, column))
    extras = [
        ("round2-v9-realmlp", "tap_iron-realmlp_td", "tap_iron", None),
        ("round2-v9-realmlp", "tap_iron-realmlp_td_s", "tap_iron", None),
        ("round2-v9-realmlp", "tap_time_len-realmlp_td", "tap_time_len", None),
        ("round2-v9-realmlp", "tap_time_len-realmlp_td_s", "tap_time_len", None),
        ("round2-v11-quantile-representation", "tap_iron-plr_quniform", "tap_iron", None),
        ("round2-v11-quantile-representation", "tap_time_len-plr_quniform", "tap_time_len", None),
        ("round2-v8-feature-attention", "tap_iron-ft_learned", "tap_iron", None),
        ("round2-v8-feature-attention", "tap_time_len-ft_learned", "tap_time_len", None),
        ("round2-v13-ple-repair", "tap_iron-ple_b_repaired", "tap_iron", None),
        ("round2-v13-ple-repair", "tap_time_len-ple_b_repaired", "tap_time_len", None),
    ]
    for directory, key, target, index in extras:
        column = load_column(directory, key, target, index)
        if column is not None:
            catalogue.append((key, target, column))
    for key, target in (("J1", "tap_iron"), ("J2", "tap_iron"), ("P_LL_I", "tap_iron"),
                        ("P_LL_T", "tap_time_len"), ("P_LH_T", "tap_time_len")):
        column = load_column("round2-v17", key, target)
        if column is not None:
            catalogue.append((f"v17/{key}", target, column))
    for key, target, index in (("joint-joint_plr001", "tap_iron", 0), ("joint-task_gated_experts", "tap_iron", 0),
                               ("joint-uniform_experts", "tap_iron", 0), ("joint-joint_ple_repaired", "tap_iron", 0)):
        directory = {"joint-joint_plr001": "round2-v12-joint-tabm",
                     "joint-task_gated_experts": "round2-v15-task-experts",
                     "joint-uniform_experts": "round2-v15-task-experts",
                     "joint-joint_ple_repaired": "round2-v14-joint-ple"}[key]
        column = load_column(directory, key, target, index)
        if column is not None:
            catalogue.append((f"{directory.split('-')[1]}/{key}", target, column))

    grid = [index / 40 for index in range(41)]
    rows = []
    for name, target, member in catalogue:
        deltas, alphas = [], []
        for held in seeds:
            other = [seed for seed in seeds if seed != held][0]
            best_alpha, best = 0.0, np.inf
            for alpha in grid:
                value = wmape(y[target], (1 - alpha) * b0[target][other] + alpha * member[other])
                if value < best:
                    best, best_alpha = value, alpha
            blended = (1 - best_alpha) * b0[target][held] + best_alpha * member[held]
            deltas.append(50 * (wmape(y[target], b0[target][held]) - wmape(y[target], blended)))
            alphas.append(best_alpha)
        rows.append({"member": name, "target": target,
                     "mean_incremental_gain": round(float(np.mean(deltas)), 6),
                     "alphas": [round(alpha, 3) for alpha in alphas]})
    rows.sort(key=lambda row: -row["mean_incremental_gain"])
    return {"count": len(rows), "grid_points": len(grid), "members": rows}


def run(root: Path | str = ROOT_DEFAULT, output: Path | str = OUTPUT_DEFAULT,
        spec_path: Path | str = "configs/round2_v18/SPEC.yaml") -> dict[str, Any]:
    root = Path(root).resolve()
    out = (root / output).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v18"):
        raise ValueError("V18 evidence is private and must stay under local/")
    if out.exists():
        raise FileExistsError("V18 surface output already exists; refusing overwrite")
    yaml.safe_load((root / spec_path).read_text())
    model_spec = yaml.safe_load((root / "configs/round2_v17/SPEC.yaml").read_text())
    report = {"platform_bounds": platform_bounds(),
              "local_surface": _design_local(root, model_spec),
              "member_screen": member_screen(root, model_spec),
              "new_fits": 0, "agent_uploads": 0}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "written", "output": str(out.relative_to(root)),
                      "top_members": report["member_screen"]["members"][:5],
                      "local_mean_delta": {name: row["local_mean_delta_vs_A35"]
                                           for name, row in report["local_surface"]["designs"].items()}},
                     indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    run(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
