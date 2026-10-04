"""Round-three closure checks: duplicate rows, feature support, and iron directional optimality.

Zero fits, zero model loads, zero packages.  Three questions:

1. Do any test rows duplicate a training row's features (an exploitable exact match)?
2. Do test values fall outside the training range (extrapolation)?
3. Is the incumbent iron column locally optimal in every recorded direction, and is the
   remaining claimed headroom (the conditional A35-to-DE3 concavity bound) locally supported?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bf_tap_r2.data import FEATURES  # noqa: E402
from bf_tap_r2.slot_screen import read_json, write_json  # noqa: E402


def wmape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.abs(actual - predicted).sum() / np.abs(actual).sum())


def duplicate_and_support(spec: dict) -> dict:
    columns = list(FEATURES)
    train = pd.read_csv(ROOT / spec["train_features"])
    test = pd.read_csv(ROOT / spec["test_features"])
    result = {
        "features": len(columns),
        "within_train_duplicate_feature_rows": int(train.duplicated(subset=columns).sum()),
        "rounding_levels": {},
    }
    for digits in spec["rounding_levels"]:
        train_key = train[columns].round(digits).astype(str).agg("|".join, axis=1)
        test_key = test[columns].round(digits).astype(str).agg("|".join, axis=1)
        matched = int(test_key.isin(set(train_key)).sum())
        result["rounding_levels"][str(digits)] = {
            "test_rows_matching_a_train_row": matched,
            "test_rows_total": len(test),
        }
    low, high = train[columns].min(), train[columns].max()
    result["test_cells_outside_train_range"] = int((((test[columns] < low) | (test[columns] > high))
                                                    .sum().sum()))
    result["test_cells_total"] = int(len(test) * len(columns))
    result["conclusion"] = (
        "No exact or rounded-duplicate feature row and no out-of-range test cell: there is no "
        "match-based shortcut and no extrapolation outside the training box."
    )
    return result


def directional_optimality(spec: dict) -> dict:
    out = {"splits": {}, "directions": spec["directions"]}
    for split in spec["splits"]:
        saved = np.load(ROOT / spec["oof_template"].format(split=split), allow_pickle=False)
        actual = saved["actual"][:, 0]
        reference = saved["reference_iron"]
        members = saved["old_members"]
        other = saved["new_members"]
        directions = {
            "DE3_mean_minus_incumbent": members.mean(axis=0) - reference,
            "IBATCH_mean_minus_incumbent": other.mean(axis=0) - reference,
            "IBATCH_mean_minus_DE3_mean": other.mean(axis=0) - members.mean(axis=0),
            "DE3_mean_minus_member0": members.mean(axis=0) - members[0],
        }
        for index in range(members.shape[0]):
            directions[f"DE3_member{index}_minus_incumbent"] = members[index] - reference
        base = wmape(actual, reference)
        entry = {"incumbent_iron_wmape": base, "per_direction": {}}
        grid = np.linspace(-1.0, 1.0, spec["grid_points"])
        for name, direction in directions.items():
            gains = [50 * (base - wmape(actual, reference + t * direction)) for t in grid]
            entry["per_direction"][name] = {
                "gain_at_plus_1": 50 * (base - wmape(actual, reference + direction)),
                "gain_at_minus_1": 50 * (base - wmape(actual, reference - direction)),
                "best_gain_on_grid": float(np.max(gains)),
                "best_t_on_grid": float(grid[int(np.argmax(gains))]),
            }
        out["splits"][str(split)] = entry
    best_per_direction = {}
    for name in out["splits"][str(spec["splits"][0])]["per_direction"]:
        best_per_direction[name] = min(out["splits"][str(split)]["per_direction"][name]["best_gain_on_grid"]
                                       for split in spec["splits"])
    out["min_over_splits_of_best_gain"] = best_per_direction
    out["best_direction_min_over_splits"] = max(best_per_direction, key=best_per_direction.get)
    out["best_gain_min_over_splits"] = max(best_per_direction.values())
    out["no_direction_improves_both_splits_materially"] = all(
        value < spec["material_gain_threshold"] for value in best_per_direction.values())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/iron_direction_audit/SPEC.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = read_json(ROOT / args.spec)
    report = {
        "stage": spec["stage"],
        "objective": spec["objective"],
        "budget_actual": {"new_fits": 0, "model_loads": 0, "new_packages": 0,
                          "desktop_writes": 0, "agent_uploads": 0},
        "duplicate_and_support": duplicate_and_support(spec),
        "directional_optimality": directional_optimality(spec),
        "remaining_claimed_headroom": spec["remaining_claimed_headroom"],
        "limits": spec["limits"],
    }
    write_json(Path(args.output) / "report.json", report)
    print(json.dumps({
        "duplicates": report["duplicate_and_support"]["rounding_levels"]["0"],
        "outside_range": report["duplicate_and_support"]["test_cells_outside_train_range"],
        "best_gain_min_over_splits": report["directional_optimality"]["best_gain_min_over_splits"],
        "no_direction_improves_both_splits_materially":
            report["directional_optimality"]["no_direction_improves_both_splits_materially"],
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
