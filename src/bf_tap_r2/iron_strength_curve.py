"""Zero-fit iron strength and seed-averaging curves from the recorded DE3 development run.

The incumbent iron column is ``V12_iron + 0.5 * (mean(3 DE3 seeds) - V12_iron)``.  The
recorded development run stores every seed's out-of-fold prediction, so both the number
of averaged seeds and the amplification factor can be re-measured without any new fit.

See ``docs/iron_strength_curve/PREREGISTRATION.md``.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bf_tap_r2.slot_screen import read_json, write_json  # noqa: E402

SEED_KEYS = ("seed_42", "seed_104729", "seed_130363")


def wmape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.abs(actual - predicted).sum() / np.abs(actual).sum())


def label_map(spec: dict) -> dict[str, float]:
    labels: dict[str, float] = {}
    for split in spec["splits"]:
        saved = np.load(ROOT / spec["label_template"].format(split=split), allow_pickle=False)
        for sample_id, value in zip(saved["ids"], saved["actual"][:, 0]):
            labels[str(sample_id)] = float(value)
    return labels


def unit(spec: dict, split: int, fold: int) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    base = ROOT / spec["source"]
    saved = np.load(base / f"tap_iron-DE3-s{split}-f{fold}/predictions.npz", allow_pickle=False)
    reference = np.load(base / f"reference-s{split}-f{fold}/predictions.npz", allow_pickle=False)
    seeds = {key: saved[key] for key in SEED_KEYS}
    return saved["query_ids"], seeds, reference["v12_iron"]


def seed_curve(spec: dict, labels: dict[str, float]) -> dict:
    pooled: dict[int, list[float]] = {size: [0.0, 0.0] for size in (1, 2, 3)}
    per_split: dict[int, dict[int, list[float]]] = {split: {s: [0.0, 0.0] for s in (1, 2, 3)}
                                                  for split in spec["splits"]}
    for split in spec["splits"]:
        for fold in range(spec["folds"]):
            query_ids, seeds, _ = unit(spec, split, fold)
            actual = np.array([labels[str(sample_id)] for sample_id in query_ids])
            denominator = float(np.abs(actual).sum())
            for size in (1, 2, 3):
                for combination in itertools.combinations(SEED_KEYS, size):
                    prediction = np.mean([seeds[key] for key in combination], axis=0)
                    residual = float(np.abs(actual - prediction).sum())
                    pooled[size][0] += residual
                    pooled[size][1] += denominator
                    per_split[split][size][0] += residual
                    per_split[split][size][1] += denominator
    pooled_wmape = {size: pooled[size][0] / pooled[size][1] for size in pooled}
    return {
        "pooled_wmape": {str(size): pooled_wmape[size] for size in pooled_wmape},
        "pooled_score_gain": {f"{a}_to_{b}": 50 * (pooled_wmape[a] - pooled_wmape[b])
                              for a, b in ((1, 2), (2, 3), (1, 3))},
        "per_split_wmape": {str(split): {str(size): per_split[split][size][0] / per_split[split][size][1]
                                         for size in per_split[split]}
                            for split in per_split},
        "per_split_gain_2_to_3": {str(split): 50 * (per_split[split][2][0] / per_split[split][2][1]
                                                    - per_split[split][3][0] / per_split[split][3][1])
                                  for split in per_split},
        "combinations_per_size": {"1": 3, "2": 3, "3": 1},
    }


def amplification_curve(spec: dict, labels: dict[str, float]) -> dict:
    grid = spec["amplification_grid"]
    pooled = {q: [0.0, 0.0] for q in grid}
    per_fold = []
    for split in spec["splits"]:
        for fold in range(spec["folds"]):
            query_ids, seeds, v12 = unit(spec, split, fold)
            actual = np.array([labels[str(sample_id)] for sample_id in query_ids])
            denominator = float(np.abs(actual).sum())
            de3 = np.mean([seeds[key] for key in SEED_KEYS], axis=0)
            values = {}
            for q in grid:
                residual = float(np.abs(actual - (v12 + q * (de3 - v12))).sum())
                pooled[q][0] += residual
                pooled[q][1] += denominator
                values[q] = residual / denominator
            per_fold.append({
                "split": split, "fold": fold,
                "gain_0.5_to_1.0": 50 * (values[0.5] - values[1.0]),
                "gain_0.5_to_0.75": 50 * (values[0.5] - values[0.75]),
            })
    curve = {q: pooled[q][0] / pooled[q][1] for q in grid}
    best = min(curve, key=curve.get)
    return {
        "grid": grid,
        "pooled_wmape": {str(q): curve[q] for q in grid},
        "score_gain_vs_0.5": {str(q): 50 * (curve[0.5] - curve[q]) for q in grid},
        "pooled_best_q": best,
        "per_fold": per_fold,
        "folds_improved_0.5_to_1.0": sum(1 for row in per_fold if row["gain_0.5_to_1.0"] > 0),
        "folds_total": len(per_fold),
        "worst_fold_gain_0.5_to_1.0": min(row["gain_0.5_to_1.0"] for row in per_fold),
        "mean_fold_gain_0.5_to_1.0": float(np.mean([row["gain_0.5_to_1.0"] for row in per_fold])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/iron_strength_curve/SPEC.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = read_json(ROOT / args.spec)
    labels = label_map(spec)
    report = {
        "stage": spec["stage"],
        "objective": spec["objective"],
        "reference": spec["reference"],
        "source": spec["source"],
        "budget_actual": {"new_fits": 0, "training_label_reads": 0, "new_packages": 0,
                          "desktop_writes": 0, "agent_uploads": 0},
        "seed_averaging_curve": seed_curve(spec, labels),
        "amplification_curve": amplification_curve(spec, labels),
        "limits": spec["limits"],
    }
    write_json(Path(args.output) / "report.json", report)
    print(json.dumps({"seed": report["seed_averaging_curve"]["pooled_score_gain"],
                      "per_split_2_to_3": report["seed_averaging_curve"]["per_split_gain_2_to_3"],
                      "amplification_gain_vs_0.5": report["amplification_curve"]["score_gain_vs_0.5"],
                      "best_q": report["amplification_curve"]["pooled_best_q"],
                      "folds_improved": f"{report['amplification_curve']['folds_improved_0.5_to_1.0']}"
                                        f"/{report['amplification_curve']['folds_total']}",
                      "mean_fold_gain": report["amplification_curve"]["mean_fold_gain_0.5_to_1.0"],
                      "worst_fold_gain": report["amplification_curve"]["worst_fold_gain_0.5_to_1.0"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
