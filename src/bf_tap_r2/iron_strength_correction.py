"""Correction: the iron strength curve was measured against the wrong base.

``docs/iron_strength_curve/RESULTS.md`` concluded that the incumbent's DE3 iron
amplification ``q=0.5`` was sub-optimal because ``v12_iron + q*(mean3 - v12_iron)``
peaked at ``q=1.0``.  That family uses the **raw V12 joint member** as its base, but the
incumbent column is ``V32_iron + 0.5*(mean3 - J42)``, whose base ``V32_iron`` is a
different, better column (it is the released V12 blend, not the raw joint member).  The
two families only share the ``mean3`` endpoint, so the earlier comparison never
contained the incumbent.

This module re-measures the family that actually contains the incumbent:

    base = incumbent_iron - 0.5*(mean3 - J42)
    iron(q) = base + q*(mean3 - J42)

See ``docs/iron_strength_correction/RESULTS.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bf_tap_r2.slot_screen import read_json, write_json  # noqa: E402


def wmape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.abs(actual - predicted).sum() / np.abs(actual).sum())


def split_curves(spec: dict) -> dict:
    grid = spec["amplification_grid"]
    out = {"splits": {}, "grid": grid}
    released_gains, incumbent_gains = [], []
    for split in spec["splits"]:
        saved = np.load(ROOT / spec["oof_template"].format(split=split), allow_pickle=False)
        ids = np.array([str(value) for value in saved["ids"]])
        actual = saved["actual"][:, 0]
        incumbent = saved["reference_iron"]
        mean3 = saved["old_members"].mean(axis=0)
        position = {sample_id: index for index, sample_id in enumerate(ids)}
        native = np.full(len(ids), np.nan)
        for fold in range(spec["folds"]):
            fold_data = np.load(ROOT / spec["fold_template"].format(split=split, fold=fold),
                                allow_pickle=False)
            for key, sample_id in enumerate(fold_data["query_ids"]):
                native[position[str(sample_id)]] = fold_data["v12_iron"][key]
        if not np.isfinite(native).all():
            raise ValueError("Native member rows did not align with the OOF rows")
        base = incumbent - 0.5 * (mean3 - native)
        curve = {q: wmape(actual, base + q * (mean3 - native)) for q in grid}
        optimum = min(curve, key=curve.get)
        released = wmape(actual, mean3)
        entry = {
            "incumbent_wmape": wmape(actual, incumbent),
            "released_plain_mean3_wmape": released,
            "released_gain_vs_incumbent": 50 * (wmape(actual, incumbent) - released),
            "curve": {str(q): curve[q] for q in grid},
            "score_gain_vs_incumbent": {str(q): 50 * (curve[0.5] - curve[q]) for q in grid},
            "optimum_q": optimum,
            "q1_gain_vs_incumbent": 50 * (curve[0.5] - curve[1.0]),
            "base_minus_native_rms": float(np.sqrt(((base - native) ** 2).mean())),
            "mean3_minus_native_rms": float(np.sqrt(((mean3 - native) ** 2).mean())),
        }
        out["splits"][str(split)] = entry
        released_gains.append(entry["released_gain_vs_incumbent"])
        incumbent_gains.append(entry["q1_gain_vs_incumbent"])
    out["released_plain_mean3_gain_mean"] = float(np.mean(released_gains))
    out["q1_gain_mean"] = float(np.mean(incumbent_gains))
    out["released_plain_mean3_is_worse_on_both_splits"] = all(gain < 0 for gain in released_gains)
    out["q1_is_worse_or_flat_on_both_splits"] = all(gain <= 0 for gain in incumbent_gains)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/iron_strength_correction/SPEC.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = read_json(ROOT / args.spec)
    result = split_curves(spec)
    report = {
        "stage": spec["stage"],
        "objective": spec["objective"],
        "superseded": spec["superseded_claim"],
        "correction": spec["correction"],
        "withdrawn_candidate": spec["withdrawn_candidate"],
        "result": result,
        "budget_actual": {"new_fits": 0, "training_label_reads": 0, "new_packages": 0,
                          "desktop_writes": 0, "agent_uploads": 0},
        "limits": spec["limits"],
    }
    write_json(Path(args.output) / "report.json", report)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
