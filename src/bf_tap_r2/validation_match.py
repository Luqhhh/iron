"""Can a matched local validation set stand in for the platform's 322 rows?

Zero new fits.  Three questions, all answered from recorded packages and recorded OOF
columns:

A. Test side, unstructured: the platform delta of a receipt is, to first order, a fixed
   linear functional of its 322-row prediction change vector with unknown per-row
   weights.  Is that weight vector identifiable from the 23 registered receipts?
B. Test side, rule-defined: the same question with the weights restricted to a small
   basis built from row features, i.e. "find a rule that picks matching rows".
C. Local side: a 322-row subset of the training rows can be chosen to score at the
   platform level (96.3979).  Does such a level-matched subset reproduce the platform's
   verdict on a candidate?

See ``docs/validation_match/PREREGISTRATION.md``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bf_tap_r2.slot_screen import read_json, write_json, read_package, template_ids  # noqa: E402

TARGETS = ("pred_tap_iron", "pred_tap_time_len")


def rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(math.sqrt(float(((a - b) ** 2).mean())))


def leave_one_out(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Minimum-norm least-squares leave-one-out predictions."""
    predictions = np.empty(len(y), dtype=float)
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        coefficients, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        predictions[index] = float(X[index] @ coefficients)
    return predictions


def task_a(records: list[dict]) -> dict:
    out = {}
    for key in TARGETS:
        subset = [record for record in records if record["changed_target"] == key]
        if len(subset) < 4:
            out[key] = {"n": len(subset), "status": "too_few_receipts"}
            continue
        X = np.array([record["delta"] for record in subset])
        y = np.array([record["platform_delta"] for record in subset])
        loo = leave_one_out(X, y)
        full, *_ = np.linalg.lstsq(X, y, rcond=None)
        generator = np.random.default_rng(20261005)
        shuffled = [rmse(leave_one_out(X, generator.permutation(y)), y) for _ in range(20)]
        out[key] = {
            "n_receipts": len(subset),
            "n_free_weights": X.shape[1],
            "ratio_receipts_to_weights": len(subset) / X.shape[1],
            "train_fit_rmse": rmse(X @ full, y),
            "leave_one_out_rmse": rmse(loo, y),
            "shuffled_label_loo_rmse_median": float(np.median(shuffled)),
            "null_zero_rmse": rmse(np.zeros_like(y), y),
            "null_fold_mean_rmse": rmse(np.full_like(y, y.mean()), y),
            "leave_one_out_rmse_over_null_zero": rmse(loo, y) / rmse(np.zeros_like(y), y),
            "ids": [record["id"] for record in subset],
        }
    return out


def row_basis(parent_column: np.ndarray, spouts: np.ndarray, kind: str = "full") -> np.ndarray:
    """Small a-priori basis.

    ``constant`` isolates the column-mean-shift direction, which the slot screen already
    measures as ``bias_pct``; ``full`` adds spout, prediction quintile and relative level.
    """
    level = parent_column / float(np.abs(parent_column).mean())
    columns = [np.ones_like(level)]
    if kind == "full":
        quantiles = np.quantile(parent_column, [0.2, 0.4, 0.6, 0.8])
        bucket = np.digitize(parent_column, quantiles)
        columns.append(level)
        columns += [(spouts == value).astype(float) for value in sorted(set(spouts.tolist()))[1:]]
        columns += [(bucket == value).astype(float) for value in range(1, 5)]
    return np.stack(columns, axis=1)


def task_b(records: list[dict]) -> dict:
    out = {}
    for key in TARGETS:
        subset = [record for record in records if record["changed_target"] == key]
        if len(subset) < 4:
            out[key] = {"n": len(subset), "status": "too_few_receipts"}
            continue
        entry = {"n_receipts": len(subset), "ids": [record["id"] for record in subset],
                 "bases": {}}
        y = np.array([record["platform_delta"] for record in subset])
        for kind in ("constant", "full"):
            designs = [record["delta"] @ row_basis(record["parent_column"], record["spouts"], kind)
                       for record in subset]
            X = np.array(designs)
            loo = leave_one_out(X, y)
            full, *_ = np.linalg.lstsq(X, y, rcond=None)
            generator = np.random.default_rng(20261005)
            shuffled = [rmse(leave_one_out(X, generator.permutation(y)), y) for _ in range(50)]
            entry["bases"][kind] = {
                "n_basis_weights": X.shape[1],
                "train_fit_rmse": rmse(X @ full, y),
                "leave_one_out_rmse": rmse(loo, y),
                "null_zero_rmse": rmse(np.zeros_like(y), y),
                "shuffled_label_loo_rmse_median": float(np.median(shuffled)),
            }
        out[key] = entry
    return out


def wmape(actual, predicted) -> float:
    return float(np.abs(np.asarray(actual) - np.asarray(predicted)).sum() / np.abs(actual).sum())


def task_c(spec: dict) -> dict:
    out = {"draws": spec["task_c"]["draws"], "rows": spec["task_c"]["rows"],
           "seed": spec["task_c"]["seed"], "splits": {}}
    generator = np.random.default_rng(spec["task_c"]["seed"])
    for split in spec["task_c"]["splits"]:
        saved = np.load(ROOT / spec["task_c"]["oof_template"].format(split=split), allow_pickle=False)
        actual, iron, incumbent = saved["actual"], saved["iron"], saved["reference"]
        candidates = {name: saved[name] for name in spec["task_c"]["candidates"]}
        rows = len(actual)
        draws = spec["task_c"]["draws"]
        size = spec["task_c"]["rows"]

        def score(selection, time_column) -> float:
            return 100.0 - 50.0 * (wmape(actual[selection, 0], iron[selection])
                                   + wmape(actual[selection, 1], time_column[selection]))

        full_incumbent = score(np.arange(rows), incumbent)
        full_candidate = {name: score(np.arange(rows), column) for name, column in candidates.items()}
        levels, deltas = np.empty(draws), {name: np.empty(draws) for name in candidates}
        for draw in range(draws):
            selection = generator.choice(rows, size=size, replace=False)
            levels[draw] = score(selection, incumbent)
            for name, column in candidates.items():
                deltas[name][draw] = score(selection, column) - levels[draw]
        level_low, level_high = spec["task_c"]["platform_level_band"]
        matched = (levels >= level_low) & (levels <= level_high)
        entry = {
            "full_oof_incumbent_score": full_incumbent,
            "full_oof_candidate_score": full_candidate,
            "full_oof_candidate_delta": {name: full_candidate[name] - full_incumbent for name in candidates},
            "subset_level_quantiles": {str(q): float(np.quantile(levels, q)) for q in (0.025, 0.25, 0.5, 0.75, 0.975)},
            "subset_level_min": float(levels.min()),
            "subset_level_max": float(levels.max()),
            "matched_subset_fraction": float(matched.mean()),
            "platform_level": spec["task_c"]["platform_level"],
            "level_matched_delta_quantiles": {},
            "delta_vs_level_spearman": {},
            "platform_delta": {},
        }
        for name in candidates:
            values = deltas[name]
            entry["level_matched_delta_quantiles"][name] = (
                {str(q): float(np.quantile(values[matched], q)) for q in (0.025, 0.25, 0.5, 0.75, 0.975)}
                if matched.any() else None)
            entry["delta_vs_level_spearman"][name] = spearman(levels, values)
            entry["platform_delta"][name] = spec["task_c"]["platform_delta"].get(name)
        out["splits"][str(split)] = entry
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(values):
        order = np.argsort(values, kind="stable")
        out = np.empty(len(values), float)
        out[order] = np.arange(1, len(values) + 1, dtype=float)
        return out
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/validation_match/SPEC.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = read_json(ROOT / args.spec)
    slot_spec = read_json(ROOT / spec["receipt_source"])
    expected = template_ids()
    cache: dict[str, dict] = {}
    records = []
    for receipt in slot_spec["receipts"]:
        for path in (receipt["dir"], receipt["parent_dir"]):
            if path not in cache:
                cache[path] = read_package(ROOT / path, expected)
        candidate, parent = cache[receipt["dir"]], cache[receipt["parent_dir"]]
        changed = [key for key in TARGETS if candidate["fields"][key] != parent["fields"][key]]
        if len(changed) != 1:
            raise ValueError(f"{receipt['id']}: identity gate failed")
        key = changed[0]
        records.append({
            "id": receipt["id"],
            "changed_target": key,
            "platform_delta": float(receipt["platform_delta"]),
            "delta": candidate["values"][key] - parent["values"][key],
            "parent_column": parent["values"][key],
        })
    spouts = read_spouts(expected)
    for record in records:
        record["spouts"] = spouts

    report = {
        "stage": spec["stage"],
        "objective": spec["objective"],
        "reference": slot_spec["reference"],
        "budget_actual": {"new_fits": 0, "training_label_reads": 0, "new_packages": 0,
                          "desktop_writes": 0, "agent_uploads": 0},
        "question": spec["question"],
        "task_a_unstructured_test_side_weights": task_a(records),
        "task_b_rule_defined_test_side_weights": task_b(records),
        "task_c_level_matched_subsets": task_c(spec),
        "limits": spec["limits"],
    }
    write_json(Path(args.output) / "report.json", report)
    weights = {}
    for key in TARGETS:
        subset = [record for record in records if record["changed_target"] == key and len(record["delta"]) == 322]
        if len(subset) < 4:
            continue
        X = np.array([record["delta"] for record in subset])
        y = np.array([record["platform_delta"] for record in subset])
        coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
        weights[key] = {"weights": coefficients, "ids": np.array([record["id"] for record in subset]),
                        "platform_delta": y, "train_rmse": rmse(X @ coefficients, y),
                        "n_receipts": len(subset), "status": "retrospective_fit_prospective_use_only"}
    np.savez(Path(args.output) / "fitted-weights.npz", **{
        f"{key}__{field}": value for key, payload in weights.items() for field, value in payload.items()})
    print(json.dumps({"A": report["task_a_unstructured_test_side_weights"],
                      "B": report["task_b_rule_defined_test_side_weights"]},
                     ensure_ascii=False, indent=1)[:4000])
    for split, entry in report["task_c_level_matched_subsets"]["splits"].items():
        print(f"--- split {split}: full incumbent {entry['full_oof_incumbent_score']:.4f} "
              f"candidate delta {entry['full_oof_candidate_delta']}")
        print(f"    subset level range {entry['subset_level_min']:.4f}..{entry['subset_level_max']:.4f} "
              f"matched fraction {entry['matched_subset_fraction']:.3f} platform level {entry['platform_level']}")
        for name, quantiles in entry["level_matched_delta_quantiles"].items():
            print(f"    {name}: level-matched delta quantiles {quantiles} "
                  f"level-delta spearman {entry['delta_vs_level_spearman'][name]:+.3f} "
                  f"platform {entry['platform_delta'][name]}")


def read_spouts(expected_ids: list[str]) -> np.ndarray:
    """Spout label per query row, aligned to the official template order."""
    import csv
    with (ROOT / "复赛_test/test_samples.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {row["sample_id"]: int(row["spout_no"]) for row in rows}
    if set(lookup) != set(expected_ids):
        raise ValueError("test_samples.csv ID set differs from the official template")
    return np.array([lookup[sid] for sid in expected_ids])


if __name__ == "__main__":
    main()
