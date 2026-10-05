"""Three-seed extension of the DE3 iron average, the only remaining positive-expectation move.

The incumbent iron column is ``V32_iron + 0.5 * (mean(3 seeds) - J42)``.  Recorded per-seed
out-of-fold predictions fit ``W(r) = A + c/r`` with residuals at 1e-4 relative, and that fit
predicts +0.0035 raw (+0.0017 amplified) for extending the average from three to six seeds.
This module trains the three new seeds on the recorded folds and measures the change.

See ``docs/de3_iron_seed6/PREREGISTRATION.md``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bf_tap_r2.component_regularization import ComponentRegressor  # noqa: E402
from bf_tap_r2.component_regularization_run import RECIPE  # noqa: E402
from bf_tap_r2.slot_screen import read_json, write_json  # noqa: E402
from bf_tap_r2.v5_library import load_v5_training_frame  # noqa: E402

TARGETS = ("tap_iron", "tap_time_len")


def wmape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.abs(actual - predicted).sum() / np.abs(actual).sum())


def fold_queries(spec: dict) -> dict:
    """Recorded held-out query ids per split and fold; the folds are reused, not regenerated."""
    layout = {}
    for split in spec["splits"]:
        folds = []
        for fold in range(spec["folds"]):
            path = ROOT / spec["fold_source"].format(split=split, fold=fold)
            with np.load(path, allow_pickle=False) as saved:
                folds.append([str(value) for value in saved["query_ids"]])
        layout[split] = folds
    return layout


def train_new_seeds(spec: dict, frame, layout: dict, output: Path, settings: dict) -> dict:
    counts = {"estimators": 0, "optimizers": 0, "selected_epochs": {}}
    for split in spec["splits"]:
        for fold in range(spec["folds"]):
            query_ids = layout[split][fold]
            training = frame.loc[~frame.sample_id.isin(query_ids)].reset_index(drop=True)
            query = (frame.set_index("sample_id").loc[query_ids].reset_index())
            actual = query["tap_iron"].to_numpy(dtype=float)
            query_frame = query.drop(columns=[name for name in TARGETS if name in query.columns])
            unit = output / f"members-s{split}-f{fold}"
            unit.mkdir(parents=True, exist_ok=False)
            columns = {}
            for seed in spec["new_seeds"]:
                directory = unit / f"training-seed-{seed}"
                directory.mkdir(exist_ok=False)
                started = time.monotonic()
                model = ComponentRegressor(RECIPE, dict(settings, random_seed=seed), "BASE", {}, directory)
                model.fit(training, training[list(TARGETS)].to_numpy())
                prediction = model.predict(query_frame)
                with (directory / "predictions.npz").open("xb") as stream:
                    np.savez_compressed(stream, prediction=prediction,
                                        query_ids=query_frame.sample_id.to_numpy(dtype=str))
                write_json(directory / "metadata.json", {
                    "training_seed": seed, "selected_epoch": model.metadata_["selected_epoch"],
                    "seconds": time.monotonic() - started,
                    "fit_rows": int(len(training)), "query_rows": int(len(query)),
                })
                columns[str(seed)] = prediction[:, 0]
                counts["estimators"] += 1
                counts["optimizers"] += 2
                counts["selected_epochs"][f"s{split}-f{fold}-{seed}"] = model.metadata_["selected_epoch"]
            write_json(unit / "columns.json", {
                "split": split, "fold": fold,
                "training_rows": int(len(training)), "query_rows": int(len(query)),
                "verified_actual_sha256": None,
            })
            np.savez_compressed(unit / "columns.npz", actual=actual, query_ids=query.sample_id.to_numpy(dtype=str),
                                **{f"new_seed_{seed}": column for seed, column in columns.items()})
    return counts


def evaluate(spec: dict, layout: dict, output: Path, frame) -> dict:
    result = {"splits": {}}
    for split in spec["splits"]:
        reference = np.load(ROOT / spec["incumbent_column_source"].format(split=split), allow_pickle=False)
        ids = [str(value) for value in reference["ids"]]
        position = {sample_id: index for index, sample_id in enumerate(ids)}
        actual = reference["actual"][:, 0]
        incumbent = reference["reference_iron"]
        old_members = reference["old_members"]
        new_members = np.zeros((len(spec["new_seeds"]), len(ids)))
        for fold in range(spec["folds"]):
            with np.load(output / f"members-s{split}-f{fold}/columns.npz", allow_pickle=False) as saved:
                query_ids = [str(value) for value in saved["query_ids"]]
                for row, sample_id in enumerate(query_ids):
                    index = position[sample_id]
                    for offset, seed in enumerate(spec["new_seeds"]):
                        new_members[offset, index] = saved[f"new_seed_{seed}"][row]
        mean3 = old_members.mean(axis=0)
        mean6 = np.concatenate([old_members, new_members], axis=0).mean(axis=0)
        candidate = incumbent + 0.5 * (mean6 - mean3)
        base_wmape = wmape(actual, incumbent)
        candidate_wmape = wmape(actual, candidate)
        result["splits"][str(split)] = {
            "incumbent_iron_wmape": base_wmape,
            "candidate_iron_wmape": candidate_wmape,
            "score_gain": 50 * (base_wmape - candidate_wmape),
            "rms_change": float(np.sqrt(((candidate - incumbent) ** 2).mean())),
            "rows": int(len(ids)),
        }
    gains = [entry["score_gain"] for entry in result["splits"].values()]
    result["mean_gain"] = float(np.mean(gains))
    result["both_splits_positive"] = all(gain > 0 for gain in gains)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/de3_iron_seed6/SPEC.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise SystemExit(f"{name} must be pinned to 1")
    spec = read_json(ROOT / args.spec)
    output = Path(args.output).resolve()
    layout = fold_queries(spec)
    frame = load_v5_training_frame(ROOT)
    if not args.evaluate_only:
        settings = yaml.safe_load((ROOT / "configs/strong_component_regularization/SPEC.yaml")
                                  .read_text(encoding="utf-8"))["training"]["tap_iron"]
        output.mkdir(parents=True, exist_ok=True)
        counts = train_new_seeds(spec, frame, layout, output, settings)
    else:
        recorded = sorted(output.glob("members-s*-f*/training-seed-*/metadata.json"))
        epochs = {}
        for path in recorded:
            meta = read_json(path)
            epochs[path.parent.name + "-" + path.parent.parent.name] = meta["selected_epoch"]
        counts = {"estimators": len(recorded), "optimizers": 2 * len(recorded),
                  "selected_epochs": epochs, "evaluated_from_recorded_units": True}
    evaluation = evaluate(spec, layout, output, frame)
    report = {
        "stage": spec["stage"],
        "candidate": spec["candidate"],
        "reference": spec["reference"],
        "expected_effect_basis": spec["expected_effect_basis"],
        "counts": counts,
        "evaluation": evaluation,
        "gate_passed": evaluation["both_splits_positive"],
        "budget_actual": {"new_estimators": counts["estimators"], "new_optimizers": counts["optimizers"],
                          "full_fits": 0, "packages": 0, "desktop_writes": 0, "agent_uploads": 0},
        "limits": spec["limits"],
    }
    write_json(output / "report.json", report)
    print(json.dumps({"counts": {k: v for k, v in counts.items() if k != "selected_epochs"},
                      "evaluation": evaluation}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
