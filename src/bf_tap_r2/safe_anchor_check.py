"""Independent saved-evidence check; never fit estimators or change predictions."""
import argparse
import json
from pathlib import Path
import zipfile

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .audit import digest, write_json
from .cv import verify_audit
from .data import TARGETS, load_config, load_snapshot
from .submission import ZIP_NAME, csv_bytes, predict_m0, validate_result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def independent_wmape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    require(y.shape == p.shape and np.isfinite(p).all(), "Invalid metric inputs")
    return float(sum(abs(y-p)) / sum(abs(y)))


def check(root, run):
    spec = yaml.safe_load((run / "config.yaml").read_text())
    summary = json.loads((run / "summary.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    require(summary["status"] == "complete", "Incomplete run")
    require(digest(root / "uv.lock") == manifest["uv_lock_sha256"], "Dependency lock changed")
    for file, sha in manifest["source_sha256"].items():
        require(digest(root / file) == sha, f"Source changed: {file}")
    require(digest(run / "config.yaml") == manifest["config_sha256"], "Config changed")
    verify_audit(root, root / "local/runs/round2-v0.1/p0-audit-r2")
    parent = root / spec["parent"]
    require(digest(parent / "summary.json") == manifest["parent_summary_sha256"], "Parent summary changed")
    for file, sha in manifest["parent_oof_sha256"].items():
        require(digest(parent / "oof" / file) == sha, "Parent OOF changed")
    require(digest(run / "fold_assignments.csv") == manifest["folds_sha256"] == digest(parent / "fold_assignments.csv"), "Fold bytes changed")
    cfg = load_config(root / "configs/round2_v0_1/data.yaml")
    frame, _ = load_snapshot(root, cfg, "train")
    frame = frame.sort_values("sample_id").reset_index(drop=True)
    test, _ = load_snapshot(root, cfg, "test")
    assignments = pd.read_csv(run / "fold_assignments.csv")
    events = [json.loads(line) for line in (run / "fit_ledger.jsonl").read_text().splitlines()]
    starts = [e for e in events if e["kind"] == "q2_fit_start"]
    ends = [e for e in events if e["kind"] == "q2_fit_complete"]
    require(len(starts) == len(ends) == 20, "Q2 fit count mismatch")
    require([e["fit_number"] for e in starts] == list(range(1, 21)), "Duplicate Q2 fit starts")
    require([e["fit_number"] for e in ends] == list(range(1, 21)), "Duplicate Q2 fit completions")
    gate = json.loads((run / "known_signal_gate.json").read_text())
    for target in TARGETS:
        report = gate[target]
        require(report["pass"] and report["ratio"] <= .1 and report["regressor_fits"] == 1, "Synthetic gate invalid")
    records = {}
    for seed in spec["fold_seeds"]:
        folds = assignments.loc[assignments.seed == seed].set_index("sample_id").loc[frame.sample_id, "fold"].to_numpy()
        for target in TARGETS:
            medians = np.empty(len(frame))
            for fold in range(5):
                medians[folds == fold] = np.median(frame.loc[folds != fold, target])
            for route in ("M0", "S25", "Q2"):
                path = run / "oof" / f"{seed}-{route}-{target}.csv"
                saved = pd.read_csv(path).set_index("sample_id")
                require(saved.index.is_unique and set(saved.index) == set(frame.sample_id), "OOF IDs invalid")
                saved = saved.loc[frame.sample_id]
                np.testing.assert_array_equal(saved.fold, folds)
                np.testing.assert_allclose(saved[target], frame[target], rtol=1e-12)
                predicted = saved[f"pred_{target}"].to_numpy()
                if route == "M0":
                    np.testing.assert_allclose(predicted, medians, rtol=1e-12)
                if route == "S25":
                    source = spec["S25"]["sources"][target]
                    old = pd.read_csv(parent / "oof" / f"{seed}-{source}-{target}.csv").set_index("sample_id").loc[frame.sample_id, f"pred_{target}"].to_numpy()
                    np.testing.assert_allclose(predicted, .75*medians + .25*old, rtol=1e-12)
                if route == "Q2":
                    for fold in range(5):
                        tag = f"{seed}-Q2-{target}-fold{fold}"
                        model_path = run / "models" / f"{tag}.joblib"
                        event = next(e for e in ends if e["tag"] == tag)
                        require(digest(model_path) == event["model_sha256"], "Model identity changed")
                        model = joblib.load(model_path)
                        with threadpool_limits(limits=1):
                            actual = model.predict(frame.loc[folds == fold])
                        np.testing.assert_allclose(predicted[folds == fold], actual, rtol=1e-12, atol=1e-10)
                metrics = {"wmape": independent_wmape(frame[target], predicted),
                           "by_fold": {str(f): independent_wmape(frame.loc[folds == f, target], predicted[folds == f]) for f in range(5)},
                           "by_spout": {str(s): independent_wmape(frame.loc[frame.spout_no == s, target], predicted[frame.spout_no == s]) for s in sorted(frame.spout_no.unique())}}
                stored = summary["metrics"][route][str(seed)][target]
                require(abs(stored["wmape"]-metrics["wmape"]) < 1e-12, "Pooled metric mismatch")
                for group in ("by_fold", "by_spout"):
                    for k, value in metrics[group].items():
                        require(abs(stored[group][k]-value) < 1e-12, "Grouped metric mismatch")
                records[seed, target, route] = metrics
    # Independently recompute eligibility from saved aligned predictions.
    for target in TARGETS:
        for route in ("S25", "Q2"):
            wins, overall, worst = 0, [], -np.inf
            for seed in spec["fold_seeds"]:
                baseline, candidate = records[seed, target, "M0"], records[seed, target, route]
                overall.append(candidate["wmape"] < baseline["wmape"]-1e-12)
                wins += sum(candidate["by_fold"][str(f)] < baseline["by_fold"][str(f)]-1e-12 for f in range(5))
                worst = max(worst, *(candidate["by_spout"][s]-baseline["by_spout"][s] for s in baseline["by_spout"]))
            eligible = all(overall) and wins >= 7 and worst <= .001+1e-12
            recorded = summary["promotion"][target][route]
            require(eligible == recorded["eligible"] and wins == recorded["improved_folds"] and abs(worst-recorded["worst_spout_delta"]) < 1e-12, "Promotion mismatch")
    m0 = root / spec["m0_release"]
    identity = json.loads((m0 / "manifest.json").read_text())
    require(digest(m0 / "manifest.json") == manifest["M0_manifest_sha256"], "M0 identity changed")
    payload = (m0 / "result.csv").read_bytes()
    validate_result(payload, test.sample_id)
    require(digest(m0 / "result.csv") == identity["result_sha256"], "M0 CSV changed")
    require(digest(m0 / ZIP_NAME) == identity["zip_sha256"], "M0 ZIP changed")
    model = json.loads((m0 / "model.json").read_text())
    require(payload == csv_bytes(test.sample_id, predict_m0(model, test.sample_id)), "M0 replay changed")
    with zipfile.ZipFile(m0 / ZIP_NAME) as archive:
        require(archive.namelist() == ["result.csv"] and archive.read("result.csv") == payload, "ZIP member mismatch")
    return {"status": "pass", "new_regressor_fits": 0, "validated_q2_models": 20,
            "experiment_regression_fits": {"synthetic": 2, "Q2": 20, "full": 0, "total": 22},
            "metrics_and_promotion": "pass", "M0_release": "unchanged_and_valid",
            "summary_sha256": digest(run / "summary.json"), "ledger_sha256": digest(run / "fit_ledger.jsonl")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = check(Path.cwd(), args.run)
    write_json(args.run / "independent_verification.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
