"""Read-only cold verification of saved folds, model outputs and reported WMAPE."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .audit import digest, write_json
from .cv import baseline_predictions, target_metrics, verify_audit
from .data import FEATURES, TARGETS, load_config, load_snapshot


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root, run_dir, audit_dir):
    verify_audit(root, audit_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    identity = json.loads((run_dir / "run_manifest.json").read_text())
    require(summary["status"] == "complete", "Run incomplete")
    for name, sha in identity["source_sha256"].items():
        require(digest(root / name) == sha, f"Training source changed: {name}")
    for name, sha in identity["config_sha256"].items():
        require(digest(run_dir / "configs" / name) == sha, f"Frozen config changed: {name}")
    require(digest(root / "uv.lock") == identity["uv_lock_sha256"], "Dependency lock changed")
    require(digest(run_dir / "data_manifest.json") == identity["data_manifest_sha256"], "Data manifest changed")
    fold_identity = json.loads((run_dir / "fold_identity.json").read_text())
    require(digest(run_dir / "fold_assignments.csv") == fold_identity["sha256"], "Fold identity changed")
    frame, _ = load_snapshot(root, load_config(run_dir / "configs/data.yaml"), "train")
    test, _ = load_snapshot(root, load_config(run_dir / "configs/data.yaml"), "test")
    frame = frame.sort_values("sample_id").reset_index(drop=True)
    folds_frame = pd.read_csv(run_dir / "fold_assignments.csv")
    events = [json.loads(line) for line in (run_dir / "fit_ledger.jsonl").read_text().splitlines()]
    starts = [e for e in events if e["kind"] == "regressor_fit_start"]
    ends = [e for e in events if e["kind"] == "regressor_fit_complete"]
    require(len(starts) == len(ends) == summary["regressor_fits"] == 70, "Fit count mismatch")
    require([e["fit_number"] for e in starts] == list(range(1, 71)), "Fit starts duplicated")
    require([e["fit_number"] for e in ends] == list(range(1, 71)), "Fit completions duplicated")
    maximum_error = 0.
    for e in ends:
        seed, route, target, fold = e["seed"], e["route"], e["target"], e["fold"]
        assignments = folds_frame.loc[folds_frame.seed == seed].set_index("sample_id")
        require(set(assignments.index) == set(frame.sample_id) and assignments.index.is_unique, "Invalid fold IDs")
        require(assignments.groupby("group_id").fold.nunique().max() == 1, "Dependent group crosses folds")
        folds = assignments.loc[frame.sample_id, "fold"].to_numpy()
        valid = frame.loc[folds == fold]
        train = frame.loc[folds != fold]
        tag = f"{seed}-{route}-{target}-fold{fold}"
        model_path = run_dir / "models" / f"{tag}.joblib"
        require(digest(model_path) == e["model_sha256"], "Model digest mismatch")
        model = joblib.load(model_path)
        if route in ("L1", "L2", "S1"):
            pre = model.estimator_.named_steps["preprocess"]
            require(set(pre.named_transformers_["spout"].categories_[0]) == set(train.spout_no), "Category fitting scope")
            if route != "S1":
                np.testing.assert_allclose(pre.named_transformers_["numeric"].mean_, train[list(FEATURES)].mean(), rtol=1e-12)
            else:
                numeric = pre.named_transformers_["numeric"]
                for i, column in enumerate(FEATURES):
                    knots = numeric.bsplines_[i].t[numeric.degree:-numeric.degree]
                    np.testing.assert_allclose(knots, np.linspace(train[column].min(), train[column].max(), 5), rtol=1e-12)
        else:
            require(model.estimator_.tree_count_ == 1500, "CatBoost iterations changed")
        require(model.target_scale_ == (float(train[target].median()) if route == "L2" else 1.), "Target scale scope")
        saved = pd.read_csv(run_dir / "oof" / f"{tag}.csv").set_index("sample_id")
        require(saved.index.is_unique and set(saved.index) == set(valid.sample_id), "Fold OOF coverage")
        with threadpool_limits(limits=1):
            cold = model.predict(valid)
            reverse = model.predict(valid.iloc[::-1])[::-1]
            test_prediction = model.predict(test)
            test_reversed = model.predict(test.iloc[::-1])[::-1]
        np.testing.assert_allclose(cold, reverse, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(test_prediction, test_reversed, rtol=1e-12, atol=1e-10)
        recorded = saved.loc[valid.sample_id, f"pred_{target}"].to_numpy()
        np.testing.assert_allclose(cold, recorded, rtol=1e-12, atol=1e-10)
        maximum_error = max(maximum_error, float(np.max(np.abs(cold-recorded))))
        full = pd.read_csv(run_dir / "oof" / f"{seed}-{route}-{target}.csv").set_index("sample_id")
        require(full.index.is_unique and set(full.index) == set(frame.sample_id), "Full OOF coverage")
        np.testing.assert_allclose(full.loc[valid.sample_id, f"pred_{target}"], recorded, rtol=1e-12)
    for name, metrics in summary["metrics"].items():
        seed_str, route = name.split("-", 1)
        folds = folds_frame.loc[folds_frame.seed == int(seed_str)].set_index("sample_id").loc[frame.sample_id, "fold"].to_numpy()
        for target in TARGETS:
            if target not in metrics:
                continue
            if route.startswith("median_"):
                predictions = baseline_predictions(frame, target, folds, route == "median_spout")
                saved = pd.read_csv(run_dir / "oof" / f"{name}.csv").set_index("sample_id")
                np.testing.assert_allclose(predictions, saved.loc[frame.sample_id, f"pred_{target}"], rtol=1e-12)
            else:
                saved = pd.read_csv(run_dir / "oof" / f"{name}-{target}.csv").set_index("sample_id")
                predictions = saved.loc[frame.sample_id, f"pred_{target}"].to_numpy()
            actual = target_metrics(frame, target, predictions, folds)
            require(abs(actual["wmape"] - metrics[target]["wmape"]) < 1e-12, "Pooled metric mismatch")
            for group in ("by_fold", "by_spout"):
                for key, value in actual[group].items():
                    require(abs(value - metrics[target][group][key]) < 1e-12, "Grouped metric mismatch")
        if all(t in metrics for t in TARGETS):
            require(abs(metrics["J"] - sum(metrics[t]["wmape"] for t in TARGETS)/2) < 1e-12, "J mismatch")
    return {"status": "pass", "models_checked": len(ends), "new_regressor_fits": 0,
            "max_cold_prediction_error": maximum_error, "pooled_and_grouped_metrics": "pass",
            "fold_preprocessing_scope": "pass", "shuffled_validation_predictions": "pass",
            "shuffled_test_predictions": "pass", "test_rows": len(test),
            "summary_sha256": digest(run_dir / "summary.json"), "fit_ledger_sha256": digest(run_dir / "fit_ledger.jsonl")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=Path("local/runs/round2-v0.1/p0-audit-r2"))
    parser.add_argument("--report-name", default="cold_verification.json")
    args = parser.parse_args()
    if Path(args.report_name).name != args.report_name or not args.report_name.endswith(".json"):
        raise ValueError("Report name must be a JSON basename")
    report = verify(Path.cwd(), args.run, args.audit)
    write_json(args.run / args.report_name, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
