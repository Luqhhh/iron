"""Bounded v0.3 protocol. Exclusive artifacts, no old model refits, no uploads."""
import argparse
import json
import importlib.metadata
import subprocess
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from .audit import digest, write_json
from .cv import append_event, target_metrics, combined_metrics, baseline_predictions
from .data import TARGETS, load_config, load_snapshot, resolve_file
from .kernel_regressor import KernelRegressor
from .marginal_estimators import hd_median
from .metrics import wmape
from .weak_models import assess_candidate


def sensitivity(frame, target, predictions, anchors, spec):
    rng = np.random.default_rng(spec["seed"])
    groups = [np.flatnonzero(frame.spout_no.to_numpy() == s) for s in sorted(frame.spout_no.unique())]
    y = frame[target].to_numpy()
    rows = []
    for _ in range(spec["repetitions"]):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        delta = {str(s): wmape(y[ix], predictions[s][ix]) - wmape(y[ix], anchors[s][ix]) for s in predictions}
        delta["mean"] = float(np.mean(list(delta.values())))
        rows.append(delta)
    distribution = pd.DataFrame(rows)
    return distribution, {k: {"q025": float(distribution[k].quantile(.025)),
                              "q975": float(distribution[k].quantile(.975))} for k in distribution}


def run(root, output):
    root = root.resolve()
    if not output.resolve().is_relative_to(root / "local/runs/round2-v0.3"):
        raise ValueError("Private fresh output required")
    output.mkdir(parents=True, exist_ok=False)
    fits = 0
    statistics = 0
    try:
        cfg_path = root / "configs/round2_v0_3/experiment.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        engineering = root / "local/runs/round2-v0.3/engineering-r1"
        for target in TARGETS:
            evidence = json.loads((engineering / f"{target}.json").read_text())
            if evidence["fits"] != 1 or evidence["status"] != "PASS":
                raise ValueError("Two engineering checks must pass before CV")
        old = root / "local/runs/round2-v0.1/p2-cv-r1"
        m0 = root / "local/runs/round2-v0.2/m0-release-r1"
        manifest = json.loads((m0 / "manifest.json").read_text())
        actual = {name: digest(resolve_file(root, name)) for name in manifest["data_sha256"]}
        if actual != manifest["data_sha256"]:
            raise ValueError("Data version changed; stop and re-audit")
        zip_sha = digest(m0 / "Luqhhh_bf_tap_predict_round2.zip")
        if zip_sha != "5adc32eb9918c2e0235b193cbc00c20220b11f5263225589c02d7c7fea5f4b61":
            raise ValueError("M0 identity mismatch")
        fold_path = old / "fold_assignments.csv"
        if digest(fold_path) != json.loads((old / "fold_identity.json").read_text())["sha256"]:
            raise ValueError("Fold identity mismatch")
        write_json(output / "manifest.json", {
            "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "data_sha256": actual, "m0_zip_sha256": zip_sha, "fold_sha256": digest(fold_path),
            "config_sha256": digest(cfg_path), "uv_lock_sha256": digest(root / "uv.lock"),
            "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib")},
            "source_sha256": {p.name: digest(p) for p in (root / "src/bf_tap_r2").glob("*.py")},
            "platform_uploads": 0, "platform_score": None})
        (output / "experiment.yaml").write_bytes(cfg_path.read_bytes())
        frame, _ = load_snapshot(root, load_config(root / "configs/round2_v0_1/data.yaml"), "train")
        frame = frame.sort_values("sample_id").reset_index(drop=True)
        assignments = pd.read_csv(fold_path)
        for name in ("models", "oof", "metrics", "sensitivity"):
            (output / name).mkdir()
        all_scores = {r: {} for r in ("M0", "M1_HD_MEDIAN", "K1_RBF_L1")}
        all_predictions = {r: {t: {} for t in TARGETS} for r in all_scores}
        fold_records = []
        for seed in cfg["seeds"]:
            assignment = assignments.loc[assignments.seed == seed].set_index("sample_id")
            if not assignment.index.is_unique or set(assignment.index) != set(frame.sample_id):
                raise ValueError("Invalid fold IDs")
            folds = assignment.loc[frame.sample_id, "fold"].to_numpy()
            if set(folds) != set(range(5)):
                raise ValueError("Expected frozen five folds")
            for r in all_scores:
                all_scores[r][seed] = {}
            for target in TARGETS:
                anchor = baseline_predictions(frame, target, folds, False)
                all_predictions["M0"][target][seed] = anchor
                all_scores["M0"][seed][target] = target_metrics(frame, target, anchor, folds)
                for route in ("M1_HD_MEDIAN", "K1_RBF_L1"):
                    pred = np.full(len(frame), np.nan)
                    for fold in range(5):
                        train, valid = frame.loc[folds != fold], frame.loc[folds == fold]
                        median = float(train[target].median())
                        record = {"seed": seed, "target": target, "route": route, "fold": fold,
                                  "ordinary_median": median}
                        if route == "M1_HD_MEDIAN":
                            estimate = hd_median(train[target])
                            statistics += 1
                            raw = np.full(len(valid), estimate)
                            train_pred = np.full(len(train), estimate)
                            record.update(hd_median=estimate, median_delta=estimate-median)
                        else:
                            if fits >= cfg["budget"]["cv_fits"]:
                                raise RuntimeError("CV budget exhausted")
                            fits += 1
                            append_event(output / "fit_ledger.jsonl", {**record, "event": "start", "fit_number": fits})
                            model = KernelRegressor(cfg["model"])
                            try:
                                model.fit(train, train[target])
                            except Exception:
                                write_json(output / "failed_fit.json", {**record, "diagnostics": getattr(model, "fit_report_", {})})
                                raise
                            raw = model.predict_raw(valid)
                            train_pred = model.predict(train)
                            model_path = output / "models" / f"{seed}-{target}-{fold}.joblib"
                            joblib.dump(model, model_path)
                            np.testing.assert_array_equal(joblib.load(model_path).predict(valid), model.predict(valid))
                            record.update(model.fit_report_)
                            record["model_sha256"] = digest(model_path)
                            append_event(output / "fit_ledger.jsonl", {**record, "event": "complete", "fit_number": fits})
                        prediction = np.maximum(0, raw) if route == "K1_RBF_L1" else raw
                        pred[folds == fold] = prediction
                        record.update(train_wmape=wmape(train[target], train_pred),
                                      valid_wmape=wmape(valid[target], prediction),
                                      baseline_valid_wmape=wmape(valid[target], anchor[folds == fold]),
                                      raw_min=float(raw.min()), clipped_count=int((raw < 0).sum()))
                        record["valid_wmape_delta"] = record["valid_wmape"]-record["baseline_valid_wmape"]
                        fold_records.append(record)
                        write_json(output / "metrics" / f"{seed}-{route}-{target}-{fold}.json", record)
                        print(json.dumps(record), flush=True)
                    all_predictions[route][target][seed] = pred
                    all_scores[route][seed][target] = target_metrics(frame, target, pred, folds)
                    oof = frame[["sample_id", "spout_no", target]].copy()
                    oof["fold"] = folds
                    oof[f"pred_{target}"] = pred
                    oof.to_csv(output / "oof" / f"{seed}-{route}-{target}.csv", index=False, mode="x")
            for route in all_scores:
                all_scores[route][seed].update(combined_metrics(all_scores[route][seed]))
        decisions, selected, bootstrap = {}, {}, {}
        for target in TARGETS:
            decisions[target] = {}
            for route in ("M1_HD_MEDIAN", "K1_RBF_L1"):
                decision = assess_candidate({s: all_scores[route][s][target] for s in cfg["seeds"]},
                                            {s: all_scores["M0"][s][target] for s in cfg["seeds"]}, cfg["selection"])
                decisions[target][route] = decision
                if decision["eligible"]:
                    distribution, intervals = sensitivity(frame, target, all_predictions[route][target],
                                                          all_predictions["M0"][target], cfg["bootstrap"])
                    distribution.to_csv(output / "sensitivity" / f"{route}-{target}.csv", index=False, mode="x")
                    bootstrap[f"{route}-{target}"] = intervals
            eligible = [r for r, d in decisions[target].items() if d["eligible"]]
            selected[target] = min(eligible, key=lambda r: decisions[target][r]["mean_wmape"]) if eligible else "M0"
            if len(eligible) == 2 and abs(decisions[target][eligible[0]]["mean_wmape"] - decisions[target][eligible[1]]["mean_wmape"]) <= cfg["tie_band"]:
                selected[target] = "M1_HD_MEDIAN"
        summary = {"G0": "PASS", "metrics": all_scores, "decisions": decisions, "selected": selected,
                   "bootstrap_intervals": bootstrap, "cv_fits": fits, "m1_fold_estimates": statistics,
                   "engineering_fits": 2, "total_regression_fits": fits + 2,
                   "full_fits": 0, "platform_uploads": 0,
                   "validation_scope": "reused development OOF; bootstrap conditional on frozen predictions"}
        write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(output / "FAILED.json", {"G0": "FAIL", "error": str(exc), "started_cv_fits": fits, "m1_estimates": statistics})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(Path.cwd(), args.output)
