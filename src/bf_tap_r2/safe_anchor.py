"""Bounded S25/Q2 comparison on frozen v0.1 folds; no platform uploads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .anchor_diagnostics import pairing_check, saved_model_diagnostics, stratified_signal
from .audit import digest, write_json
from .cv import append_event, baseline_predictions, combined_metrics, target_metrics
from .data import FEATURES, TARGETS, load_config, load_snapshot
from .verify import verify
from .weak_models import Q2Regressor, assess_candidate, s25_predictions


def load_gate(root, spec):
    reports = {}
    for target in TARGETS:
        path = root / spec["known_signal_evidence"] / f"{target}.json"
        report = json.loads(path.read_text())
        if (not report["pass"] or report["ratio"] > .1 or report["regressor_fits"] != 1
                or report["test_source_sha256"] != digest(root / "tests/test_round2_known_signal.py")):
            raise ValueError("Known-signal engineering gate failed")
        reports[target] = {**report, "report_sha256": digest(path)}
    return reports


def run(root, output):
    path = root / "configs/round2_v0_2/safe_anchor.yaml"
    spec = yaml.safe_load(path.read_text())
    gate = load_gate(root, spec)
    m0 = root / spec["m0_release"]
    if json.loads((m0 / "verification.json").read_text())["status"] != "pass":
        raise ValueError("M0 must be prepared first")
    if not output.resolve().is_relative_to(root.resolve() / "local/runs/round2-v0.2"):
        raise ValueError("Private run directory required")
    output.mkdir(parents=True, exist_ok=False)
    started_fits = 0
    try:
        for part in ("models", "oof", "metrics"):
            (output / part).mkdir()
        parent = root / spec["parent"]
        write_json(output / "parent_verification.json", verify(root, parent, root / "local/runs/round2-v0.1/p0-audit-r2"))
        write_json(output / "known_signal_gate.json", gate)
        cfg = load_config(root / "configs/round2_v0_1/data.yaml")
        frame, _ = load_snapshot(root, cfg, "train")
        frame = frame.sort_values("sample_id").reset_index(drop=True)
        test, _ = load_snapshot(root, cfg, "test")
        write_json(output / "pairing_verification.json", pairing_check(root, cfg, frame))
        with (output / "config.yaml").open("xb") as handle:
            handle.write(path.read_bytes())
        assignments_path = parent / "fold_assignments.csv"
        assignments = pd.read_csv(assignments_path)
        with (output / "fold_assignments.csv").open("xb") as handle:
            handle.write(assignments_path.read_bytes())
        write_json(output / "manifest.json", {
            "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "source_sha256": {str(p.relative_to(root)): digest(p) for p in sorted((root / "src/bf_tap_r2").glob("*.py"))},
            "uv_lock_sha256": digest(root / "uv.lock"), "config_sha256": digest(path),
            "parent_summary_sha256": digest(parent / "summary.json"), "folds_sha256": digest(assignments_path),
            "parent_oof_sha256": {p.name: digest(p) for p in sorted((parent / "oof").glob("*.csv"))},
            "M0_manifest_sha256": digest(m0 / "manifest.json"), "new_regression_budget": spec["regression_fit_budget"]})
        write_json(output / "stratified_signal.json", stratified_signal(frame, spec["permutation"]))
        write_json(output / "saved_model_diagnostics.json", saved_model_diagnostics(parent, frame, assignments))
        print("Engineering gates and stratified diagnostics PASS; starting fixed S25/Q2", flush=True)
        results = {"M0": {}, "S25": {}, "Q2": {}}
        ledger = output / "fit_ledger.jsonl"
        for target in TARGETS:
            append_event(ledger, {"kind": "known_signal_complete", "target": target, "regressor_fits": 1,
                                 "report_sha256": gate[target]["report_sha256"]})
        cold_checks = []
        for seed in spec["fold_seeds"]:
            folds = assignments.loc[assignments.seed == seed].set_index("sample_id").loc[frame.sample_id, "fold"].to_numpy()
            for name in results:
                results[name][seed] = {}
            for target in TARGETS:
                median_oof = baseline_predictions(frame, target, folds, False)
                source = spec["S25"]["sources"][target]
                old = pd.read_csv(parent / "oof" / f"{seed}-{source}-{target}.csv").set_index("sample_id").loc[frame.sample_id]
                np.testing.assert_array_equal(old["fold"], folds)
                np.testing.assert_allclose(old[target], frame[target], rtol=1e-12)
                old_base = pd.read_csv(parent / "oof" / f"{seed}-median_global.csv").set_index("sample_id").loc[frame.sample_id]
                np.testing.assert_allclose(old_base[f"pred_{target}"], median_oof, rtol=1e-12)
                s25 = s25_predictions(median_oof, old[f"pred_{target}"].to_numpy())
                q2 = np.full(len(frame), np.nan)
                for fold in range(5):
                    if started_fits >= spec["regression_fit_budget"]["q2"]:
                        raise ValueError("Q2 fit budget exhausted")
                    tag = f"{seed}-Q2-{target}-fold{fold}"
                    training, validation = frame.loc[folds != fold], frame.loc[folds == fold]
                    event = {"tag": tag, "target": target, "seed": seed, "fold": fold}
                    started_fits += 1
                    append_event(ledger, {**event, "kind": "q2_fit_start", "fit_number": started_fits,
                                         "train_rows": len(training), "validation_rows": len(validation)})
                    tick = time.monotonic()
                    with threadpool_limits(limits=1):
                        model = Q2Regressor(spec["Q2"]).fit(training, training[target])
                        prediction = model.predict(validation)
                    q2[folds == fold] = prediction
                    model_path = output / "models" / f"{tag}.joblib"
                    joblib.dump(model, model_path)
                    fold_oof = validation[["sample_id", "spout_no", target]].copy()
                    fold_oof[f"pred_{target}"] = prediction
                    fold_oof.to_csv(output / "oof" / f"{tag}.csv", index=False, mode="x")
                    elapsed = time.monotonic() - tick
                    append_event(ledger, {**event, "kind": "q2_fit_complete", "fit_number": started_fits,
                                         "seconds": elapsed, "model_sha256": digest(model_path),
                                         "nonzero_coefficients": int(np.count_nonzero(np.abs(model.estimator_.named_steps["regression"].coef_) > 1e-10))})
                    # Check the serialized estimator and preprocessing scope without refitting.
                    loaded = joblib.load(model_path)
                    numeric = loaded.estimator_.named_steps["preprocess"].named_transformers_["numeric"]
                    np.testing.assert_allclose(numeric.named_steps["scale_input"].mean_, training[list(FEATURES)].mean(), rtol=1e-12)
                    expanded = numeric.named_steps["poly"].transform(numeric.named_steps["scale_input"].transform(training[list(FEATURES)]))
                    np.testing.assert_allclose(numeric.named_steps["scale_expanded"].mean_, expanded.mean(axis=0), rtol=1e-12, atol=1e-12)
                    if expanded.shape[1] != 252 or loaded.target_scale_ != float(training[target].median()):
                        raise ValueError("Q2 expansion or target scaling contract mismatch")
                    with threadpool_limits(limits=1):
                        cold = loaded.predict(validation)
                        reverse = loaded.predict(validation.iloc[::-1])[::-1]
                        test_forward = loaded.predict(test)
                        test_reverse = loaded.predict(test.iloc[::-1])[::-1]
                    np.testing.assert_allclose(prediction, cold, rtol=1e-12, atol=1e-10)
                    np.testing.assert_allclose(cold, reverse, rtol=1e-12, atol=1e-10)
                    np.testing.assert_allclose(test_forward, test_reverse, rtol=1e-12, atol=1e-10)
                    cold_checks.append({"tag": tag, "status": "pass", "expanded_features": 252})
                    print(json.dumps({**event, "completed_q2_fits": started_fits,
                                      "wmape": wmape_value(validation[target], prediction), "seconds": elapsed}), flush=True)
                for name, prediction in (("M0", median_oof), ("S25", s25), ("Q2", q2)):
                    results[name][seed][target] = target_metrics(frame, target, prediction, folds)
                    out = frame[["sample_id", "spout_no", target]].copy()
                    out["fold"] = folds
                    out[f"pred_{target}"] = prediction
                    out.to_csv(output / "oof" / f"{seed}-{name}-{target}.csv", index=False, mode="x")
            for name in results:
                results[name][seed].update(combined_metrics(results[name][seed]))
                results[name][seed]["reference_score_not_platform"] = max(0., 100 * (1 - results[name][seed]["J"]))
                write_json(output / "metrics" / f"{seed}-{name}.json", results[name][seed])
        promotion, chosen = {}, {}
        for target in TARGETS:
            promotion[target] = {}
            for name in ("S25", "Q2"):
                promotion[target][name] = assess_candidate(
                    {s: results[name][s][target] for s in spec["fold_seeds"]},
                    {s: results["M0"][s][target] for s in spec["fold_seeds"]}, spec["promotion"])
            eligible = [n for n in ("S25", "Q2") if promotion[target][n]["eligible"]]
            if len(eligible) == 2 and abs(promotion[target]["S25"]["mean_wmape"] - promotion[target]["Q2"]["mean_wmape"]) <= spec["promotion"]["near_tie_mean_wmape"]:
                chosen[target] = "S25"
            else:
                chosen[target] = min(eligible, key=lambda n: promotion[target][n]["mean_wmape"]) if eligible else "M0"
        write_json(output / "cold_verification.json", {"status": "pass", "models": cold_checks,
                   "new_regression_fits": 0, "test_order_invariance": "pass", "fold_local_preprocessing": "pass"})
        write_json(output / "summary.json", {"status": "complete", "metrics": results, "promotion": promotion,
                   "selected_by_target": chosen, "known_signal_fits": 2, "q2_fits": started_fits,
                   "full_fits": 0, "total_new_fits": 2+started_fits, "remaining_full_fit_budget": 2,
                   "s25_new_fits": 0, "q2_preprocessor_pipeline_fits": started_fits, "platform_uploads": 0})
        print(json.dumps({"promotion": promotion, "selected": chosen}, ensure_ascii=False), flush=True)
    except Exception as exc:
        write_json(output / "FAILED.json", {"type": type(exc).__name__, "error": str(exc), "q2_fit_starts": started_fits})
        raise


def wmape_value(y, prediction):
    from .metrics import wmape
    return wmape(y, prediction)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(Path.cwd(), args.output)


if __name__ == "__main__":
    main()
