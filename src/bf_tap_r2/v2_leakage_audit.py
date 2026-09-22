"""V2 overlap, label-proxy, shift and fixed C2 negative-control diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, spearmanr
import yaml
from threadpoolctl import threadpool_limits

from .audit import digest, write_json, duplicates
from .cv import append_event, target_metrics, combined_metrics
from .data import FEATURES, TARGETS
from .metrics import wmape
from .models import SnapshotRegressor
from .v2_release import load_v2


def nearest(query, reference, self_match=False):
    """Chebyshev distance in training-IQR units, excluding the same row."""
    distances, indexes = [], []
    for start in range(0, len(query), 128):
        block = cdist(query[start:start + 128], reference, metric="chebyshev")
        if self_match:
            block[np.arange(len(block)), np.arange(start, start + len(block))] = np.inf
        index = block.argmin(axis=1)
        distances.extend(block[np.arange(len(block)), index].tolist())
        indexes.extend(index.tolist())
    return np.asarray(distances), np.asarray(indexes)


def proxy_screen(frame, features, target):
    y = frame[target].to_numpy(dtype=float)
    result = []
    for feature in features:
        x = frame[feature].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        max_error = float(np.max(np.abs(fitted - y)))
        result.append({"feature": feature, "pearson": float(np.corrcoef(x, y)[0, 1]),
                       "spearman": float(spearmanr(x, y).statistic),
                       "same_row_equal_fraction": float(np.mean(np.isclose(x, y, rtol=0, atol=1e-6))),
                       "affine_max_absolute_residual": max_error,
                       "affine_exact_at_1e_6": max_error <= 1e-6})
    return result


def permute_targets(frame, seed):
    """Jointly permute two targets within training spouts, never validation rows."""
    rng = np.random.default_rng(seed)
    values = frame[list(TARGETS)].to_numpy().copy()
    result = values.copy()
    for spout in sorted(frame.spout_no.unique()):
        indexes = np.flatnonzero(frame.spout_no.to_numpy() == spout)
        result[indexes] = values[rng.permutation(indexes)]
    return result


def run(root, output):
    if not output.resolve().is_relative_to(root.resolve() / "local/runs"):
        raise ValueError("Private output required")
    output.mkdir(parents=True, exist_ok=False)
    try:
        parent = root / "local/runs/round2-v2/comparison-r1"
        manifest = json.loads((parent / "manifest.json").read_text())
        for name, expected in {**manifest["files"], **manifest["sources"]}.items():
            if digest(root / name) != expected:
                raise ValueError(f"Changed comparison input/source: {name}")
        paths = [p for folder in ("复赛_train", "复赛_test") for p in (root / folder).iterdir() if p.is_file()]
        identity = {str(p.relative_to(root)): digest(p) for p in paths}
        write_json(output / "manifest.json", {"inputs": identity,
                   "audit_source_sha256": digest(Path(__file__)),
                   "parent_summary_sha256": digest(parent / "summary.json"),
                   "uv_lock_sha256": digest(root / "uv.lock"),
                   "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "diagnostic_fit_budget": 20, "cv_seed": 42,
                   "permutation_seed_rule": "20260922 + fold; paired targets within training spout",
                   "near_duplicate_thresholds": [0.0, 0.01, 0.05]})
        train = load_v2(root / "复赛_train", "train", 2754)
        test = load_v2(root / "复赛_test", "test", 322)
        template = pd.read_csv(root / "复赛_test/result_template.csv")
        report = {"integrity": {"train_rows": len(train), "test_rows": len(test),
                   "ID_overlap": len(set(train.sample_id) & set(test.sample_id)),
                   "ID_suffix_overlap": len(set(train.sample_id.str.rsplit("_", n=1).str[-1]) & set(test.sample_id.str.rsplit("_", n=1).str[-1])),
                   "test_target_columns": sorted(set(TARGETS) & set(test.columns)),
                   "template_filled_prediction_cells": int(template.iloc[:, 1:].notna().sum().sum()),
                   "missing_train_cells": int(train.isna().sum().sum()), "missing_test_cells": int(test.isna().sum().sum()),
                   "independent_ID_alignment": True}, "duplicates": {}, "label_proxy": {}, "shift": {}}
        # Build labels/features by independent dictionaries, not the loader's join.
        for stage, frame in (("train", train), ("test", test)):
            raw = pd.read_csv(root / f"复赛_{stage}/{stage}_features.csv").set_index("sample_id").to_dict("index")
            rebuilt = np.array([[raw[sid][f] for f in FEATURES] for sid in frame.sample_id])
            np.testing.assert_array_equal(rebuilt, frame[list(FEATURES)].to_numpy())
            report["duplicates"][stage] = duplicates(frame, list(FEATURES))
            report["integrity"][stage + "_constants"] = [f for f in FEATURES if frame[f].nunique() == 1]
            report["integrity"][stage + "_spouts"] = {str(k): int(v) for k, v in frame.spout_no.value_counts().items()}
        x, xt = train[list(FEATURES)].to_numpy(), test[list(FEATURES)].to_numpy()
        scale = np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0)
        scale = np.where(scale > 0, scale, 1.)
        for name, q, r, same in (("train_to_train", x / scale, x / scale, True),
                                 ("test_to_train", xt / scale, x / scale, False),
                                 ("test_to_test", xt / scale, xt / scale, True)):
            distances, indexes = nearest(q, r, same)
            report["duplicates"][name] = {"threshold_row_counts": {str(t): int(np.sum(distances <= t)) for t in (0., .01, .05)},
                "nearest_distance_quantiles": {str(v): float(np.quantile(distances, v)) for v in (0., .01, .5, .99, 1.)}}
            pd.DataFrame({"distance": distances, "nearest_index": indexes}).to_csv(output / f"{name}.csv", index=False, mode="x")
        for target in TARGETS:
            report["label_proxy"][target] = proxy_screen(train, FEATURES, target)
        report["label_proxy"]["target_correlation"] = float(train[list(TARGETS)].corr().iloc[0, 1])
        report["label_proxy"]["target_unique_counts"] = {t: int(train[t].nunique()) for t in TARGETS}
        report["feature_correlations"] = [{"left": a, "right": b, "pearson": float(train[a].corr(train[b]))}
            for i, a in enumerate(FEATURES) for b in FEATURES[i+1:] if abs(train[a].corr(train[b])) >= .99]
        for feature in FEATURES:
            a, b = train[feature], test[feature]
            ks = ks_2samp(a, b)
            report["shift"][feature] = {"KS": float(ks.statistic), "p_value": float(ks.pvalue),
                "bonferroni_significant_21_tests": bool(ks.pvalue < .05 / len(FEATURES)),
                "standardized_mean_difference": float((b.mean() - a.mean()) / a.std()),
                "test_outside_train_range": int(((b < a.min()) | (b > a.max())).sum())}
        # ID/order diagnostics do not make those fields available to the prediction model.
        id_numeric = train.sample_id.str.rsplit("_", n=1).str[-1].map(lambda s: int(s, 16)).to_numpy()
        report["ID_order_correlations"] = {t: {"ID_hex_spearman": float(spearmanr(id_numeric, train[t]).statistic),
                   "file_row_spearman": float(spearmanr(np.arange(len(train)), train[t]).statistic)} for t in TARGETS}
        write_json(output / "data_checks.json", report)
        print(json.dumps({"integrity": report["integrity"], "duplicates": report["duplicates"]}), flush=True)
        assignments = pd.read_csv(parent / "folds-42.csv")
        folds = assignments.set_index("sample_id").loc[train.sample_id, "fold"].to_numpy()
        if set(folds) != set(range(5)) or len(assignments) != len(train):
            raise ValueError("Invalid fold identity")
        config = yaml.safe_load((root / "configs/round2_v0_1/models.yaml").read_text())
        parent_models = {e["tag"]: e["model_sha256"] for e in map(json.loads, (parent / "fit_ledger.jsonl").read_text().splitlines()) if e["kind"] == "cv_complete"}
        diagnostics, baseline_predictions = [], {}
        for target in TARGETS:
            pred = np.full(len(train), np.nan)
            for fold in range(5):
                tag = f"42-C2-{target}-{fold}"
                path = parent / "models" / (tag + ".joblib")
                if digest(path) != parent_models[tag]:
                    raise ValueError("Parent model identity mismatch")
                model = joblib.load(path)
                if model.estimator_.feature_names_ != [*FEATURES, "spout_no"]:
                    raise ValueError("Unexpected actual model input")
                tr, va = train.loc[folds != fold], train.loc[folds == fold]
                assert not set(tr.sample_id) & set(va.sample_id)
                with threadpool_limits(limits=1):
                    vp, tp = model.predict(va), model.predict(tr)
                pred[folds == fold] = vp
                diagnostics.append({"target": target, "fold": fold, "training_wmape": wmape(tr[target], tp),
                    "validation_wmape": wmape(va[target], vp),
                    "feature_importance": dict(zip(model.estimator_.feature_names_, map(float, model.estimator_.feature_importances_)))})
            baseline_predictions[target] = pred
        results = {"original_C2": {t: target_metrics(train, t, baseline_predictions[t], folds) for t in TARGETS}}
        results["original_C2"].update(combined_metrics(results["original_C2"]))
        write_json(output / "original_models.json", diagnostics)
        (output / "models").mkdir()
        fits = 0
        for control in ("permuted_training_labels", "pig_masked"):
            predictions = np.full((len(train), 2), np.nan)
            for fold in range(5):
                tr, va = train.loc[folds != fold].copy(), train.loc[folds == fold].copy()
                y = permute_targets(tr, 20260922 + fold) if control == "permuted_training_labels" else tr[list(TARGETS)].to_numpy()
                if control == "pig_masked":
                    tr["pig"], va["pig"] = 0., 0.
                for j, target in enumerate(TARGETS):
                    fits += 1
                    if fits > 20:
                        raise ValueError("Diagnostic fit budget exceeded")
                    tag = f"{control}-{target}-{fold}"
                    append_event(output / "fit_ledger.jsonl", {"kind": "start", "fit": fits, "tag": tag})
                    tick = time.monotonic()
                    with threadpool_limits(limits=1):
                        model = SnapshotRegressor("C2", config).fit(tr, y[:, j])
                        pred = model.predict(va)
                        path = output / "models" / (tag + ".joblib")
                        joblib.dump(model, path)
                        np.testing.assert_allclose(pred, joblib.load(path).predict(va.iloc[::-1])[::-1], rtol=1e-12, atol=1e-10)
                    predictions[folds == fold, j] = pred
                    append_event(output / "fit_ledger.jsonl", {"kind": "complete", "tag": tag, "model_sha256": digest(path), "seconds": time.monotonic()-tick})
                    print(f"Audit control {fits}/20 {tag}", flush=True)
            results[control] = {t: target_metrics(train, t, predictions[:, j], folds) for j, t in enumerate(TARGETS)}
            results[control].update(combined_metrics(results[control]))
            out = train[["sample_id", *TARGETS]].copy()
            out["fold"] = folds
            out[["pred_iron", "pred_time"]] = predictions
            out.to_csv(output / (control + ".csv"), index=False, mode="x")
        if identity != {str(p.relative_to(root)): digest(p) for p in paths}:
            raise ValueError("Inputs changed during audit")
        write_json(output / "controls.json", {"metrics": results, "fits": fits,
                   "scope": "diagnostics only, seed42 fivefold; unchanged C2 parameters; no candidate promotion",
                   "permutation_limit": "one paired within-spout permutation per training fold; diagnostic, not a formal p-value",
                   "release_changes": 0, "platform_uploads": 0})
        write_json(output / "COMPLETE.json", {"G0": "PASS", "control_fits": fits, "data_report_sha256": digest(output / "data_checks.json"),
                   "control_report_sha256": digest(output / "controls.json")})
        print(json.dumps(results), flush=True)
    except Exception as exc:
        write_json(output / "FAILED.json", {"type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(Path.cwd(), args.output)


if __name__ == "__main__":
    main()
