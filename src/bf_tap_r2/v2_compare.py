"""Fixed legacy routes retrained on V2 with common folds and cold verification."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .audit import digest, write_json
from .cv import append_event, baseline_predictions, combined_metrics, target_metrics
from .data import TARGETS, SUBMISSION_COLUMNS
from .kernel_regressor import KernelRegressor
from .marginal_estimators import hd_median
from .models import SnapshotRegressor
from .splits import make_folds
from .submission import package, validate_result, deny_training_reads
from .v2_release import load_v2
from .weak_models import Q2Regressor, assess_candidate, s25_predictions


def configuration(root):
    spec = yaml.safe_load((root / "configs/round2_v2/comparison.yaml").read_text())
    return spec, {k: yaml.safe_load((root / spec[k]).read_text())
                  for k in ("legacy_models", "legacy_anchor", "legacy_kernel")}


def estimator(route, configs):
    if route == "Q2":
        return Q2Regressor(configs["legacy_anchor"]["Q2"])
    if route == "K1":
        return KernelRegressor(configs["legacy_kernel"]["model"])
    return SnapshotRegressor(route, configs["legacy_models"])


def choose(metrics, spec):
    promotion, selected = {}, {}
    for target in TARGETS:
        anchors = {s: metrics["M0"][str(s)][target] for s in spec["seeds"]}
        promotion[target] = {}
        for route in spec["routes"]:
            if route != "M0":
                promotion[target][route] = assess_candidate(
                    {s: metrics[route][str(s)][target] for s in spec["seeds"]}, anchors, spec["promotion"])
        eligible = [r for r in promotion[target] if promotion[target][r]["eligible"]]
        if eligible:
            best = min(promotion[target][r]["mean_wmape"] for r in eligible)
            selected[target] = next(r for r in spec["tie_order"] if r in eligible and
                                    promotion[target][r]["mean_wmape"] <= best + spec["tie_tolerance"])
        else:
            selected[target] = "M0"
    return promotion, selected


def predict_saved(record, folder, test):
    route = record["route"]
    if route == "MS":
        return test.spout_no.map({int(k): v for k, v in record["spout_medians"].items()}).fillna(record["median"]).to_numpy()
    if route in ("M0", "M1"):
        return np.full(len(test), record["value"])
    if digest(folder / record["model"]) != record["model_sha256"]:
        raise ValueError("Release model digest mismatch")
    model = joblib.load(folder / record["model"])
    pred = model.predict(test)
    return s25_predictions(np.full(len(test), record["median"]), pred) if route == "S25" else pred


def serialize(ids, values):
    ids = list(ids)
    if not all(re.fullmatch(r"R2S2_TEST_[0-9A-F]{12}", s) for s in ids):
        raise ValueError("V2 test IDs required")
    values = np.asarray(values, dtype=float)
    if values.shape != (len(ids), 2):
        raise ValueError("Prediction shape mismatch")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(SUBMISSION_COLUMNS)
    writer.writerows((sid, *(format(x, ".17g") for x in pair)) for sid, pair in zip(ids, values))
    payload = stream.getvalue().encode()
    validate_result(payload, ids)
    return payload


def run(root, output):
    if not output.resolve().is_relative_to(root.resolve() / "local/runs"):
        raise ValueError("Private output required")
    output.mkdir(parents=True, exist_ok=False)
    try:
        spec, configs = configuration(root)
        paths = [root / f"复赛_{s}/{s}_{k}.csv" for s in ("train", "test") for k in ("samples", "features")]
        paths += [root / "复赛_test/result_template.csv", root / "configs/round2_v2/comparison.yaml"]
        paths += [root / spec[k] for k in configs]
        manifest = {str(p.relative_to(root)): digest(p) for p in paths}
        write_json(output / "manifest.json", {"files": manifest,
                   "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "sources": {str(p.relative_to(root)): digest(p) for p in (root / "src/bf_tap_r2").glob("*.py")},
                   "uv_lock_sha256": digest(root / "uv.lock")})
        write_json(output / "configuration.json", {"spec": spec, "configs": configs})
        for folder in ("models", "oof", "release"):
            (output / folder).mkdir()
        train, test = load_v2(root / "复赛_train", "train", 2754), load_v2(root / "复赛_test", "test", 322)
        results = {r: {} for r in spec["routes"]}
        count = 0
        for seed in spec["seeds"]:
            assignments = make_folds(train, seed, spec["n_splits"])
            assignments.to_csv(output / f"folds-{seed}.csv", index=False, mode="x")
            folds = assignments.set_index("sample_id").loc[train.sample_id, "fold"].to_numpy()
            for route in results:
                results[route][str(seed)] = {}
            for target in TARGETS:
                predictions = {"M0": baseline_predictions(train, target, folds, False),
                               "MS": baseline_predictions(train, target, folds, True), "M1": np.full(len(train), np.nan)}
                for fold in range(spec["n_splits"]):
                    predictions["M1"][folds == fold] = hd_median(train.loc[folds != fold, target])
                for route in spec["regressors"]:
                    pred = np.full(len(train), np.nan)
                    for fold in range(spec["n_splits"]):
                        tag = f"{seed}-{route}-{target}-{fold}"
                        training, validation = train.loc[folds != fold], train.loc[folds == fold]
                        if count >= spec["budget"]["cv_regressor_fits"]:
                            raise ValueError("CV budget exceeded")
                        count += 1
                        append_event(output / "fit_ledger.jsonl", {"kind": "cv_start", "tag": tag, "fit": count})
                        tick = time.monotonic()
                        with threadpool_limits(limits=1):
                            model = estimator(route, configs).fit(training, training[target])
                            pred[folds == fold] = model.predict(validation)
                        path = output / "models" / f"{tag}.joblib"
                        joblib.dump(model, path)
                        append_event(output / "fit_ledger.jsonl", {"kind": "cv_complete", "tag": tag,
                                     "model_sha256": digest(path), "seconds": time.monotonic() - tick,
                                     "kernel_report": getattr(model, "fit_report_", None)})
                        print(f"CV {count}/140 {tag}", flush=True)
                    predictions[route] = pred
                source = configs["legacy_anchor"]["S25"]["sources"][target]
                predictions["S25"] = s25_predictions(predictions["M0"], predictions[source])
                oof = train[["sample_id", "spout_no", target]].copy()
                oof["fold"] = folds
                for route in spec["routes"]:
                    oof[route] = predictions[route]
                    results[route][str(seed)][target] = target_metrics(train, target, predictions[route], folds)
                oof.to_csv(output / "oof" / f"{seed}-{target}.csv", index=False, mode="x")
            for route in results:
                results[route][str(seed)].update(combined_metrics(results[route][str(seed)]))
            write_json(output / f"metrics-{seed}.json", {r: results[r][str(seed)] for r in results})
        promotion, selected = choose(results, spec)
        release, full_count = {}, 0
        for target, route in selected.items():
            median = float(train[target].median())
            record = {"route": route, "median": median}
            if route in ("M0", "M1"):
                record["value"] = median if route == "M0" else hd_median(train[target])
            elif route == "MS":
                record["spout_medians"] = {str(k): float(v) for k, v in train.groupby("spout_no")[target].median().items()}
            else:
                source = configs["legacy_anchor"]["S25"]["sources"][target] if route == "S25" else route
                full_count += 1
                if full_count > spec["budget"]["full_regressor_fits"]:
                    raise ValueError("Full fit budget exceeded")
                append_event(output / "fit_ledger.jsonl", {"kind": "full_start", "target": target, "route": source})
                with threadpool_limits(limits=1):
                    model = estimator(source, configs).fit(train, train[target])
                record["model"] = f"{target}.joblib"
                joblib.dump(model, output / "release" / record["model"])
                record["model_sha256"] = digest(output / "release" / record["model"])
                append_event(output / "fit_ledger.jsonl", {"kind": "full_complete", "target": target, "route": source})
            release[target] = record
        write_json(output / "release/models.json", release)
        with threadpool_limits(limits=1):
            values = np.column_stack([predict_saved(release[t], output / "release", test) for t in TARGETS])
        package(output / "release", serialize(test.sample_id, values), test.sample_id)
        write_json(output / "summary.json", {"metrics": results, "promotion": promotion, "selected": selected,
                   "cv_fits": count, "full_fits": full_count, "platform_uploads": 0})
        subprocess.run([sys.executable, "-m", "bf_tap_r2.v2_compare", "--output", str(output), "--verify"], check=True, cwd=root)
        subprocess.run([sys.executable, "-m", "bf_tap_r2.v2_compare", "--output", str(output), "--infer"], check=True, cwd=root)
        if (output / "release/cold.csv").read_bytes() != (output / "release/result.csv").read_bytes():
            raise ValueError("Label-free cold inference differs")
        write_json(output / "COMPLETE.json", {"G0": "PASS", "selected": selected,
                   "cold_inference_identical": True, "zip_sha256": digest(output / "release/Luqhhh_bf_tap_predict_round2.zip")})
        print((output / "COMPLETE.json").read_text(), flush=True)
    except Exception as exc:
        write_json(output / "FAILED.json", {"type": type(exc).__name__, "error": str(exc)})
        raise


def verify(root, output):
    manifest = json.loads((output / "manifest.json").read_text())
    for name, sha in {**manifest["files"], **manifest["sources"]}.items():
        if digest(root / name) != sha:
            raise ValueError(f"Identity changed: {name}")
    train = load_v2(root / "复赛_train", "train", 2754)
    test = load_v2(root / "复赛_test", "test", 322)
    config = json.loads((output / "configuration.json").read_text())
    spec, configs = config["spec"], config["configs"]
    summary = json.loads((output / "summary.json").read_text())
    events = [json.loads(line) for line in (output / "fit_ledger.jsonl").read_text().splitlines()]
    completed = [e for e in events if e["kind"] == "cv_complete"]
    if len(completed) != spec["budget"]["cv_regressor_fits"] or len({e["tag"] for e in completed}) != len(completed):
        raise ValueError("Incomplete or duplicated fit ledger")
    for event in completed:
        if digest(output / "models" / (event["tag"] + ".joblib")) != event["model_sha256"]:
            raise ValueError("CV model digest mismatch")
    recomputed = {r: {} for r in spec["routes"]}
    loaded = 0
    for seed in spec["seeds"]:
        saved = pd.read_csv(output / f"folds-{seed}.csv", dtype={"group_id": str})
        pd.testing.assert_frame_equal(saved, make_folds(train, seed, spec["n_splits"]), check_dtype=False)
        folds = saved.set_index("sample_id").loc[train.sample_id, "fold"].to_numpy()
        for route in recomputed:
            recomputed[route][str(seed)] = {}
        for target in TARGETS:
            oof = pd.read_csv(output / "oof" / f"{seed}-{target}.csv").set_index("sample_id").loc[train.sample_id]
            np.testing.assert_allclose(oof[target], train[target], rtol=1e-14)
            np.testing.assert_array_equal(oof["fold"], folds)
            pred = {"M0": baseline_predictions(train, target, folds, False), "MS": baseline_predictions(train, target, folds, True), "M1": np.empty(len(train))}
            for f in range(spec["n_splits"]):
                pred["M1"][folds == f] = hd_median(train.loc[folds != f, target])
            for route in spec["regressors"]:
                pred[route] = np.full(len(train), np.nan)
                for fold in range(spec["n_splits"]):
                    model = joblib.load(output / "models" / f"{seed}-{route}-{target}-{fold}.joblib")
                    valid = train.loc[folds == fold]
                    with threadpool_limits(limits=1):
                        p = model.predict(valid)
                        np.testing.assert_allclose(p, model.predict(valid.iloc[::-1])[::-1], rtol=1e-12, atol=1e-10)
                        np.testing.assert_allclose(model.predict(test), model.predict(test.iloc[::-1])[::-1], rtol=1e-12, atol=1e-10)
                    pred[route][folds == fold] = p
                    loaded += 1
            pred["S25"] = s25_predictions(pred["M0"], pred[configs["legacy_anchor"]["S25"]["sources"][target]])
            for route in pred:
                np.testing.assert_allclose(pred[route], oof[route], rtol=1e-12, atol=1e-10)
                recomputed[route][str(seed)][target] = target_metrics(train, target, pred[route], folds)
        for route in recomputed:
            recomputed[route][str(seed)].update(combined_metrics(recomputed[route][str(seed)]))
    def numeric_equal(a, b):
        if isinstance(a, dict):
            assert set(a) == set(b)
            for key in a:
                numeric_equal(a[key], b[key])
        else:
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    numeric_equal(recomputed, summary["metrics"])
    promotion, selected = choose(recomputed, spec)
    if selected != summary["selected"] or promotion != summary["promotion"]:
        raise ValueError("Selection mismatch")
    write_json(output / "independent_verification.json", {"status": "PASS", "cv_models_loaded": loaded,
               "oof_metrics_and_selection_recomputed": True, "row_order_invariance": True, "new_fits": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--infer", action="store_true")
    args = parser.parse_args()
    if args.infer:
        sys.addaudithook(deny_training_reads)
        release = json.loads((args.output / "release/models.json").read_text())
        test = load_v2(Path("复赛_test"), "test", 322)
        with threadpool_limits(limits=1):
            values = np.column_stack([predict_saved(release[t], args.output / "release", test) for t in TARGETS])
        with (args.output / "release/cold.csv").open("xb") as handle:
            handle.write(serialize(test.sample_id, values))
    elif args.verify:
        verify(Path.cwd(), args.output)
    else:
        run(Path.cwd(), args.output)


if __name__ == "__main__":
    main()
