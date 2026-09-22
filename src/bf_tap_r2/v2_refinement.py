"""Frozen C2-relative V2.1 refinement, with fold-safe ensembles and isolated releases."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .audit import digest, write_json
from .cv import append_event, target_metrics, combined_metrics
from .data import FEATURES, TARGETS, SUBMISSION_COLUMNS
from .low_signal_cv import sensitivity
from .metrics import wmape
from .models import SnapshotRegressor, inputs
from .submission import package, validate_result, deny_training_reads
from .v2_release import load_v2
from .weak_models import assess_candidate


def spec_at(root):
    return yaml.safe_load((root / "configs/round2_v2_1/experiment.yaml").read_text())


class L15Regressor:
    def __init__(self, config):
        self.config = dict(config)

    def transform(self, frame):
        x = inputs(frame)
        x["spout_no"] = pd.Categorical(x.spout_no, categories=self.categories_)
        return x

    def fit(self, frame, y):
        self.categories_ = sorted(frame.spout_no.unique().tolist())
        self.model_ = lgb.LGBMRegressor(**self.config)
        self.model_.fit(self.transform(frame), np.asarray(y, dtype=float), categorical_feature=["spout_no"])
        return self

    def predict(self, frame):
        pred = np.asarray(self.model_.predict(self.transform(frame)), dtype=float)
        if pred.shape != (len(frame),) or not np.isfinite(pred).all():
            raise ValueError("Invalid LightGBM predictions")
        return pred


def new_model(member, spec, legacy):
    if member == "L15":
        return L15Regressor(spec["lightgbm"])
    cfg = deepcopy(legacy)
    cfg["execution"]["model_seed"] = int(member.split("_")[-1]) if member.startswith("C2_") else 42
    if member == "D4":
        cfg["models"]["C2"] = deepcopy(spec["d4"])
    return SnapshotRegressor("C1" if member == "C1" else "C2", cfg)


def choose_c2(metrics, spec):
    decisions, selected = {}, {}
    for target in TARGETS:
        anchor = {s: metrics["C2"][str(s)][target] for s in spec["split_seeds"]}
        decisions[target] = {r: assess_candidate({s: metrics[r][str(s)][target] for s in spec["split_seeds"]}, anchor, spec["promotion"])
                             for r in spec["routes"] if r != "C2"}
        eligible = [r for r, d in decisions[target].items() if d["eligible"]]
        if not eligible:
            selected[target] = "C2"
            continue
        best = min(decisions[target][r]["mean_wmape"] for r in eligible)
        tied = [r for r in eligible if decisions[target][r]["mean_wmape"] <= best + spec["tie_tolerance"]]
        selected[target] = min(tied, key=lambda r: (*spec["inference_cost"][r], spec["routes"].index(r)))
    return decisions, selected


def fold_vector(parent, frame, split):
    assignment = pd.read_csv(parent / f"folds-{split}.csv", dtype={"group_id": str})
    if not assignment.sample_id.is_unique or set(assignment.sample_id) != set(frame.sample_id):
        raise ValueError("Frozen fold ID set mismatch")
    a = assignment.set_index("sample_id").loc[frame.sample_id]
    if set(a.fold) != set(range(5)) or not (a.seed == split).all() or a.groupby("group_id").fold.nunique().max() != 1:
        raise ValueError("Invalid frozen folds")
    return a.fold.to_numpy()


def combine_same_fold(members, required, expected_fold):
    arrays = []
    for name in required:
        fold, values = members[name]
        if fold != expected_fold:
            raise ValueError("Ensemble members must use the same held-out fold")
        arrays.append(np.asarray(values, dtype=float))
    if any(a.shape != arrays[0].shape for a in arrays) or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Invalid ensemble members")
    return np.mean(np.stack(arrays), axis=0)


def isolated_payload(base_payload, ids, target, prediction):
    rows = validate_result(base_payload, ids)
    prediction = np.asarray(prediction, dtype=float)
    if target not in TARGETS or prediction.shape != (len(rows),):
        raise ValueError("Invalid isolated replacement")
    column = "pred_" + target
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SUBMISSION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row, value in zip(rows, prediction):
        row[column] = format(value, ".17g")
        writer.writerow(row)
    result = buffer.getvalue().encode()
    rebuilt = validate_result(result, ids)
    other = "pred_" + next(t for t in TARGETS if t != target)
    if any(a[other] != b[other] for a, b in zip(validate_result(base_payload, ids), rebuilt)):
        raise ValueError("Untouched target strings changed")
    return result


def parent_model(parent, split, target, fold, member):
    route = "C1" if member == "C1" else "C2"
    return parent / "models" / f"{split}-{route}-{target}-{fold}.joblib"


def new_path(output, split, target, fold, member):
    return output / "models" / f"{split}-{target}-{fold}-{member}.joblib"


def check_identity(root, output):
    manifest = json.loads((output / "manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if digest(root / name) != sha:
            raise ValueError(f"Frozen file changed: {name}")
    return manifest


def initialize(root, output, spec):
    if not output.resolve().is_relative_to(root.resolve() / "local/runs/round2-v2.1"):
        raise ValueError("Private V2.1 run required")
    output.mkdir(parents=True, exist_ok=False)
    parent = root / spec["reference_run"]
    old = json.loads((parent / "manifest.json").read_text())
    files = {}
    for name, expected in {**old["files"], **old["sources"]}.items():
        if digest(root / name) != expected:
            raise ValueError(f"Parent identity mismatch: {name}")
        files[name] = expected
    zip_path = parent / "release/Luqhhh_bf_tap_predict_round2.zip"
    if digest(zip_path) != spec["reference_zip_sha256"]:
        raise ValueError("Original C2 package changed")
    release = json.loads((parent / "release/models.json").read_text())
    for t in TARGETS:
        record = release[t]
        if record["route"] != "C2" or digest(parent / "release" / record["model"]) != record["model_sha256"]:
            raise ValueError("Original full C2 model identity mismatch")
    for name in ("manifest.json", "summary.json", "COMPLETE.json", "independent_verification.json", "release/models.json", "release/result.csv", "release/Luqhhh_bf_tap_predict_round2.zip"):
        p = parent / name
        files[str(p.relative_to(root))] = digest(p)
    for split in spec["split_seeds"]:
        for name in [f"folds-{split}.csv", *[f"oof/{split}-{t}.csv" for t in TARGETS]]:
            p = parent / name
            files[str(p.relative_to(root))] = digest(p)
    events = [json.loads(line) for line in (parent / "fit_ledger.jsonl").read_text().splitlines()]
    for e in events:
        if e["kind"] == "cv_complete" and e["tag"].split("-")[1] in ("C1", "C2"):
            p = parent / "models" / (e["tag"] + ".joblib")
            if digest(p) != e["model_sha256"]:
                raise ValueError("Original fold model changed")
            files[str(p.relative_to(root))] = e["model_sha256"]
    for record in release.values():
        p = parent / "release" / record["model"]
        files[str(p.relative_to(root))] = digest(p)
    for p in [root / "uv.lock", root / "configs/round2_v2_1/experiment.yaml", Path(__file__)]:
        files[str(p.relative_to(root))] = digest(p)
    for folder in ("residual_audit", "models", "oof", "metrics", "selection", "release"):
        (output / folder).mkdir()
    write_json(output / "manifest.json", {"files": files, "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "data_version": spec["data_version"], "reference_route": "C2", "spec": spec})
    return parent


def residual_audit(train, parent, output, spec):
    reports, curves = [], []
    for split in spec["split_seeds"]:
        folds = fold_vector(parent, train, split)
        for target in TARGETS:
            oof = pd.read_csv(parent / "oof" / f"{split}-{target}.csv").set_index("sample_id").loc[train.sample_id]
            np.testing.assert_array_equal(oof.fold, folds)
            np.testing.assert_allclose(oof[target], train[target], rtol=1e-14)
            y, c1, c2 = train[target].to_numpy(), oof.C1.to_numpy(), oof.C2.to_numpy()
            residual = c2-y
            absolute = np.abs(residual)
            groups = {"spout": train.spout_no.astype(str).to_numpy(),
                      "true_target_quintile_report_only": pd.qcut(train[target], 5, labels=False, duplicates="drop").to_numpy()}
            record = {"split_seed": split, "target": target, "residual_definition": "prediction minus truth",
                      "mean_residual": float(residual.mean()), "median_residual": float(np.median(residual)),
                      "C1_C2_residual_correlation": float(np.corrcoef(c1-y, c2-y)[0, 1]),
                      "opposite_error_sign_fraction": float(np.mean((c1-y)*(c2-y) < 0)),
                      "C2_wmape": wmape(y, c2), "B12_wmape": wmape(y, .5*c1+.5*c2), "groups": {}}
            for grouping, values in groups.items():
                record["groups"][grouping] = {str(k): {"rows": int(np.sum(values == k)), "wmape": wmape(y[values == k], c2[values == k]),
                    "absolute_error": float(absolute[values == k].sum()), "absolute_error_contribution": float(absolute[values == k].sum()/absolute.sum())} for k in sorted(set(values))}
            reports.append(record)
            for fold in range(spec["folds"]):
                tr, va = train.loc[folds != fold], train.loc[folds == fold]
                for member, expected in (("C1", c1), ("C2_42", c2)):
                    model = joblib.load(parent_model(parent, split, target, fold, member))
                    if model.config["execution"]["model_seed"] != 42 or model.estimator_.tree_count_ != 1500:
                        raise ValueError("Unexpected original training seed/tree count")
                    with threadpool_limits(limits=1):
                        np.testing.assert_allclose(model.predict(va), expected[folds == fold], rtol=1e-12, atol=1e-10)
                    if member == "C2_42":
                        for trees in spec["diagnostic_tree_counts"]:
                            with threadpool_limits(limits=1):
                                tp = model.estimator_.predict(inputs(tr, True), ntree_end=trees)
                                vp = model.estimator_.predict(inputs(va, True), ntree_end=trees)
                            curves.append({"split_seed": split, "model_seed": 42, "target": target, "fold": fold, "trees": trees,
                                           "train_wmape": wmape(tr[target], tp), "validation_wmape": wmape(va[target], vp)})
    write_json(output / "residual_audit/summary.json", {"records": reports, "staged_predictions": curves, "new_fits": 0,
               "tree_counts_are_diagnostic_only": True, "target_quintiles_never_used_at_inference": True})
    print("P0/P1 PASS: frozen C2/C1 identity and residual diagnostics; zero new fits", flush=True)


def fit_once(model, frame, y, path, output, stage, details, limit):
    ledger = output / "fit_ledger.jsonl"
    events = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    starts = sum(e["event"] == "start" and e["stage"] == stage for e in events)
    if starts >= limit or path.exists():
        raise ValueError("Fit budget exhausted or model path exists")
    append_event(ledger, {"event": "start", "stage": stage, "fit_number": starts+1, **details,
                          "train_rows": len(frame), "training_ids_sha256": __import__("hashlib").sha256("\n".join(frame.sample_id).encode()).hexdigest()})
    tick = time.monotonic()
    with threadpool_limits(limits=1):
        model.fit(frame, y)
    joblib.dump(model, path)
    append_event(ledger, {"event": "complete", "stage": stage, **details, "model": str(path),
                          "model_sha256": digest(path), "seconds": time.monotonic()-tick})
    print(f"{stage} fit {starts+1}/{limit}: {details}", flush=True)
    return model


def calculate_oof(train, parent, output, spec, compare_saved=False):
    scores = {r: {} for r in spec["routes"]}
    all_predictions = {r: {t: {} for t in TARGETS} for r in spec["routes"]}
    for split in spec["split_seeds"]:
        folds = fold_vector(parent, train, split)
        for target in TARGETS:
            arrays = {r: np.full(len(train), np.nan) for r in spec["routes"]}
            diagnostic = {m: np.full(len(train), np.nan) for m in spec["new_cv_members"]}
            for fold in range(spec["folds"]):
                valid = train.loc[folds == fold]
                members = {}
                for name in ("C1", "C2_42", *spec["new_cv_members"]):
                    path = parent_model(parent, split, target, fold, name) if name in ("C1", "C2_42") else new_path(output, split, target, fold, name)
                    model = joblib.load(path)
                    with threadpool_limits(limits=1):
                        prediction = model.predict(valid)
                        if compare_saved:
                            np.testing.assert_allclose(prediction, model.predict(valid.iloc[::-1])[::-1], rtol=1e-12, atol=1e-10)
                    members[name] = (fold, prediction)
                    if name in diagnostic:
                        diagnostic[name][folds == fold] = prediction
                for route in arrays:
                    arrays[route][folds == fold] = combine_same_fold(members, spec["members"][route], fold)
            oof_path = output / "oof" / f"{split}-{target}.csv"
            if compare_saved:
                saved = pd.read_csv(oof_path).set_index("sample_id").loc[train.sample_id]
                np.testing.assert_array_equal(saved.fold, folds)
                np.testing.assert_allclose(saved[target], train[target], rtol=1e-14)
                for route, pred in {**arrays, **diagnostic}.items():
                    np.testing.assert_allclose(saved[route], pred, rtol=1e-12, atol=1e-10)
            else:
                saved = train[["sample_id", "spout_no", target]].copy()
                saved["fold"] = folds
                for route, pred in {**arrays, **diagnostic}.items():
                    saved[route] = pred
                saved.to_csv(oof_path, index=False, mode="x")
            parent_oof = pd.read_csv(parent / "oof" / f"{split}-{target}.csv").set_index("sample_id").loc[train.sample_id]
            np.testing.assert_allclose(arrays["C2"], parent_oof.C2, rtol=1e-12, atol=1e-10)
            for route, pred in arrays.items():
                scores[route].setdefault(str(split), {})[target] = target_metrics(train, target, pred, folds)
                all_predictions[route][target][split] = pred
        for route in scores:
            scores[route][str(split)].update(combined_metrics(scores[route][str(split)]))
    return scores, all_predictions


def numeric_equal(a, b):
    if isinstance(a, dict):
        if set(a) != set(b):
            raise ValueError("Metric key mismatch")
        for key in a:
            numeric_equal(a[key], b[key])
    elif isinstance(a, str) or a is None:
        if a != b:
            raise ValueError("Value mismatch")
    else:
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def verify_cv(root, output):
    manifest = check_identity(root, output)
    spec = manifest["spec"]
    events = [json.loads(s) for s in (output / "fit_ledger.jsonl").read_text().splitlines()]
    starts = [e for e in events if e["stage"] == "cv" and e["event"] == "start"]
    done = [e for e in events if e["stage"] == "cv" and e["event"] == "complete"]
    if len(starts) != 80 or len(done) != 80 or len({e["model"] for e in done}) != 80:
        raise ValueError("CV budget/ledger incomplete")
    for e in done:
        if digest(Path(e["model"])) != e["model_sha256"]:
            raise ValueError("New model identity changed")
    train = load_v2(root / "复赛_train", "train", 2754)
    metrics, _ = calculate_oof(train, root / spec["reference_run"], output, spec, compare_saved=True)
    saved = json.loads((output / "selection/summary.json").read_text())
    numeric_equal(metrics, saved["metrics"])
    decisions, selected = choose_c2(metrics, spec)
    numeric_equal(decisions, saved["decisions"])
    if selected != saved["selected"]:
        raise ValueError("Selection mismatch")
    write_json(output / "selection/independent_verification.json", {"status": "PASS", "new_models_read": 80,
               "reference_fold_models_read": 40, "OOF_metrics_and_selection_reproduced": True,
               "same_fold_ensemble_and_reverse_order": True, "new_fits": 0})


def release_prediction(root, record, frame):
    values = []
    for member in record["members"]:
        path = root / member["path"]
        if digest(path) != member["sha256"]:
            raise ValueError("Full model identity changed")
        model = joblib.load(path)
        with threadpool_limits(limits=1):
            pred = model.predict(frame)
            np.testing.assert_allclose(pred, model.predict(frame.iloc[::-1])[::-1], rtol=1e-12, atol=1e-10)
            chunks = np.concatenate([model.predict(frame.iloc[:1]), model.predict(frame.iloc[1:111]), model.predict(frame.iloc[111:])])
            np.testing.assert_allclose(pred, chunks, rtol=1e-12, atol=1e-10)
        values.append(pred)
    return np.mean(np.stack(values), axis=0)


def build_releases(root, output):
    manifest = check_identity(root, output)
    spec = manifest["spec"]
    if json.loads((output / "selection/independent_verification.json").read_text())["status"] != "PASS":
        raise ValueError("Independent CV check required")
    summary = json.loads((output / "selection/summary.json").read_text())
    parent = root / spec["reference_run"]
    train, test = load_v2(root / "复赛_train", "train", 2754), load_v2(root / "复赛_test", "test", 322)
    legacy = yaml.safe_load((root / spec["legacy_catboost_config"]).read_text())
    old_full = json.loads((parent / "release/models.json").read_text())
    reference_payload = (parent / "release/result.csv").read_bytes()
    records = {}
    for target, route in summary["selected"].items():
        if route == "C2":
            continue
        members = []
        for member in spec["members"][route]:
            if member == "C2_42":
                path = parent / "release" / old_full[target]["model"]
            else:
                path = output / "models" / f"full-{target}-{member}.joblib"
                fit_once(new_model(member, spec, legacy), train, train[target], path, output, "full",
                         {"target": target, "member": member}, spec["budget"]["full_fits"])
            members.append({"name": member, "path": str(path.relative_to(root)), "sha256": digest(path)})
        record = {"target": target, "route": route, "members": members}
        values = release_prediction(root, record, test)
        letter = "A" if target == "tap_iron" else "B"
        folder = output / "release" / letter
        folder.mkdir(exist_ok=False)
        package(folder, isolated_payload(reference_payload, test.sample_id, target, values), test.sample_id)
        record["package"] = str((folder / "Luqhhh_bf_tap_predict_round2.zip").relative_to(root))
        record["zip_sha256"] = digest(folder / "Luqhhh_bf_tap_predict_round2.zip")
        records[letter] = record
    write_json(output / "release/manifest.json", {"reference_csv": str((parent / "release/result.csv").relative_to(root)),
               "reference_csv_sha256": digest(parent / "release/result.csv"), "reference_zip_sha256": spec["reference_zip_sha256"],
               "packages": records, "C_status": "NOT_GENERATED_REQUIRES_POSITIVE_A_AND_B_PLATFORM_FEEDBACK",
               "remaining_account_quota": "UNKNOWN", "agent_uploads": 0})
    subprocess.run([sys.executable, "-m", "bf_tap_r2.v2_refinement", "infer", "--output", str(output)], check=True, cwd=root)
    for letter in records:
        if (output / "release" / letter / "cold.csv").read_bytes() != (output / "release" / letter / "result.csv").read_bytes():
            raise ValueError("Cold isolated release differs")
    write_json(output / "release/verification.json", {"status": "PASS", "packages": list(records), "label_free_cold_byte_identical": True,
               "untouched_column_strings_preserved": True, "shuffle_and_chunk_invariance": True, "C_generated": False})
    check_identity(root, output)
    print(json.dumps({"selected": summary["selected"], "packages": records}), flush=True)


def run(root, output):
    spec = spec_at(root)
    parent = initialize(root, output, spec)
    try:
        train = load_v2(root / "复赛_train", "train", 2754)
        legacy = yaml.safe_load((root / spec["legacy_catboost_config"]).read_text())
        residual_audit(train, parent, output, spec)
        for member in spec["new_cv_members"]:
            for split in spec["split_seeds"]:
                folds = fold_vector(parent, train, split)
                for target in TARGETS:
                    for fold in range(spec["folds"]):
                        tr = train.loc[folds != fold]
                        fit_once(new_model(member, spec, legacy), tr, tr[target], new_path(output, split, target, fold, member), output, "cv",
                                 {"split_seed": split, "target": target, "fold": fold, "member": member,
                                  "model_seed": int(member.split("_")[-1]) if member.startswith("C2_") else 42}, spec["budget"]["cv_fits"])
        metrics, predictions = calculate_oof(train, parent, output, spec)
        decisions, selected = choose_c2(metrics, spec)
        bootstrap = {}
        for target, route in selected.items():
            if route == "C2":
                continue
            distribution, intervals = sensitivity(train, target, predictions[route][target], predictions["C2"][target], spec["bootstrap"])
            distribution.to_csv(output / "selection" / f"bootstrap-{target}.csv", index=False, mode="x")
            bootstrap[target] = {"route": route, "intervals": intervals,
                 "mean_delta": float(distribution["mean"].mean()), "fraction_mean_delta_negative": float((distribution["mean"] < 0).mean()),
                 "unique_sample_units": len(train), "shared_indices_across_routes_and_split_seeds": True,
                 "scope": "fixed OOF sample-composition sensitivity, excludes retraining uncertainty"}
        write_json(output / "selection/summary.json", {"reference_route": "C2", "metrics": metrics, "decisions": decisions,
                   "selected": selected, "bootstrap": bootstrap, "cv_fits": 80, "platform_scores": None})
        subprocess.run([sys.executable, "-m", "bf_tap_r2.v2_refinement", "verify", "--output", str(output)], check=True, cwd=root)
        build_releases(root, output)
        events = [json.loads(s) for s in (output / "fit_ledger.jsonl").read_text().splitlines()]
        write_json(output / "COMPLETE.json", {"G0": "PASS", "cv_fits": sum(e["stage"] == "cv" and e["event"] == "start" for e in events),
                   "full_fits": sum(e["stage"] == "full" and e["event"] == "start" for e in events), "selected": selected,
                   "reference_zip_unchanged": digest(parent / "release/Luqhhh_bf_tap_predict_round2.zip") == spec["reference_zip_sha256"],
                   "platform_status": "AWAITING_ACCOUNT_QUOTA_AND_A_B_RESULTS", "agent_uploads": 0})
    except Exception as exc:
        write_json(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def cold_infer(root, output):
    sys.addaudithook(deny_training_reads)
    record = json.loads((output / "release/manifest.json").read_text())
    base = root / record["reference_csv"]
    if digest(base) != record["reference_csv_sha256"]:
        raise ValueError("Original CSV changed")
    test = load_v2(root / "复赛_test", "test", 322)
    for letter, package_record in record["packages"].items():
        prediction = release_prediction(root, package_record, test)
        payload = isolated_payload(base.read_bytes(), test.sample_id, package_record["target"], prediction)
        with (output / "release" / letter / "cold.csv").open("xb") as handle:
            handle.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify", "release", "infer"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"run": run, "verify": verify_cv, "release": build_releases, "infer": cold_infer}[args.command](Path.cwd(), args.output.resolve())


if __name__ == "__main__":
    main()
