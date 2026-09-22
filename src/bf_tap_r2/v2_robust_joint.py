"""V2.3 frozen normalized-loss and joint-learning development experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .audit import digest, write_json
from .candidate_tiers import classify_candidates
from .cv import append_event, target_metrics
from .data import TARGETS
from .low_signal_cv import sensitivity
from .normalized_models import JointSnapshotRegressor, NormalizedSnapshotRegressor
from .v2_checkpoint import recover_oof, verify_identity
from .v2_refinement import fold_vector
from .v2_release import load_v2


def id_digest(frame):
    return hashlib.sha256("\n".join(frame.sample_id.astype(str)).encode()).hexdigest()


def check_packages(root, spec):
    previous = root / spec["previous_run"]
    paths = [previous / "release/V22_I_ONLY/Luqhhh_bf_tap_predict_round2.zip",
             Path(spec["platform"]["desktop_package"])]
    for path in paths:
        if digest(path) != spec["platform"]["zip_sha256"]:
            raise ValueError(f"Frozen V22 package changed: {path}")
    parent = root / "local/runs/round2-v2.1/refinement-r1/release/B/Luqhhh_bf_tap_predict_round2.zip"
    if digest(parent) != "ec12c3663edf77fdc0d7e8e9fa7bb1aeb1d1c46db4107a8c6724f25c8039e1e2":
        raise ValueError("Current B anchor changed")
    def rows(path):
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != ["result.csv"] or archive.testzip():
                raise ValueError("Invalid frozen ZIP")
            with archive.open("result.csv") as stream:
                return pd.read_csv(stream, dtype=str).set_index("sample_id")
    a, b = rows(parent), rows(paths[0])
    if not a.index.is_unique or not b.index.is_unique or set(a.index) != set(b.index) or len(b) != 322:
        raise ValueError("Frozen package IDs mismatch")
    b = b.loc[a.index]
    if not (a.pred_tap_time_len == b.pred_tap_time_len).all() or (a.pred_tap_iron != b.pred_tap_iron).sum() != 322:
        raise ValueError("Frozen V22 columns changed")
    return {str(p): digest(p) for p in [*paths, parent]}


def initialize(root, output):
    if not output.resolve().is_relative_to(root / "local/runs/round2-v2.3"):
        raise ValueError("Fresh private V2.3 run required")
    spec = yaml.safe_load((root / "configs/round2_v2_3/experiment.yaml").read_text())
    old = verify_identity(root, root / spec["previous_run"])
    packages = check_packages(root, spec)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("models", "oof", "selection", "diagnostics"):
        (output / name).mkdir()
    files = dict(old["files"])
    sources = [root / "configs/round2_v2_3/experiment.yaml", root / spec["policy"],
               root / "uv.lock", Path(__file__), Path(__file__).with_name("normalized_models.py"),
               Path(__file__).with_name("candidate_tiers.py")]
    sources += list((root / spec["previous_run"] / "oof").glob("*.csv"))
    sources += [root / spec["previous_run"] / "selection/summary.json",
                root / spec["previous_run"] / "selection/independent_verification.json"]
    for path in sources:
        files[str(path.relative_to(root))] = digest(path)
    write_json(output / "manifest.json", {"files": files, "packages": packages, "spec": spec,
               "policy": yaml.safe_load((root / spec["policy"]).read_text()),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
    append_event(output / "platform_events.jsonl", {"event": "quota_user_reported", "date": "2026-09-22",
                 "remaining": 0, "daily_cap": 5, "supersedes_for_planning": "V22 initial remaining=1",
                 "next_date": "2026-09-23", "first_package": "V22_I_ONLY", "immutable_package_verified": packages,
                 "upload_actor": "user", "old_snapshot_preserved": True})
    return spec


def checked_manifest(root, output):
    manifest = verify_identity(root, output)
    if check_packages(root, manifest["spec"]) != manifest["packages"]:
        raise ValueError("Package identities changed")
    return manifest


def fit_model(output, train, valid, spec, route, target, split, fold):
    parameters = dict(spec["common"], loss_function=spec["losses"][route])
    model = JointSnapshotRegressor(parameters) if route == "J1" else NormalizedSnapshotRegressor(parameters, target)
    name = f"{split}-{fold}-{route}-{target or 'joint'}"
    path = output / "models" / f"{name}.joblib"
    identity = {"route": route, "target": target, "split": split, "heldout_fold": fold,
                "train_ids_sha256": id_digest(train), "valid_ids_sha256": id_digest(valid), "model": path.name}
    if set(train.sample_id) & set(valid.sample_id):
        raise ValueError("Train/validation ID overlap")
    append_event(output / "fit_ledger.jsonl", dict(identity, event="start", stage="cv"))
    model.model_identity_ = identity
    model.fit(train, train[list(TARGETS)] if route == "J1" else train[target])
    if path.exists():
        raise FileExistsError(path)
    joblib.dump(model, path)
    record = dict(identity, event="complete", stage="cv", model_sha256=digest(path),
                  target_scales=model.target_scales_.tolist(), target_fields=list(model.target_fields_),
                  input_fields=list(model.input_fields_), actual_parameters=model.actual_parameters_)
    write_json(path.with_suffix(".json"), record)
    append_event(output / "fit_ledger.jsonl", record)
    print(f"fit complete: {name}", flush=True)
    with threadpool_limits(limits=1):
        return model.predict(valid)


def load_predictions(root, output, spec, train, replay=False):
    predictions = {t: {r: {} for r in spec["candidates"][t]} for t in TARGETS}
    for split in spec["split_seeds"]:
        folds = fold_vector(root / spec["fold_run"], train, split)
        for target in TARGETS:
            for route in ("N1", "H1", "J1"):
                predictions[target][route][split] = np.full(len(train), np.nan)
        for fold in range(spec["folds"]):
            training, valid = train.loc[folds != fold], train.loc[folds == fold]
            for route, target in [(r, t) for r in ("N1", "H1") for t in TARGETS] + [("J1", None)]:
                if not replay:
                    values = fit_model(output, training, valid, spec, route, target, split, fold)
                else:
                    path = output / "models" / f"{split}-{fold}-{route}-{target or 'joint'}.joblib"
                    record = json.loads(path.with_suffix(".json").read_text())
                    if digest(path) != record["model_sha256"]:
                        raise ValueError("New model digest mismatch")
                    model = joblib.load(path)
                    expected_identity = {"route": route, "target": target, "split": split, "heldout_fold": fold,
                                         "train_ids_sha256": id_digest(training), "valid_ids_sha256": id_digest(valid), "model": path.name}
                    if model.model_identity_ != expected_identity or any(record[k] != v for k, v in expected_identity.items()):
                        raise ValueError("New model fold identity mismatch")
                    targets = list(TARGETS) if route == "J1" else [target]
                    np.testing.assert_allclose(model.target_scales_, training[targets].mean().to_numpy(), rtol=1e-14)
                    if (list(model.target_fields_) != targets or model.parameters != dict(spec["common"], loss_function=spec["losses"][route])
                            or model.actual_parameters_ != record["actual_parameters"] or model.estimator_.get_all_params() != record["actual_parameters"]):
                        raise ValueError("Model parameters/order mismatch")
                    with threadpool_limits(limits=1):
                        values = model.predict(valid)
                        np.testing.assert_allclose(values, model.predict(valid.iloc[::-1])[::-1], rtol=1e-12, atol=1e-10)
                        batches = np.concatenate([model.predict(valid.iloc[i:i+127]) for i in range(0, len(valid), 127)])
                        np.testing.assert_allclose(values, batches, rtol=1e-12, atol=1e-10)
                        np.testing.assert_allclose(values[:1], model.predict(valid.iloc[:1]), rtol=1e-12, atol=1e-10)
                for index, t in enumerate(TARGETS if route == "J1" else (target,)):
                    predictions[t][route][split][folds == fold] = values[:, index] if route == "J1" else values
    return predictions


def finish(root, output, spec, train, predictions, replay=False):
    metrics, diagnostics = {}, []
    # Independently restore original held-out models before using anchor/prefix OOF.
    _, old_metrics, old_predictions = recover_oof(root, root / spec["previous_run"], verify_saved=True)
    for target in TARGETS:
        reference, v22 = spec["reference_by_target"][target], spec["v22_by_target"][target]
        for route in (reference, v22):
            predictions[target][route] = old_predictions[target][route]
        metrics[target] = {r: {} for r in [reference, v22, *spec["candidates"][target]]}
        for split in spec["split_seeds"]:
            folds = fold_vector(root / spec["fold_run"], train, split)
            anchor = predictions[target][reference][split]
            # All arrays use identical sample IDs and frozen held-out folds, in original units.
            for blend, member in (("AH", "H1"), ("AJ", "J1")):
                predictions[target][blend][split] = .5 * anchor + .5 * predictions[target][member][split]
            saved = train[["sample_id", "spout_no", target]].copy()
            saved["fold"] = folds
            y = train[target].to_numpy()
            groups = pd.qcut(np.abs(anchor-y), 5, labels=False, duplicates="drop")
            for route, by_split in predictions[target].items():
                values = by_split[split]
                saved[route] = values
                metrics[target][route][str(split)] = target_metrics(train, target, values, folds)
                absolute = np.abs(values-y)
                for group in ["all", *sorted(set(groups))]:
                    mask = np.ones(len(train), dtype=bool) if group == "all" else groups == group
                    diagnostics.append({"target": target, "split": split, "route": route,
                        "anchor_error_quintile": str(group), "n": int(mask.sum()),
                        "absolute_error_sum": float(absolute[mask].sum()), "mae": float(absolute[mask].mean()),
                        "max_absolute_error_original_units": float(absolute[mask].max()),
                        "share_of_route_total_absolute_error": float(absolute[mask].sum()/absolute.sum()),
                        "wmape": float(absolute[mask].sum()/y[mask].sum()),
                        "fraction_better_than_anchor": float((absolute[mask] < np.abs(anchor-y)[mask]).mean())})
            path = output / "oof" / f"{split}-{target}.csv"
            if replay:
                stored = pd.read_csv(path)
                if not stored.sample_id.is_unique or stored.sample_id.tolist() != saved.sample_id.tolist() or list(stored.columns) != list(saved.columns):
                    raise ValueError("OOF IDs/columns mismatch")
                np.testing.assert_allclose(stored.drop(columns="sample_id"), saved.drop(columns="sample_id"), rtol=1e-12, atol=1e-10)
            else:
                saved.to_csv(path, index=False, mode="x", float_format="%.17g")
            for route in (reference, v22):
                if metrics[target][route][str(split)] != old_metrics[target][route][str(split)]:
                    raise ValueError("Recovered anchor metric mismatch")
    policy = json.loads((output / "manifest.json").read_text())["policy"]
    tiers = classify_candidates(metrics, spec, policy)
    rows = []
    for target in TARGETS:
        reference, v22 = spec["reference_by_target"][target], spec["v22_by_target"][target]
        for route in spec["candidates"][target]:
            decision = tiers["decisions"][target][route]
            row = {"target": target, "candidate": route, "mean_wmape": decision["mean_wmape"],
                   "delta_vs_current": -decision["mean_gain"],
                   "delta_vs_v22": decision["mean_wmape"] - np.mean([metrics[target][v22][str(s)]["wmape"] for s in spec["split_seeds"]]),
                   "improved_folds": decision["improved_folds"], "worst_spout_delta": decision["worst_spout_delta"],
                   "tier": decision["tier"], "failed_conditions": ",".join(decision["failed_conditions"])}
            for s in spec["split_seeds"]:
                row[f"wmape_{s}"] = metrics[target][route][str(s)]["wmape"]
                row[f"delta_current_{s}"] = row[f"wmape_{s}"] - metrics[target][reference][str(s)]["wmape"]
                row[f"delta_v22_{s}"] = row[f"wmape_{s}"] - metrics[target][v22][str(s)]["wmape"]
            rows.append(row)
    bootstraps = {}
    for record in tiers["submission_priority"]:
        target, route = record["target"], record["candidate"]
        reference = spec["reference_by_target"][target]
        distribution, intervals = sensitivity(train, target, predictions[target][route], predictions[target][reference], spec["bootstrap"])
        key = f"{target}-{route}"
        bootstraps[key] = {"intervals": intervals, "fraction_mean_delta_negative": float((distribution["mean"] < 0).mean()),
                           "interpretation": "Fixed OOF sample-composition sensitivity only; not a gate or platform guarantee"}
        path = output / "selection" / f"bootstrap-{key}.csv"
        if replay:
            np.testing.assert_allclose(distribution, pd.read_csv(path), rtol=1e-10, atol=1e-14)
        else:
            distribution.to_csv(path, index=False, mode="x")
    summary = {"metrics": metrics, "tiers": tiers, "comparisons": rows, "bootstrap": bootstraps,
               "H1_minus_N1": {t: {str(s): metrics[t]["H1"][str(s)]["wmape"] - metrics[t]["N1"][str(s)]["wmape"] for s in spec["split_seeds"]} for t in TARGETS},
               "new_cv_fits": 50, "new_full_fits": 0, "reused_development_splits": True,
               "platform_queue_first": "V22_I_ONLY", "new_packages": 0}
    if not replay:
        write_json(output / "selection/summary.json", summary)
        pd.DataFrame(rows).to_csv(output / "selection/comparison.csv", index=False, mode="x")
        pd.DataFrame(diagnostics).to_csv(output / "diagnostics/error_contributions.csv", index=False, mode="x")
    else:
        if summary != json.loads((output / "selection/summary.json").read_text()):
            raise ValueError("Independent summary/selection mismatch")
        pd.testing.assert_frame_equal(pd.read_csv(output / "diagnostics/error_contributions.csv", dtype={"anchor_error_quintile": str}),
                                      pd.DataFrame(diagnostics), check_exact=False, rtol=1e-12, atol=1e-10)
    return summary


def verify(root, output):
    manifest = checked_manifest(root, output)
    spec = manifest["spec"]
    events = [json.loads(line) for line in (output / "fit_ledger.jsonl").read_text().splitlines()]
    starts, done = ([e for e in events if e["event"] == kind] for kind in ("start", "complete"))
    if len(starts) != 50 or len(done) != 50 or len({e["model"] for e in done}) != 50 or {e["model"] for e in starts} != {e["model"] for e in done}:
        raise ValueError("Real fit budget/ledger mismatch")
    for event in done:
        if event != json.loads((output / "models" / event["model"]).with_suffix(".json").read_text()):
            raise ValueError("Ledger model record mismatch")
    train = load_v2(root / "复赛_train", "train", 2754)
    predictions = load_predictions(root, output, spec, train, replay=True)
    finish(root, output, spec, train, predictions, replay=True)
    checked_manifest(root, output)
    write_json(output / "selection/independent_verification.json", {"status": "PASS", "replayed_models": 50,
               "same_fold_id_and_train_only_scales_verified": True, "predictions_metrics_tiers_bootstrap_reproduced": True,
               "reverse_batch_single_row_predictions_verified": True, "v22_package_unchanged": True, "new_fits": 0})


def evaluate(root, output):
    spec = initialize(root, output)
    try:
        # Check old model recovery before spending any new training budget.
        recover_oof(root, root / spec["previous_run"], verify_saved=True)
        train = load_v2(root / "复赛_train", "train", 2754)
        predictions = load_predictions(root, output, spec, train)
        finish(root, output, spec, train, predictions)
        subprocess.run([sys.executable, "-m", "bf_tap_r2.v2_robust_joint", "verify", "--output", str(output)], cwd=root, check=True)
        write_json(output / "COMPLETE.json", {"status": "PASS", "new_cv_fits": 50, "new_full_fits": 0, "new_packages": 0})
    except Exception as exc:
        write_json(output / "FAILED.json", {"error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["evaluate", "verify"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    (evaluate if args.action == "evaluate" else verify)(root, args.output.resolve())


if __name__ == "__main__":
    main()
