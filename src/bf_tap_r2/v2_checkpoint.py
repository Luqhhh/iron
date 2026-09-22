"""V2.2: zero-fit checkpoint evaluation against C2 iron + full B3 time."""
from __future__ import annotations

import argparse
import csv
import io
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
from .cv import target_metrics
from .data import TARGETS
from .low_signal_cv import sensitivity
from .models import inputs
from .submission import deny_training_reads, package, validate_result
from .v2_refinement import (check_identity as check_v21_identity, combine_same_fold, fit_once,
                            fold_vector, isolated_payload, new_model, new_path, numeric_equal, parent_model)
from .v2_release import load_v2
from .weak_models import assess_candidate


def load_spec(root):
    return yaml.safe_load((root / "configs/round2_v2_2/experiment.yaml").read_text())


def prefix_predict(model, frame, ntree_start, ntree_end):
    if model.route != "C2" or ntree_start != 0 or not 0 < ntree_end <= model.estimator_.tree_count_:
        raise ValueError("Invalid frozen prefix rule")
    # Preserve SnapshotRegressor's exact input and target-unit transformations.
    result = np.asarray(model.estimator_.predict(inputs(frame, True), ntree_start=ntree_start, ntree_end=ntree_end), dtype=float) * model.target_scale_
    if result.shape != (len(frame),) or not np.isfinite(result).all():
        raise ValueError("Invalid prefix prediction")
    return result


def choose_targetwise(metrics, spec):
    decisions, selected = {}, {}
    for target in TARGETS:
        reference = spec["reference_by_target"][target]
        anchors = {s: metrics[target][reference][str(s)] for s in spec["split_seeds"]}
        decisions[target] = {r: assess_candidate({s: metrics[target][r][str(s)] for s in spec["split_seeds"]}, anchors, spec["promotion"])
                             for r in spec["candidates"][target]}
        eligible = [r for r, d in decisions[target].items() if d["eligible"]]
        selected[target] = reference
        if eligible:
            best = min(decisions[target][r]["mean_wmape"] for r in eligible)
            tied = [r for r in eligible if decisions[target][r]["mean_wmape"] <= best + spec["tie_tolerance"]]
            selected[target] = spec["iron_tie_preference"] if spec["iron_tie_preference"] in tied else tied[0]
    return decisions, selected


def single_slot_ranking(metrics, selected, spec):
    records = []
    for t, route in selected.items():
        reference = spec["reference_by_target"][t]
        if route == reference:
            continue
        gains = {str(s): metrics[t][reference][str(s)]["wmape"] - metrics[t][route][str(s)]["wmape"] for s in spec["split_seeds"]}
        records.append({"target": t, "candidate": route, "gains": gains, "minimum_gain": min(gains.values())})
    records.sort(key=lambda x: -x["minimum_gain"])
    if len(records) == 2 and records[0]["minimum_gain"] - records[1]["minimum_gain"] <= spec["single_slot_tie_tolerance"]:
        records.sort(key=lambda x: x["target"] != spec["single_slot_tie_prefer"])
    return records


def verify_identity(root, output):
    manifest = json.loads((output / "manifest.json").read_text())
    for name, sha in manifest["files"].items():
        path = root / name
        if not path.is_file() or digest(path) != sha:
            raise ValueError(f"Missing or changed frozen input: {name}")
    return manifest


def initialize(root, output, spec):
    if not output.resolve().is_relative_to(root.resolve() / "local/runs/round2-v2.2"):
        raise ValueError("Private V2.2 output required")
    output.mkdir(parents=True, exist_ok=False)
    try:
        previous = root / spec["reference"]["run"]
        old = check_v21_identity(root, previous)
        files = dict(old["files"])
        for e in map(json.loads, (previous / "fit_ledger.jsonl").read_text().splitlines()):
            if e["event"] == "complete":
                p = Path(e["model"])
                if digest(p) != e["model_sha256"]:
                    raise ValueError(f"Previous model changed: {p}")
                files[str(p.resolve().relative_to(root))] = e["model_sha256"]
        anchor = previous / spec["reference"]["package"]
        if digest(anchor) != spec["reference"]["zip_sha256"]:
            raise ValueError("96.0982 B package SHA mismatch")
        with zipfile.ZipFile(anchor) as z:
            if z.namelist() != ["result.csv"] or z.testzip() or z.read("result.csv") != (anchor.parent / "result.csv").read_bytes():
                raise ValueError("Anchor ZIP/CSV mismatch")
        test = load_v2(root / "复赛_test", "test", 322)
        base_rows = validate_result((anchor.parent / "result.csv").read_bytes(), test.sample_id)
        original = root / spec["original_c2_run"]
        original_rows = validate_result((original / "release/result.csv").read_bytes(), test.sample_id)
        if any(a["pred_tap_iron"] != b["pred_tap_iron"] for a, b in zip(base_rows, original_rows)):
            raise ValueError("Anchor iron is not original C2 strings")
        previous_release = json.loads((previous / "release/manifest.json").read_text())
        time_members = previous_release["packages"]["B"]["members"]
        if [m["name"] for m in time_members] != ["C2_42", "C2_2026", "C2_2027"]:
            raise ValueError("Anchor time members are not frozen B3")
        for m in time_members:
            if digest(root / m["path"]) != m["sha256"]:
                raise ValueError("Anchor time source changed")
            files[m["path"]] = m["sha256"]
        needed = [anchor, anchor.parent / "result.csv", previous / "manifest.json", previous / "release/manifest.json",
                  previous / "selection/summary.json", previous / "selection/independent_verification.json",
                  root / "configs/round2_v2_2/experiment.yaml", Path(__file__), root / spec["previous_config"]]
        for split in spec["split_seeds"]:
            needed += [previous / "oof" / f"{split}-{t}.csv" for t in TARGETS]
        for p in needed:
            files[str(p.relative_to(root))] = digest(p)
        for folder in ("oof", "selection", "models", "release"):
            (output / folder).mkdir()
        write_json(output / "manifest.json", {"files": files, "spec": spec,
                   "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "reference_csv": str((anchor.parent / "result.csv").relative_to(root)),
                   "time_members": time_members, "new_cv_fits_allowed": 0})
    except Exception as exc:
        write_json(output / "FAILED_P0.json", {"error": str(exc)})
        raise


def recover_oof(root, output, verify_saved=False):
    manifest = verify_identity(root, output)
    spec = manifest["spec"]
    previous, original = root / spec["reference"]["run"], root / spec["original_c2_run"]
    train = load_v2(root / "复赛_train", "train", 2754)
    metrics, predictions, rules = {}, {}, []
    for target in TARGETS:
        reference = spec["reference_by_target"][target]
        routes = [reference, *spec["candidates"][target]]
        metrics[target] = {r: {} for r in routes}
        predictions[target] = {r: {} for r in routes}
        for split in spec["split_seeds"]:
            folds = fold_vector(original, train, split)
            old = pd.read_csv(previous / "oof" / f"{split}-{target}.csv").set_index("sample_id").loc[train.sample_id]
            np.testing.assert_array_equal(old.fold, folds)
            np.testing.assert_allclose(old[target], train[target], rtol=1e-14)
            result = {r: np.full(len(train), np.nan) for r in routes}
            for fold in range(spec["folds"]):
                valid = train.loc[folds == fold]
                if target == "tap_iron":
                    c2 = joblib.load(parent_model(original, split, target, fold, "C2_42"))
                    d4 = joblib.load(new_path(previous, split, target, fold, "D4"))
                    if c2.estimator_.tree_count_ != 1500 or d4.estimator_.tree_count_ != 3000:
                        raise ValueError("Iron tree count mismatch")
                    with threadpool_limits(limits=1):
                        c, d = c2.predict(valid), d4.predict(valid)
                        np.testing.assert_allclose(c, c2.predict(valid.iloc[::-1])[::-1], rtol=1e-12, atol=1e-10)
                        np.testing.assert_allclose(d, d4.predict(valid.iloc[::-1])[::-1], rtol=1e-12, atol=1e-10)
                    np.testing.assert_allclose(c, old.C2.to_numpy()[folds == fold], rtol=1e-12, atol=1e-10)
                    np.testing.assert_allclose(d, old.D4_CATBOOST_SHALLOW.to_numpy()[folds == fold], rtol=1e-12, atol=1e-10)
                    result[reference][folds == fold] = c
                    result["I1_D4_IRON"][folds == fold] = d
                    result["I2_C2_D4_IRON_EQUAL"][folds == fold] = combine_same_fold({"c2": (fold, c), "d4": (fold, d)}, ["c2", "d4"], fold)
                else:
                    full_members, prefix_members = {}, {}
                    for seed in spec["model_seeds"]:
                        name = f"C2_{seed}"
                        path = parent_model(original, split, target, fold, name) if seed == 42 else new_path(previous, split, target, fold, name)
                        model = joblib.load(path)
                        if model.estimator_.tree_count_ != 1500 or model.config["execution"]["model_seed"] != seed:
                            raise ValueError("Time model seed/tree count mismatch")
                        with threadpool_limits(limits=1):
                            full = prefix_predict(model, valid, 0, 1500)
                            np.testing.assert_allclose(full, model.predict(valid), rtol=1e-12, atol=1e-10)
                        full_members[name] = (fold, full)
                        rules.append({"target": target, "split_seed": split, "heldout_fold": fold, "model_seed": seed,
                                      "source_model": str(path.relative_to(root)), "source_model_sha256": digest(path),
                                      "source_tree_count": 1500, "ntree_start": 0, "ntree_end": 1000})
                    names = [f"C2_{s}" for s in spec["model_seeds"]]
                    full = combine_same_fold(full_members, names, fold)
                    np.testing.assert_allclose(full, old.B3_C2_SEED_ENSEMBLE.to_numpy()[folds == fold], rtol=1e-12, atol=1e-10)
                    # No prefix is evaluated until the complete B3 has been restored.
                    for seed in spec["model_seeds"]:
                        name = f"C2_{seed}"
                        path = parent_model(original, split, target, fold, name) if seed == 42 else new_path(previous, split, target, fold, name)
                        model = joblib.load(path)
                        with threadpool_limits(limits=1):
                            pred = prefix_predict(model, valid, 0, 1000)
                            np.testing.assert_allclose(pred, prefix_predict(model, valid.iloc[::-1], 0, 1000)[::-1], rtol=1e-12, atol=1e-10)
                        prefix_members[name] = (fold, pred)
                    result[reference][folds == fold] = full
                    result["T1_B3_PREFIX1000"][folds == fold] = combine_same_fold(prefix_members, names, fold)
            path = output / "oof" / f"{split}-{target}.csv"
            if verify_saved:
                stored = pd.read_csv(path).set_index("sample_id").loc[train.sample_id]
                np.testing.assert_array_equal(stored.fold, folds)
                np.testing.assert_allclose(stored[target], train[target], rtol=1e-14)
                for r, p in result.items():
                    np.testing.assert_allclose(p, stored[r], rtol=1e-12, atol=1e-10)
            else:
                stored = train[["sample_id", "spout_no", target]].copy()
                stored["fold"] = folds
                for r, p in result.items():
                    stored[r] = p
                stored.to_csv(path, index=False, mode="x")
            for r, p in result.items():
                metrics[target][r][str(split)] = target_metrics(train, target, p, folds)
                predictions[target][r][split] = p
    if not verify_saved:
        write_json(output / "selection/cv_prefix_rules.json", {"members": rules, "aggregation": "arithmetic_mean", "member_order": spec["model_seeds"]})
    return train, metrics, predictions


def evaluate(root, output):
    spec = load_spec(root)
    initialize(root, output, spec)
    try:
        train, metrics, predictions = recover_oof(root, output)
        decisions, selected = choose_targetwise(metrics, spec)
        previous = json.loads((root / spec["reference"]["run"] / "selection/summary.json").read_text())
        numeric_equal(decisions["tap_iron"]["I1_D4_IRON"], previous["decisions"]["tap_iron"]["D4_CATBOOST_SHALLOW"])
        bootstrap = {}
        for target, route in selected.items():
            reference = spec["reference_by_target"][target]
            if route == reference:
                continue
            distribution, intervals = sensitivity(train, target, predictions[target][route], predictions[target][reference], spec["bootstrap"])
            distribution.to_csv(output / "selection" / f"bootstrap-{target}.csv", index=False, mode="x")
            bootstrap[target] = {"route": route, "reference": reference, "intervals": intervals,
                                 "fraction_mean_delta_negative": float((distribution["mean"] < 0).mean()),
                                 "shared_ID_indices_across_splits": True, "scope": "fixed OOF composition, not retraining or hidden-test uncertainty"}
        anchor_J = {str(s): sum(metrics[t][spec["reference_by_target"][t]][str(s)]["wmape"] for t in TARGETS)/2 for s in spec["split_seeds"]}
        write_json(output / "selection/summary.json", {"metrics": metrics, "decisions": decisions, "selected": selected,
                   "anchor_J_by_split": anchor_J, "anchor_J_mean": float(np.mean(list(anchor_J.values()))),
                   "bootstrap": bootstrap, "single_slot_ranking": single_slot_ranking(metrics, selected, spec),
                   "new_cv_fits": 0, "reused_development_splits": True, "I1_previous_eligibility_reproduced": True})
        subprocess.run([sys.executable, "-m", "bf_tap_r2.v2_checkpoint", "verify", "--output", str(output)], check=True, cwd=root)
        print((output / "selection/summary.json").read_text(), flush=True)
    except Exception as exc:
        write_json(output / "FAILED_EVALUATION.json", {"error": str(exc)})
        raise


def verify(root, output):
    spec = verify_identity(root, output)["spec"]
    train, metrics, predictions = recover_oof(root, output, verify_saved=True)
    saved = json.loads((output / "selection/summary.json").read_text())
    numeric_equal(metrics, saved["metrics"])
    decisions, selected = choose_targetwise(metrics, spec)
    numeric_equal(decisions, saved["decisions"])
    if selected != saved["selected"] or single_slot_ranking(metrics, selected, spec) != saved["single_slot_ranking"]:
        raise ValueError("Selection/ranking mismatch")
    for t, b in saved["bootstrap"].items():
        distribution, intervals = sensitivity(train, t, predictions[t][b["route"]], predictions[t][b["reference"]], spec["bootstrap"])
        np.testing.assert_allclose(distribution, pd.read_csv(output / "selection" / f"bootstrap-{t}.csv"), rtol=1e-10, atol=1e-14)
        numeric_equal(intervals, b["intervals"])
    write_json(output / "selection/independent_verification.json", {"status": "PASS", "old_C2_D4_B3_recovered": True,
               "model_prefix_rules_and_same_fold_members_checked": True, "metrics_selection_bootstrap_reproduced": True,
               "new_cv_fits": 0, "new_full_fits": 0})


def member_rule(root, path, seed, source_count, end):
    return {"source_model": str(path.relative_to(root)), "source_model_sha256": digest(path), "model_seed": seed,
            "source_tree_count": source_count, "ntree_start": 0, "ntree_end": end,
            "input_transform": "bf_tap_r2.models.inputs(categorical_strings=True)", "target_restore": "multiply target_scale_"}


def predict_rules(root, rules, frame):
    if rules["aggregation"] != "arithmetic_mean" or rules["member_order"] != [m["model_seed"] for m in rules["members"]]:
        raise ValueError("Ensemble rule mismatch")
    values = []
    for rule in rules["members"]:
        path = root / rule["source_model"]
        if digest(path) != rule["source_model_sha256"]:
            raise ValueError("Source model changed")
        model = joblib.load(path)
        if model.estimator_.tree_count_ != rule["source_tree_count"] or model.config["execution"]["model_seed"] != rule["model_seed"]:
            raise ValueError("Source model seed/count mismatch")
        with threadpool_limits(limits=1):
            forward = prefix_predict(model, frame, rule["ntree_start"], rule["ntree_end"])
            reverse = prefix_predict(model, frame.iloc[::-1], rule["ntree_start"], rule["ntree_end"])[::-1]
            chunks = np.concatenate([prefix_predict(model, f, rule["ntree_start"], rule["ntree_end"]) for f in (frame.iloc[:1], frame.iloc[1:111], frame.iloc[111:])])
        np.testing.assert_allclose(forward, reverse, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(forward, chunks, rtol=1e-12, atol=1e-10)
        values.append(forward)
    return np.mean(np.stack(values), axis=0)


def release(root, output):
    manifest = verify_identity(root, output)
    spec = manifest["spec"]
    if json.loads((output / "selection/independent_verification.json").read_text())["status"] != "PASS":
        raise ValueError("Independent evaluation required")
    summary = json.loads((output / "selection/summary.json").read_text())
    test = load_v2(root / "复赛_test", "test", 322)
    previous_spec = yaml.safe_load((root / spec["previous_config"]).read_text())
    legacy = yaml.safe_load((root / previous_spec["legacy_catboost_config"]).read_text())
    anchor = (root / manifest["reference_csv"]).read_bytes()
    records = {}
    scheduled = [r["target"] for r in summary["single_slot_ranking"][:spec["platform"]["release_slots_this_batch"]]]
    for target, route in summary["selected"].items():
        if route == spec["reference_by_target"][target] or target not in scheduled:
            continue
        if target == "tap_iron":
            train = load_v2(root / "复赛_train", "train", 2754)
            path = output / "models/full-D4-tap_iron.joblib"
            fit_once(new_model("D4", previous_spec, legacy), train, train[target], path, output, "full",
                     {"target": target, "member": "D4", "model_seed": 42}, 1)
            members = [member_rule(root, path, 42, 3000, 3000)]
            if route == "I2_C2_D4_IRON_EQUAL":
                original = root / spec["original_c2_run"]
                old = json.loads((original / "release/models.json").read_text())[target]
                members.insert(0, member_rule(root, original / "release" / old["model"], 42, 1500, 1500))
            name = "V22_I_ONLY"
        else:
            members = [member_rule(root, root / m["path"], int(m["name"].split("_")[-1]), 1500, 1000) for m in manifest["time_members"]]
            name = "V22_T_ONLY"
        rules = {"members": members, "member_order": [m["model_seed"] for m in members],
                 "aggregation": "arithmetic_mean", "target": target, "candidate": route}
        prediction = predict_rules(root, rules, test)
        folder = output / "release" / name
        folder.mkdir(exist_ok=False)
        package(folder, isolated_payload(anchor, test.sample_id, target, prediction), test.sample_id)
        records[name] = {"rules": rules, "package": str((folder / "Luqhhh_bf_tap_predict_round2.zip").relative_to(root)),
                         "zip_sha256": digest(folder / "Luqhhh_bf_tap_predict_round2.zip")}
    write_json(output / "release/manifest.json", {"reference_csv": manifest["reference_csv"],
               "reference_csv_sha256": digest(root / manifest["reference_csv"]), "reference_zip_sha256": spec["reference"]["zip_sha256"],
               "reference_score_user_reported": 96.0982, "packages": records, "combined_generated": False,
               "scheduled_targets": scheduled, "remaining_quota_user_reported": 1, "upload_actor": "user",
               "combined_requires": "Both isolated scores > 96.0982 and quota available", "agent_uploads": 0})
    subprocess.run([sys.executable, "-m", "bf_tap_r2.v2_checkpoint", "infer", "--output", str(output)], check=True, cwd=root)
    for name in records:
        folder = output / "release" / name
        if (folder / "cold.csv").read_bytes() != (folder / "result.csv").read_bytes():
            raise ValueError("Cold release byte mismatch")
    verify_identity(root, output)
    full_fits = sum(json.loads(line)["event"] == "start" for line in (output / "fit_ledger.jsonl").read_text().splitlines()) if (output / "fit_ledger.jsonl").exists() else 0
    write_json(output / "COMPLETE.json", {"G0": "PASS", "new_cv_fits": 0, "new_full_fits": full_fits,
               "selected": summary["selected"], "packages": list(records), "prefix_rules_bound_to_release": True,
               "label_free_cold_inference_byte_identical": True, "source_models_and_anchor_unchanged": True,
               "combined_generated": False, "agent_uploads": 0})
    print((output / "COMPLETE.json").read_text(), flush=True)


def infer(root, output):
    sys.addaudithook(deny_training_reads)
    manifest = json.loads((output / "release/manifest.json").read_text())
    base = root / manifest["reference_csv"]
    if digest(base) != manifest["reference_csv_sha256"]:
        raise ValueError("Anchor CSV changed")
    test = load_v2(root / "复赛_test", "test", 322)
    for name, record in manifest["packages"].items():
        pred = predict_rules(root, record["rules"], test)
        payload = isolated_payload(base.read_bytes(), test.sample_id, record["rules"]["target"], pred)
        with (output / "release" / name / "cold.csv").open("xb") as f:
            f.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["evaluate", "verify", "release", "infer"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"evaluate": evaluate, "verify": verify, "release": release, "infer": infer}[args.command](Path.cwd(), args.output.resolve())


if __name__ == "__main__":
    main()
