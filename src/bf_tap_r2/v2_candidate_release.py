"""Authorized isolated T1 time and AJ iron release; retain the existing V22 ZIP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import joblib
import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from .audit import digest, write_json
from .data import TARGETS
from .normalized_models import JointSnapshotRegressor
from .submission import ZIP_NAME, deny_training_reads, package, validate_result
from .v2_checkpoint import member_rule, prefix_predict
from .v2_refinement import fit_once, isolated_payload
from .v2_release import load_v2
from .v2_robust_joint import checked_manifest, check_packages, id_digest


def prefix_member(root, rule, frame):
    path = root / rule["source_model"]
    if digest(path) != rule["source_model_sha256"]:
        raise ValueError("Frozen member SHA mismatch")
    model = joblib.load(path)
    if model.estimator_.tree_count_ != rule["source_tree_count"] or model.config["execution"]["model_seed"] != rule["model_seed"]:
        raise ValueError("Frozen member seed/tree count mismatch")
    return prefix_predict(model, frame, rule["ntree_start"], rule["ntree_end"])


def predict_candidate(root, rules, frame):
    if rules["kind"] == "T1":
        if rules["aggregation"] != "arithmetic_mean" or rules["member_order"] != [42, 2026, 2027] or [m["model_seed"] for m in rules["members"]] != rules["member_order"]:
            raise ValueError("T1 member order mismatch")
        if any((m["ntree_start"], m["ntree_end"], m["source_tree_count"]) != (0, 1000, 1500) for m in rules["members"]):
            raise ValueError("T1 prefix mismatch")
        return np.mean(np.stack([prefix_member(root, m, frame) for m in rules["members"]]), axis=0)
    if rules["kind"] != "AJ" or rules["weights"] != [.5, .5] or rules["joint_target_index"] != 0:
        raise ValueError("Frozen AJ iron rule mismatch")
    if rules["target_order"] != list(TARGETS) or tuple(rules["c2_member"][k] for k in ("ntree_start", "ntree_end", "source_tree_count")) != (0, 1500, 1500):
        raise ValueError("AJ target order/C2 tree range mismatch")
    path = root / rules["joint_model"]
    if digest(path) != rules["joint_model_sha256"]:
        raise ValueError("Joint model SHA mismatch")
    model = joblib.load(path)
    if (not isinstance(model, JointSnapshotRegressor) or list(model.target_fields_) != list(TARGETS)
            or model.estimator_.tree_count_ != 1500 or model.parameters != rules["joint_parameters"]
            or model.actual_parameters_ != rules["joint_actual_parameters"]
            or model.target_scales_.tolist() != rules["joint_scales"]):
        raise ValueError("Joint model identity/target transform mismatch")
    c2 = prefix_member(root, rules["c2_member"], frame)
    return .5 * c2 + .5 * model.predict(frame)[:, 0]


def checked_predictions(root, rules, frame):
    with threadpool_limits(limits=1):
        prediction = predict_candidate(root, rules, frame)
        reverse = predict_candidate(root, rules, frame.iloc[::-1])[::-1]
        batches = np.concatenate([predict_candidate(root, rules, frame.iloc[i:i+111]) for i in range(0, len(frame), 111)])
        single = predict_candidate(root, rules, frame.iloc[:1])
    if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
        raise ValueError("Invalid release prediction")
    np.testing.assert_array_equal(prediction, reverse)
    np.testing.assert_array_equal(prediction, batches)
    np.testing.assert_array_equal(prediction[:1], single)
    return prediction


def score_scenarios(v22, v23, reported_score):
    """OOF scores plus arithmetic transfer scenarios, NOT calibrated forecasts."""
    a = v22["metrics"]
    iron, time = "tap_iron", "tap_time_len"
    mean = lambda scores: float(np.mean([x["wmape"] for x in scores.values()]))
    current_i = mean(a[iron]["C2_seed42_full1500"])
    current_t = mean(a[time]["B3_seeds42_2026_2027_full1500"])
    anchor_j = (current_i+current_t)/2
    pairs = {"V22_I_ONLY": (mean(a[iron]["I2_C2_D4_IRON_EQUAL"]), current_t),
             "V22_T_ONLY": (current_i, mean(a[time]["T1_B3_PREFIX1000"])),
             "V23_AJ_I_ONLY": (mean(v23["metrics"][iron]["AJ"]), current_t)}
    return {name: {"iron_wmape": i, "time_wmape": t, "oof_reference_score": 100*(1-(i+t)/2),
                   "oof_score_gain_vs_B": 100*(anchor_j-(i+t)/2),
                   "hypothetical_equal_gain_transfer_score": reported_score+100*(anchor_j-(i+t)/2),
                   "interpretation": "Arithmetic scenario assuming full OOF gain transfers to platform; not a forecast or confidence interval"}
            for name, (i, t) in pairs.items()}


def infer(root, output):
    sys.addaudithook(deny_training_reads)
    manifest = json.loads((output / "release_manifest.json").read_text())
    base = root / manifest["reference_csv"]
    if digest(base) != manifest["reference_csv_sha256"]:
        raise ValueError("Parent CSV identity mismatch")
    test = load_v2(root / "复赛_test", "test", 322)
    for name, record in manifest["packages"].items():
        prediction = checked_predictions(root, record["rules"], test)
        payload = isolated_payload(base.read_bytes(), test.sample_id, record["target"], prediction)
        with (output / "release" / name / "cold.csv").open("xb") as stream:
            stream.write(payload)


def release(root, output):
    if not output.is_relative_to(root / "local/runs/round2-v2.3"):
        raise ValueError("Fresh private release directory required")
    config_path = root / "configs/round2_v2_3/release_candidates.yaml"
    spec = yaml.safe_load(config_path.read_text())
    source = root / spec["source_run"]
    frozen = checked_manifest(root, source)
    if json.loads((source / "selection/independent_verification.json").read_text())["status"] != "PASS":
        raise ValueError("Independent OOF verification required")
    v23 = json.loads((source / "selection/summary.json").read_text())
    v22path = root / spec["v22_run"]
    v22 = json.loads((v22path / "selection/summary.json").read_text())
    if (v22["selected"]["tap_time_len"] != "T1_B3_PREFIX1000"
            or not any(r["target"] == "tap_iron" and r["candidate"] == "AJ" for r in v23["tiers"]["formal_selected"])):
        raise ValueError("Candidate eligibility mismatch")
    base = root / spec["reference_csv"]
    parent_zip = root / spec["reference_zip"]
    if digest(parent_zip) != spec["reference_zip_sha256"] or digest(root / spec["c2_iron"]) != spec["c2_iron_sha256"]:
        raise ValueError("Parent ZIP/C2 identity mismatch")
    with zipfile.ZipFile(parent_zip) as archive:
        if archive.read("result.csv") != base.read_bytes():
            raise ValueError("Parent CSV differs from ZIP")
    for record in spec["packages"].values():
        if Path(record["desktop"]).exists():
            raise FileExistsError("Desktop destination already exists: " + record["desktop"])
    test = load_v2(root / "复赛_test", "test", 322)
    anchor_rows = validate_result(base.read_bytes(), test.sample_id)
    t1 = json.loads((v22path / "selection/deferred_time_full_rules.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "release").mkdir()
    write_json(output / "source_manifest.json", {"source_run": spec["source_run"], "source_manifest_sha256": digest(source / "manifest.json"),
        "verified_summary_sha256": digest(source / "selection/summary.json"), "v22_summary_sha256": digest(v22path / "selection/summary.json"),
        "config_sha256": digest(config_path), "source_sha256": digest(Path(__file__)), "spec": spec,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()})
    try:
        training = load_v2(root / "复赛_train", "train", 2754)
        joint_parameters = dict(frozen["spec"]["common"], loss_function="MultiRMSE")
        model = JointSnapshotRegressor(joint_parameters)
        model.model_identity_ = {"route": "J1", "stage": "full", "train_ids_sha256": id_digest(training)}
        path = output / "models/full-J1-joint.joblib"
        fit_once(model, training, training[list(TARGETS)], path, output, "full", {"route": "J1", "targets": list(TARGETS)}, 1)
        np.testing.assert_allclose(model.target_scales_, training[list(TARGETS)].mean().to_numpy(), rtol=1e-14)
        write_json(output / "models/full-J1-joint.json", {"identity": model.model_identity_, "target_scales": model.target_scales_.tolist(),
                   "target_fields": list(model.target_fields_), "input_fields": list(model.input_fields_),
                   "actual_parameters": model.actual_parameters_, "model_sha256": digest(path)})
        rules = {"V22_T_ONLY": {"kind": "T1", "members": t1["members"], "member_order": t1["member_order"], "aggregation": "arithmetic_mean"},
                 "V23_AJ_I_ONLY": {"kind": "AJ", "weights": [.5, .5], "joint_target_index": 0,
                    "c2_member": member_rule(root, root / spec["c2_iron"], 42, 1500, 1500),
                    "joint_model": str(path.relative_to(root)), "joint_model_sha256": digest(path),
                    "joint_parameters": model.parameters, "joint_actual_parameters": model.actual_parameters_,
                    "joint_scales": model.target_scales_.tolist(), "target_order": list(TARGETS)}}
        packages = {}
        for name, rule in rules.items():
            prediction = checked_predictions(root, rule, test)
            target = spec["packages"][name]["target"]
            payload = isolated_payload(base.read_bytes(), test.sample_id, target, prediction)
            folder = output / "release" / name
            folder.mkdir()
            package(folder, payload, test.sample_id)
            rows = validate_result(payload, test.sample_id)
            changes = {"pred_"+t: sum(a["pred_"+t] != b["pred_"+t] for a, b in zip(anchor_rows, rows)) for t in TARGETS}
            other = next(t for t in TARGETS if t != target)
            if changes["pred_"+other] != 0:
                raise ValueError("Untouched column changed")
            packages[name] = {"target": target, "rules": rule, "changed_rows": changes,
                              "zip_sha256": digest(folder / ZIP_NAME), "csv_sha256": digest(folder / "result.csv"),
                              "desktop": spec["packages"][name]["desktop"]}
        write_json(output / "release_manifest.json", {"reference_csv": spec["reference_csv"], "reference_csv_sha256": digest(base),
                   "reference_zip_sha256": spec["reference_zip_sha256"], "packages": packages, "upload_actor": "user", "agent_uploads": 0,
                   "first_package": "V22_I_ONLY", "combined_generated": False})
        subprocess.run([sys.executable, "-m", "bf_tap_r2.v2_candidate_release", "infer", "--output", str(output)], cwd=root, check=True)
        for name, record in packages.items():
            folder = output / "release" / name
            if (folder / "cold.csv").read_bytes() != (folder / "result.csv").read_bytes():
                raise ValueError("Cold inference byte mismatch")
        # All checks pass before publishing any desktop copy.
        checked_manifest(root, source)
        for name, record in packages.items():
            desktop = Path(record["desktop"])
            desktop.mkdir(parents=True, exist_ok=False)
            for filename in (ZIP_NAME, "result.csv"):
                shutil.copy2(output / "release" / name / filename, desktop / filename)
            if digest(desktop / ZIP_NAME) != record["zip_sha256"]:
                raise ValueError("Desktop copy SHA mismatch")
        original_packages = check_packages(root, frozen["spec"])
        write_json(output / "score_scenarios.json", score_scenarios(v22, v23, spec["reference_platform_score_user_reported"]))
        write_json(output / "COMPLETE.json", {"G0": "PASS", "new_cv_fits": 0, "new_full_fits": 1, "new_packages": list(packages),
                   "label_free_cold_inference_byte_identical": True, "reverse_batch_single_exact": True,
                   "untouched_column_strings_preserved": True, "old_packages_unchanged": original_packages, "platform_uploads": 0})
    except Exception as exc:
        write_json(output / "FAILED.json", {"error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["release", "infer"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (release if args.action == "release" else infer)(Path.cwd().resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
