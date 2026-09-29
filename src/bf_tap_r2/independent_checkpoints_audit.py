"""Fresh-process partition, model, checkpoint and gain audit; zero fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .component_regularization import ComponentRegressor
from .component_regularization_audit import verify_saved
from .component_regularization_run import RECIPE, outputs
from .data import FEATURES, TARGETS
from .ensemble_gain_audit import audit as arithmetic_audit
from .independent_checkpoints import native_parts, clean_query, select_epochs
from .independent_checkpoints_run import SPEC, load_spec, private_path, cache_manifest
from .v3_6_networks import NumericPreprocessor
from .v5_library import fold_vector, load_v5_training_frame
from .v5_spec import load_v5_spec
from .v5_resolution import paired_summary
from .v7_periodic import digest, file_hash, write_new
from .v49_run import verify_hashes, verified_unit, unit_id, check_runtime


def cold_check(model, query, expected, atol):
    np.testing.assert_array_equal(model.predict(query), expected)
    variants = [model.predict(query.iloc[::-1])[::-1],
                np.concatenate([model.predict(query.iloc[i:i+37]) for i in range(0, len(query), 37)])]
    diff = max(float(np.max(np.abs(v-expected))) for v in variants)
    if diff > atol: raise ValueError("Cold order/chunk tolerance failed")
    return diff


def verify_checkpoint(path, fitting, y, settings):
    model = ComponentRegressor.load(path); saved = model.saved
    if saved["settings"] != settings or saved["recipe"] != RECIPE or saved["arm"] != "BASE":
        raise ValueError("Checkpoint model identity")
    if saved["trace"]["fit_ids_digest"] != digest(fitting.sample_id.tolist()):
        raise ValueError("Checkpoint fit IDs")
    if saved["preprocessing"] != NumericPreprocessor(structure="raw_tabm").fit(fitting).metadata():
        raise ValueError("Checkpoint preprocessing leakage")
    np.testing.assert_array_equal(saved["mean"], y.mean(axis=0))
    np.testing.assert_array_equal(saved["std"], y.std(axis=0))
    if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all()) for v in saved["state"].values()):
        raise ValueError("Checkpoint float32 identity")
    return model


def verify_epoch_unit(directory, training, calibration, background, target, settings):
    mask, fitting, expected_calibration = native_parts(training, settings["inner_seed"])
    if list(calibration.sample_id) != list(expected_calibration.sample_id):
        raise ValueError("Calibration partition changed")
    selection = json.loads((directory/"selection.json").read_text())
    if selection["fit_ids_digest"] != digest(fitting.sample_id.tolist()) or selection["calibration_ids_digest"] != digest(calibration.sample_id.tolist()):
        raise ValueError("Selector partition identity")
    history = selection["history"]
    y = training[outputs(target)].to_numpy()
    with np.load(directory/"epoch-predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["calibration_ids"], calibration.sample_id.to_numpy(dtype=str))
        preds = saved["standardized"].copy()
    if preds.shape != (len(history), len(calibration), len(outputs(target))) or not np.isfinite(preds).all():
        raise ValueError("Every native epoch must have calibration predictions")
    mean, std = y[mask].mean(axis=0), y[mask].std(axis=0)
    vy = torch.as_tensor((y[~mask]-mean)/std, dtype=torch.float32)
    for i, row in enumerate(history):
        native = float((torch.as_tensor(preds[i])-vy).abs().mean())
        raw = preds[i].astype(float)*std+mean
        target_error = float(np.abs(y[~mask, 0]-raw[:, 0]).mean())
        composed_error = float(np.abs(y[~mask, 0]-background-.5*raw[:, 0]).mean())
        if (native, target_error, composed_error) != (row["validation_mae"], row["target_mae"], row["compose_mae"]):
            raise ValueError("Epoch metric arithmetic changed")
        batches = int(np.ceil(len(fitting)/settings["batch_size"]))
        if row["updates"] != batches or row["gradient_evaluations"] != batches:
            raise ValueError("Native optimizer update count")
    expected = select_epochs(history, settings)
    if expected != selection["epochs"]: raise ValueError("Checkpoint selection mismatch")
    for arm, epoch in expected.items():
        model = verify_checkpoint(directory/f"selection-{arm}.pt", fitting, y[mask], settings)
        if model.saved["trace"]["history"] != history or model.saved["trace"]["selected_epoch"] != epoch:
            raise ValueError("Selected state trace mismatch")
        vx, vc = model._inputs(clean_query(calibration))
        with torch.no_grad(): actual = model.model_(vx, vc).mean(1).numpy()
        np.testing.assert_array_equal(actual, preds[epoch-1])
    return expected


def audit(root, output):
    root = Path(root).resolve(); out = private_path(root, output); spec = load_spec(root)
    manifest = json.loads((out/"manifest.json").read_text()); body = dict(manifest); body.pop("identity")
    if digest(body) != manifest["identity"] or manifest["spec_sha256"] != file_hash(root/SPEC):
        raise ValueError("Run manifest identity")
    verify_hashes(root, manifest["source_hashes"]); verify_hashes(root, manifest["data_hashes"])
    if check_runtime(spec) != manifest["versions"]: raise ValueError("Frozen runtime")
    native_cache = cache_manifest(root, spec)[0] if not manifest["development"] else None
    summary = json.loads((out/"summary.json").read_text()); queue = manifest["queue"]
    frame = load_v5_training_frame(root)
    gains = {(r["target"], r["arm"]): [] for r in summary["records"]}
    costs = dict(selector_fits=0, refit_fits=0, reused_native_units=0, factory_calls=0, component_pipeline_fits=0)
    models, native_replays, differences = 0, 0, []
    for seed in manifest["seeds"]:
        fv = fold_vector(root, frame, seed, load_v5_spec(root))
        if digest(fv.tolist()) != manifest["fold_hashes"][str(seed)]: raise ValueError("Outer split identity")
        base = {t: np.full(len(frame), np.nan) for t in TARGETS}
        columns = {key: np.full(len(frame), np.nan) for key in gains}
        for fold in range(5):
            training = frame.loc[fv != fold].reset_index(drop=True)
            query = clean_query(frame.loc[fv == fold]).reset_index(drop=True)
            rd = out/f"reference-s{seed}-f{fold}"
            verified_unit(rd, unit_id(manifest, rd.name)); ref = dict(np.load(rd/"predictions.npz", allow_pickle=False))
            np.testing.assert_array_equal(ref["query_ids"], query.sample_id.to_numpy(dtype=str))
            rmeta = json.loads((rd/"metadata.json").read_text())
            if rmeta["fit_ids_digest"] != digest(training.sample_id.tolist()): raise ValueError("Outer reference training IDs")
            np.testing.assert_array_equal(ref["tap_iron"], .5*ref["v36_iron"]+.5*ref["v12_iron"])
            np.testing.assert_array_equal(ref["tap_time_len"], .2*ref["v36_time"]+.3*ref["n_time"]+.5*ref["v7_time"])
            if "reused_from" not in rmeta:
                costs["factory_calls"] += 1; costs["component_pipeline_fits"] += 32
            for t in TARGETS: base[t][fv == fold] = ref[t]
            background = None
            if queue == "E-COMPOSE":
                bd = out/f"inner-reference-s{seed}-f{fold}"
                verified_unit(bd, unit_id(manifest, bd.name))
                bm = json.loads((bd/"metadata.json").read_text())
                _, fitting, calibration = native_parts(training)
                if bm["fit_ids_digest"] != digest(fitting.sample_id.tolist()) or bm["query_ids_digest"] != digest(calibration.sample_id.tolist()) or bm["query_labels_received"]:
                    raise ValueError("Inner reference leakage/partition mismatch")
                background = dict(np.load(bd/"predictions.npz", allow_pickle=False))
                np.testing.assert_array_equal(background["query_ids"], calibration.sample_id.to_numpy(dtype=str))
                np.testing.assert_array_equal(background["tap_iron"], .5*background["v36_iron"])
                np.testing.assert_array_equal(background["tap_time_len"], .2*background["v36_time"]+.3*background["n_time"])
                costs["factory_calls"] += 1; costs["component_pipeline_fits"] += 30
            for target in TARGETS:
                directory = out/f"{target}-{queue}-s{seed}-f{fold}"
                if not directory.exists(): continue
                verified_unit(directory, unit_id(manifest, directory.name))
                for name, sha in json.loads((directory/"nested-hashes.json").read_text()).items():
                    if file_hash(directory/name) != sha: raise ValueError("Nested model artifact changed")
                meta = json.loads((directory/"metadata.json").read_text())
                if meta["fit_ids_digest"] != digest(training.sample_id.tolist()): raise ValueError("Candidate fit IDs")
                mask, fitting, calibration = native_parts(training); y = training[outputs(target)].to_numpy()
                saved = dict(np.load(directory/"predictions.npz", allow_pickle=False))
                np.testing.assert_array_equal(saved["query_ids"], query.sample_id.to_numpy(dtype=str))
                if queue == "DE3":
                    predictions = []
                    for ts in spec["training_seeds"]:
                        sub = directory/f"training-seed-{ts}"
                        verified_unit(sub, unit_id(manifest, str(sub.relative_to(out))))
                        settings = dict(spec["training"][target], random_seed=ts)
                        mechanisms = ComponentRegressor.load(sub/"selection.pt").saved["mechanisms"]
                        selector = verify_saved(sub/"selection.pt", fitting, y[mask], "BASE", settings, mechanisms, calibration)
                        model = verify_saved(sub/"refit.pt", training, y, "BASE", settings, mechanisms,
                            expected_epoch=selector.saved["trace"]["selected_epoch"])
                        pred = dict(np.load(sub/"predictions.npz", allow_pickle=False))
                        np.testing.assert_array_equal(pred["query_ids"], query.sample_id.to_numpy(dtype=str))
                        differences.append(cold_check(model, query, pred["prediction"], spec["preflight"]["cold_predict_atol"]))
                        np.testing.assert_array_equal(saved[f"seed_{ts}"], pred["prediction"][:, 0])
                        predictions.append(pred["prediction"][:, 0]); models += 2
                        if (sub/"reuse.json").exists():
                            reuse = json.loads((sub/"reuse.json").read_text())
                            source = Path(reuse["source"])
                            if file_hash(source/"complete.json") != reuse["source_complete_sha256"]:
                                raise ValueError("Original native reuse source changed")
                            for name in ("selection.pt", "refit.pt", "predictions.npz", "metadata.json"):
                                if file_hash(source/name) != file_hash(sub/name): raise ValueError("Native reuse is not byte exact")
                            costs["reused_native_units"] += 1
                        else: costs["selector_fits"] += 1; costs["refit_fits"] += 1
                    np.testing.assert_array_equal(saved["DE3"], np.mean(predictions, axis=0))
                    np.testing.assert_array_equal(saved["seed_42"], ref["v12_iron" if target == "tap_iron" else "v7_time"])
                    native_replays += 1
                else:
                    epochs = verify_epoch_unit(directory, training, calibration, background[target], target, spec["training"][target])
                    if native_cache is not None:
                        original = ComponentRegressor.load(native_cache/f"{target}-BASE-s{seed}-f{fold}/selection.pt")
                        replay = ComponentRegressor.load(directory/"selection-E-NATIVE.pt")
                        if epochs["E-NATIVE"] != original.saved["trace"]["selected_epoch"]:
                            raise ValueError("Independent native epoch replay failed")
                        for name, tensor in original.saved["state"].items():
                            if not torch.equal(tensor, replay.saved["state"][name]):
                                raise ValueError("Independent native selector state replay failed")
                    costs["selector_fits"] += 1; models += 3
                    seen = set()
                    for arm, info in meta["refit_by_arm"].items():
                        path = directory/info["path"]
                        if file_hash(path) != info["sha256"] or info["epoch"] != epochs[arm]: raise ValueError("Refit checkpoint identity")
                        if info["path"] not in seen:
                            seen.add(info["path"])
                            if path.parent == directory: costs["reused_native_units"] += 1
                            else: costs["refit_fits"] += 1
                            models += 1
                        mechanisms = ComponentRegressor.load(path).saved["mechanisms"]
                        model = verify_saved(path, training, y, "BASE", spec["training"][target], mechanisms, expected_epoch=epochs[arm])
                        differences.append(cold_check(model, query, saved[f"full-{arm}"], spec["preflight"]["cold_predict_atol"]))
                        np.testing.assert_array_equal(model.predict(query)[:, 0], saved[arm])
                    np.testing.assert_array_equal(saved["E-NATIVE"], ref["v12_iron" if target == "tap_iron" else "v7_time"]); native_replays += 1
                old_component = ref["v12_iron" if target == "tap_iron" else "v7_time"]
                for key in columns:
                    if key[0] == target: columns[key][fv == fold] = ref[target]+.5*(saved[key[1]]-old_component)
        for row in summary["records"]:
            key = row["target"], row["arm"]; target = key[0]; y = frame[target].to_numpy(); pred = columns[key]
            if not np.isfinite(pred).all(): raise ValueError("Incomplete OOF coverage")
            gain = float(50*(np.abs(y-base[target]).sum()-np.abs(y-pred).sum())/np.abs(y).sum())
            other = next(t for t in TARGETS if t != target)
            score = 100-50*(np.abs(y-pred).sum()/np.abs(y).sum()+np.abs(frame[other].to_numpy()-base[other]).sum()/np.abs(frame[other].to_numpy()).sum())
            if abs(gain-row["seeds"][str(seed)]["gain"]) > 1e-10 or abs(score-row["seeds"][str(seed)]["candidate_score"]) > 1e-10:
                raise ValueError("Independent complete-split score mismatch")
            gains[key].append(gain)
    if costs != summary["costs"]: raise ValueError(f"Fit accounting mismatch: {costs} versus {summary['costs']}")
    selected = {}
    for target in TARGETS:
        eligible = [arm for t, arm in gains if t == target and arm != "E-NATIVE" and min(gains[t, arm]) > 0
                    and np.mean(gains[t, arm]) >= spec["promotion"]["development_mean_gain_ge"]]
        selected[target] = min(eligible, key=lambda a: (-np.mean(gains[target, a]), spec["tie_preference_by_target"][target].index(a))) if eligible else None
    if selected != summary["selected_for_confirmation"]: raise ValueError("Independent finalist selection mismatch")
    if queue == "E-COMPOSE":
        for row in summary["records"]:
            if row["arm"] != "E-COMPOSE": continue
            target = row["target"]
            diff = np.mean(gains[target, "E-COMPOSE"])-np.mean(gains[target, "E-TARGET"])
            supported = min(gains[target, "E-COMPOSE"]) > 0 and diff > 0 and all(
                c > t for c, t in zip(gains[target, "E-COMPOSE"], gains[target, "E-TARGET"]))
            if abs(diff-row["mean_gain_vs_target_only"]) > 1e-10 or supported != row["composition_mechanism_supported"]:
                raise ValueError("Composition explanation mismatch")
    if queue == "DE3":
        arithmetic = arithmetic_audit(out/"authorized-training-oof.csv", tuple(str(s) for s in manifest["seeds"]))
        if arithmetic != summary["arithmetic_audit"]: raise ValueError("DE3 cancellation audit mismatch")
        for r in summary["records"]:
            if abs(arithmetic["targets"][r["target"]]["mean_split_gain"]-r["paired"]["mean"]) > 1e-10:
                raise ValueError("Averaged prediction gain mismatch")
    if manifest["development"]:
        dev = root/manifest["development"]
        if file_hash(dev/"summary.json") != manifest["development_summary_sha256"]: raise ValueError("Development changed")
        before = json.loads((dev/"summary.json").read_text())
        for decision in summary["four_seed_decisions"]:
            t, a = decision["target"], decision["arm"]
            original = next(r for r in before["records"] if (r["target"], r["arm"]) == (t, a))
            values = [v["gain"] for v in original["seeds"].values()]+gains[t, a]
            paired = paired_summary(values)
            score = np.mean([v["candidate_score"] for v in original["seeds"].values()])
            passed = len(values) == 4 and min(values) > 0 and paired["lcb95"] > 0 and score >= spec["promotion"]["local_working_gate"]
            if passed != decision["promoted"]: raise ValueError("Four-split decision mismatch")
    report = {"status": "passed", "queue": queue, "cold_models": models, "native_replays": native_replays,
              "full_batch_cold_difference": 0.0, "maximum_order_chunk_difference": max(differences),
              "partition_provenance_verified": True, "selection_verified": True, "costs_verified": costs,
              "summary_sha256": file_hash(out/"summary.json"), "manifest_sha256": file_hash(out/"manifest.json"),
              "auditor_sha256": file_hash(Path(__file__)), "new_fits": 0, "packages": 0}
    write_new(out/"audit.json", report); print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); audit(Path.cwd(), args.output)
