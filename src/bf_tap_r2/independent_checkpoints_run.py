"""Append-only, serial DE3 then E-COMPOSE development and gated confirmation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import yaml

from .candidate_tiers import classify_candidates
from .component_regularization import ComponentRegressor, replacement
from .component_regularization_audit import verify_saved
from .component_regularization_run import RECIPE, outputs
from .data import TARGETS
from .ensemble_gain_audit import REQUIRED, MEMBERS, audit as arithmetic_audit
from .independent_checkpoints import (ARMS, AccountedRegressor, EpochSelector, native_parts, clean_query,
                                      fit_background, fresh_refit, solver_accounting)
from .v5_library import load_v5_training_frame, fold_vector
from .v5_spec import load_v5_spec
from .v5_resolution import paired_summary
from .v7_periodic import digest, file_hash, write_new
from .v49_run import (append_event, check_runtime, verify_hashes, verify_reference_cache,
                      verified_unit, unit_id, read_reference, metric_detail)

SPEC = "configs/independent_ensemble_checkpoints/SPEC.yaml"
RUN_ROOT = "local/runs/independent-ensemble-checkpoints-20260929"


def load_spec(root):
    spec = yaml.safe_load((root/SPEC).read_text())
    original = yaml.safe_load((root/"configs/strong_component_regularization/SPEC.yaml").read_text())
    for key in ("training", "reference", "split_seeds", "confirmation_seeds", "folds", "runtime_versions"):
        if spec[key] != original[key]:
            raise ValueError(f"Native contract changed: {key}")
    for key in ("min_split_seeds", "all_seed_gains_positive", "seed_level_paired_lcb95_gt",
                "local_working_gate", "fold_level", "development_mean_gain_ge"):
        if spec["promotion"][key] != original["promotion"][key]:
            raise ValueError(f"Gate changed: {key}")
    if spec["training_seeds"] != [42, 104729, 130363] or spec["calibration"]["split_seed"] != 42:
        raise ValueError("Seed/partition contract changed")
    if spec["budget"]["candidate_workers"] != 4 or spec["budget"]["reference_workers"] != 4:
        raise ValueError("Four-worker admission required")
    if spec["release"] != dict(automatic_packages=False, full_data_fits=0, desktop_writes=0, agent_uploads=0):
        raise ValueError("No automatic release")
    return spec


def private_path(root, output):
    out = (root/output).resolve()
    allowed = (root/RUN_ROOT).resolve()
    if out == allowed or not out.is_relative_to(allowed):
        raise ValueError("New private run subdirectory required")
    return out


def sources(root):
    paths = list((root/"src/bf_tap_r2").glob("*.py"))+list((root/"configs").rglob("*.yaml"))
    paths += [root/"uv.lock", root/"pyproject.toml", root/"docs/independent_ensemble_checkpoints/PREREGISTRATION.md",
              root/"docs/independent_ensemble_checkpoints/SOURCE_PLAN.md",
              root/"tests/test_independent_checkpoints.py", root/"tests/test_ensemble_gain_audit.py"]
    for name in ("local/runs/round2-v3.4-ebm-and-constrained-composition/run_l1_oof.py",
                 "local/runs/round2-v3.1-directed-search/prepared-packages-r1/manifest.json",
                 "local/runs/round2-v3.4-ebm-and-constrained-composition/release-candidates-r1/01_V34_A_MECHANICAL_CONSTRAINED/manifest.json",
                 "local/runs/round2-v3.6-loss-training-and-numeric-encoding/v36-summary.json",
                 "local/runs/round2-v3.6-loss-training-and-numeric-encoding/fixed-r2-final/fit_ledger.jsonl"):
        paths.append(root/name)
    return {str(p.relative_to(root)): file_hash(p) for p in paths}


def cache_manifest(root, spec):
    cache = root/spec["native_cache"]["directory"]
    names = {"manifest.json": "manifest_sha256", "summary.json": "summary_sha256",
             "audit.json": "audit_sha256", "audit-recovery-provenance.json": "provenance_sha256"}
    for name, key in names.items():
        if file_hash(cache/name) != spec["native_cache"][key]:
            raise ValueError(f"Native cache hash changed: {name}")
    manifest = json.loads((cache/"manifest.json").read_text())
    body = dict(manifest); body.pop("identity")
    if digest(body) != manifest["identity"]:
        raise ValueError("Native manifest identity")
    audit = json.loads((cache/"audit.json").read_text())
    if audit["status"] != "passed" or audit["native_replays"] != 20 or audit["full_batch_cold_difference"] != 0:
        raise ValueError("Native cold audit required")
    if audit["summary_sha256"] != file_hash(cache/"summary.json"):
        raise ValueError("Native audited summary changed")
    verify_hashes(root, manifest["source_hashes"])
    verify_hashes(root, manifest["data_hashes"])
    if check_runtime(spec) != manifest["versions"]:
        raise ValueError("Native runtime changed")
    return cache, manifest


def verify_original_reference(root, spec):
    """Audit the old cache under its own unchanged calibration contract.

    Only outer predictions are reused. Its seed27001 calibration predictions
    cannot be used by the new seed42 selector/background partition.
    """
    original = yaml.safe_load((root/"configs/strong_component_regularization/SPEC.yaml").read_text())
    if spec["reference_cache"] != original["reference_cache"]:
        raise ValueError("Original reference cache identity changed")
    return verify_reference_cache(root, original)


def validate_native(root, spec, frame, folds):
    cache, manifest = cache_manifest(root, spec)
    for seed, fv in folds.items():
        if digest(fv.tolist()) != manifest["fold_hashes"][str(seed)]:
            raise ValueError("Native outer folds changed")
        for fold in range(5):
            training = frame.loc[fv != fold].reset_index(drop=True)
            query = clean_query(frame.loc[fv == fold]).reset_index(drop=True)
            ref = cache/f"reference-s{seed}-f{fold}"
            verified_unit(ref, unit_id(manifest, ref.name))
            metadata = json.loads((ref/"metadata.json").read_text())
            if metadata["fit_ids_digest"] != digest(training.sample_id.tolist()):
                raise ValueError("Native reference fit IDs")
            with np.load(ref/"predictions.npz", allow_pickle=False) as refs:
                np.testing.assert_array_equal(refs["query_ids"], query.sample_id.to_numpy(dtype=str))
                for target in TARGETS:
                    source = cache/f"{target}-BASE-s{seed}-f{fold}"
                    verified_unit(source, unit_id(manifest, source.name))
                    mask, fitting, calibration = native_parts(training)
                    y = training[outputs(target)].to_numpy()
                    old_spec = yaml.safe_load((root/"configs/strong_component_regularization/SPEC.yaml").read_text())
                    selector = verify_saved(source/"selection.pt", fitting, y[mask], "BASE",
                        spec["training"][target], old_spec["mechanisms"], calibration)
                    model = verify_saved(source/"refit.pt", training, y, "BASE", spec["training"][target],
                        old_spec["mechanisms"], expected_epoch=selector.saved["trace"]["selected_epoch"])
                    if model.saved["trace"]["fit_ids_digest"] != digest(training.sample_id.tolist()):
                        raise ValueError("Native model fit IDs")
                    with np.load(source/"predictions.npz", allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved["query_ids"], query.sample_id.to_numpy(dtype=str))
                        np.testing.assert_array_equal(saved["prediction"], model.predict(query))
                        np.testing.assert_array_equal(saved["prediction"][:, 0], refs["v12_iron" if target == "tap_iron" else "v7_time"])
    return cache, manifest


def finish_unit(directory, manifest):
    key = str(directory.relative_to(Path(manifest["output_directory"])))
    result = {"identity": unit_id(manifest, key), "key": key,
              "hashes": {p.name: file_hash(p) for p in directory.iterdir() if p.is_file() and p.name != "events.jsonl"}}
    write_new(directory/"complete.json", result)
    return result


def copy_native(source, destination, original, manifest):
    completed = verified_unit(source, unit_id(original, source.name))
    if completed is None:
        raise ValueError("Native reuse source missing")
    destination.mkdir(exist_ok=False)
    for p in source.iterdir():
        if p.is_file():
            shutil.copy2(p, destination/("origin-complete.json" if p.name == "complete.json" else p.name))
    write_new(destination/"reuse.json", {"source": str(source), "source_complete_sha256": file_hash(source/"complete.json"),
                                        "source_identity": completed["identity"], "new_fits": 0})
    finish_unit(destination, manifest)


def reference_units(root, out, frame, folds, spec, manifest, native=None, shared=None):
    from .v30_reference import fit_b0
    for seed, fv in folds.items():
        for fold in range(5):
            directory = out/f"reference-s{seed}-f{fold}"
            training = frame.loc[fv != fold].reset_index(drop=True)
            query = clean_query(frame.loc[fv == fold]).reset_index(drop=True)
            directory.mkdir(exist_ok=False)
            if native is not None:
                source = native/f"reference-s{seed}-f{fold}"
                values = dict(np.load(source/"predictions.npz", allow_pickle=False))
                metadata = json.loads((source/"metadata.json").read_text())
                metadata = {**metadata, "reused_from": str(source), "new_fits": 0}
            elif shared is not None:
                shared_manifest = json.loads((shared/"manifest.json").read_text())
                source = shared/directory.name
                verified_unit(source, unit_id(shared_manifest, source.name))
                values = dict(np.load(source/"predictions.npz", allow_pickle=False))
                metadata = {**json.loads((source/"metadata.json").read_text()), "reused_from": str(source), "new_fits": 0}
            else:
                with solver_accounting(directory/"solver-fits.jsonl", "confirmation_outer_reference"):
                    values, metadata = fit_b0(root, training, query, spec)
                values["query_ids"] = query.sample_id.to_numpy(dtype=str)
            if metadata["fit_ids_digest"] != digest(training.sample_id.tolist()):
                raise ValueError("Reference training identity")
            np.testing.assert_array_equal(values["query_ids"], query.sample_id.to_numpy(dtype=str))
            with (directory/"predictions.npz").open("xb") as stream:
                np.savez_compressed(stream, **values)
            write_new(directory/"metadata.json", metadata)
            append_event(out/"fit_ledger.jsonl", {"event": "complete", **finish_unit(directory, manifest),
                "kind": "outer_reference", "factory_calls": 0 if native is not None or shared is not None else 1,
                "component_pipeline_fits": 0 if native is not None or shared is not None else 32})


def background_unit(root, out, training, seed, fold, spec, manifest):
    directory = out/f"inner-reference-s{seed}-f{fold}"
    directory.mkdir(exist_ok=False)
    _, fitting, calibration = native_parts(training)
    values, metadata = fit_background(root, fitting, calibration, spec, directory)
    with (directory/"predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, **values, query_ids=calibration.sample_id.to_numpy(dtype=str))
    write_new(directory/"metadata.json", metadata)
    return {**finish_unit(directory, manifest), "kind": "inner_reference", "factory_calls": 1,
            "component_pipeline_fits": metadata["component_pipeline_fits"]}


def candidate_unit(root, out, training, query, seed, fold, target, queue, spec, manifest, native_source=None, selected_arm=None):
    directory = out/f"{target}-{queue}-s{seed}-f{fold}"
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    append_event(directory/"events.jsonl", {"event": "start", "identity": unit_id(manifest, directory.name)})
    try:
        if queue == "DE3":
            counts = {"selector_fits": 0, "refit_fits": 0, "reused_native_units": 0}
            for train_seed in spec["training_seeds"]:
                sub = directory/f"training-seed-{train_seed}"
                if train_seed == 42 and native_source is not None:
                    original = json.loads((native_source.parent/"manifest.json").read_text())
                    copy_native(native_source, sub, original, manifest)
                    counts["reused_native_units"] += 1
                    continue
                sub.mkdir(exist_ok=False)
                settings = dict(spec["training"][target], random_seed=train_seed)
                model = AccountedRegressor(RECIPE, settings, "BASE", {}, sub)
                append_event(directory/"solver-fits.jsonl", {"event": "selector_start", "training_seed": train_seed})
                model.fit(training, training[outputs(target)].to_numpy())
                prediction = model.predict(query)
                with (sub/"predictions.npz").open("xb") as stream:
                    np.savez_compressed(stream, prediction=prediction, query_ids=query.sample_id.to_numpy(dtype=str))
                write_new(sub/"metadata.json", {"model": model.metadata_, "training_seed": train_seed,
                                               "inner_seed": settings["inner_seed"]})
                counts["selector_fits"] += 1; counts["refit_fits"] += 1
                append_event(directory/"solver-fits.jsonl", {"event": "selector_and_refit_complete", "training_seed": train_seed,
                    "selector_fits": 1, "refit_fits": 1, "epochs": model.metadata_["selected_epoch"]})
                finish_unit(sub, manifest)
            predictions = []
            for train_seed in spec["training_seeds"]:
                with np.load(directory/f"training-seed-{train_seed}/predictions.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["query_ids"], query.sample_id.to_numpy(dtype=str))
                    predictions.append(saved["prediction"][:, 0])
            values = {"DE3": np.mean(predictions, axis=0),
                      **{f"seed_{s}": p for s, p in zip(spec["training_seeds"], predictions)}}
            with np.load(out/f"reference-s{seed}-f{fold}/predictions.npz", allow_pickle=False) as ref:
                np.testing.assert_array_equal(values["seed_42"], ref["v12_iron" if target == "tap_iron" else "v7_time"])
            metadata = {**counts, "training_seeds": spec["training_seeds"], "aggregation": "fixed_arithmetic_mean"}
        else:
            with np.load(out/f"inner-reference-s{seed}-f{fold}/predictions.npz", allow_pickle=False) as background:
                _, fitting, calibration = native_parts(training)
                np.testing.assert_array_equal(background["query_ids"], calibration.sample_id.to_numpy(dtype=str))
                refmeta = json.loads((out/f"inner-reference-s{seed}-f{fold}/metadata.json").read_text())
                if refmeta["fit_ids_digest"] != digest(fitting.sample_id.tolist()):
                    raise ValueError("Leaked/mismatched background fit IDs")
                selector = EpochSelector(spec["training"][target], target, directory)
                selection = selector.select(training, background[target], query)
            metadata = {"selection": selection, "selector_fits": 1, "refit_fits": 0,
                        "reused_native_units": 0, "refit_by_arm": {}}
            append_event(directory/"solver-fits.jsonl", {"event": "selector_complete", "fits": 1,
                                                        "epochs": selection["visited_epochs"]})
            values, by_epoch = {}, {}
            native = ComponentRegressor.load(native_source/"refit.pt") if native_source is not None else None
            for arm, epoch in selection["epochs"].items():
                if selected_arm is not None and arm not in {"E-NATIVE", selected_arm, "E-TARGET"}:
                    continue
                if epoch in by_epoch:
                    path, pred = by_epoch[epoch]
                elif native is not None and epoch == native.saved["trace"]["selected_epoch"]:
                    path = f"refit-epoch-{epoch}.pt"
                    shutil.copy2(native_source/"refit.pt", directory/path)
                    pred = native.predict(query)
                    metadata["reused_native_units"] += 1
                    by_epoch[epoch] = path, pred
                else:
                    sub = directory/f"refit-epoch-{epoch}"
                    sub.mkdir(exist_ok=False)
                    model = fresh_refit(training, target, spec["training"][target], epoch, sub)
                    path = f"{sub.name}/refit.pt"; pred = model.predict(query)
                    metadata["refit_fits"] += 1
                    append_event(directory/"solver-fits.jsonl", {"event": "refit_complete", "fits": 1, "epoch": epoch})
                    by_epoch[epoch] = path, pred
                metadata["refit_by_arm"][arm] = {"path": path, "sha256": file_hash(directory/path), "epoch": epoch}
                values[arm] = pred[:, 0]
                values[f"full-{arm}"] = pred
            with np.load(out/f"reference-s{seed}-f{fold}/predictions.npz", allow_pickle=False) as ref:
                np.testing.assert_array_equal(values["E-NATIVE"], ref["v12_iron" if target == "tap_iron" else "v7_time"])
            if native_source is not None:
                old = ComponentRegressor.load(native_source/"selection.pt")
                replay = ComponentRegressor.load(directory/"selection-E-NATIVE.pt")
                if selection["epochs"]["E-NATIVE"] != old.saved["trace"]["selected_epoch"]:
                    raise ValueError("Native selected epoch replay failed")
                for key, state in old.saved["state"].items():
                    import torch
                    if not torch.equal(state, replay.saved["state"][key]):
                        raise ValueError("Native selector trajectory replay failed")
                metadata["native_selector_state_exact"] = True
        with (directory/"predictions.npz").open("xb") as stream:
            np.savez_compressed(stream, **values, query_ids=query.sample_id.to_numpy(dtype=str))
        write_new(directory/"metadata.json", {**metadata, "target": target, "queue": queue, "seed": seed, "fold": fold,
            "fit_ids_digest": digest(training.sample_id.tolist()), "seconds": time.monotonic()-started})
        # Bind nested model artifacts; predictions alone never prove lineage.
        nested = {str(p.relative_to(directory)): file_hash(p) for p in directory.rglob("*") if p.is_file()}
        write_new(directory/"nested-hashes.json", nested)
        result = finish_unit(directory, manifest)
        return {**result, "kind": "candidate", "selector_fits": metadata["selector_fits"],
                "refit_fits": metadata["refit_fits"], "reused_native_units": metadata["reused_native_units"]}
    except BaseException as exc:
        append_event(directory/"events.jsonl", {"event": "failed", "error": repr(exc)})
        raise


def select_finalists(records, spec, queue):
    result = {}
    for target in TARGETS:
        eligible = [r for r in records if r["target"] == target and r["arm"] != "E-NATIVE"
                    and min(v["gain"] for v in r["seeds"].values()) > 0
                    and r["paired"]["mean"] >= spec["promotion"]["development_mean_gain_ge"]]
        result[target] = (min(eligible, key=lambda r: (-r["paired"]["mean"],
            spec["tie_preference_by_target"][target].index(r["arm"])))["arm"] if eligible else None)
    return result


def summarize(out, frame, folds, spec, queue):
    arms = ["DE3"] if queue == "DE3" else list(ARMS)
    active = [t for t in TARGETS if (out/f"{t}-{queue}-s{next(iter(folds))}-f0").exists()]
    target_arms = {}
    for t in active:
        with np.load(out/f"{t}-{queue}-s{next(iter(folds))}-f0/predictions.npz", allow_pickle=False) as saved:
            target_arms[t] = [a for a in arms if a in saved]
    baseline = {s: {t: np.full(len(frame), np.nan) for t in (*TARGETS, "v12_iron", "v7_time")} for s in folds}
    members = {(s, t, a): np.full(len(frame), np.nan) for s in folds for t in active for a in target_arms[t]}
    seed_members = {(s, t, ts): np.full(len(frame), np.nan) for s in folds for t in TARGETS for ts in spec["training_seeds"]}
    for seed, fv in folds.items():
        for fold in range(5):
            mask = fv == fold
            with np.load(out/f"reference-s{seed}-f{fold}/predictions.npz", allow_pickle=False) as ref:
                for name in baseline[seed]: baseline[seed][name][mask] = ref[name]
            for target in active:
                path = out/f"{target}-{queue}-s{seed}-f{fold}/predictions.npz"
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["query_ids"], frame.loc[mask, "sample_id"].to_numpy(dtype=str))
                    for arm in target_arms[target]: members[seed, target, arm][mask] = saved[arm]
                    if queue == "DE3":
                        for ts in spec["training_seeds"]: seed_members[seed, target, ts][mask] = saved[f"seed_{ts}"]
    records, metrics = [], {t: {"CURRENT_FIXED_REPLACEMENT": {}} for t in TARGETS}
    for target in TARGETS:
        y = frame[target].to_numpy(); old = "v12_iron" if target == "tap_iron" else "v7_time"
        other = next(t for t in TARGETS if t != target)
        for s, fv in folds.items(): metrics[target]["CURRENT_FIXED_REPLACEMENT"][str(s)] = metric_detail(y, baseline[s][target], fv, frame.spout_no.to_numpy())
        for arm in target_arms.get(target, []):
            rows = {}; metrics[target][arm] = {}
            for s, fv in folds.items():
                member = members[s, target, arm]
                if not np.isfinite(member).all(): raise ValueError("Incomplete five-fold OOF")
                pred = replacement(baseline[s][target], baseline[s][old], member)
                detail = metric_detail(y, pred, fv, frame.spout_no.to_numpy())
                metrics[target][arm][str(s)] = detail
                bw = metrics[target]["CURRENT_FIXED_REPLACEMENT"][str(s)]["wmape"]
                gain = 50*(bw-detail["wmape"])
                score = 100-50*(detail["wmape"]+np.abs(frame[other].to_numpy()-baseline[s][other]).sum()/np.abs(frame[other].to_numpy()).sum())
                rows[str(s)] = {"gain": gain, "candidate_score": score, "wmape": detail["wmape"],
                    "standalone_component_wmape": float(np.abs(y-member).sum()/np.abs(y).sum()),
                    "positive_folds_descriptive": sum(detail["by_fold"][str(f)] < metrics[target]["CURRENT_FIXED_REPLACEMENT"][str(s)]["by_fold"][str(f)] for f in range(5))}
            records.append({"target": target, "arm": arm, "seeds": rows,
                "paired": paired_summary([v["gain"] for v in rows.values()])})
    arithmetic = None
    if queue == "DE3":
        path = out/"authorized-training-oof.csv"
        with path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REQUIRED); writer.writeheader()
            for target in active:
                old = "v12_iron" if target == "tap_iron" else "v7_time"
                for s, fv in folds.items():
                    full_members = {key: replacement(baseline[s][target], baseline[s][old], seed_members[s, target, ts])
                                    for key, ts in zip(MEMBERS, spec["training_seeds"])}
                    for i in range(len(frame)):
                        writer.writerow(dict(target=target, split_seed=s, fold=fv[i], sample_id=frame.sample_id.iloc[i],
                            actual=frame[target].iloc[i], current_prediction=baseline[s][target][i],
                            **{key: float(value[i]) for key, value in full_members.items()}))
        arithmetic = arithmetic_audit(path, tuple(str(s) for s in folds))
        write_new(out/"arithmetic-audit.json", arithmetic)
    else:
        for target in active:
            if "E-COMPOSE" not in target_arms[target]: continue
            native, target_only, compose = [next(r for r in records if (r["target"], r["arm"]) == (target, a)) for a in ARMS]
            compose["mean_gain_vs_target_only"] = compose["paired"]["mean"]-target_only["paired"]["mean"]
            compose["composition_mechanism_supported"] = min(v["gain"] for v in compose["seeds"].values()) > 0 and compose["mean_gain_vs_target_only"] > 0 and all(
                compose["seeds"][s]["gain"] > target_only["seeds"][s]["gain"] for s in compose["seeds"])
    tier_spec = dict(spec, candidates={t: [a for a in target_arms.get(t, []) if a != "E-NATIVE"] for t in TARGETS})
    tier_spec["tie_preference_by_target"] = tier_spec["candidates"]
    tiers = (classify_candidates(metrics, tier_spec, yaml.safe_load(Path("configs/candidate_tiers.yaml").read_text()))
             if set(folds) == set(spec["split_seeds"]) else None)
    ledgers = [json.loads(line) for line in (out/"fit_ledger.jsonl").read_text().splitlines()]
    costs = {key: sum(r.get(key, 0) for r in ledgers) for key in (
        "selector_fits", "refit_fits", "reused_native_units", "factory_calls", "component_pipeline_fits")}
    return {"queue": queue, "records": records, "metrics": metrics,
            "selected_for_confirmation": select_finalists(records, spec, queue), "tiers": tiers,
            "costs": costs, "arithmetic_audit": arithmetic, "stop_rule": "no_seed_or_epoch_grid_extension",
            "packages": 0, "agent_uploads": 0, "release_authorized": False}


def matching_admission(root, spec, queue, preflight_output=None):
    directory = private_path(root, preflight_output) if preflight_output is not None else root/RUN_ROOT/f"preflight-{queue}"
    preflight = directory/"report.json"
    admission = json.loads(preflight.read_text())
    if (admission["status"] != "passed" or admission["queue"] != queue
            or admission["spec_sha256"] != file_hash(root/SPEC)):
        raise ValueError("Matching successful synthetic admission required")
    verify_hashes(root, admission["source_hashes"])
    return preflight, admission


def run(root, output, queue, development=None, preflight_output=None):
    root = Path(root).resolve(); spec = load_spec(root); out = private_path(root, output)
    preflight, admission = matching_admission(root, spec, queue, preflight_output)
    check_runtime(spec)
    seeds = spec["split_seeds"]; old = None
    if development:
        dev = private_path(root, development)
        audit = json.loads((dev/"audit.json").read_text())
        if audit["status"] != "passed" or not audit["selection_verified"] or audit["summary_sha256"] != file_hash(dev/"summary.json"):
            raise ValueError("Independently audited development required")
        old = json.loads((dev/"summary.json").read_text())
        if old["queue"] != queue: raise ValueError("Confirmation queue mismatch")
        if not any(old["selected_for_confirmation"].values()):
            print(json.dumps({"status": "no_development_finalist", "queue": queue, "new_fits": 0}), flush=True)
            return
        seeds = spec["confirmation_seeds"]
    if queue == "E-COMPOSE":
        prior = root/RUN_ROOT/"development-DE3/audit.json"
        if json.loads(prior.read_text())["status"] != "passed":
            raise ValueError("DE3 must finish and audit before E-COMPOSE")
    frame = load_v5_training_frame(root)
    folds = {s: fold_vector(root, frame, s, load_v5_spec(root)) for s in seeds}
    native, native_manifest = validate_native(root, spec, frame, folds) if not development else (None, None)
    if out.exists(): raise ValueError("Run exists; preserve failure evidence, no implicit restart")
    manifest = {"spec_sha256": file_hash(root/SPEC), "versions": check_runtime(spec), "queue": queue,
                "output_directory": str(out),
                "preflight": str(preflight.relative_to(root)), "preflight_sha256": file_hash(preflight),
                "source_hashes": sources(root), "data_hashes": {str(p.relative_to(root)): file_hash(p) for p in (root/"复赛_train").glob("*.csv")},
                "fold_hashes": {str(s): digest(fv.tolist()) for s, fv in folds.items()}, "seeds": seeds,
                "development": str(development) if development else None,
                "development_summary_sha256": file_hash(dev/"summary.json") if development else None,
                "starting_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    manifest["identity"] = digest(manifest)
    out.mkdir(parents=True, exist_ok=False); write_new(out/"manifest.json", manifest)
    shared = None
    if development and queue == "E-COMPOSE":
        other = root/RUN_ROOT/"confirmation-DE3"
        if (other/"audit.json").exists() and json.loads((other/"audit.json").read_text())["status"] == "passed": shared = other
    reference_units(root, out, frame, folds, spec, manifest, native, shared)
    if queue == "E-COMPOSE":
        # Factories use four workers internally. Never multiply that by four factories.
        for seed, fv in folds.items():
            for fold in range(5):
                result = background_unit(root, out, frame.loc[fv != fold].reset_index(drop=True), seed, fold, spec, manifest)
                append_event(out/"fit_ledger.jsonl", {"event": "complete", **result})
    failures = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = {}
        for seed, fv in folds.items():
            for fold in range(5):
                training = frame.loc[fv != fold].reset_index(drop=True)
                query = clean_query(frame.loc[fv == fold]).reset_index(drop=True)
                for target in TARGETS:
                    if old is not None and old["selected_for_confirmation"][target] is None: continue
                    source = native/f"{target}-BASE-s{seed}-f{fold}" if native is not None else None
                    chosen = old["selected_for_confirmation"][target] if old is not None else None
                    job = pool.submit(candidate_unit, root, out, training, query, seed, fold, target, queue, spec, manifest, source, chosen)
                    jobs[job] = (seed, fold, target)
        for job in as_completed(jobs):
            try: append_event(out/"fit_ledger.jsonl", {"event": "complete", **job.result()})
            except Exception as exc:
                failures.append(jobs[job]); append_event(out/"fit_ledger.jsonl", {"event": "failed", "unit": jobs[job], "error": repr(exc)})
    if failures: raise RuntimeError(f"Failed units preserved: {failures}")
    verify_hashes(root, manifest["source_hashes"])
    result = summarize(out, frame, folds, spec, queue)
    if old:
        decisions = []
        for target, arm in old["selected_for_confirmation"].items():
            if arm is None: continue
            before = next(r for r in old["records"] if (r["target"], r["arm"]) == (target, arm))
            after = next(r for r in result["records"] if (r["target"], r["arm"]) == (target, arm))
            rows = {**before["seeds"], **after["seeds"]}; paired = paired_summary([v["gain"] for v in rows.values()])
            dev_score = float(np.mean([v["candidate_score"] for v in before["seeds"].values()]))
            failed = []
            if paired["positive"] != 4: failed.append("not_all_four_seeds_positive")
            if paired["lcb95"] <= 0: failed.append("nonpositive_seed_lcb95")
            if dev_score < spec["promotion"]["local_working_gate"]: failed.append("local_working_gate")
            decisions.append({"target": target, "arm": arm, "seeds": rows, "paired": paired,
                              "development_score": dev_score, "failed_conditions": failed,
                              "promoted": not failed, "release_authorized": False})
        result["four_seed_decisions"] = decisions
    write_new(out/"summary.json", result)
    print(json.dumps({"status": "complete", "queue": queue, "selected": result["selected_for_confirmation"], "costs": result["costs"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--queue", choices=["DE3", "E-COMPOSE"], required=True); p.add_argument("--development", type=Path)
    p.add_argument("--preflight", type=Path)
    a = p.parse_args(); run(Path.cwd(), a.output, a.queue, a.development, a.preflight)
