"""Conditional V17 confirmation on the two frozen derived split seeds."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import TARGETS
from .v5_library import fold_vector, load_v5_training_frame
from .v5_replicate import _identity
from .v5_resolution import paired_summary
from .v5_spec import load_v5_spec
from .v7_confirm import read_reference_fold
from .v7_periodic import digest, file_hash, write_new
from .v12_confirm import four_seed_references
from .v17_models import fit_fold
from .v17_run import context, oof_from_folds, score_candidate


def confirmed_references(root, frame, folds, spec):
    dev_seeds = spec["split_seeds"]
    confirmation_seeds = spec["confirmation_seeds"]
    _, _, a35, b0, hashes = context(root, spec, dev_seeds)
    v12_spec = yaml.safe_load((root / spec["training_source_joint"]).read_text())
    v12_confirmation = yaml.safe_load((root / "configs/round2_v12/CONFIRMATION.yaml").read_text())
    iron_refs, iron_hashes = four_seed_references(root, frame, folds, v12_spec, v12_confirmation)
    hashes.update(iron_hashes)
    for seed in dev_seeds:
        if not np.array_equal(a35[seed]["tap_iron"], iron_refs["A35"][seed]):
            raise ValueError("Development A35 iron reference mismatch")
    v12_dir = root / spec["v12_joint_confirmation"]
    v7_dir = root / spec["v7_time_confirmation"]
    v12_audit = json.loads((v12_dir / "audit-r1.json").read_text())
    v7_audit = json.loads((v7_dir / "audit-r2.json").read_text())
    if v12_audit["status"] != "passed" or v7_audit["status"] != "PASS":
        raise ValueError("Audited V12 and V7 confirmation caches required")
    for directory, audit_name in [(v12_dir, "audit-r1.json"), (v7_dir, "audit-r2.json")]:
        for name in ["manifest.json", "fit_ledger.jsonl", audit_name]:
            path = directory / name
            hashes[str(path.relative_to(root))] = file_hash(path)
    v12_events = {(e["seed"], e["fold"]): e for e in
                  map(json.loads, (v12_dir / "fit_ledger.jsonl").read_text().splitlines())}
    v7_events = {(e["seed"], e["fold"]): e for e in
                 map(json.loads, (v7_dir / "fit_ledger.jsonl").read_text().splitlines())}
    if len(v12_events) != 10 or len(v7_events) != 10:
        raise ValueError("Complete source confirmation ledgers required")
    meta = {"kind": "v36", "trial_id": "v36-s1-N-0048", "target": "tap_time_len", "line": "N"}
    for seed in confirmation_seeds:
        fv = folds[seed]
        iron_base = iron_refs["A35"][seed]
        time_base = np.full(len(frame), np.nan)
        iron_member = np.full(len(frame), np.nan)
        time_member = np.full(len(frame), np.nan)
        for fold in range(5):
            mask = fv == fold
            expected_ids = digest(frame.loc[~mask, "sample_id"].tolist())
            cache = root / f"local/runs/round2-v5-error-covariance/replication-r1/seed-{seed}/fold-{fold}.npz"
            base36, member_n = read_reference_fold(cache, _identity(frame, fv, seed, fold, meta), mask)
            time_base[mask] = .65 * base36 + .35 * member_n
            hashes[str(cache.relative_to(root))] = file_hash(cache)
            for directory, events, audit, column, destination in [
                (v12_dir, v12_events, v12_audit["hashes"], 0, iron_member),
                (v7_dir, v7_events, v7_audit["prediction_hashes"], None, time_member),
            ]:
                event = events[seed, fold]
                path = directory / f"seed-{seed}-fold-{fold}.npy"
                relative = str(path.relative_to(root))
                if event["event"] != "complete" or event["metadata"]["fit_ids_digest"] != expected_ids:
                    raise ValueError("Source fit row identity mismatch")
                if file_hash(path) != audit[relative]:
                    raise ValueError("Source confirmation prediction hash mismatch")
                values = np.load(path, allow_pickle=False)
                values = values[:, column] if column is not None else values
                if values.shape != (int(mask.sum()),) or not np.isfinite(values).all():
                    raise ValueError("Source confirmation prediction shape mismatch")
                destination[mask] = values
                hashes[relative] = file_hash(path)
        if any(not np.isfinite(v).all() for v in [iron_base, time_base, iron_member, time_member]):
            raise ValueError("Incomplete derived-seed B0 reference")
        a35[seed] = {"tap_iron": iron_base, "tap_time_len": time_base}
        b0[seed] = {"tap_iron": .5 * iron_base + .5 * iron_member,
                    "tap_time_len": .5 * time_base + .5 * time_member}
    return a35, b0, hashes


def run(root, development, output):
    root = Path(root).resolve()
    dev = (root / development).resolve()
    out = (root / output).resolve()
    allowed = root / "local/runs/round2-v17"
    if not dev.is_relative_to(allowed) or not out.is_relative_to(allowed):
        raise ValueError("Private V17 directories required")
    spec_path = root / "configs/round2_v17/SPEC.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    dev_manifest = json.loads((dev / "manifest.json").read_text())
    summary = json.loads((dev / "summary.json").read_text())
    audit = json.loads((dev / "audit-r1.json").read_text())
    if (audit["status"] != "passed" or audit["selected_for_confirmation"] != summary["selected_for_confirmation"]
            or file_hash(spec_path) != dev_manifest["spec_sha256"]):
        raise ValueError("Audited, frozen V17 development required")
    for name, sha in dev_manifest["source_hashes"].items():
        if file_hash(root / "src/bf_tap_r2" / name) != sha:
            raise ValueError(f"Development implementation changed: {name}")
    selected = {target: name for target, name in summary["selected_for_confirmation"].items() if name}
    if not selected:
        return {"status": "no_development_finalist", "confirmation_fits": 0}
    if len(selected) > 2 or any(spec["recipes"][name]["target"] != target for target, name in selected.items()):
        raise ValueError("Invalid frozen finalist selection")
    versions = {name: importlib.metadata.version(name) for name in spec["runtime_versions"]}
    if versions != dev_manifest["versions"]:
        raise ValueError("Runtime changed")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"Set {name}=1")
    frame = load_v5_training_frame(root)
    data_digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    if data_digest != dev_manifest["data_digest"]:
        raise ValueError("Training data changed")
    seeds = spec["split_seeds"] + spec["confirmation_seeds"]
    folds = {seed: fold_vector(root, frame, seed, load_v5_spec(root)) for seed in seeds}
    a35, b0, hashes = confirmed_references(root, frame, folds, spec)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "manifest.json", {"spec_sha256": file_hash(spec_path),
        "development_manifest_sha256": file_hash(dev / "manifest.json"),
        "development_summary_sha256": file_hash(dev / "summary.json"),
        "development_audit_sha256": file_hash(dev / "audit-r1.json"),
        "selected": selected, "data_digest": data_digest,
        "fold_digests": {str(s): digest(v.tolist()) for s, v in folds.items()},
        "reference_hashes": hashes, "versions": versions,
        "source_sha256": file_hash(root / "src/bf_tap_r2/v17_confirm.py")})
    joint_settings = yaml.safe_load((root / spec["training_source_joint"]).read_text())["training"]
    time_settings = yaml.safe_load((root / spec["training_source_time"]).read_text())["training"]
    failed = 0
    with ProcessPoolExecutor(max_workers=spec["workers"]) as pool:
        jobs = {}
        for target, name in selected.items():
            recipe = spec["recipes"][name]
            settings = joint_settings if recipe["kind"] == "joint" else time_settings
            for seed in spec["confirmation_seeds"]:
                for fold in range(5):
                    future = pool.submit(fit_fold, frame, folds[seed], name, recipe, settings, fold)
                    jobs[future] = (name, seed, fold)
        for future in as_completed(jobs):
            name, seed, fold = jobs[future]
            key = f"{name}-s{seed}-f{fold}"
            try:
                prediction, metadata = future.result()
                path = out / f"{key}.npy"
                with path.open("xb") as stream:
                    np.save(stream, prediction)
                event = {"event": "complete", "key": key, "metadata": metadata,
                         "prediction_sha256": file_hash(path)}
            except Exception as exc:
                failed += 1
                event = {"event": "failed", "key": key, "error": repr(exc)}
            with (out / "fit_ledger.jsonl").open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            print(json.dumps({k: v for k, v in event.items() if k != "metadata"}), flush=True)
    if failed:
        raise RuntimeError(f"V17 failed confirmation fits retained: {failed}")
    results = {}
    for target, name in selected.items():
        dev_pred = oof_from_folds(dev, frame, {s: folds[s] for s in spec["split_seeds"]}, name)
        new_pred = oof_from_folds(out, frame, {s: folds[s] for s in spec["confirmation_seeds"]}, name)
        cells = score_candidate(frame, folds, a35, b0, {**dev_pred, **new_pred}, target,
                                spec["blend_weight"])
        gains = [cells[str(seed)]["gain"] for seed in seeds]
        paired = paired_summary(gains)
        dev_score = float(np.mean([cells[str(seed)]["candidate_score"] for seed in spec["split_seeds"]]))
        results[target] = {"recipe": name, "seed_results": cells, "paired_seed_summary": paired,
            "local_working_gate_met": dev_score >= spec["promotion"]["local_working_gate"],
            "promoted": bool(all(g > 0 for g in gains) and paired["lcb95"] > 0
                             and dev_score >= spec["promotion"]["local_working_gate"])}
    write_new(out / "summary.json", {"status": "confirmation_complete", "results": results,
        "packages": 0, "uploads": 0, "release_authorized": False})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.development, args.output), default=str), flush=True)


if __name__ == "__main__":
    main()
