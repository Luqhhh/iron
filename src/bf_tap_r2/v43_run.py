"""Frozen V43 development/conditional confirmation with append-only units."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import yaml

from .candidate_tiers import classify_candidates
from .data import FEATURES, TARGETS
from .metrics import wmape
from .v3_4_bags import group_safe_inner_folds
from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import paired_summary
from .v5_spec import load_v5_spec
from .v7_periodic import digest, file_hash, write_new
from .v43_bart import fit_partition
from .v30_reference import fit_b0

# Public round renamed; private run roots retain their frozen legacy names.
SPEC = "configs/round2_v43/SPEC.yaml"


def append_event(path, event):
    with Path(path).open("a") as stream:
        stream.write(json.dumps({"time_ns": time.time_ns(), **event}, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_hashes(root):
    paths = list((root / "src/bf_tap_r2").glob("*.py"))
    paths += list((root / "configs").rglob("*.yaml"))
    paths += [root / "uv.lock", root / "pyproject.toml", root / "docs/round2_v43/PREREGISTRATION.md"]
    directories = ["round2-v3-local-search", "round2-v3.1-directed-search",
                   "round2-v3.2-ensemble-and-target-search", "round2-v3.3-structure-search",
                   "round2-v3.4-ebm-and-constrained-composition",
                   "round2-v3.6-loss-training-and-numeric-encoding"]
    for directory in directories:
        parent = root / "local/runs" / directory
        for extension in ("*.py", "*.json", "*.jsonl"):
            paths += list(parent.rglob(extension))
    return {str(path.relative_to(root)): file_hash(path) for path in sorted(set(paths))}


def verify_hashes(root, hashes):
    for name, sha in hashes.items():
        if file_hash(root / name) != sha:
            raise ValueError(f"Frozen source or evidence changed: {name}")


def check_runtime(spec):
    versions = {name: importlib.metadata.version(name) for name in spec["runtime_versions"]}
    if versions != spec["runtime_versions"]:
        raise ValueError(f"V43 runtime differs: {versions}")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"Set {name}=1")
    return versions



def verify_reference_cache(root, spec):
    cache = root / spec["reference_cache"]["directory"]
    for name in ("manifest", "audit"):
        if file_hash(cache / f"{name}.json") != spec["reference_cache"][f"{name}_sha256"]:
            raise ValueError("Reference cache manifest/audit changed")
    original = json.loads((cache / "manifest.json").read_text())
    audit = json.loads((cache / "audit.json").read_text())
    if audit["status"] != "passed" or file_hash(cache / "summary.json") != audit["summary_sha256"]:
        raise ValueError("Reference cache audit is not bound to its summary")
    verify_hashes(root, original["data_hashes"])
    commit = spec["reference_cache"]["original_commit"]
    original_spec = yaml.safe_load(subprocess.check_output(
        ["git", "show", f"{commit}:configs/round2_v27/SPEC.yaml"], cwd=root))
    for key in ("calibration", "runtime_versions", "split_seeds", "folds"):
        if (spec[key] if key != "runtime_versions" else {k: spec[key][k] for k in original_spec[key]}) != original_spec[key]:
            raise ValueError(f"Reference protocol mismatch: {key}")
    if spec["cache_original_reference"] != original_spec["reference"]:
        raise ValueError("Original component reference identity changed")
    current = dict(spec["reference"], time_weights=spec["cache_original_reference"]["time_weights"])
    if current != spec["cache_original_reference"] or spec["reference"]["time_weights"] != {"v36": .2, "n0048": .3, "v7_periodic": .5}:
        raise ValueError("Only the frozen current-platform time reweighting is allowed")
    # Renamed public files are verified against their original committed bytes.
    for name, sha in original["source_hashes"].items():
        path = root / name
        if path.is_file() and file_hash(path) == sha:
            continue
        data = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"Original reference source mismatch: {name}")
    for seed in spec["split_seeds"]:
        for fold in range(5):
            for role in ("outer", "calibration"):
                key = f"reference-{role}-s{seed}-f{fold}"
                verified_unit(cache / key, unit_id(original, key))
    return original


def read_reference(source, training, query, spec):
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata["fit_ids_digest"] != digest(training.sample_id.tolist()):
        raise ValueError("Cached reference fit rows changed")
    with np.load(source / "predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["query_ids"], query.sample_id.to_numpy(dtype=str)):
            raise ValueError("Cached reference query rows changed")
        values = {k: saved[k].copy() for k in saved.files if k != "query_ids"}
    expected_i = .5 * values["v36_iron"] + .5 * values["v12_iron"]
    expected_t = .325 * values["v36_time"] + .175 * values["n_time"] + .5 * values["v7_time"]
    if not (np.array_equal(expected_i, values["tap_iron"]) and np.array_equal(expected_t, values["tap_time_len"])):
        raise ValueError("Cached B0 endpoint arithmetic mismatch")
    values["historical_b0_time"] = values["tap_time_len"].copy()
    values["tap_time_len"] = .2*values["v36_time"]+.3*values["n_time"]+.5*values["v7_time"]
    metadata["reference_reweighting"] = "V32_TIME_A60V7_50_current_platform_best"
    metadata["reused_from"] = str(source)
    return values, metadata


def partitions(frame, folds, fold, spec):
    training = frame.loc[folds != fold].reset_index(drop=True)
    query = frame.loc[folds == fold, ["sample_id", "spout_no", *FEATURES]].reset_index(drop=True)
    inner = group_safe_inner_folds(training, seed=spec["calibration"]["split_seed"],
                                  n_splits=spec["calibration"]["n_splits"])["fold"]
    mask = inner == spec["calibration"]["held_fold"]
    fitting = training.loc[~mask].reset_index(drop=True)
    calibration = training.loc[mask].reset_index(drop=True)
    return training, query, fitting, calibration


def verified_unit(directory, identity):
    directory = Path(directory)
    if not directory.exists():
        return None
    path = directory / "complete.json"
    if not path.exists():
        raise ValueError(f"Incomplete/failed unit preserved; explicit recovery required: {directory}")
    result = json.loads(path.read_text())
    if result["identity"] != identity:
        raise ValueError(f"Unit identity mismatch: {directory}")
    for name, sha in result["hashes"].items():
        if file_hash(directory / name) != sha:
            raise ValueError(f"Unit artifact changed: {directory / name}")
    return result


def run_unit(root, directory, frame, folds, seed, fold, spec, identity, kind, target=None, recipe=None):
    directory = Path(directory)
    cached = verified_unit(directory, identity)
    if cached:
        return {"key": directory.name, "cached": True}
    directory.mkdir(parents=True, exist_ok=False)
    append_event(directory / "events.jsonl", {"event": "start", "kind": kind, "seed": seed,
                                             "fold": fold, "target": target, "recipe": recipe,
                                             "identity": identity})
    started = time.monotonic()
    try:
        training, query, fitting, calibration = partitions(frame, folds, fold, spec)
        if kind == "reference":
            train_part, query_part = ((fitting, calibration.drop(columns=list(TARGETS)))
                                      if target == "calibration" else (training, query))
            if seed in spec["split_seeds"]:
                source = root / spec["reference_cache"]["directory"] / directory.name
                values, metadata = read_reference(source, train_part, query_part, spec)
            else:
                values, metadata = fit_b0(root, train_part, query_part, spec)
                values["historical_b0_time"] = .325*values["v36_time"]+.175*values["n_time"]+.5*values["v7_time"]
            with (directory / "predictions.npz").open("xb") as stream:
                np.savez_compressed(stream, **values, query_ids=query_part.sample_id.to_numpy(dtype=str))
        else:
            parent = directory.parent
            calibration_dir = parent / f"reference-calibration-s{seed}-f{fold}"
            with np.load(calibration_dir / "predictions.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["query_ids"], calibration.sample_id.to_numpy(dtype=str)):
                    raise ValueError("Calibration reference row identity mismatch")
                base = saved[target].copy()
            model, values, metadata, calibration_pred = fit_partition(
                fitting, calibration, training, query, target, recipe, spec["training"],
                base, spec["calibration"]["blend_grid"])
            model.save(directory / "model.json")
            model.calibration_model_.save(directory / "calibration_model.json")
            with (directory / "predictions.npz").open("xb") as stream:
                np.savez_compressed(stream, prediction=values, calibration_prediction=calibration_pred,
                                    query_ids=query.sample_id.to_numpy(dtype=str),
                                    calibration_ids=calibration.sample_id.to_numpy(dtype=str))
        metadata.update(seconds=time.monotonic() - started, seed=seed, fold=fold, kind=kind,
                        target=target, recipe=recipe)
        write_new(directory / "metadata.json", metadata)
        artifacts = ["metadata.json", "predictions.npz"] + (
            ["model.json", "calibration_model.json"] if kind == "candidate" else [])
        result = {"identity": identity, "hashes": {p: file_hash(directory / p) for p in artifacts}}
        write_new(directory / "complete.json", result)
        append_event(directory / "events.jsonl", {"event": "complete", "seconds": metadata["seconds"]})
        return {"key": directory.name, "seconds": round(metadata["seconds"], 2), "cached": False}
    except Exception as exc:
        append_event(directory / "events.jsonl", {"event": "failed", "error": repr(exc)})
        raise


def unit_id(manifest, key):
    return digest({"manifest_identity": manifest["identity"], "unit": key})


def metric_detail(y, prediction, folds, spout):
    return {"wmape": wmape(y, prediction),
            "by_fold": {str(f): wmape(y[folds == f], prediction[folds == f]) for f in range(5)},
            "by_spout": {str(s): wmape(y[spout == s], prediction[spout == s]) for s in sorted(set(spout))}}


def select_finalists(records, spec):
    selected = {}
    for target in TARGETS:
        eligible = [r for r in records if r["target"] == target and r["both_seeds_positive"]
                    and r["recipe"] in spec["promotion"]["eligible_recipes"]
                    and r["paired_seed_summary"]["mean"] >= spec["promotion"]["development_mean_gain_ge"]
                    and (not any(x["target"] == target and x["recipe"] == "STUMP" for x in records)
                         or r["paired_seed_summary"]["mean"] > next(x["paired_seed_summary"]["mean"]
                             for x in records if x["target"] == target and x["recipe"] == "STUMP"))]
        selected[target] = (min(eligible, key=lambda r: (-r["paired_seed_summary"]["mean"],
                                       spec["tie_preference_by_target"][target].index(r["recipe"])))["recipe"]
                            if eligible else None)
    return selected


def summarize(out, frame, folds, spec, candidates):
    records, metrics = [], {target: {"R43_CURRENT_REFIT": {}} for target in TARGETS}
    y = {t: frame[t].to_numpy() for t in TARGETS}
    spout = frame.spout_no.to_numpy()
    base = {s: {t: np.full(len(frame), np.nan) for t in TARGETS} for s in folds}
    for seed, fv in folds.items():
        for fold in range(5):
            mask = fv == fold
            with np.load(out / f"reference-outer-s{seed}-f{fold}/predictions.npz") as saved:
                if not np.array_equal(saved["query_ids"], frame.loc[mask, "sample_id"].to_numpy(dtype=str)):
                    raise ValueError("Outer reference identity mismatch")
                for target in TARGETS:
                    base[seed][target][mask] = saved[target]
        for target in TARGETS:
            metrics[target]["R43_CURRENT_REFIT"][str(seed)] = metric_detail(y[target], base[seed][target], fv, spout)
    for target, names in candidates.items():
        for recipe in names:
            seed_results = {}
            metrics[target][recipe] = {}
            for seed, fv in folds.items():
                member, blended = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
                weights = []
                for fold in range(5):
                    mask = fv == fold
                    directory = out / f"{target}-{recipe}-s{seed}-f{fold}"
                    metadata = json.loads((directory / "metadata.json").read_text())
                    with np.load(directory / "predictions.npz") as saved:
                        if not np.array_equal(saved["query_ids"], frame.loc[mask, "sample_id"].to_numpy(dtype=str)):
                            raise ValueError("Candidate outer row identity mismatch")
                        member[mask] = saved["prediction"]
                    a = metadata["weight"]
                    weights.append(a)
                    blended[mask] = (1 - a) * base[seed][target][mask] + a * member[mask]
                other = next(t for t in TARGETS if t != target)
                bw = wmape(y[target], base[seed][target])
                cw = wmape(y[target], blended)
                base_score = 100 - 50 * (bw + wmape(y[other], base[seed][other]))
                metrics[target][recipe][str(seed)] = metric_detail(y[target], blended, fv, spout)
                seed_results[str(seed)] = {"gain": 50 * (bw - cw), "reference_score": base_score,
                    "candidate_score": base_score + 50 * (bw - cw), "weights_by_fold": weights,
                    "standalone_wmape": wmape(y[target], member), "blended_wmape": cw,
                    "baseline_wmape": bw,
                    "positive_folds_descriptive": sum(metrics[target][recipe][str(seed)]["by_fold"][str(f)]
                        < metrics[target]["R43_CURRENT_REFIT"][str(seed)]["by_fold"][str(f)] for f in range(5))}
            paired = paired_summary([v["gain"] for v in seed_results.values()])
            records.append({"target": target, "recipe": recipe, "seed_results": seed_results,
                            "paired_seed_summary": paired, "both_seeds_positive": paired["positive"] == len(folds)})
    selected = select_finalists(records, spec)
    return {"records": records, "metrics": metrics, "selected_for_confirmation": selected,
            "protocol": "outer_training_only_calibration_with_same_recipe_reference_refit",
            "packages": 0, "agent_uploads": 0}


def run(root, output, development=None):
    root = Path(root).resolve()
    out = (root / output).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v43"):
        raise ValueError("Private V43 run directory required")
    spec = yaml.safe_load((root / SPEC).read_text())
    preflight = json.loads((root / "local/runs/round2-v43/preflight-r1/report.json").read_text())
    if preflight["status"] != "passed" or preflight["spec_sha256"] != file_hash(root / SPEC):
        raise ValueError("Matching successful synthetic preflight required")
    verify_hashes(root, preflight["source_hashes"])
    versions = check_runtime(spec)
    if not development:
        verify_reference_cache(root, spec)
    candidates = spec["candidates"]
    seeds = spec["split_seeds"]
    dev = None
    if development:
        dev = (root / development).resolve()
        audit = json.loads((dev / "audit.json").read_text())
        if audit["status"] != "passed" or file_hash(dev / "summary.json") != audit["summary_sha256"]:
            raise ValueError("Audited development required")
        summary = json.loads((dev / "summary.json").read_text())
        if audit.get("selection_verified") is not True:
            raise ValueError("Independently audited selection required")
        verify_hashes(root, json.loads((dev / "manifest.json").read_text())["source_hashes"])
        candidates = {t: [name] if name else [] for t, name in summary["selected_for_confirmation"].items()}
        if not any(candidates.values()):
            print(json.dumps({"status": "no_development_finalist", "confirmation_fits": 0}), flush=True)
            return
        seeds = spec["confirmation_seeds"]
    frame = load_v5_training_frame(root)
    folds = {s: fold_vector(root, frame, s, load_v5_spec(root)) for s in seeds}
    if not development:
        original = json.loads((root / spec["reference_cache"]["directory"] / "manifest.json").read_text())
        if {str(s): digest(f.tolist()) for s, f in folds.items()} != original["fold_hashes"]:
            raise ValueError("Reference cache fold mismatch")
    data_hashes = {str(p.relative_to(root)): file_hash(p) for p in (root / "复赛_train").glob("*.csv")}
    if out.exists():
        manifest = json.loads((out / "manifest.json").read_text())
        verify_hashes(root, manifest["source_hashes"])
        if (manifest["versions"] != versions or manifest["data_hashes"] != data_hashes
                or manifest["candidates"] != candidates or manifest["seeds"] != seeds):
            raise ValueError("Run identity changed")
        if (out / "summary.json").exists():
            print(json.dumps({"status": "already_complete", "output": str(out)}), flush=True)
            return
    else:
        manifest = {"spec_sha256": file_hash(root / SPEC), "versions": versions,
                    "source_hashes": source_hashes(root), "data_hashes": data_hashes,
                    "fold_hashes": {str(s): digest(f.tolist()) for s, f in folds.items()},
                    "seeds": seeds, "candidates": candidates,
                    "starting_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                    "development_directory": str(dev.relative_to(root)) if dev else None,
                    "development_summary_sha256": file_hash(dev / "summary.json") if dev else None}
        manifest["identity"] = digest(manifest)
        out.mkdir(parents=True, exist_ok=False)
        write_new(out / "manifest.json", manifest)
    for phase in ("reference", "candidate"):
        workers = spec["budget"]["concurrent_reference_calls" if phase == "reference" else "candidate_workers"]
        tasks = []
        for seed in seeds:
            for fold in range(5):
                if phase == "reference":
                    for role in ("calibration", "outer"):
                        key = f"reference-{role}-s{seed}-f{fold}"
                        tasks.append((key, seed, fold, role, None))
                else:
                    for target, recipes in candidates.items():
                        for recipe in recipes:
                            tasks.append((f"{target}-{recipe}-s{seed}-f{fold}", seed, fold, target, recipe))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for key, seed, fold, target, recipe in tasks:
                identity = unit_id(manifest, key)
                if verified_unit(out / key, identity):
                    continue
                future = pool.submit(run_unit, root, out / key, frame, folds[seed], seed, fold,
                                     spec, identity, phase, target, recipe)
                futures[future] = key
            failed = []
            for future in as_completed(futures):
                key = futures[future]
                try:
                    result = future.result()
                    append_event(out / "fit_ledger.jsonl", {"event": "complete", **result})
                    print(json.dumps(result), flush=True)
                except Exception as exc:
                    failed.append(key)
                    append_event(out / "fit_ledger.jsonl", {"event": "failed", "key": key, "error": repr(exc)})
                    print(json.dumps({"failed": key, "error": repr(exc)}), flush=True)
            if failed:
                raise RuntimeError(f"Failed units retained: {failed}")
    verify_hashes(root, manifest["source_hashes"])
    result = summarize(out, frame, folds, spec, candidates)
    if not development:
        result["tiers"] = classify_candidates(result["metrics"], spec,
                                               yaml.safe_load((root / "configs/candidate_tiers.yaml").read_text()))
    else:
        old = json.loads((dev / "summary.json").read_text())
        result["four_seed_decisions"] = []
        for row in result["records"]:
            previous = next(r for r in old["records"] if (r["target"], r["recipe"]) == (row["target"], row["recipe"]))
            cells = {**previous["seed_results"], **row["seed_results"]}
            paired = paired_summary([v["gain"] for v in cells.values()])
            dev_mean = float(np.mean([v["candidate_score"] for v in previous["seed_results"].values()]))
            failures = []
            if paired["positive"] != 4: failures.append("not_all_four_seeds_positive")
            if paired["lcb95"] <= 0: failures.append("nonpositive_seed_lcb95")
            if dev_mean < spec["promotion"]["local_working_gate"]: failures.append("local_working_gate")
            result["four_seed_decisions"].append({"recipe": row["recipe"], "target": row["target"],
                "seed_results": cells, "paired_seed_summary": paired, "development_mean_score": dev_mean,
                "failed_conditions": failures, "promoted": not failures, "release_authorized": False})
    write_new(out / "summary.json", result)
    print(json.dumps({"status": "complete", "selected": result["selected_for_confirmation"],
                      "four_seed_decisions": result.get("four_seed_decisions")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", type=Path)
    args = parser.parse_args()
    run(Path.cwd(), args.output, args.development)


if __name__ == "__main__":
    main()
