"""Run and audit the frozen V17 complete-fold, fixed-slot experiment."""
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
from .v5_resolution import package_score, paired_summary, wmape
from .v5_spec import load_v5_spec
from .v7_periodic import digest, file_hash, write_new
from .v11_quantile import load_oof
from .v16_sequential_masks import load_references
from .v17_models import fit_fold


def context(root, spec, seeds):
    frame = load_v5_training_frame(root)
    folds = {seed: fold_vector(root, frame, seed, load_v5_spec(root)) for seed in seeds}
    if list(seeds) != spec["split_seeds"]:
        raise ValueError("Development reference loader only accepts frozen development seeds")
    reference_spec = yaml.safe_load((root / spec["reference_source"]).read_text())
    _, _, a35, _, v12, joint, hashes = load_references(root, frame, folds, reference_spec)
    time, time_hashes = load_oof(root, spec["v7_time_cache"], frame, folds,
                                 "tap_time_len", "tabm_plr001")
    hashes.update(time_hashes)
    b0 = {}
    for seed in seeds:
        b0[seed] = {
            "tap_iron": v12[seed]["tap_iron"].copy(),
            "tap_time_len": .5 * a35[seed]["tap_time_len"] + .5 * time[seed],
        }
        expected_iron = .5 * a35[seed]["tap_iron"] + .5 * joint[seed][:, 0]
        if not np.array_equal(b0[seed]["tap_iron"], expected_iron):
            raise ValueError("V12 iron slot is not the frozen half-weight column")
        if any(not np.isfinite(v).all() for v in b0[seed].values()):
            raise ValueError("Incomplete B0 local reference")
    return frame, folds, a35, b0, hashes


def oof_from_folds(directory, frame, folds, key):
    predictions = {}
    for seed, fv in folds.items():
        vector = np.full(len(frame), np.nan)
        for fold in range(5):
            path = directory / f"{key}-s{seed}-f{fold}.npy"
            pred = np.load(path, allow_pickle=False)
            if pred.shape != (int((fv == fold).sum()),) or not np.isfinite(pred).all():
                raise ValueError(f"Invalid prediction matrix: {path}")
            vector[fv == fold] = pred
        if not np.isfinite(vector).all():
            raise ValueError(f"Incomplete OOF: {key} {seed}")
        predictions[seed] = vector
    return predictions


def score_candidate(frame, folds, a35, b0, predictions, target, weight=.5):
    other = next(name for name in TARGETS if name != target)
    results = {}
    for seed in folds:
        candidate = dict(b0[seed])
        candidate[target] = (1 - weight) * a35[seed][target] + weight * predictions[seed]
        base_score = package_score(*[wmape(frame[name], b0[seed][name]) for name in TARGETS])
        candidate_score = package_score(*[wmape(frame[name], candidate[name]) for name in TARGETS])
        cells = []
        for fold in range(5):
            mask = folds[seed] == fold
            cells.append({"fold": fold, "rows": int(mask.sum()),
                          "gain": 50 * (wmape(frame.loc[mask, target], b0[seed][target][mask])
                                        - wmape(frame.loc[mask, target], candidate[target][mask]))})
        results[str(seed)] = {"b0_score": base_score, "candidate_score": candidate_score,
                              "gain": candidate_score - base_score,
                              "target_wmape": wmape(frame[target], candidate[target]),
                              "other_column_equal": bool(np.array_equal(candidate[other], b0[seed][other])),
                              "folds_descriptive": cells}
    return results


def summarize(directory, frame, folds, a35, b0, spec):
    records = []
    for name, recipe in spec["recipes"].items():
        predictions = oof_from_folds(directory, frame, folds, name)
        cells = score_candidate(frame, folds, a35, b0, predictions, recipe["target"], spec["blend_weight"])
        gains = {seed: row["gain"] for seed, row in cells.items()}
        records.append({"recipe": name, "target": recipe["target"],
                        "seed_results": cells, "development_gain": paired_summary(list(gains.values())),
                        "positive_development_seeds": all(gain > 0 for gain in gains.values())})
    selected = {}
    for target in TARGETS:
        eligible = [r for r in records if r["target"] == target and r["positive_development_seeds"]]
        selected[target] = (min(eligible, key=lambda r: (-r["development_gain"]["mean"],
                                                     list(spec["recipes"]).index(r["recipe"])))["recipe"]
                            if eligible else None)
    contrasts = {}
    by_name = {r["recipe"]: r for r in records}
    for label, left, right in [("J2_minus_J1", "J2", "J1"),
                               ("P_LH_minus_LL_iron", "P_LH_I", "P_LL_I"),
                               ("P_LH_minus_LL_time", "P_LH_T", "P_LL_T")]:
        contrasts[label] = {seed: by_name[left]["seed_results"][seed]["gain"]
                            - by_name[right]["seed_results"][seed]["gain"]
                            for seed in map(str, spec["split_seeds"])}
    return {"status": "development_complete", "fixed_blend_weight": spec["blend_weight"],
            "records": records, "mechanism_contrasts": contrasts,
            "selected_for_confirmation": selected,
            "packages": 0, "uploads": 0, "release_authorized": False}


def run(root, output):
    root = Path(root).resolve()
    spec_path = root / "configs/round2_v17/SPEC.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    versions = {name: importlib.metadata.version(name) for name in spec["runtime_versions"]}
    if versions != spec["runtime_versions"]:
        raise ValueError(f"V17 runtime mismatch: {versions}")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"Set {name}=1 before launching parallel fits")
    out = (root / output).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v17"):
        raise ValueError("Private V17 output directory required")
    frame, folds, a35, b0, hashes = context(root, spec, spec["split_seeds"])
    joint_settings = yaml.safe_load((root / spec["training_source_joint"]).read_text())["training"]
    time_settings = yaml.safe_load((root / spec["training_source_time"]).read_text())["training"]
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "manifest.json", {
        "spec_sha256": file_hash(spec_path), "plan_sha256": file_hash(root / spec["source_plan"]),
        "versions": versions,
        "data_digest": hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
        "fold_digests": {str(s): digest(v.tolist()) for s, v in folds.items()},
        "reference_hashes": hashes,
        "source_hashes": {name: file_hash(root / "src/bf_tap_r2" / name)
                          for name in ["v17_models.py", "v17_run.py", "compose_v12_v7.py",
                                       "v12_joint.py", "v7_periodic.py", "v3_6_networks.py"]},
        "b0_scores": {str(s): package_score(*[wmape(frame[t], b0[s][t]) for t in TARGETS]) for s in folds},
    })
    failed = 0
    with ProcessPoolExecutor(max_workers=spec["workers"]) as pool:
        jobs = {}
        for name, recipe in spec["recipes"].items():
            settings = joint_settings if recipe["kind"] == "joint" else time_settings
            for seed, fv in folds.items():
                for fold in range(5):
                    future = pool.submit(fit_fold, frame, fv, name, recipe, settings, fold)
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
        raise RuntimeError(f"V17 failed fits retained: {failed}")
    summary = summarize(out, frame, folds, a35, b0, spec)
    write_new(out / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "selected": summary["selected_for_confirmation"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(Path.cwd(), args.output)


if __name__ == "__main__":
    main()
