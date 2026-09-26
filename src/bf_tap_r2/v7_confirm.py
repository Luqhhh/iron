"""Confirm the highest-ranked V7 candidate, reusing verified V5 time references.

The current cache contains time only. An iron winner explicitly stops here;
it requires a new same-fold reference fit, never substitution of a time column.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .v5_library import fold_vector, load_v5_training_frame
from .v5_replicate import _identity
from .v5_resolution import admit, nested_blend, package_score, wmape
from .v5_spec import load_v5_spec
from .v7_periodic import _fit_job, file_hash, references, write_new


def strongest(summary, spec):
    if summary["status"] != "development_complete":
        raise ValueError("Complete development required before confirmation")
    expected = {(t, r) for t in spec["candidates"] for r in spec["candidates"][t]}
    if (len(summary["rows"]) != len(expected)
            or {(r["target"], r["recipe"]) for r in summary["rows"]} != expected):
        raise ValueError("Incomplete frozen candidate pool")
    seeds = {str(s) for s in spec["split_seeds"]}
    if any({str(s) for s in r["a35_nested"]["seed_gains"]} != seeds for r in summary["rows"]):
        raise ValueError("Development seed coverage mismatch")
    valid = [r for r in summary["rows"] if r["confirmation_eligible"]
             and all(float(g) > 0 for g in r["a35_nested"]["seed_gains"].values())]
    if not valid:
        raise ValueError("No candidate passed both complete development seeds")
    return min(valid, key=lambda r: (-r["a35_nested"]["seed_summary"]["mean"],
               spec["tie_preference_by_target"][r["target"]].index(r["recipe"]), r["target"]))


def read_reference_fold(path, expected_identity, expected_mask):
    with np.load(path, allow_pickle=False) as cached:
        if str(cached["identity"]) != expected_identity:
            raise ValueError("V5 reference identity mismatch")
        mask = cached["mask"]
        if mask.dtype != bool or not np.array_equal(mask, expected_mask):
            raise ValueError("V5 reference mask mismatch")
        base, candidate = np.asarray(cached["base"]), np.asarray(cached["candidate"])
        for value in (base, candidate):
            if value.shape != mask.shape or not np.isfinite(value[mask]).all() or not np.isnan(value[~mask]).all():
                raise ValueError("V5 reference coverage mismatch")
    return base[mask], candidate[mask]


def run(root, development, output):
    import importlib.metadata
    import hashlib
    root = Path(root).resolve()
    dev = (root / development).resolve()
    out = (root / output).resolve()
    allowed = root / "local/runs/round2-v7-periodic-networks"
    if not dev.is_relative_to(allowed) or not out.is_relative_to(allowed):
        raise ValueError("Private V7 paths required")
    spec_path = root / "configs/round2_v7/SPEC.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    summary = json.loads((dev / "summary.json").read_text())
    manifest = json.loads((dev / "manifest.json").read_text())
    selected = strongest(summary, spec)
    if selected["target"] != "tap_time_len":
        raise ValueError("Top candidate is iron: fresh same-fold iron reference required")
    if manifest["spec_sha256"] != file_hash(spec_path):
        raise ValueError("Development specification changed")
    if manifest["code_sha256"] != file_hash(root / "src/bf_tap_r2/v7_periodic.py"):
        raise ValueError("Development implementation changed")
    for name, expected_hash in manifest["dependency_code_hashes"].items():
        if file_hash(root / "src/bf_tap_r2" / name) != expected_hash:
            raise ValueError(f"Development dependency implementation changed: {name}")
    versions = {p: importlib.metadata.version(p) for p in spec["runtime_versions"]}
    if versions != manifest["versions"]:
        raise ValueError("Development runtime changed")
    train = load_v5_training_frame(root)
    current_hash = hashlib.sha256(pd.util.hash_pandas_object(train, index=True).values.tobytes()).hexdigest()
    if current_hash != manifest["data_digest"]:
        raise ValueError("Development training snapshot changed")
    target, recipe = selected["target"], selected["recipe"]
    legacy = load_v5_spec(root)
    folds = {s: fold_vector(root, train, s, legacy) for s in [*spec["split_seeds"], *spec["confirmation_seeds"]]}
    dev_base, _ = references(root, train, spec)
    base = {s: dev_base[s][target] for s in spec["split_seeds"]}
    candidates = {s: np.full(len(train), np.nan) for s in folds}
    for s in spec["split_seeds"]:
        for f in range(5):
            path = dev / f"{target}-{recipe}-s{s}-f{f}.npy"
            candidates[s][folds[s] == f] = np.load(path)
    cache_hashes = {}
    meta = {"kind": "v36", "trial_id": "v36-s1-N-0048", "target": target, "line": "N"}
    for s in spec["confirmation_seeds"]:
        base[s] = np.full(len(train), np.nan)
        for f in range(5):
            path = root / f"local/runs/round2-v5-error-covariance/replication-r1/seed-{s}/fold-{f}.npz"
            mask = folds[s] == f
            b36, n = read_reference_fold(path, _identity(train, folds[s], s, f, meta), mask)
            alpha = spec["reference_time_alpha"]
            base[s][mask] = (1 - alpha) * b36 + alpha * n
            cache_hashes[str(path.relative_to(root))] = file_hash(path)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "manifest.json", {"selected": selected, "spec_sha256": file_hash(spec_path),
              "development_summary_sha256": file_hash(dev / "summary.json"),
              "code_sha256": file_hash(__file__), "reference_cache_hashes": cache_hashes,
              "versions": versions, "data_digest": current_hash})
    failures = 0
    with ProcessPoolExecutor(max_workers=spec["budget"]["workers"]) as executor:
        jobs = {executor.submit(_fit_job, (train, folds[s], target, spec["recipes"][recipe], spec["training"], f)): (s, f)
                for s in spec["confirmation_seeds"] for f in range(5)}
        for future in as_completed(jobs):
            s, f = jobs[future]
            try:
                pred, metadata = future.result()
                with (out / f"seed-{s}-fold-{f}.npy").open("xb") as stream:
                    np.save(stream, pred)
                candidates[s][folds[s] == f] = pred
                event = {"event": "complete", "seed": s, "fold": f, "metadata": metadata}
            except Exception as exc:
                failures += 1
                event = {"event": "failed", "seed": s, "fold": f, "error": repr(exc)}
            with (out / "fit_ledger.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            print(json.dumps({k: v for k, v in event.items() if k != "metadata"}), flush=True)
    if failures or any(not np.isfinite(p).all() for p in candidates.values()):
        raise RuntimeError("Incomplete confirmation; failure evidence retained")
    record = nested_blend(train[target].values, folds, base, candidates, spec["blend_grid"])
    decision = admit(record, min_seeds=4, positive_cells_min=8, positive_cells_total=10,
                     enforce_fold_criteria=False)
    development_scores = {}
    for s in spec["split_seeds"]:
        alpha = float(selected["a35_nested"]["alphas"][str(s)])
        prediction = (1 - alpha) * base[s] + alpha * candidates[s]
        development_scores[s] = package_score(wmape(train.tap_iron, dev_base[s]["tap_iron"]),
                                               wmape(train.tap_time_len, prediction))
    local_score = float(np.mean(list(development_scores.values())))
    write_new(out / "summary.json", {"candidate": recipe, "target": target, "four_seed": record,
              "seed_gate": decision, "development_package_score": local_score,
              "local_working_gate_met": local_score >= spec["promotion"]["local_working_gate"],
              "packages": 0, "uploads": 0, "release_authorized": False})
    print(json.dumps({"candidate": recipe, "decision": decision, "local_score": local_score}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        if os.environ.get(key) != "1":
            parser.error(f"Set {key}=1")
    run(Path.cwd(), args.development, args.output)


if __name__ == "__main__":
    main()
