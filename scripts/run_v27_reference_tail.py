"""Execution-only V27 tail scheduling; calls the unchanged frozen unit fitter.

The original two-worker controller retains ownership of its queue. Precompute
only seed-3407 folds 1..4, well behind the currently running head. Atomic unit
directory creation prevents duplicate fitting; an active unit is never reused.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
from pathlib import Path

import yaml

from bf_tap_r2.v5_library import fold_vector, load_v5_training_frame
from bf_tap_r2.v5_spec import load_v5_spec
from bf_tap_r2.v7_periodic import digest, file_hash, write_new
from bf_tap_r2.v27_run import (
    SPEC, append_event, check_runtime, run_unit, unit_id, verified_unit, verify_hashes,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    out = root / "local/runs/round2-v27/development-r1"
    manifest = json.loads((out / "manifest.json").read_text())
    spec = yaml.safe_load((root / SPEC).read_text())
    verify_hashes(root, manifest["source_hashes"])
    verify_hashes(root, manifest["data_hashes"])
    check_runtime(spec)
    if manifest["seeds"] != [42, 3407]:
        raise ValueError("This execution amendment applies only to V27 development")
    jobs = [(f"reference-{role}-s3407-f{fold}", fold, role)
            for fold in range(1, 5) for role in ("calibration", "outer")]
    if any((out / key).exists() for key, _, _ in jobs):
        raise ValueError("Tail ownership already claimed; refuse duplicate scheduling")
    print(json.dumps({"tail_units": [j[0] for j in jobs], "additional_workers": 4,
                      "total_reference_concurrency_max": 6, "additional_fit_budget": 0}), flush=True)
    if not args.execute:
        return
    frame = load_v5_training_frame(root)
    folds = fold_vector(root, frame, 3407, load_v5_spec(root))
    if digest(folds.tolist()) != manifest["fold_hashes"]["3407"]:
        raise ValueError("Frozen fold identity mismatch")
    write_new(out / "tail-execution-amendment.json", {
        "script_sha256": file_hash(Path(__file__)), "manifest_identity": manifest["identity"],
        "planned_units": [j[0] for j in jobs], "extra_workers": 4,
        "reference_concurrency_max": 6, "additional_fit_budget": 0,
        "scientific_source_or_configuration_changed": False})
    failures = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_unit, root, out / key, frame, folds, 3407, fold, spec,
                               unit_id(manifest, key), "reference", role): key
                   for key, fold, role in jobs}
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                verified_unit(out / key, unit_id(manifest, key))
                append_event(out / "tail-execution-events.jsonl", {"event": "complete", **result})
                print(json.dumps(result), flush=True)
            except Exception as exc:
                failures.append(key)
                append_event(out / "tail-execution-events.jsonl", {"event": "failed", "key": key, "error": repr(exc)})
    if failures:
        raise RuntimeError(f"Tail failures preserved: {failures}")


if __name__ == "__main__":
    main()
