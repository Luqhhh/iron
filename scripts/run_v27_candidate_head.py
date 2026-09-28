"""Overlap the first four V27 candidate cells with unchanged B0 reference fits."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import json
from pathlib import Path
import time

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
        raise ValueError("Development-only scheduling amendment")
    jobs = [(f"{target}-{recipe}-s42-f{fold}", fold, target, recipe)
            for fold in range(4) for target in spec["targets"] for recipe in spec["recipes"]]
    if any((out / key).exists() for key, *_ in jobs):
        raise ValueError("Candidate head already claimed")
    print(json.dumps({"candidate_head_units": len(jobs), "workers": 4, "additional_fit_budget": 0}), flush=True)
    if not args.execute:
        return
    frame = load_v5_training_frame(root)
    folds = fold_vector(root, frame, 42, load_v5_spec(root))
    if digest(folds.tolist()) != manifest["fold_hashes"]["42"]:
        raise ValueError("Frozen fold identity mismatch")
    write_new(out / "candidate-head-amendment.json", {
        "script_sha256": file_hash(Path(__file__)), "manifest_identity": manifest["identity"],
        "planned_units": [j[0] for j in jobs], "workers": 4, "additional_fit_budget": 0,
        "scientific_source_or_configuration_changed": False})
    remaining, active, failures = list(jobs), {}, []
    with ProcessPoolExecutor(max_workers=4) as pool:
        while remaining or active:
            for job in list(remaining):
                if len(active) == 4:
                    break
                key, fold, target, recipe = job
                reference_key = f"reference-calibration-s42-f{fold}"
                if not (out / reference_key / "complete.json").exists():
                    event_file = out / reference_key / "events.jsonl"
                    if event_file.exists():
                        events = [json.loads(line) for line in event_file.read_text().splitlines()]
                        if any(e["event"] == "failed" for e in events):
                            raise RuntimeError(f"Required reference failed: {reference_key}")
                    continue
                verified_unit(out / reference_key, unit_id(manifest, reference_key))
                future = pool.submit(run_unit, root, out / key, frame, folds, 42, fold,
                                     spec, unit_id(manifest, key), "candidate", target, recipe)
                active[future] = key
                remaining.remove(job)
            completed, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED) if active else (set(), set())
            for future in completed:
                key = active.pop(future)
                try:
                    result = future.result()
                    verified_unit(out / key, unit_id(manifest, key))
                    append_event(out / "candidate-head-events.jsonl", {"event": "complete", **result})
                    print(json.dumps(result), flush=True)
                except Exception as exc:
                    failures.append(key)
                    append_event(out / "candidate-head-events.jsonl", {"event": "failed", "key": key, "error": repr(exc)})
                    print(json.dumps({"failed": key, "error": repr(exc)}), flush=True)
            if not active and remaining:
                time.sleep(5)
    if failures:
        raise RuntimeError(f"Candidate failures retained: {failures}")


if __name__ == "__main__":
    main()
