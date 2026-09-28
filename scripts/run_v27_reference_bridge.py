"""Fill two idle reference slots without changing the frozen V27 experiment."""
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    jobs = [(f"reference-{role}-s3407-f0", role) for role in ("calibration", "outer")]
    tail = [f"reference-{role}-s3407-f{fold}" for fold in range(1, 5)
            for role in ("calibration", "outer")]
    if manifest["seeds"] != [42, 3407] or any((out / key).exists() for key, _ in jobs):
        raise ValueError("Unexpected development identity or bridge ownership already claimed")
    print(json.dumps({"units": [key for key, _ in jobs], "workers": 2,
                      "wait_for_tail_completions": 6, "additional_fit_budget": 0}), flush=True)
    if not args.execute:
        return
    write_new(out / "bridge-execution-amendment.json", {
        "script_sha256": file_hash(Path(__file__)), "manifest_identity": manifest["identity"],
        "planned_units": [key for key, _ in jobs], "workers": 2,
        "reference_concurrency_max": 6, "additional_fit_budget": 0})
    while True:
        # Six completed tail units leave at most two tail workers. Together
        # with two primary and two bridge workers the ceiling remains six.
        done = [key for key in tail if (out / key / "complete.json").exists()]
        head_done = len(list(out.glob("reference-*-s42-f*/complete.json")))
        if head_done > 6 or any((out / key).exists() for key, _ in jobs):
            append_event(out / "bridge-execution-events.jsonl", {
                "event": "skipped", "reason": "insufficient_primary_queue_buffer"})
            print("Bridge skipped; primary controller retains both units", flush=True)
            return
        if len(done) >= 6:
            for key in done:
                verified_unit(out / key, unit_id(manifest, key))
            break
        time.sleep(15)
    frame = load_v5_training_frame(root)
    folds = fold_vector(root, frame, 3407, load_v5_spec(root))
    if digest(folds.tolist()) != manifest["fold_hashes"]["3407"]:
        raise ValueError("Frozen fold identity mismatch")
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_unit, root, out / key, frame, folds, 3407, 0, spec,
                               unit_id(manifest, key), "reference", role) for key, role in jobs]
        for future in as_completed(futures):
            result = future.result()
            append_event(out / "bridge-execution-events.jsonl", {"event": "complete", **result})
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
