"""Read-only evidence audit, independent of the resource admission helper."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import psutil
import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, run):
    spec = yaml.safe_load((root/"configs/round2_v36/SPEC.yaml").read_text())
    original = subprocess.check_output([
        "git", "show", f"{spec['optimization']['reference_commit']}:src/bf_tap_r2/v27_hard_tree.py"], cwd=root)
    assert hashlib.sha256(original).hexdigest() == spec["optimization"]["reference_source_sha256"]
    copied = (root/"src/bf_tap_r2/v36_hard_tree_reference.py").read_bytes()
    assert original.replace(b"docs/round2_v27_hard_tree/GRANDE_LICENSE.txt",
                            b"docs/round2_v36/GRANDE_LICENSE.txt") == copied
    assert sha(root/"docs/round2_v36/GRANDE_LICENSE.txt") == spec["author"]["license_sha256"]
    assert not list(run.glob("*.failed.json"))
    starts = list(run.glob("*.started.json"))
    assert len(starts) == 4
    assert sum(json.loads(p.read_text())["optimizer_budget_consumed"] for p in starts) == 2
    records, hashes = [], {}
    largest_difference = 0.
    for mode in ("equivalence", "resource"):
        for arm in ("GLOBAL", "INSTANCE"):
            path = run/f"{mode}-{arm}.json"
            record = json.loads(path.read_text())
            started = path.with_suffix(".started.json")
            identity = json.loads(started.read_text())
            assert identity["source_hashes"] == record["source_hashes"]
            assert identity["torch"] == spec["runtime_versions"]["torch"]
            assert identity["numpy"] == spec["runtime_versions"]["numpy"]
            for name, digest in record["source_hashes"].items():
                assert sha(root/name) == digest, name
            hashes[path.name], hashes[started.name] = sha(path), sha(started)
            assert record["mode"] == mode and record["arm"] == arm
            if mode == "equivalence":
                assert record["passed"] and len(record["cases"]) == 4
                assert {(c["training"], c["nonzero_logits"]) for c in record["cases"]} == {
                    (a, b) for a in (True, False) for b in (True, False)}
                for case in record["cases"]:
                    assert len(case["maximum_absolute_differences"]) == 6
                    largest_difference = max(largest_difference, *case["maximum_absolute_differences"].values())
                assert largest_difference == 0
            else:
                assert record["parameters"] == 1271808 and record["optimizer_runs"] == 1
                for key in ("training", "evaluation"):
                    values = record[f"{key}_seconds"]
                    assert len(values) == 20 and np.isfinite(values).all() and min(values) > 0
                    assert abs(float(np.percentile(values, 95))-record[f"{key}_p95_seconds"]) < 1e-12
                records.append(record)
    maximum_step = max(r["training_p95_seconds"] for r in records)
    maximum_eval = max(r["evaluation_p95_seconds"] for r in records)
    projected_seconds = 10*1.5*250*(16*maximum_step+2*maximum_eval)
    peak = max(r["peak_rss_mib"] for r in records)
    available = psutil.virtual_memory().available/1024**2
    checks = {"time": projected_seconds <= 21600, "worker_rss": peak <= 1536,
              "available_ram": 4*peak+1024 <= available}
    return {"audit_status": "passed", "resource_admitted": all(checks.values()),
            "decision": "resource_pass" if all(checks.values()) else "stop_G0_resource_refusal",
            "checks": checks, "projected_seconds": projected_seconds,
            "projected_hours": projected_seconds/3600, "peak_worker_rss_mib": peak,
            "available_ram_mib_at_audit": available, "maximum_equivalence_difference": largest_difference,
            "arms": [{k: r[k] for k in ("arm", "training_p95_seconds", "evaluation_p95_seconds", "peak_rss_mib")}
                     for r in records], "evidence_sha256": hashes, "resource_optimizer_runs": 2,
            "synthetic_learnability_runs": 0, "official_fits": 0, "packages": 0, "uploads": 0,
            "G1": "not_evaluated", "resource_scope": "core_only_preprocessing_and_formal_runner_unmeasured"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run = Path(args.run_dir).resolve()
    if not run.is_relative_to(root/"local/runs/round2-v36"):
        raise ValueError("Expected private V36 run")
    result = audit(root, run)
    with (run/"audit.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
