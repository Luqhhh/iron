"""Resume E development with an explicit zero-fit admission source bridge."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from .independent_checkpoints_recovery import verify_closed_de3
from .independent_checkpoints_run import RUN_ROOT, SPEC, load_spec, private_reference, sources
from .v7_periodic import file_hash, write_new
from .v49_run import append_event, check_runtime

RESUME = f"{RUN_ROOT}/recovery-r3"
ORIGINAL_ADMISSION = f"{RUN_ROOT}/recovery-r2/preflight-E-COMPOSE"
ORIGINAL_SOURCE_COMMIT = "81c74cc"
ROUTING_SOURCE = "src/bf_tap_r2/independent_checkpoints_run.py"
ALLOWED_CHANGED = frozenset({ROUTING_SOURCE, "tests/test_independent_checkpoints.py"})


def verify_routing_only(original, current):
    """Permit one bounded reference helper and its single metadata call only."""
    trees = [ast.parse(original), ast.parse(current)]
    helpers = [n for n in trees[1].body if isinstance(n, ast.FunctionDef) and n.name == "private_reference"]
    if len(helpers) != 1:
        raise ValueError("Exactly one scoped routing helper required")
    trees[1].body.remove(helpers[0])
    for index, tree in enumerate(trees):
        count = 0
        expected = "str(preflight.relative_to(root))" if index == 0 else "private_reference(root, preflight)"
        expected_ast = ast.dump(ast.parse(expected, mode="eval").body)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict): continue
            for i, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == "preflight":
                    if ast.dump(node.values[i]) != expected_ast:
                        raise ValueError("Unexpected routing metadata change")
                    node.values[i] = ast.Constant("verified_bounded_private_reference")
                    count += 1
        if count != 1: raise ValueError("Exactly one admission path metadata field required")
    if ast.dump(trees[0]) != ast.dump(trees[1]):
        raise ValueError("Source changed beyond routing metadata; model/recipe reuse refused")


def inspect_admission(root):
    root = Path(root).resolve(); spec = load_spec(root); check_runtime(spec)
    original = root/ORIGINAL_ADMISSION
    report = json.loads((original/"report.json").read_text())
    manifest = json.loads((original/"manifest.json").read_text())
    if (report["status"] != "passed" or report["queue"] != "E-COMPOSE"
            or report["spec_sha256"] != file_hash(root/SPEC) or not all(report["checks"].values())
            or report["source_hashes"] != manifest["source_hashes"]
            or manifest["spec_sha256"] != report["spec_sha256"]
            or manifest["synthetic_domain"] != "positive_ratio_denominators"):
        raise ValueError("Original successful matching E admission required")
    changes = {}
    for name, expected in report["source_hashes"].items():
        actual = file_hash(root/name)
        if actual == expected: continue
        if name not in ALLOWED_CHANGED:
            raise ValueError(f"Undeclared admission source change: {name}")
        old = subprocess.check_output(["git", "show", f"{ORIGINAL_SOURCE_COMMIT}:{name}"], cwd=root)
        if hashlib.sha256(old).hexdigest() != expected:
            raise ValueError("Original admission source version not recoverable")
        if name == ROUTING_SOURCE: verify_routing_only(old, (root/name).read_bytes())
        changes[name] = {"original_sha256": expected, "current_sha256": actual}
    old_artifacts = {str(path.relative_to(original)): file_hash(path)
                     for path in original.rglob("*") if path.is_file()}
    frozen = sources(root)
    closed = verify_closed_de3(root)
    available = psutil.virtual_memory().available/1024**2
    if (report["projected_hours"] > spec["preflight"]["max_projected_hours"]
            or report["peak_rss_mib"] > spec["preflight"]["max_worker_rss_mib"]
            or 4*report["peak_rss_mib"]+1024 >= available):
        raise ValueError("Unchanged admission resource gates not met")
    proof = {"status": "source_and_original_artifacts_verified", "source_changes": changes,
        "original_source_commit": ORIGINAL_SOURCE_COMMIT,
        "original_report": private_reference(root, original/"report.json"),
        "original_report_sha256": file_hash(original/"report.json"),
        "original_artifact_hashes": old_artifacts, "current_source_hashes": frozen,
        "current_available_memory_mib": available, "closed_de3": closed,
        "new_training_runs": 0, "model_recipe_or_gate_changes": False}
    return report, proof


def attest(root, directory):
    original_report, proof = inspect_admission(root)
    result = subprocess.run([sys.executable, "-m", "bf_tap_r2.independent_checkpoints_preflight", "--cold",
        str(root/ORIGINAL_ADMISSION), "--queue", "E-COMPOSE"], cwd=root, capture_output=True, text=True)
    (directory/"cold-bridge.log").write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError("Fresh-process admission cold bridge failed")
    if json.loads(result.stdout) != original_report["cold_results"]:
        raise ValueError("Original admission cold result changed")
    # The copied metrics and fit counts retain their original source provenance;
    # the new report attests reuse, rather than claiming new fits with later code.
    admission = directory/"admission-bridge"; admission.mkdir(exist_ok=False)
    proof["fresh_process_cold_verified"] = True
    write_new(admission/"provenance.json", proof)
    report = copy.deepcopy(original_report)
    report["source_hashes"] = proof["current_source_hashes"]
    report["report_kind"] = "zero_fit_revalidated_original_admission"
    report["admission_reuse"] = {"original_report": proof["original_report"],
        "original_report_sha256": proof["original_report_sha256"], "new_training_runs": 0,
        "source_provenance": private_reference(root, admission/"provenance.json"),
        "source_provenance_sha256": file_hash(admission/"provenance.json"),
        "fresh_process_cold_verified": True}
    write_new(admission/"report.json", report)
    return private_reference(root, admission)


def main(check_only=False):
    root = Path.cwd()
    if check_only:
        report, proof = inspect_admission(root)
        print(json.dumps({"status": proof["status"], "source_changes": proof["source_changes"],
            "original_report_sha256": proof["original_report_sha256"], "projected_hours": report["projected_hours"],
            "original_artifacts_verified": len(proof["original_artifact_hashes"]), "new_training_runs": 0}), flush=True)
        return
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1": raise ValueError(f"Set {key}=1")
    directory = root/RESUME; directory.mkdir(parents=True, exist_ok=False)
    write_new(directory/"workflow-start.json", {"pid": os.getpid(), "time_ns": time.time_ns(),
        "queue": "E-COMPOSE", "interval_seconds": 600, "automatic_retry": False, "new_synthetic_fits": 0})
    events = []
    append_event(directory/"workflow-events.jsonl", {"event": "start", "queue": "E-COMPOSE",
        "stage": "admission_bridge", "pid": os.getpid(), "time_ns": time.time_ns()})
    try: preflight = attest(root, directory)
    except BaseException as exc:
        event = {"stage": "admission_bridge", "returncode": 1, "error": repr(exc), "time_ns": time.time_ns()}
        append_event(directory/"workflow-events.jsonl", event)
        write_new(directory/"completion-event.json", {"status": "failed", "events": [event], "automatic_retry": False})
        raise
    event = {"stage": "admission_bridge", "returncode": 0, "new_fits": 0, "time_ns": time.time_ns()}
    events.append(event); append_event(directory/"workflow-events.jsonl", event)
    development = f"{RESUME}/development-E-COMPOSE"; confirmation = f"{RESUME}/confirmation-E-COMPOSE"
    stages = [
        ("development", "independent_checkpoints_run", ["--queue", "E-COMPOSE", "--output", development, "--preflight", preflight]),
        ("development_audit", "independent_checkpoints_audit", ["--output", development]),
        ("conditional_confirmation", "independent_checkpoints_run", ["--queue", "E-COMPOSE", "--output", confirmation,
             "--development", development, "--preflight", preflight]),
        ("confirmation_audit", "independent_checkpoints_audit", ["--output", confirmation]),
    ]
    for stage, module, args in stages:
        if stage == "confirmation_audit" and not (root/confirmation).exists():
            event = {"stage": stage, "queue": "E-COMPOSE", "skipped": "no_development_finalist", "new_fits": 0, "time_ns": time.time_ns()}
            events.append(event); append_event(directory/"workflow-events.jsonl", event); continue
        with (directory/f"E-COMPOSE-{stage}.log").open("x") as stream:
            child = subprocess.Popen([sys.executable, "-m", f"bf_tap_r2.{module}", *args], cwd=root,
                                     stdout=stream, stderr=subprocess.STDOUT)
            event = {"event": "start", "queue": "E-COMPOSE", "stage": stage, "pid": child.pid, "time_ns": time.time_ns()}
            append_event(directory/"workflow-events.jsonl", event); print(json.dumps(event), flush=True)
            while True:
                try: code = child.wait(timeout=600); break
                except subprocess.TimeoutExpired:
                    observation = {"event": "scheduled_observation", "queue": "E-COMPOSE", "stage": stage,
                                   "status": "running", "pid": child.pid, "time_ns": time.time_ns()}
                    ledger = root/(confirmation if "confirmation" in stage else development)/"fit_ledger.jsonl"
                    if ledger.exists():
                        rows = [json.loads(s) for s in ledger.read_text().splitlines()]
                        observation["completed_units"] = sum(r.get("event") == "complete" for r in rows)
                        observation["completed_candidates"] = sum(r.get("event") == "complete" and r.get("kind") == "candidate" for r in rows)
                        observation["failed_units"] = sum(r.get("event") == "failed" for r in rows)
                    append_event(directory/"scheduled-observations.jsonl", observation); print(json.dumps(observation), flush=True)
        event = {"queue": "E-COMPOSE", "stage": stage, "returncode": code, "time_ns": time.time_ns()}
        events.append(event); append_event(directory/"workflow-events.jsonl", event); print(json.dumps(event), flush=True)
        if code:
            write_new(directory/"completion-event.json", {"status": "failed", "events": events, "automatic_retry": False})
            raise SystemExit(code)
    write_new(directory/"completion-event.json", {"status": "completed", "events": events, "packages": 0, "uploads": 0})
    print(json.dumps({"status": "workflow_completed", "packages": 0, "uploads": 0}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--check", action="store_true")
    main(parser.parse_args().check)
