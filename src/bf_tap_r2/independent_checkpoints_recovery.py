"""Explicit E-only engineering recovery; original DE3 and failures stay intact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .independent_checkpoints_run import RUN_ROOT, SPEC, load_spec, select_finalists
from .v7_periodic import digest, file_hash, write_new
from .v49_run import append_event, unit_id, verified_unit, verify_hashes

RECOVERY = f"{RUN_ROOT}/recovery-r2"
ALLOWED_CHANGED_SOURCES = frozenset({
    "src/bf_tap_r2/independent_checkpoints_preflight.py",
    "src/bf_tap_r2/independent_checkpoints_run.py",
    "tests/test_independent_checkpoints.py",
})


def verify_closed_de3(root):
    """Verify old artifacts and resolve only the declared engineering source delta.

    The old result belongs to its original source freeze. No report is rewritten
    or claimed to have run with the recovery's later code. Training code, recipe,
    data, runtime, partitions and the unchanged auditor cannot cross this bridge
    with altered hashes.
    """
    root = Path(root).resolve(); out = root/RUN_ROOT/"development-DE3"
    manifest = json.loads((out/"manifest.json").read_text())
    summary = json.loads((out/"summary.json").read_text())
    audit = json.loads((out/"audit.json").read_text())
    body = dict(manifest); body.pop("identity")
    if digest(body) != manifest["identity"] or manifest["spec_sha256"] != file_hash(root/SPEC):
        raise ValueError("Original DE3 manifest/spec changed")
    if (audit["status"] != "passed" or audit["queue"] != "DE3"
            or audit["summary_sha256"] != file_hash(out/"summary.json")
            or audit["manifest_sha256"] != file_hash(out/"manifest.json")
            or audit["auditor_sha256"] != file_hash(root/"src/bf_tap_r2/independent_checkpoints_audit.py")
            or audit["cold_models"] != 120 or audit["native_replays"] != 20
            or audit["full_batch_cold_difference"] != 0):
        raise ValueError("Original independently audited DE3 result changed")
    if (summary["queue"] != "DE3" or any(summary["selected_for_confirmation"].values())
            or any(select_finalists(summary["records"], load_spec(root), "DE3").values())):
        raise ValueError("DE3 has pending earned confirmation")
    verify_hashes(root, manifest["data_hashes"])
    changes = {}
    for name, expected in manifest["source_hashes"].items():
        actual = file_hash(root/name)
        if actual == expected:
            continue
        if name not in ALLOWED_CHANGED_SOURCES:
            raise ValueError(f"Undeclared model/source change: {name}")
        original = subprocess.check_output(["git", "show", f"{manifest['starting_commit']}:{name}"], cwd=root)
        if hashlib.sha256(original).hexdigest() != expected:
            raise ValueError(f"Original frozen source cannot be recovered: {name}")
        changes[name] = {"original_sha256": expected, "recovery_sha256": actual}
    units = 0
    for path in out.rglob("complete.json"):
        directory = path.parent
        verified_unit(directory, unit_id(manifest, str(directory.relative_to(out))))
        nested = directory/"nested-hashes.json"
        if nested.exists():
            for name, expected in json.loads(nested.read_text()).items():
                if file_hash(directory/name) != expected:
                    raise ValueError("Original DE3 nested artifact changed")
        units += 1
    if units != 90:
        raise ValueError("Original DE3 complete unit coverage changed")
    preserved = {str((out/name).relative_to(root)): file_hash(out/name)
                 for name in ("manifest.json", "summary.json", "audit.json")}
    failed = root/RUN_ROOT/"preflight-E-COMPOSE"
    for path in failed.rglob("*"):
        if path.is_file(): preserved[str(path.relative_to(root))] = file_hash(path)
    for name in ("E-COMPOSE-preflight.log", "completion-event.json", "workflow-events.jsonl"):
        path = root/RUN_ROOT/name
        preserved[str(path.relative_to(root))] = file_hash(path)
    return {"status": "original_de3_and_failed_e_evidence_verified", "complete_units": units,
            "original_starting_commit": manifest["starting_commit"], "source_changes": changes,
            "preserved_evidence_hashes": preserved, "new_fits": 0,
            "original_audit_unchanged": True, "recipe_or_gate_changes": False}


def main(check_only=False):
    root = Path.cwd()
    provenance = verify_closed_de3(root)
    if check_only:
        print(json.dumps({k: v for k, v in provenance.items() if k != "preserved_evidence_hashes"}), flush=True)
        return
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1": raise ValueError(f"Set {key}=1")
    directory = root/RECOVERY; directory.mkdir(parents=True, exist_ok=False)
    write_new(directory/"engineering-provenance.json", provenance)
    write_new(directory/"workflow-start.json", {"pid": os.getpid(), "time_ns": time.time_ns(),
        "queue": "E-COMPOSE", "interval_seconds": 600, "automatic_retry": False,
        "original_de3_retrained": False, "original_failed_run_preserved": True})
    preflight = f"{RECOVERY}/preflight-E-COMPOSE"
    development = f"{RECOVERY}/development-E-COMPOSE"
    confirmation = f"{RECOVERY}/confirmation-E-COMPOSE"
    stages = [
        ("preflight", "independent_checkpoints_preflight", ["--queue", "E-COMPOSE", "--output", preflight]),
        ("development", "independent_checkpoints_run", ["--queue", "E-COMPOSE", "--output", development, "--preflight", preflight]),
        ("development_audit", "independent_checkpoints_audit", ["--output", development]),
        ("conditional_confirmation", "independent_checkpoints_run", ["--queue", "E-COMPOSE", "--output", confirmation,
             "--development", development, "--preflight", preflight]),
        ("confirmation_audit", "independent_checkpoints_audit", ["--output", confirmation]),
    ]
    events = []
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
                try:
                    code = child.wait(timeout=600); break
                except subprocess.TimeoutExpired:
                    observation = {"event": "scheduled_observation", "queue": "E-COMPOSE", "stage": stage,
                                   "status": "running", "pid": child.pid, "time_ns": time.time_ns()}
                    run = root/(confirmation if "confirmation" in stage else development if "development" in stage else preflight)
                    ledger = run/("events.jsonl" if stage == "preflight" else "fit_ledger.jsonl")
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
