"""Append-only serial recovery of the frozen eight-hour augmentation batch."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil
import yaml

from bf_tap_r2 import component_regularization_run as frozen
from bf_tap_r2.component_budget_resume import OLD_SPEC, NEW_SPEC, REPORT_SHA, validate_amendment
from bf_tap_r2.v49_run import append_event, verified_unit, unit_id, verify_hashes, check_runtime
from bf_tap_r2.v5_library import load_v5_training_frame, fold_vector
from bf_tap_r2.v5_spec import load_v5_spec
from bf_tap_r2.v7_periodic import digest, file_hash, write_new

RUN = Path("local/runs/component-augmentation-representation-budget8")
SOURCE = RUN / "development-r1"
RECOVERED = RUN / "development-r2"
CONTROL = RUN / "recovery-r2"
CONFIRMATION = RUN / "confirmation-r2"
ADMISSION_SHA = "77bd443c130f6eb5b3105d8d98899d98ac4751bfe5f4168037b37ae7e2829a8c"


def key_for(task):
    seed, fold, target, arm = task
    return f"{target}-{arm}-s{seed}-f{fold}"


def tasks_for(spec):
    if spec["diagnostics"].get("initialization_enabled", True):
        raise ValueError("This recovery admits no initialization diagnostics")
    return [(s, f, t, a) for s in spec["split_seeds"] for f in range(5)
            for t, arms in spec["candidates"].items() for a in arms]


def copy_verified(source, destination, manifest):
    result = verified_unit(source, unit_id(manifest, source.name))
    if result is None:
        raise ValueError(f"Missing source unit: {source}")
    shutil.copytree(source, destination)
    verified_unit(destination, unit_id(manifest, source.name))
    for path in source.iterdir():
        if not path.is_file() or file_hash(path) != file_hash(destination / path.name):
            raise ValueError("Recovery copy changed an artifact")
    return result


def validate_admission(root, spec):
    old = yaml.safe_load((root / OLD_SPEC).read_text())
    original = root / old["run_root"] / "preflight-r1/report.json"
    if file_hash(original) != REPORT_SHA:
        raise ValueError("Original failed admission changed")
    report = json.loads(original.read_text())
    validate_amendment(old, spec, report)
    verify_hashes(root, report["source_hashes"])
    admission = root / RUN / "preflight-r1/report.json"
    if file_hash(admission) != ADMISSION_SHA:
        raise ValueError("Original eight-hour admission changed")
    admitted = json.loads(admission.read_text())
    preflight_manifest = root / RUN / "preflight-r1/manifest.json"
    if file_hash(preflight_manifest) != admitted["readmission_manifest_sha256"]:
        raise ValueError("Readmission manifest changed")
    provenance = json.loads(preflight_manifest.read_text())
    verify_hashes(root, provenance["reused_artifact_hashes"])
    verify_hashes(root, admitted["source_hashes"])
    if (admitted["status"] != "passed" or not all(admitted["checks"].values())
            or admitted["spec_sha256"] != file_hash(root / NEW_SPEC)):
        raise ValueError("Successful matching eight-hour admission required")
    peak = max(r["peak_rss_mib"] for r in admitted["results"])
    if 4 * peak + 1024 >= psutil.virtual_memory().available / 1024**2:
        raise ValueError("Current available memory fails original admission rule")
    return admission


def prepare(root):
    source = root / SOURCE
    spec = yaml.safe_load((root / NEW_SPEC).read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    body = dict(manifest)
    body.pop("identity")
    if digest(body) != manifest["identity"] or manifest["development"] is not None:
        raise ValueError("Original development manifest identity mismatch")
    if (file_hash(root / NEW_SPEC) != manifest["spec_sha256"]
            or manifest["seeds"] != spec["split_seeds"]
            or manifest["candidates"] != spec["candidates"]):
        raise ValueError("Frozen experiment changed")
    verify_hashes(root, manifest["source_hashes"])
    verify_hashes(root, manifest["data_hashes"])
    if frozen.sources(root, spec) != manifest["source_hashes"]:
        raise ValueError("Frozen source inventory changed")
    if check_runtime(spec) != manifest["versions"]:
        raise ValueError("Original runtime changed")
    admission = validate_admission(root, spec)
    frozen.verify_reference_cache(root, spec)
    tasks = tasks_for(spec)
    expected = {key_for(t) for t in tasks}
    references = {f"reference-s{s}-f{f}" for s in spec["split_seeds"] for f in range(5)}
    complete = [p for p in source.iterdir() if p.is_dir() and (p / "complete.json").exists()]
    names = {p.name for p in complete}
    if not references.issubset(names) or names - expected - references:
        raise ValueError("Incomplete references or unexpected completed unit")
    for p in complete:
        verified_unit(p, unit_id(manifest, p.name))
    if (source / "summary.json").exists():
        raise ValueError("Source already complete; recovery refused")
    out = root / RECOVERED
    out.mkdir(exist_ok=False)
    shutil.copy2(source / "manifest.json", out / "manifest.json")
    reused = []
    for p in sorted(complete):
        result = copy_verified(p, out / p.name, manifest)
        if p.name in expected:
            reused.append(p.name)
            append_event(out / "fit_ledger.jsonl", {"event": "complete", "key": p.name,
                         "identity": result["identity"], "cached": True, "new_fits": 0})
    write_new(out / "recovery-provenance.json", {
        "authorization": "2026-09-29 user requested the other queue after BASE/EMA/SAM completion",
        "source": str(SOURCE), "original_manifest_sha256": file_hash(source / "manifest.json"),
        "original_preflight_sha256": file_hash(admission), "reused_units": reused,
        "pending_units": [key_for(t) for t in tasks if key_for(t) not in reused],
        "preserved_partial_units": [p.name for p in source.iterdir()
            if p.is_dir() and not (p / "complete.json").exists()],
        "recovery_script_sha256": file_hash(Path(__file__)),
        "source_manifest_unchanged": True, "automatic_retries": False})
    return spec, manifest, out, tasks


def development(root):
    spec, manifest, out, tasks = prepare(root)
    frame = load_v5_training_frame(root)
    folds = {s: fold_vector(root, frame, s, load_v5_spec(root)) for s in spec["split_seeds"]}
    for s, fv in folds.items():
        if digest(fv.tolist()) != manifest["fold_hashes"][str(s)]:
            raise ValueError("Original fold identity changed")
    remaining = [t for t in tasks if not (out / key_for(t) / "complete.json").exists()]
    failed = []
    with ProcessPoolExecutor(max_workers=spec["budget"]["candidate_workers"]) as pool:
        jobs = {pool.submit(frozen.fit_unit, root, out, frame, folds[s], s, f, t, a,
                           spec, manifest): (s, f, t, a) for s, f, t, a in remaining}
        for job in as_completed(jobs):
            try:
                append_event(out / "fit_ledger.jsonl", {"event": "complete", **job.result()})
            except Exception as exc:
                failed.append(jobs[job])
                append_event(out / "fit_ledger.jsonl", {"event": "failed", "unit": jobs[job], "error": repr(exc)})
    if failed:
        raise RuntimeError(f"Recovery failures preserved, no retry: {failed}")
    verify_hashes(root, manifest["source_hashes"])
    verify_hashes(root, manifest["data_hashes"])
    result = frozen.summarize(out, frame, folds, spec, spec["candidates"])
    write_new(out / "summary.json", result)
    print(json.dumps({"status": "complete", "new_units": len(remaining),
                      "selected": result["selected_for_confirmation"]}), flush=True)


def stage(root, control, name, command, output):
    with (control / f"{name}.log").open("x") as stream:
        child = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        append_event(control / "workflow-events.jsonl", {"event": "start", "stage": name, "pid": child.pid})
        while True:
            try:
                code = child.wait(timeout=600)
                break
            except subprocess.TimeoutExpired:
                ledger = root / output / "fit_ledger.jsonl"
                rows = [json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []
                append_event(control / "scheduled-observations.jsonl", {"stage": name, "pid": child.pid,
                    "status": "running", "completed_units": sum(r.get("event") == "complete" for r in rows),
                    "new_completed_units": sum(r.get("event") == "complete" and not r.get("cached") for r in rows),
                    "failed_units": sum(r.get("event") == "failed" for r in rows)})
        append_event(control / "workflow-events.jsonl", {"stage": name, "returncode": code})
        return code


def supervise(root):
    control = root / CONTROL
    control.mkdir(exist_ok=False)
    write_new(control / "workflow-start.json", {"pid": os.getpid(), "started_time_ns": time.time_ns(),
        "interval_seconds": 600, "serial_queue": True, "previous_queue": "completed",
        "script_sha256": file_hash(Path(__file__)), "frozen_module": frozen.__file__})
    py = sys.executable
    code = stage(root, control, "development", [py, str(Path(__file__).resolve()), "--mode", "development"], RECOVERED)
    if code == 0:
        code = stage(root, control, "development_audit", [py, "-m", "bf_tap_r2.component_regularization_audit",
            "--output", str(RECOVERED), "--spec", NEW_SPEC], RECOVERED)
    if code == 0:
        code = stage(root, control, "conditional_confirmation", [py, "-m", "bf_tap_r2.component_regularization_run",
            "--output", str(CONFIRMATION), "--development", str(RECOVERED), "--spec", NEW_SPEC], CONFIRMATION)
    if code == 0 and (root / CONFIRMATION).exists():
        code = stage(root, control, "confirmation_audit", [py, "-m", "bf_tap_r2.component_regularization_audit",
            "--output", str(CONFIRMATION), "--spec", NEW_SPEC], CONFIRMATION)
    write_new(control / "completion-event.json", {"status": "failed" if code else "completed",
        "returncode": code, "packages": 0, "uploads": 0, "automatic_restart": False})
    raise SystemExit(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["supervise", "development"], default="supervise")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if args.mode == "supervise":
        supervise(root)
    else:
        development(root)
