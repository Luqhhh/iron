"""Explicit, append-only recovery of the interrupted frozen component batch.

Uses a new directory and the original manifest. Completed units are verified
and copied verbatim; partial units stay in the original run. No automatic retry.
"""
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
import types

import numpy as np
import yaml

from bf_tap_r2 import component_regularization_run as frozen
from bf_tap_r2.data import TARGETS
from bf_tap_r2.v49_run import append_event, verified_unit, unit_id, verify_hashes, check_runtime
from bf_tap_r2.v5_library import load_v5_training_frame, fold_vector
from bf_tap_r2.v5_spec import load_v5_spec
from bf_tap_r2.v7_periodic import digest, file_hash, write_new

RUN = Path(frozen.RUN_ROOT)
SOURCE = RUN / "development-r1"
RECOVERED = RUN / "development-r2"
CONTROL = RUN / "recovery-r2"
CONFIRMATION = RUN / "confirmation-r2"


def key_for(task):
    seed, fold, target, arm, init = task
    return f"{target}-{arm}-s{seed}-f{fold}" + (f"-init{init}" if init != 42 else "")


def tasks_for(spec):
    regular = [(s, f, t, a, 42) for s in spec["split_seeds"] for f in range(5)
               for t, arms in spec["candidates"].items() for a in arms]
    diagnostics = [(42, 0, t, a, 1042) for t in TARGETS for a in spec["recipes"]]
    return regular, diagnostics


def copy_verified(source, destination, manifest):
    """Never import an incomplete artifact or overwrite an existing directory."""
    result = verified_unit(source, unit_id(manifest, source.name))
    if result is None:
        raise ValueError(f"Missing source unit: {source}")
    shutil.copytree(source, destination)
    verified_unit(destination, unit_id(manifest, source.name))
    for path in source.iterdir():
        if not path.is_file() or file_hash(path) != file_hash(destination / path.name):
            raise ValueError("Recovery copy changed an artifact")
    return result


def prepare(root):
    source = root / SOURCE
    spec = yaml.safe_load((root / frozen.SPEC).read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    body = dict(manifest)
    body.pop("identity")
    if digest(body) != manifest["identity"] or manifest["development"] is not None:
        raise ValueError("Original development manifest identity mismatch")
    if file_hash(root / frozen.SPEC) != manifest["spec_sha256"]:
        raise ValueError("Frozen spec changed")
    verify_hashes(root, manifest["source_hashes"])
    verify_hashes(root, manifest["data_hashes"])
    if frozen.sources(root) != manifest["source_hashes"]:
        raise ValueError("Frozen source inventory changed")
    if check_runtime(spec) != manifest["versions"]:
        raise ValueError("Original runtime changed")
    preflight = root / RUN / "preflight-r1/report.json"
    admission = json.loads(preflight.read_text())
    registered = json.loads((root / "EVIDENCE_STATUS.json").read_text())["strong_component_regularization_2026_09_29"]
    if file_hash(preflight) != registered["g0"]["report_sha256"]:
        raise ValueError("Registered original G0 report changed")
    if admission["status"] != "passed" or admission["spec_sha256"] != manifest["spec_sha256"]:
        raise ValueError("Original G0 admission required")
    frozen.verify_reference_cache(root, spec)
    regular, diagnostics = tasks_for(spec)
    expected = {key_for(t) for t in regular + diagnostics}
    complete = [p for p in source.iterdir() if p.is_dir() and (p / "complete.json").exists()]
    references = {f"reference-s{s}-f{f}" for s in spec["split_seeds"] for f in range(5)}
    if not references.issubset({p.name for p in complete}):
        raise ValueError("Original references incomplete")
    if any(p.name not in expected | references for p in complete):
        raise ValueError("Unexpected completed unit")
    # Validate every reusable unit before creating the fresh output directory.
    for p in complete:
        verified_unit(p, unit_id(manifest, p.name))
    if (source / "summary.json").exists():
        raise ValueError("Source is already complete; recovery refused")
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
    pending = [key_for(t) for t in regular + diagnostics if key_for(t) not in reused]
    write_new(out / "recovery-provenance.json", {
        "authorization": "2026-09-29 user requested serial queues, BASE/EMA/SAM first",
        "source": str(SOURCE), "original_manifest_sha256": file_hash(source / "manifest.json"),
        "original_preflight_sha256": file_hash(preflight), "reused_units": reused,
        "pending_units": pending, "preserved_partial_units": [p.name for p in source.iterdir()
        if p.is_dir() and not (p / "complete.json").exists()],
        "recovery_script_sha256": file_hash(Path(__file__)),
        "source_manifest_unchanged": True, "automatic_retries": False})
    return spec, manifest, out, regular, diagnostics


def development(root):
    spec, manifest, out, regular, diagnostics = prepare(root)
    frame = load_v5_training_frame(root)
    folds = {s: fold_vector(root, frame, s, load_v5_spec(root)) for s in spec["split_seeds"]}
    for s, fv in folds.items():
        if digest(fv.tolist()) != manifest["fold_hashes"][str(s)]:
            raise ValueError("Original fold identity changed")
    phases = [[t for t in regular if t[3] == "BASE"],
              [t for t in regular if t[3] != "BASE"], diagnostics]
    for phase, tasks in enumerate(phases):
        remaining = [t for t in tasks if not (out / key_for(t) / "complete.json").exists()]
        failed = []
        with ProcessPoolExecutor(max_workers=spec["budget"]["candidate_workers"]) as pool:
            jobs = {pool.submit(frozen.fit_unit, root, out, frame, folds[s], s, f, t, a,
                               spec, manifest, init): (s, f, t, a, init)
                    for s, f, t, a, init in remaining}
            for job in as_completed(jobs):
                try:
                    append_event(out / "fit_ledger.jsonl", {"event": "complete", **job.result()})
                except Exception as exc:
                    failed.append(jobs[job])
                    append_event(out / "fit_ledger.jsonl", {"event": "failed", "unit": jobs[job], "error": repr(exc)})
        if failed:
            raise RuntimeError(f"Recovery failures preserved, no retry: {failed}")
        print(json.dumps({"event": "phase_complete", "phase": phase, "new_units": len(remaining)}), flush=True)
    verify_hashes(root, manifest["source_hashes"])
    verify_hashes(root, manifest["data_hashes"])
    result = frozen.summarize(out, frame, folds, spec, spec["candidates"])
    base, members = frozen.collect(out, frame, folds, spec["candidates"])
    diagnostic = []
    for target in TARGETS:
        mask = folds[42] == 0
        y = frame.loc[mask, target].to_numpy()
        for arm in spec["recipes"]:
            with np.load(out / f"{target}-{arm}-s42-f0-init1042/predictions.npz") as saved:
                prediction = saved["prediction"][:, 0]
            original = members[42, target, arm][mask]
            ref = base[42][target][mask]
            old = base[42]["v12_iron" if target == "tap_iron" else "v7_time"][mask]
            before = frozen.replacement(ref, old, original)
            after = frozen.replacement(ref, old, prediction)
            diagnostic.append({"target": target, "arm": arm,
                "mean_absolute_prediction_change": float(np.abs(prediction - original).mean()),
                "standalone_score_change": float(50 * (np.abs(y-original).sum()-np.abs(y-prediction).sum()) / np.abs(y).sum()),
                "fixed_replacement_score_change": float(50 * (np.abs(y-before).sum()-np.abs(y-after).sum()) / np.abs(y).sum()),
                "scope": "single_prespecified_fold_descriptive_only"})
    result["initialization_diagnostics"] = diagnostic
    write_new(out / "summary.json", result)
    print(json.dumps({"status": "complete", "selected": result["selected_for_confirmation"]}), flush=True)


def corrected_auditor_source(source):
    old = 'fitting,fitting[outputs(target)].to_numpy(),arm,settings,spec["mechanisms"],validation)'
    new = 'fitting,training[outputs(target)].to_numpy()[inner!=0],arm,settings,spec["mechanisms"],validation)'
    if source.count(old) != 1:
        raise ValueError("Frozen auditor correction site changed")
    return source.replace(old, new)


def corrective_audit(root, out):
    """One recorded arithmetic-layout repair; every other audit check is unchanged."""
    path = root / "src/bf_tap_r2/component_regularization_audit.py"
    manifest = json.loads((root / out / "manifest.json").read_text())
    if file_hash(path) != manifest["source_hashes"][str(path.relative_to(root))]:
        raise ValueError("Frozen original auditor changed")
    corrected = corrected_auditor_source(path.read_text())
    module = types.ModuleType("bf_tap_r2.component_regularization_recovery_audit")
    module.__package__ = "bf_tap_r2"
    module.__file__ = str(Path(__file__).resolve())
    exec(compile(corrected, str(path), "exec"), module.__dict__)
    module.run(root, out)
    write_new(root / out / "audit-recovery-provenance.json", {
        "correction": "selector target array reconstructed as native y[inner != 0]",
        "original_auditor_sha256": file_hash(path), "recovery_script_sha256": file_hash(Path(__file__)),
        "effective_auditor_sha256": digest(corrected), "audit_sha256": file_hash(root / out / "audit.json"),
        "new_fits": 0, "all_other_checks_unchanged": True})


def stage(root, control, name, command, out):
    with (control / f"{name}.log").open("x") as stream:
        child = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        append_event(control / "workflow-events.jsonl", {"event": "start", "stage": name, "pid": child.pid})
        while True:
            try:
                code = child.wait(timeout=600)
                break
            except subprocess.TimeoutExpired:
                ledger = root / out / "fit_ledger.jsonl"
                rows = [json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []
                append_event(control / "scheduled-observations.jsonl", {"stage": name, "pid": child.pid,
                    "status": "running", "completed_units": sum(r.get("event") == "complete" for r in rows),
                    "failed_units": sum(r.get("event") == "failed" for r in rows)})
        append_event(control / "workflow-events.jsonl", {"stage": name, "returncode": code})
        return code


def supervise(root):
    control = root / CONTROL
    control.mkdir(exist_ok=False)
    write_new(control / "workflow-start.json", {"pid": os.getpid(), "started_time_ns": time.time_ns(),
        "interval_seconds": 600, "serial_queue": True, "other_queue": "deferred",
        "script_sha256": file_hash(Path(__file__))})
    script = str(Path(__file__).resolve())
    py = sys.executable
    code = stage(root, control, "development", [py, script, "--mode", "development"], RECOVERED)
    for label, output in [("development", RECOVERED), ("confirmation", CONFIRMATION)]:
        if code:
            break
        if label == "confirmation":
            code = stage(root, control, "conditional_confirmation", [py, "-m", "bf_tap_r2.component_regularization_run",
                "--output", str(CONFIRMATION), "--development", str(RECOVERED)], CONFIRMATION)
            if code or not (root / CONFIRMATION).exists():
                break
        code = stage(root, control, f"{label}_audit_original", [py, "-m", "bf_tap_r2.component_regularization_audit",
                     "--output", str(output)], output)
        if code:
            # Preserve the original failed audit log; never rerun training.
            code = stage(root, control, f"{label}_audit_corrective", [py, script, "--mode", "audit", "--output", str(output)], output)
    write_new(control / "completion-event.json", {"status": "failed" if code else "completed",
        "returncode": code, "packages": 0, "uploads": 0, "automatic_restart": False,
        "other_queue": "deferred"})
    raise SystemExit(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["supervise", "development", "audit"], default="supervise")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if args.mode == "supervise":
        supervise(root)
    elif args.mode == "development":
        development(root)
    else:
        if args.output not in (RECOVERED, CONFIRMATION):
            raise ValueError("Only the declared recovery outputs may be audited")
        corrective_audit(root, args.output)
