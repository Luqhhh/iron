"""Immutable execution identities for the bounded RFM design.

Preparing a manifest reads references but never fits a model or admits a run.
The controller, preflight and independent audit must exist before freezing.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys

import numpy as np

from .data import TARGETS
from .rfm_model import array_digest
from .rfm_protocol import canonical, file_hash, write_new
from .rfm_reference import load_reference_cache

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
REQUIRED_FILES = (
    "src/bf_tap_r2/rfm_run.py", "src/bf_tap_r2/rfm_preflight.py",
    "scripts/run_rfm_frozen.py", "scripts/monitor_rfm_once.py",
    "configs/rfm_metric_learning/SPEC.yaml",
)


def require_threads():
    if any(os.environ.get(v) != "1" for v in THREAD_VARS):
        raise ValueError("all four numerical thread variables must be pinned to 1")


def private_path(workspace, path):
    workspace, path = Path(workspace).resolve(), Path(path).absolute()
    resolved = path.resolve()
    if (not resolved.is_relative_to(workspace / "local")
            or resolved == workspace / "local" or resolved != path):
        raise ValueError("execution artifacts must use a direct private workspace path")
    return resolved


def source_snapshot(workspace):
    workspace = Path(workspace).resolve()
    files = {p for base in ("src", "configs", "scripts", "tests")
             for p in (workspace / base).rglob("*")
             if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".json")}
    files.update(workspace / p for p in ("uv.lock", "pyproject.toml"))
    if any(p.is_symlink() for p in files):
        raise ValueError("source snapshot cannot follow symlinks")
    return {str(p.relative_to(workspace)): file_hash(p) for p in sorted(files)}


def runtime_snapshot():
    require_threads()
    if sys.version_info[:2] != (3,12):
        raise ValueError("locked Python 3.12 required")
    return {"python":platform.python_version(), "executable":str(Path(sys.executable).resolve()),
            "packages":{d.metadata["Name"].lower():d.version for d in importlib.metadata.distributions()},
            "threads":{k:os.environ[k] for k in THREAD_VARS}}


def reference_identity(frame, folds, current, historical, audit):
    if audit.get("status") != "passed" or audit.get("new_reference_fits") != 0:
        raise ValueError("verified zero-fit native references required")
    if set(folds) != {42,3407,7777,12011}:
        raise ValueError("all four frozen reference seeds required")
    for seed, fv in folds.items():
        fv = np.asarray(fv)
        if fv.shape != (len(frame),) or set(fv) != set(range(5)):
            raise ValueError("incomplete reference folds")
        if any(np.count_nonzero(fv == f) > 551 or np.count_nonzero(fv != f) > 2204 for f in range(5)):
            raise ValueError("official partitions exceed the frozen synthetic row bounds")
    return {"audit":audit, "rows":len(frame),
            "row_ids_sha256":hashlib.sha256(canonical(frame.sample_id.tolist())).hexdigest(),
            "columns":{route:{str(s):{t:array_digest(values[s][t]) for t in TARGETS} for s in folds}
                       for route,values in (("current",current),("historical",historical))}}


def load_anchored(path, expected_sha256):
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or file_hash(path) != expected_sha256:
        raise ValueError("external artifact identity mismatch")
    return json.loads(Path(path).read_text())


def verify_manifest(path, expected_sha256, *, references=True):
    manifest = load_anchored(path, expected_sha256)
    workspace, root = Path(manifest["workspace"]), Path(manifest["reference_root"])
    if Path(__file__).resolve() != workspace / "src/bf_tap_r2/rfm_freeze.py":
        raise ValueError("RFM imported from another execution worktree")
    if source_snapshot(workspace) != manifest["source_hashes"]:
        raise ValueError("frozen execution source/configuration changed")
    if runtime_snapshot() != manifest["runtime"]:
        raise ValueError("frozen runtime changed")
    test = load_anchored(manifest["locked_tests"]["path"],manifest["locked_tests"]["sha256"])
    if test != manifest["locked_tests"]["receipt"]:
        raise ValueError("locked suite evidence changed")
    if references:
        for relative, sha in manifest["reference"]["audit"]["hashes"].items():
            path = root / relative
            if not path.resolve().is_relative_to(root) or file_hash(path) != sha:
                raise ValueError("frozen native reference changed: " + relative)
    return manifest


def reload_references(manifest):
    data = load_reference_cache(manifest["reference_root"])
    if reference_identity(*data) != manifest["reference"]:
        raise ValueError("native reference arithmetic/identity drift")
    return data


def prepare_manifest(workspace, reference_root, output, *, tests_path, tests_sha256):
    workspace, reference_root = Path(workspace).resolve(), Path(reference_root).resolve()
    output = private_path(workspace, output)
    if output.exists():
        raise FileExistsError(output)
    for name in REQUIRED_FILES:
        if not (workspace/name).is_file():
            raise ValueError("complete implementation required before freeze: " + name)
    sources = source_snapshot(workspace)
    # The receipt is produced by the locked full-suite invocation, with its
    # exact source and runtime identities, then anchored by the caller.
    tests = load_anchored(tests_path,tests_sha256)
    runtime = runtime_snapshot()
    if (tests.get("status") != "passed" or tests.get("exit_code") != 0
            or tests.get("full_suite") is not True or tests.get("source_hashes") != sources
            or tests.get("runtime") != runtime):
        raise ValueError("passed full locked suite for these exact sources/runtime required")
    changed = subprocess.check_output(["git","status","--porcelain","--",*sources],cwd=workspace,text=True)
    if changed.strip():
        raise ValueError("commit complete tested source before manifest freeze")
    data = load_reference_cache(reference_root)
    result = {"version":1,"workspace":str(workspace),"reference_root":str(reference_root),
              "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=workspace,text=True).strip(),
              "source_hashes":sources,"runtime":runtime,"reference":reference_identity(*data),
              "locked_tests":{"path":str(Path(tests_path).resolve()),"sha256":tests_sha256,"receipt":tests},
              "release_authorized":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    write_new(output,result)
    return {"path":str(output),"sha256":file_hash(output)}
