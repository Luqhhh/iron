from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd
import yaml

from .exceptions import ContractError

INFERENCE_SOURCE_NAMES = ("operation_hourly", "burden_change")


def stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_cache_key(
    *,
    code_version: str,
    data_manifest: dict[str, Any],
    feature_config: dict[str, Any],
    scenario: str,
    fit_cutoff: str,
    history_identity: str,
    sample_identity: str,
) -> str:
    return stable_digest(
        {
            "code_version": code_version,
            "data_manifest": data_manifest,
            "feature_config": feature_config,
            "scenario": scenario,
            "fit_cutoff": fit_cutoff,
            "history_identity": history_identity,
            "sample_identity": sample_identity,
        }
    )


def create_run_directory(root: str | Path, run_id: str) -> Path:
    path = Path(root) / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identities(paths: dict[str, str | Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return result


def build_inference_source_contract(
    identities: dict[str, dict[str, Any]], *, semantic_contract_sha256: str
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", semantic_contract_sha256):
        raise ContractError("semantic contract SHA256 must be lowercase hexadecimal")
    missing = set(INFERENCE_SOURCE_NAMES) - set(identities)
    if missing:
        raise ContractError(f"inference source identities missing: {sorted(missing)}")
    sources: dict[str, dict[str, Any]] = {}
    for name in INFERENCE_SOURCE_NAMES:
        identity = identities[name]
        if not isinstance(identity, dict) or not {
            "sha256",
            "bytes",
        }.issubset(identity):
            raise ContractError(f"inference source identity is invalid: {name}")
        sha256 = identity["sha256"]
        byte_count = identity["bytes"]
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ContractError(f"inference source SHA256 is invalid: {name}")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ContractError(f"inference source byte count is invalid: {name}")
        sources[name] = {"sha256": sha256, "bytes": byte_count}
    payload = {
        "schema_version": 1,
        "semantic_contract_sha256": semantic_contract_sha256,
        "sources": sources,
    }
    return {
        **payload,
        "contract_id": f"public-process-sources-v1-{stable_digest(payload)[:16]}",
    }


def validate_inference_source_contract(
    contract: dict[str, Any],
    identities: dict[str, dict[str, Any]],
    *,
    semantic_contract_sha256: str,
) -> None:
    expected_keys = {
        "schema_version",
        "contract_id",
        "semantic_contract_sha256",
        "sources",
    }
    if not isinstance(contract, dict) or set(contract) != expected_keys:
        raise ContractError("inference source contract shape is invalid")
    if not isinstance(contract["sources"], dict) or set(contract["sources"]) != set(
        INFERENCE_SOURCE_NAMES
    ):
        raise ContractError("inference source contract sources are invalid")
    for name, source in contract["sources"].items():
        if not isinstance(source, dict) or set(source) != {"sha256", "bytes"}:
            raise ContractError(f"inference source contract entry is invalid: {name}")
    rebuilt = build_inference_source_contract(
        {
            name: {
                "sha256": source["sha256"],
                "bytes": source["bytes"],
            }
            for name, source in contract["sources"].items()
        },
        semantic_contract_sha256=contract["semantic_contract_sha256"],
    )
    if rebuilt != contract:
        raise ContractError("inference source contract identity is invalid")
    if contract["semantic_contract_sha256"] != semantic_contract_sha256:
        raise ContractError("inference source contract semantic identity mismatch")
    actual = build_inference_source_contract(
        identities, semantic_contract_sha256=semantic_contract_sha256
    )
    if actual != contract:
        raise ContractError(
            "inference source identity mismatch; retrain under an explicit new source contract"
        )


def verify_file_identities(identities: dict[str, dict[str, Any]]) -> None:
    changed = []
    for name, identity in identities.items():
        path = Path(identity["path"])
        if (
            not path.is_file()
            or path.stat().st_size != identity["bytes"]
            or file_sha256(path) != identity["sha256"]
        ):
            changed.append(name)
    if changed:
        raise RuntimeError(f"input files changed during run: {sorted(changed)}")


def runtime_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "catboost": catboost.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
    }


def code_identity(repository: str | Path = ".") -> dict[str, Any]:
    root = Path(repository).resolve()

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=root, check=True, text=True, capture_output=True
        )
        return completed.stdout.strip()

    try:
        head = git("rev-parse", "HEAD")
        tree = git("rev-parse", "HEAD^{tree}")
        branch = git("branch", "--show-current")
        status = git("status", "--porcelain=v1", "--untracked-files=all")
    except (subprocess.CalledProcessError, FileNotFoundError):
        head = tree = branch = "UNAVAILABLE"
        status = "repository identity unavailable"
    source_files: list[Path] = []
    for relative in ("src", "configs", "pyproject.toml", "uv.lock"):
        candidate = root / relative
        if candidate.is_file():
            source_files.append(candidate)
        elif candidate.is_dir():
            source_files.extend(
                path
                for path in candidate.rglob("*")
                if path.is_file()
                and not path.name.endswith(".local.yaml")
                and "__pycache__" not in path.parts
            )
    snapshot = [
        {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
        for path in sorted(source_files)
    ]
    return {
        "commit": head,
        "tree": tree,
        "branch": branch,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "source_snapshot_sha256": stable_digest(snapshot),
    }


def atomic_write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
