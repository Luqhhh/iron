from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from ...artifacts import atomic_write_json, create_run_directory, stable_digest
from ...exceptions import ContractError
from .config import (
    validate_optimization_candidate_config,
    validate_optimization_common_config,
)


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def initialize_optimization_run(
    root: str | Path,
    run_id: str,
    *,
    common_config: dict[str, object],
    candidate_config: dict[str, object],
    code_identity: dict[str, object],
    data_manifest: dict[str, object],
) -> Path:
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must be a safe non-empty path segment")
    validate_optimization_common_config(common_config)
    validate_optimization_candidate_config(candidate_config)
    if not isinstance(code_identity, Mapping) or not code_identity:
        raise ContractError("code_identity must be a non-empty mapping")
    if not isinstance(data_manifest, Mapping) or not data_manifest:
        raise ContractError("data_manifest must be a non-empty mapping")

    resolved = {"candidate": candidate_config, "common": common_config}
    state = {
        "status": "CREATED",
        "candidate_id": candidate_config["candidate_id"],
        "resolved_config_sha256": stable_digest(resolved),
        "code_identity": code_identity,
        "data_manifest": data_manifest,
        "data_manifest_sha256": stable_digest(data_manifest),
    }
    destination = create_run_directory(root, run_id)
    atomic_write_json(destination / "resolved_config.json", resolved)
    atomic_write_json(destination / "run_state.json", state)
    return destination
