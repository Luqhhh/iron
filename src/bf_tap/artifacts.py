from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


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
