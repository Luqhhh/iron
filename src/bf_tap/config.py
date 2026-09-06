from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .exceptions import ContractError


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"configuration root must be a mapping: {source}")
    if value.get("schema_version") != 1:
        raise ContractError(f"unsupported schema_version in {source}")
    return value


def require_real_data_contract(config: dict[str, Any]) -> None:
    missing: list[str] = []
    for name, path in config.get("paths", {}).items():
        if path is None:
            missing.append(f"paths.{name}")
    if config.get("supervised_target_available_at") is None:
        missing.append("supervised_target_available_at")
    for name, source in config.get("sources", {}).items():
        if not source.get("available_at_column"):
            missing.append(f"sources.{name}.available_at_column")
    if missing:
        raise ContractError(
            "real-data availability contract is incomplete: " + ", ".join(missing)
        )
