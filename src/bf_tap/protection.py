from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import stable_digest
from .exceptions import ContractError, ProtectedLabelError


def _aware_timestamp(value: object, *, name: str, timezone_name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ContractError(f"{name} must be timezone-aware")
    try:
        localized = timestamp.tz_convert(timezone_name)
    except Exception as exc:
        raise ContractError(f"{name} must use timezone {timezone_name}") from exc
    if localized.utcoffset() != timestamp.utcoffset():
        raise ContractError(f"{name} offset does not match timezone {timezone_name}")
    return localized


@dataclass(frozen=True)
class ProtectionPolicy:
    contract_id: str
    timezone: str
    development_label_end_exclusive: pd.Timestamp
    protected_start: pd.Timestamp
    protected_end: pd.Timestamp
    allowed_protected_lifecycles: tuple[str, ...]
    digest: str

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ProtectionPolicy":
        allowed = {
            "schema_version",
            "contract_id",
            "timezone",
            "development_label_end_exclusive",
            "protected_interval",
            "allowed_protected_lifecycles",
        }
        unknown = set(config) - allowed
        missing = allowed - set(config)
        if unknown or missing:
            raise ContractError(
                f"protection policy keys invalid: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if config["schema_version"] != 1:
            raise ContractError("unsupported protection policy schema_version")
        timezone_name = config["timezone"]
        if not isinstance(timezone_name, str) or not timezone_name:
            raise ContractError("protection timezone must be a non-empty string")
        interval = config["protected_interval"]
        if not isinstance(interval, dict) or set(interval) != {"start", "end"}:
            raise ContractError("protected_interval must contain exactly start and end")
        development_end = _aware_timestamp(
            config["development_label_end_exclusive"],
            name="development_label_end_exclusive",
            timezone_name=timezone_name,
        )
        protected_start = _aware_timestamp(
            interval["start"], name="protected_interval.start", timezone_name=timezone_name
        )
        protected_end = _aware_timestamp(
            interval["end"], name="protected_interval.end", timezone_name=timezone_name
        )
        if development_end != protected_start or protected_start >= protected_end:
            raise ContractError("protection boundaries must be contiguous and non-empty")
        lifecycles = config["allowed_protected_lifecycles"]
        expected = {"holdout_scoring", "final_training"}
        if not isinstance(lifecycles, list) or set(lifecycles) != expected:
            raise ContractError(f"protected lifecycles must be exactly {sorted(expected)}")
        contract_id = config["contract_id"]
        if not isinstance(contract_id, str) or not contract_id:
            raise ContractError("protection contract_id must be non-empty")
        return cls(
            contract_id=contract_id,
            timezone=timezone_name,
            development_label_end_exclusive=development_end,
            protected_start=protected_start,
            protected_end=protected_end,
            allowed_protected_lifecycles=tuple(lifecycles),
            digest=stable_digest(config),
        )

    def validate_development_read(self, requested_end: object) -> pd.Timestamp:
        boundary = _aware_timestamp(
            requested_end, name="development label read end", timezone_name=self.timezone
        )
        if boundary > self.development_label_end_exclusive:
            raise ProtectedLabelError(
                f"requested label range crosses protection boundary {self.development_label_end_exclusive}"
            )
        return boundary


def load_protection_policy(path: str | Path) -> ProtectionPolicy:
    import yaml

    source = Path(path)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError("protection policy root must be a mapping")
    return ProtectionPolicy.from_config(value)


def read_access_ledger(path: str | Path | None, policy: ProtectionPolicy) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {
            "contract_id": policy.contract_id,
            "policy_digest": policy.digest,
            "consumed": False,
            "source": "ledger_absent",
            "entries": [],
        }
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("contract_id") != policy.contract_id or value.get("policy_digest") != policy.digest:
        raise ProtectedLabelError("protection ledger does not match policy identity")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise ProtectedLabelError("protection ledger entries must be a list")
    return {**value, "consumed": bool(entries), "source": "ledger"}


def record_protected_access(
    path: str | Path,
    policy: ProtectionPolicy,
    *,
    lifecycle: str,
    frozen_manifest_sha256: str,
) -> dict[str, Any]:
    if lifecycle not in policy.allowed_protected_lifecycles:
        raise ProtectedLabelError(f"unauthorized protected-label lifecycle: {lifecycle}")
    if not re.fullmatch(r"[0-9a-f]{64}", frozen_manifest_sha256):
        raise ProtectedLabelError("a frozen manifest SHA256 is required before protected access")
    destination = Path(path)
    current = read_access_ledger(destination, policy)
    entries = list(current["entries"])
    entries.append(
        {
            "lifecycle": lifecycle,
            "frozen_manifest_sha256": frozen_manifest_sha256,
            "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    value = {
        "schema_version": 1,
        "contract_id": policy.contract_id,
        "policy_digest": policy.digest,
        "entries": entries,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return {**value, "consumed": True, "source": "ledger"}
