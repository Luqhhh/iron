"""OPT-10 authorization boundary, developed exclusively with synthetic inputs.

This module does not itself open a label file or authorize an experiment.
Authorization artifacts must come from a separate, explicit approval step.
"""
from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path

from ..artifacts import file_sha256, stable_digest
from ..exceptions import ProtectedLabelError
from ..protection import ProtectionPolicy

PURPOSES = {"holdout_scoring": "frozen_report", "final_training": "frozen_final_fit"}


def authorize_protected_operation(*, policy: ProtectionPolicy, lifecycle: str,
                                  purpose: str, frozen_manifest: str | Path,
                                  authorization: dict | None, ledger: str | Path,
                                  algorithm_identity: str) -> dict:
    if lifecycle not in policy.allowed_protected_lifecycles or purpose != PURPOSES.get(lifecycle):
        raise ProtectedLabelError("protected labels cannot be used for model selection, caches or development")
    if not isinstance(authorization, dict) or set(authorization) != {
        "schema_version", "lifecycle", "policy_digest", "frozen_manifest_sha256", "approval_id"
    }:
        raise ProtectedLabelError("explicit separate lifecycle authorization is required")
    path = Path(frozen_manifest)
    if not path.is_file():
        raise ProtectedLabelError("frozen manifest is missing")
    digest = file_sha256(path)
    if (authorization["schema_version"] != "protected-authorization-v1"
            or authorization["lifecycle"] != lifecycle
            or authorization["policy_digest"] != policy.digest
            or authorization["frozen_manifest_sha256"] != digest
            or not isinstance(authorization["approval_id"], str) or not authorization["approval_id"].strip()):
        raise ProtectedLabelError("authorization lifecycle, policy or frozen manifest mismatch")
    manifest = json.loads(path.read_text())
    if (manifest.get("schema_version") != "frozen-algorithm-v3"
            or manifest.get("algorithm_identity") != algorithm_identity
            or manifest.get("policy_digest") != policy.digest
            or manifest.get("algorithm_sha256") != stable_digest(manifest.get("algorithm"))
            or not manifest.get("algorithm")):
        raise ProtectedLabelError("frozen algorithm manifest identity mismatch")
    destination = Path(ledger)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A separate v3 JSONL ledger preserves old baseline JSON ledger semantics.
    # The OS append mode and lock prevent concurrent writers losing entries.
    with destination.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        previous = [json.loads(line) for line in handle if line.strip()]
        last = None
        for row in previous:
            body = {k: v for k, v in row.items() if k != "entry_sha256"}
            if (row.get("policy_digest") != policy.digest or row.get("previous_entry_sha256") != last
                    or row.get("entry_sha256") != stable_digest(body)):
                raise ProtectedLabelError("protected access ledger chain/policy mismatch")
            last = row["entry_sha256"]
        if any(row["approval_id"] == authorization["approval_id"] for row in previous):
            raise ProtectedLabelError("approval already consumed; do not repeat protected access")
        record = {"schema_version": "protected-ledger-v3", "policy_digest": policy.digest,
                  "lifecycle": lifecycle, "purpose": purpose, "frozen_manifest_sha256": digest,
                  "algorithm_identity": algorithm_identity, "approval_id": authorization["approval_id"],
                  "previous_entry_sha256": last, "accessed_at_utc": datetime.now(timezone.utc).isoformat()}
        record["entry_sha256"] = stable_digest(record)
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        import os
        os.fsync(handle.fileno())
        return record
