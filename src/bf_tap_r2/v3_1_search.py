"""Round2 V3.1 search identity and budget helpers.

The first V3.1 step is deliberately small: freeze batch/trial identity rules and
the S1 budget contract before any new fit is allowed.  The actual directed
samplers are added on top of these helpers so trial identity can be checked
before any cache lookup.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

S1_LINES = (
    "time_catboost_neighborhood",
    "iron_catboost_neighborhood",
    "feature_and_expression",
    "smooth_and_residual",
    "far_family",
)


def load_config(root: Path | str) -> dict:
    path = Path(root) / "configs/round2_v3_1/search.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if spec.get("version") != "round2-v3.1-directed-search":
        raise ValueError("Unexpected V3.1 search configuration")
    validate_s1_budget(spec["s1_budget"])
    return spec


def validate_s1_budget(budget: Mapping[str, Any]) -> None:
    if int(budget.get("total", -1)) != 320:
        raise ValueError("V3.1 S1 total budget must be 320")
    totals = budget.get("target_totals", {})
    if int(totals.get("tap_iron", -1)) != 128 or int(totals.get("tap_time_len", -1)) != 192:
        raise ValueError("V3.1 S1 target totals must be 128 iron / 192 time")
    lines = budget.get("lines", {})
    line_sum = sum(int(lines.get(name, -1)) for name in S1_LINES)
    if line_sum != 320:
        raise ValueError(f"V3.1 S1 line budgets sum to {line_sum}, expected 320")
    if int(lines.get("time_catboost_neighborhood", -1)) != 128:
        raise ValueError("V3.1 time neighborhood budget must be 128")
    if int(lines.get("iron_catboost_neighborhood", -1)) != 64:
        raise ValueError("V3.1 iron neighborhood budget must be 64")
    if int(lines.get("feature_and_expression", -1)) != 48:
        raise ValueError("V3.1 feature/expression budget must be 48")
    if int(lines.get("smooth_and_residual", -1)) != 48:
        raise ValueError("V3.1 smooth/residual budget must be 48")
    if int(lines.get("far_family", -1)) != 32:
        raise ValueError("V3.1 far-family budget must be 32")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def canonical_trial_hash(trial: Mapping[str, Any]) -> str:
    payload = {k: v for k, v in trial.items() if k not in {"trial_id", "batch_id", "trial_hash"}}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def trial_identity(trial: Mapping[str, Any], *, batch_id: str, data_hash: str,
                   fold_hash: str, fold_ids: Sequence[int], model_seed: int,
                   stage: str, code_version: str) -> dict:
    required = {
        "batch_id": str(batch_id),
        "trial_hash": canonical_trial_hash(trial),
        "target": trial.get("target"),
        "data_hash": str(data_hash),
        "fold_hash": str(fold_hash),
        "fold_ids": tuple(int(v) for v in fold_ids),
        "model_seed": int(model_seed),
        "stage": str(stage),
        "code_version": str(code_version),
    }
    if not all(required[k] not in (None, "") for k in required):
        raise ValueError(f"Invalid V3.1 trial identity: {required}")
    if required["target"] not in {"tap_iron", "tap_time_len"}:
        raise ValueError(f"Invalid V3.1 target: {required['target']}")
    return required


def identity_matches(stored: Mapping[str, Any], desired: Mapping[str, Any]) -> bool:
    """Exact identity match; no silent cache reuse across batch/hash differences."""
    return dict(stored) == dict(desired)
