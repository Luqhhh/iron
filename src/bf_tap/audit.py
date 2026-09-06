from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .exceptions import ContractError


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_manifest(
    path: str | Path,
    *,
    key_columns: list[str] | None = None,
    time_columns: list[str] | None = None,
    protected_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Create structural evidence without reporting protected-column statistics."""
    source = Path(path)
    protected = set(protected_columns or [])
    header = list(pd.read_csv(source, nrows=0).columns)
    safe_columns = [column for column in header if column not in protected]
    frame = pd.read_csv(
        source,
        dtype={"sample_id": "string"},
        usecols=safe_columns,
    )
    manifest: dict[str, Any] = {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
        "rows": len(frame),
        "columns": header,
        "dtypes": {c: str(t) for c, t in frame.dtypes.items()},
        "null_counts": {
            c: int(frame[c].isna().sum()) for c in frame.columns if c not in protected
        },
        "protected_columns_redacted": sorted(protected & set(header)),
        "exact_duplicate_rows": int(frame.duplicated(keep=False).sum()),
    }
    if key_columns:
        missing = set(key_columns) - set(frame.columns)
        if missing:
            raise ContractError(f"{source}: missing audit keys {sorted(missing)}")
        manifest["duplicate_key_rows"] = int(frame.duplicated(key_columns, keep=False).sum())
        duplicate_keys = frame.duplicated(key_columns, keep=False)
        manifest["conflicting_key_groups_safe_columns"] = int(
            sum(
                len(group.drop_duplicates()) > 1
                for _, group in frame.loc[duplicate_keys].groupby(key_columns, dropna=False)
            )
        )
    ranges: dict[str, dict[str, str | int]] = {}
    for column in time_columns or []:
        parsed = pd.to_datetime(frame[column], errors="coerce")
        ranges[column] = {
            "min": str(parsed.min()),
            "max": str(parsed.max()),
            "parse_failures": int(parsed.isna().sum()),
        }
    manifest["time_ranges"] = ranges
    return manifest


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
