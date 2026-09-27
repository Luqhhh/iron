#!/usr/bin/env python3
"""Audit the ORIGINAL V12/V7/A35 ZIPs and optionally compose their columns.

Standard-library only. Defaults to audit-only; never trains or uploads anything.
The source ZIP identities below are frozen to iron commit 931cdd1e.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
from typing import Any
import zipfile

SOURCE_HASHES = {
    "v12": "a1c205a6722da3976a12e258458b649967c7c25130a2c55d840c5ecb2a1dc669",
    "v7": "4382523c7bd688974f87eab2f42502bf8f54b2b330008b36e797ae7672490299",
    "a35": "b4e1fc2d1287a213e9d88d1c30a7420ecc222235fcc6b7191e341dfd16924b06",
}
MAX_ZIP_BYTES = 16 * 1024 * 1024
MAX_CSV_BYTES = 8 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_original(path: Path, expected_hash: str) -> tuple[list[str], list[dict[str, str]]]:
    if path.stat().st_size > MAX_ZIP_BYTES:
        raise ValueError(f"Source ZIP exceeds the size limit: {path}")
    payload = path.read_bytes()
    if sha256(payload) != expected_hash:
        raise ValueError(f"Original ZIP hash mismatch: {path}; do not substitute a regenerated ZIP.")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.namelist() != ["result.csv"]:
            raise ValueError(f"Expected exactly result.csv in {path}")
        if archive.getinfo("result.csv").file_size > MAX_CSV_BYTES:
            raise ValueError(f"Uncompressed CSV exceeds the size limit: {path}")
        text = archive.read("result.csv").decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    headers = reader.fieldnames
    if not headers or len(headers) != len(set(headers)):
        raise ValueError(f"Missing or duplicate column names: {path}")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row: {path}")
    return headers, rows


def resolve_column(headers: list[str], explicit: str | None, kind: str) -> str:
    if explicit:
        if explicit not in headers:
            raise ValueError(f"Missing requested {kind} column: {explicit}")
        return explicit
    exact = {
        "id": {"id", "sample_id", "sampleid", "样本编号", "样本id"},
        "iron": {"tap_iron", "pred_tap_iron", "tap_iron_pred", "预测出铁量", "出铁量"},
        "time": {"tap_time_len", "pred_tap_time_len", "tap_time_len_pred", "预测出铁时长", "出铁时长"},
    }[kind]
    matches = [name for name in headers if name.casefold() in exact]
    if len(matches) != 1:
        raise ValueError(f"Cannot resolve {kind} column uniquely; pass --{kind}-col. Headers: {headers}")
    return matches[0]


def validate_number(value: str, label: str) -> None:
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Non-numeric prediction: {label}") from exc
    if not number.is_finite() or number < 0:
        raise ValueError(f"Non-finite or negative prediction: {label}")


def audit_and_compose(
    paths: dict[str, Path], *, expected_hashes: dict[str, str] | None = None,
    expected_rows: int = 322, id_col: str | None = None,
    iron_col: str | None = None, time_col: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    # Nondefault hashes/count are for synthetic unit tests, not CLI source substitution.
    hashes = SOURCE_HASHES if expected_hashes is None else expected_hashes
    tables = {name: read_original(paths[name], hashes[name]) for name in ("v12", "v7", "a35")}
    headers = tables["a35"][0]
    if any(table[0] != headers for table in tables.values()):
        raise ValueError("Source headers/order differ.")
    id_col = resolve_column(headers, id_col, "id")
    iron_col = resolve_column(headers, iron_col, "iron")
    time_col = resolve_column(headers, time_col, "time")
    if len({id_col, iron_col, time_col}) != 3:
        raise ValueError("ID, iron, and time columns must be distinct.")
    reference_ids: list[str] | None = None
    for name, (_, rows) in tables.items():
        if len(rows) != expected_rows:
            raise ValueError(f"{name}: expected {expected_rows} rows; found {len(rows)}.")
        ids = [row[id_col] for row in rows]
        if any(not value.strip() for value in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{name}: blank or duplicate IDs.")
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"{name}: source ID order differs; restore originals, do not silently reorder.")
        for index, row in enumerate(rows):
            for col in (iron_col, time_col):
                validate_number(row[col], f"{name}, row {index}, {col}")
    output_rows: list[dict[str, str]] = []
    other_cols = [name for name in headers if name not in (iron_col, time_col)]
    for row12, row7, row35 in zip(tables["v12"][1], tables["v7"][1], tables["a35"][1]):
        if any(row12[col] != row7[col] or row12[col] != row35[col] for col in other_cols):
            raise ValueError("Nontarget metadata differ across source files.")
        if row12[time_col] != row35[time_col]:
            raise ValueError("V12 is not an isolated iron change: its time field differs from A35.")
        if row7[iron_col] != row35[iron_col]:
            raise ValueError("V7 is not an isolated time change: its iron field differs from A35.")
        combined = dict(row12)
        combined[time_col] = row7[time_col]  # No numeric conversion, averaging, rounding, or recalibration.
        output_rows.append(combined)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=headers, lineterminator="\n")
    writer.writeheader()
    writer.writerows(output_rows)
    result = stream.getvalue().encode("utf-8")
    reread = list(csv.DictReader(io.StringIO(result.decode("utf-8"), newline="")))
    if reread != output_rows:
        raise ValueError("Generated CSV field-string readback failed.")
    arithmetic = Decimal("96.3526") + Decimal("96.3519") - Decimal("96.3366")
    report: dict[str, Any] = {
        "status": "SOURCE_AUDIT_PASS_NOT_PLATFORM_VERIFIED",
        "repository_commit": "931cdd1ee64946fd20a7ff227a79d44ea2882298",
        "sources": {name: {"path": str(paths[name]), "zip_sha256": hashes[name]} for name in paths},
        "row_count": expected_rows,
        "columns": {"id": id_col, "iron": iron_col, "time": time_col},
        "composition": "iron field strings from V12; time field strings from V7",
        "source_isolation_checks": "PASS: V12 time == A35 time; V7 iron == A35 iron",
        "result_csv_sha256": sha256(result),
        "conditional_score_arithmetic": str(arithmetic),
        "remaining_to_96_4_arithmetic": str(Decimal("96.4") - arithmetic),
        "arithmetic_error_bound_if_each_source_within_0_0001": "0.0003",
        "arithmetic_assumptions": [
            "All reported scores use the same hidden evaluation cohort and target denominators.",
            "The platform uses the documented additive equal-weight WMAPE metric in the unclipped region.",
            "Reported scores correspond to these original ZIPs.",
        ],
        "actual_combined_platform_score": None,
        "actual_training_runs": 0,
        "uploads": 0,
        "output_zip": None,
    }
    return result, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v12", "v7", "a35"):
        parser.add_argument(f"--{name}", type=Path, required=True, help=f"Original {name.upper()} ZIP")
    parser.add_argument("--report", type=Path, required=True, help="New JSON audit report (will not overwrite)")
    for kind in ("id", "iron", "time"):
        parser.add_argument(f"--{kind}-col", default=None)
    parser.add_argument("--output-zip", type=Path, default=None,
                        help="Explicit opt-in to create one NEW combined ZIP; omit for audit-only")
    args = parser.parse_args()
    try:
        paths = {name: getattr(args, name).resolve(strict=True) for name in SOURCE_HASHES}
        destinations = [args.report] + ([args.output_zip] if args.output_zip else [])
        if len({path.resolve() for path in destinations}) != len(destinations):
            raise ValueError("Report and ZIP must have different paths.")
        if any(path.exists() for path in destinations):
            raise ValueError("A destination already exists; nothing will be overwritten.")
        payload, report = audit_and_compose(paths, id_col=args.id_col,
                                           iron_col=args.iron_col, time_col=args.time_col)
        for path in destinations:
            path.parent.mkdir(parents=True, exist_ok=True)
        if args.output_zip:
            with zipfile.ZipFile(args.output_zip, mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("result.csv", payload)
            with zipfile.ZipFile(args.output_zip) as archive:
                if archive.namelist() != ["result.csv"] or archive.read("result.csv") != payload:
                    raise ValueError("New ZIP readback failed; do not submit it.")
            report["output_zip"] = {"path": str(args.output_zip.resolve()),
                                    "sha256": sha256(args.output_zip.read_bytes())}
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(json.dumps({"status": report["status"], "report": str(args.report),
                          "output_zip": report["output_zip"],
                          "conditional_arithmetic_not_a_platform_result": report["conditional_score_arithmetic"]},
                         indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, UnicodeError, csv.Error, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Audit failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
