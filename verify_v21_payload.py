#!/usr/bin/env python3
"""Verify and recover the historically registered V21 result payload.

This tool intentionally uses only the Python standard library.  The expected
digests are fixed in source and cannot be replaced from the command line.
It never predicts, fits, uploads, writes to a desktop, or overwrites output.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import zipfile


EXPECTED_V21_ZIP_SHA256 = "1a1d34ba96501da1391d2f97b237630f25661b718439efd070c7e52186d589a6"
EXPECTED_V21_PAYLOAD_SHA256 = "d5a2e118655107af39a76b99f3f1d053cc9469885d859113ba6f18ef20b9cece"
EXPECTED_ROWS = 335
HEADER = ["sample_id", "pred_tap_iron", "pred_tap_time_len"]
SIX_DECIMALS = re.compile(r"^(?:0|[1-9][0-9]*)\.[0-9]{6}$")


class PayloadError(ValueError):
    """The supplied archive or payload violates the frozen V21 contract."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_input(path: Path) -> tuple[bytes, dict]:
    if not path.is_file():
        raise PayloadError(f"input is not a file: {path}")
    archive_sha = sha256_file(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            if len(infos) != 1 or infos[0].filename != "result.csv" or infos[0].is_dir():
                raise PayloadError("V21 archive must contain exactly one top-level result.csv")
            if infos[0].flag_bits & 0x1:
                raise PayloadError("encrypted V21 payloads are not accepted")
            payload = archive.read(infos[0])
        return payload, {
            "input_kind": "zip",
            "archive_sha256": archive_sha,
            "archive_bytes": path.stat().st_size,
        }
    return path.read_bytes(), {
        "input_kind": "csv",
        "archive_sha256": None,
        "archive_bytes": None,
    }


def _parse_payload(payload: bytes, expected_rows: int) -> list[list[str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PayloadError("result.csv must be UTF-8 without an invalid byte sequence") from exc
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or rows[0] != HEADER:
        raise PayloadError(f"result.csv header must be exactly {HEADER}")
    body = rows[1:]
    if len(body) != expected_rows:
        raise PayloadError(f"result.csv must contain exactly {expected_rows} data rows")
    ids: list[str] = []
    for number, row in enumerate(body, 2):
        if len(row) != 3 or not row[0]:
            raise PayloadError(f"invalid row shape or sample_id at CSV line {number}")
        ids.append(row[0])
        for value in row[1:]:
            if not SIX_DECIMALS.fullmatch(value):
                raise PayloadError(f"prediction at CSV line {number} is not nonnegative six-decimal text")
            if not math.isfinite(float(value)):
                raise PayloadError(f"nonfinite prediction at CSV line {number}")
    if len(set(ids)) != len(ids):
        raise PayloadError("result.csv sample_id values must be unique")
    return body


def _metadata_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise PayloadError(f"test metadata is not a file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise PayloadError("test metadata lacks sample_id")
        ids = [str(row["sample_id"]) for row in reader]
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise PayloadError("test metadata requires nonempty unique sample_id values")
    return ids


def inspect_payload(
    input_path: str | Path,
    test_metadata: str | Path | None = None,
    *,
    expected_payload_sha256: str = EXPECTED_V21_PAYLOAD_SHA256,
    expected_zip_sha256: str = EXPECTED_V21_ZIP_SHA256,
    expected_rows: int = EXPECTED_ROWS,
) -> tuple[bytes, dict]:
    """Return exact payload bytes and a receipt after frozen-identity checks."""
    path = Path(input_path).resolve()
    payload, source = _read_input(path)
    payload_sha = sha256_bytes(payload)
    if payload_sha != expected_payload_sha256:
        raise PayloadError(
            f"V21 result.csv SHA-256 mismatch: expected {expected_payload_sha256}, got {payload_sha}"
        )
    body = _parse_payload(payload, expected_rows)
    ids = [row[0] for row in body]
    metadata_status = "NOT_RUN"
    metadata_identity = None
    if test_metadata is not None:
        metadata_path = Path(test_metadata).resolve()
        metadata = _metadata_ids(metadata_path)
        if metadata != ids:
            raise PayloadError("result.csv sample_id order differs from independent test metadata")
        metadata_status = "PASS_EXACT_ORDER"
        metadata_identity = {
            "path": str(metadata_path),
            "sha256": sha256_file(metadata_path),
            "rows": len(metadata),
        }
    original_archive_exact = (
        source["input_kind"] == "zip" and source["archive_sha256"] == expected_zip_sha256
    )
    receipt = {
        "kind": "V21_PAYLOAD_VERIFICATION_v1",
        "status": "PASS",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "input": str(path),
        **source,
        "expected_archive_sha256": expected_zip_sha256,
        "original_archive_exact": original_archive_exact,
        "payload_sha256": payload_sha,
        "expected_payload_sha256": expected_payload_sha256,
        "payload_exact": True,
        "rows": len(body),
        "columns": HEADER,
        "sample_ids_sha256": sha256_bytes("\n".join(ids).encode("utf-8")),
        "six_decimal_predictions": True,
        "metadata_check": metadata_status,
        "metadata": metadata_identity,
    }
    return payload, receipt


def recover_payload(
    input_path: str | Path,
    output_dir: str | Path,
    test_metadata: str | Path | None = None,
    *,
    expected_payload_sha256: str = EXPECTED_V21_PAYLOAD_SHA256,
    expected_zip_sha256: str = EXPECTED_V21_ZIP_SHA256,
    expected_rows: int = EXPECTED_ROWS,
) -> dict:
    """Recover a verified payload into a new, previously absent directory."""
    source = Path(input_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    payload, receipt = inspect_payload(
        source,
        test_metadata,
        expected_payload_sha256=expected_payload_sha256,
        expected_zip_sha256=expected_zip_sha256,
        expected_rows=expected_rows,
    )
    destination.mkdir(parents=True, exist_ok=False)
    result_path = destination / "result.csv"
    with result_path.open("xb") as handle:
        handle.write(payload)
    archive_path = destination / "Luqhhh_bf_tap_predict_prelim_TGATE_SPOUT1.zip"
    if receipt["original_archive_exact"]:
        with source.open("rb") as reader, archive_path.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        archive_identity = "ORIGINAL_ARCHIVE_BYTE_IDENTICAL_COPY"
    else:
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("result.csv", payload)
        archive_identity = "PAYLOAD_IDENTICAL_NEW_ARCHIVE"
    if sha256_file(result_path) != expected_payload_sha256:
        raise PayloadError("recovered result.csv changed during write")
    with zipfile.ZipFile(archive_path, "r") as archive:
        if archive.namelist() != ["result.csv"] or archive.read("result.csv") != payload:
            raise PayloadError("recovered archive payload differs")
    receipt.update({
        "recovery_status": archive_identity,
        "recovered_result": str(result_path),
        "recovered_result_sha256": sha256_file(result_path),
        "recovered_archive": str(archive_path),
        "recovered_archive_sha256": sha256_file(archive_path),
        "desktop_writes": 0,
        "platform_uploads": 0,
        "overwrites": 0,
    })
    receipt_path = destination / "recovery_receipt.json"
    with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--test-metadata", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        _, receipt = inspect_payload(args.input, args.test_metadata)
    else:
        receipt = recover_payload(args.input, args.output_dir, args.test_metadata)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
