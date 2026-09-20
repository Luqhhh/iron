"""Standard-library contracts for the single registered v0.36 endpoint blend."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


CANDIDATE = "V36T_V34_V30_EQUAL_BLEND"
PROTOCOL = "COMPLETE_ENDPOINT_TIME_EQUAL_BLEND_v036"
CSV_COLUMNS = ("sample_id", "pred_tap_iron", "pred_tap_time_len")
SIX_DECIMAL = re.compile(r"^(?:0|[1-9][0-9]*)\.[0-9]{6}$")
V34_RESULT_SHA256 = "80a74e687f74181ec962cc8a12328380706c0d6bbbc7d53dd932866a169d6466"
V34_ZIP_SHA256 = "e3f970fa96cad54e0a6c534cc473d3b19eeb88269cb8c2630411796bfa313924"
V30_RESULT_SHA256 = "97b6c3e0c648a0c6454cc9b36f518625487c03d1efd7621746d049457c4aed9a"
V30_ZIP_SHA256 = "fffcf23b04b3069cd71027682047eea764a2747c2dea48cca91320d67e113986"


class BlendContractError(ValueError):
    """Raised when a frozen source or output violates the v0.36 contract."""


@dataclass(frozen=True)
class PredictionRow:
    sample_id: str
    pred_tap_iron: str
    pred_tap_time_len: str


@dataclass(frozen=True)
class SourcePayload:
    role: str
    rows: tuple[PredictionRow, ...]
    payload: bytes
    result_sha256: str
    zip_sha256: str
    result_path: str | None
    zip_path: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_six(value: str) -> str:
    text = str(value)
    if SIX_DECIMAL.fullmatch(text) is None:
        raise BlendContractError("prediction must be canonical nonnegative six-decimal text")
    return text


def micro_integer(value: str) -> int:
    text = canonical_six(value)
    whole, fraction = text.split(".")
    return int(whole) * 1_000_000 + int(fraction)


def mean6(left: str, right: str) -> str:
    """Average canonical six-decimal endpoints in micro-units, ties to even."""
    total = micro_integer(left) + micro_integer(right)
    quotient, remainder = divmod(total, 2)
    if remainder and quotient % 2:
        quotient += 1
    return f"{quotient // 1_000_000}.{quotient % 1_000_000:06d}"


def parse_payload(payload: bytes, *, role: str, expected_rows: int | None = None) -> tuple[PredictionRow, ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BlendContractError(f"{role} result.csv must be UTF-8") from error
    if text.startswith("\ufeff"):
        raise BlendContractError(f"{role} result.csv must not contain a BOM")
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader)
        if tuple(header) != CSV_COLUMNS:
            raise BlendContractError(f"{role} must contain exactly {CSV_COLUMNS}")
        rows: list[PredictionRow] = []
        seen: set[str] = set()
        for position, fields in enumerate(reader, start=2):
            if len(fields) != 3:
                raise BlendContractError(f"{role} row {position} must have exactly three fields")
            sample_id, iron, time = fields
            if not sample_id or sample_id in seen:
                raise BlendContractError(f"{role} sample_id must be unique and nonempty")
            seen.add(sample_id)
            rows.append(PredictionRow(sample_id, canonical_six(iron), canonical_six(time)))
    except csv.Error as error:
        raise BlendContractError(f"{role} result.csv is not valid CSV") from error
    if not rows:
        raise BlendContractError(f"{role} result.csv must not be empty")
    if expected_rows is not None and len(rows) != expected_rows:
        raise BlendContractError(f"{role} row count differs: {len(rows)} != {expected_rows}")
    return tuple(rows)


def payload_bytes(rows: Sequence[PredictionRow]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        writer.writerow((row.sample_id, canonical_six(row.pred_tap_iron), canonical_six(row.pred_tap_time_len)))
    return handle.getvalue().encode("utf-8")


def read_zip_payload(path: Path, *, expected_zip_sha256: str, role: str) -> tuple[bytes, str]:
    if not path.is_file():
        raise BlendContractError(f"missing {role} ZIP: {path}")
    zip_sha256 = sha256_file(path)
    if zip_sha256 != expected_zip_sha256:
        raise BlendContractError(f"{role} ZIP SHA-256 differs")
    with ZipFile(path) as archive:
        if archive.namelist() != ["result.csv"]:
            raise BlendContractError(f"{role} ZIP must contain only result.csv")
        payload = archive.read("result.csv")
    return payload, zip_sha256


def load_source(
    *,
    role: str,
    result_path: Path,
    zip_path: Path,
    expected_result_sha256: str,
    expected_zip_sha256: str,
    expected_rows: int = 335,
) -> SourcePayload:
    archived_payload, zip_sha256 = read_zip_payload(zip_path, expected_zip_sha256=expected_zip_sha256, role=role)
    archived_sha256 = sha256_bytes(archived_payload)
    if archived_sha256 != expected_result_sha256:
        raise BlendContractError(f"{role} archived result.csv SHA-256 differs")
    selected_path: str | None = None
    payload = archived_payload
    if result_path.is_file():
        direct = result_path.read_bytes()
        if sha256_bytes(direct) != expected_result_sha256:
            raise BlendContractError(f"{role} result.csv SHA-256 differs")
        if direct != archived_payload:
            raise BlendContractError(f"{role} CSV and ZIP payload differ")
        payload = direct
        selected_path = str(result_path.resolve())
    rows = parse_payload(payload, role=role, expected_rows=expected_rows)
    return SourcePayload(
        role=role,
        rows=rows,
        payload=payload,
        result_sha256=archived_sha256,
        zip_sha256=zip_sha256,
        result_path=selected_path,
        zip_path=str(zip_path.resolve()),
    )


def compose_rows(v34_rows: Sequence[PredictionRow], v30_rows: Sequence[PredictionRow]) -> tuple[PredictionRow, ...]:
    if not v34_rows or len(v34_rows) != len(v30_rows):
        raise BlendContractError("endpoint row counts must be equal and nonzero")
    v30_by_id = {row.sample_id: row for row in v30_rows}
    if len(v30_by_id) != len(v30_rows) or set(v30_by_id) != {row.sample_id for row in v34_rows}:
        raise BlendContractError("endpoint sample_id sets differ")
    output: list[PredictionRow] = []
    changed = 0
    for parent in v34_rows:
        donor = v30_by_id[parent.sample_id]
        if parent.pred_tap_iron != donor.pred_tap_iron:
            raise BlendContractError("V34T and V30A iron strings differ")
        blended = mean6(parent.pred_tap_time_len, donor.pred_tap_time_len)
        low = min(micro_integer(parent.pred_tap_time_len), micro_integer(donor.pred_tap_time_len))
        high = max(micro_integer(parent.pred_tap_time_len), micro_integer(donor.pred_tap_time_len))
        if not low <= micro_integer(blended) <= high:
            raise BlendContractError("blended time lies outside endpoint interval")
        changed += blended != parent.pred_tap_time_len
        output.append(PredictionRow(parent.sample_id, parent.pred_tap_iron, blended))
    if changed == 0:
        raise BlendContractError("v0.36 candidate is identical to V34T at submission precision")
    return tuple(output)


def deterministic_zip(path: Path, payload: bytes) -> None:
    info = ZipInfo("result.csv", date_time=(2026, 9, 20, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    with ZipFile(path, "x", compression=ZIP_DEFLATED) as archive:
        archive.writestr(info, payload)


def build_package(
    *,
    v34_result: Path,
    v34_zip: Path,
    v30_result: Path,
    v30_zip: Path,
    output_dir: Path,
    expected_rows: int = 335,
) -> dict[str, object]:
    if output_dir.exists():
        raise BlendContractError(f"refusing to overwrite output directory: {output_dir}")
    v34 = load_source(
        role="V34T", result_path=v34_result, zip_path=v34_zip,
        expected_result_sha256=V34_RESULT_SHA256, expected_zip_sha256=V34_ZIP_SHA256,
        expected_rows=expected_rows,
    )
    v30 = load_source(
        role="V30A", result_path=v30_result, zip_path=v30_zip,
        expected_result_sha256=V30_RESULT_SHA256, expected_zip_sha256=V30_ZIP_SHA256,
        expected_rows=expected_rows,
    )
    rows = compose_rows(v34.rows, v30.rows)
    payload = payload_bytes(rows)
    reread = parse_payload(payload, role=CANDIDATE, expected_rows=expected_rows)
    if reread != rows:
        raise BlendContractError("serialized v0.36 payload differs after reread")
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "result.csv"
    result_path.write_bytes(payload)
    zip_path = output_dir / "Luqhhh_bf_tap_predict_prelim.zip"
    deterministic_zip(zip_path, payload)
    with ZipFile(zip_path) as archive:
        if archive.namelist() != ["result.csv"] or archive.read("result.csv") != payload:
            raise BlendContractError("v0.36 ZIP payload differs after reread")
    result_sha256 = sha256_file(result_path)
    zip_sha256 = sha256_file(zip_path)
    changed_vs_v34 = sum(a.pred_tap_time_len != b.pred_tap_time_len for a, b in zip(rows, v34.rows, strict=True))
    changed_vs_v30 = sum(a.pred_tap_time_len != b.pred_tap_time_len for a, b in zip(rows, v30.rows, strict=True))
    receipt: dict[str, object] = {
        "status": "PAYLOAD_VALIDATED_NOT_UPLOADED",
        "candidate": CANDIDATE,
        "protocol": PROTOCOL,
        "rows": len(rows),
        "source_identities": {
            "V34T": {"result_sha256": v34.result_sha256, "zip_sha256": v34.zip_sha256, "result_path": v34.result_path, "zip_path": v34.zip_path},
            "V30A": {"result_sha256": v30.result_sha256, "zip_sha256": v30.zip_sha256, "result_path": v30.result_path, "zip_path": v30.zip_path},
        },
        "result_sha256": result_sha256,
        "zip_sha256": zip_sha256,
        "iron_strings_exact_V34T_and_V30A": True,
        "mean6_integer_micro_units_ties_to_even": True,
        "time_within_endpoint_interval": True,
        "changed_time_rows_vs_V34T": changed_vs_v34,
        "changed_time_rows_vs_V30A": changed_vs_v30,
        "platform_feedback_used_to_select_endpoints": True,
        "source_model_cold_audit_completed_by_this_tool": False,
        "new_model_preprocessor_calibration_coefficient_fits": 0,
        "test_targets_read": False,
        "platform_uploads": 0,
        "desktop_writes": 0,
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def chunked_compose(v34_rows: Sequence[PredictionRow], v30_rows: Sequence[PredictionRow], chunks: Iterable[slice]) -> tuple[PredictionRow, ...]:
    output: list[PredictionRow] = []
    for view in chunks:
        output.extend(compose_rows(v34_rows[view], v30_rows[view]))
    return tuple(output)


__all__ = [
    "BlendContractError", "CANDIDATE", "CSV_COLUMNS", "PROTOCOL", "PredictionRow", "SourcePayload",
    "V30_RESULT_SHA256", "V30_ZIP_SHA256", "V34_RESULT_SHA256", "V34_ZIP_SHA256",
    "build_package", "canonical_six", "chunked_compose", "compose_rows", "load_source", "mean6",
    "micro_integer", "parse_payload", "payload_bytes", "read_zip_payload", "sha256_bytes", "sha256_file",
]
