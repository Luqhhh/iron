from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest

MODULE = Path(__file__).resolve().parents[1] / "verify_v21_payload.py"
SPEC = importlib.util.spec_from_file_location("verify_v21_payload", MODULE)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def payload(rows=("A", "B")) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(tool.HEADER)
    for index, sample_id in enumerate(rows, 1):
        writer.writerow([sample_id, f"{500 + index:.6f}", f"{100 + index:.6f}"])
    return handle.getvalue().encode()


def inspect(path: Path, raw: bytes, metadata=None):
    return tool.inspect_payload(
        path, metadata,
        expected_payload_sha256=hashlib.sha256(raw).hexdigest(),
        expected_zip_sha256="f" * 64,
        expected_rows=2,
    )


def test_valid_csv(tmp_path):
    raw = payload(); path = tmp_path / "result.csv"; path.write_bytes(raw)
    got, receipt = inspect(path, raw)
    assert got == raw and receipt["payload_exact"] and receipt["metadata_check"] == "NOT_RUN"


def test_valid_zip(tmp_path):
    raw = payload(); path = tmp_path / "v21.zip"
    with zipfile.ZipFile(path, "w") as archive: archive.writestr("result.csv", raw)
    got, receipt = inspect(path, raw)
    assert got == raw and receipt["input_kind"] == "zip"


def test_original_zip_identity(tmp_path):
    raw = payload(); path = tmp_path / "v21.zip"
    with zipfile.ZipFile(path, "w") as archive: archive.writestr("result.csv", raw)
    digest = tool.sha256_file(path)
    _, receipt = tool.inspect_payload(path, expected_payload_sha256=hashlib.sha256(raw).hexdigest(), expected_zip_sha256=digest, expected_rows=2)
    assert receipt["original_archive_exact"]


def test_wrong_payload_digest_rejected(tmp_path):
    path = tmp_path / "result.csv"; path.write_bytes(payload())
    with pytest.raises(tool.PayloadError, match="SHA-256 mismatch"):
        tool.inspect_payload(path, expected_payload_sha256="0" * 64, expected_rows=2)


def test_zip_extra_member_rejected(tmp_path):
    raw = payload(); path = tmp_path / "v21.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("result.csv", raw); archive.writestr("extra.txt", b"x")
    with pytest.raises(tool.PayloadError, match="exactly one"):
        inspect(path, raw)


def test_duplicate_ids_rejected(tmp_path):
    raw = payload(("A", "A")); path = tmp_path / "result.csv"; path.write_bytes(raw)
    with pytest.raises(tool.PayloadError, match="unique"):
        inspect(path, raw)


def test_bad_header_rejected(tmp_path):
    raw = payload().replace(b"pred_tap_iron", b"iron", 1); path = tmp_path / "result.csv"; path.write_bytes(raw)
    with pytest.raises(tool.PayloadError, match="header"):
        inspect(path, raw)


def test_wrong_row_count_rejected(tmp_path):
    raw = payload(("A",)); path = tmp_path / "result.csv"; path.write_bytes(raw)
    with pytest.raises(tool.PayloadError, match="exactly 2"):
        tool.inspect_payload(path, expected_payload_sha256=hashlib.sha256(raw).hexdigest(), expected_rows=2)


def test_non_six_decimal_rejected(tmp_path):
    raw = payload().replace(b"501.000000", b"501.0", 1); path = tmp_path / "result.csv"; path.write_bytes(raw)
    with pytest.raises(tool.PayloadError, match="six-decimal"):
        inspect(path, raw)


def test_negative_rejected(tmp_path):
    raw = payload().replace(b"501.000000", b"-1.000000", 1); path = tmp_path / "result.csv"; path.write_bytes(raw)
    with pytest.raises(tool.PayloadError, match="six-decimal"):
        inspect(path, raw)


def test_metadata_order_checked(tmp_path):
    raw = payload(); result = tmp_path / "result.csv"; result.write_bytes(raw)
    metadata = tmp_path / "metadata.csv"; metadata.write_text("sample_id\nB\nA\n")
    with pytest.raises(tool.PayloadError, match="order"):
        inspect(result, raw, metadata)


def test_recovery_new_archive_and_no_overwrite(tmp_path):
    raw = payload(); result = tmp_path / "result.csv"; result.write_bytes(raw)
    output = tmp_path / "recovered"
    receipt = tool.recover_payload(result, output, expected_payload_sha256=hashlib.sha256(raw).hexdigest(), expected_zip_sha256="f" * 64, expected_rows=2)
    assert receipt["recovery_status"] == "PAYLOAD_IDENTICAL_NEW_ARCHIVE"
    assert json.loads((output / "recovery_receipt.json").read_text())["payload_exact"]
    with zipfile.ZipFile(receipt["recovered_archive"]) as archive:
        assert archive.namelist() == ["result.csv"] and archive.read("result.csv") == raw
    with pytest.raises(FileExistsError):
        tool.recover_payload(result, output, expected_payload_sha256=hashlib.sha256(raw).hexdigest(), expected_rows=2)
