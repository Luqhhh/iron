"""Compose original scored V12 iron and V7 time, without training or retuning.

Standard-library only. The public CLI pins original ZIP hashes and the official
V2 test-ID order; an arbitrary rebuilt model is not an accepted substitute.
Outputs stay below the working directory's local/ and are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

IRON_SHA = "a1c205a6722da3976a12e258458b649967c7c25130a2c55d840c5ecb2a1dc669"
TIME_SHA = "4382523c7bd688974f87eab2f42502bf8f54b2b330008b36e797ae7672490299"
ID_SHA = "1dfcce63f72ad5274e21e0e7c22e71f8dd5dc2d94d861bac2c0b65227735d83c"
COLUMNS = ("sample_id", "pred_tap_iron", "pred_tap_time_len")
ZIP_NAME = "Luqhhh_bf_tap_predict_round2.zip"
CANDIDATE = "V12_A50_IRON_V7_A50_TIME_COMBINED"
MAX_BYTES = 2 * 1024 * 1024


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def bounded_bytes(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Artifact exceeds the bounded release size")
    return raw


def zip_payload(path, expected_sha):
    raw = bounded_bytes(path)
    if sha(raw) != expected_sha:
        raise ValueError(f"Original source ZIP hash mismatch: {path}")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if (len(entries) != 1 or entries[0].filename != "result.csv"
                or entries[0].file_size > MAX_BYTES or entries[0].flag_bits & 1):
            raise ValueError("ZIP must contain only one bounded unencrypted result.csv")
        return archive.read(entries[0])  # Also verifies the entry CRC.


def read_rows(payload, *, template=False):
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig")), strict=True))
    if not rows or tuple(rows[0]) != COLUMNS:
        raise ValueError("Exact submission column order required")
    records = rows[1:]
    if len(records) != 322 or any(len(row) != 3 for row in records):
        raise ValueError("Require exactly 322 three-field rows")
    ids = [row[0] for row in records]
    if len(set(ids)) != 322 or any(not sid.startswith("R2S2_TEST_") for sid in ids):
        raise ValueError("Require unique V2 test IDs")
    if not template:
        for row in records:
            for field in row[1:]:
                if not field or field != field.strip():
                    raise ValueError("Empty or whitespace-padded prediction")
                value = float(field)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Predictions must be finite and nonnegative")
    return records


def template_ids(path):
    ids = [row[0] for row in read_rows(bounded_bytes(path), template=True)]
    if sha("\n".join(ids).encode("utf-8")) != ID_SHA:
        raise ValueError("Frozen official test-ID order mismatch")
    return ids


def compose_payload(iron_zip, time_zip, ids, *, iron_sha=IRON_SHA, time_sha=TIME_SHA):
    """Core supports synthetic identity fixtures; CLI never overrides real hashes."""
    ids = list(ids)
    if len(ids) != 322 or len(set(ids)) != 322:
        raise ValueError("Require 322 unique expected IDs")
    if Path(iron_zip).resolve() == Path(time_zip).resolve():
        raise ValueError("Two different original source packages required")
    iron = {row[0]: row for row in read_rows(zip_payload(iron_zip, iron_sha))}
    time = {row[0]: row for row in read_rows(zip_payload(time_zip, time_sha))}
    if set(iron) != set(ids) or set(time) != set(ids):
        raise ValueError("Original source IDs must match the template exactly")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows((sid, iron[sid][1], time[sid][2]) for sid in ids)
    payload = stream.getvalue().encode("utf-8")
    # Compare raw field strings, not float approximations, after CSV read-back.
    actual = read_rows(payload)
    if any(row[1] != iron[row[0]][1] or row[2] != time[row[0]][2] for row in actual):
        raise ValueError("Selected original field strings changed")
    if sha(bounded_bytes(iron_zip)) != iron_sha or sha(bounded_bytes(time_zip)) != time_sha:
        raise ValueError("Source packages changed during composition")
    return payload


def audit_release(iron_zip, time_zip, template, output):
    ids = template_ids(template)
    expected = compose_payload(iron_zip, time_zip, ids)
    raw_zip = bounded_bytes(output / ZIP_NAME)
    stored = zip_payload(output / ZIP_NAME, sha(raw_zip))
    if stored != expected or bounded_bytes(output / "result.csv") != expected:
        raise ValueError("Released ZIP/CSV differ from original-column composition")
    return {"status": "passed", "rows": 322, "unique_ids": 322,
            "iron_field_mismatches": 0, "time_field_mismatches": 0,
            "result_sha256": sha(expected), "zip_sha256": sha(raw_zip),
            "original_zip_hashes_verified": True, "template_order_verified": True,
            "official_model_fits": 0, "platform_uploads": 0}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_package(output, payload):
    read_rows(payload)
    with (output / "result.csv").open("xb") as stream:
        stream.write(payload)
    info = zipfile.ZipInfo("result.csv", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(output / ZIP_NAME, "x") as archive:
        archive.writestr(info, payload)


def build_release(iron_zip, time_zip, template, output):
    output = output.resolve()
    private = (Path.cwd() / "local").resolve()
    if output == private or not output.is_relative_to(private):
        raise ValueError("Release must be a new directory below local/")
    if output.exists():
        raise FileExistsError(output)
    ids = template_ids(template)
    payload = compose_payload(iron_zip, time_zip, ids)
    # Missing/incorrect sources fail before any release directory is created.
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_package(output, payload)
        command = [sys.executable, str(Path(__file__).resolve()), "audit",
                   "--iron-zip", str(iron_zip.resolve()), "--time-zip", str(time_zip.resolve()),
                   "--template", str(template.resolve()), "--output", str(output)]
        cold = subprocess.run(command, check=True, capture_output=True, text=True)
        verified = json.loads(cold.stdout)
        verified["fresh_process_audit_passed"] = True
        write_json(output / "verification.json", verified)
        manifest = {"candidate": CANDIDATE, "state": "ORIGINAL_COLUMNS_COMPOSED_AUDITED",
                    "sources": {"V12_iron_zip_sha256": IRON_SHA, "V7_time_zip_sha256": TIME_SHA},
                    "test_id_order_sha256": ID_SHA, "source_sha256": sha(Path(__file__).read_bytes()),
                    **verified, "platform_score": None, "new_model_fits": 0,
                    "conditional_score_arithmetic": "96.3526+96.3519-96.3366=96.3679",
                    "conditional_score_error_allowance": 0.0003,
                    "score_is_not_measured_or_guaranteed": True}
        write_json(output / "manifest.json", manifest)
        with (output / "README.txt").open("x", encoding="utf-8") as stream:
            stream.write("V12原始铁量A50 + V7原始时长A50；不重训、不重选权。\n"
                         "96.3679仅为条件算术，非平台实测分数；不能承诺96.4。用户自行上传。\n")
        return manifest
    except Exception as exc:
        write_json(output / "FAILED.json", {"error_type": type(exc).__name__, "error": str(exc)})
        raise


def deny_training_reads(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        name = str(args[0])
        if any(token in name for token in ("复赛_train", "train_samples.csv", "train_features.csv", "tap_history_train.csv")):
            raise RuntimeError("Composition must not read training data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "audit"])
    parser.add_argument("--iron-zip", type=Path, required=True)
    parser.add_argument("--time-zip", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path("复赛_test/result_template.csv"))
    parser.add_argument("--output", type=Path, default=Path("local/runs/round2-v12-v7-combined/release-r1"))
    args = parser.parse_args()
    sys.addaudithook(deny_training_reads)
    result = (build_release if args.command == "build" else audit_release)(
        args.iron_zip, args.time_zip, args.template, args.output)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
