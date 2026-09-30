"""Synthetic checks for exact source identity and isolated package arithmetic."""
import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "today_five_builder", Path(__file__).resolve().parents[1] / "scripts/build_today_five.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@pytest.fixture
def ids():
    return [f"R2S2_TEST_{i:012X}" for i in range(322)]


@pytest.fixture
def frames(ids):
    return {name: [dict(zip(builder.HEADER, (sid, iron, time))) for sid in ids]
            for name, iron, time in (("DE3", "501.000000", "100"),
                                     ("Q75", "500.0000", "101.000000"),
                                     ("Q100", "500.0000", "105"),
                                     ("RESERVE", "501.000000", "101.000000"),
                                     ("PTARL_TIME_Q20", "500.0000", "93"))}


def test_composition_keeps_exact_original_field_strings(ids, frames):
    blobs = builder.composites(frames)
    parsed = [builder.parse(blob, ids) for blob in blobs]
    assert parsed[0][0]["pred_tap_iron"] == "501.000000"
    assert float(parsed[0][0]["pred_tap_time_len"]) == 97
    assert float(parsed[1][0]["pred_tap_time_len"]) == 103
    assert all(r["pred_tap_iron"] == "501.000000" for data in parsed for r in data)


@pytest.mark.parametrize("bad", ["nan", "inf", "-1", "1e999"])
def test_nonfinite_and_negative_predictions_refused(ids, frames, bad):
    frames["Q75"][0]["pred_tap_time_len"] = bad
    with pytest.raises(ValueError):
        builder.parse(builder.csv_bytes(frames["Q75"]), ids)


@pytest.mark.parametrize("corruption", ["reordered", "duplicate", "missing", "extra", "foreign"])
def test_ids_and_order_are_checked_against_official_template(ids, frames, corruption):
    rows = frames["Q75"]
    if corruption == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif corruption == "duplicate":
        rows[0]["sample_id"] = rows[1]["sample_id"]
    elif corruption == "missing":
        rows.pop()
    elif corruption == "extra":
        rows.append(rows[-1])
    else:
        rows[0]["sample_id"] = "R2S_TEST_FFFFFFFFFFFF"
    with pytest.raises(ValueError):
        builder.parse(builder.csv_bytes(rows), ids)


def test_blends_refuse_different_source_iron_fields(frames):
    frames["PTARL_TIME_Q20"][0]["pred_tap_iron"] = "499"
    with pytest.raises(ValueError, match="share"):
        builder.composites(frames)


def test_zip_requires_known_hash_and_only_result_csv(tmp_path, ids, frames):
    path = tmp_path / "package.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("result.csv", builder.csv_bytes(frames["Q75"]))
    builder.load_zip(path, ids, builder.digest(path))
    with pytest.raises(ValueError, match="identity"):
        builder.load_zip(path, ids, "0" * 64)
    with zipfile.ZipFile(path, "a") as z:
        z.writestr("extra.txt", "not part of a competition submission")
    with pytest.raises(ValueError, match="contents"):
        builder.load_zip(path, ids)


@pytest.fixture
def received(tmp_path, monkeypatch, frames):
    """An artificial complete release, with test-only substituted original hashes."""
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_zip(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = builder.csv_bytes(rows)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("result.csv", data)
        return data

    root = tmp_path / "root"
    originals = {}
    for name in builder.ORIGINALS:
        relative = "inputs/" + name + ".zip"
        path = root / relative
        write_zip(path, frames[name])
        originals[name] = (builder.digest(path), relative)
    monkeypatch.setattr(builder, "ORIGINALS", originals)
    status = root / "EVIDENCE_STATUS.json"
    write_json(status, {"round2_current_platform_best": builder.CURRENT})
    template = root / "复赛_test/result_template.csv"
    template.parent.mkdir()
    template.write_bytes(builder.csv_bytes(frames["Q75"]))
    release = root / "release"
    entries = []
    for name in builder.NEW_NAMES:
        rows = [r.copy() for r in frames["Q75"]]
        column = builder.HEADER[1] if name == builder.NEW_NAMES[0] else builder.HEADER[2]
        for row in rows:
            row[column] = "502" if column == builder.HEADER[1] else "93"
        work = release / name
        blob = write_zip(work / "package" / builder.ZIP, rows)
        sha = builder.digest(work / "package" / builder.ZIP)
        write_json(work / "audit.json", {"G0": "passed"})
        write_json(work / "cold.json", {"G0": "passed", "training_reads_prohibited": True})
        write_json(work / "independent-package-audit.json", {
            "G0": "passed", "rows": 322, "new_fits": 0,
            "unchanged_field_mismatches": 0, "training_reads_prohibited": True,
            "cold_csv_byte_identical": True, "zip_sha256": sha,
        })
        entries.append({"candidate": name, "zip_sha256": sha,
                        "csv_sha256": hashlib.sha256(blob).hexdigest(),
                        "audit_sha256": builder.digest(work / "audit.json"),
                        "cold_sha256": builder.digest(work / "cold.json")})
    write_json(release / "manifest.json", {"best": builder.CURRENT})
    write_json(release / "release-summary.json", {"comparison": builder.CURRENT, "packages": entries})
    write_json(release / "completion-event.json", {"status": "completed", "packages": 2})
    return SimpleNamespace(artifact_root=root, input_dir=None, release_dir=release,
                           status_file=None, output=root / "local/output")


def test_complete_evidence_and_column_checks_pass(received):
    ids, frames, payloads, inventory, _ = builder.collect(received)
    assert len(ids) == 322 and len(payloads) == 6
    assert len(builder.composites(frames)) == 2
    assert str(Path(builder.__file__).resolve()) in inventory
    assert not received.output.exists()


@pytest.mark.parametrize("corruption", ["absent_original", "unfinished", "stale_reference", "audit_changed", "wrong_other_column"])
def test_incomplete_or_corrupted_scientific_evidence_never_creates_output(received, corruption):
    root, release = received.artifact_root, received.release_dir
    if corruption == "absent_original":
        (root / "inputs/DE3.zip").unlink()
    elif corruption == "unfinished":
        (release / "completion-event.json").write_text('{"status":"running","packages":0}')
    elif corruption == "stale_reference":
        (root / "EVIDENCE_STATUS.json").write_text('{"round2_current_platform_best":{"candidate":"DE3"}}')
    elif corruption == "audit_changed":
        (release / builder.NEW_NAMES[0] / "audit.json").write_text('{"G0":"failed"}')
    else:
        path = release / builder.NEW_NAMES[0] / "independent-package-audit.json"
        data = json.loads(path.read_text())
        data["unchanged_field_mismatches"] = 1
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        builder.collect(received)
    assert not received.output.exists()


@pytest.mark.parametrize("corrupt_arithmetic", [False, True])
def test_independent_five_pack_readback_and_rehashed_arithmetic_corruption(received, corrupt_arithmetic):
    ids, frames, payloads, inventory, _ = builder.collect(received)
    derived = builder.composites(frames)
    blobs = [payloads["PTARL_TIME_Q20"], payloads["EMA_IRON_EMA_TIME"],
             derived[0], derived[1], payloads["RESERVE"]]
    if corrupt_arithmetic:
        rows = builder.parse(blobs[2], ids)
        rows[0]["pred_tap_time_len"] = "96"
        blobs[2] = builder.csv_bytes(rows)
    out = received.output
    out.mkdir(parents=True)
    records = []
    for name, data in zip(builder.OUTPUT_NAMES, blobs, strict=True):
        folder = out / name
        folder.mkdir()
        path = folder / builder.ZIP
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("result.csv", data)
        records.append({"candidate": name, "sha256": builder.digest(path)})
    (out / "receipt.json").write_text(json.dumps({"source_inventory": inventory, "packages": records}))
    if corrupt_arithmetic:
        with pytest.raises(ValueError, match="arithmetic"):
            builder.verify(out, ids, frames, payloads, inventory)
    else:
        report = builder.verify(out, ids, frames, payloads, inventory)
        assert report["packages"] == 5
        assert report["candidate_05"] == "reserve_not_in_current_upload_queue"
