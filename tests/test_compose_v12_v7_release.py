import csv
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compose_v12_v7_release.py"
spec = importlib.util.spec_from_file_location("original_column_composer", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, *, mutation=None):
    ids = [f"R2S2_TEST_{i:012X}" for i in range(322)]
    iron = [[sid, f"{i + 400}.000000000000001", "80.000"] for i, sid in enumerate(ids)]
    time = [[sid, "410.000", f"{i + 80}.123450000000000"] for i, sid in enumerate(ids)]
    if mutation:
        mutation(iron, time)
    paths, hashes = [], []
    for name, rows in [("iron", iron), ("time", time)]:
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(module.COLUMNS)
        writer.writerows(rows)
        path = tmp_path / f"{name}.zip"
        with zipfile.ZipFile(path, "x") as archive:
            archive.writestr("result.csv", stream.getvalue().encode())
        paths.append(path)
        hashes.append(module.sha(path.read_bytes()))
    return ids, paths, hashes


def compose(ids, paths, hashes):
    return module.compose_payload(*paths, ids, iron_sha=hashes[0], time_sha=hashes[1])


def test_original_field_text_and_template_order_are_exact(tmp_path):
    ids, paths, hashes = fixture(tmp_path, mutation=lambda iron, time: time.reverse())
    before = [path.read_bytes() for path in paths]
    payload = compose(ids, paths, hashes)
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    assert [row["sample_id"] for row in rows] == ids
    for i, row in enumerate(rows):
        assert row["pred_tap_iron"] == f"{i + 400}.000000000000001"
        assert row["pred_tap_time_len"] == f"{i + 80}.123450000000000"
    assert before == [path.read_bytes() for path in paths]
    assert compose(ids, paths, hashes) == payload


@pytest.mark.parametrize("field", ["", "nan", "inf", "-1", " 1", "=SUM(1)"])
def test_invalid_source_numbers_are_rejected(tmp_path, field):
    ids, paths, hashes = fixture(tmp_path, mutation=lambda iron, time: time[0].__setitem__(2, field))
    with pytest.raises(ValueError):
        compose(ids, paths, hashes)


@pytest.mark.parametrize("kind", ["duplicate", "missing", "extra", "different", "old_snapshot"])
def test_id_boundaries_are_rejected(tmp_path, kind):
    def mutate(iron, time):
        if kind == "duplicate":
            time[0][0] = time[1][0]
        elif kind == "missing":
            time.pop()
        elif kind == "extra":
            time[0].append("99")
        elif kind == "different":
            time[0][0] = "R2S2_TEST_FFFFFFFFFFFF"
        else:
            time[0][0] = "R2S_TEST_OLD"
    ids, paths, hashes = fixture(tmp_path, mutation=mutate)
    with pytest.raises(ValueError):
        compose(ids, paths, hashes)


def test_original_hash_is_mandatory_and_role_cannot_be_swapped(tmp_path):
    ids, paths, hashes = fixture(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        module.compose_payload(*paths, ids)
    with pytest.raises(ValueError, match="hash mismatch"):
        module.compose_payload(paths[1], paths[0], ids, iron_sha=hashes[0], time_sha=hashes[1])
    with pytest.raises(ValueError, match="different"):
        module.compose_payload(paths[0], paths[0], ids, iron_sha=hashes[0], time_sha=hashes[0])


@pytest.mark.parametrize("name", ["../result.csv", "extra.json", "result.csv/result.csv"])
def test_noncanonical_archives_are_rejected(tmp_path, name):
    path = tmp_path / "invalid.zip"
    with zipfile.ZipFile(path, "x") as archive:
        archive.writestr(name, "ignored")
    with pytest.raises(ValueError, match="only one"):
        module.zip_payload(path, module.sha(path.read_bytes()))


def test_existing_outputs_and_public_paths_are_not_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "missing.zip"
    with pytest.raises(ValueError, match="below local"):
        module.build_release(missing, missing, missing, tmp_path / "public")
    with pytest.raises(ValueError, match="below local"):
        module.build_release(missing, missing, missing, tmp_path / "local")
    existing = tmp_path / "local/existing"
    existing.mkdir(parents=True)
    sentinel = existing / "original.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        module.build_release(missing, missing, missing, existing)
    assert sentinel.read_text() == "keep"


def test_cli_missing_material_creates_no_package(tmp_path):
    template = SCRIPT.parents[1] / "复赛_test/result_template.csv"
    output = tmp_path / "local/release"
    result = subprocess.run([sys.executable, str(SCRIPT), "build", "--iron-zip", "absent-iron.zip",
                             "--time-zip", "absent-time.zip", "--template", str(template),
                             "--output", str(output)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
    assert not output.exists()


def test_official_template_order_and_training_read_guard():
    ids = module.template_ids(SCRIPT.parents[1] / "复赛_test/result_template.csv")
    assert len(ids) == 322
    with pytest.raises(RuntimeError):
        module.deny_training_reads("open", ("复赛_train/train_samples.csv",))


def test_template_id_reordering_is_not_accepted(tmp_path):
    template = SCRIPT.parents[1] / "复赛_test/result_template.csv"
    rows = list(csv.reader(io.StringIO(template.read_text(encoding="utf-8-sig"))))
    rows[1], rows[2] = rows[2], rows[1]
    path = tmp_path / "template.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError, match="order mismatch"):
        module.template_ids(path)


def test_zip_is_deterministic_readback_exact_and_never_overwritten(tmp_path):
    ids, paths, hashes = fixture(tmp_path)
    payload = compose(ids, paths, hashes)
    archives = []
    for i in range(2):
        folder = tmp_path / f"output-{i}"
        folder.mkdir()
        module.write_package(folder, payload)
        archive = folder / module.ZIP_NAME
        archives.append(archive.read_bytes())
        with zipfile.ZipFile(archive) as handle:
            assert handle.namelist() == ["result.csv"]
            assert handle.read("result.csv") == payload
            assert handle.testzip() is None
        with pytest.raises(FileExistsError):
            module.write_package(folder, payload)
        assert archive.read_bytes() == archives[-1]
    assert archives[0] == archives[1]


def test_failed_fresh_process_audit_retains_failed_evidence(tmp_path, monkeypatch):
    ids, paths, hashes = fixture(tmp_path)
    payload = compose(ids, paths, hashes)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "template_ids", lambda path: ids)
    monkeypatch.setattr(module, "compose_payload", lambda *args: payload)
    def fail(*args, **kwargs):
        assert kwargs["check"] and "audit" in args[0]
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(module.subprocess, "run", fail)
    folder = tmp_path / "local/release"
    with pytest.raises(subprocess.CalledProcessError):
        module.build_release(*paths, tmp_path / "unused-template", folder)
    assert (folder / "FAILED.json").exists()
    assert (folder / "result.csv").read_bytes() == payload
    assert not (folder / "verification.json").exists()
    assert not (folder / "manifest.json").exists()


def test_oversized_inputs_are_rejected_before_parsing(tmp_path):
    path = tmp_path / "oversize"
    path.write_bytes(b"0" * (module.MAX_BYTES + 1))
    with pytest.raises(ValueError, match="bounded"):
        module.zip_payload(path, module.sha(path.read_bytes()))
