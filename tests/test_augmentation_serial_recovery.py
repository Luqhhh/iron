"""Recovery must reject damaged, mismatched, partial and overwritten caches."""
import importlib.util
import json
from pathlib import Path

import pytest
from bf_tap_r2.v7_periodic import file_hash
from bf_tap_r2.v49_run import unit_id


@pytest.fixture
def recovery():
    path = Path(__file__).resolve().parents[1] / "scripts/resume_augmentation_queue.py"
    spec = importlib.util.spec_from_file_location("augmentation_recovery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache(directory, manifest):
    directory.mkdir()
    artifact = directory / "saved.json"
    artifact.write_text('{"state": "frozen"}\n')
    (directory / "complete.json").write_text(json.dumps({
        "identity": unit_id(manifest, directory.name), "hashes": {artifact.name: file_hash(artifact)}}))


def test_verbatim_copy_preserves_source_and_rejects_overwrite(tmp_path, recovery):
    manifest = {"identity": "frozen"}
    old = tmp_path / "old" / "unit"; old.parent.mkdir()
    new = tmp_path / "new" / "unit"; new.parent.mkdir()
    cache(old, manifest)
    before = {p.name: p.read_bytes() for p in old.iterdir()}
    recovery.copy_verified(old, new, manifest)
    assert {p.name: p.read_bytes() for p in new.iterdir()} == before
    assert {p.name: p.read_bytes() for p in old.iterdir()} == before
    with pytest.raises(FileExistsError):
        recovery.copy_verified(old, new, manifest)


def test_corrupted_cache_is_rejected_before_destination_creation(tmp_path, recovery):
    old = tmp_path / "unit"; manifest = {"identity": "frozen"}; cache(old, manifest)
    (old / "saved.json").write_text("tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        recovery.copy_verified(old, tmp_path / "new", manifest)
    assert not (tmp_path / "new").exists()


def test_wrong_manifest_is_rejected_before_destination_creation(tmp_path, recovery):
    old = tmp_path / "unit"; cache(old, {"identity": "original"})
    with pytest.raises(ValueError, match="identity mismatch"):
        recovery.copy_verified(old, tmp_path / "new", {"identity": "different"})
    assert not (tmp_path / "new").exists()


def test_partial_evidence_is_never_imported_or_removed(tmp_path, recovery):
    partial = tmp_path / "partial"; partial.mkdir()
    (partial / "events.jsonl").write_bytes(b'{"event":"start"}\n')
    with pytest.raises(ValueError, match="Incomplete/failed"):
        recovery.copy_verified(partial, tmp_path / "new", {"identity": "frozen"})
    assert (partial / "events.jsonl").read_bytes() == b'{"event":"start"}\n'
    assert not (tmp_path / "new").exists()
