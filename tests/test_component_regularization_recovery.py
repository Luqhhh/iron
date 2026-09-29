"""Recovery must preserve cache identity and every original failed artifact."""
import importlib.util
import json
from pathlib import Path

import pytest

from bf_tap_r2.v7_periodic import file_hash
from bf_tap_r2.v49_run import unit_id


@pytest.fixture
def recovery():
    path = Path(__file__).resolve().parents[1] / "scripts/resume_component_regularization.py"
    spec = importlib.util.spec_from_file_location("component_recovery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache(directory, manifest):
    directory.mkdir()
    artifact = directory / "saved.json"
    artifact.write_text('{"state": "frozen"}\n')
    (directory / "complete.json").write_text(json.dumps({
        "identity": unit_id(manifest, directory.name), "hashes": {artifact.name: file_hash(artifact)}}))


def test_copy_is_verbatim_and_rejects_tamper_and_overwrite(tmp_path, recovery):
    manifest = {"identity": "frozen"}
    old = tmp_path / "old" / "unit"
    old.parent.mkdir()
    new = tmp_path / "new" / "unit"
    new.parent.mkdir()
    cache(old, manifest)
    before = {p.name: p.read_bytes() for p in old.iterdir()}
    recovery.copy_verified(old, new, manifest)
    assert {p.name: p.read_bytes() for p in new.iterdir()} == before
    assert {p.name: p.read_bytes() for p in old.iterdir()} == before
    with pytest.raises(FileExistsError):
        recovery.copy_verified(old, new, manifest)
    (old / "saved.json").write_text("tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        recovery.copy_verified(old, tmp_path / "bad", manifest)
    assert not (tmp_path / "bad").exists()


def test_partial_unit_is_never_imported_or_removed(tmp_path, recovery):
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "selection.pt").write_bytes(b"interrupted selector")
    with pytest.raises(ValueError, match="Incomplete/failed"):
        recovery.copy_verified(partial, tmp_path / "new", {"identity": "frozen"})
    assert (partial / "selection.pt").read_bytes() == b"interrupted selector"
    assert not (tmp_path / "new").exists()


def test_recovery_covers_only_frozen_units_and_keeps_diagnostics_separate(recovery):
    import yaml
    spec = yaml.safe_load(Path("configs/strong_component_regularization/SPEC.yaml").read_text())
    regular, diagnostics = recovery.tasks_for(spec)
    assert len(regular) == 60 and len(diagnostics) == 6
    keys = {recovery.key_for(t) for t in regular + diagnostics}
    assert len(keys) == 66
    assert all(t[-1] == 42 for t in regular)
    assert all(t[:2] == (42, 0) and t[-1] == 1042 for t in diagnostics)


def test_corrective_auditor_changes_only_selector_target_array_layout(recovery):
    source = Path("src/bf_tap_r2/component_regularization_audit.py").read_text()
    corrected = recovery.corrected_auditor_source(source)
    old = 'fitting,fitting[outputs(target)].to_numpy(),arm,settings,spec["mechanisms"],validation)'
    new = 'fitting,training[outputs(target)].to_numpy()[inner!=0],arm,settings,spec["mechanisms"],validation)'
    assert corrected.replace(new, old) == source
    compile(corrected, "corrective_audit", "exec")
    with pytest.raises(ValueError, match="correction site changed"):
        recovery.corrected_auditor_source(corrected)
