import json
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import pytest

from bf_tap.artifacts import stable_digest
from bf_tap.config import load_yaml


ROOT = Path(__file__).parents[2]
CONFIG_ROOT = ROOT / "configs" / "optimization" / "v0.2"


def _run_module():
    module = find_spec("bf_tap.models.optimization_v02.run")
    assert module is not None, "optimization run initializer must exist"
    return import_module("bf_tap.models.optimization_v02.run")


def test_initialize_run_writes_bound_config_code_and_data_manifest(tmp_path):
    run = _run_module()
    common = load_yaml(CONFIG_ROOT / "common.yaml")
    candidate = load_yaml(CONFIG_ROOT / "m1_blend.yaml")
    code_identity = {"commit": "abc", "source_snapshot_sha256": "1" * 64}
    data_manifest = {
        "train_samples": {"bytes": 123, "sha256": "2" * 64},
        "operation_hourly": {"bytes": 456, "sha256": "3" * 64},
    }

    destination = run.initialize_optimization_run(
        tmp_path,
        "M1_BLEND-20240907T010203Z",
        common_config=common,
        candidate_config=candidate,
        code_identity=code_identity,
        data_manifest=data_manifest,
    )

    assert destination == tmp_path / "M1_BLEND-20240907T010203Z"
    resolved = json.loads((destination / "resolved_config.json").read_text())
    state = json.loads((destination / "run_state.json").read_text())
    assert resolved == {"candidate": candidate, "common": common}
    assert state == {
        "candidate_id": "M1_BLEND",
        "code_identity": code_identity,
        "data_manifest": data_manifest,
        "data_manifest_sha256": stable_digest(data_manifest),
        "resolved_config_sha256": stable_digest(resolved),
        "status": "CREATED",
    }


def test_initialize_run_never_overwrites_existing_evidence(tmp_path):
    run = _run_module()
    common = load_yaml(CONFIG_ROOT / "common.yaml")
    candidate = load_yaml(CONFIG_ROOT / "m1_blend.yaml")
    arguments = {
        "common_config": common,
        "candidate_config": candidate,
        "code_identity": {"commit": "abc"},
        "data_manifest": {"train_samples": {"sha256": "1" * 64}},
    }
    destination = run.initialize_optimization_run(tmp_path, "run-1", **arguments)
    before = (destination / "run_state.json").read_bytes()

    with pytest.raises(FileExistsError):
        run.initialize_optimization_run(tmp_path, "run-1", **arguments)

    assert (destination / "run_state.json").read_bytes() == before


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "", "."])
def test_initialize_run_rejects_unsafe_run_id(tmp_path, run_id):
    run = _run_module()
    common = load_yaml(CONFIG_ROOT / "common.yaml")
    candidate = load_yaml(CONFIG_ROOT / "m1_blend.yaml")

    with pytest.raises(ValueError, match="run_id"):
        run.initialize_optimization_run(
            tmp_path,
            run_id,
            common_config=common,
            candidate_config=candidate,
            code_identity={"commit": "abc"},
            data_manifest={"train_samples": {"sha256": "1" * 64}},
        )
