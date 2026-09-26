"""Protect the locked CPU test path from missing neural dependencies."""
import importlib
from pathlib import Path
import tomllib

import pytest
import yaml


@pytest.mark.parametrize(
    "distribution,module,spec_path",
    [
        ("torch", "torch", "configs/round2_v7/SPEC.yaml"),
        ("tabm", "tabm", "configs/round2_v7/SPEC.yaml"),
        ("rtdl-num-embeddings", "rtdl_num_embeddings", "configs/round2_v7/SPEC.yaml"),
        ("rtdl-revisiting-models", "rtdl_revisiting_models", "configs/round2_v8/SPEC.yaml"),
    ],
)
def test_round2_extra_installs_frozen_neural_libraries(distribution, module, spec_path):
    project = tomllib.loads(Path("pyproject.toml").read_text())
    versions = yaml.safe_load(Path(spec_path).read_text())["runtime_versions"]
    # PEP 440's public version matches the CPU build installed via the explicit index.
    version = str(versions[distribution]).split("+")[0]
    assert f"{distribution}=={version}" in project["project"]["optional-dependencies"]["round2"]
    importlib.import_module(module)  # Fail visibly instead of silently skipping coverage.


def test_torch_source_is_explicit_cpu_index():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    assert project["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cpu"}
    indexes = {item["name"]: item for item in project["tool"]["uv"]["index"]}
    assert indexes["pytorch-cpu"]["url"] == "https://download.pytorch.org/whl/cpu"
    assert indexes["pytorch-cpu"]["explicit"] is True
