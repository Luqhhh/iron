import pytest

from bf_tap.artifacts import (
    build_inference_source_contract,
    create_run_directory,
    feature_cache_key,
    file_identities,
    validate_inference_source_contract,
)
from bf_tap.exceptions import ContractError


def test_cache_key_changes_for_every_identity_boundary():
    base = dict(code_version="a", data_manifest={"x": 1}, feature_config={"w": 6}, scenario="D", fit_cutoff="c", history_identity="h", sample_identity="s")
    key = feature_cache_key(**base)
    for field, value in [("code_version", "b"), ("data_manifest", {"x": 2}), ("feature_config", {"w": 24}), ("scenario", "E"), ("fit_cutoff", "d"), ("history_identity", "i"), ("sample_identity", "t")]:
        changed = {**base, field: value}
        assert feature_cache_key(**changed) != key


def test_run_directory_never_overwrites(tmp_path):
    create_run_directory(tmp_path, "r1")
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "r1")


def test_inference_source_contract_is_path_independent_and_content_bound(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for root in (left, right):
        (root / "operation.csv").write_bytes(b"same operation")
        (root / "burden.csv").write_bytes(b"same burden")
    contract = build_inference_source_contract(
        file_identities(
            {
                "operation_hourly": left / "operation.csv",
                "burden_change": left / "burden.csv",
            }
        ),
        semantic_contract_sha256="a" * 64,
    )
    same_content = file_identities(
        {
            "operation_hourly": right / "operation.csv",
            "burden_change": right / "burden.csv",
        }
    )
    validate_inference_source_contract(
        contract, same_content, semantic_contract_sha256="a" * 64
    )

    (right / "operation.csv").write_bytes(b"different operation")
    with pytest.raises(ContractError, match="inference source identity mismatch"):
        validate_inference_source_contract(
            contract,
            file_identities(
                {
                    "operation_hourly": right / "operation.csv",
                    "burden_change": right / "burden.csv",
                }
            ),
            semantic_contract_sha256="a" * 64,
        )
