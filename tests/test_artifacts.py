import pytest

from bf_tap.artifacts import create_run_directory, feature_cache_key


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
