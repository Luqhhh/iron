from pathlib import Path

import pandas as pd
import pytest

from bf_tap_r2.audit import near_pairs, run
from bf_tap_r2.data import FEATURES, join_snapshot, load_config, read_table, resolve_file


def frames():
    samples = pd.DataFrame({"sample_id": ["R2S_TEST_1", "R2S_TEST_2"], "spout_no": [1, 2]})
    features = pd.DataFrame({"sample_id": samples.sample_id, **{c: [1., 2.] for c in FEATURES}})
    return samples, features


def test_join_is_id_aligned_and_preserves_sample_order():
    samples, features = frames()
    expected = join_snapshot(samples, features, "test")
    pd.testing.assert_frame_equal(expected, join_snapshot(samples, features.iloc[::-1], "test"))
    shuffled = join_snapshot(samples.iloc[::-1], features, "test").sort_values("sample_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, shuffled)


@pytest.mark.parametrize("damage", ["duplicate", "missing", "extra", "foreign", "null"])
def test_invalid_ids_are_rejected(damage):
    samples, features = frames()
    if damage == "duplicate":
        features.loc[1, "sample_id"] = features.loc[0, "sample_id"]
    elif damage == "missing":
        features = features.iloc[:1]
    elif damage == "extra":
        features.loc[2] = ["R2S_TEST_3", *([3.] * len(FEATURES))]
    elif damage == "foreign":
        features.loc[0, "sample_id"] = "BF4_1"
    else:
        features.loc[0, "sample_id"] = None
    with pytest.raises(ValueError):
        join_snapshot(samples, features, "test")


def test_preliminary_paths_and_symlinks_rejected(tmp_path):
    with pytest.raises(ValueError):
        resolve_file(tmp_path, "初赛数据集/train_samples.csv")
    (tmp_path / "复赛_train").mkdir()
    (tmp_path / "复赛_train/train_samples.csv").symlink_to(tmp_path / "old.csv")
    with pytest.raises(ValueError):
        resolve_file(tmp_path, "复赛_train/train_samples.csv")


def test_schema_rejects_extra_and_duplicate_fields(tmp_path):
    path = tmp_path / "samples.csv"
    for content in ("sample_id,spout_no,tap_iron\nR2S_TEST_1,1,100\n",
                    "sample_id,spout_no,spout_no\nR2S_TEST_1,1,1\n",
                    "sample_id,spout_no\nR2S_TEST_1,text\n"):
        path.write_text(content)
        with pytest.raises(ValueError):
            read_table(path, ("sample_id", "spout_no"))


def test_near_duplicate_screen_covers_cross_split_and_spout():
    samples, features = frames()
    train = join_snapshot(samples, features, "test")
    test = train.iloc[:1].copy()
    test["sample_id"] = "R2S_TEST_3"
    test["spout_no"] = 9
    report = near_pairs(train, test, .01, 1)
    assert report["counts"] == {"train": 0, "test": 0, "cross": 1}
    assert report["pairs"][0]["same_spout"] is False


def test_run_preserves_failure_and_refuses_overwrite(tmp_path):
    output = tmp_path / "local/runs/round2-v0.1/p0"
    with pytest.raises(FileNotFoundError):
        run(tmp_path, tmp_path / "absent.yaml", output)
    before = (output / "FAILED.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(tmp_path, tmp_path / "absent.yaml", output)
    assert (output / "FAILED.json").read_bytes() == before
    with pytest.raises(ValueError):
        run(tmp_path, tmp_path / "absent.yaml", tmp_path / "public")


def test_frozen_config():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/round2_v0_1/data.yaml")
    assert len(cfg["numeric_features"]) == 21
    assert "sample_id" not in cfg["numeric_features"]
