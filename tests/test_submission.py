from zipfile import ZipFile

import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.submission import pack_submission, validate_submission, write_submission


def valid():
    return pd.DataFrame({"sample_id": pd.Series(["001", "002"], dtype="string"), "pred_tap_iron": [1.0, 2.0], "pred_tap_time_len": [3.0, 4.0]})


def test_roundtrip_and_zip_root(tmp_path):
    result = tmp_path / "result.csv"
    write_submission(valid(), valid().sample_id, result)
    assert pd.read_csv(result, dtype="string").sample_id.tolist() == ["001", "002"]
    archive = pack_submission(result, stage="test_a", team_name="team", output_dir=tmp_path, expected_ids=valid().sample_id)
    with ZipFile(archive) as zf:
        assert zf.namelist() == ["result.csv"]


@pytest.mark.parametrize(
    "mutation", ["columns", "ids", "duplicate", "negative", "infinite", "empty"]
)
def test_invalid_submission_fails(mutation):
    frame = valid()
    if mutation == "columns": frame = frame[["sample_id", "pred_tap_time_len", "pred_tap_iron"]]
    if mutation == "ids": frame.loc[1, "sample_id"] = "999"
    if mutation == "duplicate": frame.loc[1, "sample_id"] = "001"
    if mutation == "negative": frame.loc[1, "pred_tap_iron"] = -1
    if mutation == "infinite": frame.loc[1, "pred_tap_iron"] = float("inf")
    if mutation == "empty": frame = frame.iloc[0:0]
    with pytest.raises(ContractError):
        validate_submission(frame, valid().sample_id)


def test_pack_rejects_wrong_stage_and_overwrite(tmp_path):
    result = tmp_path / "result.csv"
    write_submission(valid(), valid().sample_id, result)
    with pytest.raises(ContractError, match="unknown stage"):
        pack_submission(result, stage="unknown", team_name="team", output_dir=tmp_path, expected_ids=valid().sample_id)
    pack_submission(result, stage="test_b", team_name="team", output_dir=tmp_path, expected_ids=valid().sample_id)
    with pytest.raises(ContractError, match="overwrite"):
        pack_submission(result, stage="test_b", team_name="team", output_dir=tmp_path, expected_ids=valid().sample_id)


def test_pack_revalidates_csv_content(tmp_path):
    result = tmp_path / "result.csv"
    result.write_text("not_a_submission\n1\n", encoding="utf-8")
    with pytest.raises(ContractError, match="columns"):
        pack_submission(
            result,
            stage="test_c",
            team_name="team",
            output_dir=tmp_path,
            expected_ids=valid().sample_id,
        )
