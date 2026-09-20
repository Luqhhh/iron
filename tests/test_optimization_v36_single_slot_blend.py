from __future__ import annotations

import hashlib
import io
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from bf_tap.optimization import single_slot_blend_v36 as v36
from bf_tap.optimization import single_slot_blend_v36_run as v36_run


def rows(time=("3.000000", "4.000000")):
    return (
        v36.PredictionRow("a", "1.000000", time[0]),
        v36.PredictionRow("b", "2.000000", time[1]),
    )


def write_source(root: Path, value, *, include_csv=True):
    root.mkdir()
    payload = v36.payload_bytes(value)
    result = root / "result.csv"
    if include_csv:
        result.write_bytes(payload)
    archive = root / "source.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as handle:
        handle.writestr("result.csv", payload)
    return result, archive, hashlib.sha256(payload).hexdigest(), v36.sha256_file(archive)


def test_registered_identity_is_frozen():
    assert v36.CANDIDATE == "V36T_V34_V30_EQUAL_BLEND"
    assert v36.PROTOCOL == "COMPLETE_ENDPOINT_TIME_EQUAL_BLEND_v036"


def test_registration_freezes_single_slot_and_zero_fit_budget():
    registered = v36_run.registration()
    assert registered["candidate"] == v36.CANDIDATE
    assert registered["single_slot_user_constraint"]["remaining_platform_submissions"] == 1
    assert registered["budget"]["new_model_fits"] == 0
    assert registered["budget"]["new_candidate_packages"] == 1


def test_lifecycle_public_sources_exist():
    assert all(path.is_file() for path in v36_run._source_files())


def test_canonical_six_accepts_zero_and_positive_values():
    assert v36.canonical_six("0.000000") == "0.000000"
    assert v36.canonical_six("123.456789") == "123.456789"


@pytest.mark.parametrize("value", ["1", "1.2", "01.000000", "+1.000000"])
def test_canonical_six_rejects_noncanonical_text(value):
    with pytest.raises(v36.BlendContractError, match="canonical"):
        v36.canonical_six(value)


@pytest.mark.parametrize("value", ["-1.000000", "nan", "inf", "1.0000000"])
def test_canonical_six_rejects_negative_nonfinite_and_extra_precision(value):
    with pytest.raises(v36.BlendContractError, match="canonical"):
        v36.canonical_six(value)


def test_micro_integer_is_exact():
    assert v36.micro_integer("12.345678") == 12_345_678


def test_mean6_exact_integer_average():
    assert v36.mean6("80.000000", "120.000000") == "100.000000"


def test_mean6_ties_to_even_down():
    assert v36.mean6("1.000000", "1.000001") == "1.000000"


def test_mean6_ties_to_even_up():
    assert v36.mean6("1.000001", "1.000002") == "1.000002"


def test_payload_roundtrip_is_utf8_lf_and_six_decimal():
    payload = v36.payload_bytes(rows())
    assert payload == b"sample_id,pred_tap_iron,pred_tap_time_len\na,1.000000,3.000000\nb,2.000000,4.000000\n"
    assert v36.parse_payload(payload, role="fixture", expected_rows=2) == rows()


def test_parse_rejects_wrong_columns():
    payload = b"sample_id,pred_tap_time_len,pred_tap_iron\na,3.000000,1.000000\n"
    with pytest.raises(v36.BlendContractError, match="exactly"):
        v36.parse_payload(payload, role="fixture")


def test_parse_rejects_duplicate_ids():
    payload = b"sample_id,pred_tap_iron,pred_tap_time_len\na,1.000000,3.000000\na,2.000000,4.000000\n"
    with pytest.raises(v36.BlendContractError, match="unique"):
        v36.parse_payload(payload, role="fixture")


def test_parse_rejects_bom():
    payload = "\ufeffsample_id,pred_tap_iron,pred_tap_time_len\n".encode()
    with pytest.raises(v36.BlendContractError, match="BOM"):
        v36.parse_payload(payload, role="fixture")


def test_parse_rejects_non_utf8():
    with pytest.raises(v36.BlendContractError, match="UTF-8"):
        v36.parse_payload(b"\xff", role="fixture")


def test_parse_rejects_wrong_row_count():
    with pytest.raises(v36.BlendContractError, match="row count"):
        v36.parse_payload(v36.payload_bytes(rows()), role="fixture", expected_rows=3)


def test_compose_aligns_donor_by_id_and_preserves_parent_order():
    donor = tuple(reversed(rows(("5.000000", "8.000000"))))
    result = v36.compose_rows(rows(), donor)
    assert [row.sample_id for row in result] == ["a", "b"]
    assert [row.pred_tap_time_len for row in result] == ["4.000000", "6.000000"]
    assert [row.pred_tap_iron for row in result] == ["1.000000", "2.000000"]


def test_compose_rejects_iron_difference():
    donor = (v36.PredictionRow("a", "9.000000", "5.000000"), rows()[1])
    with pytest.raises(v36.BlendContractError, match="iron"):
        v36.compose_rows(rows(), donor)


def test_compose_rejects_id_set_difference():
    donor = (v36.PredictionRow("x", "1.000000", "5.000000"), rows()[1])
    with pytest.raises(v36.BlendContractError, match="sample_id"):
        v36.compose_rows(rows(), donor)


def test_compose_rejects_complete_noop():
    with pytest.raises(v36.BlendContractError, match="identical"):
        v36.compose_rows(rows(), rows())


def test_load_source_uses_certified_zip_when_csv_is_absent(tmp_path):
    result, archive, result_hash, zip_hash = write_source(tmp_path / "source", rows(), include_csv=False)
    loaded = v36.load_source(
        role="fixture", result_path=result, zip_path=archive,
        expected_result_sha256=result_hash, expected_zip_sha256=zip_hash, expected_rows=2,
    )
    assert loaded.rows == rows()
    assert loaded.result_path is None


def test_load_source_rejects_extra_zip_member(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    archive = root / "source.zip"
    with ZipFile(archive, "x") as handle:
        handle.writestr("result.csv", v36.payload_bytes(rows()))
        handle.writestr("extra.txt", b"no")
    with pytest.raises(v36.BlendContractError, match="only result.csv"):
        v36.read_zip_payload(archive, expected_zip_sha256=v36.sha256_file(archive), role="fixture")


def test_load_source_rejects_hash_mismatch(tmp_path):
    _, archive, _, _ = write_source(tmp_path / "source", rows())
    with pytest.raises(v36.BlendContractError, match="SHA-256"):
        v36.read_zip_payload(archive, expected_zip_sha256="0" * 64, role="fixture")


def test_build_package_writes_only_required_private_outputs(tmp_path, monkeypatch):
    parent = rows()
    donor = rows(("5.000001", "8.000000"))
    p_result, p_zip, p_result_hash, p_zip_hash = write_source(tmp_path / "parent", parent)
    d_result, d_zip, d_result_hash, d_zip_hash = write_source(tmp_path / "donor", donor)
    monkeypatch.setattr(v36, "V34_RESULT_SHA256", p_result_hash)
    monkeypatch.setattr(v36, "V34_ZIP_SHA256", p_zip_hash)
    monkeypatch.setattr(v36, "V30_RESULT_SHA256", d_result_hash)
    monkeypatch.setattr(v36, "V30_ZIP_SHA256", d_zip_hash)
    output = tmp_path / "output"
    receipt = v36.build_package(
        v34_result=p_result, v34_zip=p_zip, v30_result=d_result, v30_zip=d_zip,
        output_dir=output, expected_rows=2,
    )
    assert sorted(path.name for path in output.iterdir()) == ["Luqhhh_bf_tap_predict_prelim.zip", "receipt.json", "result.csv"]
    assert receipt["status"] == "PAYLOAD_VALIDATED_NOT_UPLOADED"
    assert receipt["source_model_cold_audit_completed_by_this_tool"] is False
    with ZipFile(output / "Luqhhh_bf_tap_predict_prelim.zip") as archive:
        assert archive.namelist() == ["result.csv"]
        assert archive.read("result.csv") == (output / "result.csv").read_bytes()


def test_build_package_refuses_overwrite(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(v36.BlendContractError, match="overwrite"):
        v36.build_package(
            v34_result=Path("missing"), v34_zip=Path("missing"),
            v30_result=Path("missing"), v30_zip=Path("missing"), output_dir=output,
        )
