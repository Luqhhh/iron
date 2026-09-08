import json
import hashlib
from zipfile import ZipFile

import pytest

from bf_tap.artifacts import file_sha256
from bf_tap.exceptions import ContractError
from bf_tap.optimization.v3_active import active_bundle, export_archived_submission


def test_active_pointer_checks_exact_bundle_before_prediction(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle.json").write_text(json.dumps({"candidate": "CB-FC-CVcal"}))
    cfg = {"schema_version": 1, "release_protocol": "development-fc-catboost-v1", "active_candidate": "CB-FC-CVcal",
           "promotion": "USER_OVERRIDE_G1_FAIL", "platform_score_status": "NOT_YET_TESTED", "bundle": str(bundle),
           "bundle_sha256": file_sha256(bundle / "bundle.json")}
    path = tmp_path / "active.yaml"
    path.write_text(json.dumps(cfg))
    assert active_bundle(path) == bundle
    (bundle / "bundle.json").write_text(json.dumps({"candidate": "E16"}))
    with pytest.raises(ContractError, match="identity"):
        active_bundle(path)
    cfg["bundle_sha256"] = file_sha256(bundle / "bundle.json")
    path.write_text(json.dumps(cfg))
    with pytest.raises(ContractError, match="candidate"):
        active_bundle(path)


def test_archived_export_preserves_bytes_and_refuses_wrong_stage_or_ids(tmp_path):
    payload = b"sample_id,pred_tap_iron,pred_tap_time_len\na,450,120\n"
    source = tmp_path / "submission.zip"
    with ZipFile(source, "x") as archive:
        archive.writestr("result.csv", payload)
    config = tmp_path / "active.yaml"
    config.write_text(json.dumps({"schema_version": 1, "release_protocol": "archived-e16-submission-v1",
        "active_candidate": "E16_TIMECAL_E12", "test_a_zip": str(source), "test_a_zip_sha256": file_sha256(source),
        "test_a_result_sha256": hashlib.sha256(payload).hexdigest()}))
    samples = tmp_path / "samples.csv"
    samples.write_text("sample_id,tap_iron\na,DO_NOT_READ\n")
    data = tmp_path / "data.yaml"
    data.write_text(json.dumps({"schema_version": 1, "paths": {"test_a_samples": str(samples)}}))
    out = export_archived_submission(config, data, "test_a", tmp_path / "export")
    assert out.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        export_archived_submission(config, data, "test_a", tmp_path / "export")
    with pytest.raises(ContractError, match="test_a only"):
        export_archived_submission(config, data, "test_b", tmp_path / "wrong-stage")
    samples.write_text("sample_id,tap_iron\nb,DO_NOT_READ\n")
    with pytest.raises(ContractError, match="IDs"):
        export_archived_submission(config, data, "test_a", tmp_path / "wrong-ids")
    source.write_bytes(b"damaged archive")
    with pytest.raises(ContractError, match="identity"):
        export_archived_submission(config, data, "test_a", tmp_path / "damaged")
