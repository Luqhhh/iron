import csv
import io
import zipfile

import numpy as np
import pytest

from bf_tap_r2.submission import ZIP_NAME, csv_bytes, deny_training_reads, package, predict_m0, validate_result


def test_m0_precision_packaging_and_shape(tmp_path):
    ids = [f"R2S_TEST_{i}" for i in range(322)]
    model = {"candidate": "R2_M0_GLOBAL_MEDIAN", "medians": {"tap_iron": 523.2390964285737, "tap_time_len": 124.62969065337148}}
    payload = csv_bytes(ids, predict_m0(model, ids))
    rows = validate_result(payload, ids)
    assert float(rows[0]["pred_tap_iron"]) == model["medians"]["tap_iron"]
    package(tmp_path, payload, ids)
    with zipfile.ZipFile(tmp_path / ZIP_NAME) as archive:
        assert archive.namelist() == ["result.csv"]
        assert archive.read("result.csv") == payload
    with pytest.raises(FileExistsError):
        package(tmp_path, payload, ids)


@pytest.mark.parametrize("bad", [-1., float("nan"), float("inf")])
def test_invalid_predictions_rejected(bad):
    with pytest.raises(ValueError):
        csv_bytes(["R2S_TEST_1"], [[bad, 1.]])


def test_input_id_contract_and_no_label_reads():
    with pytest.raises(ValueError):
        csv_bytes(["BF4_1"], [[1., 1.]])
    with pytest.raises(ValueError):
        validate_result(csv_bytes(["R2S_TEST_1"], [[1., 1.]]), ["R2S_TEST_1"])
    with pytest.raises(RuntimeError):
        deny_training_reads("open", ("复赛_train/train_samples.csv", "r"))
    deny_training_reads("open", ("复赛_test/test_samples.csv", "r"))
