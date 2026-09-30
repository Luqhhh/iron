import pytest
from bf_tap_r2.t2g_run import validate_completion,validate_formal_manifest
from bf_tap_r2.t2g_protocol import expected_units

def test_missing_cell_incomplete():
    keys=expected_units("development",["tap_iron","tap_time_len"])
    records=[{"key":k.__dict__,"optimizer_starts":2} for k in keys]
    with pytest.raises(ValueError): validate_completion("development",["tap_iron","tap_time_len"],records[:-1])
    validate_completion("development",["tap_iron","tap_time_len"],records)
    with pytest.raises(ValueError): validate_completion("development",["tap_iron","tap_time_len"],records+[records[0]])

def test_formal_force_epochs_rejected():
    with pytest.raises(ValueError): validate_formal_manifest({"force_epochs":2})
