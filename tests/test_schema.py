import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.io import parse_local_time
from bf_tap.schema import validate_samples


def samples():
    return pd.DataFrame(
        {
            "sample_id": pd.Series(["0001", "0002"], dtype="string"),
            "tap_no": [1, 2],
            "spout_no": [1, 2],
            "reference_time": parse_local_time(pd.Series(["2024-01-01", "2024-01-02"]), "t"),
            "tap_iron": [10.0, 11.0],
            "tap_time_len": [5.0, 6.0],
        }
    )


def test_sample_id_string_and_schema():
    frame = samples()
    assert validate_samples(frame, labeled=True).rows == 2
    assert frame.sample_id.tolist() == ["0001", "0002"]


def test_duplicate_or_negative_label_fails():
    frame = samples()
    frame.loc[1, "sample_id"] = "0001"
    with pytest.raises(ContractError, match="duplicate"):
        validate_samples(frame, labeled=True)
    frame = samples()
    frame.loc[1, "tap_iron"] = -1
    with pytest.raises(ContractError, match="nonnegative"):
        validate_samples(frame, labeled=True)
