import pandas as pd

from bf_tap.io import read_development_history, read_development_labels


def test_development_reader_skips_protected_rows_and_preserves_id(tmp_path):
    source = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "sample_id": ["0001", "0002"],
            "tap_no": [1, 2],
            "spout_no": [1, 2],
            "reference_time": ["2024-10-31 23:00:00", "2024-11-01 00:00:00"],
            "tap_iron": [1.0, 999999.0],
            "tap_time_len": [2.0, 999999.0],
        }
    ).to_csv(source, index=False)
    safe = read_development_labels(source, "2024-11-01T00:00:00+08:00")
    assert safe.sample_id.tolist() == ["0001"]
    assert safe.tap_iron.tolist() == [1.0]


def test_history_reader_requires_reference_and_availability_before_origin(tmp_path):
    source = tmp_path / "history.csv"
    pd.DataFrame(
        {
            "sample_id": ["safe", "unfinished", "future"],
            "reference_time": ["2024-10-30", "2024-10-31", "2024-11-02"],
            "tap_end_time": ["2024-10-30 01:00", "2024-11-01 01:00", "2024-11-02 01:00"],
            "tap_iron": [1.0, 999.0, 888.0],
            "tap_time_len": [2.0, 999.0, 888.0],
        }
    ).to_csv(source, index=False)
    safe = read_development_history(
        source,
        "2024-11-01T00:00:00+08:00",
        available_at_column="tap_end_time",
    )
    assert safe.sample_id.tolist() == ["safe"]
    assert safe.tap_iron.tolist() == [1.0]
