import pandas as pd
import pytest

from bf_tap.data import normalize_event_source
from bf_tap.exceptions import ContractError


def test_unresolved_availability_mapping_fails_closed():
    frame = pd.DataFrame({"clock": ["2024-01-01"], "value": [1.0]})
    with pytest.raises(ContractError, match="unresolved"):
        normalize_event_source(
            frame,
            event_time_column="clock",
            available_at_column=None,
            value_columns=["value"],
        )


def test_explicit_mapping_normalizes_timezone():
    frame = pd.DataFrame(
        {"clock": ["2024-01-01 00:00:00"], "published": ["2024-01-01 01:00:00"], "value": [1.0]}
    )
    result = normalize_event_source(
        frame,
        event_time_column="clock",
        available_at_column="published",
        value_columns=["value"],
    )
    assert str(result.event_time.dt.tz) == "Asia/Shanghai"
    assert result.available_at.iloc[0] > result.event_time.iloc[0]


def test_one_official_timestamp_can_define_event_and_availability():
    frame = pd.DataFrame({"clock": ["2024-01-01 00:00:00"], "value": [1.0]})
    result = normalize_event_source(
        frame,
        event_time_column="clock",
        available_at_column="clock",
        value_columns=["value"],
    )
    assert result.event_time.iloc[0] == result.available_at.iloc[0]
    assert result.value.iloc[0] == 1.0
