import pandas as pd
import pytest

from bf_tap.availability import assert_no_future, eligible_asof, freeze_history_origin
from bf_tap.exceptions import ContractError


def t(value):
    return pd.Timestamp(value, tz="Asia/Shanghai")


def test_asof_includes_boundary_and_excludes_future():
    source = pd.DataFrame({"available_at": [t("2024-01-01 09:00"), t("2024-01-01 09:01")], "x": [1, 2]})
    assert eligible_asof(source, t("2024-01-01 09:00")).x.tolist() == [1]


def test_frozen_origin_excludes_post_cutoff_even_for_later_samples():
    history = pd.DataFrame(
        {
            "reference_time": [t("2024-01-01"), t("2024-01-02")],
            "available_at": [t("2024-01-01 01:00"), t("2024-01-03")],
            "tap_iron": [10, 20],
            "tap_time_len": [5, 6],
        }
    )
    frozen = freeze_history_origin(history, t("2024-01-02 12:00"))
    assert frozen.tap_iron.tolist() == [10]


def test_future_audit_fails():
    audit = pd.DataFrame({"reference_time": [t("2024-01-01")], "max_available_at": [t("2024-01-02")]})
    with pytest.raises(ContractError, match="future"):
        assert_no_future(audit)
