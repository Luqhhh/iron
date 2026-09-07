import numpy as np
import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.optimization.process_change import add_process_change_features


def config():
    return {
        "feature_set_id": "signed_operation_level_deltas_v1",
        "value_columns": ["air"],
        "transforms": [
            "latest_minus_6h_mean",
            "latest_minus_24h_mean",
            "mean_6h_minus_24h_mean",
        ],
    }


def test_process_change_features_are_signed_differences_of_asof_aggregates():
    frame = pd.DataFrame(
        {
            "spout_no": pd.Series(["1", "2"], dtype="string"),
            "operation__air__latest": [12.0, np.nan],
            "operation__air__6h__mean": [10.0, 4.0],
            "operation__air__24h__mean": [8.0, 3.0],
        }
    )
    result = add_process_change_features(
        frame, config(), baseline_value_columns=["air"]
    )
    assert result.loc[0, "process_change__air__latest_minus_6h_mean"] == 2.0
    assert result.loc[0, "process_change__air__latest_minus_24h_mean"] == 4.0
    assert result.loc[0, "process_change__air__mean_6h_minus_24h_mean"] == 2.0
    assert pd.isna(result.loc[1, "process_change__air__latest_minus_6h_mean"])
    assert result.loc[1, "process_change__air__mean_6h_minus_24h_mean"] == 1.0


def test_process_change_requires_exact_baseline_value_columns():
    with pytest.raises(ContractError, match="differ"):
        add_process_change_features(
            pd.DataFrame(), config(), baseline_value_columns=["other"]
        )
