import pandas as pd
import pytest

from bf_tap.optimization_v03.diagnostics import (
    diagnostic_metrics,
    paired_daily_block_bootstrap,
    scenario_summary,
)


TZ = "Asia/Shanghai"


def test_diagnostics_report_required_groups_and_signed_bias():
    ids = [f"s{i}" for i in range(8)]
    actual = pd.DataFrame(
        {
            "sample_id": ids,
            "reference_time": pd.to_datetime(
                [
                    "2024-01-01 01:00+08:00",
                    "2024-01-01 02:00+08:00",
                    "2024-01-01 07:00+08:00",
                    "2024-01-01 08:00+08:00",
                    "2024-02-01 13:00+08:00",
                    "2024-02-01 14:00+08:00",
                    "2024-02-01 19:00+08:00",
                    "2024-02-01 20:00+08:00",
                ],
                utc=True,
            ).tz_convert(TZ),
            "spout_no": [1, 2, 1, 2, 1, 2, 1, 2],
            "tap_iron": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "tap_time_len": [
                100.0,
                110.0,
                120.0,
                130.0,
                140.0,
                150.0,
                160.0,
                170.0,
            ],
        }
    )
    predicted = pd.DataFrame(
        {
            "sample_id": ids,
            "pred_tap_iron": actual["tap_iron"] + 1.0,
            "pred_tap_time_len": actual["tap_time_len"] - 2.0,
        }
    )

    result = diagnostic_metrics(actual, predicted)

    assert result["overall"]["iron"]["wmape"] == pytest.approx(8.0 / 360.0)
    assert result["overall"]["time"]["wmape"] == pytest.approx(16.0 / 1080.0)
    assert result["overall"]["loss"] == pytest.approx(
        0.5 * (8.0 / 360.0 + 16.0 / 1080.0)
    )
    assert result["overall"]["iron"]["signed_bias"] == 1.0
    assert result["overall"]["time"]["signed_bias"] == -2.0
    assert set(result["by_month"]) == {"2024-01", "2024-02"}
    assert set(result["by_spout"]) == {"1", "2"}
    assert set(result["by_hour_bucket"]) == {
        "00-05",
        "06-11",
        "12-17",
        "18-23",
    }
    assert set(result["by_actual_quartile"]["tap_iron"]) == {
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    }
    assert set(result["by_actual_quartile"]["tap_time_len"]) == {
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    }


def test_scenario_summary_makes_repeated_sample_ids_scenario_unique():
    rows = pd.DataFrame(
        {
            "candidate_id": ["A", "A"],
            "unit_id": ["U1", "U2"],
            "sample_id": ["same", "same"],
            "reference_time": pd.to_datetime(
                ["2024-01-01 01:00+08:00", "2024-02-01 07:00+08:00"],
                utc=True,
            ).tz_convert(TZ),
            "spout_no": [1, 2],
            "tap_iron": [10.0, 20.0],
            "tap_time_len": [100.0, 100.0],
            "pred_tap_iron": [11.0, 22.0],
            "pred_tap_time_len": [99.0, 98.0],
        }
    )

    result = scenario_summary(rows)

    assert result["A"]["scenario_rows"] == 2
    assert result["A"]["overall"]["iron"]["wmape"] == pytest.approx(3.0 / 30.0)
    assert set(result["A"]["by_month"]) == {"2024-01", "2024-02"}


def _bootstrap_rows():
    records = []
    for day, actual_iron, actual_time in (
        ("2024-01-01 01:00+08:00", 10.0, 100.0),
        ("2024-01-02 01:00+08:00", 20.0, 200.0),
    ):
        for candidate, iron_error, time_error in (
            ("CANDIDATE", 0.0, 0.0),
            ("CONTROL", 2.0, 4.0),
        ):
            records.append(
                {
                    "candidate_id": candidate,
                    "unit_id": "U1",
                    "sample_id": day[:10],
                    "reference_time": pd.Timestamp(day),
                    "tap_iron": actual_iron,
                    "tap_time_len": actual_time,
                    "pred_tap_iron": actual_iron + iron_error,
                    "pred_tap_time_len": actual_time + time_error,
                }
            )
    return pd.DataFrame(records)


def test_daily_block_bootstrap_is_paired_deterministic_and_signed():
    rows = _bootstrap_rows()

    first = paired_daily_block_bootstrap(
        rows, "CANDIDATE", "CONTROL", repetitions=200, seed=20260908
    )
    second = paired_daily_block_bootstrap(
        rows, "CANDIDATE", "CONTROL", repetitions=200, seed=20260908
    )
    reversed_result = paired_daily_block_bootstrap(
        rows, "CONTROL", "CANDIDATE", repetitions=200, seed=20260908
    )

    assert first == second
    assert first["point_delta"] < 0
    assert first["ci_low"] <= first["point_delta"] <= first["ci_high"]
    assert first["blocks"] == 2
    assert first["repetitions"] == 200
    assert first["seed"] == 20260908
    assert reversed_result["point_delta"] == pytest.approx(-first["point_delta"])

def test_scenario_summary_excludes_screening_horizon_zero():
    rows = pd.DataFrame(
        {
            "candidate_id": ["C1_RECENT30_TIME", "C1_RECENT30_TIME"],
            "unit_id": ["DEV_LONG", "O202406_H1"],
            "horizon": [0, 1],
            "sample_id": ["screening", "grid"],
            "reference_time": pd.to_datetime(
                ["2024-07-01T00:00:00+08:00", "2024-07-01T00:00:00+08:00"]
            ),
            "spout_no": [1, 1],
            "tap_iron": [100.0, 100.0],
            "tap_time_len": [10.0, 10.0],
            "pred_tap_iron": [0.0, 100.0],
            "pred_tap_time_len": [0.0, 10.0],
        }
    )

    result = scenario_summary(rows)

    assert result["C1_RECENT30_TIME"]["scenario_rows"] == 1
    assert result["C1_RECENT30_TIME"]["overall"]["iron"]["n"] == 1
