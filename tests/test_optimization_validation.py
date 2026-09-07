import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.optimization.ensemble import (
    apply_global_residual_calibration,
    convex_blend,
    shrink_toward_b1,
)
from bf_tap.optimization.validation import aggregate_grid, evaluate_acceptance, select_partitions
from bf_tap.optimization.config import EvaluationUnit


TZ = "Asia/Shanghai"


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz=TZ)


def test_split_requires_label_availability_at_origin():
    unit = EvaluationUnit(
        "O_H1", "O", 1, ts("2024-07-01"), ts("2024-03-01"),
        ts("2024-07-01"), ts("2024-08-01"),
    )
    samples = pd.DataFrame(
        {
            "sample_id": ["ready", "late", "eval"],
            "reference_time": [ts("2024-06-01"), ts("2024-06-02"), ts("2024-07-02")],
            "label_available_at": [ts("2024-06-01 01:00"), ts("2024-07-01 00:01"), ts("2024-07-02 01:00")],
        }
    )
    train, evaluation, evidence = select_partitions(samples, unit)
    assert train["sample_id"].tolist() == ["ready"]
    assert evaluation["sample_id"].tolist() == ["eval"]
    assert evidence["excluded_label_unavailable_rows"] == 1


def test_shrinkage_uses_separate_target_weights_and_checks_ids():
    learned = pd.DataFrame(
        {"sample_id": ["a"], "pred_tap_iron": [10.0], "pred_tap_time_len": [20.0]}
    )
    b1 = pd.DataFrame(
        {"sample_id": ["a"], "pred_tap_iron": [2.0], "pred_tap_time_len": [4.0]}
    )
    result = shrink_toward_b1(
        learned, b1, {"tap_iron": 0.25, "tap_time_len": 0.75}
    )
    assert result.loc[0, "pred_tap_iron"] == 4.0
    assert result.loc[0, "pred_tap_time_len"] == 16.0
    with pytest.raises(ContractError, match="IDs differ"):
        shrink_toward_b1(learned, b1.assign(sample_id="b"), {"tap_iron": 1, "tap_time_len": 1})


def test_convex_blend_supports_separate_target_weights():
    e02 = pd.DataFrame(
        {"sample_id": ["a"], "pred_tap_iron": [10.0], "pred_tap_time_len": [20.0]}
    )
    e04 = pd.DataFrame(
        {"sample_id": ["a"], "pred_tap_iron": [2.0], "pred_tap_time_len": [4.0]}
    )
    result = convex_blend(
        {"E02": e02, "E04": e04},
        {
            "tap_iron": {"E02": 0.5, "E04": 0.5},
            "tap_time_len": {"E02": 1.0, "E04": 0.0},
        },
    )
    assert result.loc[0, "pred_tap_iron"] == 6.0
    assert result.loc[0, "pred_tap_time_len"] == 20.0


def test_global_residual_calibration_is_target_specific_and_nonnegative():
    prediction = pd.DataFrame(
        {"sample_id": ["a"], "pred_tap_iron": [1.0], "pred_tap_time_len": [2.0]}
    )
    result = apply_global_residual_calibration(
        prediction, {"tap_iron": 0.0, "tap_time_len": 3.0}
    )
    assert result.loc[0, "pred_tap_iron"] == 1.0
    assert result.loc[0, "pred_tap_time_len"] == 0.0


def test_grid_J_is_equal_weighted_by_horizon_not_by_cell_count():
    metrics = {}
    counts = {1: 5, 2: 4, 3: 3, 4: 2}
    losses = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    for horizon, count in counts.items():
        for origin in range(count):
            target = {"wmape": losses[horizon]}
            metrics[f"O{origin}_H{horizon}"] = {
                "horizon": horizon,
                "candidates": {
                    "E00": {
                        "overall": {
                            "loss": losses[horizon], "iron": target, "time": target
                        }
                    }
                },
            }
    summary = aggregate_grid(metrics)
    assert summary["E00"]["J"] == pytest.approx(0.25)
    assert summary["E00"]["horizons"]["H1"]["origins"] == 5


def test_acceptance_refuses_partial_suite():
    result = evaluate_acceptance({}, {"E00": {}}, {})
    assert result["status"] == "INCOMPLETE"
