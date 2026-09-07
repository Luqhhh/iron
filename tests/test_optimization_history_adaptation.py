import numpy as np
import pandas as pd

from bf_tap.features.history import build_history_features
from bf_tap.optimization.history_adaptation import build_history_views


TZ = "Asia/Shanghai"


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz=TZ)


def test_history_views_are_grouped_weighted_and_auditable():
    samples = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d", "e"],
            "reference_time": [ts(f"2024-03-{day:02d}") for day in range(1, 6)],
            "tap_iron": [10.0, 11.0, 12.0, 13.0, 14.0],
            "tap_time_len": [5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )
    result = build_history_views(samples, (0, 7, 30, 60, 90))
    assert len(result.samples) == 25
    assert not result.samples["view_id"].duplicated().any()
    grouped = result.samples.groupby("original_sample_id")
    assert grouped.size().eq(5).all()
    assert np.allclose(grouped["sample_weight"].sum(), 1.0)
    assert grouped["inner_partition"].nunique().eq(1).all()
    assert set(result.audit) >= {
        "original_sample_id", "view_id", "history_origin", "view_contract_sha256"
    }


def test_synthetic_origin_masks_entire_unavailable_history_rows():
    history = pd.DataFrame(
        {
            "sample_id": ["old", "recent"],
            "tap_no": [1, 2],
            "spout_no": [1, 1],
            "reference_time": [ts("2024-01-01"), ts("2024-01-09")],
            "tap_end_time": [ts("2024-01-01 01:00"), ts("2024-01-09 01:00")],
            "available_at": [ts("2024-01-01 01:00"), ts("2024-01-09 01:00")],
            "tap_iron": [10.0, 999.0],
            "tap_time_len": [5.0, 999.0],
        }
    )
    samples = pd.DataFrame(
        {
            "sample_id": ["view"],
            "original_sample_id": ["current"],
            "view_id": ["current::history_age_7d"],
            "spout_no": [1],
            "reference_time": [ts("2024-01-10")],
            "history_origin": [ts("2024-01-03")],
        }
    )
    features, audit = build_history_features(
        samples, history, fit_cutoff=ts("2024-01-11"), last_k=(3,)
    )
    assert features.loc[0, "history__all__tap_iron__latest"] == 10.0
    assert features.loc[0, "history__all__tap_iron__last3_count"] == 1.0
    assert audit.loc[0, "max_available_at"] <= audit.loc[0, "history_origin"]
