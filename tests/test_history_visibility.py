import pandas as pd

from bf_tap.features.history import build_history_features


def t(value):
    return pd.Timestamp(value, tz="Asia/Shanghai")


def history():
    return pd.DataFrame(
        {
            "sample_id": ["old", "current", "future-origin"],
            "spout_no": [1, 1, 2],
            "reference_time": [t("2024-01-01 08:00"), t("2024-01-01 09:00"), t("2024-01-02")],
            "available_at": [t("2024-01-01 08:30"), t("2024-01-01 10:30"), t("2024-01-02 01:00")],
            "tap_iron": [10.0, 999.0, 777.0],
            "tap_time_len": [5.0, 999.0, 777.0],
        }
    )


def test_unfinished_current_and_cross_origin_labels_do_not_leak():
    samples = pd.DataFrame({"sample_id": ["current"], "spout_no": [1], "reference_time": [t("2024-01-01 09:40")]})
    x, audit = build_history_features(samples, history(), fit_cutoff=t("2024-01-01 10:00"), last_k=(3, 10))
    assert x.loc[0, "history__all__tap_iron__latest"] == 10
    assert x.loc[0, "history__all__tap_iron__last3_count"] == 1
    assert audit.loc[0, "max_available_at"] == t("2024-01-01 08:30")


def test_early_sample_cannot_see_record_available_before_cutoff_but_after_sample():
    samples = pd.DataFrame({"sample_id": ["early"], "spout_no": [1], "reference_time": [t("2024-01-01 08:15")]})
    x, _ = build_history_features(samples, history(), fit_cutoff=t("2024-01-01 10:00"), last_k=(3,))
    assert pd.isna(x.loc[0, "history__all__tap_iron__latest"])
