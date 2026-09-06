import pandas as pd

from bf_tap.splits import TemporalSplit, select_split


def t(value):
    return pd.Timestamp(value, tz="Asia/Shanghai")


def test_fit_boundary_requires_label_availability():
    frame = pd.DataFrame(
        {
            "sample_id": ["ok", "late", "eval"],
            "reference_time": [t("2024-06-01"), t("2024-06-30"), t("2024-07-01")],
            "label_available_at": [t("2024-06-02"), t("2024-07-02"), t("2024-07-02")],
        }
    )
    split = TemporalSplit("D", "development", t("2024-07-01"), t("2024-03-01"), t("2024-07-01"), t("2024-08-01"))
    train, evaluation, evidence = select_split(frame, split)
    assert train.sample_id.tolist() == ["ok"]
    assert evaluation.sample_id.tolist() == ["eval"]
    assert evidence["excluded_label_unavailable_rows"] == 1
