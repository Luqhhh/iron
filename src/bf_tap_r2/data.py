"""Strict snapshot reads, preserving sample-table order and explicit ID sets."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import yaml

FEATURES = tuple("air_volume cold_air_press hot_air_press oxygen hot_air_temp coal_rate humidity gas_rate furnace_top_press upper_press_diff lower_press_diff total_press_diff air_press_ratio furnace_top_temp_avg air_speed furnace_throat_temp pig all_quality consumption fuel_rate coke_rate".split())
TARGETS = ("tap_iron", "tap_time_len")
SUBMISSION_COLUMNS = ("sample_id", "pred_tap_iron", "pred_tap_time_len")
FILES = {
    "train_samples": "复赛_train/train_samples.csv",
    "train_features": "复赛_train/train_features.csv",
    "test_samples": "复赛_test/test_samples.csv",
    "test_features": "复赛_test/test_features.csv",
    "template": "复赛_test/result_template.csv",
}
SUPPORT_FILES = tuple(f"{folder}/{name}" for folder in ("复赛_train", "复赛_test")
                      for name in ("readme.txt", "data_dictionary.xlsx"))


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if any(cfg.get(key) != value for key, value in FILES.items()):
        raise ValueError("Only the round-two snapshot file allowlist is accepted")
    if (tuple(cfg["numeric_features"]) != FEATURES or tuple(cfg["targets"]) != TARGETS
            or cfg["categorical_features"] != ["spout_no"] or cfg["id_column"] != "sample_id"
            or set(cfg["support_files"]) != set(SUPPORT_FILES)
            or cfg["expected_rows"] != {"train": 2754, "test": 322}):
        raise ValueError("Round-two schema contract mismatch")
    return cfg


def resolve_file(root: Path, relative: str) -> Path:
    if relative not in {*FILES.values(), *SUPPORT_FILES}:
        raise ValueError(f"Not a round-two input: {relative}")
    path = root / relative
    if path.resolve() != root.resolve() / relative:
        raise ValueError("Symlinked data inputs are not allowed")
    return path


def read_table(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    if len(header) != len(set(header)) or set(header) != set(columns):
        raise ValueError(f"Unexpected or duplicate columns: {path}")
    frame = pd.read_csv(path, dtype={"sample_id": "string"})
    for name in columns:
        if name != "sample_id" and not pd.api.types.is_numeric_dtype(frame[name]):
            raise ValueError(f"Non-numeric field {name}: {path}")
    return frame.loc[:, list(columns)]


def join_snapshot(samples: pd.DataFrame, features: pd.DataFrame, stage: str) -> pd.DataFrame:
    prefix = {"train": "R2S_TRAIN_", "test": "R2S_TEST_"}[stage]
    for frame in (samples, features):
        ids = frame["sample_id"]
        if ids.isna().any() or ids.duplicated().any() or not ids.str.startswith(prefix).all():
            raise ValueError(f"Invalid, duplicate or foreign {stage} sample IDs")
    left, right = set(samples.sample_id), set(features.sample_id)
    if left != right:
        raise ValueError(f"ID set mismatch: missing features={len(left-right)}, extra features={len(right-left)}")
    return samples.merge(features, on="sample_id", validate="one_to_one", how="left", sort=False)


def load_snapshot(root: Path, cfg: dict, stage: str) -> tuple[pd.DataFrame, dict]:
    columns = ("sample_id", "spout_no") + (TARGETS if stage == "train" else ())
    samples = read_table(resolve_file(root, cfg[f"{stage}_samples"]), columns)
    features = read_table(resolve_file(root, cfg[f"{stage}_features"]), ("sample_id",) + FEATURES)
    merged = join_snapshot(samples, features, stage)
    if len(merged) != cfg["expected_rows"][stage]:
        raise ValueError(f"Unexpected {stage} row count: {len(merged)}")
    return merged, {"sample_rows": len(samples), "feature_rows": len(features),
                    "sample_unique_ids": samples.sample_id.nunique(),
                    "feature_unique_ids": features.sample_id.nunique(),
                    "missing_feature_ids": 0, "extra_feature_ids": 0,
                    "merged_rows": len(merged), "dtypes": merged.dtypes.astype(str).to_dict()}
