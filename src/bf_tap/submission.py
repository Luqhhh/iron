from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from .exceptions import ContractError

SUBMISSION_COLUMNS = ["sample_id", "pred_tap_iron", "pred_tap_time_len"]
STAGE_SUFFIX = {
    "test_a": "prelim",
    "test_b": "round2",
    "test_c": "semifinal",
}


def validate_submission(frame: pd.DataFrame, expected_ids: pd.Series) -> None:
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise ContractError(f"submission columns must be exactly {SUBMISSION_COLUMNS}")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ContractError("submission sample_id must be non-null and unique")
    actual_ids = frame["sample_id"].astype(str).tolist()
    required_ids = expected_ids.astype(str).tolist()
    if actual_ids != required_ids:
        raise ContractError("submission IDs or row order do not match stage sample table")
    predictions = frame[SUBMISSION_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(predictions).all():
        raise ContractError("submission predictions must be finite")
    if (predictions < 0).any():
        raise ContractError("submission predictions must be nonnegative")


def write_submission(frame: pd.DataFrame, expected_ids: pd.Series, path: str | Path) -> None:
    validate_submission(frame, expected_ids)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8", float_format="%.6f")
    reloaded = pd.read_csv(destination, dtype={"sample_id": "string"})
    validate_submission(reloaded, expected_ids)


def pack_submission(
    result_path: str | Path,
    *,
    stage: str,
    team_name: str,
    output_dir: str | Path,
) -> Path:
    if stage not in STAGE_SUFFIX:
        raise ContractError(f"unknown stage: {stage}")
    if not team_name or any(c in team_name for c in "/\\"):
        raise ContractError("invalid team name")
    result = Path(result_path)
    if result.name != "result.csv" or not result.is_file():
        raise ContractError("pack input must be an existing result.csv")
    destination = Path(output_dir) / f"{team_name}_bf_tap_predict_{STAGE_SUFFIX[stage]}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ContractError(f"refusing to overwrite {destination}")
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        archive.write(result, arcname="result.csv")
    with ZipFile(destination) as archive:
        if archive.namelist() != ["result.csv"]:
            raise ContractError("submission ZIP root must contain only result.csv")
    return destination
