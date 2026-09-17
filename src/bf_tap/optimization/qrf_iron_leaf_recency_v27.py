"""Pure contracts for the two preregistered v0.27 QRF experiments."""
from __future__ import annotations

import csv
import io
import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..exceptions import ContractError
from .qrf_partition_v26 import (
    POOLED_AGGREGATION,
    PRIMARY_AGGREGATION,
    TARGETS,
    TOLERANCE,
    exposure_pooled_summary,
    macro_origin_summary,
    roundtrip_six,
    verify_isolated_deltas,
)


CANDIDATE_A = "V27I_ABS_QRF_DIRECT_IRON"
CANDIDATE_B = "V27T_ABS_QRF_LEAF_RECENCY60"
PARENT = "V26A_PARENT"
PREDICTION_COLUMNS = ("pred_tap_iron", "pred_tap_time_len")


def recency60_weights(reference_ns: Sequence[int], cutoff_ns: int) -> np.ndarray:
    """Return unnormalised binary64 60-day half-life weights."""
    reference = np.asarray(reference_ns, dtype=np.int64)
    cutoff = int(cutoff_ns)
    if reference.ndim != 1 or len(reference) == 0 or (reference >= cutoff).any():
        raise ContractError("training reference times must be a nonempty vector before cutoff")
    days = (cutoff - reference).astype(np.float64) / (86400.0 * 1e9)
    result = np.exp2(-days / 60.0).astype(np.float64)
    if not np.isfinite(result).all() or (result <= 0).any() or (result > 1.0).any():
        raise ContractError("invalid recency60 training weights")
    return result


def isolate_iron(parent: pd.DataFrame, iron: Sequence[float]) -> pd.DataFrame:
    expected = ["sample_id", *PREDICTION_COLUMNS]
    if list(parent) != expected or parent.sample_id.isna().any() or parent.sample_id.astype(str).duplicated().any():
        raise ContractError("valid ordered V26A parent required")
    result = parent.copy()
    result["pred_tap_iron"] = roundtrip_six(iron)
    if not np.array_equal(result.pred_tap_time_len.to_numpy(), parent.pred_tap_time_len.to_numpy()):
        raise ContractError("iron-only candidate changed parent time")
    return result


def isolate_time(parent: pd.DataFrame, time: Sequence[float]) -> pd.DataFrame:
    expected = ["sample_id", *PREDICTION_COLUMNS]
    if list(parent) != expected or parent.sample_id.isna().any() or parent.sample_id.astype(str).duplicated().any():
        raise ContractError("valid ordered V26A parent required")
    result = parent.copy()
    result["pred_tap_time_len"] = roundtrip_six(time)
    if not np.array_equal(result.pred_tap_iron.to_numpy(), parent.pred_tap_iron.to_numpy()):
        raise ContractError("time-only candidate changed parent iron")
    return result


def submission_bytes_preserving_other_target(
    parent_strings: pd.DataFrame,
    sample_ids: Sequence[str],
    values: Sequence[float],
    *,
    changed_target: str,
) -> bytes:
    """Serialize one changed target while copying the other parent's text exactly."""
    expected = ["sample_id", *PREDICTION_COLUMNS]
    if list(parent_strings) != expected or parent_strings.isna().any().any():
        raise ContractError("complete string-valued V26A result required")
    ids = [str(value) for value in sample_ids]
    if parent_strings.sample_id.astype(str).tolist() != ids or len(set(ids)) != len(ids):
        raise ContractError("candidate/parent submission IDs or order differ")
    if changed_target not in ("tap_iron", "tap_time_len"):
        raise ContractError("exactly one registered changed target required")
    fixed_column = "pred_tap_time_len" if changed_target == "tap_iron" else "pred_tap_iron"
    fixed = parent_strings[fixed_column].astype(str).tolist()
    six = re.compile(r"^(?:0|[1-9][0-9]*)\.[0-9]{6}$")
    if any(six.fullmatch(value) is None for value in fixed):
        raise ContractError("unchanged parent strings must be nonnegative six-decimal values")
    changed = roundtrip_six(values)
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(expected)
    for index, sample_id in enumerate(ids):
        iron = f"{changed[index]:.6f}" if changed_target == "tap_iron" else fixed[index]
        time = fixed[index] if changed_target == "tap_iron" else f"{changed[index]:.6f}"
        writer.writerow([sample_id, iron, time])
    return handle.getvalue().encode("utf-8")


__all__ = [
    "CANDIDATE_A", "CANDIDATE_B", "PARENT", "TARGETS", "TOLERANCE",
    "PRIMARY_AGGREGATION", "POOLED_AGGREGATION", "recency60_weights",
    "isolate_iron", "isolate_time", "submission_bytes_preserving_other_target",
    "macro_origin_summary", "exposure_pooled_summary", "verify_isolated_deltas",
    "roundtrip_six",
]
