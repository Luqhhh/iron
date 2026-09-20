"""Pure contracts for the two v0.35 time-leaf aggregation candidates."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..exceptions import ContractError
from .fixed_blend_v28 import submission_bytes, validate_endpoint
from .qrf_partition_v26 import (
    POOLED_AGGREGATION, PRIMARY_AGGREGATION, TARGETS, TOLERANCE,
    exposure_pooled_summary, macro_origin_summary, roundtrip_six, verify_isolated_deltas,
)


CANDIDATE_A = "V35A_OOB_MIDPOINT_LEAF_MEAN_TIME"
CANDIDATE_B = "V35B_OOB_LOWER_MEDIAN_OF_LEAF_POINTS_TIME"
PARENT = "V34T_OOB_LEAF_MEDIAN_BAGGING_TIME"
V30A = "V30A_OOB_BOTH_TARGETS"
V21_REPLAY = "V21_REPLAY"
V1 = "V1"
PROTOCOL = "FROZEN_OOB_TIME_LEAF_LOCATION_AND_VOTE_v035"
ATTACHMENT_PROTOCOL = "QRF_FROZEN_FOREST_OOB_LEAF_RESPONSE_v029"
PARENT_TABLE_PROTOCOL = "FROZEN_OOB_LEAF_LOWER_MEDIAN_BAGGING_v034"


def compose_time_candidate(parent_v34t: pd.DataFrame, postprocessed_time: Sequence[float], *, label: str) -> pd.DataFrame:
    if label not in (CANDIDATE_A, CANDIDATE_B):
        raise ContractError("unknown v0.35 candidate label")
    parent = validate_endpoint(parent_v34t, label=PARENT)
    values = np.asarray(postprocessed_time, dtype=np.float64)
    if values.shape != (len(parent),) or not np.isfinite(values).all() or (values < 0).any():
        raise ContractError("v0.35 postprocessed time endpoint is invalid")
    result = parent.copy()
    result["pred_tap_time_len"] = [f"{value:.6f}" for value in roundtrip_six(values)]
    if result.pred_tap_iron.tolist() != parent.pred_tap_iron.tolist():
        raise ContractError("v0.35 candidate changed parent iron strings")
    return validate_endpoint(result, label=label)


__all__ = [
    "ATTACHMENT_PROTOCOL", "CANDIDATE_A", "CANDIDATE_B", "PARENT", "PARENT_TABLE_PROTOCOL",
    "POOLED_AGGREGATION", "PRIMARY_AGGREGATION", "PROTOCOL", "TARGETS", "TOLERANCE",
    "V1", "V21_REPLAY", "V30A", "compose_time_candidate", "exposure_pooled_summary",
    "macro_origin_summary", "submission_bytes", "validate_endpoint", "verify_isolated_deltas",
]
