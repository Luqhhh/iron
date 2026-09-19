"""Pure composition contracts for the two v0.33 time-only candidates."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..exceptions import ContractError
from .fixed_blend_v28 import submission_bytes, validate_endpoint
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


CANDIDATE_A = "V33A_OOB_TIME_FEATURE_THIRD"
CANDIDATE_B = "V33B_OOB_TIME_HALF_BOOTSTRAP"
PARENT = "V30A_OOB_BOTH_TARGETS"
V21_REPLAY = "V21_REPLAY"
V1 = "V1"
FOREST_PROTOCOL = "QRF_TIME_FEATURE_AND_ROW_SAMPLING_FOREST_v033"
OOB_PROTOCOL = "QRF_SAMPLED_FOREST_OOB_LEAF_RESPONSE_v033"


def compose_time_candidate(parent_v30a: pd.DataFrame, postprocessed_time: Sequence[float], *, label: str) -> pd.DataFrame:
    if label not in (CANDIDATE_A, CANDIDATE_B):
        raise ContractError("registered v0.33 candidate label required")
    parent = validate_endpoint(parent_v30a, label=PARENT)
    values = np.asarray(postprocessed_time, dtype=np.float64)
    if values.shape != (len(parent),):
        raise ContractError("v0.33 time endpoint row count differs")
    result = parent.copy()
    result["pred_tap_time_len"] = [f"{value:.6f}" for value in roundtrip_six(values)]
    if result.pred_tap_iron.tolist() != parent.pred_tap_iron.tolist():
        raise ContractError("v0.33 time candidate changed V30A iron strings")
    return validate_endpoint(result, label=label)


__all__ = [
    "CANDIDATE_A", "CANDIDATE_B", "PARENT", "V21_REPLAY", "V1", "FOREST_PROTOCOL", "OOB_PROTOCOL",
    "PRIMARY_AGGREGATION", "POOLED_AGGREGATION", "TARGETS", "TOLERANCE", "compose_time_candidate",
    "macro_origin_summary", "exposure_pooled_summary", "verify_isolated_deltas", "submission_bytes", "validate_endpoint",
]
