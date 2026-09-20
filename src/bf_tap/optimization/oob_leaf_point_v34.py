"""Pure contracts for the two v0.34 OOB leaf-point bagging candidates."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..exceptions import ContractError
from .fixed_blend_v28 import equal_blend_six, submission_bytes, validate_endpoint
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


CANDIDATE_A = "V34I_OOB_LEAF_MEDIAN_BAGGING_BLEND"
CANDIDATE_B = "V34T_OOB_LEAF_MEDIAN_BAGGING_TIME"
PARENT = "V30A_OOB_BOTH_TARGETS"
V21_REPLAY = "V21_REPLAY"
V1 = "V1"
LEAF_POINT_PROTOCOL = "FROZEN_OOB_LEAF_LOWER_MEDIAN_BAGGING_v034"
ATTACHMENT_PROTOCOL = "QRF_FROZEN_FOREST_OOB_LEAF_RESPONSE_v029"


def _aligned(parent: pd.DataFrame, source: pd.DataFrame, label: str):
    left = validate_endpoint(parent, label=PARENT)
    right = validate_endpoint(source, label=label)
    if left.sample_id.duplicated().any() or right.sample_id.duplicated().any():
        raise ContractError("sample IDs must be unique before v0.34 alignment")
    if set(left.sample_id) != set(right.sample_id):
        raise ContractError(f"{label}/parent sample ID sets differ")
    return left, right.set_index("sample_id").loc[left.sample_id].reset_index()


def compose_iron_candidate(
    parent_v30a: pd.DataFrame,
    complete_v26a: pd.DataFrame,
    leaf_point_qrf_iron: Sequence[float],
) -> pd.DataFrame:
    """Rebuild the original CatBoost/QRF 50/50 iron endpoint."""
    parent, catboost = _aligned(parent_v30a, complete_v26a, "V26A complete endpoint")
    values = np.asarray(leaf_point_qrf_iron, dtype=np.float64)
    if values.shape != (len(parent),) or not np.isfinite(values).all() or (values < 0).any():
        raise ContractError("v0.34 iron leaf-point endpoint is invalid")
    q6 = [f"{value:.6f}" for value in roundtrip_six(values)]
    result = parent.copy()
    result["pred_tap_iron"] = [
        equal_blend_six(complete, qrf)
        for complete, qrf in zip(catboost.pred_tap_iron, q6, strict=True)
    ]
    if result.pred_tap_time_len.tolist() != parent.pred_tap_time_len.tolist():
        raise ContractError("v0.34 iron candidate changed parent time strings")
    return validate_endpoint(result, label=CANDIDATE_A)


def compose_time_candidate(
    parent_v30a: pd.DataFrame,
    postprocessed_time: Sequence[float],
) -> pd.DataFrame:
    """Replace only time after the frozen v0.15 gate/V21 shrink replay."""
    parent = validate_endpoint(parent_v30a, label=PARENT)
    values = np.asarray(postprocessed_time, dtype=np.float64)
    if values.shape != (len(parent),) or not np.isfinite(values).all() or (values < 0).any():
        raise ContractError("v0.34 postprocessed time endpoint is invalid")
    result = parent.copy()
    result["pred_tap_time_len"] = [f"{value:.6f}" for value in roundtrip_six(values)]
    if result.pred_tap_iron.tolist() != parent.pred_tap_iron.tolist():
        raise ContractError("v0.34 time candidate changed parent iron strings")
    return validate_endpoint(result, label=CANDIDATE_B)


__all__ = [
    "ATTACHMENT_PROTOCOL", "CANDIDATE_A", "CANDIDATE_B", "LEAF_POINT_PROTOCOL",
    "PARENT", "POOLED_AGGREGATION", "PRIMARY_AGGREGATION", "TARGETS", "TOLERANCE",
    "V1", "V21_REPLAY", "compose_iron_candidate", "compose_time_candidate",
    "exposure_pooled_summary", "macro_origin_summary", "submission_bytes",
    "validate_endpoint", "verify_isolated_deltas",
]
