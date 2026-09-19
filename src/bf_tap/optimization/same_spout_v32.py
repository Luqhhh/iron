"""Pure contracts for the two v0.32 same-spout OOB candidates."""
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


CANDIDATE_A = "V32I_SAME_SPOUT_OOB_BLEND"
CANDIDATE_B = "V32T_SAME_SPOUT_OOB_TIME"
PARENT = "V30A_OOB_BOTH_TARGETS"
V29I = "V29I_OOB_LEAF_QRF_BLEND"
V29T = "V29T_OOB_LEAF_QRF_TIME"
V28I = "V28I_PARENT"
V26A_PARENT = "V26A_PARENT"
V21_REPLAY = "V21_REPLAY"
V1 = "V1"
SAME_SPOUT_PROTOCOL = "QRF_FROZEN_FOREST_SAME_SPOUT_OOB_RESPONSE_v032"
ATTACHMENT_PROTOCOL = "QRF_FROZEN_FOREST_OOB_LEAF_RESPONSE_v029"


def _aligned(parent: pd.DataFrame, source: pd.DataFrame, label: str):
    left = validate_endpoint(parent, label=PARENT)
    right = validate_endpoint(source, label=label)
    if set(left.sample_id) != set(right.sample_id):
        raise ContractError(f"{label}/parent sample ID sets differ")
    if left.sample_id.duplicated().any() or right.sample_id.duplicated().any():
        raise ContractError("sample IDs must be unique before v0.32 alignment")
    return left, right.set_index("sample_id").loc[left.sample_id].reset_index()


def compose_iron_candidate(
    parent_v30a: pd.DataFrame,
    complete_v26a: pd.DataFrame,
    same_spout_qrf_iron: Sequence[float],
) -> pd.DataFrame:
    """Rebuild the original 50/50 iron structure and copy parent time exactly."""
    parent, catboost = _aligned(parent_v30a, complete_v26a, "V26A complete endpoint")
    values = np.asarray(same_spout_qrf_iron, dtype=np.float64)
    if values.shape != (len(parent),):
        raise ContractError("same-spout iron endpoint row count differs")
    q6 = [f"{value:.6f}" for value in roundtrip_six(values)]
    result = parent.copy()
    result["pred_tap_iron"] = [
        equal_blend_six(complete, qrf) for complete, qrf in zip(catboost.pred_tap_iron, q6, strict=True)
    ]
    if result.pred_tap_time_len.tolist() != parent.pred_tap_time_len.tolist():
        raise ContractError("v0.32 iron candidate changed parent time strings")
    return validate_endpoint(result, label=CANDIDATE_A)


def compose_time_candidate(parent_v30a: pd.DataFrame, postprocessed_time: Sequence[float]) -> pd.DataFrame:
    """Replace only time after the frozen v0.15 gate/V21 shrink chain."""
    parent = validate_endpoint(parent_v30a, label=PARENT)
    values = np.asarray(postprocessed_time, dtype=np.float64)
    if values.shape != (len(parent),):
        raise ContractError("same-spout time endpoint row count differs")
    result = parent.copy()
    result["pred_tap_time_len"] = [f"{value:.6f}" for value in roundtrip_six(values)]
    if result.pred_tap_iron.tolist() != parent.pred_tap_iron.tolist():
        raise ContractError("v0.32 time candidate changed parent iron strings")
    return validate_endpoint(result, label=CANDIDATE_B)


__all__ = [
    "CANDIDATE_A", "CANDIDATE_B", "PARENT", "V29I", "V29T", "V28I", "V26A_PARENT",
    "V21_REPLAY", "V1", "SAME_SPOUT_PROTOCOL", "ATTACHMENT_PROTOCOL", "PRIMARY_AGGREGATION",
    "POOLED_AGGREGATION", "TARGETS", "TOLERANCE", "compose_iron_candidate",
    "compose_time_candidate", "macro_origin_summary", "exposure_pooled_summary",
    "verify_isolated_deltas", "submission_bytes", "validate_endpoint",
]
