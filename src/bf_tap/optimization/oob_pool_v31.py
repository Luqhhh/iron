"""Pure contracts for the two v0.31 occurrence-pooled OOB candidates."""
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


CANDIDATE_A = "V31I_OOB_OCCURRENCE_POOL_BLEND"
CANDIDATE_B = "V31T_OOB_OCCURRENCE_POOL_TIME"
PARENT = "V30A_OOB_BOTH_TARGETS"
V29I = "V29I_OOB_LEAF_QRF_BLEND"
V29T = "V29T_OOB_LEAF_QRF_TIME"
V28I = "V28I_PARENT"
V26A_PARENT = "V26A_PARENT"
V21_REPLAY = "V21_REPLAY"
V1 = "V1"
POOLED_PROTOCOL = "QRF_FROZEN_FOREST_OOB_OCCURRENCE_POOLING_v031"
ATTACHMENT_PROTOCOL = "QRF_FROZEN_FOREST_OOB_LEAF_RESPONSE_v029"


def _aligned(parent: pd.DataFrame, source: pd.DataFrame, label: str):
    left = validate_endpoint(parent, label=PARENT)
    right = validate_endpoint(source, label=label)
    if set(left.sample_id) != set(right.sample_id):
        raise ContractError(f"{label}/parent sample ID sets differ")
    if left.sample_id.duplicated().any() or right.sample_id.duplicated().any():
        raise ContractError("sample IDs must be unique before v0.31 alignment")
    return left, right.set_index("sample_id").loc[left.sample_id].reset_index()


def compose_iron_candidate(
    parent_v30a: pd.DataFrame,
    complete_v26a: pd.DataFrame,
    pooled_qrf_iron: Sequence[float],
) -> pd.DataFrame:
    """Average complete V26A iron with the new occurrence-pooled OOB iron.

    The parent's V30A time string column is copied verbatim.  The blend is not
    ``0.5 * V30A_iron + 0.5 * new_QRF``: V30A already contains the old 50/50
    V29I composition, so the new candidate is re-based on the original V26A
    complete endpoint as registered.
    """
    parent, catboost = _aligned(parent_v30a, complete_v26a, "V26A complete endpoint")
    values = np.asarray(pooled_qrf_iron, dtype=np.float64)
    if values.shape != (len(parent),):
        raise ContractError("new occurrence-pooled iron endpoint row count differs")
    q6 = [f"{value:.6f}" for value in roundtrip_six(values)]
    result = parent.copy()
    result["pred_tap_iron"] = [
        equal_blend_six(complete, pooled) for complete, pooled in zip(catboost.pred_tap_iron, q6, strict=True)
    ]
    if result.pred_tap_time_len.tolist() != parent.pred_tap_time_len.tolist():
        raise ContractError("iron candidate changed the parent V30A time strings")
    return validate_endpoint(result, label=CANDIDATE_A)


def compose_time_candidate(parent_v30a: pd.DataFrame, postprocessed_time: Sequence[float]) -> pd.DataFrame:
    """Replace only time after the frozen original gate/post-processing chain."""
    parent = validate_endpoint(parent_v30a, label=PARENT)
    values = np.asarray(postprocessed_time, dtype=np.float64)
    if values.shape != (len(parent),):
        raise ContractError("occurrence-pooled time endpoint row count differs")
    result = parent.copy()
    result["pred_tap_time_len"] = [f"{value:.6f}" for value in roundtrip_six(values)]
    if result.pred_tap_iron.tolist() != parent.pred_tap_iron.tolist():
        raise ContractError("time candidate changed the parent V30A iron strings")
    return validate_endpoint(result, label=CANDIDATE_B)


def assert_legacy_reproduces_parent(
    parent_v30a: pd.DataFrame,
    complete_v26a: pd.DataFrame,
    legacy_oob_iron: Sequence[float],
    legacy_postprocessed_time: Sequence[float],
) -> dict:
    """Regression identity for switching the pooled rule back to equal trees."""
    parent = validate_endpoint(parent_v30a, label=PARENT)
    legacy_iron = compose_iron_candidate(parent, complete_v26a, legacy_oob_iron)
    legacy_time = compose_time_candidate(parent, legacy_postprocessed_time)
    if not legacy_iron.equals(parent):
        raise ContractError("legacy iron switch-back does not reproduce the V30A parent")
    if not legacy_time.equals(parent):
        raise ContractError("legacy time switch-back does not reproduce the V30A parent")
    return {
        "legacy_iron_switch_back_exact": True,
        "legacy_time_switch_back_exact": True,
        "legacy_iron_rows": len(legacy_iron),
        "legacy_time_rows": len(legacy_time),
    }


__all__ = [
    "CANDIDATE_A", "CANDIDATE_B", "PARENT", "V29I", "V29T", "V28I",
    "V26A_PARENT", "V21_REPLAY", "V1", "POOLED_PROTOCOL", "ATTACHMENT_PROTOCOL",
    "PRIMARY_AGGREGATION", "POOLED_AGGREGATION", "TARGETS", "TOLERANCE",
    "compose_iron_candidate", "compose_time_candidate", "assert_legacy_reproduces_parent",
    "macro_origin_summary", "exposure_pooled_summary", "verify_isolated_deltas",
    "submission_bytes", "validate_endpoint",
]
