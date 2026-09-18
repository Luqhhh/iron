"""Pure composition contracts for frozen-forest OOB leaf responses."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..exceptions import ContractError
from .fixed_blend_v28 import compose_candidate, validate_endpoint
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


CANDIDATE_A = "V29I_OOB_LEAF_QRF_BLEND"
CANDIDATE_B = "V29T_OOB_LEAF_QRF_TIME"
PARENT = "V28I_PARENT"
IRON_COMPLETE_ENDPOINT = "V26A_COMPLETE_IRON"
V26A_PARENT = "V26A_PARENT"
V21_REPLAY = "V21_REPLAY"


def _aligned(parent: pd.DataFrame, source: pd.DataFrame, label: str):
    left = validate_endpoint(parent, label="V28I parent")
    right = validate_endpoint(source, label=label)
    if set(left.sample_id) != set(right.sample_id):
        raise ContractError(f"{label}/parent sample ID sets differ")
    return left, right.set_index("sample_id").loc[left.sample_id].reset_index()


def _six(values: Sequence[float]) -> list[str]:
    rounded = roundtrip_six(values)
    return [f"{value:.6f}" for value in rounded]


def compose_iron_candidate(parent_v28i: pd.DataFrame, complete_catboost: pd.DataFrame, oob_qrf_iron: Sequence[float]) -> pd.DataFrame:
    """Average complete CatBoost-chain iron with the new OOB-QRF endpoint."""
    parent, catboost = _aligned(parent_v28i, complete_catboost, "V26A complete endpoint")
    if parent.pred_tap_time_len.tolist() != catboost.pred_tap_time_len.tolist():
        raise ContractError("V28I/V26A time endpoint identity differs")
    if len(oob_qrf_iron) != len(parent):
        raise ContractError("OOB iron endpoint row count differs")
    donor = catboost.copy()
    donor["pred_tap_iron"] = _six(oob_qrf_iron)
    result = compose_candidate(catboost, donor, changed_target="tap_iron")
    result["pred_tap_time_len"] = parent.pred_tap_time_len.tolist()
    if result.pred_tap_time_len.tolist() != parent.pred_tap_time_len.tolist():
        raise ContractError("iron OOB candidate changed V28I time strings")
    return validate_endpoint(result, label=CANDIDATE_A)


def compose_time_candidate(parent_v28i: pd.DataFrame, oob_postprocessed_time: Sequence[float]) -> pd.DataFrame:
    """Replace only time after the frozen V21 gate/post-processing chain."""
    parent = validate_endpoint(parent_v28i, label="V28I parent")
    if len(oob_postprocessed_time) != len(parent):
        raise ContractError("OOB time endpoint row count differs")
    result = parent.copy()
    result["pred_tap_time_len"] = _six(oob_postprocessed_time)
    if result.pred_tap_iron.tolist() != parent.pred_tap_iron.tolist():
        raise ContractError("time OOB candidate changed V28I iron strings")
    return validate_endpoint(result, label=CANDIDATE_B)


__all__ = [
    "CANDIDATE_A", "CANDIDATE_B", "PARENT", "IRON_COMPLETE_ENDPOINT",
    "V26A_PARENT", "V21_REPLAY", "PRIMARY_AGGREGATION", "POOLED_AGGREGATION",
    "TARGETS", "TOLERANCE", "compose_iron_candidate", "compose_time_candidate",
    "macro_origin_summary", "exposure_pooled_summary", "verify_isolated_deltas",
]
