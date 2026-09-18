"""Pure contracts for the two preregistered v0.30 candidates."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

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


CANDIDATE_A = "V30A_OOB_BOTH_TARGETS"
CANDIDATE_B = "V30B_OOB_TIME_1024"
PARENT = "V28I_PARENT"
IRON_SOURCE = "V29I_OOB_LEAF_QRF_BLEND"
TIME_SOURCE = "V29T_OOB_LEAF_QRF_TIME"
V26A_PARENT = "V26A_PARENT"
V21_REPLAY = "V21_REPLAY"
V1 = "V1"
OOB_ATTACHMENT_PROTOCOL = "QRF_OOB_LEAF_APPENDED_1024_v030"
APPENDED_FOREST_PROTOCOL = "QRF_TIME_WARM_START_APPEND_1024_v030"


def align_sources(iron_source: pd.DataFrame, time_source: pd.DataFrame):
    """Align the two complete v0.29 endpoints by sample ID, V29I order first."""
    left = validate_endpoint(iron_source, label=IRON_SOURCE)
    right = validate_endpoint(time_source, label=TIME_SOURCE)
    if set(left.sample_id) != set(right.sample_id):
        raise ContractError("v0.29 iron/time source sample ID sets differ")
    right = right.set_index("sample_id").loc[left.sample_id].reset_index()
    if right.sample_id.tolist() != left.sample_id.tolist():
        raise ContractError("v0.29 source alignment changed sample ID order")
    return left, right


def compose_both_targets(iron_source: pd.DataFrame, time_source: pd.DataFrame) -> pd.DataFrame:
    """Candidate A: copy both complete source columns without recomputation."""
    left, right = align_sources(iron_source, time_source)
    result = left.copy()
    result["pred_tap_time_len"] = right.pred_tap_time_len.tolist()
    if result.pred_tap_iron.tolist() != left.pred_tap_iron.tolist():
        raise ContractError("candidate A changed an iron string while composing time")
    if result.pred_tap_time_len.tolist() != right.pred_tap_time_len.tolist():
        raise ContractError("candidate A changed a time string while composing iron")
    return validate_endpoint(result, label=CANDIDATE_A)


def compose_1024_time(candidate_a: pd.DataFrame, postprocessed_time: Sequence[float]) -> pd.DataFrame:
    """Candidate B: replace only time after the frozen gate/post-processing chain."""
    parent = validate_endpoint(candidate_a, label=CANDIDATE_A)
    values = np.asarray(postprocessed_time, dtype=np.float64)
    if len(values) != len(parent):
        raise ContractError("v0.30 1024-tree time endpoint row count differs")
    result = parent.copy()
    result["pred_tap_time_len"] = [f"{value:.6f}" for value in roundtrip_six(values)]
    if result.pred_tap_iron.tolist() != parent.pred_tap_iron.tolist():
        raise ContractError("1024-tree time candidate changed the A iron strings")
    return validate_endpoint(result, label=CANDIDATE_B)


def _cell_index(scorecard: pd.DataFrame):
    required = {"algorithm", "unit", "target", "n", "target_sum", "wmape", "E"}
    if required - set(scorecard):
        raise ContractError("complete v0.30 scorecard required for composition identities")
    result = {}
    for algorithm, part in scorecard.groupby("algorithm", sort=False):
        for unit, cell in part.groupby("unit", sort=False):
            by_target = cell.set_index("target")
            if set(by_target.index) != set(TARGETS) or len(cell) != 2:
                raise ContractError(f"cell {algorithm}/{unit} requires exactly two target rows")
            if cell.n.nunique() != 1 or cell.E.nunique() != 1:
                raise ContractError(f"cell {algorithm}/{unit} target rows disagree on n/E")
            result[(algorithm, unit)] = {
                "n": int(cell.n.iloc[0]),
                "target_sum": {target: float(by_target.loc[target, "target_sum"]) for target in TARGETS},
                "iron": float(by_target.loc["tap_iron", "wmape"]),
                "time": float(by_target.loc["tap_time_len", "wmape"]),
                "E": float(cell.E.iloc[0]),
            }
    return result


def _macro_index(summary: pd.DataFrame):
    primary = summary.loc[summary.aggregation == PRIMARY_AGGREGATION]
    result = {}
    for row in primary.itertuples(index=False):
        result[(row.algorithm, row.scope)] = {
            "iron": float(row.wmape_iron), "time": float(row.wmape_time), "E": float(row.E),
        }
    return result


def _check_scope(name: str, values: Mapping, algorithms, *, tolerance: float):
    checks = {
        "A_iron_exact_copy": abs(values[CANDIDATE_A]["iron"] - values[IRON_SOURCE]["iron"]),
        "A_time_exact_copy": abs(values[CANDIDATE_A]["time"] - values[TIME_SOURCE]["time"]),
        "B_iron_exact_copy": abs(values[CANDIDATE_B]["iron"] - values[CANDIDATE_A]["iron"]),
        "A_additivity": abs(
            values[CANDIDATE_A]["E"] - values[PARENT]["E"]
            - (values[IRON_SOURCE]["E"] - values[PARENT]["E"])
            - (values[TIME_SOURCE]["E"] - values[PARENT]["E"])
        ),
        "A_vs_time_half_delta": abs(
            values[CANDIDATE_A]["E"] - values[TIME_SOURCE]["E"]
            - 0.5 * (values[CANDIDATE_A]["iron"] - values[TIME_SOURCE]["iron"])
        ),
        "B_vs_A_half_delta": abs(
            values[CANDIDATE_B]["E"] - values[CANDIDATE_A]["E"]
            - 0.5 * (values[CANDIDATE_B]["time"] - values[CANDIDATE_A]["time"])
        ),
    }
    for label, residual in checks.items():
        if residual > tolerance:
            raise ContractError(f"{name}: {label} exceeded tolerance ({residual} > {tolerance})")
    return checks


def verify_composition_identities(scorecard: pd.DataFrame, summary: pd.DataFrame, *, tolerance: float = TOLERANCE) -> dict:
    """Prove the registered column copies, additivity and half-delta identities."""
    cells = _cell_index(scorecard)
    macros = _macro_index(summary)
    algorithms = (PARENT, IRON_SOURCE, TIME_SOURCE, CANDIDATE_A, CANDIDATE_B)
    for name, index in (("cell", cells), ("macro", macros)):
        for algorithm in algorithms:
            if not any(key[0] == algorithm for key in index):
                raise ContractError(f"{name} scorecard is missing a registered v0.30 algorithm: {algorithm}")
    maximum_residual = 0.0
    checked_cells = 0
    checked_macros = 0
    for unit in sorted({key[1] for key in cells if key[0] == PARENT}):
        values = {algorithm: cells[(algorithm, unit)] for algorithm in algorithms}
        if len({values[algorithm]["n"] for algorithm in algorithms}) != 1:
            raise ContractError(f"cell {unit}: candidate/reference denominators differ")
        for target in TARGETS:
            denominators = {values[algorithm]["target_sum"][target] for algorithm in algorithms}
            if len(denominators) != 1:
                raise ContractError(f"cell {unit}/{target}: target denominators differ")
        checks = _check_scope(f"cell {unit}", values, algorithms, tolerance=tolerance)
        maximum_residual = max(maximum_residual, *checks.values())
        checked_cells += 1
    for scope in sorted({key[1] for key in macros if key[0] == PARENT}):
        values = {algorithm: macros[(algorithm, scope)] for algorithm in algorithms}
        checks = _check_scope(f"macro {scope}", values, algorithms, tolerance=tolerance)
        maximum_residual = max(maximum_residual, *checks.values())
        checked_macros += 1
    return {
        "checked_cells": checked_cells,
        "checked_macro_summaries": checked_macros,
        "maximum_absolute_delta_identity_residual": maximum_residual,
        "tolerance": tolerance,
    }


__all__ = [
    "CANDIDATE_A", "CANDIDATE_B", "PARENT", "IRON_SOURCE", "TIME_SOURCE",
    "V26A_PARENT", "V21_REPLAY", "V1", "OOB_ATTACHMENT_PROTOCOL",
    "APPENDED_FOREST_PROTOCOL", "PRIMARY_AGGREGATION", "POOLED_AGGREGATION",
    "TARGETS", "TOLERANCE", "align_sources", "compose_both_targets",
    "compose_1024_time", "verify_composition_identities",
    "macro_origin_summary", "exposure_pooled_summary", "verify_isolated_deltas",
    "submission_bytes",
]
