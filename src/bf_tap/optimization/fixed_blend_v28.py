"""Pure contracts for the two preregistered v0.28 fixed equal blends."""
from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN

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
    verify_isolated_deltas,
)


CANDIDATE_A = "V28I_CB_QRF_EQUAL_BLEND"
CANDIDATE_B = "V28T_L1_L2_QRF_EQUAL_BLEND"
PARENT = "V26A_PARENT"
IRON_DONOR = "V27I_ENDPOINT"
TIME_DONOR = "V21_REPLAY"
PREDICTION_COLUMNS = ("pred_tap_iron", "pred_tap_time_len")
CSV_COLUMNS = ("sample_id", *PREDICTION_COLUMNS)
SIX_DECIMAL = re.compile(r"^(?:0|[1-9][0-9]*)\.[0-9]{6}$")
MICRO = Decimal("0.000001")


def canonical_six(value: str) -> str:
    """Canonicalise an already-six-decimal numeric value without changing it."""
    text = str(value)
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ContractError("finite nonnegative six-decimal endpoint required") from error
    if not number.is_finite() or number < 0 or number != number.quantize(MICRO, rounding=ROUND_HALF_EVEN):
        raise ContractError("finite nonnegative six-decimal endpoint required")
    result = format(number, ".6f")
    if SIX_DECIMAL.fullmatch(result) is None:
        raise ContractError("finite nonnegative six-decimal endpoint required")
    return result


def micro_integer(value: str) -> int:
    return int(Decimal(canonical_six(value)) * 1_000_000)


def equal_blend_six(left: str, right: str) -> str:
    """Average two six-decimal endpoints in integer micro-units, ties to even."""
    total = micro_integer(left) + micro_integer(right)
    quotient, remainder = divmod(total, 2)
    if remainder and quotient % 2:
        quotient += 1
    return f"{quotient // 1_000_000}.{quotient % 1_000_000:06d}"


def validate_endpoint(value: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if list(value) != list(CSV_COLUMNS) or value.isna().any().any():
        raise ContractError(f"{label} must contain exactly the three submission columns")
    result = value.copy()
    result["sample_id"] = result.sample_id.astype(str)
    if result.sample_id.duplicated().any() or (result.sample_id == "").any():
        raise ContractError(f"{label} sample IDs must be unique and nonempty")
    for column in PREDICTION_COLUMNS:
        result[column] = [canonical_six(item) for item in result[column].astype(str)]
    return result


def align_endpoint(parent: pd.DataFrame, donor: pd.DataFrame, *, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = validate_endpoint(parent, label="parent")
    right = validate_endpoint(donor, label=label)
    if set(left.sample_id) != set(right.sample_id):
        raise ContractError(f"{label}/parent sample ID sets differ")
    right = right.set_index("sample_id").loc[left.sample_id].reset_index()
    return left, right


def compose_candidate(parent: pd.DataFrame, donor: pd.DataFrame, *, changed_target: str) -> pd.DataFrame:
    """Compose one registered all-row 50/50 blend after complete endpoints."""
    left, right = align_endpoint(parent, donor, label="donor")
    if changed_target not in TARGETS:
        raise ContractError("exactly one registered target must change")
    changed = f"pred_{changed_target}"
    unchanged = "pred_tap_time_len" if changed_target == "tap_iron" else "pred_tap_iron"
    if left[unchanged].tolist() != right[unchanged].tolist():
        raise ContractError("donor changed the target required to remain isolated")
    result = left.copy()
    result[changed] = [equal_blend_six(a, b) for a, b in zip(left[changed], right[changed], strict=True)]
    if result[unchanged].tolist() != left[unchanged].tolist():
        raise ContractError("blend changed the isolated parent target")
    return result


def submission_bytes(value: pd.DataFrame) -> bytes:
    checked = validate_endpoint(value, label="candidate")
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    writer.writerows(checked.itertuples(index=False, name=None))
    return handle.getvalue().encode("utf-8")


def cancellation_diagnostic(
    actual: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    rounded: np.ndarray,
) -> Mapping[str, float | int]:
    """Return the absolute-error cancellation identity for one cell/target."""
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    r = np.asarray(rounded, dtype=np.float64)
    if any(item.ndim != 1 for item in (y, p, q, r)) or not (len(y) == len(p) == len(q) == len(r)) or not len(y):
        raise ContractError("aligned nonempty one-dimensional diagnostic arrays required")
    if not all(np.isfinite(item).all() for item in (y, p, q, r)):
        raise ContractError("finite diagnostic values required")
    e, f = p - y, q - y
    cancellation = (np.abs(e) + np.abs(f) - np.abs(e + f)) / 2.0
    endpoint_average = float((np.abs(e).sum() + np.abs(f).sum()) / 2.0)
    exact = (p + q) / 2.0
    exact_error = float(np.abs(exact - y).sum())
    identity_residual = abs(endpoint_average - float(cancellation.sum()) - exact_error)
    if identity_residual > 1e-9:
        raise ContractError("absolute-error cancellation identity failed")
    rounded_error = float(np.abs(r - y).sum())
    rounding_effect = rounded_error - exact_error
    bound = len(y) * 0.5e-6
    if abs(rounding_effect) > bound + 1e-9:
        raise ContractError("six-decimal blend exceeded rounding error bound")
    return {
        "n": int(len(y)),
        "opposite_sign_count": int(np.sum(e * f < 0)),
        "opposite_sign_rate": float(np.mean(e * f < 0)),
        "cancellation_sum": float(cancellation.sum()),
        "endpoint_average_abs_error_sum": endpoint_average,
        "exact_unrounded_blend_abs_error_sum": exact_error,
        "rounded_blend_abs_error_sum": rounded_error,
        "rounding_effect_abs_error_sum": rounding_effect,
        "rounding_effect_bound": bound,
        "identity_residual": identity_residual,
        "endpoint_error_correlation": float(np.corrcoef(e, f)[0, 1]) if len(y) > 1 and np.std(e) and np.std(f) else np.nan,
    }


__all__ = [
    "CANDIDATE_A", "CANDIDATE_B", "PARENT", "IRON_DONOR", "TIME_DONOR",
    "PRIMARY_AGGREGATION", "POOLED_AGGREGATION", "TARGETS", "TOLERANCE",
    "canonical_six", "micro_integer", "equal_blend_six", "validate_endpoint",
    "align_endpoint", "compose_candidate", "submission_bytes", "cancellation_diagnostic",
    "macro_origin_summary", "exposure_pooled_summary", "verify_isolated_deltas",
]
