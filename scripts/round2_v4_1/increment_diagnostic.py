"""Exact L1 diagnostics for a FIXED candidate direction; no model training.

An alpha fitted and evaluated on the same OOF labels is a descriptive oracle,
not an unbiased validation score, a deployable weight or a submission authority.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np


def _vectors(y: Any, baseline: Any, candidate: Any) -> tuple[np.ndarray, ...]:
    values = tuple(np.asarray(v, dtype=np.float64) for v in (y, baseline, candidate))
    if any(v.ndim != 1 or not len(v) for v in values):
        raise ValueError("Inputs must be nonempty 1-D arrays")
    if len({len(v) for v in values}) != 1:
        raise ValueError("Inputs must have the same length and aligned sample IDs")
    if any(not np.isfinite(v).all() for v in values):
        raise ValueError("Inputs contain NaN or infinity")
    with np.errstate(over="raise", invalid="raise"):
        try:
            r, d = values[0] - values[1], values[2] - values[1]
        except FloatingPointError as exc:
            raise ValueError("Subtraction overflow") from exc
    return *values, r, d


def _loss(r: np.ndarray, d: np.ndarray, alpha: float) -> float:
    with np.errstate(over="raise", invalid="raise"):
        try:
            value = float(np.abs(r - alpha * d).sum())
        except FloatingPointError as exc:
            raise ValueError("Loss overflow") from exc
    if not math.isfinite(value):
        raise ValueError("Nonfinite loss")
    return value


def exact_l1_alpha(y: Any, baseline: Any, candidate: Any) -> float:
    """Minimise sum(abs(y - baseline - alpha*(candidate-baseline))) on [0,1].

    For d != 0, |r-alpha*d| = |d|*|r/d-alpha|.  A weighted median
    (weights |d|) is therefore optimal. Clipping ratios to [0,1] preserves
    the constrained minimiser. Time O(n log n), auxiliary storage O(n).
    Flat optima use the lower weighted median; a tie with baseline uses 0.
    This function is a training primitive, not a validation protocol.
    """
    _, _, _, r, d = _vectors(y, baseline, candidate)
    active = d != 0.0
    if not active.any():
        return 0.0
    weights = np.abs(d[active])
    weights /= weights.max()  # preserve relative weights; avoid overflow
    with np.errstate(over="ignore", divide="ignore", invalid="raise"):
        ratios = np.clip(r[active] / d[active], 0.0, 1.0)
    order = np.argsort(ratios, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, cumulative[-1] / 2.0, side="left")), len(order)-1)
    alpha = float(ratios[order[index]])
    if _loss(r, d, alpha) >= _loss(r, d, 0.0):
        return 0.0
    return alpha


def diagnose_direction(y: Any, baseline: Any, candidate: Any) -> dict[str, Any]:
    """Descriptive diagnostics on one fixed, ID-aligned evaluation sample set.

    Package deltas hold the other target unchanged. The oracle alpha is
    selected using THESE labels and must NEVER be reported as a test score.
    At exact residual ties the right derivative contributes +|d|, not zero.
    """
    y, b, c, r, d = _vectors(y, baseline, candidate)
    if (y < 0).any() or not (0.0 < y.sum() < np.inf):
        raise ValueError("WMAPE requires nonnegative actual targets and a finite positive sum")
    denominator = float(y.sum())
    alpha = exact_l1_alpha(y, b, c)
    zero = r == 0.0
    slope = float(-np.sum(np.sign(r[~zero]) * d[~zero]) + np.abs(d[zero]).sum())
    base, single, half, oracle = (_loss(r, d, a) for a in (0.0, 1.0, 0.5, alpha))
    return {
        "evidence_level": "DESCRIPTIVE_SAME_OOF_LABELS_NOT_VALIDATION",
        "n_rows": int(len(y)),
        "exact_zero_residual_rows": int(zero.sum()),
        "baseline_target_wmape": base/denominator,
        "single_target_wmape": single/denominator,
        "half_blend_target_wmape": half/denominator,
        "single_package_delta": 50.0*(base-single)/denominator,
        "half_blend_package_delta": 50.0*(base-half)/denominator,
        "oracle_alpha_same_labels": alpha,
        "oracle_package_delta_same_labels": 50.0*(base-oracle)/denominator,
        "loss_right_derivative_at_zero": slope,
        "score_right_derivative_at_zero": -50.0*slope/denominator,
        "descent_direction_on_these_rows": bool(slope < 0.0),
        "half_blend_missed_descent": bool(half >= base and oracle < base),
        "model_training_fits": 0,
        "platform_authority": False,
    }
