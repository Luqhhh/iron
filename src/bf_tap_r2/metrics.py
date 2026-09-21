"""Local comparison metrics; deliberately no unverified platform score mapping."""
from __future__ import annotations

import numpy as np


def wmape(actual, predicted) -> float:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError("Expected equally sized nonempty one-dimensional arrays")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require finite values")
    denominator = np.abs(y).sum()
    if denominator <= 0:
        raise ValueError("WMAPE denominator must be positive")
    return float(np.abs(y - p).sum() / denominator)


def score_targets(iron_actual, iron_predicted, time_actual, time_predicted) -> dict[str, float]:
    if len(iron_actual) != len(time_actual):
        raise ValueError("Both targets must cover the same aligned samples")
    iron = wmape(iron_actual, iron_predicted)
    time = wmape(time_actual, time_predicted)
    return {"W_I": iron, "W_T": time, "J": (iron + time) / 2}
