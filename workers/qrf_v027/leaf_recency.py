"""Frozen-forest leaf-local recency weighting for v0.27 candidate B."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from qrf_model import distribution_weights, lower_median


PROTOCOL = "QRF_FROZEN_PARTITION_LEAF_RECENCY60_v027"
HALF_LIFE_DAYS = 60.0


def recency60_weights(reference_ns, cutoff_ns):
    reference = np.asarray(reference_ns, dtype=np.int64)
    cutoff = int(cutoff_ns)
    if reference.ndim != 1 or len(reference) == 0 or (reference >= cutoff).any():
        raise ValueError("training reference times must precede cutoff")
    age_days = (cutoff - reference).astype(np.float64) / (86400.0 * 1e9)
    result = np.exp2(-age_days / HALF_LIFE_DAYS).astype(np.float64)
    if not np.isfinite(result).all() or (result <= 0).any() or (result > 1).any():
        raise ValueError("invalid persisted binary64 recency weights")
    return result


def distribution_weights_recency(members, recency, n):
    """Normalise recency separately inside each leaf, preserving equal tree mass."""
    values = np.asarray(recency, dtype=np.float64)
    if values.shape != (n,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("positive aligned binary64 recency vector required")
    if np.all(values == values[0]):
        return distribution_weights(members, n)
    weights = np.zeros(n, dtype=np.float64)
    tree_mass = []
    for indices in members:
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0 or len(set(indices.tolist())) != len(indices):
            raise ValueError("empty or repeated full-training leaf members")
        denominator = float(values[indices].sum())
        contribution = values[indices] / denominator / len(members)
        weights[indices] += contribution
        tree_mass.append(float(contribution.sum()))
    if not np.isfinite(weights).all() or (weights < 0).any() or abs(float(weights.sum()) - 1.0) > 1e-12:
        raise ValueError("invalid leaf-recency distribution mass")
    expected = 1.0 / len(members)
    if max(abs(value - expected) for value in tree_mass) > 1e-14:
        raise ValueError("tree mass is not equal after leaf-local normalisation")
    return weights


def _exact_cdf_at(y, members, recency, value):
    total = Fraction()
    for indices in members:
        denominator = sum((Fraction.from_float(float(recency[index])) for index in indices), Fraction())
        numerator = sum((Fraction.from_float(float(recency[index])) for index in indices if y[index] <= value), Fraction())
        total += numerator / denominator
    return total / len(members)


def lower_median_recency(y, weights, members, recency):
    """Return the lower weighted median, with a binary64-rational boundary oracle."""
    response = np.asarray(y, dtype=np.float64)
    order = np.argsort(response, kind="stable")
    cdf = np.cumsum(np.asarray(weights, dtype=np.float64)[order])
    index = min(int(np.searchsorted(cdf, 0.5, side="left")), len(response) - 1)
    bound = len(response) * np.finfo(np.float64).eps
    nearby = np.flatnonzero(np.abs(cdf - 0.5) <= bound)
    if len(nearby):
        values = sorted(set(response[order].tolist()))
        for value in values:
            if _exact_cdf_at(response, members, recency, float(value)) >= Fraction(1, 2):
                return float(value)
        raise ValueError("exact recency median oracle found no valid response")
    return float(response[order[index]])


def predict_with_recency(model, x, recency, training_months=None):
    if x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("leaf-recency prediction requires finite float32 input")
    if not hasattr(model, "ids") or len(recency) != len(model.ids):
        raise ValueError("model/recency training identity differs")
    routed = np.column_stack([tree.apply(x) for tree in model.forest.estimators_])
    median, mean, diagnostics = [], [], []
    for row in routed:
        members = [model.leaves[index][int(leaf)] for index, leaf in enumerate(row)]
        weights = distribution_weights_recency(members, recency, len(model.y))
        value = lower_median_recency(model.y, weights, members, recency)
        if value < model.y.min() or value > model.y.max():
            raise ValueError("leaf-recency median left training response support")
        median.append(value)
        mean.append(float(np.sum(model.y * weights)))
        diagnostic = {
            "effective_neighbors": float(1.0 / np.sum(weights * weights)),
            "maximum_weight": float(weights.max()),
            "weight_mass": float(weights.sum()),
            "tree_mass_equal": True,
        }
        if training_months is not None:
            months = np.asarray(training_months)
            diagnostic["training_month_weights"] = {
                str(month): float(weights[months == month].sum()) for month in sorted(set(months))
            }
        diagnostics.append(diagnostic)
    return np.asarray(median), np.asarray(mean), diagnostics


def assert_all_ones_reproduces(model, x):
    ones = np.ones(len(model.y), dtype=np.float64)
    original = model.predict(x)
    current = predict_with_recency(model, x, ones)
    if not np.array_equal(original[0], current[0]) or not np.array_equal(original[1], current[1]):
        raise ValueError("all-ones leaf-recency rule does not reproduce parent distribution")
    return True
