"""Pure v0.35 time-leaf location and cross-tree vote core."""
from __future__ import annotations

from fractions import Fraction

import numpy as np


PROTOCOL = "FROZEN_OOB_TIME_LEAF_LOCATION_AND_VOTE_v035"
MIDPOINT_FIELDS = {"leaf_trees", "leaf_nodes", "selected_counts", "lower_raw", "upper_raw"}


def median_interval_raw(response, members):
    values = np.asarray(response, dtype=np.float64)
    index = np.asarray(members, dtype=np.int64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty one-dimensional response required")
    if index.ndim != 1 or not len(index) or (index < 0).any() or (index >= len(values)).any():
        raise ValueError("nonempty valid selected members required")
    if len(np.unique(index)) != len(index):
        raise ValueError("selected members must be unique within a tree leaf")
    ordered = np.sort(values[index], kind="mergesort")
    return float(ordered[(len(ordered) - 1) // 2]), float(ordered[len(ordered) // 2])


def derive_midpoint_table(response, attachment):
    required = {"leaf_trees", "leaf_nodes", "member_offsets", "members"}
    if required - set(attachment):
        raise ValueError("certified OOB attachment fields missing")
    trees = np.asarray(attachment["leaf_trees"])
    nodes = np.asarray(attachment["leaf_nodes"])
    offsets = np.asarray(attachment["member_offsets"])
    members = np.asarray(attachment["members"])
    if len(trees) != len(nodes) or len(offsets) != len(trees) + 1 or offsets[0] != 0 or offsets[-1] != len(members):
        raise ValueError("certified OOB selected-member table shape differs")
    if (np.diff(offsets) <= 0).any():
        raise ValueError("every certified selected set must be nonempty")
    keys = list(zip(trees.astype(int), nodes.astype(int), strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate tree/leaf key in attachment")
    lower, upper, counts = [], [], []
    for position in range(len(trees)):
        selected = members[offsets[position]:offsets[position + 1]]
        low, high = median_interval_raw(response, selected)
        lower.append(low); upper.append(high); counts.append(len(selected))
    table = {
        "leaf_trees": trees.astype(np.int16, copy=True),
        "leaf_nodes": nodes.astype(np.int64, copy=True),
        "selected_counts": np.asarray(counts, dtype=np.int32),
        "lower_raw": np.asarray(lower, dtype=np.float64),
        "upper_raw": np.asarray(upper, dtype=np.float64),
    }
    validate_midpoint_table(response, table)
    return table


def validate_midpoint_table(response, table):
    values = np.asarray(response, dtype=np.float64)
    if set(table) != MIDPOINT_FIELDS:
        raise ValueError("v0.35 midpoint table fields differ")
    length = len(table["leaf_trees"])
    if not length or any(len(table[name]) != length for name in MIDPOINT_FIELDS):
        raise ValueError("v0.35 midpoint table vector lengths differ")
    keys = list(zip(np.asarray(table["leaf_trees"]).astype(int), np.asarray(table["leaf_nodes"]).astype(int), strict=True))
    lower, upper = np.asarray(table["lower_raw"], dtype=np.float64), np.asarray(table["upper_raw"], dtype=np.float64)
    counts = np.asarray(table["selected_counts"])
    if len(keys) != len(set(keys)) or (counts <= 0).any():
        raise ValueError("v0.35 midpoint table keys/counts invalid")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (upper < lower).any():
        raise ValueError("v0.35 median interval ordering invalid")
    if (lower < values.min()).any() or (upper > values.max()).any():
        raise ValueError("v0.35 median interval left raw response support")
    if not np.array_equal(lower[counts % 2 == 1], upper[counts % 2 == 1]):
        raise ValueError("odd selected sets must have a degenerate median interval")
    return True


def exact_binary64_mean(values):
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 1 or not len(points) or not np.isfinite(points).all():
        raise ValueError("finite nonempty leaf points required")
    return float(sum((Fraction.from_float(float(value)) for value in points), Fraction()) / len(points))


def exact_midpoint_mean(lower, upper):
    low, high = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if low.shape != high.shape or low.ndim != 1 or not len(low) or not np.isfinite(low).all() or not np.isfinite(high).all() or (high < low).any():
        raise ValueError("valid paired median interval endpoints required")
    total = sum((Fraction.from_float(float(value)) for value in low), Fraction())
    total += sum((Fraction.from_float(float(value)) for value in high), Fraction())
    return float(total / (2 * len(low)))


def exact_mean_half_interval_width(lower, upper):
    low, high = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if low.shape != high.shape or low.ndim != 1 or not len(low) or (high < low).any():
        raise ValueError("valid paired median interval endpoints required")
    width = sum((Fraction.from_float(float(h)) - Fraction.from_float(float(l)) for l, h in zip(low, high, strict=True)), Fraction())
    return float(width / (2 * len(low)))


def lower_median_vote(values):
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 1 or not len(points) or not np.isfinite(points).all():
        raise ValueError("finite nonempty tree points required")
    ordered = np.sort(points, kind="mergesort")
    return float(ordered[(len(ordered) - 1) // 2])


def _lookup(table, lower_field, upper_field=None):
    result = {}
    for position, (tree, node) in enumerate(zip(table["leaf_trees"], table["leaf_nodes"], strict=True)):
        key = (int(tree), int(node))
        result[key] = (float(table[lower_field][position]),
                       float(table[upper_field][position]) if upper_field else None)
    return result


def _route(forest, transformed):
    matrix = np.asarray(transformed)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("v0.35 prediction requires finite float32 matrix")
    if len(forest.estimators_) != 256:
        raise ValueError("v0.35 requires exactly 256 frozen trees")
    return np.column_stack([tree.apply(matrix) for tree in forest.estimators_])


def predict_midpoint_mean(forest, transformed, table):
    routed, lookup = _route(forest, transformed), _lookup(table, "lower_raw", "upper_raw")
    prediction, parent, diagnostics = [], [], []
    for row in routed:
        pairs = [lookup[(tree, int(node))] for tree, node in enumerate(row)]
        lower = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        upper = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        current, baseline = exact_midpoint_mean(lower, upper), exact_binary64_mean(lower)
        if current < baseline:
            raise ValueError("v0.35 midpoint prediction fell below lower-only parent")
        prediction.append(current); parent.append(baseline)
        diagnostics.append({
            "nonzero_interval_tree_count": int(np.count_nonzero(upper > lower)),
            "mean_half_interval_width": exact_mean_half_interval_width(lower, upper),
            "raw_delta": float(current - baseline),
        })
    return np.asarray(prediction), np.asarray(parent), diagnostics


def predict_lower_vote(forest, transformed, lower_table):
    routed, lookup = _route(forest, transformed), _lookup(lower_table, "lower_median_raw")
    prediction, parent, diagnostics = [], [], []
    for row in routed:
        points = np.asarray([lookup[(tree, int(node))][0] for tree, node in enumerate(row)], dtype=np.float64)
        vote, baseline = lower_median_vote(points), exact_binary64_mean(points)
        prediction.append(vote); parent.append(baseline)
        diagnostics.append({
            "leaf_point_minimum": float(points.min()), "leaf_point_maximum": float(points.max()),
            "distinct_leaf_point_count": int(len(np.unique(points))), "raw_delta": float(vote - baseline),
        })
    return np.asarray(prediction), np.asarray(parent), diagnostics
