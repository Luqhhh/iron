"""Pure v0.34 OOB leaf-lower-median bagging core."""
from __future__ import annotations

from fractions import Fraction

import numpy as np


PROTOCOL = "FROZEN_OOB_LEAF_LOWER_MEDIAN_BAGGING_v034"
TABLE_FIELDS = {"leaf_trees", "leaf_nodes", "selected_counts", "lower_median_raw"}


def lower_median_raw(response, members) -> float:
    values = np.asarray(response, dtype=np.float64)
    index = np.asarray(members, dtype=np.int64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty one-dimensional response required")
    if index.ndim != 1 or not len(index) or (index < 0).any() or (index >= len(values)).any():
        raise ValueError("nonempty valid selected members required")
    if len(np.unique(index)) != len(index):
        raise ValueError("selected members must be unique within a tree leaf")
    ordered = np.sort(values[index], kind="mergesort")
    return float(ordered[(len(ordered) - 1) // 2])


def derive_point_table(response, attachment):
    values = np.asarray(response, dtype=np.float64)
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
    points, counts = [], []
    for index in range(len(trees)):
        selected = members[offsets[index]:offsets[index + 1]]
        counts.append(len(selected))
        points.append(lower_median_raw(values, selected))
    table = {
        "leaf_trees": trees.astype(np.int16, copy=True),
        "leaf_nodes": nodes.astype(np.int64, copy=True),
        "selected_counts": np.asarray(counts, dtype=np.int32),
        "lower_median_raw": np.asarray(points, dtype=np.float64),
    }
    validate_point_table(values, table)
    return table


def validate_point_table(response, table):
    values = np.asarray(response, dtype=np.float64)
    if set(table) != TABLE_FIELDS:
        raise ValueError("v0.34 leaf point table fields differ")
    length = len(table["leaf_trees"])
    if not length or any(len(table[name]) != length for name in TABLE_FIELDS):
        raise ValueError("v0.34 leaf point table vector lengths differ")
    keys = list(zip(np.asarray(table["leaf_trees"]).astype(int), np.asarray(table["leaf_nodes"]).astype(int), strict=True))
    if len(keys) != len(set(keys)) or (np.asarray(table["selected_counts"]) <= 0).any():
        raise ValueError("v0.34 leaf point table keys/counts invalid")
    points = np.asarray(table["lower_median_raw"], dtype=np.float64)
    if not np.isfinite(points).all() or (points < values.min()).any() or (points > values.max()).any():
        raise ValueError("v0.34 leaf points left raw response support")
    return True


def exact_binary64_mean(values) -> float:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 1 or not len(points) or not np.isfinite(points).all():
        raise ValueError("finite nonempty leaf points required")
    total = sum((Fraction.from_float(float(value)) for value in points), Fraction(0, 1))
    return float(total / len(points))


def point_lookup(table):
    return {
        (int(tree), int(node)): float(point)
        for tree, node, point in zip(table["leaf_trees"], table["leaf_nodes"], table["lower_median_raw"], strict=True)
    }


def predict(forest, transformed, table):
    matrix = np.asarray(transformed)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("v0.34 prediction requires finite float32 matrix")
    if len(forest.estimators_) != 256:
        raise ValueError("v0.34 requires exactly 256 frozen trees")
    lookup = point_lookup(table)
    routed = np.column_stack([tree.apply(matrix) for tree in forest.estimators_])
    predictions, diagnostics = [], []
    for row in routed:
        points = np.asarray([lookup[(tree, int(node))] for tree, node in enumerate(row)], dtype=np.float64)
        value = exact_binary64_mean(points)
        predictions.append(value)
        diagnostics.append({
            "tree_count": 256,
            "leaf_point_minimum": float(points.min()),
            "leaf_point_maximum": float(points.max()),
            "leaf_point_range": float(points.max() - points.min()),
            "leaf_point_standard_deviation": float(points.std(ddof=0)),
            "distinct_leaf_point_count": int(len(np.unique(points))),
        })
    return np.asarray(predictions, dtype=np.float64), diagnostics
