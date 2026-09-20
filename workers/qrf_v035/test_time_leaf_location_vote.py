from fractions import Fraction

import numpy as np
import pytest

from time_leaf_location_vote import (
    PROTOCOL, derive_midpoint_table, exact_binary64_mean, exact_mean_half_interval_width, exact_midpoint_mean,
    lower_median_vote, median_interval_raw, predict_lower_vote, predict_midpoint_mean,
    validate_midpoint_table,
)


def attachment(groups):
    trees, nodes, offsets, members = [], [], [0], []
    for tree, node, selected in groups:
        trees.append(tree); nodes.append(node); members.extend(selected); offsets.append(len(members))
    return {"leaf_trees": np.asarray(trees), "leaf_nodes": np.asarray(nodes),
            "member_offsets": np.asarray(offsets), "members": np.asarray(members)}


class Tree:
    def __init__(self, node): self.node = node
    def apply(self, matrix): return np.full(len(matrix), self.node, dtype=np.int64)


class Forest:
    def __init__(self): self.estimators_ = [Tree(tree + 10) for tree in range(256)]


def midpoint_table(lower, upper):
    return {"leaf_trees": np.arange(256, dtype=np.int16), "leaf_nodes": np.arange(10, 266),
            "selected_counts": np.full(256, 2, dtype=np.int32),
            "lower_raw": np.asarray(lower, dtype=np.float64), "upper_raw": np.asarray(upper, dtype=np.float64)}


def lower_table(points):
    return {"leaf_trees": np.arange(256, dtype=np.int16), "leaf_nodes": np.arange(10, 266),
            "selected_counts": np.ones(256, dtype=np.int32), "lower_median_raw": np.asarray(points, dtype=np.float64)}


def test_protocol():
    assert PROTOCOL == "FROZEN_OOB_TIME_LEAF_LOCATION_AND_VOTE_v035"


@pytest.mark.parametrize(("members", "expected"), [([0], (80.0, 80.0)), ([0, 1], (80.0, 120.0)), ([0, 1, 2], (100.0, 100.0)), ([0, 1, 2, 3], (100.0, 120.0))])
def test_median_interval(members, expected):
    assert median_interval_raw([80.0, 120.0, 100.0, 180.0], members) == expected


def test_exact_midpoint_does_not_round_each_leaf():
    low, high = [0.1, 0.3], [0.2, 0.4]
    expected = float(sum((Fraction.from_float(v) for v in low + high), Fraction()) / 4)
    assert exact_midpoint_mean(low, high) == expected
    width = sum((Fraction.from_float(h) - Fraction.from_float(l) for l, h in zip(low, high, strict=True)), Fraction())
    assert exact_mean_half_interval_width(low, high) == float(width / 4)


def test_lower_median_vote_uses_smaller_middle():
    assert lower_median_vote([80.0, 110.0, 104.0, 150.0]) == 104.0
    assert lower_median_vote([1.0, 2.0, 3.0]) == 2.0


def test_synthetic_example_is_non_equivalent():
    lower, upper = [80.0, 110.0, 104.0, 150.0], [120.0, 110.0, 106.0, 150.0]
    assert exact_binary64_mean(lower) == 111.0
    assert exact_midpoint_mean(lower, upper) == 116.25
    assert lower_median_vote(lower) == 104.0


def test_table_derivation_binds_same_members_and_odd_intervals():
    table = derive_midpoint_table([80.0, 100.0, 120.0], attachment([(0, 7, [0, 2]), (1, 8, [0, 1, 2])]))
    assert table["lower_raw"].tolist() == [80.0, 100.0]
    assert table["upper_raw"].tolist() == [120.0, 100.0]
    assert table["selected_counts"].tolist() == [2, 3]


@pytest.mark.parametrize("members", [[], [-1], [3], [0, 0]])
def test_invalid_members(members):
    with pytest.raises(ValueError): median_interval_raw([1.0, 2.0, 3.0], members)


def test_midpoint_prediction_is_monotone_and_parent_exact():
    lower = np.arange(256, dtype=float); upper = lower + 2.0
    x = np.asarray([[1.0], [2.0]], dtype=np.float32)
    current, parent, diagnostic = predict_midpoint_mean(Forest(), x, midpoint_table(lower, upper))
    assert np.array_equal(parent, np.full(2, exact_binary64_mean(lower)))
    assert np.array_equal(current, np.full(2, exact_midpoint_mean(lower, upper)))
    assert (current >= parent).all() and diagnostic[0]["nonzero_interval_tree_count"] == 256


def test_lower_vote_prediction_and_order_invariance():
    points = np.arange(256, dtype=float); x = np.asarray([[1.0], [2.0]], dtype=np.float32)
    current, parent, diagnostic = predict_lower_vote(Forest(), x, lower_table(points))
    reverse = predict_lower_vote(Forest(), x[::-1], lower_table(points))[0]
    assert current.tolist() == [127.0, 127.0]
    assert parent.tolist() == [127.5, 127.5]
    assert np.array_equal(reverse, current[::-1]) and diagnostic[0]["distinct_leaf_point_count"] == 256


def test_midpoint_collapse_recovers_parent():
    lower = np.linspace(1.0, 2.0, 256)
    current, parent, _ = predict_midpoint_mean(Forest(), np.asarray([[0.0]], dtype=np.float32), midpoint_table(lower, lower))
    assert np.array_equal(current, parent)


def test_validation_rejects_upper_below_lower_and_wrong_odd_interval():
    table = midpoint_table(np.ones(256), np.ones(256)); table["upper_raw"][0] = 0.0
    with pytest.raises(ValueError, match="ordering"): validate_midpoint_table([0.0, 1.0], table)
    table = midpoint_table(np.ones(256), np.ones(256) * 2); table["selected_counts"][:] = 1
    with pytest.raises(ValueError, match="odd"): validate_midpoint_table([1.0, 2.0], table)


def test_prediction_rejects_wrong_dtype_and_tree_count():
    with pytest.raises(ValueError, match="float32"):
        predict_lower_vote(Forest(), np.asarray([[1.0]], dtype=float), lower_table(np.ones(256)))
    forest = Forest(); forest.estimators_.pop()
    with pytest.raises(ValueError, match="256"):
        predict_lower_vote(forest, np.asarray([[1.0]], dtype=np.float32), lower_table(np.ones(256)))
