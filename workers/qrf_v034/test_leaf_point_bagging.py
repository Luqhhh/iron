from fractions import Fraction

import numpy as np
import pytest

from leaf_point_bagging import (
    PROTOCOL,
    derive_point_table,
    exact_binary64_mean,
    lower_median_raw,
    point_lookup,
    predict,
    validate_point_table,
)


def attachment(groups):
    trees, nodes, offsets, members = [], [], [0], []
    for tree, node, selected in groups:
        trees.append(tree); nodes.append(node); members.extend(selected); offsets.append(len(members))
    return {
        "leaf_trees": np.asarray(trees), "leaf_nodes": np.asarray(nodes),
        "member_offsets": np.asarray(offsets), "members": np.asarray(members),
    }


class Tree:
    def __init__(self, node): self.node = node
    def apply(self, matrix): return np.full(len(matrix), self.node, dtype=np.int64)


class Forest:
    def __init__(self): self.estimators_ = [Tree(tree + 10) for tree in range(256)]


def full_table(points):
    return {
        "leaf_trees": np.arange(256, dtype=np.int16),
        "leaf_nodes": np.arange(10, 266, dtype=np.int64),
        "selected_counts": np.ones(256, dtype=np.int32),
        "lower_median_raw": np.asarray(points, dtype=np.float64),
    }


def test_protocol_is_fixed():
    assert PROTOCOL == "FROZEN_OOB_LEAF_LOWER_MEDIAN_BAGGING_v034"


@pytest.mark.parametrize(("selected", "expected"), [([0], 80.0), ([0, 1], 80.0), ([0, 1, 2], 110.0), ([0, 1, 2, 3], 110.0)])
def test_lower_median_uses_smaller_middle(selected, expected):
    assert lower_median_raw([80.0, 110.0, 200.0, 300.0], selected) == expected


def test_leaf_median_mean_differs_from_old_operations():
    first = lower_median_raw([80, 80, 200, 110, 110, 110], [0, 1, 2])
    second = lower_median_raw([80, 80, 200, 110, 110, 110], [3, 4, 5])
    assert exact_binary64_mean([first, second]) == 95.0
    assert np.mean([80, 80, 200, 110, 110, 110]) == 115.0
    assert np.sort([80, 80, 200, 110, 110, 110])[2] == 110.0


def test_exact_mean_matches_fraction_of_binary64_values():
    values = [0.1, 0.2, 0.3]
    expected = float(sum((Fraction.from_float(value) for value in values), Fraction()) / 3)
    assert exact_binary64_mean(values) == expected


def test_table_derivation_and_source_arrays_are_unchanged():
    response = np.asarray([80.0, 110.0, 200.0])
    source = attachment([(0, 7, [0, 1]), (1, 8, [1, 2, 0])])
    before_response = response.copy(); before_members = source["members"].copy()
    table = derive_point_table(response, source)
    assert table["lower_median_raw"].tolist() == [80.0, 110.0]
    assert table["selected_counts"].tolist() == [2, 3]
    assert point_lookup(table) == {(0, 7): 80.0, (1, 8): 110.0}
    assert np.array_equal(response, before_response)
    assert np.array_equal(source["members"], before_members)


@pytest.mark.parametrize("members", [[], [-1], [3], [0, 0]])
def test_invalid_members_are_rejected(members):
    with pytest.raises(ValueError):
        lower_median_raw([1.0, 2.0, 3.0], members)


def test_empty_selected_set_and_duplicate_key_are_rejected():
    with pytest.raises(ValueError):
        derive_point_table([1.0], attachment([(0, 7, [])]))
    with pytest.raises(ValueError):
        derive_point_table([1.0], attachment([(0, 7, [0]), (0, 7, [0])]))


def test_point_prediction_is_exact_mean_and_order_invariant():
    points = np.linspace(1.0, 256.0, 256)
    table = full_table(points)
    matrix = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
    expected = exact_binary64_mean(points)
    actual, diagnostic = predict(Forest(), matrix, table)
    reverse = predict(Forest(), matrix[::-1], table)[0]
    assert actual.tolist() == [expected] * 3
    assert np.array_equal(reverse, actual[::-1])
    assert diagnostic[0]["tree_count"] == 256
    assert diagnostic[0]["distinct_leaf_point_count"] == 256


def test_point_prediction_may_not_be_training_response_but_stays_in_support():
    points = np.asarray([1.0] * 128 + [2.0] * 128)
    value = predict(Forest(), np.asarray([[0.0]], dtype=np.float32), full_table(points))[0][0]
    assert value == 1.5
    assert 1.0 <= value <= 2.0


def test_prediction_rejects_wrong_dtype_and_tree_count():
    with pytest.raises(ValueError, match="float32"):
        predict(Forest(), np.asarray([[1.0]], dtype=np.float64), full_table(np.ones(256)))
    forest = Forest(); forest.estimators_.pop()
    with pytest.raises(ValueError, match="256"):
        predict(forest, np.asarray([[1.0]], dtype=np.float32), full_table(np.ones(256)))


def test_table_validation_rejects_out_of_support_and_missing_field():
    table = full_table(np.ones(256)); table["lower_median_raw"][0] = 3.0
    with pytest.raises(ValueError, match="support"):
        validate_point_table([1.0, 2.0], table)
    table.pop("leaf_nodes")
    with pytest.raises(ValueError, match="fields"):
        validate_point_table([1.0, 2.0], table)
