from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "qrf_v015"))

from same_spout import (
    PROTOCOL,
    QUERY_KNOWN,
    QUERY_MISSING,
    QUERY_UNKNOWN,
    condition_members,
    conditioned_lower_median,
    exact_tokens,
    query_state,
    validate_training_spout,
)


class Preprocessor:
    training_ids = ["a", "b", "c", "d"]
    vocabulary = ["1", "2"]
    transformed_columns = ["x", "spout_no:known:1", "spout_no:known:2", "spout_no:missing", "spout_no:unknown"]


def test_protocol_frozen():
    assert PROTOCOL == "QRF_FROZEN_FOREST_SAME_SPOUT_OOB_RESPONSE_v032"


def test_exact_tokens_preserve_text_without_numeric_coercion():
    assert exact_tokens(np.asarray(["1", "1.0", "", "2"])).tolist() == ["1", "1.0", "", "2"]


def test_exact_tokens_reject_multidimensional_metadata():
    with pytest.raises(ValueError, match="one-dimensional"):
        exact_tokens(np.asarray([["1"]]))


def test_training_spout_identity_and_vocabulary_pass():
    tokens = validate_training_spout(["1", "2", "1", "2"], ["a", "b", "c", "d"], ["a", "b", "c", "d"], Preprocessor())
    assert tokens.tolist() == ["1", "2", "1", "2"]


@pytest.mark.parametrize("ids,model_ids", [
    (["a", "b", "c", "d"], ["b", "a", "c", "d"]),
    (["a", "b", "c", "x"], ["a", "b", "c", "d"]),
    (["a", "a", "c", "d"], ["a", "a", "c", "d"]),
])
def test_training_spout_rejects_bad_id_identity(ids, model_ids):
    with pytest.raises(ValueError, match="alignment|IDs"):
        validate_training_spout(["1", "2", "1", "2"], ids, model_ids, Preprocessor())


def test_training_spout_rejects_token_vocabulary_mismatch():
    with pytest.raises(ValueError, match="vocabulary"):
        validate_training_spout(["1", "2", "3", "2"], ["a", "b", "c", "d"], ["a", "b", "c", "d"], Preprocessor())


def test_training_spout_rejects_transformed_schema_mismatch():
    preprocessor = Preprocessor()
    preprocessor.transformed_columns = ["x", "spout_no:known:1", "spout_no:missing", "spout_no:unknown"]
    with pytest.raises(ValueError, match="schema"):
        validate_training_spout(["1", "2", "1", "2"], ["a", "b", "c", "d"], ["a", "b", "c", "d"], preprocessor)


@pytest.mark.parametrize("value,state", [("1", QUERY_KNOWN), ("2", QUERY_KNOWN), ("", QUERY_MISSING), ("3", QUERY_UNKNOWN), ("1.0", QUERY_UNKNOWN)])
def test_query_state_exact(value, state):
    assert query_state(value, ["1", "2"])[1] == state


def test_same_spout_intersection_is_used_when_nonempty():
    selected, diagnostic = condition_members([np.asarray([0, 1, 2])], ["1", "2", "1", "2"], "1", ["1", "2"])
    assert selected[0].tolist() == [0, 2]
    assert diagnostic["same_spout_nonempty_tree_count"] == 1
    assert diagnostic["condition_fallback_tree_count"] == 0


def test_empty_intersection_returns_original_selected_set_not_full_leaf():
    original = np.asarray([1, 3], dtype=np.int64)
    selected, diagnostic = condition_members([original], ["1", "2", "1", "2"], "1", ["1", "2"])
    assert np.array_equal(selected[0], original)
    assert diagnostic["condition_fallback_tree_count"] == 1


@pytest.mark.parametrize("query", ["", "3", "1.0"])
def test_missing_or_unknown_query_bypasses_all_trees(query):
    original = [np.asarray([0, 1]), np.asarray([2, 3])]
    selected, diagnostic = condition_members(original, ["1", "2", "1", "2"], query, ["1", "2"])
    assert all(np.array_equal(left, right) for left, right in zip(original, selected, strict=True))
    assert diagnostic["conditioning_bypassed"] is True


@pytest.mark.parametrize("query,expected", [("1", [[0, 2], [0]]), ("2", [[1], [1, 3]])])
def test_rule_is_symmetric_between_registered_spouts(query, expected):
    selected, _ = condition_members([np.asarray([0, 1, 2]), np.asarray([0, 1, 3])], ["1", "2", "1", "2"], query, ["1", "2"])
    assert [part.tolist() for part in selected] == expected


def test_one_same_spout_member_is_legal():
    selected, _ = condition_members([np.asarray([0, 1, 3])], ["1", "2", "1", "2"], "1", ["1", "2"])
    assert selected[0].tolist() == [0]


def test_cross_tree_repeat_remains_present_in_each_tree():
    selected, _ = condition_members([np.asarray([0, 1]), np.asarray([0, 2])], ["1", "2", "1"], "1", ["1", "2"])
    assert selected[0].tolist() == [0]
    assert selected[1].tolist() == [0, 2]


def test_conditioning_changes_equal_tree_lower_median_without_occurrence_pooling():
    y = np.asarray([0.0, 10.0, 20.0, 30.0])
    selected = [np.asarray([0, 1]), np.asarray([2, 3])]
    value, legacy, diagnostic = conditioned_lower_median(y, selected, ["1", "2", "1", "2"], "1", ["1", "2"])
    assert legacy == 10.0
    assert value == 0.0
    assert diagnostic["new_cross_spout_mass"] == 0.0


def test_empty_intersection_tree_keeps_equal_tree_mass():
    y = np.asarray([0.0, 10.0, 20.0, 30.0])
    selected = [np.asarray([0, 2]), np.asarray([1, 3])]
    value, legacy, diagnostic = conditioned_lower_median(y, selected, ["1", "2", "1", "2"], "1", ["1", "2"])
    assert diagnostic["condition_fallback_tree_count"] == 1
    assert diagnostic["same_spout_nonempty_tree_count"] == 1
    assert diagnostic["new_cross_spout_mass"] == pytest.approx(0.5)
    assert value == legacy == 10.0


def test_original_oob_fallback_and_condition_fallback_are_separate():
    _, _, diagnostic = conditioned_lower_median(
        [0.0, 10.0, 20.0, 30.0], [np.asarray([0, 2]), np.asarray([1, 3])],
        ["1", "2", "1", "2"], "1", ["1", "2"], original_fallback_modes=[True, False],
    )
    assert diagnostic["original_oob_fallback_tree_count"] == 1
    assert diagnostic["condition_fallback_tree_count"] == 1


@pytest.mark.parametrize("query", ["1", "2"])
def test_cross_spout_mass_never_increases_for_known_query(query):
    _, _, diagnostic = conditioned_lower_median(
        np.arange(6.0), [np.asarray([0, 1, 2]), np.asarray([2, 3, 4, 5])],
        ["1", "2", "1", "2", "1", "2"], query, ["1", "2"],
    )
    assert diagnostic["new_cross_spout_mass"] <= diagnostic["original_cross_spout_mass"] + 1e-12


@pytest.mark.parametrize("query,state", [("", QUERY_MISSING), ("3", QUERY_UNKNOWN)])
def test_bypass_median_exactly_matches_legacy(query, state):
    value, legacy, diagnostic = conditioned_lower_median(
        [1.0, 2.0, 3.0], [np.asarray([0, 1]), np.asarray([1, 2])], ["1", "2", "1"], query, ["1", "2"],
    )
    assert value == legacy
    assert diagnostic["query_state"] == int(state)
    assert np.isnan(diagnostic["new_cross_spout_mass"])


def test_known_all_same_spout_is_exact_noop():
    value, legacy, diagnostic = conditioned_lower_median(
        [1.0, 2.0, 3.0], [np.asarray([0, 1]), np.asarray([1, 2])], ["1", "1", "1"], "1", ["1"],
    )
    assert value == legacy
    assert diagnostic["prediction_changed"] is False
    assert diagnostic["condition_fallback_tree_count"] == 0


def test_new_selected_counts_and_neighbors_reported():
    _, _, diagnostic = conditioned_lower_median(
        np.arange(5.0), [np.asarray([0, 1, 2]), np.asarray([2, 3, 4])], ["1", "2", "1", "2", "1"], "1", ["1", "2"],
    )
    assert diagnostic["new_minimum_selected_count"] == 2
    assert diagnostic["new_maximum_selected_count"] == 2
    assert diagnostic["new_distinct_selected_rows"] == 3
    assert diagnostic["new_effective_neighbors"] > 0


@pytest.mark.parametrize("selected,match", [
    ([], "at least one"),
    ([np.asarray([], dtype=int)], "nonempty"),
    ([np.asarray([0, 0])], "more than once"),
    ([np.asarray([0, 4])], "outside"),
    ([np.asarray([0.0, 1.0])], "integer"),
])
def test_invalid_original_selected_members_rejected(selected, match):
    with pytest.raises(ValueError, match=match):
        condition_members(selected, ["1", "2", "1", "2"], "1", ["1", "2"])


def test_nonfinite_response_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        conditioned_lower_median([1.0, np.nan], [np.asarray([0, 1])], ["1", "2"], "1", ["1", "2"])


def test_response_spout_length_mismatch_rejected():
    with pytest.raises(ValueError, match="identity"):
        conditioned_lower_median([1.0], [np.asarray([0])], ["1", "2"], "1", ["1", "2"])


def test_original_fallback_flag_length_mismatch_rejected():
    with pytest.raises(ValueError, match="flags"):
        conditioned_lower_median([1.0, 2.0], [np.asarray([0, 1])], ["1", "2"], "1", ["1", "2"], original_fallback_modes=[])
