from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "qrf_v015"))

from pooled_oob_reference import PROTOCOL, occurrence_counts, pooled_lower_median
from pooled_response import selected_member_diagnostics
from qrf_model import distribution_weights, lower_median


def test_equal_leaf_sizes_match_legacy_exactly():
    rng = np.random.default_rng(2026)
    for rows in (7, 12, 25):
        for leaves in (1, 3, 5):
            y = np.round(rng.normal(size=rows) * 10 + 100, 6)
            size = 4
            selected = [rng.choice(rows, size=size, replace=False).astype(np.int64) for _ in range(leaves)]
            value, counts = pooled_lower_median(y, selected)
            weights = distribution_weights(selected, rows)
            legacy = lower_median(y, weights, selected)
            assert value == legacy
            assert np.array_equal(counts, np.bincount(np.concatenate(selected), minlength=rows))


def test_two_tree_unequal_leaf_mass_is_occurrence_based():
    y = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    selected = [np.asarray([0], dtype=np.int64), np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)]
    value, counts = pooled_lower_median(y, selected)
    # Old equal-tree mass puts 1/2 on row 0; new mass puts 1/10 on row 0.
    assert int(counts.sum()) == 10
    assert int(counts[0]) == 1
    assert value == 50.0


def test_even_total_returns_smaller_of_two_central_values():
    y = np.asarray([1.0, 2.0, 3.0, 4.0])
    selected = [np.asarray([0, 1], dtype=np.int64), np.asarray([2, 3], dtype=np.int64)]
    value, counts = pooled_lower_median(y, selected)
    assert int(counts.sum()) == 4
    assert value == 2.0


def test_same_row_in_multiple_trees_counts_multiple_times_but_distinct_is_one():
    y = np.asarray([1.0, 2.0, 3.0, 4.0])
    selected = [np.asarray([0, 1], dtype=np.int64), np.asarray([0, 1], dtype=np.int64)]
    value, diagnostic = selected_member_diagnostics(y, selected)
    assert value == 1.0
    assert diagnostic["S_occurrence_total"] == 4
    assert diagnostic["distinct_selected_rows"] == 2
    assert diagnostic["fallback_mass_fraction"] == 0.0


def test_occurrence_total_excludes_bootstrap_multiplicity():
    y = np.asarray([1.0, 2.0, 3.0])
    selected = [np.asarray([1], dtype=np.int64), np.asarray([1, 2], dtype=np.int64)]
    counts = occurrence_counts(selected, len(y))
    assert counts.tolist() == [0, 2, 1]
    assert int(counts.sum()) == 3


def test_duplicate_member_within_one_tree_rejected():
    y = np.asarray([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="repeated training row"):
        occurrence_counts([np.asarray([0, 0], dtype=np.int64)], len(y))


def test_empty_or_out_of_range_members_rejected():
    y = np.asarray([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="nonempty"):
        occurrence_counts([np.asarray([], dtype=np.int64)], len(y))
    with pytest.raises(ValueError, match="outside"):
        occurrence_counts([np.asarray([0, 3], dtype=np.int64)], len(y))


def test_fallback_mass_is_leaf_occurrence_mass_over_S():
    y = np.asarray([1.0, 2.0, 3.0, 4.0])
    selected = [np.asarray([0, 1], dtype=np.int64), np.asarray([2], dtype=np.int64), np.asarray([3], dtype=np.int64)]
    value, diagnostic = selected_member_diagnostics(y, selected, fallback_modes=[False, True, True])
    assert diagnostic["S_occurrence_total"] == 4
    assert diagnostic["fallback_tree_count"] == 2
    assert diagnostic["fallback_mass_numerator"] == 2
    assert diagnostic["fallback_mass_fraction"] == pytest.approx(0.5)
    assert value == 2.0


def test_protocol_identifier_frozen():
    assert PROTOCOL == "QRF_FROZEN_FOREST_OOB_OCCURRENCE_POOLING_v031"


def test_equal_tree_mass_equivalence_exhaustive_small_synthetic_grid():
    rng = np.random.default_rng(31)
    for rows in range(2, 9):
        response = np.round(rng.normal(size=rows) * 5 + 50, 6)
        for size in range(1, rows + 1):
            for trees in range(1, 5):
                for _ in range(25):
                    selected = [
                        rng.choice(rows, size=size, replace=False).astype(np.int64)
                        for _ in range(trees)
                    ]
                    value, counts = pooled_lower_median(response, selected)
                    weights = distribution_weights(selected, rows)
                    assert value == lower_median(response, weights, selected)
                    concatenated = np.concatenate(selected)
                    assert np.array_equal(counts, np.bincount(concatenated, minlength=rows))
                    assert int(counts.sum()) == size * trees
