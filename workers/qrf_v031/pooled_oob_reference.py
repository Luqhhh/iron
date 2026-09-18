"""Exact occurrence-pooled OOB response reference for optimization v0.31.

The registered change is deliberately small and entirely in the response
aggregation: every occurrence of one original training row in one selected
per-tree leaf contributes one equal unit of mass.  Thus, with selected member
counts ``n_b`` and total occurrence mass ``S = sum_b n_b``, the response weight
of original training row ``i`` is ``c_i / S`` where ``c_i`` counts its
occurrences over the queried trees.  No bootstrap multiplicity and no
cross-tree identity de-duplication enter the weight.

This module is intentionally pure: it does not import a fitted model, a
preprocessor, the old equal-tree ``lower_median`` implementation, or any
scoring code.  The worker uses the integer cumulative-count definition below,
not the legacy Fraction fallback, to certify the new rule.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


PROTOCOL = "QRF_FROZEN_FOREST_OOB_OCCURRENCE_POOLING_v031"


def _as_members(members, n: int, label: str) -> np.ndarray:
    array = np.asarray(members)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{label} must be a nonempty one-dimensional member vector")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(array == np.asarray(array, dtype=np.int64)):
            raise ValueError(f"{label} member positions must be integer valued")
        array = np.asarray(array, dtype=np.int64)
    else:
        array = np.asarray(array, dtype=np.int64)
    if (array < 0).any() or (array >= n).any():
        raise ValueError(f"{label} member position outside original training identity")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"{label} contains a repeated training row")
    return array


def occurrence_counts(members_by_tree: Sequence[Iterable[int]], n: int) -> np.ndarray:
    """Count each tree x selected-member occurrence exactly once.

    ``members_by_tree`` holds exactly one selected member vector for each fixed
    forest tree, in tree order.  A training position occurring in several trees
    is counted several times; a position repeated inside one selected leaf is
    rejected (the frozen v0.29 attachment already certifies leaf uniqueness).
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError("original training row count must be a positive integer")
    n = int(n)
    counts = np.zeros(n, dtype=np.int64)
    tree_count = 0
    for tree, members in enumerate(members_by_tree):
        selected = _as_members(members, n, f"tree {tree} selected members")
        counts[selected] += 1
        tree_count += 1
    if tree_count == 0:
        raise ValueError("at least one queried tree is required")
    if int(counts.sum()) != sum(len(np.asarray(members)) for members in members_by_tree):
        raise ValueError("occurrence count total differs from selected member total")
    return counts


def lower_median_from_counts(y: Sequence[float], counts: np.ndarray) -> float:
    """Exact smaller weighted median using integer occurrence counts.

    The response is sorted stably.  The first response whose cumulative integer
    occurrence count reaches ``ceil(S / 2)`` is returned.  For even ``S`` this
    is the smaller of the two central response values, never their average.
    """
    response = np.asarray(y, dtype=np.float64)
    occurrence = np.asarray(counts)
    if response.ndim != 1 or occurrence.ndim != 1 or len(response) != len(occurrence):
        raise ValueError("response and occurrence vectors must be aligned one-dimensional arrays")
    if not np.isfinite(response).all() or (response < 0).any():
        raise ValueError("original responses must be finite nonnegative")
    if not np.issubdtype(occurrence.dtype, np.integer):
        raise ValueError("occurrence counts must be integer typed")
    if (occurrence < 0).any():
        raise ValueError("occurrence counts must be nonnegative")
    total = int(occurrence.sum(dtype=np.int64))
    if total <= 0:
        raise ValueError("at least one selected occurrence is required")
    order = np.argsort(response, kind="stable")
    cumulative = np.cumsum(occurrence[order], dtype=np.int64)
    threshold = (total + 1) // 2
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    if index < 0 or index >= len(response) or int(cumulative[index]) < threshold:
        raise ValueError("occurrence cumulative count did not reach the lower-median threshold")
    value = float(response[order[index]])
    if value < float(np.min(response)) or value > float(np.max(response)):
        raise ValueError("pooled lower median left original response support")
    return value


def pooled_lower_median(y: Sequence[float], members_by_tree: Sequence[Iterable[int]]):
    """Return ``(value, counts)`` for one query using the v0.31 rule."""
    response = np.asarray(y, dtype=np.float64)
    counts = occurrence_counts(members_by_tree, len(response))
    return lower_median_from_counts(response, counts), counts


__all__ = [
    "PROTOCOL",
    "occurrence_counts",
    "lower_median_from_counts",
    "pooled_lower_median",
]
