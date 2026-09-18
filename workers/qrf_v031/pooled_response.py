"""Worker-facing wrappers around the exact v0.31 pooled response reference."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from pooled_oob_reference import (  # noqa: F401  (top-level worker path)
    PROTOCOL,
    lower_median_from_counts,
    occurrence_counts,
    pooled_lower_median,
)


def selected_member_diagnostics(
    y: Sequence[float],
    members_by_tree: Sequence[Iterable[int]],
    *,
    fallback_modes: Sequence[bool] | None = None,
    n: int | None = None,
) -> tuple[float, dict]:
    """Return the exact pooled median and per-query pooling diagnostics.

    The diagnostics describe this response estimator only.  ``S`` is the
    total number of tree x selected-member occurrences; it is not a new sample
    count or an independent-observation count.
    """
    response = np.asarray(y, dtype=np.float64)
    row_count = len(response)
    if n is None:
        n = row_count
    if int(n) != row_count:
        raise ValueError("n must match the original response vector length")
    counts = occurrence_counts(members_by_tree, row_count)
    total = int(counts.sum(dtype=np.int64))
    distinct = int(np.count_nonzero(counts))
    sizes = np.asarray([len(np.asarray(members)) for members in members_by_tree], dtype=np.int64)
    if len(sizes) == 0:
        raise ValueError("at least one tree is required")
    if sizes.min() < 1:
        raise ValueError("selected leaf members must be nonempty for every tree")
    finite = np.isfinite(response).all() and (response >= 0).all()
    value = lower_median_from_counts(response, counts) if finite else float("nan")
    new_weights = counts.astype(np.float64) / total
    if fallback_modes is None:
        fallback = np.zeros(len(sizes), dtype=bool)
    else:
        fallback = np.asarray(fallback_modes, dtype=bool)
        if fallback.shape != sizes.shape:
            raise ValueError("fallback modes must align with selected trees")
    old_weights = np.zeros(row_count, dtype=np.float64)
    tree_count = len(sizes)
    for members, size in zip(members_by_tree, sizes, strict=True):
        selected = np.asarray(members, dtype=np.int64)
        old_weights[selected] += 1.0 / (tree_count * int(size))
    diagnostics = {
        "tree_count": int(tree_count),
        "minimum_selected_count": int(sizes.min()),
        "maximum_selected_count": int(sizes.max()),
        "S_occurrence_total": total,
        "distinct_selected_rows": distinct,
        "fallback_tree_count": int(np.count_nonzero(fallback)),
        "fallback_tree_fraction": float(np.mean(fallback)),
        "fallback_mass_numerator": int(sizes[fallback].sum(dtype=np.int64)),
        "fallback_mass_fraction": float(sizes[fallback].sum(dtype=np.int64) / total),
        "weight_l1_change_from_equal_tree": float(np.abs(new_weights - old_weights).sum()),
        "effective_neighbors_new": float(1.0 / np.sum(new_weights * new_weights)),
        "effective_neighbors_old_equal_tree": float(1.0 / np.sum(old_weights * old_weights)),
        "maximum_weight_new": float(new_weights.max(initial=0.0)),
        "weight_mass_new": float(new_weights.sum()),
    }
    return value, diagnostics


__all__ = [
    "PROTOCOL",
    "occurrence_counts",
    "lower_median_from_counts",
    "pooled_lower_median",
    "selected_member_diagnostics",
]
