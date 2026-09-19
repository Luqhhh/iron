"""Same-spout conditioning inside certified v0.29 selected members.

This module never changes a tree, bootstrap draw, response, or selected-member
attachment.  For a known query spout, a tree uses the same-spout intersection
when nonempty and otherwise keeps the original selected set byte-for-byte.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from qrf_model import distribution_weights, lower_median


PROTOCOL = "QRF_FROZEN_FOREST_SAME_SPOUT_OOB_RESPONSE_v032"
QUERY_KNOWN = np.uint8(0)
QUERY_MISSING = np.uint8(1)
QUERY_UNKNOWN = np.uint8(2)


def exact_tokens(values: Sequence[object]) -> np.ndarray:
    """Apply the frozen preprocessor's exact ``str`` tokenization only."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("spout metadata must be one-dimensional")
    return np.asarray([str(value) for value in array], dtype=str)


def validate_training_spout(
    training_spout: Sequence[object],
    training_ids: Sequence[object],
    model_ids: Sequence[object],
    preprocessor,
) -> np.ndarray:
    """Prove raw spout metadata has the frozen model/preprocessor row identity."""
    tokens = exact_tokens(training_spout)
    ids = [str(value) for value in training_ids]
    frozen_ids = [str(value) for value in model_ids]
    preprocessor_ids = [str(value) for value in preprocessor.training_ids]
    if len(tokens) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("training spout/ID alignment differs")
    if ids != frozen_ids or ids != preprocessor_ids:
        raise ValueError("training spout IDs differ from frozen model/preprocessor order")
    expected_vocabulary = sorted({token for token in tokens if token})
    actual_vocabulary = [str(value) for value in preprocessor.vocabulary]
    if expected_vocabulary != actual_vocabulary:
        raise ValueError("training spout tokens differ from frozen preprocessor vocabulary")
    transformed_spout_columns = [
        column for column in preprocessor.transformed_columns if column.startswith("spout_no:")
    ]
    expected_columns = [
        *[f"spout_no:known:{token}" for token in actual_vocabulary],
        "spout_no:missing",
        "spout_no:unknown",
    ]
    if transformed_spout_columns != expected_columns:
        raise ValueError("frozen transformed spout schema differs from vocabulary")
    return tokens


def query_state(value: object, vocabulary: Sequence[object]) -> tuple[str, np.uint8]:
    token = str(value)
    frozen = {str(item) for item in vocabulary}
    if not token:
        return token, QUERY_MISSING
    if token not in frozen:
        return token, QUERY_UNKNOWN
    return token, QUERY_KNOWN


def _validated_members(selected: Sequence[np.ndarray], training_rows: int) -> list[np.ndarray]:
    if not selected:
        raise ValueError("at least one selected-member tree is required")
    result = []
    for members in selected:
        current = np.asarray(members)
        if current.ndim != 1 or current.dtype.kind not in "iu" or len(current) == 0:
            raise ValueError("each original selected-member set must be nonempty integer positions")
        current = current.astype(np.int64, copy=False)
        if (current < 0).any() or (current >= training_rows).any():
            raise ValueError("selected member outside frozen training identity")
        if len(np.unique(current)) != len(current):
            raise ValueError("same tree selected a training row more than once")
        result.append(current)
    return result


def condition_members(
    selected: Sequence[np.ndarray],
    training_spout: Sequence[object],
    query_spout: object,
    vocabulary: Sequence[object],
) -> tuple[list[np.ndarray], dict]:
    """Condition each original set, retaining it exactly on an empty intersection."""
    tokens = exact_tokens(training_spout)
    original = _validated_members(selected, len(tokens))
    token, state = query_state(query_spout, vocabulary)
    if state != QUERY_KNOWN:
        return original, {
            "query_token": token,
            "query_state": int(state),
            "same_spout_nonempty_tree_count": 0,
            "condition_fallback_tree_count": 0,
            "conditioning_bypassed": True,
        }

    conditioned = []
    same_nonempty = 0
    empty_intersection = 0
    for members in original:
        same = members[tokens[members] == token]
        if len(same):
            conditioned.append(same)
            same_nonempty += 1
        else:
            conditioned.append(members)
            empty_intersection += 1
    for before, after in zip(original, conditioned, strict=True):
        if len(after) == 0 or not np.isin(after, before).all():
            raise ValueError("conditioned members must be a nonempty subset of the original set")
    return conditioned, {
        "query_token": token,
        "query_state": int(state),
        "same_spout_nonempty_tree_count": same_nonempty,
        "condition_fallback_tree_count": empty_intersection,
        "conditioning_bypassed": False,
    }


def conditioned_lower_median(
    response: Sequence[float],
    selected: Sequence[np.ndarray],
    training_spout: Sequence[object],
    query_spout: object,
    vocabulary: Sequence[object],
    *,
    original_fallback_modes: Sequence[bool] | None = None,
) -> tuple[float, float, dict]:
    """Return conditioned and legacy medians plus exact equal-tree diagnostics."""
    y = np.asarray(response, dtype=np.float64)
    tokens = exact_tokens(training_spout)
    if y.ndim != 1 or len(y) != len(tokens) or not np.isfinite(y).all():
        raise ValueError("frozen response/spout identity differs or response is non-finite")
    original = _validated_members(selected, len(y))
    conditioned, metadata = condition_members(original, tokens, query_spout, vocabulary)
    old_weights = distribution_weights(original, len(y))
    new_weights = distribution_weights(conditioned, len(y))
    if not np.isclose(old_weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("legacy equal-tree response mass differs from one")
    if not np.isclose(new_weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("conditioned equal-tree response mass differs from one")
    legacy = float(lower_median(y, old_weights, original))
    value = float(lower_median(y, new_weights, conditioned))

    fallback = [False] * len(original) if original_fallback_modes is None else list(original_fallback_modes)
    if len(fallback) != len(original):
        raise ValueError("original OOB fallback flags differ from tree count")
    token = metadata["query_token"]
    known = metadata["query_state"] == int(QUERY_KNOWN)
    if known:
        cross = tokens != token
        old_cross = float(old_weights[cross].sum())
        new_cross = float(new_weights[cross].sum())
        if new_cross > old_cross + 1e-12:
            raise ValueError("same-spout conditioning increased cross-spout response mass")
    else:
        old_cross = np.nan
        new_cross = np.nan
        if any(not np.array_equal(before, after) for before, after in zip(original, conditioned, strict=True)):
            raise ValueError("missing/unknown query spout did not bypass conditioning")

    diagnostic = {
        **metadata,
        "tree_count": len(original),
        "original_oob_fallback_tree_count": int(sum(bool(value) for value in fallback)),
        "original_cross_spout_mass": old_cross,
        "new_cross_spout_mass": new_cross,
        "original_minimum_selected_count": int(min(map(len, original))),
        "original_maximum_selected_count": int(max(map(len, original))),
        "new_minimum_selected_count": int(min(map(len, conditioned))),
        "new_maximum_selected_count": int(max(map(len, conditioned))),
        "original_distinct_selected_rows": int(np.count_nonzero(old_weights)),
        "new_distinct_selected_rows": int(np.count_nonzero(new_weights)),
        "original_effective_neighbors": float(1.0 / np.sum(old_weights * old_weights)),
        "new_effective_neighbors": float(1.0 / np.sum(new_weights * new_weights)),
        "original_maximum_weight": float(np.max(old_weights)),
        "new_maximum_weight": float(np.max(new_weights)),
        "prediction_delta": value - legacy,
        "prediction_changed": bool(value != legacy),
    }
    return value, legacy, diagnostic


__all__ = [
    "PROTOCOL", "QUERY_KNOWN", "QUERY_MISSING", "QUERY_UNKNOWN", "exact_tokens",
    "validate_training_spout", "query_state", "condition_members", "conditioned_lower_median",
]
