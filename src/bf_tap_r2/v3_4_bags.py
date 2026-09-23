"""Group-safe EBM bag protocol for Round2 V3.4.

The public V3.3 wrapper called ``ExplainableBoostingRegressor.fit(X, y)``
without explicit bags.  V3.4 introduces ``group-safe-bags-v1``: inside the
current training part, a frozen duplicate-group/spout-stratified five-fold
split supplies four EBM outer bags.  Fold 0/1/2/3 are used as the internal
validation part for bags 0/1/2/3; all other folds are the training part.  The
fifth fold is therefore always in the training part and never used as an
internal validation fold in this protocol.

InterpretML 0.6.10 accepts a ``(outer_bags, n_samples)`` int8 matrix with +1
for training and -1 for internal validation.  This module creates that matrix,
hashes it, and asserts that no duplicate group is split across a bag boundary.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .splits import make_folds

BAG_PROTOCOL = "group-safe-bags-v1"
BAG_VALIDATION_FOLDS = (0, 1, 2, 3)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_group_ids(values: np.ndarray) -> np.ndarray:
    return np.asarray([str(v) for v in values], dtype=object)


def assert_group_isolated(bags: np.ndarray, groups: np.ndarray) -> None:
    """Assert every group has a single role inside every bag."""
    matrix = np.asarray(bags)
    group_values = _canonical_group_ids(np.asarray(groups))
    if matrix.ndim != 2 or matrix.shape[1] != group_values.shape[0]:
        raise ValueError("Bag matrix shape does not match group vector")
    if any(len(set(matrix[row][group_values == group])) != 1 for group in set(group_values)
           for row in range(matrix.shape[0])):
        raise ValueError("A duplicate group was split across a bag boundary")


def group_safe_inner_folds(frame: pd.DataFrame, *, n_splits: int = 5,
                           seed: int = 42) -> dict[str, Any]:
    """Return an order-aligned, duplicate-group-safe fold assignment.

    The helper delegates to :func:`bf_tap_r2.splits.make_folds`, which uses
    exact feature-vector duplicate hashes as groups and stratifies by
    ``spout_no``.  The returned fold vector is aligned to ``frame``'s current
    row order (it does not assume the frame is sorted by ``sample_id``).
    """
    if "sample_id" not in frame or "spout_no" not in frame:
        raise ValueError("Group-safe folds require sample_id and spout_no")
    assignment = make_folds(frame, seed=int(seed), n_splits=int(n_splits))
    aligned = assignment.set_index("sample_id").loc[frame["sample_id"]].reset_index()
    fold = aligned["fold"].to_numpy(dtype=int)
    group_ids = aligned["group_id"].astype(str).to_numpy(dtype=object)
    if len(fold) != len(frame) or (fold < 0).any():
        raise ValueError("Incomplete group-safe fold assignment")
    group_hash = _sha256_bytes("".join(group_ids.tolist()).encode("utf-8"))
    fold_hash = _sha256_bytes(np.ascontiguousarray(fold, dtype=np.int64).tobytes())
    return {
        "fold": fold,
        "group_id": group_ids,
        "n_splits": int(n_splits),
        "seed": int(seed),
        "group_hash": group_hash,
        "inner_fold_hash": fold_hash,
    }


def build_group_safe_bags(frame: pd.DataFrame, *, n_outer_bags: int = 4,
                          n_inner_splits: int = 5, seed: int = 42) -> dict[str, Any]:
    """Build the frozen four-bag matrix for one current training part.

    Returns a dictionary containing the int8 bag matrix, its hash, the group
    hash, the inner fold hash, and a JSON-safe metadata block.  Every bag has a
    nonempty training and validation part.  All values are +1 (training) or -1
    (internal validation); no zero padding is used.
    """
    n_bags = int(n_outer_bags)
    n_folds = int(n_inner_splits)
    if n_bags < 1:
        raise ValueError("n_outer_bags must be positive")
    if n_folds < 2:
        raise ValueError("n_inner_splits must be at least 2")
    if n_bags > n_folds:
        raise ValueError("n_outer_bags cannot exceed n_inner_splits")
    if n_bags != 4:
        raise ValueError("V3.4 group-safe-bags-v1 is frozen at four outer bags")
    if n_folds != 5:
        raise ValueError("V3.4 group-safe-bags-v1 is frozen at five inner folds")

    folded = group_safe_inner_folds(frame, n_splits=n_folds, seed=seed)
    fold = folded["fold"]
    groups = folded["group_id"]
    n_samples = len(frame)
    bags = np.full((n_bags, n_samples), 0, dtype=np.int8)
    validation_folds: list[int] = []
    for bag_index in range(n_bags):
        valid_mask = fold == bag_index
        train_mask = ~valid_mask
        if not train_mask.any():
            raise ValueError(f"Bag {bag_index} has an empty training part")
        if not valid_mask.any():
            raise ValueError(f"Bag {bag_index} has an empty validation part")
        bags[bag_index, train_mask] = 1
        bags[bag_index, valid_mask] = -1
        validation_folds.append(int(bag_index))

    if not np.isin(bags, (-1, 1)).all():
        raise ValueError("Group-safe bag matrix contains values outside -1/+1")
    assert_group_isolated(bags, groups)
    bag_hash = _sha256_bytes(np.ascontiguousarray(bags, dtype=np.int8).tobytes())
    metadata = {
        "protocol": BAG_PROTOCOL,
        "n_outer_bags": n_bags,
        "n_inner_splits": n_folds,
        "bag_seed": int(seed),
        "validation_folds": validation_folds,
        "feature_folds_not_used_as_validation": [int(v) for v in range(n_bags, n_folds)],
        "bag_hash": bag_hash,
        "group_hash": folded["group_hash"],
        "inner_fold_hash": folded["inner_fold_hash"],
        "n_samples": int(n_samples),
        "bag_train_counts": [int((bags[i] == 1).sum()) for i in range(n_bags)],
        "bag_validation_counts": [int((bags[i] == -1).sum()) for i in range(n_bags)],
    }
    return {
        "bags": bags,
        "fold": fold,
        "group_id": groups,
        "bag_hash": bag_hash,
        "group_hash": folded["group_hash"],
        "inner_fold_hash": folded["inner_fold_hash"],
        "metadata": metadata,
    }


def protocol_bag_hash(frame: pd.DataFrame, *, n_outer_bags: int = 4,
                      n_inner_splits: int = 5, seed: int = 42) -> str:
    """Return the bag hash for a protocol on a complete training table.

    The runner uses this label-free value in the trial identity.  Actual
    per-outer-fold bag hashes are recorded in ``fit_meta`` after fitting.
    """
    return str(build_group_safe_bags(frame, n_outer_bags=n_outer_bags,
                                     n_inner_splits=n_inner_splits,
                                     seed=seed)["bag_hash"])


def bag_protocol_metadata(n_outer_bags: int = 4, n_inner_splits: int = 5,
                          seed: int = 42) -> dict[str, Any]:
    """Return identity fields that are valid before seeing any model fit."""
    return {
        "protocol": BAG_PROTOCOL,
        "n_outer_bags": int(n_outer_bags),
        "n_inner_splits": int(n_inner_splits),
        "bag_seed": int(seed),
        "validation_folds": list(BAG_VALIDATION_FOLDS[: int(n_outer_bags)]),
    }
