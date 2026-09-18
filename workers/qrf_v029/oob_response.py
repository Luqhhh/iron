"""Frozen-forest OOB leaf-response estimation for optimization v0.29.

The fitted forest is never changed.  For each tree/leaf we retain the original
training rows that were not present in that tree's bootstrap draw.  A leaf
with no such row falls back to all original rows in the same leaf.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from qrf_model import distribution_weights, lower_median


PROTOCOL = "QRF_FROZEN_FOREST_OOB_LEAF_RESPONSE_v029"
MODE_OOB = np.uint8(0)
MODE_FULL_SAME_LEAF_FALLBACK = np.uint8(1)


def array_identity(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": array.dtype.str, "shape": array.shape}, sort_keys=True).encode())
    digest.update(array.tobytes())
    return {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": digest.hexdigest()}


def tree_structure_identity(forest):
    digest = hashlib.sha256()
    for index, estimator in enumerate(forest.estimators_):
        tree = estimator.tree_
        digest.update(index.to_bytes(4, "little"))
        digest.update(str(estimator.random_state).encode())
        for value in (tree.children_left, tree.children_right, tree.feature, tree.threshold):
            current = np.ascontiguousarray(value)
            digest.update(current.dtype.str.encode())
            digest.update(np.asarray(current.shape, dtype=np.int64).tobytes())
            digest.update(current.tobytes())
    return digest.hexdigest()


def _validate_training(model, transformed):
    matrix = np.asarray(transformed)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("OOB derivation requires finite float32 training matrix")
    n = len(model.ids)
    if len(matrix) != n or len(model.y) != n or len(set(model.ids)) != n:
        raise ValueError("frozen model training ID/response/matrix alignment differs")
    if len(model.forest.estimators_) != len(model.leaves) or not model.leaves:
        raise ValueError("frozen forest/tree-leaf identity differs")
    return matrix, n


def derive_attachment(model, transformed):
    """Derive typed arrays and a certificate from a trusted fitted model."""
    matrix, n = _validate_training(model, transformed)
    draws = np.asarray(model.forest.estimators_samples_, dtype=np.int32)
    if draws.shape != (len(model.forest.estimators_), n):
        raise ValueError("bootstrap draw shape differs from bootstrap=True/max_samples=None")
    if (draws < 0).any() or (draws >= n).any():
        raise ValueError("bootstrap draw position outside original training rows")

    leaf_trees, leaf_nodes, offsets = [], [], [0]
    members, full_counts, oob_counts, modes = [], [], [], []
    fallback_leaves = 0
    structure_before = tree_structure_identity(model.forest)
    random_states = []
    for tree_index, (estimator, draw) in enumerate(zip(model.forest.estimators_, draws, strict=True)):
        assigned = estimator.apply(matrix)
        if assigned.shape != (n,):
            raise ValueError("training leaf assignment shape differs")
        counts = np.bincount(draw, minlength=n)
        random_states.append(int(estimator.random_state))
        stored = model.leaves[tree_index]
        if set(map(int, stored)) != set(map(int, np.unique(assigned))):
            raise ValueError("stored/full leaf node set differs")
        partition = []
        for node in sorted(stored):
            full = np.asarray(stored[node], dtype=np.int32)
            rebuilt = np.flatnonzero(assigned == int(node)).astype(np.int32)
            if not np.array_equal(full, rebuilt):
                raise ValueError("stored full-leaf mapping differs from tree.apply")
            if len(full) == 0 or len(np.unique(full)) != len(full):
                raise ValueError("invalid full same-leaf membership")
            partition.append(full)
            weighted = int(counts[full].sum())
            unique_inbag = int(np.count_nonzero(counts[full]))
            if not np.isclose(estimator.tree_.weighted_n_node_samples[int(node)], weighted, rtol=0.0, atol=1e-12):
                raise ValueError("bootstrap multiplicity differs from weighted_n_node_samples")
            if int(estimator.tree_.n_node_samples[int(node)]) != unique_inbag:
                raise ValueError("unique in-bag rows differ from n_node_samples")
            oob = full[counts[full] == 0]
            if len(oob):
                selected, mode = oob, MODE_OOB
            else:
                selected, mode = full, MODE_FULL_SAME_LEAF_FALLBACK
                fallback_leaves += 1
            leaf_trees.append(tree_index)
            leaf_nodes.append(int(node))
            full_counts.append(len(full))
            oob_counts.append(len(oob))
            modes.append(mode)
            members.append(selected)
            offsets.append(offsets[-1] + len(selected))
        projected = np.concatenate(partition)
        if len(projected) != n or not np.array_equal(np.sort(projected), np.arange(n)):
            raise ValueError("full leaf mapping does not partition every original row once")

    arrays = {
        "draws": draws,
        "leaf_trees": np.asarray(leaf_trees, dtype=np.int16),
        "leaf_nodes": np.asarray(leaf_nodes, dtype=np.int64),
        "member_offsets": np.asarray(offsets, dtype=np.int64),
        "members": np.concatenate(members).astype(np.int32, copy=False),
        "full_counts": np.asarray(full_counts, dtype=np.int32),
        "oob_counts": np.asarray(oob_counts, dtype=np.int32),
        "modes": np.asarray(modes, dtype=np.uint8),
    }
    validate_attachment(model, matrix, arrays, rederive=False)
    if structure_before != tree_structure_identity(model.forest):
        raise ValueError("frozen tree structure changed while deriving OOB responses")
    certificate = {
        "protocol": PROTOCOL,
        "training_rows": n,
        "trees": len(model.forest.estimators_),
        "draws_per_tree": n,
        "leaf_count": len(leaf_nodes),
        "fallback_leaf_count": fallback_leaves,
        "fallback_leaf_fraction": fallback_leaves / len(leaf_nodes),
        "minimum_full_count": int(np.min(arrays["full_counts"])),
        "maximum_full_count": int(np.max(arrays["full_counts"])),
        "minimum_oob_count": int(np.min(arrays["oob_counts"])),
        "maximum_oob_count": int(np.max(arrays["oob_counts"])),
        "minimum_selected_count": int(np.min(np.diff(arrays["member_offsets"]))),
        "maximum_selected_count": int(np.max(np.diff(arrays["member_offsets"]))),
        "tree_random_states": random_states,
        "tree_structure_sha256": structure_before,
        "arrays": {name: array_identity(value) for name, value in arrays.items()},
        "one_oob_member_allowed": True,
        "fallback": "FULL_SAME_LEAF_FALLBACK",
        "bootstrap_multiplicity_not_used_for_response_weight": True,
        "each_tree_total_mass_equal": True,
    }
    return arrays, certificate


def validate_attachment(model, transformed, arrays, *, rederive=True):
    matrix, n = _validate_training(model, transformed)
    required = {"draws", "leaf_trees", "leaf_nodes", "member_offsets", "members", "full_counts", "oob_counts", "modes"}
    if set(arrays) != required:
        raise ValueError("OOB attachment fields differ")
    trees = len(model.forest.estimators_)
    if arrays["draws"].shape != (trees, n):
        raise ValueError("OOB attachment bootstrap shape differs")
    count = len(arrays["leaf_nodes"])
    if any(len(arrays[name]) != count for name in ("leaf_trees", "full_counts", "oob_counts", "modes")):
        raise ValueError("OOB leaf certificate vector lengths differ")
    offsets = arrays["member_offsets"]
    if len(offsets) != count + 1 or offsets[0] != 0 or offsets[-1] != len(arrays["members"]) or (np.diff(offsets) <= 0).any():
        raise ValueError("OOB selected-member offsets invalid")
    if (arrays["members"] < 0).any() or (arrays["members"] >= n).any():
        raise ValueError("OOB selected member outside training identity")
    if not np.isin(arrays["modes"], [MODE_OOB, MODE_FULL_SAME_LEAF_FALLBACK]).all():
        raise ValueError("unknown OOB leaf selection mode")
    if rederive:
        rebuilt, _ = derive_attachment(model, matrix)
        for name in required:
            if not np.array_equal(arrays[name], rebuilt[name]):
                raise ValueError(f"persisted OOB attachment differs from recovered bootstrap: {name}")
    return True


def _member_lookup(arrays):
    lookup = {}
    offsets = arrays["member_offsets"]
    for index, (tree, node) in enumerate(zip(arrays["leaf_trees"], arrays["leaf_nodes"], strict=True)):
        key = (int(tree), int(node))
        if key in lookup:
            raise ValueError("duplicate tree/leaf in OOB attachment")
        lookup[key] = arrays["members"][offsets[index]:offsets[index + 1]]
    return lookup


def predict(model, transformed, arrays, training_months=None):
    matrix = np.asarray(transformed)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("OOB prediction requires finite float32 input")
    lookup = _member_lookup(arrays)
    routed = np.column_stack([tree.apply(matrix) for tree in model.forest.estimators_])
    medians, means, diagnostics = [], [], []
    months = None if training_months is None else np.asarray(training_months)
    if months is not None and len(months) != len(model.y):
        raise ValueError("training-month metadata differs from response identity")
    fallback_keys = {
        (int(tree), int(node))
        for tree, node, mode in zip(arrays["leaf_trees"], arrays["leaf_nodes"], arrays["modes"], strict=True)
        if int(mode) == int(MODE_FULL_SAME_LEAF_FALLBACK)
    }
    for row in routed:
        selected = [lookup[(tree, int(node))] for tree, node in enumerate(row)]
        weights = distribution_weights(selected, len(model.y))
        value = lower_median(model.y, weights, selected)
        if value < np.min(model.y) or value > np.max(model.y):
            raise ValueError("OOB lower median left original response support")
        fallback_count = sum((tree, int(node)) in fallback_keys for tree, node in enumerate(row))
        diagnostic = {
            "effective_neighbors": float(1.0 / np.sum(weights * weights)),
            "maximum_weight": float(np.max(weights)),
            "weight_mass": float(np.sum(weights)),
            "fallback_tree_count": int(fallback_count),
            "fallback_tree_fraction": float(fallback_count / len(selected)),
            "minimum_selected_count": int(min(map(len, selected))),
            "maximum_selected_count": int(max(map(len, selected))),
        }
        if months is not None:
            diagnostic["training_month_weights"] = {
                str(month): float(weights[months == month].sum()) for month in sorted(set(months))
            }
        medians.append(value)
        means.append(float(np.sum(model.y * weights)))
        diagnostics.append(diagnostic)
    return np.asarray(medians), np.asarray(means), diagnostics


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}
