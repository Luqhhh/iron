"""Versioned OOB response attachment supporting training rows N != draws m."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from qrf_model import distribution_weights, lower_median
from sampled_forest import expected_draws


PROTOCOL = "QRF_SAMPLED_FOREST_OOB_LEAF_RESPONSE_v033"
MODE_OOB = np.uint8(0)
MODE_FULL_SAME_LEAF_FALLBACK = np.uint8(1)
ARRAY_FIELDS = {
    "draws", "leaf_trees", "leaf_nodes", "full_offsets", "full_members",
    "oob_offsets", "oob_members", "selected_offsets", "selected_members",
    "full_counts", "oob_counts", "modes",
}


def array_identity(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": array.dtype.str, "shape": array.shape}, sort_keys=True).encode())
    digest.update(array.tobytes())
    return {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": digest.hexdigest()}


def _combined_identity(arrays, names):
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update(array_identity(arrays[name])["sha256"].encode())
    return digest.hexdigest()


def tree_structure_identity(forest):
    digest = hashlib.sha256()
    for index, estimator in enumerate(forest.estimators_):
        tree = estimator.tree_
        digest.update(index.to_bytes(4, "little"))
        digest.update(str(estimator.random_state).encode())
        for value in (
            tree.children_left, tree.children_right, tree.feature, tree.threshold,
            tree.impurity, tree.n_node_samples, tree.weighted_n_node_samples,
        ):
            current = np.ascontiguousarray(value)
            digest.update(current.dtype.str.encode())
            digest.update(np.asarray(current.shape, dtype=np.int64).tobytes())
            digest.update(current.tobytes())
    return digest.hexdigest()


def _training(model, transformed):
    matrix = np.asarray(transformed)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("v0.33 OOB derivation requires finite float32 input")
    n = len(model.ids)
    if len(matrix) != n or len(model.y) != n or len(set(model.ids)) != n:
        raise ValueError("v0.33 training ID/response/matrix identity differs")
    if len(model.forest.estimators_) != 256 or len(model.leaves) != 256:
        raise ValueError("v0.33 forest/leaf identity differs")
    return matrix, n


def derive_attachment(model, transformed):
    matrix, n = _training(model, transformed)
    m = expected_draws(model.candidate, n)
    draws = np.asarray(model.forest.estimators_samples_, dtype=np.int32)
    if draws.shape != (256, m) or (draws < 0).any() or (draws >= n).any():
        raise ValueError("actual bootstrap draws do not match registered (trees,m) identity")

    trees, nodes = [], []
    full_offsets, oob_offsets, selected_offsets = [0], [0], [0]
    full_members, oob_members, selected_members = [], [], []
    full_counts, oob_counts, modes = [], [], []
    fallback = 0
    structure_before = tree_structure_identity(model.forest)
    unique_inbag_global, oob_global = [], []
    for tree_index, (estimator, draw) in enumerate(zip(model.forest.estimators_, draws, strict=True)):
        assigned = estimator.apply(matrix)
        counts = np.bincount(draw, minlength=n)
        unique_inbag_global.append(int(np.count_nonzero(counts)))
        oob_global.append(int(np.count_nonzero(counts == 0)))
        if int(counts.sum()) != m:
            raise ValueError("bootstrap multiplicity sum differs from m")
        stored = model.leaves[tree_index]
        if set(map(int, stored)) != set(map(int, np.unique(assigned))):
            raise ValueError("stored full leaf nodes differ from tree projection")
        partition = []
        for node in sorted(stored):
            full = np.asarray(stored[node], dtype=np.int32)
            rebuilt = np.flatnonzero(assigned == int(node)).astype(np.int32)
            if not np.array_equal(full, rebuilt) or not len(full) or len(np.unique(full)) != len(full):
                raise ValueError("stored full leaf membership differs")
            partition.append(full)
            weighted = int(counts[full].sum())
            unique = int(np.count_nonzero(counts[full]))
            if not np.isclose(estimator.tree_.weighted_n_node_samples[int(node)], weighted, rtol=0, atol=1e-12):
                raise ValueError("leaf draw multiplicity differs from weighted_n_node_samples")
            if int(estimator.tree_.n_node_samples[int(node)]) != unique:
                raise ValueError("leaf unique in-bag rows differ from n_node_samples")
            oob = full[counts[full] == 0]
            if len(oob):
                selected, mode = oob, MODE_OOB
            else:
                selected, mode = full, MODE_FULL_SAME_LEAF_FALLBACK
                fallback += 1
            trees.append(tree_index); nodes.append(int(node))
            full_counts.append(len(full)); oob_counts.append(len(oob)); modes.append(mode)
            full_members.append(full); oob_members.append(oob); selected_members.append(selected)
            full_offsets.append(full_offsets[-1] + len(full))
            oob_offsets.append(oob_offsets[-1] + len(oob))
            selected_offsets.append(selected_offsets[-1] + len(selected))
        projected = np.concatenate(partition)
        if len(projected) != n or not np.array_equal(np.sort(projected), np.arange(n)):
            raise ValueError("full leaf mapping must cover all N unique rows once")

    arrays = {
        "draws": draws,
        "leaf_trees": np.asarray(trees, dtype=np.int16),
        "leaf_nodes": np.asarray(nodes, dtype=np.int64),
        "full_offsets": np.asarray(full_offsets, dtype=np.int64),
        "full_members": np.concatenate(full_members).astype(np.int32, copy=False),
        "oob_offsets": np.asarray(oob_offsets, dtype=np.int64),
        "oob_members": np.concatenate(oob_members).astype(np.int32, copy=False),
        "selected_offsets": np.asarray(selected_offsets, dtype=np.int64),
        "selected_members": np.concatenate(selected_members).astype(np.int32, copy=False),
        "full_counts": np.asarray(full_counts, dtype=np.int32),
        "oob_counts": np.asarray(oob_counts, dtype=np.int32),
        "modes": np.asarray(modes, dtype=np.uint8),
    }
    validate_attachment(model, matrix, arrays, rederive=False)
    structure_after = tree_structure_identity(model.forest)
    if structure_before != structure_after:
        raise ValueError("tree structure changed during attachment derivation")
    selected_sizes = np.diff(arrays["selected_offsets"])
    certificate = {
        "protocol": PROTOCOL,
        "candidate": model.candidate,
        "training_rows": n,
        "draws_per_tree": m,
        "trees": 256,
        "leaf_count": len(nodes),
        "fallback_leaf_count": fallback,
        "fallback_leaf_fraction": fallback / len(nodes),
        "minimum_full_count": int(np.min(arrays["full_counts"])),
        "maximum_full_count": int(np.max(arrays["full_counts"])),
        "minimum_oob_count": int(np.min(arrays["oob_counts"])),
        "maximum_oob_count": int(np.max(arrays["oob_counts"])),
        "minimum_selected_count": int(np.min(selected_sizes)),
        "maximum_selected_count": int(np.max(selected_sizes)),
        "unique_inbag_global": {"minimum": min(unique_inbag_global), "maximum": max(unique_inbag_global), "mean": float(np.mean(unique_inbag_global))},
        "global_oob": {"minimum": min(oob_global), "maximum": max(oob_global), "mean": float(np.mean(oob_global))},
        "draw_sha256": array_identity(draws)["sha256"],
        "tree_sha256": structure_before,
        "full_mapping_sha256": _combined_identity(arrays, ("leaf_trees", "leaf_nodes", "full_offsets", "full_members")),
        "oob_mapping_sha256": _combined_identity(arrays, ("leaf_trees", "leaf_nodes", "oob_offsets", "oob_members")),
        "selected_mapping_sha256": _combined_identity(arrays, ("leaf_trees", "leaf_nodes", "selected_offsets", "selected_members", "modes")),
        "arrays": {name: array_identity(value) for name, value in arrays.items()},
        "all_N_rows_projected_per_tree": True,
        "bootstrap_multiplicity_not_used_for_response_weight": True,
        "one_oob_member_allowed": True,
        "empty_oob_fallback": "FULL_SAME_LEAF",
        "each_tree_total_mass_equal": True,
    }
    return arrays, certificate


def validate_attachment(model, transformed, arrays, *, rederive=True):
    matrix, n = _training(model, transformed)
    m = expected_draws(model.candidate, n)
    if set(arrays) != ARRAY_FIELDS or arrays["draws"].shape != (256, m):
        raise ValueError("v0.33 attachment fields or N/m draw shape differs")
    count = len(arrays["leaf_nodes"])
    if any(len(arrays[name]) != count for name in ("leaf_trees", "full_counts", "oob_counts", "modes")):
        raise ValueError("v0.33 leaf certificate vector lengths differ")
    for prefix in ("full", "oob", "selected"):
        offsets, members = arrays[f"{prefix}_offsets"], arrays[f"{prefix}_members"]
        if len(offsets) != count + 1 or offsets[0] != 0 or offsets[-1] != len(members) or (np.diff(offsets) < 0).any():
            raise ValueError(f"invalid {prefix} mapping offsets")
        if (members < 0).any() or (members >= n).any():
            raise ValueError(f"{prefix} member outside N training rows")
    if (np.diff(arrays["selected_offsets"]) <= 0).any():
        raise ValueError("selected sets must never be empty")
    if not np.isin(arrays["modes"], [MODE_OOB, MODE_FULL_SAME_LEAF_FALLBACK]).all():
        raise ValueError("unknown selected member mode")
    if rederive:
        rebuilt, _ = derive_attachment(model, matrix)
        for name in ARRAY_FIELDS:
            if not np.array_equal(arrays[name], rebuilt[name]):
                raise ValueError(f"attachment differs from actual model bootstrap: {name}")
    return True


def _lookup(arrays):
    result = {}
    offsets = arrays["selected_offsets"]
    for index, (tree, node) in enumerate(zip(arrays["leaf_trees"], arrays["leaf_nodes"], strict=True)):
        key = (int(tree), int(node))
        if key in result:
            raise ValueError("duplicate tree/leaf key")
        result[key] = arrays["selected_members"][offsets[index]:offsets[index + 1]]
    return result


def predict(model, transformed, arrays):
    matrix = np.asarray(transformed)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("v0.33 prediction requires finite float32 input")
    lookup = _lookup(arrays)
    routed = np.column_stack([tree.apply(matrix) for tree in model.forest.estimators_])
    fallback_keys = {
        (int(tree), int(node))
        for tree, node, mode in zip(arrays["leaf_trees"], arrays["leaf_nodes"], arrays["modes"], strict=True)
        if int(mode) == int(MODE_FULL_SAME_LEAF_FALLBACK)
    }
    medians, means, diagnostics = [], [], []
    for row in routed:
        selected = [lookup[(tree, int(node))] for tree, node in enumerate(row)]
        weights = distribution_weights(selected, len(model.y))
        median = lower_median(model.y, weights, selected)
        fallback_count = sum((tree, int(node)) in fallback_keys for tree, node in enumerate(row))
        medians.append(median)
        means.append(float(np.sum(model.y * weights)))
        diagnostics.append({
            "effective_neighbors": float(1.0 / np.sum(weights * weights)),
            "maximum_weight": float(np.max(weights)),
            "fallback_tree_count": int(fallback_count),
            "fallback_tree_fraction": float(fallback_count / 256),
            "minimum_selected_count": int(min(map(len, selected))),
            "maximum_selected_count": int(max(map(len, selected))),
            "weight_mass": float(np.sum(weights)),
        })
    return np.asarray(medians), np.asarray(means), diagnostics


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}
