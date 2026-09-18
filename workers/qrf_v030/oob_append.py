"""1024-tree OOB attachment glue for the v0.30 append protocol.

The low-level derivation and prediction math stays in the v0.29 core
(``workers/qrf_v029/oob_response.py``); this module only registers the new
1024-tree protocol, slices certified prefixes, rebuilds restricted views and
synthesizes the full-leaf attachment used by the source regression check.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

ATTACHMENT_PROTOCOL = "QRF_OOB_LEAF_APPENDED_1024_v030"
CORE_PROTOCOL = "QRF_FROZEN_FOREST_OOB_LEAF_RESPONSE_v029"
REGRESSION_TREES = 256
MODE_FULL_SAME_LEAF_FALLBACK = np.uint8(1)
LEAF_VECTOR_FIELDS = ("leaf_trees", "leaf_nodes", "full_counts", "oob_counts", "modes")


def slice_prefix_arrays(arrays, trees: int) -> dict:
    """Return the certified tree-prefix slice of a leaf attachment."""
    if trees < 1 or trees > len(arrays["draws"]):
        raise ValueError("attachment prefix tree count differs")
    leaf_trees = np.asarray(arrays["leaf_trees"])
    count = int(np.count_nonzero(leaf_trees < trees))
    if count and int(np.max(leaf_trees[:count])) >= trees:
        raise ValueError("attachment leaf entries are not ordered by tree")
    offsets = np.asarray(arrays["member_offsets"])
    if len(offsets) != len(leaf_trees) + 1:
        raise ValueError("attachment offsets differ from leaf entries")
    return {
        "draws": np.asarray(arrays["draws"])[:trees],
        "leaf_trees": leaf_trees[:count].copy(),
        "leaf_nodes": np.asarray(arrays["leaf_nodes"])[:count].copy(),
        "full_counts": np.asarray(arrays["full_counts"])[:count].copy(),
        "oob_counts": np.asarray(arrays["oob_counts"])[:count].copy(),
        "modes": np.asarray(arrays["modes"])[:count].copy(),
        "member_offsets": offsets[: count + 1] - offsets[0],
        "members": np.asarray(arrays["members"])[offsets[0]:offsets[count]].copy(),
    }


def assert_prefix_equal(full_arrays, prefix_arrays, trees: int, *, label: str) -> dict:
    """Prove that the new 1024-tree attachment repeats the certified prefix."""
    sliced = slice_prefix_arrays(full_arrays, trees)
    if not np.array_equal(prefix_arrays["draws"], sliced["draws"]):
        raise ValueError(f"{label}: prefix bootstrap draws differ")
    for name in LEAF_VECTOR_FIELDS:
        if not np.array_equal(prefix_arrays[name], sliced[name]):
            raise ValueError(f"{label}: prefix field differs: {name}")
    if not np.array_equal(prefix_arrays["member_offsets"], sliced["member_offsets"]):
        raise ValueError(f"{label}: prefix member offsets differ")
    if not np.array_equal(prefix_arrays["members"], sliced["members"]):
        raise ValueError(f"{label}: prefix leaf members differ")
    return {
        "prefix_trees": trees,
        "prefix_leaf_count": len(sliced["leaf_nodes"]),
        "prefix_member_count": len(sliced["members"]),
        "prefix_leaf_members_exact": True,
    }


class RestrictedView:
    """Expose the certified first-``trees`` estimators as a frozen model view."""

    def __init__(self, model, trees: int):
        total = len(model.forest.estimators_)
        if trees != REGRESSION_TREES or trees > total:
            raise ValueError("restricted view requires the registered 256-tree prefix")
        self.forest = SimpleNamespace(estimators_=model.forest.estimators_[:trees])
        self.y = model.y
        self.ids = model.ids
        self.leaves = model.leaves[:trees]
        self.training_months = getattr(model, "training_months", None)


def full_leaf_arrays(model, *, trees: int | None = None) -> dict:
    """Synthesize a full-leaf attachment that reproduces the source QRF output."""
    views = model.leaves if trees is None else model.leaves[:trees]
    leaf_trees, leaf_nodes, offsets, members, counts, modes = [], [], [0], [], [], []
    for tree_index, mapping in enumerate(views):
        for node in sorted(mapping):
            full = np.asarray(mapping[node], dtype=np.int32)
            if not len(full) or len(np.unique(full)) != len(full):
                raise ValueError("invalid full same-leaf membership")
            leaf_trees.append(tree_index)
            leaf_nodes.append(int(node))
            full_counts_val = len(full)
            members.append(full)
            counts.append(full_counts_val)
            modes.append(MODE_FULL_SAME_LEAF_FALLBACK)
            offsets.append(offsets[-1] + full_counts_val)
    return {
        "leaf_trees": np.asarray(leaf_trees, dtype=np.int16),
        "leaf_nodes": np.asarray(leaf_nodes, dtype=np.int64),
        "member_offsets": np.asarray(offsets, dtype=np.int64),
        "members": np.concatenate(members).astype(np.int32, copy=False),
        "full_counts": np.asarray(counts, dtype=np.int32),
        "oob_counts": np.zeros(len(counts), dtype=np.int32),
        "modes": np.asarray(modes, dtype=np.uint8),
    }


def require_1024_identity(model) -> dict:
    """Refuse old 256-tree loaders before any 1024-tree derivation."""
    trees = len(model.forest.estimators_)
    if trees != 1024 or len(model.leaves) != 1024:
        raise ValueError("v0.30 attachment derivation requires exactly 1024 appended trees")
    if getattr(model, "protocol", None) != "QRF_TIME_WARM_START_APPEND_1024_v030":
        raise ValueError("v0.30 appends must retain the registered warm-start protocol")
    if getattr(model, "appended_new_trees", None) != 768 or getattr(model, "warm_start_registered", None) is not True:
        raise ValueError("v0.30 appends must register 768 new warm-start trees")
    if getattr(model, "prefix_verified", None) is not True:
        raise ValueError("v0.30 appends require a verified 256-tree prefix")
    return {"trees": trees, "new_trees": 768, "protocol": model.protocol}


__all__ = [
    "ATTACHMENT_PROTOCOL", "CORE_PROTOCOL", "REGRESSION_TREES",
    "MODE_FULL_SAME_LEAF_FALLBACK", "LEAF_VECTOR_FIELDS",
    "slice_prefix_arrays", "assert_prefix_equal", "RestrictedView",
    "full_leaf_arrays", "require_1024_identity",
]
