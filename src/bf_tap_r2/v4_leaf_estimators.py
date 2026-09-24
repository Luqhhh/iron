"""V4 leaf-estimation and partition mechanisms (F line).

The module deliberately separates three things that are easy to conflate:

* the partition learned by a forest (which rows are neighbours);
* the response estimate inside each partition (mean, median, or a
  per-tree empirical distribution quantile);
* honest sample splitting for the response estimate.

Nothing here reads an outer-validation label.  All response statistics are
estimated from the training rows passed to :meth:`fit`.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.tree import DecisionTreeRegressor


TREE_CRITERIA = frozenset({"squared_error", "absolute_error", "friedman_mse", "poisson"})
LEAF_METHODS = frozenset({"mean", "distribution_median", "median_of_medians"})
LIGHTGBM_L1_OBJECTIVES = frozenset({"regression_l1", "mae", "mean_absolute_error"})


def _as_2d_float(x: Any, name: str = "X") -> np.ndarray:
    array = np.asarray(x, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty 2-D array")
    return array


def _as_1d_float(y: Any, name: str = "y") -> np.ndarray:
    array = np.asarray(y, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty 1-D array")
    return array


def _validate_lengths(x: np.ndarray, y: np.ndarray) -> None:
    if len(x) != len(y):
        raise ValueError("X and y must contain the same number of rows")


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = _as_1d_float(values, "values")
    weights = _as_1d_float(weights, "weights")
    if len(values) != len(weights):
        raise ValueError("values and weights must align")
    if (weights <= 0).any():
        raise ValueError("weights must be positive")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative = cumulative / cumulative[-1]
    return float(np.interp(float(q), cumulative, sorted_values))


def _sha256_groups(groups: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for value in groups:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


@dataclass
class LeafResponseIndex:
    """Training response values routed through one fitted tree."""

    leaf_ids: dict[int, np.ndarray]
    fallback: float

    @classmethod
    def from_tree(cls, tree: DecisionTreeRegressor, x: np.ndarray, y: np.ndarray) -> "LeafResponseIndex":
        routed = np.asarray(tree.apply(x), dtype=int)
        leaves: dict[int, np.ndarray] = {}
        for leaf in np.unique(routed):
            leaves[int(leaf)] = np.asarray(y[routed == leaf], dtype=float)
        if not leaves:
            raise ValueError("Tree routed no training rows to a leaf")
        return cls(leaf_ids=leaves, fallback=float(np.median(y)))


class LeafPartitionForest:
    """F1--F4: a forest with several response estimators over one partition.

    Parameters
    ----------
    criterion:
        Split criterion.  ``squared_error`` and ``absolute_error`` are the
        pre-registered F4 contrast.  The criterion changes the partition, not
        the response summary.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 32,
        max_depth: int | None = 5,
        min_samples_leaf: int = 5,
        criterion: str = "squared_error",
        max_features: float | int | str | None = 1.0,
        random_state: int = 42,
    ) -> None:
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be positive")
        if str(criterion) not in TREE_CRITERIA:
            raise ValueError(f"Unsupported split criterion: {criterion!r}")
        if max_depth is not None and int(max_depth) < 1:
            raise ValueError("max_depth must be positive or None")
        if int(min_samples_leaf) < 1:
            raise ValueError("min_samples_leaf must be positive")
        self.n_estimators = int(n_estimators)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.criterion = str(criterion)
        self.max_features = max_features
        self.random_state = int(random_state)

    def fit(self, x: Any, y: Any) -> "LeafPartitionForest":
        x_array = _as_2d_float(x)
        y_array = _as_1d_float(y)
        _validate_lengths(x_array, y_array)
        self.estimators_: list[DecisionTreeRegressor] = []
        self.response_index_: list[LeafResponseIndex] = []
        self.n_features_in_ = int(x_array.shape[1])
        for index in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state + index,
            )
            tree.fit(x_array, y_array)
            self.estimators_.append(tree)
            self.response_index_.append(LeafResponseIndex.from_tree(tree, x_array, y_array))
        self.fit_meta_ = {
            "mechanism": "v4_leaf_partition_forest",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "criterion": self.criterion,
            "response_methods": sorted(LEAF_METHODS),
            "n_training_rows": int(len(x_array)),
            "n_features": int(x_array.shape[1]),
        }
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("LeafPartitionForest is not fitted")

    def _leaf_arrays(self, x: np.ndarray) -> list[np.ndarray]:
        routed = [np.asarray(tree.apply(x), dtype=int) for tree in self.estimators_]
        return routed

    def predict(self, x: Any, method: str = "mean") -> np.ndarray:
        """Predict with one of the pre-registered leaf response rules."""
        self._check_fitted()
        if method not in LEAF_METHODS:
            raise ValueError(f"Unknown leaf response method: {method!r}")
        x_array = _as_2d_float(x, "X")
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count than fit")
        routed = self._leaf_arrays(x_array)

        if method == "mean":
            predictions = np.mean(
                np.vstack([tree.predict(x_array) for tree in self.estimators_]), axis=0
            )
            return np.asarray(predictions, dtype=float)

        if method == "median_of_medians":
            per_tree = np.empty((len(self.estimators_), len(x_array)), dtype=float)
            for tree_index, leaf_ids in enumerate(routed):
                index = self.response_index_[tree_index]
                per_tree[tree_index] = [
                    float(np.median(index.leaf_ids.get(int(leaf_id), np.asarray([index.fallback]))))
                    for leaf_id in leaf_ids
                ]
            return np.mean(per_tree, axis=0)

        # F2: each tree contributes its leaf's empirical response distribution
        # with equal total tree weight.  Leaf size is not a between-tree weight.
        out = np.empty(len(x_array), dtype=float)
        tree_weight = 1.0 / len(self.estimators_)
        for row_index, leaf_ids in enumerate(zip(*routed)):
            values: list[float] = []
            weights: list[float] = []
            for tree_index, leaf_id in enumerate(leaf_ids):
                responses = self.response_index_[tree_index].leaf_ids.get(int(leaf_id))
                if responses is None or len(responses) == 0:
                    responses = np.asarray([self.response_index_[tree_index].fallback], dtype=float)
                values.extend(float(v) for v in responses)
                weights.extend([tree_weight / len(responses)] * len(responses))
            out[row_index] = _weighted_quantile(np.asarray(values, dtype=float), np.asarray(weights, dtype=float), 0.5)
        return out


class HonestLeafForest:
    """F5: honest partition/response split at duplicate-group level.

    In each repeat, groups are split into a structure part (used to grow the
    tree) and a disjoint response part (used to populate leaf responses).  The
    response part is never used to choose a split.  Multiple repeats average
    their leaf estimates.  Empty response leaves fall back to the response-part
    global median (or mean).
    """

    def __init__(
        self,
        *,
        n_estimators: int = 16,
        max_depth: int | None = 4,
        min_samples_leaf: int = 3,
        criterion: str = "squared_error",
        random_state: int = 42,
        n_repeats: int = 4,
        structure_fraction: float = 0.5,
    ) -> None:
        if int(n_estimators) < 1 or int(n_repeats) < 1:
            raise ValueError("n_estimators and n_repeats must be positive")
        if not 0.0 < float(structure_fraction) < 1.0:
            raise ValueError("structure_fraction must be strictly between 0 and 1")
        if str(criterion) not in TREE_CRITERIA:
            raise ValueError(f"Unsupported split criterion: {criterion!r}")
        self.n_estimators = int(n_estimators)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.criterion = str(criterion)
        self.random_state = int(random_state)
        self.n_repeats = int(n_repeats)
        self.structure_fraction = float(structure_fraction)

    def fit(self, x: Any, y: Any, groups: Iterable[Any]) -> "HonestLeafForest":
        x_array = _as_2d_float(x)
        y_array = _as_1d_float(y)
        group_array = np.asarray(list(groups))
        if len(x_array) != len(y_array) or len(group_array) != len(x_array):
            raise ValueError("X, y and groups must have the same length")
        unique_groups = np.asarray(pd_unique(group_array), dtype=object)
        if len(unique_groups) < 2:
            raise ValueError("Honest splitting requires at least two groups")
        self.estimators_ = []
        self.response_parts_ = []
        self.structure_groups_: list[tuple[Any, ...]] = []
        self.response_groups_: list[tuple[Any, ...]] = []
        self.fallback_counts_: list[int] = []

        for repeat in range(self.n_repeats):
            rng = np.random.default_rng(self.random_state + 1009 * repeat)
            permuted = unique_groups[rng.permutation(len(unique_groups))]
            n_structure = min(
                max(1, int(round(len(unique_groups) * self.structure_fraction))),
                len(unique_groups) - 1,
            )
            structure_groups = tuple(permuted[:n_structure].tolist())
            response_groups = tuple(permuted[n_structure:].tolist())
            structure_mask = np.isin(group_array, np.asarray(structure_groups, dtype=object))
            response_mask = ~structure_mask
            if not structure_mask.any() or not response_mask.any():
                raise ValueError("Honest split produced an empty structure or response part")
            structure_x, structure_y = x_array[structure_mask], y_array[structure_mask]
            response_y = y_array[response_mask]
            for tree_index in range(self.n_estimators):
                tree = DecisionTreeRegressor(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state + 1009 * repeat + tree_index,
                )
                tree.fit(structure_x, structure_y)
                routed = np.asarray(tree.apply(x_array[response_mask]), dtype=int)
                leaf_responses: dict[int, np.ndarray] = {}
                for leaf in np.unique(routed):
                    leaf_responses[int(leaf)] = np.asarray(response_y[routed == leaf], dtype=float)
                self.estimators_.append(tree)
                self.response_parts_.append(
                    {
                        "leaf_responses": leaf_responses,
                        "fallback_median": float(np.median(response_y)),
                        "fallback_mean": float(np.mean(response_y)),
                    }
                )
                self.fallback_counts_.append(0)
            self.structure_groups_.append(structure_groups)
            self.response_groups_.append(response_groups)

        self.n_features_in_ = int(x_array.shape[1])
        self.fit_meta_ = {
            "mechanism": "v4_honest_leaf_forest",
            "n_estimators": self.n_estimators,
            "n_repeats": self.n_repeats,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "structure_fraction": self.structure_fraction,
            "structure_group_hashes": [_sha256_groups(values) for values in self.structure_groups_],
            "response_group_hashes": [_sha256_groups(values) for values in self.response_groups_],
            "group_isolation_checked": True,
            "total_trees": int(len(self.estimators_)),
            "n_training_rows": int(len(x_array)),
            "n_groups": int(len(unique_groups)),
        }
        return self

    @property
    def empty_leaf_fallback_count_(self) -> int:
        return int(sum(self.fallback_counts_))

    def predict(self, x: Any, method: str = "median") -> np.ndarray:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("HonestLeafForest is not fitted")
        if method not in {"median", "mean"}:
            raise ValueError("HonestLeafForest supports method='median' or 'mean'")
        x_array = _as_2d_float(x, "X")
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count than fit")
        per_tree = np.empty((len(self.estimators_), len(x_array)), dtype=float)
        fallback_key = "fallback_median" if method == "median" else "fallback_mean"
        for tree_index, (tree, response) in enumerate(zip(self.estimators_, self.response_parts_)):
            routed = np.asarray(tree.apply(x_array), dtype=int)
            leaf_responses = response["leaf_responses"]
            values = []
            for leaf in routed:
                estimate = leaf_responses.get(int(leaf))
                if estimate is None or len(estimate) == 0:
                    self.fallback_counts_[tree_index] += 1
                    estimate = np.asarray([response[fallback_key]], dtype=float)
                values.append(float(np.median(estimate) if method == "median" else np.mean(estimate)))
            per_tree[tree_index] = np.asarray(values, dtype=float)
        return np.mean(per_tree, axis=0)


def pd_unique(values: np.ndarray) -> list[Any]:
    """Small unique helper that preserves first-seen order and treats NaN safely."""
    seen: set[Any] = set()
    out: list[Any] = []
    for value in values.tolist():
        key = ("__nan__",) if isinstance(value, float) and np.isnan(value) else value
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def fit_lightgbm_linear_tree(
    x: Any,
    y: Any,
    *,
    objective: str = "regression",
    n_estimators: int = 20,
    num_leaves: int = 7,
    learning_rate: float = 0.1,
    min_child_samples: int = 5,
    random_state: int = 42,
    **params: Any,
) -> Any:
    """F6 capability-checked LightGBM linear-leaf model.

    ``regression_l1`` and its aliases are rejected because the locked LightGBM
    documentation states that linear trees are not supported for L1.  The
    returned model records the requested and effective ``linear_tree`` flag.
    """
    objective = str(objective)
    if objective in LIGHTGBM_L1_OBJECTIVES:
        raise ValueError("LightGBM linear_tree is not supported for regression_l1/MAE")
    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("LightGBM is unavailable") from exc

    merged = dict(params)
    merged.update(
        objective=objective,
        n_estimators=int(n_estimators),
        num_leaves=int(num_leaves),
        learning_rate=float(learning_rate),
        min_child_samples=int(min_child_samples),
        random_state=int(random_state),
        linear_tree=True,
    )
    model = LGBMRegressor(**merged)
    model.fit(_as_2d_float(x), _as_1d_float(y))
    effective = model.get_params()
    if effective.get("linear_tree") is not True:
        raise RuntimeError("LightGBM did not retain linear_tree=True")
    model.linear_tree_effective_ = {
        "requested": True,
        "actual": bool(effective.get("linear_tree")),
        "objective": str(effective.get("objective")),
        "num_leaves": int(effective.get("num_leaves")),
    }
    return model


def assert_lightgbm_linear_tree_supported(objective: str) -> None:
    """Raise for the documented unsupported L1 + linear-tree combination."""
    if str(objective) in LIGHTGBM_L1_OBJECTIVES:
        raise ValueError("regression_l1/MAE is not supported with linear_tree")


__all__ = [
    "HonestLeafForest",
    "LEAF_METHODS",
    "LIGHTGBM_L1_OBJECTIVES",
    "LeafPartitionForest",
    "assert_lightgbm_linear_tree_supported",
    "fit_lightgbm_linear_tree",
]
