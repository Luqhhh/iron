"""Versioned V4.1 EBM wrapper with explicit pair and triple interactions.

V3.5/V3.6 deliberately freeze the interaction semantics to automatic counts and
explicit *pairs*.  The V4.1 strong-increment task needs a bridge that can keep a
parent model's fitted pair set and add at most two training-selected triples
without mutating the old wrappers.  This module therefore provides an
independent ``V41EBMRegressor``.  It reuses the frozen target transforms,
feature construction, group-safe bag protocol, and V3.6 effective-parameter
audit, but normalises and validates interaction specifications itself.

The wrapper never reads query labels.  ``fit`` receives a training frame only;
``predict`` receives a query frame only.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .v3_4_bags import (
    BAG_PROTOCOL,
    assert_group_isolated,
    build_group_safe_bags,
    group_safe_inner_folds,
)
from .v3_5_models import (
    EBM_KINDS,
    _feature_frame,
    _finite_1d,
    _sha256_bytes,
    _validate_custom_bags,
)
from .v3_6_models import (
    V36_EBM_ALLOWED,
    V36_EBM_EFFECTIVENESS_KEYS,
    V36_EBM_OBJECTIVES,
    V36EBMRegressor,
    _jsonable,
    _values_equal,
)
from .v3_local_search import fit_target_transform

__all__ = [
    "V41EBMRegressor",
    "extract_pair_terms_by_name",
    "extract_terms_by_name",
    "fitted_feature_names",
    "normalise_interactions",
    "resolve_interactions",
    "select_shared_edge_triples",
    "select_triples_from_parent_model",
    "structure_audit",
    "terms_as_names",
]


def _term_indices(term: Sequence[Any], feature_names: Sequence[str]) -> tuple[int, ...]:
    """Map one feature-name/index term to canonical ascending integer indices."""
    values: list[int] = []
    for value in term:
        if isinstance(value, str):
            if value not in feature_names:
                raise ValueError(f"Unknown interaction feature name: {value!r}")
            values.append(list(feature_names).index(value))
        else:
            index = int(value)
            if index < 0 or index >= len(feature_names):
                raise ValueError(f"Interaction index out of range: {value!r}")
            values.append(index)
    if len(values) not in (2, 3):
        raise ValueError(f"Only explicit pair/triple interactions are supported, got {term!r}")
    if len(set(values)) != len(values):
        raise ValueError(f"Interaction term repeats a variable: {term!r}")
    return tuple(sorted(values))


def normalise_interactions(interactions: Any) -> int | list[tuple[int, ...]]:
    """Normalise an interaction request without changing V3.5/V3.6 semantics.

    Automatic counts stay integers.  Explicit lists may contain pair or triple
    tuples and are canonicalised by sorted indices; duplicates and repeated
    variables are rejected.  Empty explicit lists are rejected rather than
    silently turning into an additive model.
    """
    if isinstance(interactions, bool):
        raise ValueError("Boolean interactions are not a valid interaction specification")
    if isinstance(interactions, (int, np.integer)):
        value = int(interactions)
        if value < 0:
            raise ValueError("Automatic interaction count must be nonnegative")
        return value
    if isinstance(interactions, (list, tuple)):
        out: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for raw in interactions:
            if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple, np.ndarray)):
                raise ValueError(f"Explicit interaction must be a pair/triple tuple, got {raw!r}")
            values = [int(v) for v in raw]
            if len(values) not in (2, 3):
                raise ValueError(f"Explicit interaction must be a pair or triple, got {raw!r}")
            if any(v < 0 for v in values) or len(set(values)) != len(values):
                raise ValueError(f"Invalid explicit interaction: {raw!r}")
            key = tuple(sorted(values))
            if key not in seen:
                seen.add(key)
                out.append(key)
        if not out:
            raise ValueError("Explicit interaction list may not be empty; use 0 for additive EBM")
        return out
    raise ValueError(f"Unsupported interaction specification: {interactions!r}")


def resolve_interactions(
    interactions: Any,
    feature_names: Sequence[str],
) -> int | list[tuple[int, ...]]:
    """Resolve a name-based interaction request against an actual feature frame.

    ``interactions`` may be an automatic count, or a sequence of terms whose
    members are either integer positions in ``feature_names`` or feature names
    present in that frame.  The returned explicit terms are sorted integer
    index tuples; automatic counts are returned unchanged.
    """
    if isinstance(interactions, (int, np.integer)) and not isinstance(interactions, bool):
        return normalise_interactions(interactions)
    if not isinstance(interactions, (list, tuple)):
        raise ValueError("Interactions must be an integer count or an explicit sequence of terms")
    resolved: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for term in interactions:
        if isinstance(term, (str, bytes)) or not isinstance(term, (list, tuple, np.ndarray)):
            raise ValueError(f"Interaction term must be a sequence, got {term!r}")
        key = _term_indices(term, feature_names)
        if key not in seen:
            seen.add(key)
            resolved.append(key)
    if not resolved:
        raise ValueError("Explicit interaction list may not be empty; use 0 for additive EBM")
    return resolved


def fitted_feature_names(model: Any) -> list[str]:
    """Return the training feature order recorded by a fitted wrapper/model."""
    for owner in (model, getattr(model, "impl", None)):
        if owner is None:
            continue
        names = getattr(owner, "input_columns_", None)
        if names is not None:
            return [str(v) for v in names]
    estimator = getattr(model, "estimator_", None)
    if estimator is not None:
        names = getattr(estimator, "feature_names_in_", None)
        if names is not None:
            return [str(v) for v in names]
        names = getattr(estimator, "feature_names_", None)
        if names is not None:
            return [str(v) for v in names]
    raise ValueError("Fitted model does not expose a feature-name order")


def _model_terms(model: Any) -> list[tuple[int, ...]]:
    estimator = getattr(model, "estimator_", None)
    if estimator is None:
        estimator = getattr(getattr(model, "impl", None), "estimator_", None)
    if estimator is None:
        raise ValueError("Fitted model does not expose an underlying EBM estimator")
    raw = getattr(estimator, "term_features_", None)
    if raw is None:
        raise ValueError("Fitted EBM does not expose term_features_")
    return [tuple(int(v) for v in term) for term in raw]


def terms_as_names(
    terms: Iterable[Sequence[int]],
    feature_names: Sequence[str],
) -> list[tuple[str, ...]]:
    """Convert canonical integer terms to feature-name terms for reporting."""
    names = list(feature_names)
    out: list[tuple[str, ...]] = []
    for term in terms:
        values = tuple(int(v) for v in term)
        if any(v < 0 or v >= len(names) for v in values) or len(set(values)) != len(values):
            raise ValueError(f"Invalid fitted term: {term!r}")
        out.append(tuple(names[v] for v in values))
    return out


def extract_terms_by_name(
    model: Any,
    feature_names: Sequence[str] | None = None,
) -> list[tuple[str, ...]]:
    """Extract all fitted EBM terms and express them as feature names."""
    names = list(feature_names) if feature_names is not None else fitted_feature_names(model)
    return terms_as_names(_model_terms(model), names)


def extract_pair_terms_by_name(
    model: Any,
    feature_names: Sequence[str] | None = None,
) -> list[tuple[str, ...]]:
    """Extract fitted pair terms as canonical feature-name pairs."""
    names = list(feature_names) if feature_names is not None else fitted_feature_names(model)
    pairs = [term for term in _model_terms(model) if len(term) == 2]
    return terms_as_names(pairs, names)


def select_shared_edge_triples(
    pair_terms: Sequence[Sequence[Any]],
    pair_importances: Sequence[float],
    *,
    feature_names: Sequence[str] | None = None,
    max_candidates: int = 8,
) -> list[tuple[int, ...]]:
    """Generate training-part triple candidates from shared-feature pair edges.

    Pair terms may be names (with ``feature_names`` supplied) or integer
    indices.  Two ordered pair terms sharing exactly one feature form the union
    triple.  Candidate triples are deduplicated canonically and ranked by the
    sum of the two parent pair importances.  Triples are *candidates only*; this
    function does not fit models or read validation labels.
    """
    if len(pair_terms) != len(pair_importances):
        raise ValueError("pair_terms and pair_importances must have equal length")
    if max_candidates <= 0:
        return []
    names = None if feature_names is None else list(feature_names)
    pairs: list[tuple[int, ...]] = []
    for term in pair_terms:
        if names is not None:
            resolved = _term_indices(term, names)
            if len(resolved) != 2:
                raise ValueError(f"Expected a valid pair term, got {term!r}")
            pairs.append(resolved)
        else:
            values = [int(v) for v in term]
            if len(values) != 2 or len(set(values)) != 2 or any(v < 0 for v in values):
                raise ValueError(f"Expected a valid pair term, got {term!r}")
            pairs.append(tuple(sorted(values)))
    imp = np.asarray(pair_importances, dtype=float)
    if not np.isfinite(imp).all():
        raise ValueError("pair_importances must be finite")
    best: dict[tuple[int, ...], float] = {}
    n_pairs = len(pairs)
    for i in range(n_pairs):
        left = pairs[i]
        for j in range(i + 1, n_pairs):
            right = pairs[j]
            shared = set(left) & set(right)
            if len(shared) != 1:
                continue
            triple = tuple(sorted(set(left) | set(right)))
            if len(triple) != 3:
                continue
            score = float(imp[i] + imp[j])
            if score > best.get(triple, -np.inf):
                best[triple] = score
    ordered = sorted(((score, term) for term, score in best.items()), key=lambda item: (-item[0], item[1]))
    return [term for _, term in ordered[: max(0, int(max_candidates))]]


def _pair_importances(model: Any) -> tuple[list[tuple[int, ...]], np.ndarray, list[str]]:
    names = fitted_feature_names(model)
    terms = _model_terms(model)
    estimator = getattr(model, "estimator_", None)
    if estimator is None:
        estimator = getattr(getattr(model, "impl", None), "estimator_", None)
    raw_importance = getattr(estimator, "term_importances", None)
    if raw_importance is None:
        raise ValueError("Fitted EBM does not expose term_importances()")
    importances = np.asarray(estimator.term_importances(), dtype=float)
    if importances.shape != (len(terms),):
        raise ValueError("Fitted EBM term/importance length mismatch")
    pair_indices = [index for index, term in enumerate(terms) if len(term) == 2]
    pairs = [terms[index] for index in pair_indices]
    return pairs, importances[pair_indices], names


def select_triples_from_parent_model(
    model: Any,
    *,
    max_candidates: int = 8,
) -> list[tuple[int, ...]]:
    """Generate canonical triple *indices* from one fitted parent EBM."""
    pairs, importances, _ = _pair_importances(model)
    return select_shared_edge_triples(pairs, importances, max_candidates=max_candidates)


def structure_audit(
    model: V36EBMRegressor,
    *,
    expected_pairs: Sequence[Sequence[int]],
    expected_triples: Sequence[Sequence[int]] | None = None,
    require_pair_preservation: bool = True,
) -> dict[str, Any]:
    """Check the fitted EBM's actual terms against the requested explicit set."""
    actual = _model_terms(model)
    actual_pairs = [term for term in actual if len(term) == 2]
    actual_triples = [term for term in actual if len(term) == 3]
    expected_pair_set = {tuple(sorted(int(v) for v in term)) for term in expected_pairs}
    actual_pair_set = {tuple(sorted(term)) for term in actual_pairs}
    expected_triple_set = ({tuple(sorted(int(v) for v in term)) for term in expected_triples}
                           if expected_triples is not None else set())
    actual_triple_set = {tuple(sorted(term)) for term in actual_triples}
    if require_pair_preservation and actual_pair_set != expected_pair_set:
        raise AssertionError(
            f"Fitted pair terms do not match the explicit control set: "
            f"expected={sorted(expected_pair_set)!r}, actual={sorted(actual_pair_set)!r}"
        )
    if expected_triples is not None:
        if actual_triple_set != expected_triple_set:
            raise AssertionError(
                f"Fitted triple terms do not match the selected set: "
                f"expected={sorted(expected_triple_set)!r}, actual={sorted(actual_triple_set)!r}"
            )
        if len(actual_triple_set) > 2:
            raise AssertionError(f"Fitted EBM produced more than two triples: {sorted(actual_triple_set)!r}")
    all_terms = [tuple(int(v) for v in term) for term in actual]
    for term in all_terms:
        if len(term) not in (1, 2, 3) or len(set(term)) != len(term):
            raise AssertionError(f"Fitted EBM produced an invalid term: {term!r}")
        if any(value < 0 for value in term):
            raise AssertionError(f"Fitted EBM produced a negative term index: {term!r}")
    return {
        "actual_pair_terms": sorted(actual_pair_set),
        "actual_triple_terms": sorted(actual_triple_set),
        "n_pairs": len(actual_pair_set),
        "n_triples": len(actual_triple_set),
        "pair_set_preserved": actual_pair_set == expected_pair_set,
        "triple_set_preserved": actual_triple_set == expected_triple_set,
    }


class V41EBMRegressor(V36EBMRegressor):
    """Independent V4.1 EBM wrapper supporting explicit pairs and triples.

    This class deliberately does **not** call ``V36EBMRegressor.__init__``: the
    V3.5 initialiser rejects triples.  It initialises the same attributes and
    then reuses the inherited V3.6 ``_params``/effective-parameter machinery, so
    the old wrappers are not broadened in place.
    """

    def __init__(self, trial: Mapping[str, Any]):
        self.trial = deepcopy(dict(trial))
        kind = str(self.trial.get("kind", "ebm"))
        if kind not in EBM_KINDS and kind != "ebm_loss":
            raise ValueError(f"Not a V4.1 EBM trial: {kind!r}")
        self.kind = kind
        self.target = str(self.trial["target"])
        self.target_transform = str(self.trial["target_transform"])
        self.feature_set = str(self.trial.get("feature_set", "raw"))
        self.parameters = dict(self.trial.get("parameters", {}))
        self.protocol = dict(self.trial.get("protocol", {}))
        self.inner_splits = int(self.protocol.get("inner_splits", 5))
        self.bag_seed = int(self.protocol.get("bag_seed", self.parameters.get("random_state", 42)))
        self._interactions = normalise_interactions(self.parameters.get("interactions", 0))
        unknown = sorted(set(self.parameters) - V36_EBM_ALLOWED)
        if unknown:
            raise ValueError(f"V4.1 EBM trial carries unsupported parameters: {unknown}")
        objective = str(self.parameters.get("objective", "rmse"))
        if objective not in V36_EBM_OBJECTIVES:
            raise ValueError(f"Unsupported V4.1 EBM objective: {objective!r}")

    def _params(self, n_outer_bags: int) -> dict[str, Any]:
        params = super()._params(n_outer_bags)
        if isinstance(params.get("interactions"), list):
            params["interactions"] = [list(term) for term in self._interactions]
        return params

    def _validate_effective_params(self) -> None:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("V4.1 EBM has not been fitted")
        actual_all = self.estimator_.get_params(deep=False)
        requested = {
            key: deepcopy(value)
            for key, value in self.parameters.items()
            if key in V36_EBM_ALLOWED
        }
        requested["interactions"] = (
            int(self._interactions) if isinstance(self._interactions, int)
            else [list(term) for term in self._interactions]
        )
        requested["outer_bags"] = int(self.estimator_.outer_bags)
        missing = sorted(set(requested) - set(actual_all))
        if missing:
            raise ValueError(f"InterpretML silently dropped V4.1 EBM parameters: {missing}")
        checks: dict[str, dict[str, Any]] = {}
        for key, requested_value in requested.items():
            actual_value = actual_all[key]
            if not _values_equal(requested_value, actual_value):
                raise ValueError(
                    f"V4.1 EBM parameter {key!r} did not take effect: "
                    f"requested={requested_value!r}, actual={actual_value!r}"
                )
            checks[key] = {
                "requested": _jsonable(requested_value),
                "actual": _jsonable(actual_value),
                "effective": True,
            }
        ignored = sorted(set(self.parameters) - set(requested))
        if ignored:
            raise ValueError(f"V4.1 EBM ignored unsupported parameters: {ignored}")
        self.requested_params_ = requested
        self.effective_params_ = {
            key: _jsonable(actual_all[key])
            for key in V36_EBM_EFFECTIVENESS_KEYS
            if key in actual_all
        }
        self.parameter_checks_ = checks

    def fit(self, frame: pd.DataFrame, target: np.ndarray,
            bags: np.ndarray | None = None) -> "V41EBMRegressor":
        from interpret.glassbox import ExplainableBoostingRegressor

        y = _finite_1d(np.asarray(target, dtype=float), "V4.1 EBM labels")
        if len(frame) != len(y):
            raise ValueError("V4.1 EBM frame/label length mismatch")
        z, self.target_state_ = fit_target_transform(y, self.target_transform)
        x = _feature_frame(frame, self.feature_set)
        if bags is None:
            bag_info = build_group_safe_bags(
                frame, n_outer_bags=4, n_inner_splits=self.inner_splits, seed=self.bag_seed
            )
            matrix = bag_info["bags"]
            self.bag_info_ = bag_info
        else:
            matrix = _validate_custom_bags(np.asarray(bags), len(frame), 4)
            folded = group_safe_inner_folds(frame, n_splits=self.inner_splits, seed=self.bag_seed)
            assert_group_isolated(matrix, folded["group_id"])
            bag_hash = _sha256_bytes(np.ascontiguousarray(matrix, dtype=np.int8).tobytes())
            self.bag_info_ = {
                "bags": matrix,
                "fold": folded["fold"],
                "group_id": folded["group_id"],
                "bag_hash": bag_hash,
                "group_hash": folded["group_hash"],
                "inner_fold_hash": folded["inner_fold_hash"],
                "metadata": {
                    "protocol": BAG_PROTOCOL,
                    "n_outer_bags": int(matrix.shape[0]),
                    "n_inner_splits": int(self.inner_splits),
                    "bag_seed": int(self.bag_seed),
                    "bag_hash": bag_hash,
                    "group_hash": folded["group_hash"],
                    "inner_fold_hash": folded["inner_fold_hash"],
                    "n_samples": int(len(frame)),
                    "bag_train_counts": [int((matrix[i] == 1).sum()) for i in range(matrix.shape[0])],
                    "bag_validation_counts": [int((matrix[i] == -1).sum()) for i in range(matrix.shape[0])],
                },
            }
        params = self._params(matrix.shape[0])
        feature_types = ["continuous"] * len(x.columns)
        spout_index = list(x.columns).index("spout_no")
        feature_types[spout_index] = "nominal"
        if isinstance(self._interactions, list):
            for term in self._interactions:
                if any(index >= len(x.columns) for index in term):
                    raise ValueError("Explicit interaction index exceeds fitted feature count")
        self.estimator_ = ExplainableBoostingRegressor(
            feature_names=list(x.columns),
            feature_types=feature_types,
            **params,
        )
        self.estimator_.fit(x, z, bags=matrix)
        self.input_columns_ = tuple(x.columns)
        self.bag_hash_ = str(self.bag_info_["bag_hash"])
        self.group_hash_ = str(self.bag_info_["group_hash"])
        self.inner_fold_hash_ = str(self.bag_info_["inner_fold_hash"])
        try:
            best_iteration = np.asarray(self.estimator_.best_iteration_, dtype=int).ravel().tolist()
        except Exception:  # pragma: no cover - library attribute guard
            best_iteration = []
        self.best_iteration_ = [int(v) for v in best_iteration]
        self._validate_effective_params()
        self.fit_meta_ = {
            "model_kind": self.kind,
            "target_transform": self.target_transform,
            "feature_set": self.feature_set,
            "interactions": _jsonable(self._interactions),
            "bag_protocol": BAG_PROTOCOL,
            "bag_hash": self.bag_hash_,
            "group_hash": self.group_hash_,
            "inner_fold_hash": self.inner_fold_hash_,
            "n_outer_bags": int(matrix.shape[0]),
            "n_inner_splits": int(self.inner_splits),
            "bag_seed": int(self.bag_seed),
            "bag_train_counts": list(self.bag_info_["metadata"]["bag_train_counts"]),
            "bag_validation_counts": list(self.bag_info_["metadata"]["bag_validation_counts"]),
            "best_iteration": list(self.best_iteration_),
            "configured_max_rounds": int(params.get("max_rounds", 0)),
            "requested_params": _jsonable(self.requested_params_),
            "effective_params": _jsonable(self.effective_params_),
            "parameter_checks": _jsonable(self.parameter_checks_),
            "ignored_params": [],
        }
        return self
