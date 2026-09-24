"""V4 smooth-basis mechanisms (S line).

The S line asks whether explicit low-order continuous bases explain the current
finite data better than axis-aligned trees.  The implementation keeps the
degrees of freedom deliberately small:

* S1 is an additive univariate spline model;
* S2 adds a small number of explicitly declared two-way tensor products;
* S3 adds at most a handful of three-way tensor products to S2;
* S4 checks and uses InterpretML's explicit high-order tuple terms.

The module never accepts outer-validation residuals for term selection.  The
only selection helper uses an internal K-fold on the training matrix supplied
by the caller.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer, StandardScaler


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


def _validate_terms(terms: Iterable[Sequence[int]], n_features: int, arity: int, name: str) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for raw_term in terms:
        term = tuple(int(value) for value in raw_term)
        if len(term) != arity:
            raise ValueError(f"{name} must contain only {arity}-tuples, got {term!r}")
        if len(set(term)) != arity:
            raise ValueError(f"{name} term has repeated feature index: {term!r}")
        if min(term) < 0 or max(term) >= n_features:
            raise ValueError(f"{name} term index out of range: {term!r}")
        canonical = tuple(sorted(term))
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    difference = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(difference * difference)))


class _TensorSplineFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build additive, pair-tensor and triple-tensor spline bases."""

    def __init__(
        self,
        *,
        n_knots: int = 4,
        degree: int = 3,
        include_bias: bool = False,
        standardize: bool = True,
        pairs: Sequence[Sequence[int]] = (),
        triples: Sequence[Sequence[int]] = (),
    ) -> None:
        if int(n_knots) < 2:
            raise ValueError("n_knots must be at least 2")
        if int(degree) < 1:
            raise ValueError("degree must be positive")
        self.n_knots = int(n_knots)
        self.degree = int(degree)
        self.include_bias = bool(include_bias)
        self.standardize = bool(standardize)
        self.pairs = tuple(tuple(int(v) for v in pair) for pair in pairs)
        self.triples = tuple(tuple(int(v) for v in triple) for triple in triples)

    def fit(self, x: Any, y: Any = None) -> "_TensorSplineFeatureBuilder":
        x_array = _as_2d_float(x)
        self.n_features_in_ = int(x_array.shape[1])
        self.pairs_ = _validate_terms(self.pairs, self.n_features_in_, 2, "pairs")
        self.triples_ = _validate_terms(self.triples, self.n_features_in_, 3, "triples")
        self.scaler_ = StandardScaler() if self.standardize else None
        scaled = self.scaler_.fit_transform(x_array) if self.scaler_ is not None else x_array
        self.spline_transformers_ = []
        for feature in range(self.n_features_in_):
            transformer = SplineTransformer(
                n_knots=self.n_knots,
                degree=self.degree,
                include_bias=self.include_bias,
                extrapolation="constant",
            )
            transformer.fit(scaled[:, [feature]])
            self.spline_transformers_.append(transformer)
        self.feature_names_ = self._feature_names()
        return self

    def transform(self, x: Any) -> np.ndarray:
        if not hasattr(self, "spline_transformers_"):
            raise RuntimeError("_TensorSplineFeatureBuilder is not fitted")
        x_array = _as_2d_float(x, "X")
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count than fit")
        scaled = self.scaler_.transform(x_array) if self.scaler_ is not None else x_array
        bases = [
            np.asarray(transformer.transform(scaled[:, [feature]]), dtype=float)
            for feature, transformer in enumerate(self.spline_transformers_)
        ]
        blocks: list[np.ndarray] = [*bases]
        for left, right in self.pairs_:
            blocks.append(np.einsum("ij,ik->ijk", bases[left], bases[right]).reshape(len(x_array), -1))
        for first, second, third in self.triples_:
            blocks.append(
                np.einsum("ij,ik,il->ijkl", bases[first], bases[second], bases[third]).reshape(len(x_array), -1)
            )
        return np.hstack(blocks)

    def _feature_names(self) -> list[str]:
        names = [f"add_{feature}" for feature in range(self.n_features_in_)]
        names.extend(f"pair_{left}_{right}" for left, right in self.pairs_)
        names.extend(f"triple_{first}_{second}_{third}" for first, second, third in self.triples_)
        return names

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("_TensorSplineFeatureBuilder is not fitted")
        return np.asarray(self.feature_names_, dtype=object)


class LinearControlRegressor:
    """S1/P1 control: the same input features with only a linear map."""

    def __init__(self, *, alpha: float = 1.0, standardize: bool = True, fit_intercept: bool = True) -> None:
        if float(alpha) < 0:
            raise ValueError("alpha must be nonnegative")
        self.alpha = float(alpha)
        self.standardize = bool(standardize)
        self.fit_intercept = bool(fit_intercept)

    def fit(self, x: Any, y: Any) -> "LinearControlRegressor":
        x_array = _as_2d_float(x)
        y_array = _as_1d_float(y)
        if len(x_array) != len(y_array):
            raise ValueError("X and y must have the same number of rows")
        self.scaler_ = StandardScaler() if self.standardize else None
        design = self.scaler_.fit_transform(x_array) if self.scaler_ is not None else x_array
        self.model_ = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept).fit(design, y_array)
        self.n_features_in_ = int(x_array.shape[1])
        self.fit_meta_ = {
            "mechanism": "v4_linear_control",
            "alpha": self.alpha,
            "standardize": self.standardize,
            "n_features": self.n_features_in_,
            "selection_data": "training_rows_only",
        }
        return self

    def predict(self, x: Any) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("LinearControlRegressor is not fitted")
        x_array = _as_2d_float(x, "X")
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count than fit")
        design = self.scaler_.transform(x_array) if self.scaler_ is not None else x_array
        prediction = np.asarray(self.model_.predict(design), dtype=float)
        if prediction.shape != (len(x_array),) or not np.isfinite(prediction).all():
            raise ValueError("Invalid linear-control prediction")
        return prediction


class SmoothAdditiveRegressor:
    """S1/S2/S3 estimator: low-degree spline basis followed by ridge."""

    def __init__(
        self,
        *,
        n_knots: int = 4,
        degree: int = 3,
        alpha: float = 1.0,
        standardize: bool = True,
        pairs: Sequence[Sequence[int]] = (),
        triples: Sequence[Sequence[int]] = (),
        fit_intercept: bool = True,
    ) -> None:
        self.n_knots = int(n_knots)
        self.degree = int(degree)
        self.alpha = float(alpha)
        self.standardize = bool(standardize)
        self.pairs = tuple(tuple(int(v) for v in pair) for pair in pairs)
        self.triples = tuple(tuple(int(v) for v in triple) for triple in triples)
        self.fit_intercept = bool(fit_intercept)
        if self.alpha < 0:
            raise ValueError("alpha must be nonnegative")

    def fit(self, x: Any, y: Any) -> "SmoothAdditiveRegressor":
        x_array = _as_2d_float(x)
        y_array = _as_1d_float(y)
        if len(x_array) != len(y_array):
            raise ValueError("X and y must have the same number of rows")
        self.builder_ = _TensorSplineFeatureBuilder(
            n_knots=self.n_knots,
            degree=self.degree,
            include_bias=False,
            standardize=self.standardize,
            pairs=self.pairs,
            triples=self.triples,
        ).fit(x_array)
        basis = self.builder_.transform(x_array)
        self.model_ = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        self.model_.fit(basis, y_array)
        self.n_features_in_ = int(x_array.shape[1])
        self.fit_meta_ = {
            "mechanism": "v4_smooth_tensor_spline",
            "n_knots": self.n_knots,
            "degree": self.degree,
            "alpha": self.alpha,
            "standardize": self.standardize,
            "pairs": list(self.builder_.pairs_),
            "triples": list(self.builder_.triples_),
            "n_basis_columns": int(basis.shape[1]),
            "spline_degree_univariate": True,
            "selection_data": "training_rows_only",
        }
        return self

    def predict(self, x: Any) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("SmoothAdditiveRegressor is not fitted")
        x_array = _as_2d_float(x, "X")
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count than fit")
        prediction = np.asarray(self.model_.predict(self.builder_.transform(x_array)), dtype=float)
        if prediction.shape != (len(x_array),) or not np.isfinite(prediction).all():
            raise ValueError("Invalid smooth-model prediction")
        return prediction


class PairwiseTensorProductRegressor(SmoothAdditiveRegressor):
    """S2: additive splines plus explicitly declared two-way tensor products."""

    def __init__(self, *, pairs: Sequence[Sequence[int]], **kwargs: Any) -> None:
        if not pairs:
            raise ValueError("PairwiseTensorProductRegressor requires at least one pair")
        super().__init__(pairs=pairs, **kwargs)


class TripleTensorProductRegressor(SmoothAdditiveRegressor):
    """S3: additive + pair terms plus at most a few three-way tensor terms."""

    def __init__(
        self,
        *,
        triples: Sequence[Sequence[int]],
        pairs: Sequence[Sequence[int]] = (),
        **kwargs: Any,
    ) -> None:
        if not triples:
            raise ValueError("TripleTensorProductRegressor requires at least one triple")
        super().__init__(pairs=pairs, triples=triples, **kwargs)


def select_pair_terms_internal_validation(
    x: Any,
    y: Any,
    candidates: Sequence[Sequence[int]],
    *,
    n_splits: int = 3,
    alpha: float = 1.0,
    n_knots: int = 3,
    degree: int = 2,
    random_state: int = 42,
) -> list[tuple[int, ...]]:
    """Rank candidate pairs by an internal K-fold RMSE on training rows only."""
    x_array = _as_2d_float(x)
    y_array = _as_1d_float(y)
    _validate_terms(candidates, x_array.shape[1], 2, "candidates")
    splitter = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
    scored: list[tuple[float, tuple[int, ...]]] = []
    for pair in candidates:
        predicted = np.full(len(y_array), np.nan, dtype=float)
        for train_index, valid_index in splitter.split(x_array):
            model = PairwiseTensorProductRegressor(
                pairs=[pair], n_knots=n_knots, degree=degree, alpha=alpha
            ).fit(x_array[train_index], y_array[train_index])
            predicted[valid_index] = model.predict(x_array[valid_index])
        if not np.isfinite(predicted).all():
            raise RuntimeError("Internal-validation selection left an uncovered row")
        scored.append((_rmse(y_array, predicted), tuple(int(v) for v in pair)))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [pair for _, pair in scored]


def assert_spline_transformer_univariate(transformer: SplineTransformer, n_features: int) -> None:
    """Assert that a ``SplineTransformer`` produced one univariate basis per feature.

    A degree-3 spline is not a three-feature interaction.  The assertion is
    intentionally structural: the number of fitted B-spline blocks must equal
    the number of input columns.
    """
    if not hasattr(transformer, "bsplines_"):
        raise RuntimeError("SplineTransformer has not been fitted")
    if len(transformer.bsplines_) != int(n_features):
        raise AssertionError(
            f"Expected {n_features} univariate spline bases, got {len(transformer.bsplines_)}"
        )


def assert_ebm_term_arity(model: Any, arity: int) -> tuple[int, ...]:
    """Return an actual fitted InterpretML term with the requested arity."""
    if not hasattr(model, "term_features_"):
        raise RuntimeError("EBM has not been fitted")
    wanted = int(arity)
    matches = [tuple(int(v) for v in term) for term in model.term_features_ if len(term) == wanted]
    if not matches:
        raise AssertionError(f"Fitted EBM has no term of arity {wanted}")
    return matches[0]


def assert_ebm_has_no_terms_above(model: Any, max_arity: int = 2) -> None:
    """Assert an EBM is additive/pairwise-only, as V3.5 advertises."""
    if not hasattr(model, "term_features_"):
        raise RuntimeError("EBM has not been fitted")
    too_high = [term for term in model.term_features_ if len(term) > int(max_arity)]
    if too_high:
        raise AssertionError(f"Expected no terms above arity {max_arity}, found {too_high!r}")


def build_explicit_tuple_interactions(tuples: Iterable[Sequence[int]]) -> list[tuple[int, ...]]:
    """Normalise explicit InterpretML tuple interactions without promoting arity."""
    out: list[tuple[int, ...]] = []
    for raw in tuples:
        term = tuple(int(v) for v in raw)
        if len(term) < 3:
            raise ValueError(f"Explicit high-order tuple must have at least 3 entries, got {term!r}")
        if len(set(term)) != len(term):
            raise ValueError(f"Explicit tuple has repeated feature indices: {term!r}")
        canonical = tuple(sorted(term))
        if canonical not in out:
            out.append(canonical)
    if not out:
        raise ValueError("At least one explicit high-order tuple is required")
    return out


def fit_ebm_with_explicit_triple(
    x: Any,
    y: Any,
    triple: Sequence[int],
    *,
    parameters: Mapping[str, Any] | None = None,
) -> Any:
    """Fit InterpretML with an explicit three-feature tuple and assert it exists."""
    from interpret.glassbox import ExplainableBoostingRegressor

    x_array = _as_2d_float(x)
    y_array = _as_1d_float(y)
    term = _validate_terms([triple], x_array.shape[1], 3, "triple")[0]
    params = dict(parameters or {})
    params.update(interactions=[term], max_interaction_bins=int(params.get("max_interaction_bins", 8)))
    model = ExplainableBoostingRegressor(**params)
    model.fit(x_array, y_array)
    found = assert_ebm_term_arity(model, 3)
    model.explicit_triple_term_ = found
    return model


__all__ = [
    "LinearControlRegressor",
    "PairwiseTensorProductRegressor",
    "SmoothAdditiveRegressor",
    "TripleTensorProductRegressor",
    "_TensorSplineFeatureBuilder",
    "assert_ebm_has_no_terms_above",
    "assert_ebm_term_arity",
    "assert_spline_transformer_univariate",
    "build_explicit_tuple_interactions",
    "fit_ebm_with_explicit_triple",
    "select_pair_terms_internal_validation",
]
