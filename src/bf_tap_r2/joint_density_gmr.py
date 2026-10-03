"""Auditable joint-density Gaussian mixture regression; no query-label API.

Each native EM start is a separate sklearn 1.8.0 object.  The selected start
maximizes final fit-only likelihood, not a held-out loss or the pre-M-step
``lower_bound_``.  Prediction is the conditional mixture median, not sklearn's
component classifier.  Numeric exports also retain every unsuccessful-to-win
start so a different process can audit selection without unpickling a model.
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import sklearn
from scipy.optimize import brentq
from scipy.special import log_ndtr, logsumexp
from sklearn.mixture import GaussianMixture


INPUT_DIM = 23
STARTS = 5
REG_COVAR = .01
MAX_ITER = 2000
EM_TOL = 1e-5
RANDOM_STATE = 42
RAW_MEDIAN_TOL = 1e-9
SKLEARN_VERSION = "1.8.0"


def _array_hash(array):
    array = np.ascontiguousarray(array)
    identity = f"{array.dtype.str}:{array.shape}:".encode("ascii")
    return hashlib.sha256(identity + array.tobytes()).hexdigest()


def _features(x, dimension, *, allow_empty=True, check_spout=False):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != dimension:
        raise ValueError(f"Expected an (n, {dimension}) feature-only matrix")
    if not np.isfinite(x).all() or (not allow_empty and len(x) == 0):
        raise ValueError("Features must be nonempty where required and finite")
    if check_spout and (not np.isin(x[:, -2:], [0., 1.]).all()
                        or not np.all(x[:, -2:].sum(axis=1) == 1.)):
        raise ValueError("Last two columns must encode exactly spout 1 or 2")
    return x


def _normalization(x, y):
    x_mean, x_scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    x_scale = np.where(x_scale == 0., 1., x_scale)
    y_mean, y_scale = float(y.mean()), float(y.std(ddof=0))
    if y_scale == 0.:
        y_scale = 1.
    if (not np.isfinite(x_mean).all() or not np.isfinite(x_scale).all()
            or not math.isfinite(y_mean) or not math.isfinite(y_scale)):
        raise ValueError("Fit-only normalization overflowed")
    return x_mean, x_scale, y_mean, y_scale


class TrackedGaussianMixture(GaussianMixture):
    """Only observe native initialization/EM; do not change native arithmetic."""

    def __init__(self, *, n_components, random_state, on_event=None,
                 on_iteration=None, start_index=0):
        super().__init__(n_components=n_components, covariance_type="full",
                         reg_covar=REG_COVAR, max_iter=MAX_ITER, tol=EM_TOL,
                         n_init=1, init_params="kmeans", random_state=random_state,
                         warm_start=False)
        self.on_event = on_event
        self.on_iteration = on_iteration
        self.start_index = start_index
        self.observed_iterations = 0
        self.observed_lower_bounds = []
        self.kmeans_started = 0
        self.kmeans_completed = 0

    def _event(self, name, **extra):
        if self.on_event is not None:
            self.on_event(dict(event=name, start_index=self.start_index,
                               random_state=self.random_state, **extra))

    def _initialize_parameters(self, X, random_state, xp=None):
        self.kmeans_started += 1
        self._event("native_kmeans_started", fit_rows=len(X))
        result = super()._initialize_parameters(X, random_state, xp=xp)
        self.kmeans_completed += 1
        self._event("native_kmeans_completed", fit_rows=len(X))
        return result

    def _e_step(self, X, xp=None):
        result = super()._e_step(X, xp=xp)
        self._pre_m_lower_bound = float(result[0])
        return result

    def _m_step(self, X, log_resp, xp=None):
        result = super()._m_step(X, log_resp, xp=xp)
        self.observed_iterations += 1
        lower_bound = self._pre_m_lower_bound
        if not math.isfinite(lower_bound):
            raise ValueError("Nonfinite native EM likelihood")
        self.observed_lower_bounds.append(lower_bound)
        if self.on_iteration is not None:
            self.on_iteration(dict(start_index=self.start_index,
                                   random_state=self.random_state,
                                   native_fit_index=self.start_index + 1,
                                   iteration=self.observed_iterations,
                                   fit_rows=len(X),
                                   pre_m_lower_bound=lower_bound))
        return result


class JointDensityGMR:
    """Frozen K=1 control / K=8 candidate with five audited native starts."""

    def __init__(self, k):
        if not isinstance(k, int) or isinstance(k, bool) or k not in (1, 8):
            raise ValueError("Only frozen K=1 and K=8 recipes are permitted")
        self.k = k
        self._attempted = False
        self._state = None

    def fit(self, x, y, on_event=None, on_iteration=None):
        if self._attempted:
            raise ValueError("A GMR object cannot be fitted or retried twice")
        self._attempted = True
        self._metadata = dict(k=self.k, n_components=self.k, fit_rows=0, native_fit_count=0,
                              native_kmeans_count=0,
                              native_kmeans_started=0,
                              native_kmeans_completed=0, selected_start=None,
                              status="validating", starts=[],
                              sklearn_version=sklearn.__version__,
                              params=dict(covariance_type="full", reg_covar=REG_COVAR,
                                          n_init=STARTS, native_n_init=1,
                                          max_iter=MAX_ITER, tol=EM_TOL,
                                          init_params="kmeans", warm_start=False,
                                          random_states=list(range(42, 47)),
                                          selection="maximum_final_fit_log_likelihood;earliest_tie",
                                          prediction="conditional_mixture_median"))
        if sklearn.__version__ != SKLEARN_VERSION:
            raise RuntimeError("Native EM observation requires locked sklearn 1.8.0")
        x = _features(x, INPUT_DIM, allow_empty=False, check_spout=True)
        y = np.asarray(y, dtype=np.float64)
        if (y.shape != (len(x),) or not np.isfinite(y).all()
                or np.any(y < 0.) or len(x) < max(2, self.k)):
            raise ValueError("Fit target must be aligned finite nonnegative time")
        x_mean, x_scale, y_mean, y_scale = _normalization(x, y)
        joint = np.column_stack(((x - x_mean)/x_scale, (y - y_mean)/y_scale))
        if not np.isfinite(joint).all():
            raise ValueError("Normalized joint fit matrix is nonfinite")
        self._metadata.update(fit_rows=len(x), status="fitting",
                              normalized_joint_hash=_array_hash(joint))
        starts = []

        def emit(event):
            name = event["event"]
            if name == "native_kmeans_started":
                self._metadata["native_kmeans_started"] += 1
            if name == "native_kmeans_completed":
                self._metadata["native_kmeans_completed"] += 1
                self._metadata["native_kmeans_count"] += 1
            if on_event is not None:
                on_event(dict(event))

        for index in range(STARTS):
            record = dict(start_index=index, random_state=42 + index,
                          native_fit_index=index + 1, fit_rows=len(x), status="started")
            self._metadata["starts"].append(record)
            self._metadata["native_fit_count"] += 1
            model = TrackedGaussianMixture(n_components=self.k, random_state=42 + index,
                                           on_event=emit, on_iteration=on_iteration,
                                           start_index=index)
            try:
                emit(dict(event="native_fit_started", **record))
                model.fit(joint)
                iterations = int(model.n_iter_)
                history = np.asarray(model.lower_bounds_, dtype=np.float64)
                if (not 1 <= iterations <= MAX_ITER or len(history) != iterations
                        or model.observed_iterations != iterations
                        or not np.array_equal(history, model.observed_lower_bounds)
                        or model.kmeans_started != 1 or model.kmeans_completed != 1):
                    raise ValueError("Native initialization/iteration accounting does not close")
                score = float(model.score(joint))
                state = dict(weights=np.asarray(model.weights_, float).copy(),
                             means=np.asarray(model.means_, float).copy(),
                             covariances=np.asarray(model.covariances_, float).copy())
                _validate_components(state, INPUT_DIM)
                lower_bound = float(model.lower_bound_)
                if not math.isfinite(score) or not math.isfinite(lower_bound):
                    raise ValueError("Final fit log-likelihood is nonfinite")
                record.update(status="completed", n_iter=iterations,
                              lower_bound=lower_bound, final_fit_score=score,
                              converged=bool(model.converged_),
                              native_kmeans_count=1,
                              em_lower_bounds=history.tolist(),
                              hashes={name: _array_hash(value) for name, value in state.items()})
                starts.append((state, history, score, lower_bound, iterations,
                               bool(model.converged_)))
                emit(dict(event="native_fit_completed", **record))
                if not model.converged_:
                    raise ValueError(f"Native start {index} did not converge; no retry or fallback")
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__,
                              error=str(error))
                self._metadata["status"] = "failed"
                emit(dict(event="native_fit_failed", **record))
                raise

        scores = np.array([start[2] for start in starts])
        selected = int(np.argmax(scores))  # np.argmax selects the first exact tie.
        padded_history = np.zeros((STARTS, MAX_ITER), dtype=np.float64)
        for index, start in enumerate(starts):
            padded_history[index, :len(start[1])] = start[1]
        self._state = dict(x_mean=x_mean.copy(), x_scale=x_scale.copy(),
                           y_mean=np.array(y_mean), y_scale=np.array(y_scale),
                           weights=np.stack([s[0]["weights"] for s in starts]),
                           means=np.stack([s[0]["means"] for s in starts]),
                           covariances=np.stack([s[0]["covariances"] for s in starts]),
                           em_lower_bounds=padded_history,
                           final_fit_scores=scores,
                           lower_bounds=np.array([s[3] for s in starts]),
                           n_iters=np.array([s[4] for s in starts], dtype=np.int64),
                           converged=np.array([s[5] for s in starts], dtype=np.bool_),
                           random_states=np.arange(42, 47, dtype=np.int64),
                           selected_start=np.array(selected, dtype=np.int64),
                           fit_rows=np.array(len(x), dtype=np.int64),
                           native_fit_count=np.array(STARTS, dtype=np.int64),
                           native_kmeans_count=np.array(STARTS, dtype=np.int64),
                           reg_covar=np.array(REG_COVAR),
                           max_iter=np.array(MAX_ITER, dtype=np.int64), tol=np.array(EM_TOL))
        self._metadata.update(status="fitted", selected_start=selected,
                              state_hashes={n: _array_hash(v) for n, v in self._state.items()})
        return self

    def metadata(self):
        if not self._attempted:
            raise ValueError("No fit has been attempted")
        return copy.deepcopy(self._metadata)

    def export_state(self):
        if self._state is None:
            raise ValueError("No complete successfully fitted GMR state")
        return {name: value.copy() for name, value in self._state.items()}

    def predict(self, x):
        if self._state is None:
            raise ValueError("GMR has not completed its five native fits")
        return _predict_native(self._state, x)


def _component_view(state, start_index=None):
    """Resolve canonical five-start exports or single-component test fixtures."""
    weights = np.asarray(state["weights"])
    if weights.ndim == 1:
        if start_index is not None:
            raise ValueError("Single-start arrays have no start index")
        return {name: state[name] for name in ("weights", "means", "covariances")}
    if weights.ndim != 2 or len(weights) != STARTS:
        raise ValueError("Expected five retained native starts")
    if start_index is None:
        value = np.asarray(state["selected_start"])
        if value.shape != () or value.dtype.kind not in "iu":
            raise ValueError("Selected start must be an integer scalar")
        start_index = int(value)
    if (not isinstance(start_index, int) or isinstance(start_index, bool)
            or not 0 <= start_index < STARTS):
        raise ValueError("Invalid native start index")
    return {name: np.asarray(state[name])[start_index]
            for name in ("weights", "means", "covariances")}


def _validate_components(state, dimension):
    state = _component_view(state)
    weights = np.asarray(state["weights"], dtype=np.float64)
    means = np.asarray(state["means"], dtype=np.float64)
    covariances = np.asarray(state["covariances"], dtype=np.float64)
    k = len(weights) if weights.ndim == 1 else 0
    if (k == 0 or means.shape != (k, dimension + 1)
            or covariances.shape != (k, dimension + 1, dimension + 1)
            or not np.isfinite(weights).all() or not np.isfinite(means).all()
            or not np.isfinite(covariances).all() or np.any(weights <= 0.)
            or abs(float(weights.sum()) - 1.) > 1e-10):
        raise ValueError("Invalid joint Gaussian mixture arrays")
    for covariance in covariances:
        if not np.allclose(covariance, covariance.T, rtol=0., atol=1e-12):
            raise ValueError("Joint covariance is not symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("Joint covariance is not positive definite") from error
    return weights, means, covariances


def _validate_state(state):
    x_mean = np.asarray(state["x_mean"], float)
    x_scale = np.asarray(state["x_scale"], float)
    y_mean, y_scale = float(state["y_mean"]), float(state["y_scale"])
    if (x_mean.ndim != 1 or len(x_mean) == 0 or x_scale.shape != x_mean.shape
            or not np.isfinite(x_mean).all() or not np.isfinite(x_scale).all()
            or np.any(x_scale <= 0.) or not math.isfinite(y_mean)
            or not math.isfinite(y_scale) or y_scale <= 0.):
        raise ValueError("Invalid fit-only normalization state")
    mixture = _validate_components(state, len(x_mean))
    return x_mean, x_scale, y_mean, y_scale, mixture


def _predict_native(state, x):
    x_mean, x_scale, y_mean, y_scale, (priors, means, covs) = _validate_state(state)
    d = len(x_mean)
    x = _features(x, d, check_spout=(d == INPUT_DIM))
    components = []
    for mean, cov in zip(means, covs, strict=True):
        chol = np.linalg.cholesky(cov[:d, :d])
        coefficient = np.linalg.solve(cov[:d, :d], cov[:d, d])
        variance = float(cov[d, d] - np.dot(cov[d, :d], coefficient))
        if not math.isfinite(variance) or variance <= 0.:
            raise ValueError("Conditional variance is not positive")
        components.append((mean, chol, coefficient, math.sqrt(variance)*y_scale,
                           float(np.log(np.diag(chol)).sum())))
    result = np.empty(len(x))
    for row_index, row in enumerate(x):
        z = (row - x_mean)/x_scale
        log_weights, locations, scales = [], [], []
        for prior, (mean, chol, coefficient, scale, logdet) in zip(priors, components, strict=True):
            difference = z - mean[:d]
            whitened = np.linalg.solve(chol, difference)
            log_weights.append(math.log(float(prior)) - .5*(d*math.log(2*math.pi)
                               + 2*logdet + float(np.dot(whitened, whitened))))
            locations.append(y_mean + y_scale*(float(mean[d])
                             + float(np.dot(coefficient, difference))))
            scales.append(scale)
        maximum = max(log_weights)
        unnormalized = [math.exp(value - maximum) for value in log_weights]
        total = math.fsum(unnormalized)
        weights = np.asarray([value/total for value in unnormalized])
        locations, scales = np.asarray(locations), np.asarray(scales)
        if not np.isfinite(log_weights).all() or not np.isfinite(locations).all():
            raise ValueError("Conditional inference overflowed")
        low, high = float(locations.min()), float(locations.max())
        if low == high:
            median = low
        else:
            def centered_cdf(value):
                return _centered_cdf_logtails(value, weights, locations, scales)
            median = brentq(centered_cdf, low, high, xtol=1e-10,
                            rtol=4*np.finfo(float).eps, maxiter=1000)
        if not math.isfinite(median):
            raise ValueError("Conditional median is nonfinite; no clipping")
        result[row_index] = median
    return result


def _centered_cdf_logtails(value, weights, locations, scales):
    """Sign-preserving rescaled F(value)-.5 without subtracting rounded CDFs.

    For components to the left of value, Phi(z)=1-Q(z); for those to the
    right, Phi(z)=Q(-z).  Sum positive and negative tails separately in log
    space.  The rescaling is strictly positive, so roots are unchanged.
    """
    positive, negative, left_weights = [], [], []
    for weight, location, scale in zip(weights, locations, scales, strict=True):
        weight, location, scale = float(weight), float(location), float(scale)
        if weight == 0.:
            continue
        log_tail = math.log(weight) + float(log_ndtr(-abs((value - location)/scale)))
        if value >= location:
            left_weights.append(weight)
            negative.append(log_tail)
        else:
            positive.append(log_tail)
    delta = math.fsum(left_weights) - .5
    if delta > 0.:
        positive.append(math.log(delta))
    elif delta < 0.:
        negative.append(math.log(-delta))
    log_positive = float(logsumexp(positive)) if positive else -math.inf
    log_negative = float(logsumexp(negative)) if negative else -math.inf
    if log_positive == log_negative:
        return 0.
    if log_positive > log_negative:
        return -math.expm1(log_negative - log_positive)
    return math.expm1(log_positive - log_negative)


def _manual_cholesky(matrix):
    """Scalar cold path: no numpy/scipy linear algebra or native estimator."""
    n = len(matrix)
    lower = [[0.] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = float(matrix[i][j]) - math.fsum(lower[i][q]*lower[j][q]
                                                   for q in range(j))
            if i == j:
                if not math.isfinite(value) or value <= 0.:
                    raise ValueError("Cold covariance is not positive definite")
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value/lower[j][j]
    return lower


def _forward_substitute(lower, vector):
    output = []
    for i in range(len(vector)):
        output.append((float(vector[i]) - math.fsum(lower[i][j]*output[j]
                                                    for j in range(i)))/lower[i][i])
    return output


def _back_substitute_transpose(lower, vector):
    output = [0.] * len(vector)
    for i in reversed(range(len(vector))):
        output[i] = (float(vector[i]) - math.fsum(lower[j][i]*output[j]
                                                 for j in range(i + 1, len(vector))))/lower[i][i]
    return output


def _cold_log_survival(z):
    """Independent scalar log Q(z), z>=0; 12-term Mills series for z>=26."""
    if not math.isfinite(z) or z < 0.:
        raise ValueError("Cold normal tail requires finite nonnegative z")
    if z < 26.:
        return math.log(.5*math.erfc(z/math.sqrt(2.)))
    square = z*z
    term = 1.
    terms = [term]
    for order in range(1, 12):
        term *= -(2*order - 1)/square
        terms.append(term)
    correction = math.fsum(terms)
    if correction <= 0.:
        raise ValueError("Cold Mills expansion is invalid")
    return -.5*square - math.log(z) - .5*math.log(2*math.pi) + math.log(correction)


def _cold_logsumexp(values):
    if not values:
        return -math.inf
    maximum = max(values)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def _cold_centered_cdf_logtails(value, weights, locations, scales):
    """Independent scalar tail balance; never forms 1+erf or a rounded CDF."""
    positive, negative, left_mass = [], [], []
    for index in range(len(weights)):
        weight = float(weights[index])
        if weight == 0.:
            continue
        location, scale = float(locations[index]), float(scales[index])
        term = math.log(weight) + _cold_log_survival(abs((value - location)/scale))
        if value >= location:
            left_mass.append(weight)
            negative.append(term)
        else:
            positive.append(term)
    delta = math.fsum(left_mass) - .5
    if delta > 0.:
        positive.append(math.log(delta))
    elif delta < 0.:
        negative.append(math.log(-delta))
    positive_log, negative_log = _cold_logsumexp(positive), _cold_logsumexp(negative)
    if positive_log == negative_log:
        return 0.
    if positive_log > negative_log:
        return -math.expm1(negative_log - positive_log)
    return math.expm1(positive_log - negative_log)


def _cold_components(state, dimension):
    state = _component_view(state)
    weights = np.asarray(state["weights"], float)
    means = np.asarray(state["means"], float)
    covs = np.asarray(state["covariances"], float)
    k = len(weights) if weights.ndim == 1 else 0
    if (k == 0 or means.shape != (k, dimension + 1)
            or covs.shape != (k, dimension + 1, dimension + 1)
            or not np.isfinite(weights).all() or not np.isfinite(means).all()
            or not np.isfinite(covs).all() or np.any(weights <= 0.)
            or abs(math.fsum(float(w) for w in weights) - 1.) > 1e-10):
        raise ValueError("Invalid cold joint component arrays")
    for cov in covs:
        for i in range(dimension + 1):
            for j in range(dimension + 1):
                if abs(float(cov[i, j]) - float(cov[j, i])) > 1e-12:
                    raise ValueError("Cold joint covariance is not symmetric")
        _manual_cholesky(cov)
    return weights, means, covs


def independent_predict(state, x):
    """Cold median using scalar Cholesky/log-tails/bisection; absolute tol only."""
    x_mean, x_scale = np.asarray(state["x_mean"], float), np.asarray(state["x_scale"], float)
    y_mean, y_scale = float(state["y_mean"]), float(state["y_scale"])
    if (x_mean.ndim != 1 or len(x_mean) == 0 or x_scale.shape != x_mean.shape
            or not np.isfinite(x_mean).all() or not np.isfinite(x_scale).all()
            or np.any(x_scale <= 0.) or not math.isfinite(y_mean)
            or not math.isfinite(y_scale) or y_scale <= 0.):
        raise ValueError("Invalid cold normalization state")
    d = len(x_mean)
    x = _features(x, d, check_spout=(d == INPUT_DIM))
    priors, means, covs = _cold_components(state, d)
    parts = []
    for mean, cov in zip(means, covs, strict=True):
        lower = _manual_cholesky(cov[:d, :d])
        coefficient = _back_substitute_transpose(lower,
                            _forward_substitute(lower, cov[:d, d]))
        variance = float(cov[d, d]) - math.fsum(float(cov[d, j])*coefficient[j]
                                                for j in range(d))
        if not math.isfinite(variance) or variance <= 0.:
            raise ValueError("Cold conditional variance is not positive")
        parts.append((mean, lower, coefficient, math.sqrt(variance)*y_scale,
                      math.fsum(math.log(lower[j][j]) for j in range(d))))
    output = []
    for row in x:
        z = [(float(row[j]) - float(x_mean[j]))/float(x_scale[j]) for j in range(d)]
        log_weights, locations, scales = [], [], []
        for prior, (mean, lower, coefficient, scale, logdet) in zip(priors, parts, strict=True):
            difference = [z[j] - float(mean[j]) for j in range(d)]
            whitened = _forward_substitute(lower, difference)
            log_weights.append(math.log(float(prior)) - .5*(d*math.log(2*math.pi)
                               + 2*logdet + math.fsum(v*v for v in whitened)))
            locations.append(y_mean + y_scale*(float(mean[d])
                             + math.fsum(coefficient[j]*difference[j] for j in range(d))))
            scales.append(scale)
        if not all(math.isfinite(v) for v in log_weights + locations + scales):
            raise ValueError("Cold conditional inference overflowed")
        maximum = max(log_weights)
        weights = [math.exp(value - maximum) for value in log_weights]
        normalizer = math.fsum(weights)
        weights = [value/normalizer for value in weights]
        low, high = min(locations), max(locations)
        def centered_cdf(value):
            return _cold_centered_cdf_logtails(value, weights, locations, scales)
        median = low
        if low != high:
            if centered_cdf(low) > 0. or centered_cdf(high) < 0.:
                raise ValueError("Cold conditional median bracket is invalid")
            for _ in range(2000):
                middle = low + (high - low)/2.
                if high - low <= RAW_MEDIAN_TOL:
                    median = middle
                    break
                if middle == low or middle == high:
                    raise ValueError("Cold median cannot meet raw absolute tolerance")
                balance = centered_cdf(middle)
                if balance == 0.:
                    median = middle
                    break
                if balance < 0.:
                    low = middle
                else:
                    high = middle
            else:
                raise ValueError("Cold median bisection did not complete")
        if not math.isfinite(median):
            raise ValueError("Cold median is nonfinite; no clipping")
        output.append(median)
    return np.asarray(output, dtype=np.float64)


def _cold_log_components(state, joint, start_index=None):
    joint = np.asarray(joint, float)
    d = len(np.asarray(state["x_mean"]))
    if joint.ndim != 2 or joint.shape[1] != d + 1 or not np.isfinite(joint).all():
        raise ValueError("Expected a finite normalized joint fit matrix")
    components = _component_view(state, start_index)
    priors, means, covs = _cold_components(components, d)
    lower = [_manual_cholesky(cov) for cov in covs]
    constants = [math.log(float(w)) - .5*((d + 1)*math.log(2*math.pi)
                 + 2*math.fsum(math.log(l[j][j]) for j in range(d + 1)))
                 for w, l in zip(priors, lower, strict=True)]
    values = np.empty((len(joint), len(priors)))
    for i, row in enumerate(joint):
        for k, (mean, factor, constant) in enumerate(zip(means, lower, constants, strict=True)):
            difference = [float(row[j]) - float(mean[j]) for j in range(d + 1)]
            whitened = _forward_substitute(factor, difference)
            value = constant - .5*math.fsum(v*v for v in whitened)
            if not math.isfinite(value):
                raise ValueError("Cold fit likelihood overflowed")
            values[i, k] = value
    return values


def independent_score_samples(state, joint_normalized, start_index=None):
    """Independent scalar full joint score, for all five fit-only start audits."""
    values = _cold_log_components(state, joint_normalized, start_index)
    scores = []
    for row in values:
        maximum = max(float(v) for v in row)
        scores.append(maximum + math.log(math.fsum(math.exp(float(v) - maximum) for v in row)))
    return np.asarray(scores, dtype=np.float64)
