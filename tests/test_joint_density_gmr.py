"""Pure arrays and fake native objects only: these tests never fit a model."""
import math

import numpy as np
import pytest

import bf_tap_r2.joint_density_gmr as gmr


class FakeTrackedMixture:
    scores = [0., 4., 4., -1., 2.]
    instances = []
    nonconverged_start = None
    failed_start = None
    bad_count_start = None

    def __init__(self, *, n_components, random_state, on_event,
                 on_iteration, start_index):
        self.n_components, self.random_state = n_components, random_state
        self.on_event, self.on_iteration = on_event, on_iteration
        self.start_index = start_index
        type(self).instances.append(self)

    def fit(self, joint):
        self.fit_joint = np.asarray(joint).copy()
        self.kmeans_started = 1
        self.on_event(dict(event="native_kmeans_started", start_index=self.start_index,
                           random_state=self.random_state, fit_rows=len(joint)))
        if self.start_index == type(self).failed_start:
            raise RuntimeError("fake native initialization failed")
        self.kmeans_completed = 1
        self.on_event(dict(event="native_kmeans_completed", start_index=self.start_index,
                           random_state=self.random_state, fit_rows=len(joint)))
        self.n_iter_ = 2
        # Lower bounds deliberately rank differently from final fit score.
        self.lower_bounds_ = [-10. + self.start_index, float(self.start_index)]
        self.lower_bound_ = self.lower_bounds_[-1]
        self.observed_iterations = (1 if self.start_index == type(self).bad_count_start else 2)
        self.observed_lower_bounds = list(self.lower_bounds_)
        for iteration, likelihood in enumerate(self.lower_bounds_, 1):
            if self.on_iteration is not None:
                self.on_iteration(dict(start_index=self.start_index,
                                       random_state=self.random_state,
                                       native_fit_index=self.start_index + 1,
                                       iteration=iteration, fit_rows=len(joint),
                                       pre_m_lower_bound=likelihood))
        self.converged_ = self.start_index != type(self).nonconverged_start
        self.weights_ = np.full(self.n_components, 1./self.n_components)
        self.means_ = np.zeros((self.n_components, 24))
        self.means_[:, -1] = self.start_index/10.
        self.covariances_ = np.repeat(np.eye(24)[None], self.n_components, axis=0)
        return self

    def score(self, joint):
        np.testing.assert_array_equal(joint, self.fit_joint)
        return type(self).scores[self.start_index]


@pytest.fixture(autouse=True)
def no_native_fits(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A pure-array/fake-model test attempted a real fit")
    monkeypatch.setattr(gmr.GaussianMixture, "fit", forbidden)


@pytest.fixture
def fake_native(monkeypatch):
    FakeTrackedMixture.instances = []
    FakeTrackedMixture.scores = [0., 4., 4., -1., 2.]
    FakeTrackedMixture.nonconverged_start = None
    FakeTrackedMixture.failed_start = None
    FakeTrackedMixture.bad_count_start = None
    monkeypatch.setattr(gmr, "TrackedGaussianMixture", FakeTrackedMixture)
    monkeypatch.setattr(gmr.sklearn, "__version__", "1.8.0")
    return FakeTrackedMixture


def fit_arrays():
    numeric = np.arange(12*21, dtype=float).reshape(12, 21)
    numeric[:, 3] = 7.  # A fit-only constant column has scale one.
    spout = np.eye(2)[np.arange(12) % 2]
    return np.column_stack((numeric, spout)), np.arange(100., 112.)


def single_state(weights=(1.,), means=((0., 0.),), covariances=None):
    if covariances is None:
        covariances = np.repeat(np.eye(2)[None], len(weights), axis=0)
    return dict(x_mean=np.array([0.]), x_scale=np.array([1.]),
                y_mean=np.array(100.), y_scale=np.array(10.),
                weights=np.array(weights), means=np.array(means),
                covariances=np.array(covariances), reg_covar=np.array(.01))


@pytest.mark.parametrize("k", [1, 8])
def test_five_native_starts_fit_only_normalization_and_final_score_selection(fake_native, k):
    x, y = fit_arrays()
    events, iterations = [], []
    model = gmr.JointDensityGMR(k).fit(x, y, events.append, iterations.append)
    assert [instance.random_state for instance in fake_native.instances] == list(range(42, 47))
    for instance in fake_native.instances:
        np.testing.assert_allclose(instance.fit_joint.mean(axis=0), 0., atol=1e-14)
        np.testing.assert_array_equal(instance.fit_joint[:, 3], np.zeros(len(x)))
        np.testing.assert_array_equal(instance.fit_joint[:, -1], (y - y.mean())/y.std())
    metadata, state = model.metadata(), model.export_state()
    assert metadata["native_fit_count"] == metadata["native_kmeans_count"] == 5
    assert metadata["native_kmeans_started"] == metadata["native_kmeans_completed"] == 5
    assert metadata["fit_rows"] == 12
    assert metadata["n_components"] == k
    assert metadata["selected_start"] == 1  # Earliest final-score tie; not lower_bound_.
    assert [r["n_iter"] for r in metadata["starts"]] == [2]*5
    assert [r["final_fit_score"] for r in metadata["starts"]] == [0., 4., 4., -1., 2.]
    for index in range(5):
        assert [event["event"] for event in events if event["start_index"] == index] == [
            "native_fit_started", "native_kmeans_started", "native_kmeans_completed",
            "native_fit_completed"]
        assert [event["iteration"] for event in iterations if event["start_index"] == index] == [1, 2]
    assert state["weights"].shape == (5, k)
    assert state["means"].shape == (5, k, 24)
    assert state["covariances"].shape == (5, k, 24, 24)
    assert state["em_lower_bounds"].shape == (5, 2000)
    assert np.isfinite(state["em_lower_bounds"]).all()
    np.testing.assert_array_equal(state["em_lower_bounds"][:, 2:], np.zeros((5, 1998)))
    np.testing.assert_array_equal(state["x_mean"], x.mean(axis=0))
    assert state["x_scale"][3] == 1.
    assert int(state["selected_start"]) == 1
    assert all(isinstance(value, np.ndarray) and value.dtype.kind in "fiub" for value in state.values())
    # Both export and metadata return copies rather than mutable fit state.
    state["means"][1, :, -1] = -1000.
    metadata["starts"][0]["random_state"] = -1
    assert model.metadata()["starts"][0]["random_state"] == 42
    np.testing.assert_array_equal(model.export_state()["means"][1, :, -1], np.full(k, .1))


def test_selected_arrays_are_used_without_query_labels(fake_native):
    x, y = fit_arrays()
    model = gmr.JointDensityGMR(8).fit(x, y)
    expected = y.mean() + .1*y.std()
    np.testing.assert_array_equal(model.predict(x[:3]), np.full(3, expected))
    np.testing.assert_array_equal(gmr.independent_predict(model.export_state(), x[:3]),
                                  np.full(3, expected))
    with pytest.raises(ValueError, match="feature-only"):
        model.predict(np.column_stack((x, y)))
    assert model.predict(x[:0]).shape == (0,)
    assert gmr.independent_predict(model.export_state(), x[:0]).shape == (0,)


@pytest.mark.parametrize("kind", ["nonconverged", "failed", "bad_count"])
def test_any_native_failure_stops_without_new_start_retry_or_fallback(fake_native, kind):
    setattr(fake_native, kind + "_start", 2)
    x, y = fit_arrays()
    events = []
    model = gmr.JointDensityGMR(8)
    with pytest.raises((ValueError, RuntimeError)):
        model.fit(x, y, events.append)
    assert len(fake_native.instances) == 3
    metadata = model.metadata()
    assert metadata["native_fit_count"] == 3
    assert metadata["status"] == "failed"
    assert metadata["starts"][2]["status"] == "failed"
    assert events[-1]["event"] == "native_fit_failed"
    assert metadata["selected_start"] is None
    with pytest.raises(ValueError):
        model.export_state()
    with pytest.raises(ValueError, match="retried"):
        model.fit(x, y)


def test_resource_callback_exception_is_not_retried(fake_native):
    x, y = fit_arrays()
    model = gmr.JointDensityGMR(1)
    def resource_gate(event):
        if event["iteration"] == 2:
            raise MemoryError("fake 1536 MiB resource gate")
    with pytest.raises(MemoryError):
        model.fit(x, y, on_iteration=resource_gate)
    assert len(fake_native.instances) == 1
    assert model.metadata()["starts"][0]["error_type"] == "MemoryError"


@pytest.mark.parametrize("case", ["extra_target", "nan", "unknown_spout", "target_nan", "negative_time"])
def test_invalid_fit_arrays_do_not_start_any_native_fit(fake_native, case):
    x, y = fit_arrays()
    if case == "extra_target":
        x = np.column_stack((x, y))
    elif case == "nan":
        x[0, 0] = np.nan
    elif case == "unknown_spout":
        x[0, -2:] = 0.
    elif case == "target_nan":
        y[0] = np.nan
    else:
        y[0] = -1.
    with pytest.raises(ValueError):
        gmr.JointDensityGMR(1).fit(x, y)
    assert fake_native.instances == []


@pytest.mark.parametrize("k", [0, 2, True, 1., "8"])
def test_constructor_rejects_unfrozen_components(k):
    with pytest.raises(ValueError):
        gmr.JointDensityGMR(k)


def test_locked_version_is_mandatory_before_native_fit(fake_native, monkeypatch):
    monkeypatch.setattr(gmr.sklearn, "__version__", "1.9.0")
    with pytest.raises(RuntimeError, match="locked"):
        gmr.JointDensityGMR(1).fit(*fit_arrays())
    assert fake_native.instances == []


def test_native_observer_uses_one_init_and_does_not_count_final_e_step(monkeypatch):
    events, iterations = [], []
    observer = gmr.TrackedGaussianMixture(n_components=8, random_state=44,
                start_index=2, on_event=events.append, on_iteration=iterations.append)
    assert observer.n_init == 1
    assert observer.init_params == "kmeans"
    assert observer.reg_covar == .01
    assert observer.max_iter == 2000 and observer.tol == 1e-5
    joint, logs = np.zeros((3, 24)), np.zeros((3, 8))
    calls = []
    def initialize(self, X, random_state, xp=None):
        calls.append(("initialize", X, random_state, xp))
    def e_step(self, X, xp=None):
        calls.append(("e_step", X, xp))
        return -3.5, logs
    def m_step(self, X, log_resp, xp=None):
        calls.append(("m_step", X, log_resp, xp))
    monkeypatch.setattr(gmr.GaussianMixture, "_initialize_parameters", initialize)
    monkeypatch.setattr(gmr.GaussianMixture, "_e_step", e_step)
    monkeypatch.setattr(gmr.GaussianMixture, "_m_step", m_step)
    observer._initialize_parameters(joint, "fake RNG", xp="fake namespace")
    observer._e_step(joint, xp="fake namespace")
    observer._m_step(joint, logs, xp="fake namespace")
    observer._e_step(joint, xp="fake namespace")  # Native fit's final consistency E.
    assert [entry[0] for entry in calls] == ["initialize", "e_step", "m_step", "e_step"]
    assert observer.observed_iterations == 1
    assert observer.observed_lower_bounds == [-3.5]
    assert observer.kmeans_started == observer.kmeans_completed == 1
    assert [event["event"] for event in events] == ["native_kmeans_started", "native_kmeans_completed"]
    assert iterations == [dict(start_index=2, random_state=44, native_fit_index=3,
                               iteration=1, fit_rows=3, pre_m_lower_bound=-3.5)]


def test_k1_conditional_affine_mean_and_variance_are_analytic():
    state = single_state(covariances=[[[4., 2.], [2., 3.]]])
    state.update(x_mean=np.array([2.]), x_scale=np.array([5.]))
    x = np.array([[2.], [7.], [-3.]])
    expected = np.array([100., 105., 95.])
    np.testing.assert_array_equal(gmr._predict_native(state, x), expected)
    np.testing.assert_array_equal(gmr.independent_predict(state, x), expected)


def test_mixture_median_not_mean_and_cold_matches_brentq():
    state = single_state(weights=(.4, .6), means=((-2., -2.), (2., 2.)),
                         covariances=[[[1., 0.], [0., .25]], [[1., 0.], [0., .25]]])
    x = np.array([[0.], [-1.], [1.], [-4.], [4.]])
    native = gmr._predict_native(state, x)
    cold = gmr.independent_predict(state, x)
    np.testing.assert_allclose(cold, native, rtol=0., atol=1e-8)
    assert native[0] > 110. and abs(native[0] - 104.) > 6.
    assert native[1] < native[0] < native[2]
    for prediction, row in zip(cold, x, strict=True):
        logs = [math.log(prior) - .5*(float(row[0]) - mean[0])**2
                for prior, mean in zip((.4, .6), ((-2., -2.), (2., 2.)), strict=True)]
        weights = [math.exp(log - max(logs)) for log in logs]
        total = math.fsum(weights)
        cdf = math.fsum(w/total*.5*(1. + math.erf((prediction - location)/(5.*math.sqrt(2.))))
                        for w, location in zip(weights, (80., 120.), strict=True))
        assert abs(cdf - .5) < 1e-9


def test_cold_forward_does_not_call_scipy_or_numpy_linear_algebra(monkeypatch):
    state = single_state(weights=(.4, .6), means=((-1., -1.), (1., 1.)))
    def forbidden(*args, **kwargs):
        raise AssertionError("Cold path called native numerical inference")
    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(gmr, "brentq", forbidden)
    monkeypatch.setattr(gmr, "log_ndtr", forbidden)
    monkeypatch.setattr(gmr, "logsumexp", forbidden)
    assert np.isfinite(gmr.independent_predict(state, [[0.], [1.]])).all()
    assert np.isfinite(gmr.independent_score_samples(state, [[0., 0.]])).all()


@pytest.mark.parametrize("forward", [gmr._predict_native, gmr.independent_predict])
def test_row_chunk_and_reverse_invariance(forward):
    state = single_state(weights=(.3, .7), means=((-1., -1.), (1., 2.)))
    x = np.linspace(-3., 3., 11)[:, None]
    full = forward(state, x)
    np.testing.assert_array_equal(full, forward(state, x[::-1])[::-1])
    np.testing.assert_array_equal(full, np.concatenate([forward(state, x[:4]), forward(state, x[4:])]))
    assert forward(state, x[:0]).shape == (0,)


@pytest.mark.parametrize("forward", [gmr._predict_native, gmr.independent_predict])
@pytest.mark.parametrize("case", ["bad_covariance", "zero_scale", "nan_query"])
def test_inference_failure_is_explicit_and_never_clipped(forward, case):
    state = single_state()
    x = np.array([[0.]])
    if case == "bad_covariance":
        state["covariances"][0, 1, 1] = -1.
    elif case == "zero_scale":
        state["y_scale"] = np.array(0.)
    else:
        x[0, 0] = np.nan
    with pytest.raises(ValueError):
        forward(state, x)


def test_full_joint_cold_likelihood_is_analytic():
    joint = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    state = single_state(covariances=[[[1.01, 0.], [0., 1.01]]])
    expected = -math.log(2*math.pi) - math.log(1.01) - 1./1.01
    np.testing.assert_allclose(gmr.independent_score_samples(state, joint),
                               np.full(4, expected), rtol=0., atol=1e-14)


@pytest.mark.parametrize("forward", [gmr._predict_native, gmr.independent_predict])
def test_finite_negative_raw_component_is_not_clipped(forward):
    state = single_state()
    state["y_mean"] = np.array(-1.)
    np.testing.assert_array_equal(forward(state, [[0.]]), [-1.])


def test_retained_start_likelihood_can_be_audited_by_index(fake_native):
    model = gmr.JointDensityGMR(1).fit(*fit_arrays())
    state = model.export_state()
    joint = np.zeros((2, 24))
    score0 = gmr.independent_score_samples(state, joint, 0)
    score1 = gmr.independent_score_samples(state, joint, 1)
    np.testing.assert_allclose(score1 - score0, -.005, atol=1e-14, rtol=0.)
    np.testing.assert_array_equal(gmr.independent_score_samples(state, joint), score1)
    with pytest.raises(ValueError):
        gmr.independent_score_samples(state, joint, 5)


@pytest.mark.parametrize("separation", [10., 100.])
def test_wide_symmetric_mixture_has_stable_center_not_a_cdf_rounding_plateau(separation):
    state = single_state(weights=(.5, .5), means=((0., -separation), (0., separation)))
    state.update(y_mean=np.array(100.), y_scale=np.array(1.))
    x = np.array([[0.], [-1.], [1.], [2.]])
    native, cold = gmr._predict_native(state, x), gmr.independent_predict(state, x)
    np.testing.assert_allclose(native, np.full(len(x), 100.), rtol=0., atol=1e-8)
    np.testing.assert_allclose(cold, np.full(len(x), 100.), rtol=0., atol=1e-8)
    locations, scales = [100. - separation, 100. + separation], [1., 1.]
    for balance in (gmr._centered_cdf_logtails, gmr._cold_centered_cdf_logtails):
        assert balance(100. - .01, [.5, .5], locations, scales) < 0.
        assert balance(100., [.5, .5], locations, scales) == 0.
        assert balance(100. + .01, [.5, .5], locations, scales) > 0.


@pytest.mark.parametrize("separation", [10., 100.])
@pytest.mark.parametrize("epsilon", [-1e-12, 1e-12, -1e-8, 1e-8])
def test_near_symmetric_wide_mixture_is_stable_and_order_invariant(separation, epsilon):
    state = single_state(weights=(.5 - epsilon, .5 + epsilon),
                         means=((0., -separation), (0., separation)))
    state.update(y_mean=np.array(100.), y_scale=np.array(1.))
    x = np.array([[0.], [-1.], [1.], [2.]])
    native, cold = gmr._predict_native(state, x), gmr.independent_predict(state, x)
    np.testing.assert_allclose(cold, native, rtol=0., atol=1e-8)
    assert (native[0] - 100.)*epsilon > 0.
    for forward in (gmr._predict_native, gmr.independent_predict):
        np.testing.assert_array_equal(forward(state, x), forward(state, x[::-1])[::-1])
        np.testing.assert_array_equal(forward(state, x), np.concatenate([
            forward(state, x[:1]), forward(state, x[1:])]))
    reversed_state = dict(state, weights=state["weights"][::-1],
                          means=state["means"][::-1], covariances=state["covariances"][::-1])
    np.testing.assert_array_equal(gmr.independent_predict(state, x),
                                  gmr.independent_predict(reversed_state, x))


@pytest.mark.parametrize("z", [0., 1., 10., 25.999, 26., 30., 100.])
def test_independent_log_survival_matches_primary_tail(z):
    np.testing.assert_allclose(gmr._cold_log_survival(z), float(gmr.log_ndtr(-z)),
                               rtol=0., atol=2e-12)
