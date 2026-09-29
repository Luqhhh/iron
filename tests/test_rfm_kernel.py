"""Independent, tiny mathematical witnesses; no official-data or model fits."""
import numpy as np
import pytest

from bf_tap_r2.rfm_kernel import (
    laplace_kernel, numeric_gradient, solve_kernel, training_bandwidth, update_metric,
)


def test_kernel_and_gradient_oracles():
    centers = np.array([[0., 1.], [2., -1.], [-1., 3.]])
    category = np.eye(2)[[0, 1, 0]]
    metric = np.array([[2., .3], [.3, .7]])
    query = np.array([[.4, .6], [-.7, -2.]])
    qcat = np.eye(2)
    alpha = np.array([.3, -.8, 1.4])
    h = 1.3
    kernel = laplace_kernel(query, qcat, centers, category, metric, h)
    oracle = np.array([[np.exp(-np.sqrt(
        (x-z) @ metric @ (x-z) + np.sum((c-d)**2))/h)
        for z, d in zip(centers, category)] for x, c in zip(query, qcat)])
    np.testing.assert_allclose(kernel, oracle, rtol=1e-13, atol=1e-14)
    gradient = numeric_gradient(query, qcat, centers, category, metric, h, alpha)
    fd = np.empty_like(query)
    for row in range(len(query)):
        for col in range(query.shape[1]):
            plus, minus = query.copy(), query.copy()
            plus[row, col] += 1e-6
            minus[row, col] -= 1e-6
            fd[row, col] = ((laplace_kernel(plus, qcat, centers, category, metric, h)
                            - laplace_kernel(minus, qcat, centers, category, metric, h))
                           @ alpha)[row] / 2e-6
    np.testing.assert_allclose(gradient, fd, rtol=1e-5, atol=1e-6)
    self_kernel = laplace_kernel(centers, category, centers, category, metric, h)
    np.testing.assert_array_equal(np.diag(self_kernel), np.ones(3))
    np.testing.assert_allclose(self_kernel, self_kernel.T, atol=1e-14)


def test_coincident_centers_and_category_distance():
    z = np.array([[1., 2.], [1., 2.]])
    c = np.eye(2)
    k = laplace_kernel(z, c, z, c, np.eye(2), 1)
    assert k[0, 1] == pytest.approx(np.exp(-np.sqrt(2)))
    np.testing.assert_array_equal(
        numeric_gradient(z, c, z, c, np.eye(2), 1, np.array([1., 2.])),
        np.zeros((2, 2)),
    )


def test_coincident_contribution_excluded_not_entire_gradient():
    query = np.array([[0.]])
    centers = np.array([[0.], [1.]])
    g = numeric_gradient(query, np.zeros((1, 0)), centers, np.zeros((2, 0)),
                         np.eye(1), 1., np.array([1e8, 2.]))
    np.testing.assert_allclose(g, [[2 / np.e]], rtol=1e-14)


def test_block_order_chunk_and_input_immutability():
    rng = np.random.default_rng(56001)
    z, centers = rng.normal(size=(259, 3)), rng.normal(size=(9, 3))
    c, cc = np.eye(2)[np.arange(259) % 2], np.eye(2)[np.arange(9) % 2]
    metric = update_metric(rng.normal(size=(8, 3)))
    alpha = rng.normal(size=9)
    snapshots = [a.copy() for a in (z, c, centers, cc, metric, alpha)]
    for fn, extra in [(laplace_kernel, ()), (numeric_gradient, (alpha,))]:
        full = fn(z, c, centers, cc, metric, 1., *extra)
        reverse = fn(z[::-1], c[::-1], centers, cc, metric, 1., *extra)[::-1]
        chunked = np.concatenate([fn(z[i:i+7], c[i:i+7], centers, cc, metric, 1., *extra)
                                  for i in range(0, len(z), 7)])
        np.testing.assert_array_equal(full, reverse)
        np.testing.assert_array_equal(full, chunked)
    for before, after in zip(snapshots, (z, c, centers, cc, metric, alpha)):
        np.testing.assert_array_equal(before, after)


def test_cholesky_residual_and_literal_diagonal_regularizer():
    x = np.array([[0.], [0.], [2.]])
    c = np.zeros((3, 0))
    kernel = laplace_kernel(x, c, x, c, np.eye(1), 2.)
    y = np.array([1., 2., -1.])
    before = kernel.copy()
    alpha = solve_kernel(kernel, y)
    np.testing.assert_allclose((kernel + .01*np.eye(3)) @ alpha, y, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(before, kernel)
    with pytest.raises(ValueError, match="exactly"):
        solve_kernel(kernel, y, regularizer=.03)
    with pytest.raises(np.linalg.LinAlgError):
        solve_kernel(np.array([[1., 2.], [2., 1.]]), np.ones(2))


def test_metric_trace_psd_and_uncentered_agop():
    # Constant nonzero gradients must remain learnable: AGOP is not covariance.
    gradients = np.zeros((4, 21))
    gradients[:, 0] = 3.
    metric = update_metric(gradients)
    expected = np.eye(21)*.01
    expected[0, 0] += .99*21
    np.testing.assert_allclose(metric, expected, atol=1e-13)
    assert np.trace(metric) == pytest.approx(21., abs=1e-10)
    assert np.linalg.eigvalsh(metric).min() >= .01 - 1e-12
    np.testing.assert_array_equal(metric, metric.T)
    np.testing.assert_allclose(update_metric(gradients*13), metric, atol=1e-12)
    with pytest.raises(ValueError, match="trace"):
        update_metric(np.zeros((4, 21)))


def test_bandwidth_all_positive_pairs_including_category_and_duplicates():
    z = np.array([[0.], [0.], [3.]])
    c = np.zeros((3, 0))
    assert training_bandwidth(z, c) == 3.
    c = np.eye(2)[[0, 1, 0]]
    assert training_bandwidth(z, c) == 3.  # sqrt(2), 3, sqrt(11)
    with pytest.raises(ValueError, match="no positive"):
        training_bandwidth(np.zeros((3, 1)), np.zeros((3, 0)))


@pytest.mark.parametrize("metric", [np.diag([1., -1.]), np.array([[1., 1.], [0., 1.]]),
                                     np.eye(3), np.full((2, 2), np.nan)])
def test_invalid_metric_rejected(metric):
    with pytest.raises(ValueError):
        laplace_kernel(np.zeros((1, 2)), np.zeros((1, 0)), np.ones((2, 2)),
                       np.zeros((2, 0)), metric, 1.)


def test_invalid_shapes_values_and_bandwidth_rejected():
    x = np.zeros((2, 2))
    c = np.zeros((2, 0))
    for h in (0., -1., np.nan, np.inf):
        with pytest.raises(ValueError):
            laplace_kernel(x, c, x, c, np.eye(2), h)
    with pytest.raises(ValueError):
        numeric_gradient(x, c, x, c, np.eye(2), 1., np.ones(3))
    with pytest.raises(ValueError):
        laplace_kernel(x, c[:1], x, c, np.eye(2), 1.)
    with pytest.raises(ValueError):
        solve_kernel(np.eye(2), np.array([1., np.nan]))
