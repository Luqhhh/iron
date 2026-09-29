"""Float64 primitives for the bounded RFM design; no data loading or fit runner.

The experiment uses 21 numeric coordinates. Primitives also accept smaller
dimensions so independent analytic tests need no competition data. Category
coordinates affect distances but are never part of the learned metric.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist, pdist

QUERY_BLOCK = 128
PSD_TOLERANCE = 1e-10
COINCIDENT_TOLERANCE = 1e-12


def _array(value, ndim, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return value


def _metric(value, dimension):
    matrix = _array(value, 2, "metric")
    if matrix.shape != (dimension, dimension):
        raise ValueError("metric shape does not match numeric coordinates")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
        raise ValueError("metric must be symmetric")
    matrix = (matrix + matrix.T) / 2
    eigenvalues, vectors = np.linalg.eigh(matrix)
    if eigenvalues.min() < -PSD_TOLERANCE:
        raise ValueError("metric must be positive semidefinite")
    # This is only a roundoff projection, never a solver jitter schedule.
    eigenvalues = np.maximum(eigenvalues, 0)
    root = vectors * np.sqrt(eigenvalues)
    return root @ root.T, root


def _inputs(z, c, centers, center_c, metric, bandwidth):
    z = _array(z, 2, "numeric queries")
    centers = _array(centers, 2, "numeric centers")
    c = _array(c, 2, "categorical queries")
    center_c = _array(center_c, 2, "categorical centers")
    if not centers.shape[0] or not centers.shape[1]:
        raise ValueError("nonempty numeric centers required")
    if z.shape[1] != centers.shape[1]:
        raise ValueError("numeric coordinate counts differ")
    if c.shape[0] != z.shape[0] or center_c.shape[0] != centers.shape[0]:
        raise ValueError("numeric/categorical row counts differ")
    if c.shape[1] != center_c.shape[1]:
        raise ValueError("categorical coordinate counts differ")
    bandwidth = float(bandwidth)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("positive finite bandwidth required")
    matrix, root = _metric(metric, z.shape[1])
    # Rowwise contraction avoids batch-size dependent BLAS reductions. Direct
    # distances avoid cancellation in ||x||^2 + ||c||^2 - 2*x.c near duplicates.
    tz = np.einsum("ij,jk->ik", z, root, optimize=False)
    tc = np.einsum("ij,jk->ik", centers, root, optimize=False)
    if not np.isfinite(tz).all() or not np.isfinite(tc).all():
        raise ValueError("nonfinite metric-transformed coordinates")
    return z, c, centers, center_c, matrix, bandwidth, tz, tc


def _distance(z, c, centers, center_c):
    squared = cdist(z, centers, metric="sqeuclidean")
    squared += cdist(c, center_c, metric="sqeuclidean")
    if not np.isfinite(squared).all():
        raise ValueError("nonfinite squared distance")
    return np.sqrt(np.maximum(squared, 0))


def laplace_kernel(z, c, centers, center_c, M, h):
    """Return exp(-sqrt((z-center)' M (z-center) + category_distance)/h)."""
    z, c, centers, center_c, _, h, tz, tc = _inputs(z, c, centers, center_c, M, h)
    result = np.empty((len(z), len(centers)), dtype=np.float64)
    for start in range(0, len(z), QUERY_BLOCK):
        part = slice(start, start + QUERY_BLOCK)
        result[part] = np.exp(-_distance(tz[part], c[part], tc, center_c) / h)
    return result


def numeric_gradient(z, c, centers, center_c, M, h, alpha):
    """Derivative of K(z, centers) @ alpha in numeric coordinates only.

    Each center at total distance <=1e-12 contributes zero. At coincident
    points this is the frozen convention for a nondifferentiable kernel,
    not a claim that the usual derivative exists there.
    """
    z, c, centers, center_c, matrix, h, tz, tc = _inputs(z, c, centers, center_c, M, h)
    alpha = _array(alpha, 1, "alpha")
    if alpha.shape != (len(centers),):
        raise ValueError("one coefficient per center required")
    result = np.empty_like(z)
    for start in range(0, len(z), QUERY_BLOCK):
        part = slice(start, start + QUERY_BLOCK)
        radius = _distance(tz[part], c[part], tc, center_c)
        coefficient = np.zeros_like(radius)
        np.divide(np.exp(-radius / h), h * radius, out=coefficient,
                  where=radius > COINCIDENT_TOLERANCE)
        coefficient *= alpha[None, :]
        # At most one block x centers temporary per coordinate, never n*n*d.
        # Subtract coordinates before summing to retain near-center accuracy.
        difference = np.empty((len(z[part]), z.shape[1]))
        for j in range(z.shape[1]):
            difference[:, j] = np.sum(
                coefficient * (centers[None, :, j] - z[part, j, None]), axis=1
            )
        result[part] = np.einsum("ij,jk->ik", difference, matrix, optimize=False)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite numeric gradient")
    return result


def update_metric(gradients):
    """Normalize uncentered AGOP to trace d, shrink 1% toward I_d."""
    gradients = _array(gradients, 2, "gradients")
    n, dimension = gradients.shape
    if not n or not dimension:
        raise ValueError("nonempty gradients required")
    agop = gradients.T @ gradients / n
    agop = (agop + agop.T) / 2
    if not np.isfinite(agop).all():
        raise ValueError("nonfinite AGOP")
    eigenvalues, vectors = np.linalg.eigh(agop)
    if eigenvalues.min() < -PSD_TOLERANCE:
        raise ValueError("materially negative AGOP eigenvalue")
    eigenvalues = np.maximum(eigenvalues, 0)
    trace = eigenvalues.sum()
    if not np.isfinite(trace) or trace <= 0:
        raise ValueError("positive finite AGOP trace required")
    normalized = (vectors * (dimension * eigenvalues / trace)) @ vectors.T
    return 0.99 * normalized + 0.01 * np.eye(dimension)


def training_bandwidth(z, c):
    """Median positive identity-metric distance over unordered training pairs."""
    z = _array(z, 2, "training numeric coordinates")
    c = _array(c, 2, "training categorical coordinates")
    if len(z) != len(c) or len(z) < 2 or not z.shape[1]:
        raise ValueError("aligned training rows and at least two rows required")
    distance = pdist(np.concatenate([z, c], axis=1), metric="euclidean")
    if not np.isfinite(distance).all():
        raise ValueError("nonfinite training distance")
    positive = distance[distance > 0]
    if not len(positive):
        raise ValueError("no positive training pair distance")
    return float(np.median(positive))


def solve_kernel(K, y, regularizer=0.01):
    """Solve (K + .01 I) alpha = y once; never adapt jitter or retry."""
    kernel = _array(K, 2, "kernel")
    y = _array(y, 1, "target")
    if not len(y) or kernel.shape != (len(y), len(y)):
        raise ValueError("nonempty square kernel matching target required")
    if not np.allclose(kernel, kernel.T, rtol=0, atol=1e-12):
        raise ValueError("kernel must be symmetric")
    if regularizer != 0.01:
        raise ValueError("the frozen diagonal regularizer is exactly 0.01")
    system = kernel.copy()
    system.flat[::len(y) + 1] += regularizer
    factor = cho_factor(system, lower=True, check_finite=True)
    alpha = cho_solve(factor, y, check_finite=True)
    if not np.isfinite(alpha).all():
        raise ValueError("nonfinite kernel coefficients")
    return alpha
