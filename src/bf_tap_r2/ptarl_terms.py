"""Original float64 equations for the pinned PTaRL code-inspired auxiliary route.

Single-atom transport has its unique feasible plan equal to the coordinates.
No Sinkhorn/POT dependency and no externally trained prototypes are required.
"""
from __future__ import annotations

import math
import numpy as np
import torch


def _matrix(x, name):
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float64 or x.ndim != 2:
        raise ValueError(f"{name} must be a float64 matrix")
    if not x.shape[0] or not x.shape[1] or not torch.isfinite(x).all():
        raise ValueError(f"{name} must be finite and nonempty")


def cosine_matrix(left, right, epsilon=1e-12):
    _matrix(left, "left"); _matrix(right, "right")
    if left.shape[1] != right.shape[1] or left.device != right.device:
        raise ValueError("Cosine feature width/device mismatch")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Positive finite norm epsilon required")
    a = left / left.square().sum(1, keepdim=True).clamp_min(epsilon**2).sqrt()
    b = right / right.square().sum(1, keepdim=True).clamp_min(epsilon**2).sqrt()
    return (a @ b.T).clamp(-1., 1.)


def _coordinates(r, prototypes=None):
    _matrix(r, "coordinates")
    if (r < 0).any() or not torch.allclose(r.sum(1), torch.ones(len(r), dtype=r.dtype, device=r.device),
                                         atol=1e-12, rtol=0):
        raise ValueError("Coordinates must be probability-simplex rows")
    if prototypes is not None and r.shape[1] != prototypes.shape[0]:
        raise ValueError("Coordinate/prototype width mismatch")


def projection_cost(hidden, coordinates, prototypes, epsilon=1e-12):
    """mean_i sum_k r_ik (1-cos(h_i,B_k)); singleton OT, exact not approximate."""
    _matrix(hidden, "hidden"); _matrix(prototypes, "prototypes")
    _coordinates(coordinates, prototypes)
    if len(hidden) != len(coordinates) or hidden.device != coordinates.device:
        raise ValueError("Projection row/device mismatch")
    return (coordinates * (1. - cosine_matrix(hidden, prototypes, epsilon))).sum(1).mean()


def regression_bins(y):
    """Pinned code rule: 1+floor(log2(batch rows)), equal-width on the FULL batch.

    Constant target batches map to bin0 explicitly. These bins exist only in
    training regularization and must never be computed from a query target.
    """
    if not isinstance(y, torch.Tensor) or y.dtype != torch.float64 or y.ndim != 1:
        raise ValueError("Targets must be a float64 vector")
    if not len(y) or not torch.isfinite(y).all():
        raise ValueError("Targets must be finite and nonempty")
    count = 1 + int(math.log2(len(y)))
    minimum, maximum = y.min(), y.max()
    if maximum == minimum:
        return torch.zeros(len(y), dtype=torch.long, device=y.device)
    return ((y - minimum) / ((maximum - minimum) / count)).floor().long().clamp(0, count-1)


def sampled_rows(rows, random_seed, step):
    """Half the batch, at least one row; independent explicit RNG, no global use."""
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError("Positive integer row count required")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (random_seed, step)):
        raise ValueError("Nonnegative integer seed/step required")
    rng = np.random.default_rng(np.random.SeedSequence([random_seed, step, 54003]))
    return rng.choice(rows, size=max(1, rows//2), replace=False).tolist()


def coordinate_diversity(coordinates, y, selected, epsilon=1e-12):
    """Global all-pair denominator, positive-count mean, INCLUDING diagonals.

    It is not row-wise InfoNCE. Bins are calculated before taking the half sample.
    """
    _coordinates(coordinates)
    bins = regression_bins(y)
    if len(y) != len(coordinates) or y.device != coordinates.device:
        raise ValueError("Coordinate/target row/device mismatch")
    if (not isinstance(selected, (list, tuple)) or not selected
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(y) for i in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("Unique valid selected rows required")
    r = coordinates[list(selected)]
    similarity = cosine_matrix(r, r, epsilon)
    groups = bins[list(selected)]
    positive = groups[:, None] == groups[None, :]
    denominator = torch.logsumexp(similarity.reshape(-1), 0)
    return denominator - similarity[positive].mean()


def prototype_orthogonalization(prototypes, epsilon=1e-12):
    """Author-code coefficient0.5, not the paper's coefficient1.

    An orthonormal bank has loss1, NOT0. A zero prototype is rejected, because
    the claimed unit cosine diagonal and the loss's lower bound require it.
    """
    _matrix(prototypes, "prototypes")
    if (prototypes.square().sum(1) <= epsilon**2).any():
        raise ValueError("Nonzero prototype rows required")
    matrix = cosine_matrix(prototypes, prototypes, epsilon).abs()
    l1 = matrix.sum()
    l2 = matrix.square().sum()
    return l1/l2 + .5*(l1-len(prototypes)).abs()


def auxiliary_terms(hidden, coordinates, prototypes, y, selected, epsilon=1e-12):
    """Return every term separately so saved traces cannot hide inactive losses."""
    return {"projection": projection_cost(hidden, coordinates, prototypes, epsilon),
            "diversity": coordinate_diversity(coordinates, y, selected, epsilon),
            "orthogonalization": prototype_orthogonalization(prototypes, epsilon)}
