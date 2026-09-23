"""Round2 V3.1 fusion primitives: pooled metrics, monotonic triage, exact-weight LP."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .metrics import wmape

V31_ALPHAS = (0.10, 0.25, 0.50)


def pooled_wmape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Pooled WMAPE over one complete aligned coverage set."""
    return wmape(np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float))


def triage_delta_v31(delta: float) -> str:
    """Monotonic V3.1 budget-management bands.

    >= 0.15 is always primary; values above 0.20 are not reassigned back to
    candidate_pool as in the original V3 helper.
    """
    value = float(delta)
    if value < 0.02:
        return "not_independent"
    if value < 0.05:
        return "fallback"
    if value < 0.15:
        return "candidate_pool"
    return "primary"


def simple_mix(reference: np.ndarray, candidate: np.ndarray, alpha: float) -> np.ndarray:
    """(1-alpha) * reference + alpha * candidate in original units."""
    a = float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.shape != cand.shape or ref.ndim != 1 or not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError("Invalid simple-mix arrays")
    return (1.0 - a) * ref + a * cand


def _prepare_matrices(y_by_seed: Mapping[str, np.ndarray],
                      p_by_seed: Mapping[str, np.ndarray]) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    seeds = sorted(y_by_seed)
    if seeds != sorted(p_by_seed):
        raise ValueError("Seed sets differ")
    n = None
    m = None
    ys, ps = [], []
    for seed in seeds:
        y = np.asarray(y_by_seed[seed], dtype=float)
        p = np.asarray(p_by_seed[seed], dtype=float)
        if y.ndim != 1 or p.ndim != 2 or p.shape[0] != y.shape[0] or not len(y):
            raise ValueError("Invalid fusion matrix shape")
        if n is None:
            n, m = p.shape
        if p.shape != (n, m):
            raise ValueError("Fusion matrix shape mismatch across seeds")
        if not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError("Fusion matrices must be finite")
        if np.abs(y).sum() <= 0:
            raise ValueError("Positive WMAPE denominator required")
        ys.append(y)
        ps.append(p)
    return seeds, ys, ps


def lp_simplex_weights(y_by_seed: Mapping[str, np.ndarray],
                       p_by_seed: Mapping[str, np.ndarray]) -> dict:
    """Solve the fixed-member nonnegative simplex WMAPE LP exactly.

    Variables are `[w (M), t (S*N)]`:

        minimize  mean_s  sum_i t[s, i] / sum_i |y[s, i]|
        subject to:
          P[s] w - t[s] <= y[s]
         -P[s] w - t[s] <= -y[s]
          w >= 0, sum(w) = 1, t >= 0
    """
    seeds, ys, ps = _prepare_matrices(y_by_seed, p_by_seed)
    n = ys[0].shape[0]
    m = ps[0].shape[1]
    s_count = len(seeds)
    objective = np.zeros(m + s_count * n, dtype=float)
    a_ub_blocks = []
    b_ub = []
    ident = sparse.identity(n, format="csr")
    for idx, (y, p) in enumerate(zip(ys, ps)):
        denom = float(np.abs(y).sum())
        objective[m + idx * n: m + (idx + 1) * n] = 1.0 / (s_count * denom)
        p_sparse = sparse.csr_matrix(p)
        left_zeros = sparse.csr_matrix((n, idx * n), dtype=float)
        right_zeros = sparse.csr_matrix((n, (s_count - idx - 1) * n), dtype=float)
        top = sparse.hstack([p_sparse, left_zeros, -ident, right_zeros], format="csr")
        bottom = sparse.hstack([-p_sparse, left_zeros, -ident, right_zeros], format="csr")
        a_ub_blocks.append(sparse.vstack([top, bottom], format="csr"))
        b_ub.extend(y.tolist())
        b_ub.extend((-y).tolist())
    a_ub = sparse.vstack(a_ub_blocks, format="csr")
    a_eq = sparse.hstack([
        sparse.csr_matrix(np.ones((1, m), dtype=float)),
        sparse.csr_matrix((1, s_count * n), dtype=float),
    ], format="csr")
    bounds = [(0.0, 1.0)] * m + [(0.0, None)] * (s_count * n)
    result = linprog(objective, A_ub=a_ub, b_ub=np.asarray(b_ub, dtype=float),
                     A_eq=a_eq, b_eq=np.ones(1, dtype=float), bounds=bounds,
                     method="highs")
    if not result.success or result.x is None:
        raise ValueError(f"V3.1 simplex LP failed: status={result.status} {result.message}")
    weights = np.asarray(result.x[:m], dtype=float)
    weights[weights < 1e-12] = 0.0
    total = float(weights.sum())
    if abs(total - 1.0) > 1e-7:
        raise ValueError(f"V3.1 simplex LP weight sum is {total}")
    weights /= total
    recomputed = float(np.mean([
        pooled_wmape(y, p @ weights) for y, p in zip(ys, ps)
    ]))
    if abs(recomputed - float(result.fun)) > 1e-8 * max(1.0, abs(recomputed)):
        raise ValueError(f"V3.1 simplex LP objective mismatch: {result.fun} vs {recomputed}")
    return {
        "weights": weights,
        "objective": recomputed,
        "solver_objective": float(result.fun),
        "status": int(result.status),
        "message": str(result.message),
        "seeds": seeds,
    }


def greedy_forward_select_lp(y_by_seed: Mapping[str, np.ndarray],
                             p_by_seed: Mapping[str, np.ndarray],
                             names: Sequence[str],
                             max_members: int = 5) -> dict:
    """Small greedy forward member selection with exact LP refitting."""
    seeds = sorted(y_by_seed)
    if len(names) != np.asarray(p_by_seed[seeds[0]]).shape[1]:
        raise ValueError("Candidate name count mismatch")
    chosen: list[int] = []
    history: list[dict] = []
    for _ in range(max(0, int(max_members))):
        best = None
        for j in range(len(names)):
            if j in chosen:
                continue
            columns = [*chosen, j]
            p = {s: np.asarray(p_by_seed[s])[:, columns] for s in seeds}
            try:
                fit = lp_simplex_weights(y_by_seed, p)
            except ValueError:
                continue
            if best is None or fit["objective"] < best["objective"]:
                best = {"objective": fit["objective"], "columns": columns}
        if best is None:
            break
        if history and best["objective"] >= history[-1]["objective"] - 1e-12:
            break
        chosen = list(best["columns"])
        fit = lp_simplex_weights(y_by_seed, {s: np.asarray(p_by_seed[s])[:, chosen] for s in seeds})
        history.append({
            "members": [names[j] for j in chosen],
            "columns": chosen,
            "objective": fit["objective"],
            "weights": {names[j]: float(w) for j, w in zip(chosen, fit["weights"])},
        })
    if not chosen:
        raise ValueError("No V3.1 LP fusion members selected")
    return {
        "members": history[-1]["members"],
        "columns": chosen,
        "weights": history[-1]["weights"],
        "objective": history[-1]["objective"],
        "history": history,
    }
