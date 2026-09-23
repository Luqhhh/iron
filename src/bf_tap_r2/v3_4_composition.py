"""Constrained V3.4 composition primitives.

The main rule is deliberately narrow: keep the frozen L1 prediction and allow
at most two new experts whose total weight may not exceed 0.5.  All member and
weight learning happens on the supplied training/development predictions only;
the caller is responsible for never passing outer-validation labels to these
functions.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .metrics import wmape

SINGLE_L1_NAME = "L1"


def _prepare(y_by_seed: Mapping[str, np.ndarray],
             p_by_seed: Mapping[str, np.ndarray]) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    seeds = sorted(y_by_seed)
    if seeds != sorted(p_by_seed):
        raise ValueError("Prediction and label seed sets differ")
    n = None
    m = None
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for seed in seeds:
        y = np.asarray(y_by_seed[seed], dtype=float)
        p = np.asarray(p_by_seed[seed], dtype=float)
        if y.ndim != 1 or p.ndim != 2 or p.shape[0] != y.shape[0] or not len(y):
            raise ValueError("Invalid constrained-composition matrix shape")
        if n is None:
            n, m = p.shape
        if p.shape != (n, m):
            raise ValueError("Constrained-composition matrix shape mismatch")
        if not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError("Constrained-composition matrices must be finite")
        if np.abs(y).sum() <= 0:
            raise ValueError("WMAPE denominator must be positive")
        ys.append(y)
        ps.append(p)
    assert m is not None
    return seeds, ys, ps


def pooled_wmape_by_seed(y_by_seed: Mapping[str, np.ndarray],
                         p_by_seed: Mapping[str, np.ndarray]) -> dict[str, float]:
    seeds, ys, ps = _prepare(y_by_seed, p_by_seed)
    return {seed: float(wmape(y, p)) for seed, y, p in zip(seeds, ys, ps)}


def evaluate_weights(y_by_seed: Mapping[str, np.ndarray],
                     p_by_seed: Mapping[str, np.ndarray],
                     weights: np.ndarray) -> dict[str, Any]:
    seeds, ys, ps = _prepare(y_by_seed, p_by_seed)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape[0] != ps[0].shape[1]:
        raise ValueError("Weight vector shape mismatch")
    if (w < -1e-12).any() or abs(float(w.sum()) - 1.0) > 1e-7:
        raise ValueError("Weights must be nonnegative and sum to one")
    per_seed = {seed: float(wmape(y, p @ w)) for seed, y, p in zip(seeds, ys, ps)}
    return {
        "seeds": seeds,
        "per_seed_wmape": per_seed,
        "mean_wmape": float(np.mean(list(per_seed.values()))),
        "weights": w.tolist(),
    }


def _linearized_constraints(ys: Sequence[np.ndarray], ps: Sequence[np.ndarray],
                            m: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    n = ys[0].shape[0]
    count = len(ys)
    a_blocks = []
    b_ub: list[float] = []
    ident = sparse.identity(n, format="csr")
    for idx, (y, p) in enumerate(zip(ys, ps)):
        p_sparse = sparse.csr_matrix(p)
        left_zeros = sparse.csr_matrix((n, idx * n), dtype=float)
        right_zeros = sparse.csr_matrix((n, (count - idx - 1) * n), dtype=float)
        top = sparse.hstack([p_sparse, left_zeros, -ident, right_zeros], format="csr")
        bottom = sparse.hstack([-p_sparse, left_zeros, -ident, right_zeros], format="csr")
        a_blocks.append(sparse.vstack([top, bottom], format="csr"))
        b_ub.extend(y.tolist())
        b_ub.extend((-y).tolist())
    return sparse.vstack(a_blocks, format="csr"), np.asarray(b_ub, dtype=float)


def lp_constrained_weights(y_by_seed: Mapping[str, np.ndarray],
                           p_by_seed: Mapping[str, np.ndarray],
                           new_indices: Sequence[int],
                           max_new_weight: float = 0.5) -> dict[str, Any]:
    """Exact simplex LP with an upper bound on total new-expert weight."""
    seeds, ys, ps = _prepare(y_by_seed, p_by_seed)
    n = ys[0].shape[0]
    m = ps[0].shape[1]
    limit = float(max_new_weight)
    if not 0.0 <= limit <= 1.0:
        raise ValueError("max_new_weight must be in [0, 1]")
    new = np.zeros(m, dtype=float)
    for index in new_indices:
        idx = int(index)
        if not 0 <= idx < m:
            raise ValueError("New-expert column index out of range")
        new[idx] = 1.0
    count = len(seeds)
    objective = np.zeros(m + count * n, dtype=float)
    for idx, y in enumerate(ys):
        denom = float(np.abs(y).sum())
        objective[m + idx * n: m + (idx + 1) * n] = 1.0 / (count * denom)
    a_ub, b_ub = _linearized_constraints(ys, ps, m)
    if new.any():
        constraint_row = sparse.hstack([
            sparse.csr_matrix(new.reshape(1, m)),
            sparse.csr_matrix((1, count * n)),
        ], format="csr")
        a_ub = sparse.vstack([a_ub, constraint_row], format="csr")
        b_ub = np.concatenate([b_ub, np.asarray([limit], dtype=float)])
    a_eq = sparse.hstack([
        sparse.csr_matrix(np.ones((1, m), dtype=float)),
        sparse.csr_matrix((1, count * n), dtype=float),
    ], format="csr")
    bounds = [(0.0, 1.0)] * m + [(0.0, None)] * (count * n)
    result = linprog(objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq,
                     b_eq=np.ones(1, dtype=float), bounds=bounds, method="highs")
    if not result.success or result.x is None:
        raise ValueError(f"V3.4 constrained LP failed: status={result.status} {result.message}")
    weights = np.asarray(result.x[:m], dtype=float)
    weights[weights < 1e-12] = 0.0
    total = float(weights.sum())
    if abs(total - 1.0) > 1e-7:
        raise ValueError(f"V3.4 constrained LP weight sum is {total}")
    weights /= total
    new_total = float(np.dot(weights, new))
    if new_total > limit + 1e-7:
        raise ValueError(f"V3.4 constrained LP new weight {new_total} exceeds {limit}")
    recomputed = float(np.mean([wmape(y, p @ weights) for y, p in zip(ys, ps)]))
    if abs(recomputed - float(result.fun)) > 1e-8 * max(1.0, abs(recomputed)):
        raise ValueError(f"V3.4 constrained LP objective mismatch: {result.fun} vs {recomputed}")
    return {
        "weights": weights,
        "new_weight_total": new_total,
        "objective": recomputed,
        "solver_objective": float(result.fun),
        "status": int(result.status),
        "message": str(result.message),
        "seeds": seeds,
    }


def constrained_forward_select(y_by_seed: Mapping[str, np.ndarray],
                               p_by_seed: Mapping[str, np.ndarray],
                               names: Sequence[str],
                               l1_index: int,
                               candidate_indices: Sequence[int],
                               max_new_experts: int = 2,
                               max_new_weight: float = 0.5) -> dict[str, Any]:
    """Enumerate L1 + up to two new experts and solve the constrained LP.

    The empty new-expert set is included and is exactly the L1 control.
    """
    seeds, ys, ps = _prepare(y_by_seed, p_by_seed)
    if len(names) != ps[0].shape[1]:
        raise ValueError("Member names do not match prediction columns")
    l1 = int(l1_index)
    if not 0 <= l1 < len(names):
        raise ValueError("L1 column index out of range")
    candidates = sorted({int(i) for i in candidate_indices if int(i) != l1})
    limit_new = max(0, int(max_new_experts))
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for size in range(0, min(limit_new, len(candidates)) + 1):
        for subset in combinations(candidates, size):
            columns = [l1, *subset]
            p_selected = {seed: p[:, columns] for seed, p in zip(seeds, ps)}
            if subset:
                fit = lp_constrained_weights(
                    y_by_seed, p_selected,
                    new_indices=list(range(1, len(columns))),
                    max_new_weight=max_new_weight,
                )
                objective = float(fit["objective"])
                weights = fit["weights"]
            else:
                prediction = ps[0][:, l1]
                per_seed = {seed: float(wmape(y, p[:, l1])) for seed, y, p in zip(seeds, ys, ps)}
                objective = float(np.mean(list(per_seed.values())))
                weights = np.asarray([1.0], dtype=float)
                fit = {"weights": weights, "new_weight_total": 0.0, "objective": objective,
                       "solver_objective": objective, "status": 0, "message": "L1 control",
                       "seeds": seeds}
            record = {
                "columns": columns,
                "members": [str(names[i]) for i in columns],
                "new_experts": [str(names[i]) for i in subset],
                "new_weight_total": float(fit.get("new_weight_total", 0.0)),
                "objective": objective,
                "weights": [float(v) for v in weights],
                "seeds": seeds,
            }
            history.append(record)
            if best is None or (objective, len(subset), str(record["members"])) < (
                best["objective"], len(best["new_experts"]), str(best["members"])
            ):
                best = record
    if best is None:
        raise RuntimeError("No V3.4 constrained composition candidate")
    baseline = [h for h in history if not h["new_experts"]]
    if not baseline:
        raise RuntimeError("L1 control missing from constrained composition history")
    return {
        **best,
        "baseline_l1_objective": float(baseline[0]["objective"]),
        "delta_vs_l1": float(baseline[0]["objective"] - best["objective"]),
        "history": history,
        "max_new_experts": limit_new,
        "max_new_weight": float(max_new_weight),
    }


def simple_mix_alphas(actual_by_seed: Mapping[str, np.ndarray],
                      reference_by_seed: Mapping[str, np.ndarray],
                      candidate_by_seed: Mapping[str, np.ndarray],
                      alphas: Sequence[float] = (0.10, 0.25, 0.50)) -> list[dict[str, Any]]:
    """Evaluate (1-alpha)*L1 + alpha*candidate against actual development labels."""
    seeds = sorted(actual_by_seed)
    if seeds != sorted(reference_by_seed) or seeds != sorted(candidate_by_seed):
        raise ValueError("Simple-mix seed sets differ")
    out: list[dict[str, Any]] = []
    for alpha in alphas:
        a = float(alpha)
        if not 0.0 <= a <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        per_seed = {}
        for seed in seeds:
            y = np.asarray(actual_by_seed[seed], dtype=float)
            ref = np.asarray(reference_by_seed[seed], dtype=float)
            cand = np.asarray(candidate_by_seed[seed], dtype=float)
            if y.shape != ref.shape or y.shape != cand.shape or y.ndim != 1:
                raise ValueError("Simple-mix arrays must be aligned one-dimensional vectors")
            per_seed[seed] = float(wmape(y, (1.0 - a) * ref + a * cand))
        out.append({
            "alpha": a,
            "per_seed_wmape": per_seed,
            "mean_wmape": float(np.mean(list(per_seed.values()))),
        })
    return out


def select_simple_mix_models(actual_by_seed: Mapping[str, np.ndarray],
                             l1_by_seed: Mapping[str, np.ndarray],
                             ebm_by_seed: Mapping[str, Mapping[str, np.ndarray]],
                             alphas: Sequence[float] = (0.10, 0.25, 0.50)) -> dict[str, Any]:
    """Select one strong EBM and one mix-effective, structurally distinct EBM.

    The second model is selected by actual improvement over the L1 control on
    the supplied development labels.  If no candidate improves, only the
    independent best EBM is retained and the caller must not force a second.
    """
    seeds = sorted(actual_by_seed)
    if not seeds:
        raise ValueError("No development seeds supplied")
    if sorted(l1_by_seed) != seeds:
        raise ValueError("L1 predictions do not cover every development seed")
    candidate_names = sorted(ebm_by_seed)
    if not candidate_names:
        raise ValueError("No EBM candidates supplied")
    single_scores = {}
    for name in candidate_names:
        if sorted(ebm_by_seed[name]) != seeds:
            raise ValueError(f"Candidate {name} does not cover every seed")
        single_scores[name] = float(np.mean([
            wmape(np.asarray(actual_by_seed[s], dtype=float), np.asarray(ebm_by_seed[name][s], dtype=float))
            for s in seeds
        ]))
    l1_mean = float(np.mean([
        wmape(np.asarray(actual_by_seed[s], dtype=float), np.asarray(l1_by_seed[s], dtype=float))
        for s in seeds
    ]))
    strongest = sorted(candidate_names, key=lambda name: (single_scores[name], name))[0]
    second = None
    second_scan: list[dict[str, Any]] = []
    for name in candidate_names:
        if name == strongest:
            continue
        scan = simple_mix_alphas(actual_by_seed, l1_by_seed, ebm_by_seed[name], alphas)
        best = min(scan, key=lambda row: (row["mean_wmape"], row["alpha"]))
        improvement = l1_mean - float(best["mean_wmape"])
        record = {
            "name": name,
            **best,
            "single_mean_wmape": single_scores[name],
            "improvement_vs_l1": float(improvement),
        }
        second_scan.append(record)
        if improvement > 1e-12 and (
            second is None
            or (improvement, -float(best["mean_wmape"]), name)
            > (second["improvement_vs_l1"], -float(second["mean_wmape"]), second["name"])
        ):
            second = record
    selected = [strongest]
    if second is not None:
        selected.append(second["name"])
    return {
        "selected": selected,
        "strongest_single": strongest,
        "l1_mean_wmape": l1_mean,
        "single_mean_wmape": {name: single_scores[name] for name in candidate_names},
        "second_selection_scan": sorted(
            second_scan, key=lambda row: (-row["improvement_vs_l1"], row["name"])
        ),
    }


def s0_rule_records(actual_by_seed: Mapping[str, np.ndarray],
                    l1_by_seed: Mapping[str, np.ndarray],
                    ebm_by_seed: Mapping[str, Mapping[str, np.ndarray]],
                    alphas: Sequence[float] = (0.10, 0.25, 0.50)) -> list[dict[str, Any]]:
    """Return the up-to-12 S0 simple-mix rule records for one target.

    Each record is a development-screen rule; no outer-validation labels are
    consumed here.  The caller is responsible for keeping target-level package
    arithmetic separate from the target-local WMAPE record.
    """
    selection = select_simple_mix_models(actual_by_seed, l1_by_seed, ebm_by_seed, alphas)
    records: list[dict[str, Any]] = []
    for name in selection["selected"]:
        scan = simple_mix_alphas(actual_by_seed, l1_by_seed, ebm_by_seed[name], alphas)
        for row in scan:
            records.append({
                "target_model": name,
                "alpha": row["alpha"],
                "mean_wmape": row["mean_wmape"],
                "per_seed_wmape": row["per_seed_wmape"],
                "single_mean_wmape": selection["single_mean_wmape"][name],
                "l1_mean_wmape": selection["l1_mean_wmape"],
                "delta_vs_l1": float(selection["l1_mean_wmape"] - row["mean_wmape"]),
                "package_delta_from_target": float(50.0 * (selection["l1_mean_wmape"] - row["mean_wmape"])),
                "role": "independent_best" if name == selection["strongest_single"] else "mix_effective_second",
            })
    return records
