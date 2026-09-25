"""Round2 V5 evaluation resolution: nested selection, paired statistics, LCB.

The V5 pre-registration replaces the previous "two split seeds with the same sign
and a mean of at least +0.005" gate.  Two facts drive the change:

* the per-fold gain of the historical N2 candidate spans roughly -0.009 to
  +0.028; a fold-level 95% lower confidence bound on those ten cells is about
  +0.008, so a fold-level bound *admits* a candidate that lost 0.0201 on the
  platform;
* the same ten cells were used to select the candidate, so they cannot be
  independent evidence.

The effective promotion unit is therefore the split seed: at least four split
seeds with a positive paired lower bound.  Fold-level statistics are retained,
but only as descriptive evidence.

Nothing in this module fits, selects or promotes anything by itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .metrics import wmape as _wmape

__all__ = [
    "CellObservation",
    "wmape",
    "package_score",
    "package_score_from_targets",
    "paired_cells",
    "paired_summary",
    "seed_gains",
    "seed_level_summary",
    "t_quantile_one_sided",
    "select_alpha",
    "nested_blend",
    "blend_record",
    "fold_criteria_met",
    "admit",
]

#: Two-sided Student-t quantiles for a one-sided 95% bound, used when SciPy is
#: unavailable.  The V5 environment ships SciPy; the table keeps the statistic
#: reproducible in a minimal interpreter.
_T_TABLE_95 = {
    1: 6.3138, 2: 2.9200, 3: 2.3534, 4: 2.1318, 5: 2.0150, 6: 1.9432,
    7: 1.8946, 8: 1.8595, 9: 1.8331, 10: 1.8125, 11: 1.7959, 12: 1.7823,
    13: 1.7709, 14: 1.7613, 15: 1.7531, 16: 1.7459, 17: 1.7396, 18: 1.7341,
    19: 1.7291, 20: 1.7247, 21: 1.7207, 22: 1.7171, 23: 1.7139, 24: 1.7109,
    25: 1.7081, 26: 1.7056, 27: 1.7033, 28: 1.7011, 29: 1.6991, 30: 1.6973,
}


def wmape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Original-unit WMAPE, matching :func:`bf_tap_r2.metrics.wmape`."""
    return float(_wmape(actual, predicted))


def package_score(iron_wmape: float, time_wmape: float) -> float:
    """The frozen complete-package score: equal-target WMAPE."""
    return float(100.0 - 50.0 * (float(iron_wmape) + float(time_wmape)))


def package_score_from_targets(target_wmape: Mapping[str, float]) -> float:
    return float(100.0 - 50.0 * float(sum(float(v) for v in target_wmape.values())))


@dataclass(frozen=True)
class CellObservation:
    """One ``(split seed, fold)`` pair of the base and candidate columns."""

    seed: int
    fold: int
    rows: int
    wmape_base: float
    wmape_candidate: float

    @property
    def delta_wmape(self) -> float:
        return float(self.wmape_base - self.wmape_candidate)

    @property
    def delta_score(self) -> float:
        """Score change for the full package when only this column changes."""
        return float(50.0 * self.delta_wmape)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "fold": int(self.fold),
            "rows": int(self.rows),
            "wmape_base": float(self.wmape_base),
            "wmape_candidate": float(self.wmape_candidate),
            "delta_wmape": float(self.delta_wmape),
            "delta_score": float(self.delta_score),
        }


def paired_cells(y: Sequence[float], folds: Mapping[int, np.ndarray],
                 base_by_seed: Mapping[int, np.ndarray],
                 candidate_by_seed: Mapping[int, np.ndarray]) -> list[CellObservation]:
    """Per ``(seed, fold)`` paired comparison of base and candidate columns."""
    actual = np.asarray(y, dtype=float)
    if sorted(base_by_seed) != sorted(candidate_by_seed):
        raise ValueError("V5 base and candidate seed sets differ")
    cells: list[CellObservation] = []
    for seed in sorted(base_by_seed):
        fold_vector = np.asarray(folds[int(seed)])
        base = np.asarray(base_by_seed[int(seed)], dtype=float)
        candidate = np.asarray(candidate_by_seed[int(seed)], dtype=float)
        if not (base.shape == candidate.shape == actual.shape):
            raise ValueError("V5 paired-cell shape mismatch")
        for fold in sorted(set(int(v) for v in fold_vector)):
            mask = fold_vector == fold
            cells.append(CellObservation(
                seed=int(seed),
                fold=int(fold),
                rows=int(mask.sum()),
                wmape_base=wmape(actual[mask], base[mask]),
                wmape_candidate=wmape(actual[mask], candidate[mask]),
            ))
    return cells


def paired_summary(values: Sequence[float]) -> dict[str, Any]:
    """Mean, standard error and one-sided 95% lower bound of a sample."""
    sample = np.asarray([float(v) for v in values], dtype=float)
    n = int(sample.size)
    if n == 0:
        raise ValueError("V5 paired summary requires at least one value")
    mean = float(sample.mean())
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "se": None, "lcb95": None, "positive": int((sample > 0).sum())}
    sd = float(sample.std(ddof=1))
    se = float(sd / math.sqrt(n))
    lcb = float(mean - t_quantile_one_sided(n - 1, 0.95) * se)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "lcb95": lcb,
        "positive": int((sample > 0).sum()),
    }


def seed_gains(cells: Iterable[CellObservation]) -> dict[int, float]:
    """Mean score gain per split seed, pooled over that seed's folds."""
    buckets: dict[int, list[float]] = {}
    for cell in cells:
        buckets.setdefault(int(cell.seed), []).append(float(cell.delta_score))
    return {int(seed): float(np.mean(values)) for seed, values in sorted(buckets.items())}


def seed_level_summary(gains: Mapping[int, float]) -> dict[str, Any]:
    """Paired summary whose unit is the split seed, not the fold."""
    values = [float(v) for _, v in sorted(gains.items())]
    summary = paired_summary(values)
    summary["positive_seeds"] = int(sum(1 for v in values if v > 0))
    summary["seeds"] = sorted(int(s) for s in gains)
    return summary


def t_quantile_one_sided(df: int, level: float = 0.95) -> float:
    """One-sided Student-t quantile; SciPy when present, table otherwise."""
    df = int(df)
    if df <= 0:
        raise ValueError("V5 t quantile requires positive degrees of freedom")
    if abs(float(level) - 0.95) < 1e-12:
        try:  # pragma: no cover - SciPy is present in the locked environment
            from scipy import stats  # type: ignore

            return float(stats.t.ppf(0.95, df))
        except Exception:
            pass
        if df in _T_TABLE_95:
            return float(_T_TABLE_95[df])
        return 1.6449
    try:  # pragma: no cover - SciPy is present in the locked environment
        from scipy import stats  # type: ignore

        return float(stats.t.ppf(float(level), df))
    except Exception:
        raise ValueError("V5 non-standard confidence levels require SciPy")


def select_alpha(y: Sequence[float], base: np.ndarray, candidate: np.ndarray,
                 grid: Sequence[float]) -> tuple[float, float]:
    """Choose the interior blend weight minimising the fit-seed WMAPE."""
    actual = np.asarray(y, dtype=float)
    best_alpha = float(grid[0])
    best_value = math.inf
    for alpha in grid:
        value = wmape(actual, (1.0 - float(alpha)) * base + float(alpha) * candidate)
        if value < best_value:
            best_value, best_alpha = float(value), float(alpha)
    return best_alpha, best_value


def nested_blend(y: Sequence[float], folds: Mapping[int, np.ndarray],
                 base_by_seed: Mapping[int, np.ndarray],
                 candidate_by_seed: Mapping[int, np.ndarray],
                 grid: Sequence[float]) -> dict[str, Any]:
    """Fit the blend weight on one split seed, evaluate the paired cells on the other.

    Held-out labels never enter the weight choice: for each held-out seed the
    weight is selected on the other seed's aligned OOF predictions only.
    """
    seeds = sorted(base_by_seed)
    if len(seeds) < 2:
        raise ValueError("V5 nested blend requires at least two split seeds")
    cells: list[CellObservation] = []
    alphas: dict[int, float] = {}
    for held in seeds:
        fit_seeds = [s for s in seeds if s != held]
        combined_y = np.asarray(y, dtype=float)
        best_alpha = float(grid[0])
        best_value = math.inf
        for alpha in grid:
            values = []
            for seed in fit_seeds:
                prediction = ((1.0 - float(alpha)) * np.asarray(base_by_seed[seed], dtype=float)
                              + float(alpha) * np.asarray(candidate_by_seed[seed], dtype=float))
                values.append(wmape(combined_y, prediction))
            value = float(np.mean(values))
            if value < best_value:
                best_value, best_alpha = value, float(alpha)
        alphas[int(held)] = best_alpha
        held_cells = paired_cells(
            y,
            {int(held): np.asarray(folds[int(held)])},
            {int(held): np.asarray(base_by_seed[int(held)], dtype=float)},
            {int(held): (1.0 - best_alpha) * np.asarray(base_by_seed[int(held)], dtype=float)
             + best_alpha * np.asarray(candidate_by_seed[int(held)], dtype=float)},
        )
        cells.extend(held_cells)
    gains = seed_gains(cells)
    return {
        "alphas": alphas,
        "cells": [cell.as_dict() for cell in cells],
        "fold_summary": paired_summary([cell.delta_score for cell in cells]),
        "seed_gains": gains,
        "seed_summary": seed_level_summary(gains),
    }


def blend_record(y: Sequence[float], folds: Mapping[int, np.ndarray],
                 base_by_seed: Mapping[int, np.ndarray],
                 candidate_by_seed: Mapping[int, np.ndarray],
                 grid: Sequence[float]) -> dict[str, Any]:
    """Nested blend plus the in-sample (all-seed) blend, reported side by side."""
    nested = nested_blend(y, folds, base_by_seed, candidate_by_seed, grid)
    all_seeds = sorted(base_by_seed)
    best_alpha = float(grid[0])
    best_value = math.inf
    for alpha in grid:
        value = float(np.mean([
            wmape(y, (1.0 - float(alpha)) * np.asarray(base_by_seed[s], dtype=float)
                  + float(alpha) * np.asarray(candidate_by_seed[s], dtype=float))
            for s in all_seeds
        ]))
        if value < best_value:
            best_value, best_alpha = value, float(alpha)
    in_sample_cells = paired_cells(
        y, folds, base_by_seed,
        {s: (1.0 - best_alpha) * np.asarray(base_by_seed[s], dtype=float)
            + best_alpha * np.asarray(candidate_by_seed[s], dtype=float) for s in all_seeds},
    )
    nested["in_sample"] = {
        "alpha": best_alpha,
        "fold_summary": paired_summary([c.delta_score for c in in_sample_cells]),
        "seed_gains": seed_gains(in_sample_cells),
    }
    return nested


def fold_criteria_met(fold_summary: Mapping[str, Any], positive_min: int,
                      total_min: int, both_initial_seeds_positive: bool) -> tuple[bool, list[str]]:
    """The descriptive fold-level criteria, kept for reporting and backtest."""
    reasons: list[str] = []
    if int(fold_summary.get("positive") or 0) < int(positive_min):
        reasons.append("positive_cells_below_minimum")
    if int(fold_summary.get("n") or 0) < int(total_min):
        reasons.append("fold_cells_incomplete")
    if not bool(both_initial_seeds_positive):
        reasons.append("initial_split_seed_not_positive")
    return (not reasons), reasons


def admit(record: Mapping[str, Any], *, min_seeds: int, positive_cells_min: int,
          positive_cells_total: int, lcb_level: float = 0.95) -> dict[str, Any]:
    """Apply the full V5 promotion rule to one candidate's evidence record.

    The rule admits a candidate only when the split-seed-level paired lower
    confidence bound is positive over at least ``min_seeds`` seeds.  Fold-level
    evidence is required but explicitly insufficient on its own.
    """
    fold_summary = dict(record["fold_summary"])
    gains = {int(k): float(v) for k, v in record["seed_gains"].items()}
    seed_summary = seed_level_summary(gains)
    ok_folds, fold_reasons = fold_criteria_met(
        fold_summary, positive_cells_min, positive_cells_total,
        both_initial_seeds_positive=all(v > 0 for v in gains.values()) if gains else False,
    )
    reasons: list[str] = []
    if len(gains) < int(min_seeds):
        reasons.append("insufficient_split_seeds")
    if seed_summary.get("lcb95") is None or float(seed_summary["lcb95"]) <= 0.0:
        reasons.append("seed_level_lcb_not_positive")
    reasons.extend(fold_reasons)
    return {
        "admitted": not reasons,
        "reasons": reasons,
        "fold_summary": fold_summary,
        "seed_summary": seed_summary,
        "seed_gains": gains,
        "required_split_seeds": int(min_seeds),
    }
