"""V3.5 constrained-composition primitives.

This module contains the pre-registered S0 cap-release screen.  It deliberately
keeps the L1 curve fixed and only changes the upper bound on the total weight of
the selected expert set.  All weights are learned from the supplied development
predictions only; the caller must never pass outer-validation labels.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import wmape
from .v3_4_composition import lp_constrained_weights


def _as_seed_mapping(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in values.items():
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 1 or not len(arr) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid development prediction vector for seed {key}")
        out[str(key)] = arr
    if not out:
        raise ValueError("No development prediction vectors supplied")
    return out


def select_g2_expert(y_by_seed: Mapping[str, np.ndarray],
                     experts_by_seed: Mapping[str, Mapping[str, np.ndarray]],
                     candidate_names: Sequence[str] | None = None) -> dict[str, Any]:
    """Select the single strongest expert by mean development WMAPE.

    Ties are resolved by trial/expert name, exactly as frozen in the task book.
    """
    y = _as_seed_mapping(y_by_seed)
    seeds = sorted(y)
    names = sorted(candidate_names) if candidate_names is not None else sorted(experts_by_seed)
    if not names:
        raise ValueError("No G2 candidate experts supplied")
    rows: list[dict[str, Any]] = []
    for name in names:
        if name not in experts_by_seed:
            raise KeyError(f"G2 expert {name} has no prediction vectors")
        pred = _as_seed_mapping(experts_by_seed[name])
        if sorted(pred) != seeds:
            raise ValueError(f"G2 expert {name} does not cover every development seed")
        per_seed = {seed: float(wmape(y[seed], pred[seed])) for seed in seeds}
        rows.append({
            "name": str(name),
            "mean_wmape": float(np.mean(list(per_seed.values()))),
            "per_seed_wmape": per_seed,
        })
    rows.sort(key=lambda row: (row["mean_wmape"], row["name"]))
    return {"selected": rows[0]["name"], "ranking": rows}


def _align_experts(y_by_seed: Mapping[str, np.ndarray],
                   l1_by_seed: Mapping[str, np.ndarray],
                   experts_by_seed: Mapping[str, Mapping[str, np.ndarray]],
                   names: Sequence[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    y = _as_seed_mapping(y_by_seed)
    l1 = _as_seed_mapping(l1_by_seed)
    seeds = sorted(y)
    if sorted(l1) != seeds:
        raise ValueError("L1 development vectors do not cover every seed")
    aligned: dict[str, np.ndarray] = {}
    for name in names:
        if name not in experts_by_seed:
            raise KeyError(f"Missing development expert {name}")
        values = _as_seed_mapping(experts_by_seed[name])
        if sorted(values) != seeds:
            raise ValueError(f"Expert {name} does not cover every development seed")
        for seed in seeds:
            if values[seed].shape != y[seed].shape or l1[seed].shape != y[seed].shape:
                raise ValueError(f"Development prediction shape mismatch for {name}/{seed}")
        aligned[str(name)] = values
    return y, l1, aligned


def _predict_columns(seed: str, l1: Mapping[str, np.ndarray],
                     experts: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.column_stack([l1[seed], *[experts[name][seed] for name in names]])


def s0_cap_records(y_by_seed: Mapping[str, np.ndarray],
                   l1_by_seed: Mapping[str, np.ndarray],
                   experts_by_seed: Mapping[str, Mapping[str, np.ndarray]],
                   *,
                   g1_names: Sequence[str],
                   g2_candidates: Sequence[str] | None = None,
                   caps: Sequence[float] = (0.65, 0.80, 1.00),
                   old_cap: float = 0.50) -> dict[str, Any]:
    """Evaluate the pre-registered S0 G1/G2 cap-release rules.

    Returns the 12 new cap rules plus the old ``cap=0.50`` control for each
    group.  A separate joint selection step must choose a single ``(G, cap)``
    rule per target and then freeze it before any outer split is read.
    """
    if len(g1_names) != 2:
        raise ValueError("S0 G1 must contain exactly the two existing A experts")
    caps_tuple = tuple(float(c) for c in caps)
    if caps_tuple != (0.65, 0.80, 1.00):
        raise ValueError("V3.5 S0 caps are frozen at 0.65/0.80/1.00")
    if float(old_cap) != 0.50:
        raise ValueError("V3.5 S0 old-cap control is frozen at 0.50")
    y, l1, experts = _align_experts(y_by_seed, l1_by_seed, experts_by_seed, list(g1_names))
    seeds = sorted(y)

    g2_selection = select_g2_expert(y, experts, candidate_names=g2_candidates or list(g1_names))
    groups: dict[str, list[str]] = {
        "G1": [str(name) for name in g1_names],
        "G2": [g2_selection["selected"]],
    }
    records: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for group_name, names in groups.items():
        for cap in caps_tuple:
            p_by_seed = {seed: _predict_columns(seed, l1, experts, names) for seed in seeds}
            fit = lp_constrained_weights(
                y_by_seed, p_by_seed, new_indices=list(range(1, len(names) + 1)),
                max_new_weight=float(cap),
            )
            weights = [float(v) for v in fit["weights"]]
            records.append({
                "group": group_name,
                "cap": float(cap),
                "members": ["L1", *names],
                "weights": weights,
                "new_weight_total": float(fit["new_weight_total"]),
                "objective": float(fit["objective"]),
                "per_seed_wmape": {
                    seed: float(wmape(y[seed], _predict_columns(seed, l1, experts, names) @ np.asarray(weights)))
                    for seed in seeds
                },
                "seeds": seeds,
                "status": "new_rule",
            })
        p_by_seed = {seed: _predict_columns(seed, l1, experts, names) for seed in seeds}
        fit = lp_constrained_weights(
            y_by_seed, p_by_seed, new_indices=list(range(1, len(names) + 1)),
            max_new_weight=float(old_cap),
        )
        controls.append({
            "group": group_name,
            "cap": float(old_cap),
            "members": ["L1", *names],
            "weights": [float(v) for v in fit["weights"]],
            "new_weight_total": float(fit["new_weight_total"]),
            "objective": float(fit["objective"]),
            "per_seed_wmape": {
                seed: float(wmape(y[seed], _predict_columns(seed, l1, experts, names) @ np.asarray(fit["weights"])))
                for seed in seeds
            },
            "seeds": seeds,
            "status": "old_cap_control",
        })
    return {
        "g1_names": [str(name) for name in g1_names],
        "g2_selection": g2_selection,
        "rules": records,
        "old_cap_controls": controls,
        "caps": list(caps_tuple),
        "old_cap": float(old_cap),
        "baseline_l1": {
            seed: float(wmape(y[seed], l1[seed])) for seed in seeds
        },
    }


def select_s0_rule(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Jointly choose one S0 rule by development mean WMAPE, with frozen ties."""
    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError("No S0 rules to select")
    rows.sort(key=lambda row: (
        float(row.get("objective", float("inf"))),
        str(row.get("group")),
        float(row.get("cap", -1.0)),
        str(row.get("members")),
    ))
    return rows[0]


def apply_s0_rule(l1_by_seed: Mapping[str, np.ndarray],
                  experts_by_seed: Mapping[str, Mapping[str, np.ndarray]],
                  rule: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Apply a frozen S0 rule to already-available prediction vectors."""
    y_like = {seed: np.asarray(v, dtype=float) for seed, v in l1_by_seed.items()}
    names = [str(name) for name in rule["members"] if str(name) != "L1"]
    y, l1, experts = _align_experts(y_like, l1_by_seed, experts_by_seed, names)
    weights = np.asarray(rule["weights"], dtype=float)
    if weights.shape != (len(names) + 1,):
        raise ValueError("S0 rule weight length does not match member list")
    out: dict[str, np.ndarray] = {}
    for seed in sorted(y):
        out[seed] = _predict_columns(seed, l1, experts, names) @ weights
    return out


def cap_for_record(record: Mapping[str, Any]) -> float:
    return float(record.get("cap", record.get("max_new_weight", 0.0)))
