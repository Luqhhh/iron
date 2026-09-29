"""Frozen isolated-column RFM scores and complete-split admission decisions.

Decisions require independently audited caller artifacts. These functions do
not authorize fits, packaging or uploads. Four seeds reuse the same rows.
"""
import numpy as np

from .rfm_protocol import ARMS, TARGETS

DEVELOPMENT_SEEDS = (42, 3407)
CONFIRMATION_SEEDS = (7777, 12011)
LCB_MULTIPLIER = 2.3533634348018264


def _vector(value, rows=None):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("finite nonempty score vectors required")
    if rows is not None and len(result) != rows:
        raise ValueError("score row count mismatch")
    return result


def wmape(y, prediction):
    y = _vector(y)
    prediction = _vector(prediction, len(y))
    denominator = float(np.abs(y).sum())
    if denominator <= 0:
        raise ValueError("positive WMAPE denominator required")
    return float(np.abs(y-prediction).sum()/denominator)


def score_seed(y, folds, current, historical, members, *, target, seed):
    if (target not in TARGETS or seed not in (*DEVELOPMENT_SEEDS, *CONFIRMATION_SEEDS)
            or set(y) != set(TARGETS) or set(current) != set(TARGETS)
            or set(historical) != set(TARGETS) or set(members) != set(ARMS)):
        raise ValueError("incomplete target/arm/seed score definition")
    y = {t: _vector(y[t]) for t in TARGETS}
    n = len(y[TARGETS[0]])
    if any(len(v) != n for v in y.values()):
        raise ValueError("target rows do not align")
    folds = np.asarray(folds)
    if folds.shape != (n,) or not np.issubdtype(folds.dtype, np.integer) or set(folds) != set(range(5)):
        raise ValueError("complete five-fold coverage required")
    current = {t: _vector(current[t], n) for t in TARGETS}
    historical = {t: _vector(historical[t], n) for t in TARGETS}
    base = {t: wmape(y[t], current[t]) for t in TARGETS}
    old = {t: wmape(y[t], historical[t]) for t in TARGETS}
    base_score = 100 - 50*sum(base.values())
    historical_score = 100 - 50*sum(old.values())
    other = next(t for t in TARGETS if t != target)
    records = []
    for arm in ARMS:
        raw = _vector(members[arm], n)
        changed = .8*current[target]+.2*raw
        error = wmape(y[target], changed)
        candidate_score = 100-50*(error+base[other])
        records.append({"target": target, "arm": arm, "seed": int(seed), "rows": n,
            "blend_weight": .2, "base_score": base_score, "candidate_score": candidate_score,
            "gain": 50*(base[target]-error), "target_wmape": error,
            "standalone_wmape": wmape(y[target], raw), "base_target_wmape": base[target],
            "historical_target_gain": 50*(old[target]-error),
            "historical_package_score": historical_score,
            "candidate_minus_historical_package": candidate_score-historical_score,
            "other_target_unchanged": True,
            "folds_descriptive": [{"fold": f, "rows": int((folds==f).sum()),
                "gain": 50*(wmape(y[target][folds==f], current[target][folds==f])
                            -wmape(y[target][folds==f], changed[folds==f]))} for f in range(5)]})
    return records


def _index(records, seeds, targets):
    expected = {(t, a, s) for t in targets for a in ARMS for s in seeds}
    indexed = {}
    for row in records:
        key = row["target"], row["arm"], row["seed"]
        if key not in expected or key in indexed:
            raise ValueError("unexpected or duplicate decision record")
        if (row["blend_weight"] != .2 or row["other_target_unchanged"] is not True
                or not np.isfinite([row["gain"], row["candidate_score"]]).all()):
            raise ValueError("invalid decision input")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("incomplete decision coverage")
    return indexed


def decide_development(records):
    rows = _index(records, DEVELOPMENT_SEEDS, TARGETS)
    decisions = {}
    for target in TARGETS:
        candidate = [rows[target, "FULL_RFM", s] for s in DEVELOPMENT_SEEDS]
        gains = np.array([r["gain"] for r in candidate])
        contrast = gains-np.array([rows[target, "FIXED_KRR", s]["gain"] for s in DEVELOPMENT_SEEDS])
        checks = {"both_complete_seeds_positive": bool((gains > 0).all()),
                  "mean_gain_at_least_0_01": bool(gains.mean() >= .01),
                  "positive_control_advantage": bool(contrast.mean() > 0),
                  "local_package_at_least_96_25": bool(np.mean([r["candidate_score"] for r in candidate]) >= 96.25)}
        decisions[target] = {"eligible": all(checks.values()), "checks": checks,
            "failure_reasons": [k for k,v in checks.items() if not v],
            "mean_gain": float(gains.mean()), "mean_control_advantage": float(contrast.mean())}
    return {"phase": "development", "decisions": decisions,
            "eligible_targets": [t for t in TARGETS if decisions[t]["eligible"]],
            "release_authorized": False, "requires_independent_artifact_audit": True}


def decide_confirmation(development_records, confirmation_records):
    development = decide_development(development_records)
    targets = development["eligible_targets"]
    confirmed = _index(confirmation_records, CONFIRMATION_SEEDS, targets)
    rows = _index(development_records, DEVELOPMENT_SEEDS, TARGETS)
    rows.update(confirmed)
    decisions = {}
    for target in targets:
        seeds = (*DEVELOPMENT_SEEDS, *CONFIRMATION_SEEDS)
        gains = np.array([rows[target, "FULL_RFM", s]["gain"] for s in seeds])
        control = np.array([rows[target, "FIXED_KRR", s]["gain"] for s in seeds])
        lower = float(gains.mean()-LCB_MULTIPLIER*gains.std(ddof=1)/2)
        checks = {"all_four_seeds_positive": bool((gains > 0).all()),
                  "paired_seed_lcb95_positive": bool(lower > 0),
                  "positive_four_seed_control_advantage": bool((gains-control).mean() > 0)}
        decisions[target] = {"promoted": all(checks.values()), "checks": checks,
            "failure_reasons": [k for k,v in checks.items() if not v],
            "mean_gain": float(gains.mean()), "seed_lcb95": lower,
            "mean_control_advantage": float((gains-control).mean())}
    return {"phase": "confirmation", "decisions": decisions,
            "promoted_targets": [t for t in targets if decisions[t]["promoted"]],
            "release_authorized": False, "requires_independent_artifact_audit": True,
            "interpretation": "Repeated splits of the same rows; fold signs descriptive; not a platform forecast."}
