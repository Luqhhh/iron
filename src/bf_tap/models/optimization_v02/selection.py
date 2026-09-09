"""Local OOF quality gates, distinct from remote equal-horizon J."""
from .oof import score_oof_predictions


def quality_evidence(actual, candidate, catboost, b1):
    scores = {name: score_oof_predictions(actual, pred)
              for name, pred in (("candidate", candidate), ("catboost", catboost), ("B1", b1))}
    c, base, control = (scores[n] for n in ("candidate", "catboost", "B1"))
    deltas = {fold: c["folds"][fold]["loss"] - base["folds"][fold]["loss"]
              for fold in c["folds"]}
    wins = sum(delta < 0 for delta in deltas.values())
    target_delta = {
        t: c["pooled"][t]["wmape"] - min(base["pooled"][t]["wmape"], control["pooled"][t]["wmape"])
        for t in ("iron", "time")
    }
    checks = {
        "beats_both_pooled": c["pooled"]["loss"] < min(base["pooled"]["loss"], control["pooled"]["loss"]),
        "per_target_regression": max(target_delta.values()) <= .002,
        "majority_fold_wins": wins / len(deltas) >= 2 / 3,
    }
    return {"pass": all(checks.values()), "checks": checks,
            "per_target_delta": target_delta, "fold_loss_delta": deltas,
            "winning_folds": wins, "worst_fold_delta": max(deltas.values()),
            "metrics": c}
