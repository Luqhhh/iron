"""A local-only, stability-gated pairwise convex ensemble."""
from itertools import combinations

import numpy as np
import pandas as pd

from ....exceptions import ContractError
from ..oof import score_oof_predictions
from ..selection import quality_evidence

LOCAL_MEMBERS = {"M1_BLEND", "M2_RECENCY", "M3_RESIDUAL"}
TARGETS = ("tap_iron", "tap_time_len")
POLICY = {
    "member_gate": "local_pooled_and_target_and_majority_fold_checks",
    "max_residual_correlation": .95,
    "probe_weight": .5,
    "weight_grid": [.25, .5, .75],
    "minimum_probe_improvement": 1e-12,
}


def apply_ensemble(predictions, frozen):
    if frozen.get("status") != "ENABLED":
        raise ContractError("M4 requires an enabled frozen selection")
    members = frozen["members"]
    if not members or not set(members) <= LOCAL_MEMBERS or set(members) - set(predictions):
        raise ContractError("M4 members must be available local candidates")
    first = predictions[members[0]]
    keys = ["fold_id", "sample_id"] if "fold_id" in first else ["sample_id"]
    columns = ["pred_" + t for t in TARGETS]
    result = first[keys].copy()
    for key in keys:
        result[key] = result[key].astype("string")
    if result[keys].isna().any().any() or result.duplicated(keys).any():
        raise ContractError("invalid ensemble prediction keys")
    aligned = {}
    for member in members:
        frame = predictions[member].copy()
        if set(keys + columns) - set(frame) or (("fold_id" in frame) != ("fold_id" in keys)):
            raise ContractError("ensemble prediction shape mismatch")
        for key in keys:
            frame[key] = frame[key].astype("string")
        if (len(frame) != len(result) or frame[keys].isna().any().any()
                or frame.duplicated(keys).any()
                or set(map(tuple, frame[keys].to_numpy())) != set(map(tuple, result[keys].to_numpy()))):
            raise ContractError("ensemble prediction keys mismatch")
        aligned[member] = result.merge(frame, on=keys, validate="one_to_one", sort=False)[columns]
        if not np.isfinite(aligned[member].to_numpy(float)).all():
            raise ContractError("ensemble predictions must be finite")
    if set(frozen["weights"]) != set(TARGETS):
        raise ContractError("ensemble requires weights for both targets")
    for target in TARGETS:
        weights = frozen["weights"][target]
        if (set(weights) != set(members)
                or any(not np.isfinite(w) or w < 0 for w in weights.values())
                or not np.isclose(sum(weights.values()), 1., atol=1e-12, rtol=0)):
            raise ContractError("ensemble weights must form a finite convex combination")
        result["pred_" + target] = sum(
            weights[member] * aligned[member]["pred_" + target].to_numpy()
            for member in members)
    return result


def _frozen(pair, weight):
    return {"status": "ENABLED", "members": list(pair),
            "weights": {t: {pair[0]: weight, pair[1]: 1. - weight} for t in TARGETS}}


def select_ensemble(actual, predictions, catboost, b1):
    if set(predictions) - LOCAL_MEMBERS:
        raise ContractError("M4 accepts local M1/M2/M3 candidates only")
    evidence = {name: quality_evidence(actual, pred, catboost, b1)
                for name, pred in predictions.items()}
    stable = sorted(name for name, info in evidence.items() if info["pass"])
    report = {"status": "SKIPPED", "reason": "insufficient_stable_complementary_members",
              "policy": POLICY, "member_quality": evidence, "pairs": []}
    eligible = []
    for pair in combinations(stable, 2):
        probe = apply_ensemble(predictions, _frozen(pair, .5))
        score = score_oof_predictions(actual, probe)
        residuals = []
        for member in pair:
            aligned = actual.merge(predictions[member], on=["fold_id", "sample_id"],
                                   validate="one_to_one", sort=False)
            residuals.append(np.column_stack([
                aligned["pred_" + t].to_numpy() - aligned[t].to_numpy() for t in TARGETS]))
        correlations = []
        for i in range(2):
            left, right = residuals[0][:, i], residuals[1][:, i]
            correlations.append(float(np.corrcoef(left, right)[0, 1])
                                if np.std(left) > 0 and np.std(right) > 0 else 1.)
        better = min(evidence[n]["metrics"]["pooled"]["loss"] for n in pair)
        complementarity = (min(correlations) < POLICY["max_residual_correlation"]
                           and better - score["pooled"]["loss"] > POLICY["minimum_probe_improvement"])
        report["pairs"].append({"members": list(pair), "residual_correlations": correlations,
                                "probe_loss": score["pooled"]["loss"], "pass": complementarity})
        if not complementarity:
            continue
        options = []
        for w in POLICY["weight_grid"]:
            state = _frozen(pair, w)
            frame = apply_ensemble(predictions, state)
            options.append((w, score_oof_predictions(actual, frame)["pooled"]))
        frozen = _frozen(pair, .5)
        for target, metric in zip(TARGETS, ("iron", "time")):
            chosen = min(options, key=lambda entry: (entry[1][metric]["wmape"], abs(entry[0] - .5), entry[0]))[0]
            frozen["weights"][target] = {pair[0]: chosen, pair[1]: 1. - chosen}
        frame = apply_ensemble(predictions, frozen)
        frozen["metrics"] = score_oof_predictions(actual, frame)
        eligible.append(frozen)
    if eligible:
        best = min(eligible, key=lambda item: (item["metrics"]["pooled"]["loss"], item["members"]))
        report.update(best)
        report["reason"] = "stable_members_and_complementary_residuals"
    return report
