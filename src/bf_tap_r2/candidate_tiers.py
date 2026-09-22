"""Prospective candidate triage; never changes frozen selectors or releases."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from .weak_models import assess_candidate


def classify_candidates(metrics, spec, policy):
    """Classify a predeclared pool against target-specific references.

    This is a shortlist, not evidence verification or release authorization.
    Existing identity, same-fold OOF and cold-inference checks remain required.
    """
    if policy["version"] != "candidate-tiers-v1" or policy["max_exploration_per_round"] != 1:
        raise ValueError("Unsupported frozen tier policy")
    rules = policy["promotion"]
    tolerance = rules["numerical_equality_tolerance"]
    seeds = [str(s) for s in spec["split_seeds"]]
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("Expected two distinct split seeds")
    records, formal, exploration = {}, [], []
    for target, pool in spec["candidates"].items():
        reference = spec["reference_by_target"][target]
        if len(set(pool)) != len(pool) or reference in pool:
            raise ValueError("Candidate pool must be unique and exclude reference")
        order = spec["tie_preference_by_target"][target]
        if set(order) != set(pool) or len(order) != len(pool):
            raise ValueError("Tie preference must enumerate candidate pool")
        anchors = metrics[target][reference]
        for route in [reference, *pool]:
            scores = metrics[target][route]
            if set(scores) != set(seeds):
                raise ValueError("Split identity mismatch")
            for seed in seeds:
                for group in ("by_fold", "by_spout"):
                    values = scores[seed][group]
                    if not values or set(values) != set(anchors[seed][group]):
                        raise ValueError("Fold/spout identity mismatch")
                    if group == "by_fold" and len(values) != spec["folds"]:
                        raise ValueError("Incomplete folds")
                values = [scores[seed]["wmape"], *scores[seed]["by_fold"].values(),
                          *scores[seed]["by_spout"].values()]
                if any(not math.isfinite(v) or v < 0 for v in values):
                    raise ValueError("Invalid WMAPE")
        records[target] = {}
        for route in pool:
            scores = metrics[target][route]
            decision = assess_candidate(scores, anchors, rules)
            gains = {s: anchors[s]["wmape"] - scores[s]["wmape"] for s in seeds}
            failed = []
            if not all(decision["pooled_improvements"].values()):
                failed.append("both_splits_improve")
            if decision["improved_folds"] < rules["minimum_improved_folds"]:
                failed.append("minimum_improved_folds")
            if decision["worst_spout_delta"] > rules["max_spout_wmape_degradation"] + tolerance:
                failed.append("spout_risk")
            mean_gain = sum(gains.values()) / len(seeds)
            tier = "formal" if decision["eligible"] else "exploration" if mean_gain > tolerance else "not_shortlisted"
            record = dict(decision, target=target, candidate=route, reference=reference,
                          tier=tier, failed_conditions=failed, gains_by_split=gains,
                          mean_gain=mean_gain, minimum_gain=min(gains.values()))
            records[target][route] = record
        passing = [r for r in records[target].values() if r["tier"] == "formal"]
        if passing:
            best = min(r["mean_wmape"] for r in passing)
            tied = [r for r in passing if r["mean_wmape"] <= best + policy["tie_tolerance"]]
            # Explicit per-target ordering is frozen before evaluating results.
            formal.append(min(tied, key=lambda r: order.index(r["candidate"])))
        exploration.extend(r for r in records[target].values() if r["tier"] == "exploration")
    formal.sort(key=lambda r: (-r["minimum_gain"], r["target"], r["candidate"]))
    exploration.sort(key=lambda r: (-r["mean_gain"], -r["minimum_gain"], r["target"], r["candidate"]))
    shortlist = exploration[:policy["max_exploration_per_round"]]
    return {"policy": policy["version"], "decisions": records,
            "formal_selected": formal, "exploration_selected": shortlist,
            "submission_priority": formal + shortlist,
            "release_authorized": False,
            "interpretation": "Development shortlist; not proof of platform improvement. No automatic release."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, help="JSON metrics or summary containing metrics")
    parser.add_argument("--spec", type=Path, required=True, help="Prospective frozen candidate pool YAML")
    parser.add_argument("--policy", type=Path, default=Path("configs/candidate_tiers.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("local").resolve()):
        parser.error("Output must remain under local/")
    raw = json.loads(args.metrics.read_text())
    result = classify_candidates(raw.get("metrics", raw), yaml.safe_load(args.spec.read_text()),
                                 yaml.safe_load(args.policy.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
