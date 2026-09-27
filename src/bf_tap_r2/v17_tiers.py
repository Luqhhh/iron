"""Apply the frozen candidate-tier policy to audited V17 fixed-slot columns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .candidate_tiers import classify_candidates
from .data import TARGETS
from .v7_periodic import file_hash, score_detail, write_new
from .v17_run import context, oof_from_folds


def classify(root, development, output):
    root = Path(root).resolve()
    dev = (root / development).resolve()
    output = (root / output).resolve()
    if not dev.is_relative_to(root / "local/runs/round2-v17") or not output.is_relative_to(dev):
        raise ValueError("Private audited V17 development paths required")
    spec_path = root / "configs/round2_v17/SPEC.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    tier_path = root / "configs/round2_v17/TIER_SPEC.yaml"
    tier_spec = yaml.safe_load(tier_path.read_text())
    summary = json.loads((dev / "summary.json").read_text())
    audit = json.loads((dev / "audit-r1.json").read_text())
    manifest = json.loads((dev / "manifest.json").read_text())
    if audit["status"] != "passed" or file_hash(spec_path) != manifest["spec_sha256"]:
        raise ValueError("Frozen audited development required")
    if (tier_spec["source_experiment"] != str(spec_path.relative_to(root))
            or tier_spec["split_seeds"] != spec["split_seeds"]
            or tier_spec["folds"] != spec["folds"]
            or [name for pool in tier_spec["candidates"].values() for name in pool] != list(spec["recipes"])):
        raise ValueError("Candidate tier pool differs from predeclared V17 order")
    frame, folds, a35, b0, _ = context(root, spec, spec["split_seeds"])
    metrics = {}
    for target in TARGETS:
        y = frame[target].to_numpy()
        reference = tier_spec["reference_by_target"][target]
        metrics[target] = {reference: {str(seed): score_detail(y, b0[seed][target], folds[seed],
                                                                   frame.spout_no.to_numpy())
                                       for seed in folds}}
        for name in tier_spec["candidates"][target]:
            prediction = oof_from_folds(dev, frame, folds, name)
            metrics[target][name] = {}
            for seed in folds:
                column = .5 * a35[seed][target] + .5 * prediction[seed]
                metrics[target][name][str(seed)] = score_detail(y, column, folds[seed],
                                                                frame.spout_no.to_numpy())
    policy = yaml.safe_load((root / "configs/candidate_tiers.yaml").read_text())
    decisions = classify_candidates(metrics, tier_spec, policy)
    result = {"status": "classified_not_released", "tier_spec_sha256": file_hash(tier_path),
              "policy_sha256": file_hash(root / "configs/candidate_tiers.yaml"),
              "development_summary_sha256": file_hash(dev / "summary.json"),
              "decisions": decisions, "packages": 0, "uploads": 0}
    write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = classify(Path.cwd(), args.development, args.output)
    print(json.dumps({"status": result["status"], "formal": len(result["decisions"]["formal_selected"]),
                      "exploration": len(result["decisions"]["exploration_selected"])}))


if __name__ == "__main__":
    main()
