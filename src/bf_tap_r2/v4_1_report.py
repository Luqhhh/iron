"""Build the V4.1 candidate-tier evidence from frozen private predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .candidate_tiers import classify_candidates
from .metrics import wmape
from .v3_run import load_training_frame


def _complete_events(ledger: Path) -> dict[int, dict[str, Any]]:
    events: dict[int, dict[str, Any]] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") == "complete" and event.get("target") == "tap_time_len":
            events[int(event["seed"])] = event
    if set(events) != {42, 3407}:
        raise ValueError("V4.1 complete ledger lacks both time-target seeds")
    return events


def build_tier_inputs(root: Path, coarse_summary: Path,
                      complete_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    coarse = json.loads(coarse_summary.read_text(encoding="utf-8"))
    pool = [
        str(row["name"]) for row in coarse["qualifying"]
        if row["target"] == "tap_time_len"
    ]
    if not pool or len(pool) != len(set(pool)):
        raise ValueError("Invalid coarse-derived V4.1 complete candidate pool")
    events = _complete_events(complete_dir / "fit_ledger.jsonl")
    train = load_training_frame(root).set_index("sample_id")
    reference = "PUBLIC_EBM_ANCHOR"
    metrics: dict[str, Any] = {"tap_time_len": {reference: {}}}
    for name in pool:
        metrics["tap_time_len"][name] = {}
    for seed in (42, 3407):
        event = events[seed]
        # coarse-r1 pred files used object string arrays.  Numeric arrays are
        # trusted local evidence and IDs are immediately converted to strings;
        # future V4.1 outputs use a non-object Unicode dtype.
        archive = np.load(complete_dir / f"pred-tap_time_len-seed{seed}.npz", allow_pickle=True)
        ids = archive["sample_id"].astype(str)
        frame = train.loc[ids]
        labels = archive["target"].astype(float)
        spout = frame["spout_no"].astype(str).to_numpy()
        fold_rows = {str(row["fold"]): row for row in event["fold_rows"]}
        routes = {reference: archive["anchor"].astype(float)}
        routes.update({name: archive[f"candidate__{name}"].astype(float) for name in pool})
        for route, prediction in routes.items():
            by_fold = {}
            for fold, row in fold_rows.items():
                if route == reference:
                    by_fold[fold] = float(row["anchor_wmape"])
                else:
                    by_fold[fold] = float(row["candidates"][route]["wmape"])
            by_spout = {
                value: float(wmape(labels[spout == value], prediction[spout == value]))
                for value in sorted(set(spout.tolist()))
            }
            metrics["tap_time_len"][route][str(seed)] = {
                "wmape": float(wmape(labels, prediction)),
                "by_fold": by_fold,
                "by_spout": by_spout,
            }
    spec = {
        "split_seeds": [42, 3407],
        "folds": 5,
        "reference_by_target": {"tap_time_len": reference},
        "candidates": {"tap_time_len": pool},
        # Coarse ranking existed before complete-r1 and is the frozen cost tie order.
        "tie_preference_by_target": {"tap_time_len": pool},
        "pool_source": str(coarse_summary),
        "claim_boundary": "public-anchor-relative development classification only",
    }
    return metrics, spec


def write_report(root: Path, coarse_summary: Path, complete_dir: Path,
                 output_dir: Path) -> dict[str, Any]:
    if not output_dir.resolve().is_relative_to((root / "local").resolve()):
        raise ValueError("V4.1 tier outputs must remain under local/")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, spec = build_tier_inputs(root, coarse_summary, complete_dir)
    policy = yaml.safe_load((root / "configs/candidate_tiers.yaml").read_text(encoding="utf-8"))
    result = classify_candidates(metrics, spec, policy)
    paths = {
        "metrics.json": {"metrics": metrics},
        "candidate-tiers.json": result,
    }
    for name, payload in paths.items():
        with (output_dir / name).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    with (output_dir / "candidate-tier-spec.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, allow_unicode=True, sort_keys=False)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-summary", type=Path, required=True)
    parser.add_argument("--complete-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = write_report(Path.cwd(), args.coarse_summary, args.complete_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
